"""Tests for extraction (milestone M5) with a fake LLM, so they run offline and for free."""

import json

import pytest

from erasmus_agent import db, extract
from erasmus_agent.models import Course, ProgrammeRecord, RequiredDocument

PAGE = {"page": "admission", "url": "https://example.org/admission"}


def fake_llm(replies):
    """Return a stand-in for chat_json that gives these replies in order and records the messages."""
    seen = []

    def chat_json(messages, schema_name, schema, model="main", reasoning_effort="low"):
        seen.append(messages)
        return json.dumps(replies[len(seen) - 1]), 100
    return chat_json, seen


def test_valid_reply_is_accepted_and_provenance_is_ours(monkeypatch):
    reply = {"programme_id": "something-else", "courses": [{"name": "Machine Learning", "source_url": "x",
                                                            "topics": ["machine learning"]}]}
    fake, _ = fake_llm([reply])
    monkeypatch.setattr(extract, "chat_json", fake)
    record, calls, _ = extract.extract_chunk("deai", PAGE, "text", "main")
    assert calls == 1
    assert record.programme_id == "deai"                            # not what the model said
    assert record.courses[0].source_url == PAGE["url"]


def test_invalid_reply_gets_one_retry_with_the_error(monkeypatch):
    bad = {"courses": [{"name": "Blockchain", "topics": ["blockchain"]}]}   # topic not in vocabulary
    good = {"courses": [{"name": "Blockchain", "topics": []}]}
    fake, seen = fake_llm([bad, good])
    monkeypatch.setattr(extract, "chat_json", fake)
    record, calls, _ = extract.extract_chunk("deai", PAGE, "text", "main")
    assert calls == 2 and record.courses[0].topics == []
    assert "topic vocabulary" in seen[1][-1]["content"]             # the retry quoted the error


def test_two_invalid_replies_become_missing_information(monkeypatch):
    bad = {"deadlines": [{"label": "application", "due": "1999-01-01", "quote": "x"}]}
    fake, _ = fake_llm([bad, bad])
    monkeypatch.setattr(extract, "chat_json", fake)
    reading = {"programme_id": "deai", "pages": [{**PAGE, "text": "short page"}]}
    record, stats = extract.extract_programme(reading)
    assert stats["failed_chunks"] == 1
    assert any("could not extract" in note for note in record.missing_information)


def test_merge_removes_duplicates_and_keeps_highest_count():
    url = "https://example.org"
    a = ProgrammeRecord(programme_id="deai", accepted_degree_fields=["computer science"],
                        required_documents=[RequiredDocument(doc_type="recommendation_letter", count=1)],
                        courses=[Course(name="Machine Learning", source_url=url)])
    b = ProgrammeRecord(programme_id="deai", accepted_degree_fields=["computer science", "mathematics"],
                        required_documents=[RequiredDocument(doc_type="recommendation_letter", count=2)],
                        courses=[Course(name="machine learning", source_url=url)])
    merged = extract.merge("deai", [a, b])
    assert merged.accepted_degree_fields == ["computer science", "mathematics"]
    assert merged.required_documents[0].count == 2
    assert len(merged.courses) == 1


def test_tagging_retries_once_then_applies_topics(monkeypatch):
    record = ProgrammeRecord(programme_id="ilspsd", courses=[Course(name="Public International Law", source_url="x"),
                                                             Course(name="Master Thesis", source_url="x")])
    bad = {"courses": [{"name": "Public International Law", "topics": ["legal stuff"]}]}
    good = {"courses": [{"name": "Public International Law", "topics": ["international law"]},
                        {"name": "Master Thesis", "topics": []}]}
    fake, seen = fake_llm([bad, good])
    monkeypatch.setattr(extract, "chat_json", fake)
    calls, _ = extract.tag_courses(record, "main")
    assert calls == 2 and "not in the vocabulary" in seen[1][-1]["content"]
    assert record.courses[0].topics == ["international law"] and record.courses[1].topics == []


def test_changed_pages_are_saved_as_pending(tmp_path):
    conn = db.connect(tmp_path / "test.db")
    db.init_db(conn)
    db.add_programme(conn, "deai", "101241133", "DEAI")
    url = "https://example.org"
    first = ProgrammeRecord(programme_id="deai", courses=[Course(name="Machine Learning", source_url=url)])
    _, status = extract.save_programme_record(conn, {"programme_id": "deai", "changed": False, "fingerprint": "aaa",
                                                     "pages": [{"fetched_at": "2026-10-23"}]}, first, "test")
    assert status == "approved"
    second = ProgrammeRecord(programme_id="deai", other_conditions=["new rule"])
    version_id, status = extract.save_programme_record(conn, {"programme_id": "deai", "changed": True, "fingerprint": "bbb",
                                                              "pages": [{"fetched_at": "2026-11-20"}]}, second, "test")
    assert status == "pending"
    assert db.get_current(conn, "programme_versions", {"programme_id": "deai"})["page_fingerprint"] == "aaa"
    assert conn.execute("SELECT COUNT(*) FROM courses").fetchone()[0] == 1
    conn.close()

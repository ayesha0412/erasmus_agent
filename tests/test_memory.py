"""Tests for student memory (milestone M6): onboarding, new document versions, isolation, stored answers."""

from pathlib import Path

import pytest

from erasmus_agent import db
from erasmus_agent.memory import lookup_profile, record_condition_answer, save_document, save_profile
from erasmus_agent.models import Document, load_profile

PROFILES = Path(__file__).resolve().parent.parent / "data" / "profiles"


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.db")
    db.init_db(connection)
    save_profile(connection, load_profile(PROFILES / "stu-sara.json"), "onboarding")
    save_profile(connection, load_profile(PROFILES / "stu-omar.json"), "onboarding")
    db.add_programme(connection, "codas", "101128281", "CoDaS")
    yield connection
    connection.close()


def test_onboarding_twice_creates_no_duplicates(conn):
    save_profile(conn, load_profile(PROFILES / "stu-sara.json"), "onboarding")   # a retry
    assert conn.execute("SELECT COUNT(*) FROM profile_versions WHERE student_id = 'stu-sara'").fetchone()[0] == 1
    assert len(lookup_profile(conn, "stu-sara")["documents"]) == 4


def test_new_ielts_becomes_current_and_old_one_is_kept(conn):
    new = Document(doc_type="english_test", version=2, added_on="2026-11-20",
                   details={"test": "IELTS Academic", "overall": 7.0, "taken_on": "2026-11-08"})
    save_document(conn, "stu-sara", new, "cycle 2")
    tests = [d for d in lookup_profile(conn, "stu-sara")["documents"] if d["doc_type"] == "english_test"]
    assert len(tests) == 1 and tests[0]["version"] == 2 and tests[0]["details"]["overall"] == 7.0
    history = db.get_history(conn, "document_versions", {"student_id": "stu-sara", "doc_id": "english_test"})
    assert [h["data"]["details"]["overall"] for h in history] == [6.5, 7.0]


def test_students_never_see_each_other(conn):
    sara, omar = lookup_profile(conn, "stu-sara"), lookup_profile(conn, "stu-omar")
    assert sara["profile"]["degree"]["degree_field"] == "computer science"
    assert omar["profile"]["interest_topics"] == ["robotics"]
    assert not any(d["doc_type"] == "recommendation_letter" for d in sara["documents"])   # only Omar has one


def test_condition_answer_is_remembered_once(conn):
    assert record_condition_answer(conn, "stu-sara", 1, "codas", "no previous master's", True) is True
    assert record_condition_answer(conn, "stu-sara", 1, "codas", "no previous master's", True) is False
    answers = lookup_profile(conn, "stu-sara")["condition_answers"]
    assert answers == [{"programme_id": "codas", "condition": "no previous master's", "met": True}]
    assert lookup_profile(conn, "stu-omar")["condition_answers"] == []


def test_unknown_student_is_rejected(conn):
    with pytest.raises(ValueError):
        lookup_profile(conn, "stu-nobody")

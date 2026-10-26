"""Tests for the deterministic rules (milestone M6): all three eligibility results and missing documents."""

import datetime as dt

from erasmus_agent.models import Course, ProgrammeRecord
from erasmus_agent.rules import check_eligibility, excluded_by_preferences, find_missing_documents, match_score

TODAY = dt.date(2026, 10, 25)
URL = "https://example.org"


def student(**profile_changes) -> dict:
    """A Sara-like student: CS, IELTS 6.5, two documents, interested in AI."""
    profile = {"degree": {"degree_field": "computer science"}, "interest_topics": ["artificial intelligence"],
               "languages": []}
    profile.update(profile_changes)
    return {"profile": profile, "preferences": [], "condition_answers": [],
            "documents": [{"doc_type": "english_test", "details": {"test": "IELTS Academic", "overall": 6.5}},
                          {"doc_type": "recommendation_letter"}]}


def programme(**record_fields) -> dict:
    """A clear programme that accepts CS, IELTS 6.0, with a future deadline; tests change one thing."""
    fields = {"programme_id": "prog", "accepted_degree_fields": ["computer science"],
              "english_requirements": [{"test": "IELTS", "min_overall": 6.0}],
              "deadlines": [{"label": "Scholarship application deadline", "due": "2027-01-11", "quote": "11 Jan"}]}
    fields.update(record_fields)
    return {"programme_id": "prog", "record": ProgrammeRecord.model_validate(fields)}


def test_eligible_when_every_check_passes():
    result = check_eligibility(student(), programme(), TODAY)
    assert result["status"] == "eligible" and result["next_deadline"] == "2027-01-11"


def test_not_eligible_when_degree_field_is_not_accepted():
    result = check_eligibility(student(), programme(accepted_degree_fields=["law"]), TODAY)
    assert result["status"] == "not_eligible" and "not an accepted degree field" in result["blockers"][0]


def test_not_eligible_when_english_score_is_too_low():
    result = check_eligibility(student(), programme(english_requirements=[{"test": "IELTS Academic", "min_overall": 7.0}]), TODAY)
    assert result["status"] == "not_eligible" and "below the minimum 7.0" in result["blockers"][0]


def test_uncertain_when_field_rule_is_vague_or_missing():
    vague = check_eligibility(student(), programme(accepted_degree_fields=["law"], degree_rule_vague=True), TODAY)
    unknown = check_eligibility(student(), programme(accepted_degree_fields=[]), TODAY)
    assert vague["status"] == unknown["status"] == "uncertain"


def test_condition_is_uncertain_until_the_student_answers():
    prog = programme(other_conditions=["top 10% of class"])
    assert check_eligibility(student(), prog, TODAY)["status"] == "uncertain"
    answered = student()
    answered["condition_answers"] = [{"programme_id": "prog", "condition": "top 10% of class", "met": True}]
    assert check_eligibility(answered, prog, TODAY)["status"] == "eligible"
    answered["condition_answers"][0]["met"] = False
    assert check_eligibility(answered, prog, TODAY)["status"] == "not_eligible"


def test_other_language_blocks_only_when_required_for_all():
    french = {"language": "french", "min_level": "B2", "quote": "French B2"}
    for_all = check_eligibility(student(), programme(other_language_requirements=[{**french, "applies_to": "all"}]), TODAY)
    one_track = check_eligibility(student(), programme(other_language_requirements=[{**french, "applies_to": "some_tracks"}]), TODAY)
    speaks_it = check_eligibility(student(languages=[{"language": "french", "level": "C1"}]),
                                  programme(other_language_requirements=[{**french, "applies_to": "all"}]), TODAY)
    assert for_all["status"] == "not_eligible"
    assert one_track["status"] == "eligible" and one_track["notes"]
    assert speaks_it["status"] == "eligible"


def test_deadline_passed_unpublished_or_missing():
    passed = programme(deadlines=[{"label": "Scholarship application deadline", "due": "2022-02-14", "quote": "x"}])
    unpublished = programme(deadlines=[{"label": "Application deadline", "due": None, "quote": "to be announced"},
                                       {"label": "Scholarship application deadline", "due": "2022-02-14", "quote": "x"}])
    assert check_eligibility(student(), passed, TODAY)["status"] == "not_eligible"
    result = check_eligibility(student(), unpublished, TODAY)
    assert result["status"] == "uncertain" and "has passed" in result["notes"][0]
    assert check_eligibility(student(), programme(deadlines=[]), TODAY)["status"] == "uncertain"


def test_deadline_ignores_start_dates_uploads_and_self_funded_track():
    prog = programme(deadlines=[
        {"label": "Grant application start period", "due": "2026-10-20", "quote": "x"},
        {"label": "Deadline for uploading reference letters", "due": "2026-11-01", "quote": "x"},
        {"label": "Self‑funded application deadline", "due": "2026-12-01", "quote": "x"},
        {"label": "Deadline for the grant application period", "due": "2027-01-11", "quote": "x"}])
    assert check_eligibility(student(), prog, TODAY)["next_deadline"] == "2027-01-11"


def test_missing_documents_counts_per_type():
    record = programme(required_documents=[{"doc_type": "recommendation_letter", "count": 2},
                                           {"doc_type": "english_test"}, {"doc_type": "other"}])["record"]
    result = find_missing_documents(student(), record)
    assert result["missing"] == [{"doc_type": "recommendation_letter", "required": 2, "owned": 1, "missing": 1}]


def test_match_counts_broader_topics_and_gives_examples():
    record = ProgrammeRecord(programme_id="prog", courses=[
        Course(name="Statistical Learning", source_url=URL, topics=["machine learning"]),   # broader: AI
        Course(name="Databases", source_url=URL, topics=["data engineering"]),
        Course(name="Academic Writing", source_url=URL)])
    result = match_score(student(), record)
    assert result["matched"] == 1 and result["total"] == 3
    assert result["by_interest"]["artificial intelligence"]["examples"] == ["Statistical Learning"]


def test_exclude_topic_preference_removes_mostly_law_programmes():
    law = ProgrammeRecord(programme_id="law", courses=[
        Course(name="Public International Law", source_url=URL, topics=["international law"]),
        Course(name="Human Rights Law", source_url=URL, topics=["human rights law"]),
        Course(name="Security", source_url=URL, topics=["security studies"])])
    cs = ProgrammeRecord(programme_id="cs", courses=[
        Course(name="Machine Learning", source_url=URL, topics=["machine learning"]),
        Course(name="Law and Data", source_url=URL, topics=["law"]),
        Course(name="Databases", source_url=URL, topics=["data engineering"])])
    sara = student()
    sara["preferences"] = [{"kind": "exclude_topic", "value": "law"}]
    assert excluded_by_preferences(sara, law) is not None
    assert excluded_by_preferences(sara, cs) is None

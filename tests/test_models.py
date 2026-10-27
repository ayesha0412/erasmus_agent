"""Tests for the pydantic models (milestone M3): good data passes, bad data is rejected."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from erasmus_agent.models import Course, Deadline, Profile, ProgrammeRecord, load_cycle, load_profile

DATA = Path(__file__).resolve().parent.parent / "data"


def sara() -> dict:
    """Sara's profile as a plain dict, so a test can break one field at a time."""
    return json.loads((DATA / "profiles" / "stu-sara.json").read_text(encoding="utf-8"))


def test_all_sample_profiles_and_cycles_are_valid():
    profiles = [load_profile(p) for p in sorted((DATA / "profiles").glob("*.json"))]
    cycles = [load_cycle(c) for c in sorted((DATA / "cycles").glob("cycle*.json"))]
    assert {p.student_id for p in profiles} == {"stu-sara", "stu-omar"}
    assert [c.cycle for c in cycles] == [1, 2, 3]


def test_doc_id_defaults_to_doc_type():
    profile = Profile.model_validate(sara())
    assert profile.documents[0].doc_id == profile.documents[0].doc_type


def test_interest_outside_vocabulary_is_rejected():
    data = sara()
    data["interest_topics"] = ["AI"]           # the vocabulary says "artificial intelligence"
    with pytest.raises(ValidationError, match="topic vocabulary"):
        Profile.model_validate(data)


def test_degree_field_outside_vocabulary_is_rejected():
    data = sara()
    data["degree"]["degree_field"] = "computing"
    with pytest.raises(ValidationError, match="degree-field vocabulary"):
        Profile.model_validate(data)


def test_impossible_date_is_rejected():
    data = sara()
    data["documents"][0]["added_on"] = "2026-02-30"
    with pytest.raises(ValidationError):
        Profile.model_validate(data)


def test_cgpa_above_scale_is_rejected():
    data = sara()
    data["grades"] = {"cgpa": 4.4, "scale": 4.0}
    with pytest.raises(ValidationError, match="higher than its scale"):
        Profile.model_validate(data)


def test_unknown_key_is_rejected():
    data = sara()
    data["interst_topics"] = ["robotics"]      # typo
    with pytest.raises(ValidationError):
        Profile.model_validate(data)


def test_english_test_needs_a_score():
    data = sara()
    data["documents"][2]["details"] = {"test": "IELTS Academic"}
    with pytest.raises(ValidationError, match="english_test"):
        Profile.model_validate(data)


def test_course_topics_come_from_vocabulary_or_are_empty():
    url = "https://example.org/courses"
    assert Course(name="Academic Writing", source_url=url).topics == []
    assert Course(name="Statistical Learning", source_url=url, topics=["machine learning"]).topics
    with pytest.raises(ValidationError, match="topic vocabulary"):
        Course(name="Blockchain", source_url=url, topics=["blockchain"])


def test_deadline_must_be_plausible_or_explicitly_unknown():
    assert Deadline(label="application", due=None, quote="Deadlines will be published soon").due is None
    with pytest.raises(ValidationError, match="outside"):
        Deadline(label="application", due="1999-01-11", quote="January 11th, 1999")


def test_language_requirement_needs_a_valid_cefr_level_and_scope():
    record = ProgrammeRecord(programme_id="ilspsd", other_language_requirements=[
        {"language": "french", "min_level": "B2", "applies_to": "some_tracks", "quote": "French ... level B2"}])
    assert record.other_language_requirements[0].min_level == "B2"
    with pytest.raises(ValidationError):
        ProgrammeRecord(programme_id="x", other_language_requirements=[
            {"language": "french", "min_level": "B3", "applies_to": "all", "quote": "q"}])
    with pytest.raises(ValidationError):
        ProgrammeRecord(programme_id="x", other_language_requirements=[
            {"language": "French", "min_level": "B2", "applies_to": "everyone", "quote": "q"}])


def test_programme_record_rejects_unknown_degree_field():
    with pytest.raises(ValidationError, match="degree-field vocabulary"):
        ProgrammeRecord(programme_id="deai", accepted_degree_fields=["data wizardry"])
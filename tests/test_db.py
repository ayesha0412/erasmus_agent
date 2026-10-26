"""Tests for the SQLite memory (milestone M2). Each test gets a fresh, empty database file."""

import pytest

from erasmus_agent import db


@pytest.fixture
def conn(tmp_path):
    """A new database in a temporary folder, with one student and one programme."""
    connection = db.connect(tmp_path / "test.db")
    db.init_db(connection)
    db.add_student(connection, "stu-sara", "Sara", is_fictional=True)
    db.add_programme(connection, "deai", "101241133", "Data Engineering and Artificial Intelligence")
    yield connection
    connection.close()


def test_init_creates_all_tables_and_is_safe_to_rerun(conn):
    db.init_db(conn)  # second run must not fail or lose data
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    expected = {"students", "programmes", "profile_versions", "document_versions", "preference_versions",
                "programme_versions", "courses", "decisions", "checklist_items", "cycles", "action_log"}
    assert expected <= tables
    assert conn.execute("SELECT COUNT(*) FROM students").fetchone()[0] == 1


def test_correction_adds_version_and_keeps_history(conn):
    key = {"student_id": "stu-sara", "doc_id": "english_test"}
    v1 = db.add_version(conn, "document_versions", key, {"overall": 6.5}, "onboarding", extra={"doc_type": "english_test"})
    v2 = db.add_version(conn, "document_versions", key, {"overall": 7.0}, "cycle 2", extra={"doc_type": "english_test"})

    current = db.get_current(conn, "document_versions", key)
    history = db.get_history(conn, "document_versions", key)

    assert current["id"] == v2 and current["data"] == {"overall": 7.0}
    assert [h["version"] for h in history] == [1, 2]
    assert history[0]["is_current"] == 0 and history[0]["superseded_by"] == v2  # old version kept, marked
    assert v1 != v2


def test_saving_the_same_data_twice_creates_no_new_version(conn):
    key = {"student_id": "stu-sara", "doc_id": "cv"}
    first = db.add_version(conn, "document_versions", key, {"pages": 2}, "onboarding", extra={"doc_type": "cv"})
    retry = db.add_version(conn, "document_versions", key, {"pages": 2}, "onboarding", extra={"doc_type": "cv"})
    assert first == retry
    assert len(db.get_history(conn, "document_versions", key)) == 1


def test_pending_version_is_not_used_until_approved(conn):
    key = {"programme_id": "deai"}
    old = db.add_version(conn, "programme_versions", key, {"deadline": "2027-01-11"}, "cycle 1")
    new = db.add_version(conn, "programme_versions", key, {"deadline": "2027-01-25"}, "cycle 2", status="pending")

    assert db.get_current(conn, "programme_versions", key)["id"] == old  # decisions still use the approved one

    db.approve_version(conn, "programme_versions", new)
    history = db.get_history(conn, "programme_versions", key)
    assert db.get_current(conn, "programme_versions", key)["data"] == {"deadline": "2027-01-25"}
    assert history[0]["superseded_by"] == new and history[0]["data"] == {"deadline": "2027-01-11"}


def test_duplicate_checklist_item_gives_one_row(conn):
    key = db.idempotency_key("stu-sara", "deai", "recommendation_letter", 1)
    row = {"student_id": "stu-sara", "programme_id": "deai", "item": "recommendation_letter",
           "cycle": 1, "idempotency_key": key}
    assert db.insert_once(conn, "checklist_items", row) is True
    assert db.insert_once(conn, "checklist_items", row) is False  # a retry
    assert conn.execute("SELECT COUNT(*) FROM checklist_items").fetchone()[0] == 1


def test_records_are_separated_by_student(conn):
    db.add_student(conn, "stu-omar", "Omar", is_fictional=True)
    db.add_version(conn, "document_versions", {"student_id": "stu-sara", "doc_id": "passport"}, {}, "onboarding",
                   extra={"doc_type": "passport"})
    assert db.get_current(conn, "document_versions", {"student_id": "stu-omar", "doc_id": "passport"}) is None


def test_unknown_table_or_wrong_key_is_rejected(conn):
    with pytest.raises(ValueError):
        db.add_version(conn, "students", {"student_id": "stu-sara"}, {}, "test")
    with pytest.raises(ValueError):
        db.add_version(conn, "document_versions", {"student_id": "stu-sara"}, {}, "test")  # doc_id missing
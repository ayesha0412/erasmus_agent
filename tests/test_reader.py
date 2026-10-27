"""Tests for read_programme (milestone M4): same pages -> not changed, changed copy -> changed."""

import pytest

from erasmus_agent import db
from erasmus_agent.reader import fingerprint, read_programme

CHANGED_COPY = {"deai/admission": "data/pages/test/deai/admission_deadline_changed.html"}


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.db")
    db.init_db(connection)
    db.add_programme(connection, "deai", "101241133", "Data Engineering and Artificial Intelligence")
    yield connection
    connection.close()


def remember(conn, reading):
    """Save a programme version holding this reading's fingerprint, as extraction would."""
    db.add_version(conn, "programme_versions", {"programme_id": "deai"}, {"note": "test"}, "test",
                   extra={"page_fingerprint": reading["fingerprint"]})


def test_first_read_is_marked_and_not_changed(conn):
    reading = read_programme(conn, "deai")
    assert reading["first_read"] is True and reading["changed"] is False
    assert [p["page"] for p in reading["pages"]] == ["courses", "admission", "profiles"]
    assert "<td>" not in reading["pages"][1]["text"]       # HTML cleaned to text


def test_same_pages_twice_are_not_changed(conn):
    first = read_programme(conn, "deai")
    remember(conn, first)
    second = read_programme(conn, "deai")
    assert second["fingerprint"] == first["fingerprint"]
    assert second["changed"] is False


def test_changed_copy_is_detected(conn):
    remember(conn, read_programme(conn, "deai"))
    reading = read_programme(conn, "deai", CHANGED_COPY)
    admission = next(p for p in reading["pages"] if p["page"] == "admission")
    assert reading["changed"] is True
    assert admission["test_copy"] is True and "December 18th, 2026" in admission["text"]


def test_fingerprint_ignores_whitespace_only_differences():
    assert fingerprint(["Deadline:  11 January\r\n"]) == fingerprint(["Deadline: 11 January"])
    assert fingerprint(["Deadline: 11 January"]) != fingerprint(["Deadline: 25 January"])


def test_unknown_programme_is_rejected(conn):
    with pytest.raises(ValueError):
        read_programme(conn, "not-a-programme")

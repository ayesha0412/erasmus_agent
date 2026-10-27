"""Tests for the command-line interface (milestone M9), each on its own temporary database."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from erasmus_agent import db
from erasmus_agent.cli import app
from erasmus_agent.extract import save_programme_record
from erasmus_agent.memory import get_programme
from erasmus_agent.models import ProgrammeRecord
from erasmus_agent.reader import read_programme

ROOT = Path(__file__).resolve().parent.parent
runner = CliRunner()


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "cli.db"
    assert runner.invoke(app, ["--db", str(path), "init"]).exit_code == 0
    assert runner.invoke(app, ["--db", str(path), "onboard", "--from", str(ROOT / "data/profiles/stu-sara.json")]).exit_code == 0
    return path


def run(database, *args):
    result = runner.invoke(app, ["--db", str(database), *args])
    assert result.exit_code == 0, result.output
    return result.output


def test_init_loads_the_five_saved_records_and_is_safe_to_repeat(database):
    output = run(database, "init")
    conn = db.connect(database)
    assert conn.execute("SELECT COUNT(*) FROM programme_versions").fetchone()[0] == 5
    assert "deai" in output and "version 1" in output


def test_status_shows_state_evidence_and_next_action(database):
    output = run(database, "status", "--student", "stu-sara", "--date", "2026-10-25")
    assert "deai       ELIGIBLE" in output and "deadline 2027-01-11" in output
    assert "APPROVALS NEEDED: none" in output and "run-cycle 1" in output


def test_programme_correction_keeps_history_and_survives_re_extraction(database):
    output = run(database, "correct", "--file", str(ROOT / "data/corrections/ilspsd-french-scope.json"))
    assert "version 1 -> 2" in output
    conn = db.connect(database)
    current = get_programme(conn, "ilspsd")
    assert current["record"].other_language_requirements[0].applies_to == "some_tracks"
    assert len(current["record"].courses) == 32                                  # courses carried over
    history = db.get_history(conn, "programme_versions", {"programme_id": "ilspsd"})
    assert history[0]["data"]["other_language_requirements"][0]["applies_to"] == "all"   # old version kept

    # A later extraction still says "all", but the stored correction is applied before saving.
    raw = ProgrammeRecord.model_validate_json((ROOT / "outputs/records/ilspsd.json").read_text(encoding="utf-8"))
    raw.other_conditions = ["something new on the page"]
    reading = {**read_programme(conn, "ilspsd"), "changed": True, "fingerprint": "new"}
    pending_id, status = save_programme_record(conn, reading, raw, "re-extraction")
    pending = db.get_history(conn, "programme_versions", {"programme_id": "ilspsd"})[-1]
    assert status == "pending" and pending["data"]["other_language_requirements"][0]["applies_to"] == "some_tracks"


def test_student_correction_adds_a_profile_version(database):
    run(database, "correct", "--student", "stu-sara", "--field", "grades.cgpa", "--value", "3.5",
        "--reason", "transcript re-checked")
    history = db.get_history(db.connect(database), "profile_versions", {"student_id": "stu-sara"})
    assert [h["data"]["grades"]["cgpa"] for h in history] == [3.4, 3.5]


def test_invalid_correction_is_rejected_and_nothing_changes(database):
    result = runner.invoke(app, ["--db", str(database), "correct", "--student", "stu-sara", "--field", "grades.cgpa",
                                 "--value", "9", "--reason", "typo"])
    assert result.exit_code != 0                                                 # 9 > scale 4.0
    assert len(db.get_history(db.connect(database), "profile_versions", {"student_id": "stu-sara"})) == 1


def test_forget_deletes_everything_about_the_student(database):
    run(database, "forget", "stu-sara", "--yes")
    conn = db.connect(database)
    for table in ("students", "profile_versions", "document_versions"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM programme_versions").fetchone()[0] == 5   # shared data kept


def test_onboarding_rejects_an_invalid_profile(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text((ROOT / "data/profiles/stu-sara.json").read_text(encoding="utf-8").replace('"artificial intelligence"', '"AI"'))
    result = runner.invoke(app, ["--db", str(tmp_path / "x.db"), "onboard", "--from", str(bad)])
    assert result.exit_code == 1 and "not valid" in result.output

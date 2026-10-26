"""Tests for the agent loop (milestone M7) with scripted decisions instead of the LLM.

The loop, the tool rules, logging, checkpoints and memory are all real; only the choice
of the next tool is scripted, so each test is free, fast and repeatable.
"""

import json
from pathlib import Path

import pytest

from erasmus_agent import db, tools
from erasmus_agent.agent import ScriptedInteraction, run_cycle
from erasmus_agent.extract import save_programme_record
from erasmus_agent.memory import save_profile
from erasmus_agent.models import Deadline, ProgrammeRecord, load_cycle, load_profile
from erasmus_agent.reader import read_programme

ROOT = Path(__file__).resolve().parent.parent
CYCLE1, CYCLE2 = load_cycle(ROOT / "data/cycles/cycle1.json"), load_cycle(ROOT / "data/cycles/cycle2.json")


def script(*calls):
    """A stand-in for the LLM that returns these (tool, arguments) pairs in order."""
    queue = list(calls)

    def decide(ctx, model):
        name, args = queue.pop(0)
        return name, json.dumps(args), 0, "main"
    return decide


def count(conn, sql):
    return conn.execute(sql).fetchone()[0]


def settle_reply(verdicts):
    """A stand-in for the model that settles conditions from profile facts."""
    def chat_json(messages, schema_name, schema, model="main", reasoning_effort="low"):
        return json.dumps({"verdicts": verdicts}), 0
    return chat_json


@pytest.fixture(autouse=True)
def nothing_settled_by_default(monkeypatch):
    """Unless a test says otherwise, the model settles no conditions (and no network is used)."""
    monkeypatch.setattr(tools, "chat_json", settle_reply([]))


@pytest.fixture
def conn(tmp_path):
    """A database with Sara onboarded and three programmes saved from the extracted records."""
    connection = db.connect(tmp_path / "test.db")
    db.init_db(connection)
    save_profile(connection, load_profile(ROOT / "data/profiles/stu-sara.json"), "onboarding")
    for pid, name in [("deai", "DEAI"), ("codas", "CoDaS"), ("ilspsd", "ILSPSD")]:
        db.add_programme(connection, pid, pid, name)
        record = ProgrammeRecord.model_validate_json((ROOT / f"outputs/records/{pid}.json").read_text(encoding="utf-8"))
        save_programme_record(connection, read_programme(connection, pid), record, "test setup")
    yield connection
    connection.close()


def test_full_cycle_writes_the_plan_once(conn):
    decide = script(("read_programmes", {"programme_ids": ["deai", "codas"]}),
                    ("check_programmes", {"programme_ids": ["deai", "codas"]}),
                    ("request_approval", {"kind": "shortlist", "programme_ids": ["deai"], "reason": "eligible"}),
                    ("find_documents", {"programme_ids": ["deai"]}),
                    ("write_plan", {"programme_ids": ["deai"]}),
                    ("finish", {"summary": "DEAI shortlisted"}))
    outcome = run_cycle(conn, CYCLE1, ScriptedInteraction(), decide=decide)
    items = count(conn, "SELECT COUNT(*) FROM checklist_items")
    assert outcome["status"] == "done" and items > 0
    assert count(conn, "SELECT COUNT(*) FROM decisions WHERE kind = 'shortlist'") == 1
    assert count(conn, "SELECT COUNT(*) FROM action_log WHERE error IS NULL") == 6

    again = run_cycle(conn, CYCLE1, ScriptedInteraction(), decide=script())      # rerun after it finished
    assert again["already_finished"] and count(conn, "SELECT COUNT(*) FROM checklist_items") == items


def test_plan_before_approval_is_refused_and_logged(conn):
    decide = script(("write_plan", {"programme_ids": ["deai"]}), ("finish", {"summary": "stopped early"}))
    outcome = run_cycle(conn, CYCLE1, ScriptedInteraction(), decide=decide)
    error = conn.execute("SELECT error FROM action_log WHERE step = 1").fetchone()["error"]
    assert outcome["status"] == "done" and "not approved a shortlist" in error
    assert count(conn, "SELECT COUNT(*) FROM checklist_items") == 0


def test_not_eligible_programme_cannot_be_shortlisted(conn):
    decide = script(("check_programmes", {"programme_ids": ["ilspsd"]}),
                    ("request_approval", {"kind": "shortlist", "programme_ids": ["ilspsd"], "reason": "x"}),
                    ("finish", {"summary": "done"}))
    interaction = ScriptedInteraction()
    run_cycle(conn, CYCLE1, interaction, decide=decide)
    assert interaction.approvals == []                          # the student was never even asked
    assert "cannot be shortlisted" in conn.execute("SELECT error FROM action_log WHERE step = 2").fetchone()["error"]


def test_changed_page_waits_for_approval(conn, monkeypatch):
    def fake_extraction(reading, model):
        record = ProgrammeRecord.model_validate_json((ROOT / "outputs/records/deai.json").read_text(encoding="utf-8"))
        record.deadlines = [Deadline(label="Deadline for the grant application period", due="2027-01-25", quote="January 25th, 2027")]
        return record, {"calls": 0, "tokens": 0, "failed_chunks": 0}
    monkeypatch.setattr(tools, "run_extraction", fake_extraction)

    decide = script(("read_programmes", {"programme_ids": ["deai"]}),
                    ("extract_programme", {"programme_id": "deai"}),
                    ("check_programmes", {"programme_ids": ["deai"]}),
                    ("request_approval", {"kind": "shortlist", "programme_ids": ["deai"], "reason": "eligible"}),
                    ("write_plan", {"programme_ids": ["deai"]}),                       # refused: change pending
                    ("request_approval", {"kind": "programme_change", "programme_ids": ["deai"], "reason": "deadline moved"}),
                    ("finish", {"summary": "deadline change approved"}))
    interaction = ScriptedInteraction()
    run_cycle(conn, CYCLE2, interaction, decide=decide)

    readings = json.loads(conn.execute("SELECT result FROM action_log WHERE step = 1").fetchone()["result"])
    assert readings["deai"]["changed_since_saved_version"] is True
    assert "waiting for approval" in conn.execute("SELECT error FROM action_log WHERE step = 5").fetchone()["error"]
    assert "2027-01-25" in interaction.approvals[-1]                                # the student saw what changed
    history = db.get_history(conn, "programme_versions", {"programme_id": "deai"})
    assert [h["status"] for h in history] == ["approved", "approved"] and history[0]["is_current"] == 0


def test_answer_is_remembered_and_never_asked_again(conn):
    condition = ProgrammeRecord.model_validate_json(
        (ROOT / "outputs/records/codas.json").read_text(encoding="utf-8")).other_conditions[0]
    ask = ("ask_student", {"programme_id": "codas", "condition": condition})
    interaction = ScriptedInteraction(answers={"Master": "yes"})
    run_cycle(conn, CYCLE1, interaction, decide=script(ask, ("finish", {"summary": "asked"})))
    run_cycle(conn, CYCLE2, interaction, decide=script(ask, ("finish", {"summary": "not asked again"})))
    assert len(interaction.asked) == 1
    assert "already answered" in conn.execute("SELECT error FROM action_log WHERE cycle = 2 AND step = 1").fetchone()["error"]


def test_repeated_refusals_stop_the_cycle(conn):
    bad = ("write_plan", {"programme_ids": ["deai"]})
    outcome = run_cycle(conn, CYCLE1, ScriptedInteraction(), decide=script(bad, bad, bad))
    assert outcome["status"] == "stopped" and outcome["steps"] == 3


def test_out_of_llm_budget_stops_cleanly_and_can_resume(conn):
    from test_llm import limit_error

    def no_budget(ctx, model):
        raise limit_error("Rate limit reached ... on tokens per day (TPD)")
    outcome = run_cycle(conn, CYCLE1, ScriptedInteraction(), decide=no_budget)
    assert outcome["status"] == "stopped" and "rate limit" in outcome["reason"]
    resumed = run_cycle(conn, CYCLE1, ScriptedInteraction(), decide=script(("finish", {"summary": "resumed"})))
    assert resumed["status"] == "done"


def test_unknown_tool_and_unknown_programme_are_refused(conn):
    decide = script(("send_email", {"to": "x@example.com"}), ("check_programmes", {"programme_ids": ["madeup"]}),
                    ("finish", {"summary": "done"}))
    outcome = run_cycle(conn, CYCLE1, ScriptedInteraction(), decide=decide)
    errors = [row["error"] for row in conn.execute("SELECT error FROM action_log ORDER BY step")]
    assert "not a tool" in errors[0] and "unknown" in errors[1] and outcome["status"] == "done"


def codas_condition() -> str:
    return ProgrammeRecord.model_validate_json(
        (ROOT / "outputs/records/codas.json").read_text(encoding="utf-8")).other_conditions[0]


def test_condition_settled_from_profile_is_not_asked(conn, monkeypatch):
    verdict = {"condition": codas_condition(), "verdict": "met", "facts_used": ["degree.level"],
               "reason": "bachelor only, no previous master's"}
    monkeypatch.setattr(tools, "chat_json", settle_reply([verdict]))
    interaction = ScriptedInteraction()
    run_cycle(conn, CYCLE1, interaction, decide=script(("check_programmes", {"programme_ids": ["codas"]}),
                                                       ("finish", {"summary": "done"})))
    result = json.loads(conn.execute("SELECT result FROM action_log WHERE step = 1").fetchone()["result"])
    stored = conn.execute("SELECT decided_by, data FROM decisions WHERE kind = 'condition_answer'").fetchone()
    assert result["codas"]["status"] == "eligible" and interaction.asked == []
    assert stored["decided_by"] == "agent" and "degree.level=bachelor" in json.loads(stored["data"])["reason"]


def test_verdict_without_a_real_fact_is_ignored(conn, monkeypatch):
    unbacked = [{"condition": codas_condition(), "verdict": "met", "facts_used": [], "reason": "probably fine"},
                {"condition": codas_condition(), "verdict": "met", "facts_used": ["degree.honours"], "reason": "made up"}]
    monkeypatch.setattr(tools, "chat_json", settle_reply(unbacked))
    run_cycle(conn, CYCLE1, ScriptedInteraction(), decide=script(("check_programmes", {"programme_ids": ["codas"]}),
                                                                 ("finish", {"summary": "done"})))
    result = json.loads(conn.execute("SELECT result FROM action_log WHERE step = 1").fetchone()["result"])
    assert result["codas"]["status"] == "uncertain" and result["codas"]["unanswered_conditions"]
    assert count(conn, "SELECT COUNT(*) FROM decisions WHERE kind = 'condition_answer'") == 0

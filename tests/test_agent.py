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
    def chat_json(messages, schema_name, schema, model="main", reasoning_effort="low", **kwargs):
        return json.dumps({"verdicts": verdicts}), 0
    return chat_json


@pytest.fixture(autouse=True)
def nothing_settled_by_default(monkeypatch, tmp_path):
    """Unless a test says otherwise, the model settles no conditions (no network), and files go to a temp folder."""
    monkeypatch.setattr(tools, "chat_json", settle_reply([]))
    monkeypatch.setattr("erasmus_agent.exports.PLANS_DIR", tmp_path / "plans")


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
        record.deadlines = [Deadline(label="Deadline for the grant application period", due="2026-12-18", quote="December 18th, 2026")]
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
    assert "2026-12-18" in interaction.approvals[-1]                                # the student saw what changed
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


# ---------- M8: approved plan, Excel and calendar ----------

PLAN_STEPS = (("check_programmes", {"programme_ids": ["deai"]}),
              ("request_approval", {"kind": "shortlist", "programme_ids": ["deai"], "reason": "eligible"}),
              ("write_plan", {"programme_ids": ["deai"]}))


def ics_uids(path) -> list[str]:
    return [line.split(":", 1)[1] for line in Path(path).read_text().splitlines() if line.startswith("UID:")]


def test_plan_needs_approval_and_a_second_run_writes_no_duplicates(conn, tmp_path):
    first = run_cycle(conn, CYCLE1, ScriptedInteraction(), decide=script(*PLAN_STEPS, ("finish", {"summary": "x"})))
    rows = count(conn, "SELECT COUNT(*) FROM checklist_items")
    result = json.loads(conn.execute("SELECT result FROM action_log WHERE step = 3").fetchone()["result"])
    uids = ics_uids(result["files"]["calendar"])
    assert first["status"] == "done" and rows == result["added"] > 0
    assert "Shortlist: deai (deadline 2027-01-11)" in first["facts"]           # facts come from the database

    # Same plan again in a later cycle: nothing new, files regenerated with the same calendar ids.
    run_cycle(conn, CYCLE2.model_copy(update={"page_overrides": {}}), ScriptedInteraction(),
              decide=script(*PLAN_STEPS, ("finish", {"summary": "y"})))
    again = json.loads(conn.execute("SELECT result FROM action_log WHERE cycle = 2 AND step = 3").fetchone()["result"])
    assert count(conn, "SELECT COUNT(*) FROM checklist_items") == rows
    assert again["changes"].startswith("none") and ics_uids(again["files"]["calendar"]) == uids


def test_rejected_plan_writes_nothing(conn):
    interaction = ScriptedInteraction()
    interaction.approve = lambda message: "shortlist" in message.lower()      # approve the shortlist, not the plan
    run_cycle(conn, CYCLE1, interaction, decide=script(*PLAN_STEPS, ("finish", {"summary": "x"})))
    assert count(conn, "SELECT COUNT(*) FROM checklist_items") == 0


def test_new_document_closes_items_and_new_deadline_moves_them(conn):
    run_cycle(conn, CYCLE1, ScriptedInteraction(), decide=script(*PLAN_STEPS, ("finish", {"summary": "x"})))
    before = {r["item"] for r in conn.execute("SELECT item FROM checklist_items WHERE status = 'open'")}
    letter = {"doc_type": "recommendation_letter", "version": 1, "added_on": "2026-12-15", "details": {"referee": "R1"}}
    record = ProgrammeRecord.model_validate_json((ROOT / "outputs/records/deai.json").read_text(encoding="utf-8"))
    record.deadlines = [Deadline(label="Deadline for the grant application period", due="2026-12-18", quote="18 Dec")]
    db.add_version(conn, "programme_versions", {"programme_id": "deai"}, record.model_dump(mode="json", exclude={"courses"}),
                   "test: approved deadline change", extra={"page_fingerprint": read_programme(conn, "deai")["fingerprint"]})
    cycle3 = type(CYCLE1).model_validate({**CYCLE1.model_dump(), "cycle": 3, "new_documents": [letter]})
    run_cycle(conn, cycle3, ScriptedInteraction(), decide=script(*PLAN_STEPS, ("finish", {"summary": "z"})))
    open_rows = {r["item"]: r["due_date"] for r in conn.execute("SELECT item, due_date FROM checklist_items WHERE status = 'open'")}
    done = {r["item"] for r in conn.execute("SELECT item FROM checklist_items WHERE status = 'done'")}
    assert "obtain 2 more recommendation_letter" in before and "obtain 2 more recommendation_letter" in done
    assert open_rows["obtain 1 more recommendation_letter"] == "2026-12-18"
    assert open_rows["submit the application"] == "2026-12-18"                 # carried forward, rescheduled
    assert count(conn, "SELECT COUNT(*) FROM checklist_items WHERE item = 'submit the application'") == 1


# ---------- M10 robustness ----------

def test_reply_without_a_tool_call_is_asked_again(conn, monkeypatch):
    from erasmus_agent import agent
    replies = [(None, "I think we should check first.", 5, "local"), ("finish", '{"summary": "ok"}', 5, "local")]
    monkeypatch.setattr(agent, "chat_tools", lambda messages, tools, model: replies.pop(0))
    outcome = run_cycle(conn, CYCLE1, ScriptedInteraction())
    assert outcome["status"] == "done" and outcome["steps"] == 1


def test_immediate_identical_repeat_is_refused(conn):
    check = ("check_programmes", {"programme_ids": ["deai"]})
    run_cycle(conn, CYCLE1, ScriptedInteraction(), decide=script(check, check, ("finish", {"summary": "x"})))
    assert "you just made exactly this call" in conn.execute("SELECT error FROM action_log WHERE step = 2").fetchone()["error"]


# ---------- M11: run log ----------

def test_run_log_records_a_full_cycle(conn):
    from erasmus_agent import logbook
    run_cycle(conn, CYCLE1, ScriptedInteraction(), decide=script(*PLAN_STEPS, ("finish", {"summary": "x"})))
    events = logbook.read_events("stu-sara", 1)
    kinds = [e["event"] for e in events]
    assert kinds.index("cycle_start") < kinds.index("tool_call") and kinds[-1] == "cycle_end"
    assert {"memory_read", "memory_write", "tool_call"} <= set(kinds)
    tool_calls = [e for e in events if e["event"] == "tool_call"]
    assert [e["tool"] for e in tool_calls] == ["check_programmes", "request_approval", "write_plan", "finish"]
    assert all({"ts", "run_id", "latency_ms", "tokens", "model"} <= e.keys() for e in tool_calls)
    assert any(e["event"] == "memory_write" and e["table"] == "checklist_items" for e in events)


def test_forget_removes_the_students_log_lines(conn):
    from erasmus_agent import logbook
    run_cycle(conn, CYCLE1, ScriptedInteraction(), decide=script(("finish", {"summary": "x"})))
    assert logbook.read_events("stu-sara")
    removed = logbook.forget_student("stu-sara")
    assert removed > 0 and logbook.read_events("stu-sara") == []


# ---------- M12: failure and safe retry ----------

def test_crash_during_write_then_retry_gives_no_duplicates(conn):
    from erasmus_agent import logbook
    from erasmus_agent.exports import duplicate_items
    write = ("write_plan", {"programme_ids": ["deai"]})
    interaction = ScriptedInteraction()
    outcome = run_cycle(conn, CYCLE1, interaction, fail_once={"write_plan"},
                        decide=script(*PLAN_STEPS, write, ("finish", {"summary": "retried"})))

    failed = conn.execute("SELECT error FROM action_log WHERE step = 3").fetchone()["error"]
    retry = json.loads(conn.execute("SELECT result FROM action_log WHERE step = 4").fetchone()["result"])
    total = count(conn, "SELECT COUNT(*) FROM checklist_items")
    assert outcome["status"] == "done" and failed.startswith("ToolFailed: simulated crash after saving 1 of")
    assert retry["added"] == total - 1                     # the retry added only what was missing
    assert duplicate_items(conn, "stu-sara") == 0
    assert "+ deai" in interaction.approvals[-1] and interaction.approvals[-1].count("+ deai") == total - 1

    # A third attempt (e.g. a retry after a restart) writes nothing and asks nothing.
    run_cycle(conn, CYCLE2.model_copy(update={"page_overrides": {}}), ScriptedInteraction(),
              decide=script(*PLAN_STEPS, ("finish", {"summary": "again"})))
    assert count(conn, "SELECT COUNT(*) FROM checklist_items") == total and duplicate_items(conn, "stu-sara") == 0
    writes = [e for e in logbook.read_events("stu-sara") if e["event"] == "memory_write" and e["table"] == "checklist_items"]
    assert sum(e["result"] == "inserted" for e in writes) == total

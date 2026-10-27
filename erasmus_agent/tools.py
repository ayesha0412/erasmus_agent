"""The agent's tools (milestone M7): what the LLM may ask for, and the rules around each one.

The LLM only chooses WHICH tool to call next and with which arguments. Every tool is plain
code (extract_programme runs the validated extractor), and every permission rule is checked
here, so a confused or manipulated model cannot skip an approval or invent an action.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from . import db
from .config import DEFAULT_MODEL, EXTRACT_MODEL
from .extract import extract_programme as run_extraction
from .exports import export_plan, facts_summary
from .extract import save_programme_record
from .llm import chat_json
from .memory import get_programme, lookup_profile, record_condition_answer
from .models import CycleInput, ProgrammeRecord
from .reader import read_programme
from .rules import application_deadlines, check_eligibility, excluded_by_preferences, find_missing_documents, match_score


class ToolRefused(Exception):
    """A tool call broke a rule (e.g. writing a plan before approval). The agent is told why and continues."""


class ToolFailed(Exception):
    """A tool crashed while working (e.g. a simulated failure). Its writes are idempotent, so a retry is safe."""


@dataclass
class Context:
    """Everything a tool may use during one cycle. `state` is the cycle's working memory (checkpointed)."""
    conn: sqlite3.Connection
    cycle: CycleInput
    interact: object                    # has ask(question) -> str and approve(message) -> bool
    state: dict = field(default_factory=dict)
    model: str = DEFAULT_MODEL           # the model that makes decisions (also settles conditions)
    extract_model: str = EXTRACT_MODEL
    plans_dir: Path | None = None        # where Excel and .ics go (None = outputs/plans)
    fail_once: set = field(default_factory=set)   # tools that crash once part-way (failure simulation, M12)

    @property
    def student_id(self) -> str:
        return self.cycle.student_id


# ---------- Helpers ----------

def known_programmes(ctx: Context, programme_ids: list[str]) -> list[str]:
    """Refuse ids that are not registered programmes (the model cannot invent one)."""
    known = {row["programme_id"] for row in ctx.conn.execute("SELECT programme_id FROM programmes")}
    unknown = [pid for pid in programme_ids if pid not in known]
    if unknown or not programme_ids:
        raise ToolRefused(f"unknown or missing programme ids {unknown}; known ids: {sorted(known)}")
    return programme_ids


def describe_change(old: ProgrammeRecord, new: ProgrammeRecord) -> list[str]:
    """Plain-code summary of what changed between two versions, for the approval message."""
    def deadlines(record):
        return sorted(f"{d.label}: {d.due}" for d in application_deadlines(record))
    changes = []
    for label, before, after in [
        ("application deadlines", deadlines(old), deadlines(new)),
        ("accepted degree fields", sorted(old.accepted_degree_fields), sorted(new.accepted_degree_fields)),
        ("English requirements", sorted(f"{e.test} {e.min_overall}" for e in old.english_requirements),
         sorted(f"{e.test} {e.min_overall}" for e in new.english_requirements)),
    ]:
        if before != after:
            changes.append(f"{label}: {before} -> {after}")
    return changes or ["the page text changed, but no key requirement changed"]


# ---------- Settling admission conditions from the profile ----------

SETTLE_PROMPT = """You check whether a student meets admission conditions, using ONLY the facts given.
For each condition answer:
- "met" if the condition is NO obstacle for this student: either it is satisfied, or it only applies to
  other situations (e.g. "if you have not completed your degree ...") and the facts show it does not apply;
- "not_met" if the condition stops this student from being admitted;
- give "met" or "not_met" only if one or more of the listed facts clearly settles it, and name those facts
  in facts_used (copy their keys exactly);
- "unknown" if the facts do not settle it. Never assume knowledge, skills, class rank or anything else
  that is not in the facts; the student will be asked about unknown conditions."""

SETTLE_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {"condition": {"type": "string"},
                       "verdict": {"type": "string", "enum": ["met", "not_met", "unknown"]},
                       "facts_used": {"type": "array", "items": {"type": "string"}},
                       "reason": {"type": "string"}},
        "required": ["condition", "verdict", "facts_used", "reason"], "additionalProperties": False}}},
    "required": ["verdicts"], "additionalProperties": False,
}


def profile_facts(student: dict, ctx: Context) -> dict:
    """The facts the model may use, as simple key -> value pairs (nothing else about the student)."""
    profile = student["profile"]
    facts = {
        "degree.level": profile["degree"]["level"],
        "degree.field": profile["degree"]["degree_field"],
        "degree.graduation_year": profile["degree"]["graduation_year"],
        "degree.ects": profile["degree"]["ects"],
        "grades.cgpa": f"{profile['grades']['cgpa']} out of {profile['grades']['scale']}",
        "languages": ", ".join(f"{l['language']} {l['level']}" for l in profile.get("languages", [])) or "none listed",
        "documents_owned": ", ".join(d["doc_type"] for d in student["documents"]),
        "today": ctx.cycle.date.isoformat(),
    }
    for document in student["documents"]:
        if document["doc_type"] == "english_test":
            facts["english_test"] = f"{document['details']['test']} {document['details']['overall']}"
    return facts


def settle_conditions_from_profile(ctx: Context, programme_id: str, conditions: list[str]) -> list[dict]:
    """Ask the LLM which conditions the profile facts settle; code accepts a verdict only if it cites real facts.

    Settled conditions are saved as answers decided by the agent (with the reason), so they are
    not worked out again; the rest stay open for ask_student.
    """
    if not conditions:
        return []
    student = lookup_profile(ctx.conn, ctx.student_id)
    facts = profile_facts(student, ctx)
    message = json.dumps({"facts": facts, "conditions": conditions}, ensure_ascii=False)
    reply, tokens = chat_json([{"role": "system", "content": SETTLE_PROMPT}, {"role": "user", "content": message}],
                              "condition_verdicts", SETTLE_SCHEMA, model=ctx.model, purpose="settle_conditions")
    ctx.state["llm_tokens_decisions"] = ctx.state.get("llm_tokens_decisions", 0) + tokens
    try:
        verdicts = json.loads(reply)["verdicts"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []                                   # nothing settled; the student can still be asked
    settled = []
    for verdict in verdicts:
        cited = [f for f in verdict.get("facts_used", []) if f in facts]
        if verdict.get("condition") not in conditions or verdict.get("verdict") not in ("met", "not_met") or not cited:
            continue                                # unknown, or not backed by a real fact: leave it open
        met = verdict["verdict"] == "met"
        reason = f"{verdict['reason']} (facts: {', '.join(f'{k}={facts[k]}' for k in cited)})"
        record_condition_answer(ctx.conn, ctx.student_id, ctx.cycle.cycle, programme_id, verdict["condition"],
                                met, decided_by="agent", reason=reason)
        settled.append({"condition": verdict["condition"], "met": met, "reason": reason})
    return settled


# ---------- Tools ----------

def read_programmes(ctx: Context, programme_ids: list[str]) -> dict:
    """Evidence retrieval: read the saved pages and report which ones are new or changed."""
    results = {}
    for pid in known_programmes(ctx, programme_ids):
        reading = read_programme(ctx.conn, pid, ctx.cycle.page_overrides)
        stored = get_programme(ctx.conn, pid)
        results[pid] = {
            "first_read": reading["first_read"],
            "changed_since_saved_version": reading["changed"],
            "pending_change_waiting_for_approval": bool(stored and stored["pending_version_id"]),
            "test_copy_used": any(p["test_copy"] for p in reading["pages"]),
            "pages": [p["url"] for p in reading["pages"]],
        }
    ctx.state.setdefault("readings", {}).update(results)
    return results


def extract_programme(ctx: Context, programme_id: str) -> dict:
    """Run the validated LLM extractor on a new or changed programme and save it as a version."""
    known_programmes(ctx, [programme_id])
    reading = read_programme(ctx.conn, programme_id, ctx.cycle.page_overrides)
    stored = get_programme(ctx.conn, programme_id)
    if not (reading["first_read"] or reading["changed"]):
        raise ToolRefused(f"{programme_id}: pages unchanged since the saved version; no extraction needed")
    if stored and stored["pending_version_id"]:
        pending = ctx.conn.execute("SELECT page_fingerprint FROM programme_versions WHERE id = ?",
                                   [stored["pending_version_id"]]).fetchone()
        if pending["page_fingerprint"] == reading["fingerprint"]:
            raise ToolRefused(f"{programme_id}: this change is already extracted and waiting for approval")
    record, stats = run_extraction(reading, model=ctx.extract_model)
    version_id, status = save_programme_record(ctx.conn, reading, record, source=f"cycle {ctx.cycle.cycle}")
    ctx.state["llm_tokens_extraction"] = ctx.state.get("llm_tokens_extraction", 0) + stats["tokens"]
    result = {"programme_id": programme_id, "saved_as": status, "version_id": version_id,
              "courses": len(record.courses), "failed_chunks": stats["failed_chunks"],
              "suspected_instructions": record.suspected_instructions}
    if status == "pending" and stored:
        result["what_changed"] = describe_change(stored["record"], record)
    if record.suspected_instructions:
        ctx.state.setdefault("flagged_instructions", {})[programme_id] = record.suspected_instructions
    return result


def check_programmes(ctx: Context, programme_ids: list[str]) -> dict:
    """Eligibility, course match and preference check for each programme.

    Conditions the profile can settle are settled first (once, stored); the checks themselves are plain code.
    """
    results = {}
    for pid in known_programmes(ctx, programme_ids):
        programme = get_programme(ctx.conn, pid)
        if programme is None:
            results[pid] = {"status": "no_saved_record", "hint": "call extract_programme first"}
            continue
        answered = {a["condition"] for a in lookup_profile(ctx.conn, ctx.student_id)["condition_answers"]
                    if a["programme_id"] == pid}
        open_conditions = [c for c in programme["record"].other_conditions if c not in answered]
        settled = settle_conditions_from_profile(ctx, pid, open_conditions)
        student = lookup_profile(ctx.conn, ctx.student_id)
        eligibility = check_eligibility(student, programme, ctx.cycle.date)
        results[pid] = {
            "status": eligibility["status"],
            "blockers": eligibility["blockers"],
            "unknowns": eligibility["unknowns"],
            "notes": eligibility["notes"],
            "next_deadline": eligibility["next_deadline"],
            "match": match_score(student, programme["record"])["summary"],
            "excluded_by_preference": excluded_by_preferences(student, programme["record"]),
            "pending_change_waiting_for_approval": programme["pending_version_id"] is not None,
            "unanswered_conditions": [c for c in programme["record"].other_conditions
                                      if f"needs the student's confirmation: {c}" in eligibility["unknowns"]],
            "settled_from_profile_now": settled,
        }
    ctx.state.setdefault("assessments", {}).update(results)
    return results


def find_documents(ctx: Context, programme_ids: list[str]) -> dict:
    """Required minus owned documents for each programme (plain code)."""
    student = lookup_profile(ctx.conn, ctx.student_id)
    results = {}
    for pid in known_programmes(ctx, programme_ids):
        programme = get_programme(ctx.conn, pid)
        if programme is None:
            raise ToolRefused(f"{pid}: no saved record yet")
        results[pid] = find_missing_documents(student, programme["record"])
    ctx.state.setdefault("missing_documents", {}).update(results)
    return results


def ask_student(ctx: Context, programme_id: str, condition: str) -> dict:
    """Ask the student whether they meet an admission condition, and remember the answer."""
    programme = get_programme(ctx.conn, known_programmes(ctx, [programme_id])[0])
    if programme is None or condition not in programme["record"].other_conditions:
        raise ToolRefused("you can only ask about a condition listed in that programme's other_conditions, copied exactly")
    answered = {(a["programme_id"], a["condition"]) for a in lookup_profile(ctx.conn, ctx.student_id)["condition_answers"]}
    if (programme_id, condition) in answered:
        raise ToolRefused("the student already answered this condition; the answer is in memory")
    reply = (ctx.interact.ask(f"[{programme['name']}] Do you meet this admission condition?\n  \"{condition}\"\n  (yes / no / unsure)") or "").strip().lower()
    met = {"yes": True, "y": True, "no": False, "n": False}.get(reply)
    record_condition_answer(ctx.conn, ctx.student_id, ctx.cycle.cycle, programme_id, condition, met)
    return {"programme_id": programme_id, "condition": condition, "answer": {True: "yes", False: "no"}.get(met, "unsure")}


def request_approval(ctx: Context, kind: str, programme_ids: list[str], reason: str) -> dict:
    """Ask the student to approve a changed programme version, or the shortlist. Nothing is used before approval."""
    programme_ids = known_programmes(ctx, programme_ids)
    if kind == "programme_change":
        return approve_changes(ctx, programme_ids, reason)
    if kind == "shortlist":
        return approve_shortlist(ctx, programme_ids, reason)
    raise ToolRefused("kind must be 'programme_change' or 'shortlist'")


def approve_changes(ctx: Context, programme_ids: list[str], reason: str) -> dict:
    results = {}
    for pid in programme_ids:
        programme = get_programme(ctx.conn, pid)
        if programme is None or programme["pending_version_id"] is None:
            raise ToolRefused(f"{pid} has no change waiting for approval")
        pending_id = programme["pending_version_id"]
        new = get_programme_version(ctx.conn, pending_id)
        message = (f"[{programme['name']}] The saved pages changed.\n  " + "\n  ".join(describe_change(programme["record"], new))
                   + f"\n  Agent's reason: {reason}\n  Use the new version?")
        approved = bool(ctx.interact.approve(message))
        (db.approve_version if approved else db.reject_version)(ctx.conn, "programme_versions", pending_id)
        db.insert_once(ctx.conn, "decisions", {
            "student_id": ctx.student_id, "cycle": ctx.cycle.cycle, "programme_id": pid,
            "kind": "approve_change" if approved else "reject_change", "decided_by": "student",
            "data": json.dumps({"version_id": pending_id, "reason": reason}),
            "idempotency_key": db.idempotency_key(ctx.student_id, "change", pid, pending_id)})
        results[pid] = "approved: the new version is now current; the old one is kept as history" if approved \
            else "rejected: the previous version stays in use"
    return results


def get_programme_version(conn: sqlite3.Connection, version_id: int) -> ProgrammeRecord:
    """Load any stored version (e.g. a pending one) as a ProgrammeRecord, courses included."""
    row = conn.execute("SELECT data FROM programme_versions WHERE id = ?", [version_id]).fetchone()
    courses = [{"name": c["name"], "source_url": c["source_url"], "topics": json.loads(c["topics"])}
               for c in conn.execute("SELECT * FROM courses WHERE programme_version_id = ?", [version_id])]
    return ProgrammeRecord.model_validate({**json.loads(row["data"]), "courses": courses})


def approve_shortlist(ctx: Context, programme_ids: list[str], reason: str) -> dict:
    assessments = ctx.state.get("assessments", {})
    for pid in programme_ids:
        assessment = assessments.get(pid)
        if assessment is None:
            raise ToolRefused(f"{pid} has not been checked in this cycle; call check_programmes first")
        if assessment["status"] == "not_eligible" or assessment["excluded_by_preference"]:
            raise ToolRefused(f"{pid} is not eligible or is excluded by a preference; it cannot be shortlisted")
    lines = [f"  - {pid}: {assessments[pid]['status']}, deadline {assessments[pid]['next_deadline']}, "
             f"{assessments[pid]['match']}" for pid in programme_ids]
    approved = bool(ctx.interact.approve("Proposed shortlist:\n" + "\n".join(lines) + f"\n  Agent's reason: {reason}\n  Approve?"))
    db.insert_once(ctx.conn, "decisions", {
        "student_id": ctx.student_id, "cycle": ctx.cycle.cycle, "kind": "shortlist" if approved else "shortlist_rejected",
        "decided_by": "student", "data": json.dumps({"programme_ids": programme_ids, "reason": reason}),
        "idempotency_key": db.idempotency_key(ctx.student_id, "shortlist", ctx.cycle.cycle, ",".join(sorted(programme_ids)))})
    if approved:
        ctx.state["approved_shortlist"] = programme_ids
    return {"approved": approved, "programme_ids": programme_ids}


def plan_items(ctx: Context, pid: str) -> tuple[list[str], str | None]:
    """What should be on the checklist for one programme now: missing documents plus the application."""
    student = lookup_profile(ctx.conn, ctx.student_id)
    programme = get_programme(ctx.conn, pid)
    deadline = check_eligibility(student, programme, ctx.cycle.date)["next_deadline"]
    items = [f"obtain {m['missing']} more {m['doc_type']}" for m in find_missing_documents(student, programme["record"])["missing"]]
    return items + ["submit the application"], deadline


def write_plan(ctx: Context, programme_ids: list[str]) -> dict:
    """Persistent write, after the student approves the exact changes: checklist rows, then Excel and .ics.

    Items still needed are carried forward (never duplicated), items no longer needed are marked done,
    due dates follow the current approved deadline. Running it again changes nothing.
    """
    shortlist = ctx.state.get("approved_shortlist")
    if not shortlist:
        raise ToolRefused("the student has not approved a shortlist in this cycle; call request_approval first")
    if not set(programme_ids) <= set(shortlist):
        raise ToolRefused(f"only approved programmes can be planned: {shortlist}")
    for pid in programme_ids:
        if get_programme(ctx.conn, pid)["pending_version_id"] is not None:
            raise ToolRefused(f"{pid} has a changed version waiting for approval; approve or reject it first")

    # 1. Work out the changes without writing anything.
    to_add, to_close, to_reschedule = [], [], []
    for pid in programme_ids:
        items, deadline = plan_items(ctx, pid)
        open_rows = ctx.conn.execute("SELECT id, item, due_date FROM checklist_items WHERE student_id = ? "
                                     "AND programme_id = ? AND status = 'open'", [ctx.student_id, pid]).fetchall()
        open_items = {row["item"]: row for row in open_rows}
        to_add += [(pid, item, deadline) for item in items if item not in open_items]
        to_close += [(pid, row["id"], row["item"]) for row in open_rows if row["item"] not in items]
        to_reschedule += [(pid, row["id"], row["item"], row["due_date"], deadline)
                          for row in open_rows if row["item"] in items and row["due_date"] != deadline]
    if not (to_add or to_close or to_reschedule):
        files = export_plan(ctx.conn, ctx.student_id, ctx.plans_dir)
        ctx.state["plan_written"] = programme_ids
        return {"changes": "none, the checklist is already up to date", "files": files}

    # 2. Show the exact changes and write only if the student approves.
    lines = ([f"  + {pid}: {item} (due {due})" for pid, item, due in to_add]
             + [f"  done {pid}: {item}" for pid, _, item in to_close]
             + [f"  moved {pid}: {item} {old} -> {new}" for pid, _, item, old, new in to_reschedule])
    if not ctx.interact.approve("Checklist and calendar changes:\n" + "\n".join(lines) + "\n  Write them?"):
        return {"approved": False, "written": 0}

    with ctx.conn:
        for pid, row_id, _ in to_close:
            ctx.conn.execute("UPDATE checklist_items SET status = 'done' WHERE id = ?", [row_id])
        for pid, row_id, _, _, new_due in to_reschedule:
            ctx.conn.execute("UPDATE checklist_items SET due_date = ? WHERE id = ?", [new_due, row_id])
    written = 0
    for pid, item, due in to_add:
        written += db.insert_once(ctx.conn, "checklist_items", {
            "student_id": ctx.student_id, "programme_id": pid, "item": item, "cycle": ctx.cycle.cycle, "due_date": due,
            "idempotency_key": db.idempotency_key(ctx.student_id, pid, item, ctx.cycle.cycle)})
        if "write_plan" in ctx.fail_once:          # failure simulation: crash after the first row is saved
            ctx.fail_once.discard("write_plan")
            raise ToolFailed(f"simulated crash after saving {written} of {len(to_add)} checklist items; "
                             "every write has an idempotency key, so retrying write_plan is safe")
    db.insert_once(ctx.conn, "decisions", {
        "student_id": ctx.student_id, "cycle": ctx.cycle.cycle, "kind": "plan_approved", "decided_by": "student",
        "data": json.dumps({"added": len(to_add), "closed": len(to_close), "rescheduled": len(to_reschedule)}),
        "idempotency_key": db.idempotency_key(ctx.student_id, "plan", ctx.cycle.cycle, ",".join(sorted(programme_ids)))})
    files = export_plan(ctx.conn, ctx.student_id, ctx.plans_dir)
    ctx.state["plan_written"] = programme_ids
    return {"approved": True, "added": written, "closed": len(to_close), "rescheduled": len(to_reschedule), "files": files}


def finish(ctx: Context, summary: str) -> dict:
    """End the cycle. The facts shown to the student come from the database; the model's text is only a note."""
    ctx.state["finished"] = summary
    ctx.state["facts"] = facts_summary(ctx.conn, ctx.student_id)
    return {"finished": True}


# ---------- What the LLM sees ----------

TOOLS = {
    "read_programmes": read_programmes,
    "extract_programme": extract_programme,
    "check_programmes": check_programmes,
    "find_documents": find_documents,
    "ask_student": ask_student,
    "request_approval": request_approval,
    "write_plan": write_plan,
    "finish": finish,
}


def _schema(name: str, description: str, properties: dict) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(properties),
                       "additionalProperties": False}}}


IDS = {"type": "array", "items": {"type": "string"}, "description": "programme ids, e.g. [\"deai\", \"codas\"]"}
TOOL_SCHEMAS = [
    _schema("read_programmes", "Read the saved pages of programmes; reports first reads and changed pages.", {"programme_ids": IDS}),
    _schema("extract_programme", "Extract and save a programme that is read for the first time or whose pages changed. "
            "A changed programme is saved as pending until the student approves it.", {"programme_id": {"type": "string"}}),
    _schema("check_programmes", "Check eligibility, course match and preferences for programmes (plain code).", {"programme_ids": IDS}),
    _schema("find_documents", "List the documents the student still needs for programmes.", {"programme_ids": IDS}),
    _schema("ask_student", "Ask the student whether they meet one admission condition (copy it exactly).",
            {"programme_id": {"type": "string"}, "condition": {"type": "string"}}),
    _schema("request_approval", "Ask the student to approve either a programme_change or a shortlist.",
            {"kind": {"type": "string", "enum": ["programme_change", "shortlist"]}, "programme_ids": IDS,
             "reason": {"type": "string"}}),
    _schema("write_plan", "Write or update the checklist, Excel file and calendar for approved shortlisted programmes "
            "(the student approves the exact changes).", {"programme_ids": IDS}),
    _schema("finish", "End this cycle with a short summary for the student.", {"summary": {"type": "string"}}),
]

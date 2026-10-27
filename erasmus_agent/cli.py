"""Command-line interface (milestone M9). Run `python -m erasmus_agent --help`.

Typical use:
    python -m erasmus_agent init
    python -m erasmus_agent onboard --from data/profiles/stu-sara.json
    python -m erasmus_agent run-cycle 1
    python -m erasmus_agent status --student stu-sara
"""

import datetime as dt
import json
import shutil
from pathlib import Path

import typer
from pydantic import ValidationError

from . import db, logbook
from .agent import MAX_STEPS, ConsoleInteraction, ScriptedInteraction, print_step, run_cycle
from .config import DEFAULT_MODEL, EXTRACT_MODEL, MODELS
from .exports import PLANS_DIR, facts_summary
from .extract import extract_programme, save_programme_record
from .llm import chat_json
from .logbook import read_events, set_context
from .memory import correct_programme, correct_student, get_programme, lookup_profile, save_profile
from .models import Profile, ProgrammeRecord, load_cycle, load_profile, vocabulary
from .reader import read_programme
from .rules import check_eligibility, excluded_by_preferences, match_score
from .tools import describe_change, get_programme_version

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELECTION_FILE = PROJECT_ROOT / "data" / "selected_programmes.json"
RECORDS_DIR = PROJECT_ROOT / "outputs" / "records"
CYCLES_DIR = PROJECT_ROOT / "data" / "cycles"

app = typer.Typer(help="Erasmus Mundus Tracker: an agent that tracks scholarship applications across cycles.",
                  no_args_is_help=True, add_completion=False)
settings = {"db": db.DEFAULT_DB_PATH}


@app.callback()
def main(database: Path = typer.Option(db.DEFAULT_DB_PATH, "--db", help="SQLite memory file.")):
    """Erasmus Mundus Tracker."""
    settings["db"] = database


def connect():
    """Open the memory and make sure every table exists (safe to repeat)."""
    conn = db.connect(settings["db"])
    db.init_db(conn)
    return conn


def check_model(model: str) -> str:
    if model not in MODELS:
        raise typer.BadParameter(f"choose one of: {', '.join(MODELS)}")
    return model


# ---------- Setting up ----------

@app.command()
def init(load_records: bool = typer.Option(True, help="Load the saved extracted records (no LLM needed).")):
    """Create the database, register the 5 programmes and load their saved records."""
    conn = connect()
    for programme in json.loads(SELECTION_FILE.read_text(encoding="utf-8"))["programmes"]:
        pid = programme["programme_id"]
        db.add_programme(conn, pid, programme["catalogue_id"], programme["name"], programme["role"])
        record_file = RECORDS_DIR / f"{pid}.json"
        if load_records and get_programme(conn, pid) is None and record_file.is_file():
            record = ProgrammeRecord.model_validate_json(record_file.read_text(encoding="utf-8"))
            save_programme_record(conn, read_programme(conn, pid), record, "saved extraction (outputs/records)")
        current = get_programme(conn, pid)
        typer.echo(f"{pid:10} " + (f"version {current['version']}, {len(current['record'].courses)} courses"
                                   if current else "no record yet (run: extract)"))
    typer.echo(f"Memory ready: {settings['db']}")


@app.command()
def extract(model: str = typer.Option(EXTRACT_MODEL, "--model", help="main, fast or local.")):
    """Extract programmes that are new or whose saved pages changed (changed ones wait for approval)."""
    conn = connect()
    for (pid,) in conn.execute("SELECT programme_id FROM programmes ORDER BY programme_id").fetchall():
        reading = read_programme(conn, pid)
        if not (reading["first_read"] or reading["changed"]):
            typer.echo(f"{pid}: unchanged, skipped")
            continue
        record, stats = extract_programme(reading, model=check_model(model))
        _, saved = save_programme_record(conn, reading, record, f"extraction ({MODELS[model]})")
        typer.echo(f"{pid}: saved as {saved}; {stats['calls']} calls, {stats['tokens']} tokens")


# ---------- Onboarding ----------

def suggest_topics(text: str, model: str) -> list[str]:
    """The LLM proposes 1-3 vocabulary topics for the student's own words; the student confirms."""
    topics = sorted(vocabulary()["topics"])
    schema = {"type": "object", "properties": {"topics": {"type": "array", "items": {"type": "string", "enum": topics}}},
              "required": ["topics"], "additionalProperties": False}
    try:
        reply, _ = chat_json([{"role": "system", "content": "Pick 1 to 3 topics from the list that best match the "
                               "student's interests. Only use topics from the list."},
                              {"role": "user", "content": f"Interests: {text}\nTopics: {', '.join(topics)}"}],
                             "interest_topics", schema, model=model, purpose="suggest_topics")
        return [t for t in json.loads(reply)["topics"] if t in vocabulary()["topics"]][:3]
    except Exception:          # no LLM available: the student picks from the list instead
        return []


def pick(prompt: str, options: list[str], suggested: list[str]) -> list[str]:
    """Show numbered options; Enter accepts the suggestion, or type numbers like 1,4."""
    for number, option in enumerate(options, start=1):
        typer.echo(f"  {number:2}. {option}")
    default = ",".join(str(options.index(s) + 1) for s in suggested) if suggested else ""
    answer = typer.prompt(prompt, default=default)
    return [options[int(n) - 1] for n in answer.replace(" ", "").split(",") if n.isdigit() and 0 < int(n) <= len(options)]


def interactive_profile(model: str) -> Profile:
    """Ask the student for their details; vocabulary values are suggested and then confirmed."""
    student_id = typer.prompt("Student id (e.g. stu-ayesha)")
    name = typer.prompt("Name")
    level = typer.prompt("Degree level", default="bachelor")
    field_text = typer.prompt("Degree subject (as on your transcript)")
    fields = sorted(vocabulary()["degree_fields"])
    guess = [f for f in fields if f in field_text.lower() or field_text.lower() in f][:1]
    degree_field = (pick("Which field matches your degree?", fields, guess) or [None])[0]
    interests_text = typer.prompt("Your interests, in your own words")
    suggested = suggest_topics(interests_text, model)
    topics = pick("Which topics match your interests?", sorted(vocabulary()["topics"]), suggested)
    test = typer.prompt("English test (e.g. IELTS Academic)")
    documents = [{"doc_type": "english_test", "version": 1, "added_on": dt.date.today().isoformat(),
                  "details": {"test": test, "overall": float(typer.prompt("Overall score")),
                              "taken_on": typer.prompt("Test date (YYYY-MM-DD)")}}]
    owned = typer.prompt("Other documents you have, comma separated (e.g. transcript, cv, passport)", default="")
    documents += [{"doc_type": d.strip(), "version": 1, "added_on": dt.date.today().isoformat()}
                  for d in owned.split(",") if d.strip()]
    return Profile.model_validate({
        "student_id": student_id, "name": name, "fictional": False,
        "degree": {"level": level, "field": field_text, "degree_field": degree_field,
                   "institution": typer.prompt("University"), "graduation_year": int(typer.prompt("Graduation year")),
                   "ects": int(typer.prompt("Credits (ECTS)", default="240"))},
        "grades": {"cgpa": float(typer.prompt("CGPA")), "scale": float(typer.prompt("CGPA scale", default="4.0"))},
        "interests_text": interests_text, "interest_topics": topics, "documents": documents})


@app.command()
def onboard(from_file: Path = typer.Option(None, "--from", help="A profile file, e.g. data/profiles/stu-sara.json."),
            model: str = typer.Option(DEFAULT_MODEL, "--model", help="Model that suggests interest topics.")):
    """Register a student, interactively or from a file. Both go through the same validation."""
    try:
        profile = load_profile(from_file) if from_file else interactive_profile(check_model(model))
    except (ValidationError, ValueError) as problem:
        typer.echo(f"Profile not saved, it is not valid:\n{problem}")
        raise typer.Exit(1)
    conn = connect()
    set_context(profile.student_id)
    save_profile(conn, profile, f"onboarding from {from_file.name}" if from_file else "onboarding (interactive)")
    student = lookup_profile(conn, profile.student_id)
    typer.echo(f"Saved {profile.student_id}: {student['profile']['degree']['degree_field']}, "
               f"interests {student['profile']['interest_topics']}, {len(student['documents'])} documents.")


# ---------- Running the agent ----------

@app.command("run-cycle")
def run_cycle_command(
        cycle: str = typer.Argument(..., help="Cycle number (uses data/cycles/cycleN.json) or a cycle file."),
        model: str = typer.Option(DEFAULT_MODEL, "--model", help="Decision model: main, fast or local."),
        auto: bool = typer.Option(False, "--auto", help="No questions: approve everything, answer 'unsure'."),
        answers: Path = typer.Option(None, "--answers", help="With --auto: JSON {text in question: answer}."),
        max_steps: int = typer.Option(MAX_STEPS, "--max-steps"),
        fail: list[str] = typer.Option([], "--fail", help="Failure simulation: this tool crashes once part-way, e.g. write_plan."),
        no_memory: bool = typer.Option(False, "--no-memory", help="Comparison run: ignore everything earlier cycles stored.")):
    """Run one cycle, or resume it from its checkpoint. A finished cycle is not run again."""
    cycle_file = CYCLES_DIR / f"cycle{cycle}.json" if cycle.isdigit() else Path(cycle)
    cycle_input = load_cycle(cycle_file)
    conn = connect()
    interaction = ScriptedInteraction(json.loads(answers.read_text()) if answers else {}) if auto else ConsoleInteraction()
    outcome = run_cycle(conn, cycle_input, interaction, model=check_model(model), max_steps=max_steps, on_step=print_step,
                        fail_once=set(fail), memory=not no_memory)
    typer.echo(f"\nCycle {cycle_input.cycle}: {outcome['status']} after {outcome['steps']} steps, "
               f"{outcome.get('tokens', 0)} tokens." + (" (already finished)" if outcome.get("already_finished") else ""))
    if outcome.get("reason"):
        typer.echo(f"Reason: {outcome['reason']}")
    typer.echo(f"\nFacts (from the database):\n{facts_summary(conn, cycle_input.student_id)}")
    if outcome.get("summary"):
        typer.echo(f"\nAgent's note (model text, not checked): {outcome['summary']}")


@app.command()
def approve(programme: str, reject: bool = typer.Option(False, "--reject"),
            student: str = typer.Option("stu-sara", "--student")):
    """Approve (or reject) a programme change that is waiting, outside a cycle."""
    set_context(student)
    conn = connect()
    current = get_programme(conn, programme)
    if current is None or current["pending_version_id"] is None:
        typer.echo(f"{programme}: nothing is waiting for approval.")
        raise typer.Exit(1)
    pending = current["pending_version_id"]
    for line in describe_change(current["record"], get_programme_version(conn, pending)):
        typer.echo(f"  {line}")
    (db.reject_version if reject else db.approve_version)(conn, "programme_versions", pending)
    last = conn.execute("SELECT COALESCE(MAX(cycle), 0) FROM cycles WHERE student_id = ?", [student]).fetchone()[0]
    db.insert_once(conn, "decisions", {
        "student_id": student, "cycle": last, "programme_id": programme, "decided_by": "student",
        "kind": "reject_change" if reject else "approve_change", "data": json.dumps({"version_id": pending}),
        "idempotency_key": db.idempotency_key(student, "change", programme, pending)})
    typer.echo(f"{programme}: change {'rejected; the old version stays' if reject else 'approved; the old version is kept as history'}.")


@app.command()
def correct(file: Path = typer.Option(None, "--file", help="Programme correction file (see data/corrections/)."),
            student: str = typer.Option(None, "--student"), field: str = typer.Option(None, "--field"),
            value: str = typer.Option(None, "--value"), reason: str = typer.Option(None, "--reason")):
    """Correct a programme fact (from a file) or a student fact. A new version is saved; history is kept."""
    conn = connect()
    if file:
        fix = json.loads(file.read_text(encoding="utf-8"))
        before = get_programme(conn, fix["programme_id"])
        version_id = correct_programme(conn, fix["programme_id"], fix["field"], fix["value"], fix["reason"])
        after = get_programme(conn, fix["programme_id"])
        typer.echo(f"{fix['programme_id']}.{fix['field']}: version {before['version']} -> {after['version']} "
                   f"(row {version_id}); version {before['version']} kept as history.\n"
                   f"  before: {json.dumps(before['record'].model_dump(mode='json')[fix['field']], ensure_ascii=False)[:300]}\n"
                   f"  after:  {json.dumps(after['record'].model_dump(mode='json')[fix['field']], ensure_ascii=False)[:300]}")
    elif student and field and value is not None and reason:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        correct_student(conn, student, field, parsed, reason)
        typer.echo(f"{student}.{field} = {parsed!r}; the previous profile version is kept as history.")
    else:
        typer.echo("Use --file FILE, or --student ID --field PATH --value VALUE --reason TEXT.")
        raise typer.Exit(1)


# ---------- Looking at memory ----------

def recommended_action(conn, student_id: str, pending: list[str], last: dict | None, open_items: int) -> str:
    """The next thing the student should do, worked out by code from the stored state."""
    if pending:
        return f"Review the changed programme(s) {', '.join(pending)}: python -m erasmus_agent approve <id> (or --reject)"
    if last is None:
        return "Run the first cycle: python -m erasmus_agent run-cycle 1"
    if last["status"] != "done":
        return f"Resume cycle {last['cycle']}: python -m erasmus_agent run-cycle {last['cycle']}"
    if open_items:
        return "Collect the open checklist items before their deadlines, then run the next cycle with the new documents."
    return "Nothing open. Run the next cycle when new information arrives."


@app.command()
def status(student: str = typer.Option("stu-sara", "--student"),
           date: str = typer.Option(None, "--date", help="Judge deadlines on this date (default: last cycle's date).")):
    """Current state, recommended next action, supporting evidence and any approval needed."""
    conn = connect()
    view = lookup_profile(conn, student)
    profile = view["profile"]
    last = conn.execute("SELECT cycle, cycle_date, status, step FROM cycles WHERE student_id = ? ORDER BY cycle DESC LIMIT 1",
                        [student]).fetchone()
    last = dict(last) if last else None
    today = dt.date.fromisoformat(date or (last["cycle_date"] if last else dt.date.today().isoformat()))

    typer.echo(f"STUDENT  {student} ({profile['name']}): {profile['degree']['degree_field']}, interests "
               f"{', '.join(profile['interest_topics'])}")
    typer.echo(f"  documents: {', '.join(f'{d['doc_id']} v{d['version']}' for d in view['documents'])}")
    typer.echo(f"  preferences: {', '.join(f'{p['kind']} {p['value']}' for p in view['preferences']) or 'none'}")
    typer.echo(f"  last cycle: " + (f"{last['cycle']} on {last['cycle_date']}, {last['status']} after {last['step']} steps"
                                    if last else "none yet"))

    typer.echo(f"\nPROGRAMMES (judged on {today})")
    pending = []
    for (pid,) in conn.execute("SELECT programme_id FROM programmes ORDER BY programme_id").fetchall():
        programme = get_programme(conn, pid)
        if programme is None:
            typer.echo(f"  {pid:10} no record yet")
            continue
        result = check_eligibility(view, programme, today)
        excluded = excluded_by_preferences(view, programme["record"])
        typer.echo(f"  {pid:10} {('EXCLUDED' if excluded else result['status'].upper()):13} deadline "
                   f"{result['next_deadline'] or 'unknown':10}  {match_score(view, programme['record'])['summary']}  "
                   f"[v{programme['version']}, pages fetched {(programme['fetched_at'] or '')[:10]}]")
        for reason in (result["blockers"] + result["unknowns"])[:2]:
            typer.echo(f"             - {reason[:120]}")
        flagged = list(programme["record"].suspected_instructions)
        if programme["pending_version_id"]:
            pending.append(pid)
            waiting = get_programme_version(conn, programme["pending_version_id"])
            flagged += [text for text in waiting.suspected_instructions if text not in flagged]
            for line in describe_change(programme["record"], waiting):
                typer.echo(f"             ! waiting for approval: {line}")
        for text in flagged:
            typer.echo(f"             ! text addressed to AI tools was found on its pages and IGNORED: \"{text[:100]}\"")

    open_items = [r for r in conn.execute("SELECT 1 FROM checklist_items WHERE student_id = ? AND status = 'open'", [student])]
    typer.echo(f"\nPLAN\n{facts_summary(conn, student)}")
    typer.echo(f"\nAPPROVALS NEEDED: {', '.join(pending) if pending else 'none'}")
    typer.echo(f"NEXT ACTION: {recommended_action(conn, student, pending, last, len(open_items))}")


def describe_event(event: dict) -> str:
    """One readable line for a run-log event."""
    kind = event["event"]
    if kind == "llm_call":
        return (f"LLM  {event['purpose']:17} {event['model'][:20]:20} {event['tokens']:>6} tok {event['latency_ms']:>6} ms"
                + ("  (fallback)" if event.get("fallback") else ""))
    if kind == "rate_limit":
        return f"LIMIT {event['limit']} on {event['model']}: {event['action']}"
    if kind == "tool_call":
        return f"TOOL step {event['step']} {event['tool']}: " + ("ok" if event["ok"] else f"{'FAILED' if event['error'].startswith('ToolFailed') else 'REFUSED'} {event['error']}")[:110]
    if kind == "memory_read":
        return f"READ  {event['what']} {event['target']}"
    if kind == "memory_write":
        return f"WRITE {event['table']} {json.dumps(event.get('key'), ensure_ascii=False)[:60]}: {event['result']}"
    return f"{kind.upper()} " + json.dumps({k: v for k, v in event.items() if k not in ("ts", "event", "run_id")},
                                           ensure_ascii=False)[:120]


@app.command()
def log(student: str = typer.Option("stu-sara", "--student"), cycle: int = typer.Option(None, "--cycle"),
        limit: int = typer.Option(60, "--limit"),
        events: bool = typer.Option(False, "--events", help="Show the full run log (LLM calls, memory reads and writes).")):
    """Every agent step: time, tool, model, tokens, latency and the outcome or the refusal."""
    if events:
        found = read_events(student, cycle)
        for event in found[-limit:]:
            typer.echo(f"{event['ts'][:19]}  c{event.get('cycle')}  {describe_event(event)}")
        calls = [e for e in found if e["event"] == "llm_call"]
        ignored = sum(e["event"] == "memory_write" and "duplicate ignored" in e.get("result", "") for e in found)
        typer.echo(f"\n{len(found)} events in {logbook.LOG_FILE}: {len(calls)} LLM calls, "
                   f"{sum(e['tokens'] for e in calls)} tokens, "
                   f"{sum(e['event'] == 'memory_read' for e in found)} memory reads, "
                   f"{sum(e['event'] == 'memory_write' for e in found)} memory writes, "
                   f"{ignored} duplicate writes ignored")
        return
    conn = connect()
    sql = "SELECT * FROM action_log WHERE student_id = ?" + (" AND cycle = ?" if cycle else "") + " ORDER BY id LIMIT ?"
    rows = conn.execute(sql, [student, cycle, limit] if cycle else [student, limit]).fetchall()
    for r in rows:
        outcome = f"{'FAILED' if r['error'].startswith('ToolFailed') else 'REFUSED'} {r['error']}" if r["error"] else (r["result"] or "")
        typer.echo(f"{r['created_at'][:19]}  c{r['cycle']}.{r['step']:<2} {r['tool']:18} {(r['model'] or '')[:20]:20} "
                   f"{r['tokens'] or 0:>6} tok {r['latency_ms'] or 0:>6} ms  {outcome[:110]}")
    totals = conn.execute("SELECT COUNT(*), COALESCE(SUM(tokens), 0), COALESCE(AVG(latency_ms), 0) FROM action_log "
                          "WHERE student_id = ?", [student]).fetchone()
    typer.echo(f"\n{totals[0]} steps, {totals[1]} tokens in decisions, {totals[2]:.0f} ms average per step")


@app.command()
def forget(student: str, yes: bool = typer.Option(False, "--yes", help="Do not ask for confirmation.")):
    """Delete everything stored about one student (database rows and generated files)."""
    if not yes and not typer.confirm(f"Permanently delete all data about {student}?"):
        raise typer.Exit(1)
    deleted = db.forget_student(connect(), student)
    shutil.rmtree(PLANS_DIR / student, ignore_errors=True)
    lines = logbook.forget_student(student)
    typer.echo(f"Deleted {sum(deleted.values())} rows about {student}: {deleted}; {lines} run-log lines removed")

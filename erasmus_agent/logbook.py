"""Run log (milestone M11): one JSON object per line in logs/run.jsonl.

Every LLM call, rate-limit event, tool call, memory read and memory write is appended
with a UTC timestamp and the run id, so a whole run can be replayed, measured and audited.
The action_log table keeps the agent's steps for the CLI; this file has everything.
"""

import datetime as dt
import json
import os
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = Path(os.getenv("ERASMUS_LOG", PROJECT_ROOT / "logs" / "run.jsonl"))

# Who and what the current events belong to; set at the start of a command or cycle.
context = {"run_id": uuid.uuid4().hex[:12], "student_id": None, "cycle": None}


def set_context(student_id: str | None = None, cycle: int | None = None) -> None:
    context["student_id"], context["cycle"] = student_id, cycle


def log_event(event: str, **fields) -> None:
    """Append one event. Logging must never break the agent, so write errors are ignored."""
    record = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"), "event": event, **context, **fields}
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def read_events(student_id: str | None = None, cycle: int | None = None) -> list[dict]:
    """All logged events, optionally only those of one student (and cycle)."""
    if not LOG_FILE.is_file():
        return []
    events = []
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (student_id is None or event.get("student_id") == student_id) and (cycle is None or event.get("cycle") == cycle):
            events.append(event)
    return events


def forget_student(student_id: str) -> int:
    """Remove one student's events from the log (part of deleting a student). Returns lines removed."""
    if not LOG_FILE.is_file():
        return 0
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if f'"student_id": "{student_id}"' not in line]
    LOG_FILE.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(lines) - len(kept)

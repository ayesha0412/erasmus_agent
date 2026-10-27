"""Generated views of the plan (milestone M8): an Excel checklist, an .ics calendar and a fact summary.

Everything here is READ from SQLite and written out fresh each time. These files are views:
they never change the database, so editing or deleting them cannot overwrite a decision.
Calendar events have stable ids, so importing the file again updates events instead of
duplicating them.
"""

import datetime as dt
import hashlib
import json
import os
import sqlite3
from pathlib import Path

from icalendar import Alarm, Calendar, Event
from openpyxl import Workbook
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLANS_DIR = Path(os.getenv("ERASMUS_PLANS", PROJECT_ROOT / "outputs" / "plans"))


def checklist_rows(conn: sqlite3.Connection, student_id: str) -> list[dict]:
    """All checklist items of ONE student, open ones first, then by deadline."""
    rows = conn.execute(
        "SELECT c.programme_id, p.name, c.item, c.due_date, c.status, c.cycle, c.created_at "
        "FROM checklist_items c JOIN programmes p ON p.programme_id = c.programme_id "
        "WHERE c.student_id = ? ORDER BY c.status = 'done', c.due_date, c.programme_id, c.id", [student_id])
    return [dict(row) for row in rows]


def duplicate_items(conn: sqlite3.Connection, student_id: str) -> int:
    """How many open checklist items appear more than once (should always be 0)."""
    rows = conn.execute("SELECT COUNT(*) - 1 AS extra FROM checklist_items WHERE student_id = ? AND status = 'open' "
                        "GROUP BY programme_id, item HAVING COUNT(*) > 1", [student_id]).fetchall()
    return sum(row["extra"] for row in rows)


def latest_shortlist(conn: sqlite3.Connection, student_id: str) -> list[str]:
    """The programmes in the student's most recent approved shortlist."""
    row = conn.execute("SELECT data FROM decisions WHERE student_id = ? AND kind = 'shortlist' ORDER BY id DESC LIMIT 1",
                       [student_id]).fetchone()
    return json.loads(row["data"])["programme_ids"] if row else []


def facts_summary(conn: sqlite3.Connection, student_id: str) -> str:
    """A summary built only from the database, so every date and count in it is exact."""
    rows = [r for r in checklist_rows(conn, student_id) if r["status"] == "open"]
    shortlist = latest_shortlist(conn, student_id)
    deadlines = {r["programme_id"]: r["due_date"] for r in rows}
    lines = [f"Shortlist: " + (", ".join(f"{pid} (deadline {deadlines.get(pid) or 'unknown'})" for pid in shortlist)
                                or "none approved yet")]
    lines.append(f"Open checklist items: {len(rows)} (duplicates: {duplicate_items(conn, student_id)})")
    for row in rows:
        lines.append(f"  - {row['programme_id']}: {row['item']} (due {row['due_date'] or 'unknown'})")
    return "\n".join(lines)


def write_excel(rows: list[dict], path: Path, student_id: str) -> None:
    """The checklist as a spreadsheet, with a note that it is a generated view."""
    book = Workbook()
    sheet = book.active
    sheet.title = "Checklist"
    headers = ["Programme", "Programme name", "Item", "Due date", "Status", "Added in cycle"]
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for row in rows:
        sheet.append([row["programme_id"], row["name"], row["item"], row["due_date"], row["status"], row["cycle"]])
    for column, width in zip("ABCDEF", [12, 45, 40, 12, 10, 14]):
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = "A2"

    about = book.create_sheet("About")
    about.append(["Student", student_id])
    about.append(["Generated at (UTC)", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")])
    about.append(["Source", "erasmus.db (checklist_items); this file is a generated view"])
    about.append(["Note", "Changes made here are not read back. Record decisions through the agent."])
    book.save(path)


def stable_uid(*parts: str) -> str:
    """The same programme always gets the same event id, so re-importing updates instead of duplicating."""
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24] + "@erasmus-agent"


def write_ics(rows: list[dict], path: Path, student_id: str) -> int:
    """One all-day event per programme deadline, listing its open items, with a reminder 7 days before."""
    calendar = Calendar()
    calendar.add("prodid", "-//Erasmus Mundus Tracker (student project)//EN")
    calendar.add("version", "2.0")
    by_programme: dict[str, list[dict]] = {}
    for row in rows:
        if row["status"] == "open" and row["due_date"]:
            by_programme.setdefault(row["programme_id"], []).append(row)
    for programme_id, items in sorted(by_programme.items()):
        due = dt.date.fromisoformat(items[0]["due_date"])
        event = Event()
        event.add("uid", stable_uid(student_id, programme_id, "application-deadline"))
        event.add("dtstamp", dt.datetime.now(dt.timezone.utc))
        event.add("dtstart", due)
        event.add("dtend", due + dt.timedelta(days=1))
        event.add("summary", f"Application deadline: {items[0]['name']}")
        event.add("description", "Open items:\n" + "\n".join(f"- {i['item']}" for i in items))
        reminder = Alarm()
        reminder.add("action", "DISPLAY")
        reminder.add("description", f"7 days left: {items[0]['name']}")
        reminder.add("trigger", dt.timedelta(days=-7))
        event.add_component(reminder)
        calendar.add_component(event)
    path.write_bytes(calendar.to_ical())
    return len(by_programme)


def export_plan(conn: sqlite3.Connection, student_id: str, out_dir: Path | None = None) -> dict:
    """Regenerate both files for one student from the database. Returns their paths."""
    folder = (out_dir or PLANS_DIR) / student_id
    folder.mkdir(parents=True, exist_ok=True)
    rows = checklist_rows(conn, student_id)
    excel, ics = folder / "checklist.xlsx", folder / "deadlines.ics"
    write_excel(rows, excel, student_id)
    events = write_ics(rows, ics, student_id)
    return {"excel": excel.as_posix(), "calendar": ics.as_posix(), "items": len(rows), "calendar_events": events}

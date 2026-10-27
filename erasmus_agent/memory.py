"""Student and programme memory: the layer between the agent and SQLite (milestone M6).

Writes go through add_version (history kept, retries harmless). Every student read
filters by student_id, so one student's data can never appear in another's view.
"""

import json
import sqlite3

from . import db
from .logbook import log_event
from .models import Course, Document, Preference, Profile, ProgrammeRecord


# ---------- Writing student memory ----------

def save_profile(conn: sqlite3.Connection, profile: Profile, source: str) -> None:
    """Onboard a student, or save their profile again: each part gets a new version only if it changed."""
    db.add_student(conn, profile.student_id, profile.name, is_fictional=profile.fictional)
    core = profile.model_dump(mode="json", exclude={"documents", "preferences"})
    db.add_version(conn, "profile_versions", {"student_id": profile.student_id}, core, source)
    for document in profile.documents:
        save_document(conn, profile.student_id, document, source)
    for preference in profile.preferences:
        save_preference(conn, profile.student_id, preference, source)


def save_document(conn: sqlite3.Connection, student_id: str, document: Document, source: str) -> int:
    """Add a document, or a new version of one (e.g. a retaken IELTS). The database numbers the versions."""
    data = document.model_dump(mode="json", exclude={"doc_id", "doc_type", "version"})
    return db.add_version(conn, "document_versions", {"student_id": student_id, "doc_id": document.doc_id},
                          data, source, extra={"doc_type": document.doc_type})


def save_preference(conn: sqlite3.Connection, student_id: str, preference: Preference, source: str) -> int:
    """Add a preference such as 'exclude_topic:law'."""
    return db.add_version(conn, "preference_versions", {"student_id": student_id, "pref_key": preference.pref_key},
                          preference.model_dump(mode="json"), source)


def record_condition_answer(conn: sqlite3.Connection, student_id: str, cycle: int, programme_id: str,
                            condition: str, met: bool | None, decided_by: str = "student",
                            reason: str = "") -> bool:
    """Remember whether an admission condition is met, so it is never asked or worked out again.

    met is True or False, or None when the student is not sure (the condition then stays uncertain).
    decided_by is "student" (they answered) or "agent" (settled from profile facts, with a reason).
    """
    return db.insert_once(conn, "decisions", {
        "student_id": student_id, "cycle": cycle, "kind": "condition_answer", "programme_id": programme_id,
        "data": json.dumps({"condition": condition, "met": met, "reason": reason}), "decided_by": decided_by,
        "idempotency_key": db.idempotency_key(student_id, "condition_answer", programme_id, condition),
    })


# ---------- Reading student memory ----------

def lookup_profile(conn: sqlite3.Connection, student_id: str) -> dict:
    """Everything the agent may know about ONE student: current profile, documents, preferences,
    past decisions, answered conditions and open checklist items."""
    profile = db.get_current(conn, "profile_versions", {"student_id": student_id})
    log_event("memory_read", what="lookup_profile", target=student_id, found=profile is not None)
    if profile is None:
        raise ValueError(f"unknown student '{student_id}'")
    documents = [{"doc_id": r["doc_id"], "doc_type": r["doc_type"], "version": r["version"], **r["data"]}
                 for r in db.list_current(conn, "document_versions", student_id)]
    preferences = [r["data"] for r in db.list_current(conn, "preference_versions", student_id)]
    decisions = [{**dict(r), "data": json.loads(r["data"])}
                 for r in conn.execute("SELECT * FROM decisions WHERE student_id = ? ORDER BY id", [student_id])]
    checklist = [dict(r) for r in conn.execute(
        "SELECT * FROM checklist_items WHERE student_id = ? AND status = 'open' ORDER BY due_date", [student_id])]
    return {
        "student_id": student_id,
        "profile": profile["data"],
        "profile_version": profile["version"],
        "documents": documents,
        "preferences": preferences,
        "decisions": decisions,
        "condition_answers": [{"programme_id": d["programme_id"], "decided_by": d["decided_by"], **d["data"]}
                              for d in decisions if d["kind"] == "condition_answer"],
        "open_checklist": checklist,
    }


def profile_view(profile: Profile) -> dict:
    """The same shape as lookup_profile, built straight from a profile file (no database)."""
    return {
        "student_id": profile.student_id,
        "profile": profile.model_dump(mode="json", exclude={"documents", "preferences"}),
        "documents": [{"doc_id": d.doc_id, "doc_type": d.doc_type, "version": d.version,
                       **d.model_dump(mode="json", exclude={"doc_id", "doc_type", "version"})} for d in profile.documents],
        "preferences": [p.model_dump(mode="json") for p in profile.preferences],
        "decisions": [],
        "condition_answers": [],
        "open_checklist": [],
    }


# ---------- Reading programme memory ----------

def get_programme(conn: sqlite3.Connection, programme_id: str) -> dict | None:
    """The current APPROVED version of a programme, with its courses, plus any version waiting for approval."""
    row = db.get_current(conn, "programme_versions", {"programme_id": programme_id})
    log_event("memory_read", what="get_programme", target=programme_id, version=row["version"] if row else None)
    if row is None:
        return None
    courses = [Course(name=c["name"], ects=c["ects"], semester=c["semester"], university=c["university"],
                      description=c["description"], source_url=c["source_url"], topics=json.loads(c["topics"]))
               for c in conn.execute("SELECT * FROM courses WHERE programme_version_id = ? ORDER BY id", [row["id"]])]
    record = ProgrammeRecord.model_validate({**row["data"], "courses": [c.model_dump() for c in courses]})
    pending = conn.execute("SELECT id FROM programme_versions WHERE programme_id = ? AND status = 'pending' "
                           "ORDER BY id DESC LIMIT 1", [programme_id]).fetchone()
    name = conn.execute("SELECT name FROM programmes WHERE programme_id = ?", [programme_id]).fetchone()["name"]
    return {
        "programme_id": programme_id,
        "name": name,
        "version_id": row["id"],
        "version": row["version"],
        "fetched_at": row["fetched_at"],
        "record": record,
        "pending_version_id": pending["id"] if pending else None,
    }


# ---------- Corrections (history kept, never overwritten) ----------

CORRECTABLE_PROGRAMME_FIELDS = set(ProgrammeRecord.model_fields) - {"programme_id", "courses"}


def programme_corrections(conn: sqlite3.Connection, programme_id: str) -> dict:
    """The latest correction for each field of a programme, as {field: value}."""
    rows = conn.execute("SELECT field, value FROM corrections WHERE target = 'programme' AND target_id = ? ORDER BY id",
                        [programme_id])
    return {row["field"]: json.loads(row["value"]) for row in rows}


def apply_corrections(conn: sqlite3.Connection, record: ProgrammeRecord) -> ProgrammeRecord:
    """Return the record with every stored human correction applied (validated again)."""
    fixes = programme_corrections(conn, record.programme_id)
    if not fixes:
        return record
    return ProgrammeRecord.model_validate({**record.model_dump(mode="json"), **fixes})


def correct_programme(conn: sqlite3.Connection, programme_id: str, field: str, value, reason: str) -> int:
    """Correct one field of a programme: store the correction, then save a new approved version with it.

    The old version is kept (superseded); later extractions re-apply the correction automatically.
    """
    if field not in CORRECTABLE_PROGRAMME_FIELDS:
        raise ValueError(f"'{field}' cannot be corrected; choose one of {sorted(CORRECTABLE_PROGRAMME_FIELDS)}")
    current = get_programme(conn, programme_id)
    if current is None:
        raise ValueError(f"programme '{programme_id}' has no saved record")
    corrected = ProgrammeRecord.model_validate({**current["record"].model_dump(mode="json"), field: value})  # validate first
    db.insert_once(conn, "corrections", {
        "target": "programme", "target_id": programme_id, "field": field,
        "value": json.dumps(value, ensure_ascii=False), "reason": reason,
        "idempotency_key": db.idempotency_key("programme", programme_id, field, json.dumps(value, sort_keys=True))})
    row = conn.execute("SELECT page_fingerprint, fetched_at FROM programme_versions WHERE id = ?",
                       [current["version_id"]]).fetchone()
    version_id = db.add_version(conn, "programme_versions", {"programme_id": programme_id},
                                corrected.model_dump(mode="json", exclude={"courses"}), f"correction: {reason}",
                                extra={"page_fingerprint": row["page_fingerprint"], "fetched_at": row["fetched_at"]})
    with conn:   # the courses did not change: copy them to the new version (once)
        if not conn.execute("SELECT COUNT(*) FROM courses WHERE programme_version_id = ?", [version_id]).fetchone()[0]:
            conn.execute("INSERT INTO courses (programme_version_id, name, ects, semester, university, description, "
                         "source_url, topics) SELECT ?, name, ects, semester, university, description, source_url, topics "
                         "FROM courses WHERE programme_version_id = ?", [version_id, current["version_id"]])
    return version_id


def correct_student(conn: sqlite3.Connection, student_id: str, field: str, value, reason: str) -> int:
    """Correct one profile fact, e.g. field 'grades.cgpa'. A new profile version is saved; the old one is kept."""
    current = db.get_current(conn, "profile_versions", {"student_id": student_id})
    if current is None:
        raise ValueError(f"unknown student '{student_id}'")
    data = json.loads(json.dumps(current["data"]))
    *path, last = field.split(".")
    target = data
    for part in path:
        if part not in target:
            raise ValueError(f"unknown profile field '{field}'")
        target = target[part]
    if last not in target:
        raise ValueError(f"unknown profile field '{field}'")
    target[last] = value
    Profile.model_validate({**data, "documents": [], "preferences": []})     # the corrected profile must still be valid
    db.insert_once(conn, "corrections", {
        "target": "student", "target_id": student_id, "field": field, "value": json.dumps(value), "reason": reason,
        "idempotency_key": db.idempotency_key("student", student_id, field, json.dumps(value))})
    return db.add_version(conn, "profile_versions", {"student_id": student_id}, data, f"correction: {reason}")

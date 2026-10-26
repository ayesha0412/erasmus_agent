"""SQLite memory: the single source of truth for students, programmes, decisions and logs.

Three rules hold for every table:
- Nothing is overwritten. A correction adds a new version; the old one is marked superseded.
- A write that may be retried carries a UNIQUE idempotency key, so a retry never duplicates it.
- Every student-owned table has a student_id column, and every query on it filters by student_id.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(os.getenv("ERASMUS_DB", "erasmus.db"))
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id    TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    is_fictional  INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS programmes (
    programme_id  TEXT PRIMARY KEY,
    catalogue_id  TEXT NOT NULL,
    name          TEXT NOT NULL,
    role          TEXT,
    created_at    TEXT NOT NULL
);

-- Versioned tables: same pattern everywhere. Exactly one row per record has is_current = 1;
-- older rows point to the row that replaced them through superseded_by.

CREATE TABLE IF NOT EXISTS profile_versions (
    id            INTEGER PRIMARY KEY,
    student_id    TEXT NOT NULL REFERENCES students(student_id),
    version       INTEGER NOT NULL,
    data          TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('approved', 'pending', 'rejected')),
    is_current    INTEGER NOT NULL DEFAULT 0,
    superseded_by INTEGER REFERENCES profile_versions(id),
    source        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    UNIQUE (student_id, version)
);

CREATE TABLE IF NOT EXISTS document_versions (
    id            INTEGER PRIMARY KEY,
    student_id    TEXT NOT NULL REFERENCES students(student_id),
    doc_id        TEXT NOT NULL,
    doc_type      TEXT NOT NULL,
    version       INTEGER NOT NULL,
    data          TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('approved', 'pending', 'rejected')),
    is_current    INTEGER NOT NULL DEFAULT 0,
    superseded_by INTEGER REFERENCES document_versions(id),
    source        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    UNIQUE (student_id, doc_id, version)
);

CREATE TABLE IF NOT EXISTS preference_versions (
    id            INTEGER PRIMARY KEY,
    student_id    TEXT NOT NULL REFERENCES students(student_id),
    pref_key      TEXT NOT NULL,
    version       INTEGER NOT NULL,
    data          TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('approved', 'pending', 'rejected')),
    is_current    INTEGER NOT NULL DEFAULT 0,
    superseded_by INTEGER REFERENCES preference_versions(id),
    source        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    UNIQUE (student_id, pref_key, version)
);

CREATE TABLE IF NOT EXISTS programme_versions (
    id               INTEGER PRIMARY KEY,
    programme_id     TEXT NOT NULL REFERENCES programmes(programme_id),
    version          INTEGER NOT NULL,
    data             TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('approved', 'pending', 'rejected')),
    is_current       INTEGER NOT NULL DEFAULT 0,
    superseded_by    INTEGER REFERENCES programme_versions(id),
    source           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    page_fingerprint TEXT,
    fetched_at       TEXT,
    UNIQUE (programme_id, version)
);

-- Courses are stored exactly as scraped; topics are an extra column, never a replacement.
CREATE TABLE IF NOT EXISTS courses (
    id                   INTEGER PRIMARY KEY,
    programme_version_id INTEGER NOT NULL REFERENCES programme_versions(id),
    name                 TEXT NOT NULL,
    ects                 REAL,
    semester             TEXT,
    university           TEXT,
    description          TEXT,
    source_url           TEXT NOT NULL,
    topics               TEXT NOT NULL DEFAULT '[]'
);

-- History of what was decided: shortlists, rejections, approvals.
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY,
    student_id      TEXT NOT NULL REFERENCES students(student_id),
    cycle           INTEGER NOT NULL,
    kind            TEXT NOT NULL,
    programme_id    TEXT REFERENCES programmes(programme_id),
    data            TEXT NOT NULL DEFAULT '{}',
    decided_by      TEXT NOT NULL CHECK (decided_by IN ('student', 'agent')),
    created_at      TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS checklist_items (
    id              INTEGER PRIMARY KEY,
    student_id      TEXT NOT NULL REFERENCES students(student_id),
    programme_id    TEXT NOT NULL REFERENCES programmes(programme_id),
    item            TEXT NOT NULL,
    cycle           INTEGER NOT NULL,
    due_date        TEXT,
    status          TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done')),
    created_at      TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE
);

-- Working state of one cycle, checkpointed after every step so a crash can resume.
CREATE TABLE IF NOT EXISTS cycles (
    id          INTEGER PRIMARY KEY,
    student_id  TEXT NOT NULL REFERENCES students(student_id),
    cycle       INTEGER NOT NULL,
    cycle_date  TEXT NOT NULL,
    goal        TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT '{}',
    step        INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'running',
    updated_at  TEXT NOT NULL,
    UNIQUE (student_id, cycle)
);

CREATE TABLE IF NOT EXISTS action_log (
    id          INTEGER PRIMARY KEY,
    created_at  TEXT NOT NULL,
    student_id  TEXT,
    cycle       INTEGER,
    step        INTEGER,
    tool        TEXT NOT NULL,
    args        TEXT,
    result      TEXT,
    error       TEXT,
    model       TEXT,
    tokens      INTEGER,
    latency_ms  INTEGER
);
"""

# Which columns identify one record in each versioned table (its "key").
VERSIONED_TABLES = {
    "profile_versions": ("student_id",),
    "document_versions": ("student_id", "doc_id"),
    "preference_versions": ("student_id", "pref_key"),
    "programme_versions": ("programme_id",),
}

# Tables whose rows are written at most once, guarded by their UNIQUE idempotency_key.
IDEMPOTENT_TABLES = ("decisions", "checklist_items")


def now_utc() -> str:
    """Current time in UTC, e.g. '2026-10-26T10:15:00+00:00'."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the database with rows readable by column name and foreign keys enforced."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create every table if it does not exist yet. Safe to run again: existing data is kept."""
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def idempotency_key(*parts) -> str:
    """Build a stable key from its parts, e.g. 'stu-sara|deai|recommendation_letter|1'."""
    return "|".join(str(part) for part in parts)


# ---------- Students and programmes ----------

def add_student(conn: sqlite3.Connection, student_id: str, display_name: str, is_fictional: bool = False) -> None:
    """Register a student once; running it again changes nothing."""
    with conn:
        conn.execute(
            "INSERT INTO students (student_id, display_name, is_fictional, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (student_id) DO NOTHING",
            (student_id, display_name, int(is_fictional), now_utc()),
        )


def add_programme(conn: sqlite3.Connection, programme_id: str, catalogue_id: str, name: str, role: str | None = None) -> None:
    """Register a programme once; running it again changes nothing."""
    with conn:
        conn.execute(
            "INSERT INTO programmes (programme_id, catalogue_id, name, role, created_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (programme_id) DO NOTHING",
            (programme_id, catalogue_id, name, role, now_utc()),
        )


# ---------- Versioned records ----------

def _where(table: str, key: dict) -> tuple[str, list]:
    """Check the table and key against the allowed list, and build 'col = ? AND ...'."""
    if table not in VERSIONED_TABLES:
        raise ValueError(f"'{table}' is not a versioned table")
    if tuple(key) != VERSIONED_TABLES[table]:
        raise ValueError(f"{table} needs the key columns {VERSIONED_TABLES[table]}, got {tuple(key)}")
    return " AND ".join(f"{column} = ?" for column in key), list(key.values())


def _make_current(conn: sqlite3.Connection, table: str, key: dict, version_id: int) -> None:
    """Mark version_id as the current, approved version and supersede the previous one."""
    where, values = _where(table, key)
    conn.execute(
        f"UPDATE {table} SET is_current = 0, superseded_by = ? WHERE {where} AND is_current = 1",
        [version_id, *values],
    )
    conn.execute(f"UPDATE {table} SET is_current = 1, status = 'approved' WHERE id = ?", [version_id])


def add_version(
    conn: sqlite3.Connection,
    table: str,
    key: dict,
    data: dict,
    source: str,
    status: str = "approved",
    extra: dict | None = None,
) -> int:
    """Save a new version of a record, in one transaction, and return its id.

    key    identifies the record, e.g. {"student_id": "stu-sara", "doc_id": "english_test"}.
    status 'approved' makes the new version current at once. 'pending' stores it without
           replacing the current one, until approve_version() is called (e.g. a changed deadline).
    extra  values for table-specific columns, e.g. {"doc_type": ...} or {"page_fingerprint": ...}.

    If the same data is already current (or already waiting for approval), nothing is
    written and that row's id is returned, so retrying the same save is harmless.
    """
    where, values = _where(table, key)
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
    with conn:  # one transaction: either every statement below happens, or none does
        same = conn.execute(
            f"SELECT id FROM {table} WHERE {where} AND data = ? AND (is_current = 1 OR status = 'pending')",
            [*values, payload],
        ).fetchone()
        if same:
            return same["id"]

        latest = conn.execute(f"SELECT COALESCE(MAX(version), 0) FROM {table} WHERE {where}", values).fetchone()[0]
        row = {
            **key,
            "version": latest + 1,
            "data": payload,
            "status": status,
            "is_current": 0,
            "source": source,
            "created_at": now_utc(),
            **(extra or {}),
        }
        placeholders = ", ".join("?" for _ in row)
        cursor = conn.execute(f"INSERT INTO {table} ({', '.join(row)}) VALUES ({placeholders})", list(row.values()))
        new_id = cursor.lastrowid
        if status == "approved":
            _make_current(conn, table, key, new_id)
    return new_id


def approve_version(conn: sqlite3.Connection, table: str, version_id: int) -> None:
    """Approve a pending version: it becomes current and the old version is superseded."""
    if table not in VERSIONED_TABLES:
        raise ValueError(f"'{table}' is not a versioned table")
    row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", [version_id]).fetchone()
    if row is None or row["status"] != "pending":
        raise ValueError(f"{table} row {version_id} is not waiting for approval")
    key = {column: row[column] for column in VERSIONED_TABLES[table]}
    with conn:
        _make_current(conn, table, key, version_id)


def reject_version(conn: sqlite3.Connection, table: str, version_id: int) -> None:
    """Reject a pending version: the current version stays in use, the rejected one is kept as history."""
    if table not in VERSIONED_TABLES:
        raise ValueError(f"'{table}' is not a versioned table")
    with conn:
        updated = conn.execute(f"UPDATE {table} SET status = 'rejected' WHERE id = ? AND status = 'pending'",
                               [version_id]).rowcount
    if not updated:
        raise ValueError(f"{table} row {version_id} is not waiting for approval")


def get_current(conn: sqlite3.Connection, table: str, key: dict) -> dict | None:
    """Return the current version of a record (with 'data' decoded), or None if there is none."""
    where, values = _where(table, key)
    row = conn.execute(f"SELECT * FROM {table} WHERE {where} AND is_current = 1", values).fetchone()
    return _decode(row) if row else None


def get_history(conn: sqlite3.Connection, table: str, key: dict) -> list[dict]:
    """Return every version of a record, oldest first, including superseded and pending ones."""
    where, values = _where(table, key)
    rows = conn.execute(f"SELECT * FROM {table} WHERE {where} ORDER BY version", values).fetchall()
    return [_decode(row) for row in rows]


STUDENT_TABLES = ("profile_versions", "document_versions", "preference_versions")


def list_current(conn: sqlite3.Connection, table: str, student_id: str) -> list[dict]:
    """All current records of ONE student in a student-owned versioned table."""
    if table not in STUDENT_TABLES:
        raise ValueError(f"'{table}' is not a student-owned versioned table")
    rows = conn.execute(f"SELECT * FROM {table} WHERE student_id = ? AND is_current = 1 ORDER BY id", [student_id])
    return [_decode(row) for row in rows]


def _decode(row: sqlite3.Row) -> dict:
    """Turn a database row into a dict and parse its JSON 'data' column."""
    record = dict(row)
    record["data"] = json.loads(record["data"])
    return record


# ---------- Write-once rows (idempotent) ----------

def insert_once(conn: sqlite3.Connection, table: str, row: dict) -> bool:
    """Insert a row unless one with the same idempotency_key exists. Returns True if inserted."""
    if table not in IDEMPOTENT_TABLES:
        raise ValueError(f"'{table}' is not a write-once table")
    if "idempotency_key" not in row:
        raise ValueError("row needs an idempotency_key")
    row = {"created_at": now_utc(), **row}
    placeholders = ", ".join("?" for _ in row)
    with conn:
        cursor = conn.execute(
            f"INSERT INTO {table} ({', '.join(row)}) VALUES ({placeholders}) "
            "ON CONFLICT (idempotency_key) DO NOTHING",
            list(row.values()),
        )
    return cursor.rowcount == 1
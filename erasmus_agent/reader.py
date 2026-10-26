"""read_programme: the agent's evidence-retrieval tool (milestone M4).

Reads a programme's saved pages (never the live website), cleans them to text, and
fingerprints the text with SHA-256. Comparing that fingerprint with the one stored in
the current approved programme version tells the agent whether the pages have changed.
"""

import hashlib
import json
import re
import sqlite3
from pathlib import Path

from . import db
from .text import html_to_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = PROJECT_ROOT / "data" / "pages" / "real"


def fingerprint(texts: list[str]) -> str:
    """SHA-256 of the page texts with whitespace collapsed, so spacing or line endings never count as a change."""
    normalised = "\n".join(re.sub(r"\s+", " ", text).strip() for text in texts)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def read_programme(conn: sqlite3.Connection, programme_id: str, page_overrides: dict[str, str] | None = None) -> dict:
    """Return the programme's page texts with their sources, a fingerprint and a `changed` flag.

    page_overrides maps "programme/page" to a test copy, e.g. the cycle 2 deadline change.
    """
    meta_file = PAGES_DIR / programme_id / "meta.json"
    if not meta_file.is_file():
        raise ValueError(f"no saved pages for programme '{programme_id}'")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    overrides = page_overrides or {}

    pages = []
    for page in meta["pages"]:
        override = overrides.get(f"{programme_id}/{page['page']}")
        file = override or page["file"]
        pages.append({
            "page": page["page"],
            "url": page["url"],
            "fetched_at": page["fetched_at"],
            "file": file,
            "test_copy": override is not None,
            "text": html_to_text((PROJECT_ROOT / file).read_text(encoding="utf-8")),
        })

    current = fingerprint([p["text"] for p in pages])
    previous = db.get_current(conn, "programme_versions", {"programme_id": programme_id})
    previous_fingerprint = previous["page_fingerprint"] if previous else None
    return {
        "programme_id": programme_id,
        "name": meta["name"],
        "pages": pages,
        "fingerprint": current,
        "previous_fingerprint": previous_fingerprint,
        "first_read": previous is None,
        "changed": previous is not None and previous_fingerprint != current,
    }

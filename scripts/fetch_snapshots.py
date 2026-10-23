"""Polite scraper for the Erasmus Mundus data (milestone M1).

Two commands, run from the project root:
    python scripts/fetch_snapshots.py catalogue   # all programmes -> data/catalogue.json
    python scripts/fetch_snapshots.py pages       # full pages of the selected programmes

Every request checks robots.txt, waits 2 seconds and sends an honest User-Agent.
Raw HTML is saved with its URL and fetch time (and a SHA-256 fingerprint for
programme pages), so the agent and the tests read these files and never visit
live websites.
"""

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx2 as httpx  # openai 3.x ships httpx2, the renamed successor of httpx (same API)
from bs4 import BeautifulSoup

CATALOGUE_URL = "https://www.eacea.ec.europa.eu/scholarships/erasmus-mundus-catalogue_en"
USER_AGENT = (
    "ErasmusMundusTracker/0.1 (student university project; "
    "+https://github.com/ayesha0412/erasmus_agent)"
)
PAUSE_SECONDS = 2.0  # wait before every request so we never hammer the server
# EU project numbers come in two formats: new "101082454" and old "619804-EPP-1-2020-1-BE-EPPKA1-JMD-MOB".
PROJECT_ID_PATTERN = re.compile(r"\d{6}-EPP-[0-9A-Z-]+|\b10\d{7}\b")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "catalogue"
CATALOGUE_FILE = PROJECT_ROOT / "data" / "catalogue.json"
SELECTION_FILE = PROJECT_ROOT / "data" / "selected_programmes.json"
PAGES_DIR = PROJECT_ROOT / "data" / "pages" / "real"


# ---------- Shared: polite fetching ----------

def now_utc() -> str:
    """Current time in UTC, e.g. '2026-10-23T10:15:00+00:00'."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_robots(client: httpx.Client, url: str) -> RobotFileParser:
    """Download the site's robots.txt and return a parser that answers 'may I fetch this?'.

    Same rules as Python's own RobotFileParser.read(): 401/403 means everything is
    forbidden, any other error (e.g. 404, no robots.txt) means nothing is forbidden.
    """
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    response = client.get(robots_url)
    robots = RobotFileParser(robots_url)
    if response.status_code in (401, 403):
        robots.disallow_all = True
    elif response.status_code >= 400:
        robots.allow_all = True
    else:
        robots.parse(response.text.splitlines())
    return robots


def robots_for(client: httpx.Client, url: str, cache: dict) -> RobotFileParser:
    """Return the robots.txt rules for the site of `url`, downloading them once per site."""
    host = urlparse(url).netloc
    if host not in cache:
        cache[host] = load_robots(client, url)
    return cache[host]


def polite_get(client: httpx.Client, robots: RobotFileParser, url: str) -> str:
    """Fetch one URL as text, but only if robots.txt allows it, and only after a pause."""
    if not robots.can_fetch(USER_AGENT, url):
        raise PermissionError(f"robots.txt does not allow fetching {url}")
    time.sleep(PAUSE_SECONDS)
    response = client.get(url)
    response.raise_for_status()  # stop on 404, 500, etc. instead of saving an error page
    return response.text


def new_client() -> httpx.Client:
    """An HTTP client that always identifies itself with our User-Agent."""
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True)


# ---------- Command 1: the catalogue (M1a) ----------

def page_url(page_number: int) -> str:
    """Catalogue pages are numbered from 0; page 0 has no '?page=' part."""
    return CATALOGUE_URL if page_number == 0 else f"{CATALOGUE_URL}?page={page_number}"


def fetch_and_save_page(client: httpx.Client, robots: RobotFileParser, page_number: int) -> tuple[BeautifulSoup, dict]:
    """Download one catalogue page, save the raw HTML, and return it parsed plus its metadata."""
    url = page_url(page_number)
    html = polite_get(client, robots, url)
    raw_file = RAW_DIR / f"page_{page_number:02d}.html"
    raw_file.write_text(html, encoding="utf-8")
    meta = {
        "page": page_number,
        "url": url,
        "fetched_at": now_utc(),
        "raw_file": raw_file.relative_to(PROJECT_ROOT).as_posix(),
    }
    return BeautifulSoup(html, "html.parser"), meta


def parse_total(soup: BeautifulSoup) -> int:
    """Read the total number of programmes from the heading 'Projects (218)'."""
    for heading in soup.find_all("h2"):
        match = re.search(r"Projects\s*\((\d+)\)", heading.get_text(" ", strip=True))
        if match:
            return int(match.group(1))
    raise ValueError("No 'Projects (N)' heading found; the page layout may have changed.")


def parse_programmes(soup: BeautifulSoup, page_number: int) -> list[dict]:
    """Turn each programme card on one catalogue page into a small dictionary.

    Cards come in several layouts (title linked or plain text, links in different
    places), so links are classified by where they point, not where they sit:
    a europa.eu link is the EU project overview, any other link is the website.
    """
    programmes = []
    for card in soup.select("article.ecl-card"):
        title = card.select_one(".ecl-content-block__title")
        if title is None:
            continue  # not a programme card

        # The description starts with the acronym: "DEAI - Project overview" or just "SPREE -".
        description = card.select_one(".ecl-content-block__description")
        first_line = description.get_text("\n", strip=True).split("\n")[0] if description else ""
        acronym = first_line.rsplit(" -", 1)[0].strip() if " -" in first_line else None

        links = [urljoin(CATALOGUE_URL, a["href"]) for a in card.select("a[href]")]
        eu_links = [url for url in links if urlparse(url).netloc.endswith("europa.eu")]
        other_links = [url for url in links if url not in eu_links]
        website = other_links[0] if other_links else None
        overview_url = eu_links[0] if eu_links else None

        # The EU project number is a stable official ID; a few older projects have none.
        match = PROJECT_ID_PATTERN.search(overview_url) if overview_url else None
        project_id = match.group(0) if match else None

        programmes.append({
            "catalogue_id": project_id or acronym,
            "project_id": project_id,
            "acronym": acronym,
            "name": title.get_text(" ", strip=True),
            "website": website,
            "overview_url": overview_url,
            "catalogue_page": page_number,
        })
    return programmes


def fetch_catalogue() -> None:
    """Download every catalogue page and write data/catalogue.json."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    started_at = now_utc()

    with new_client() as client:
        robots = load_robots(client, CATALOGUE_URL)
        print(f"robots.txt allows the catalogue: {robots.can_fetch(USER_AGENT, CATALOGUE_URL)}")

        # Page 0 tells us the total, so we know exactly how many pages to fetch.
        soup, meta = fetch_and_save_page(client, robots, 0)
        total = parse_total(soup)
        programmes = parse_programmes(soup, 0)
        if not programmes:
            raise ValueError("No programme cards found on page 0; the page layout may have changed.")
        per_page = len(programmes)
        page_count = math.ceil(total / per_page)
        pages = [meta]
        print(f"Catalogue lists {total} programmes, {per_page} per page -> {page_count} pages")
        print(f"Page 1/{page_count}: {per_page} programmes")

        for page_number in range(1, page_count):
            soup, meta = fetch_and_save_page(client, robots, page_number)
            found = parse_programmes(soup, page_number)
            programmes.extend(found)
            pages.append(meta)
            print(f"Page {page_number + 1}/{page_count}: {len(found)} programmes")

    # Sanity checks: did we get every programme, exactly once?
    ids = [p["catalogue_id"] for p in programmes]
    duplicates = len(ids) - len(set(ids))
    missing_ids = ids.count(None)

    catalogue = {
        "source_url": CATALOGUE_URL,
        "user_agent": USER_AGENT,
        "fetched_at": started_at,
        "total_listed": total,
        "count": len(programmes),
        "pages": pages,
        "programmes": programmes,
    }
    CATALOGUE_FILE.write_text(json.dumps(catalogue, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved {len(programmes)} programmes to {CATALOGUE_FILE.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Expected {total} | duplicate IDs: {duplicates} | missing IDs: {missing_ids}")
    if len(programmes) != total or duplicates or missing_ids:
        print("WARNING: the numbers don't add up; check the raw HTML before using this file.")
        sys.exit(1)


# ---------- Command 2: pages of the selected programmes (M1b) ----------

def load_selection() -> list[dict]:
    """Read data/selected_programmes.json and check every programme exists in the catalogue."""
    selection = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))["programmes"]
    catalogue = json.loads(CATALOGUE_FILE.read_text(encoding="utf-8"))["programmes"]
    known_ids = {p["catalogue_id"] for p in catalogue}
    unknown = [p["programme_id"] for p in selection if p["catalogue_id"] not in known_ids]
    if unknown:
        raise ValueError(f"Not found in data/catalogue.json: {unknown}")
    return selection


def fetch_pages() -> None:
    """Save each selected programme's pages, plus a meta.json describing where they came from."""
    selection = load_selection()
    robots_cache: dict = {}
    page_total = 0

    with new_client() as client:
        for programme in selection:
            programme_id = programme["programme_id"]
            out_dir = PAGES_DIR / programme_id
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = []

            for page_name, url in programme["pages"].items():
                html = polite_get(client, robots_for(client, url, robots_cache), url)
                content = html.encode("utf-8")
                raw_file = out_dir / f"{page_name}.html"
                raw_file.write_bytes(content)  # exact bytes, so the fingerprint matches the file
                saved.append({
                    "page": page_name,
                    "url": url,
                    "fetched_at": now_utc(),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "file": raw_file.relative_to(PROJECT_ROOT).as_posix(),
                })
                page_total += 1
                print(f"{programme_id}/{page_name}: saved {len(content):,} bytes")

            meta = {
                "programme_id": programme_id,
                "catalogue_id": programme["catalogue_id"],
                "name": programme["name"],
                "role": programme["role"],
                "user_agent": USER_AGENT,
                "pages": saved,
            }
            meta_file = out_dir / "meta.json"
            meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved {page_total} pages for {len(selection)} programmes in {PAGES_DIR.relative_to(PROJECT_ROOT).as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Polite scraper for Erasmus Mundus data.")
    parser.add_argument("command", choices=["catalogue", "pages"], help="what to download")
    args = parser.parse_args()
    if args.command == "catalogue":
        fetch_catalogue()
    else:
        fetch_pages()


if __name__ == "__main__":
    main()
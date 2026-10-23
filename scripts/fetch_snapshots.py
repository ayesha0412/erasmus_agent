"""Polite scraper for the Erasmus Mundus catalogue (milestone M1a).

Downloads every page of the public EACEA catalogue once, saves the raw HTML
as evidence, and writes data/catalogue.json with one entry per programme.
The agent never visits live websites: it only reads the files saved here.

Run from the project root:
    python scripts/fetch_snapshots.py
"""

import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx2 as httpx 
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
OUTPUT_FILE = PROJECT_ROOT / "data" / "catalogue.json"


def now_utc() -> str:
    """Current time in UTC, e.g. '2026-10-23T10:15:00+00:00'."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_robots(client: httpx.Client, url: str) -> RobotFileParser:
    """Download the site's robots.txt and return a parser that answers 'may I fetch this?'."""
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    response = client.get(robots_url)
    response.raise_for_status()
    robots = RobotFileParser(robots_url)
    robots.parse(response.text.splitlines())
    return robots


def polite_get(client: httpx.Client, robots: RobotFileParser, url: str) -> str:
    """Fetch one URL as text, but only if robots.txt allows it, and only after a pause."""
    if not robots.can_fetch(USER_AGENT, url):
        raise PermissionError(f"robots.txt does not allow fetching {url}")
    time.sleep(PAUSE_SECONDS)
    response = client.get(url)
    response.raise_for_status()  # stop on 404, 500, etc. instead of saving an error page
    return response.text


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


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    started_at = now_utc()

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True) as client:
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
    OUTPUT_FILE.write_text(json.dumps(catalogue, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved {len(programmes)} programmes to {OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Expected {total} | duplicate IDs: {duplicates} | missing IDs: {missing_ids}")
    if len(programmes) != total or duplicates or missing_ids:
        print("WARNING: the numbers don't add up; check the raw HTML before using this file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
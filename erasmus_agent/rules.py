"""Deterministic decision rules (milestone M6): eligibility, missing documents, course match.

No LLM here: the same inputs always give the same answer, and every answer carries
reasons a person can check. Eligibility depends only on entry requirements; course
match only ranks programmes and never makes a student eligible or not.

See the results for a profile against the saved programmes:
    python -m erasmus_agent.rules data/profiles/stu-sara.json 2026-10-25
"""

import datetime as dt
import re
import sys
from collections import Counter

from .models import ProgrammeRecord, vocabulary

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

# English test families, so "IELTS (Academic)" and "IELTS Academic" compare as the same test.
TEST_FAMILIES = {"ielts": "ielts", "toefl": "toefl", "pearson": "pte", "pte": "pte",
                 "cambridge": "cambridge", "oxford": "oxford", "duolingo": "duolingo"}

# The scholarship application deadline: labels about applying, but not start dates, results,
# uploads of single documents or the self-funded track.
APPLICATION_LABEL = re.compile(r"application|scholarship|grant")
NOT_APPLICATION_LABEL = re.compile(
    r"start|open|result|decision|publication|accept|self.?funded|deposit|certif|upload|reference|"
    r"recommendation|proof|interview|notification")


def test_family(name: str) -> str | None:
    """Which English test a name refers to, e.g. 'IELTS (Academic)' -> 'ielts'."""
    lowered = name.lower()
    return next((family for word, family in TEST_FAMILIES.items() if word in lowered), None)


def application_deadlines(record: ProgrammeRecord) -> list:
    """The deadlines that are about submitting the (scholarship) application itself."""
    return [d for d in record.deadlines
            if APPLICATION_LABEL.search(d.label.lower()) and not NOT_APPLICATION_LABEL.search(d.label.lower())]


def check_eligibility(student: dict, programme: dict, today: dt.date) -> dict:
    """Eligible, Not eligible or Uncertain, with the reason for every check.

    blockers -> Not eligible; otherwise unknowns -> Uncertain; otherwise Eligible.
    """
    record: ProgrammeRecord = programme["record"]
    blockers, unknowns, passes, notes = [], [], [], []
    next_deadline = None

    # 1. Degree field
    field = student["profile"]["degree"]["degree_field"]
    accepted = record.accepted_degree_fields
    if field in accepted:
        passes.append(f"degree field '{field}' is accepted")
    elif not accepted:
        unknowns.append("the pages do not say which degree fields are accepted")
    elif record.degree_rule_vague:
        unknowns.append(f"'{field}' is not listed, but the page also accepts related or equivalent fields")
    else:
        blockers.append(f"'{field}' is not an accepted degree field (accepted: {', '.join(accepted)})")

    # 2. English
    tests = [d for d in student["documents"] if d["doc_type"] == "english_test"]
    required = [r for r in record.english_requirements if r.min_overall]   # skip tests with no numeric minimum
    if not required:
        unknowns.append("no minimum English score is stated")
    elif not tests:
        unknowns.append("no English test on file")
    else:
        test = tests[0]["details"]
        same = [r for r in required if test_family(r.test) == test_family(test["test"])]
        if not same:
            unknowns.append(f"accepted tests are {', '.join(r.test for r in required)}; the student has {test['test']}")
        elif any(test["overall"] >= r.min_overall for r in same):
            passes.append(f"{test['test']} {test['overall']} meets the minimum {min(r.min_overall for r in same)}")
        else:
            blockers.append(f"{test['test']} {test['overall']} is below the minimum {min(r.min_overall for r in same)}")

    # 3. Other languages
    known = {skill["language"]: skill["level"] for skill in student["profile"].get("languages", [])}
    for req in record.other_language_requirements:
        have = known.get(req.language)
        level = req.min_level or "a stated level"
        if have and (req.min_level is None or CEFR_LEVELS.index(have) >= CEFR_LEVELS.index(req.min_level)):
            passes.append(f"{req.language} {have} meets {level}")
        elif req.applies_to == "some_tracks":
            notes.append(f"{req.language} {level} is needed only for some study tracks: \"{req.quote[:150]}\"")
        else:
            blockers.append(f"{req.language} {level} is required for all students")

    # 4. Other conditions: the student's stored answers settle them; otherwise ask.
    answers = {a["condition"]: a["met"] for a in student.get("condition_answers", [])
               if a["programme_id"] == programme["programme_id"]}
    for condition in record.other_conditions:
        met = answers.get(condition)
        if met is True:
            passes.append(f"confirmed by the student: {condition}")
        elif met is False:
            blockers.append(f"the student does not meet: {condition}")
        else:
            unknowns.append(f"needs the student's confirmation: {condition}")

    # 5. Application deadline, judged on the cycle's date (manual clock).
    candidates = application_deadlines(record)
    upcoming = sorted(d.due for d in candidates if d.due and d.due >= today)
    passed = sorted(d.due for d in candidates if d.due and d.due < today)
    if upcoming:
        next_deadline = upcoming[0]
        passes.append(f"next application deadline {next_deadline} ({(next_deadline - today).days} days left)")
    elif any(d.due is None for d in candidates):
        unknowns.append("the application deadline is not published yet")
        if passed:
            notes.append(f"an older page shows {passed[-1]}, which has passed, so it was ignored")
    elif passed:
        blockers.append(f"the application deadline {passed[-1]} has passed")
    else:
        unknowns.append("no application deadline is stated")

    status = "not_eligible" if blockers else "uncertain" if unknowns else "eligible"
    return {
        "programme_id": programme["programme_id"],
        "status": status,
        "blockers": blockers,
        "unknowns": unknowns,
        "passes": passes,
        "notes": notes,
        "next_deadline": next_deadline.isoformat() if next_deadline else None,
        "evidence": [q for q in [record.degree_rule_quote] if q],
    }


def find_missing_documents(student: dict, record: ProgrammeRecord) -> dict:
    """Required minus owned, counted per document type (e.g. 2 letters required, 1 owned -> 1 missing)."""
    owned = Counter(d["doc_type"] for d in student["documents"])
    missing = []
    for req in record.required_documents:
        if req.doc_type == "other":
            continue  # too vague to check
        lacking = req.count - owned[req.doc_type]
        if lacking > 0:
            missing.append({"doc_type": req.doc_type, "required": req.count,
                            "owned": owned[req.doc_type], "missing": lacking})
    return {"programme_id": record.programme_id, "documents_listed": bool(record.required_documents),
            "missing": missing}


def covers(topic: str, interest: str) -> bool:
    """A course topic covers an interest if it IS the interest or its broader topic is."""
    return topic == interest or vocabulary()["broader"].get(topic) == interest


def match_score(student: dict, record: ProgrammeRecord) -> dict:
    """How many courses cover the student's interests, with example course names as evidence."""
    interests = student["profile"]["interest_topics"]
    total = len(record.courses)
    matched = [c for c in record.courses if any(covers(t, i) for t in c.topics for i in interests)]
    by_interest = {}
    for interest in interests:
        names = [c.name for c in record.courses if any(covers(t, interest) for t in c.topics)]
        by_interest[interest] = {"count": len(names), "examples": names[:3]}
    return {
        "programme_id": record.programme_id,
        "matched": len(matched),
        "total": total,
        "score": round(len(matched) / total, 2) if total else 0.0,
        "by_interest": by_interest,
        "summary": f"{len(matched)} of {total} courses cover {' or '.join(interests)}",
    }


def excluded_by_preferences(student: dict, record: ProgrammeRecord) -> str | None:
    """A programme is excluded by 'exclude_topic: X' if at least half of its tagged courses are about X."""
    tagged = [c for c in record.courses if c.topics]
    for preference in student["preferences"]:
        if preference["kind"] != "exclude_topic" or not tagged:
            continue
        share = sum(1 for c in tagged if any(covers(t, preference["value"]) for t in c.topics)) / len(tagged)
        if share >= 0.5:
            return f"excluded by the preference 'exclude {preference['value']}': {share:.0%} of its tagged courses are about it"
    return None


def main() -> None:
    """Print eligibility, match and missing documents for one profile file against erasmus.db."""
    from . import db
    from .memory import get_programme, profile_view
    from .models import load_profile

    student = profile_view(load_profile(sys.argv[1]))
    today = dt.date.fromisoformat(sys.argv[2])
    conn = db.connect()
    for (programme_id,) in conn.execute("SELECT programme_id FROM programmes ORDER BY programme_id"):
        programme = get_programme(conn, programme_id)
        result = check_eligibility(student, programme, today)
        match = match_score(student, programme["record"])
        missing = find_missing_documents(student, programme["record"])
        print(f"\n{programme_id}: {result['status'].upper()} | {match['summary']} | next deadline {result['next_deadline']}")
        for label in ("blockers", "unknowns", "notes"):
            for reason in result[label]:
                print(f"    {label[:-1]}: {reason}")
        print(f"    missing documents: {[(m['doc_type'], m['missing']) for m in missing['missing']] or 'none'}"
              + ("" if missing["documents_listed"] else " (no document list on the pages)"))


if __name__ == "__main__":
    main()

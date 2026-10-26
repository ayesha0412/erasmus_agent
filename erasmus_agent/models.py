"""Pydantic models: the shape and rules of every record before it reaches the database.

Profiles, cycle inputs and extracted programme records are all validated here.
Unknown fields, tags outside the frozen vocabulary, impossible dates and impossible
values are rejected with a clear error instead of being stored.
"""

import datetime as dt
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAGS_FILE = PROJECT_ROOT / "data" / "tags.json"

DocType = Literal[
    "transcript", "degree_certificate", "cv", "english_test", "passport",
    "recommendation_letter", "motivation_letter", "motivation_video", "residence_proof", "other",
]

CefrLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]   # Common European Framework of Reference levels

# Deadlines outside this range are almost certainly extraction mistakes.
EARLIEST_YEAR, LATEST_YEAR = 2020, 2035


@lru_cache
def vocabulary() -> dict:
    """Load the frozen vocabulary (data/tags.json) once and keep it in memory."""
    raw = json.loads(TAGS_FILE.read_text(encoding="utf-8"))
    return {
        "version": raw["version"],
        "topics": {t["topic"] for t in raw["topics"]},
        "broader": {t["topic"]: t["broader"] for t in raw["topics"]},
        "degree_fields": {f["field"] for f in raw["degree_fields"]},
    }


def known_topics(values: list[str]) -> list[str]:
    """Reject any topic that is not in the frozen vocabulary."""
    unknown = [v for v in values if v not in vocabulary()["topics"]]
    if unknown:
        raise ValueError(f"not in the topic vocabulary (data/tags.json): {unknown}")
    return values


def known_degree_fields(values: list[str]) -> list[str]:
    """Reject any degree field that is not in the frozen vocabulary."""
    unknown = [v for v in values if v not in vocabulary()["degree_fields"]]
    if unknown:
        raise ValueError(f"not in the degree-field vocabulary (data/tags.json): {unknown}")
    return values


class Strict(BaseModel):
    """Base for every model: unknown keys are errors, so a typo cannot slip through."""
    model_config = ConfigDict(extra="forbid")


# ---------- Student side ----------

class Degree(Strict):
    level: Literal["bachelor", "master"]
    field: str = Field(min_length=1)          # as the student wrote it
    degree_field: str                          # the matching vocabulary value, confirmed by the student
    institution: str
    graduation_year: int = Field(ge=1980, le=2030)
    ects: int = Field(gt=0, le=400)

    @field_validator("degree_field")
    @classmethod
    def _known_field(cls, value: str) -> str:
        return known_degree_fields([value])[0]


class Grades(Strict):
    cgpa: float = Field(ge=0)
    scale: float = Field(gt=0)

    @model_validator(mode="after")
    def _cgpa_within_scale(self):
        if self.cgpa > self.scale:
            raise ValueError(f"cgpa {self.cgpa} is higher than its scale {self.scale}")
        return self


class Document(Strict):
    doc_type: DocType
    doc_id: str | None = None                  # defaults to doc_type; set it for a second letter etc.
    version: int = Field(ge=1)
    added_on: dt.date
    details: dict = {}
    note: str | None = None

    @model_validator(mode="after")
    def _fill_and_check(self):
        if self.doc_id is None:
            self.doc_id = self.doc_type
        if self.doc_type == "english_test":
            score = self.details.get("overall")
            if not isinstance(score, (int, float)) or not self.details.get("test"):
                raise ValueError("an english_test document needs details.test and a numeric details.overall")
        return self


class Preference(Strict):
    kind: Literal["exclude_topic"]
    value: str
    added_on: dt.date
    reason: str = ""

    @field_validator("value")
    @classmethod
    def _known_topic(cls, value: str) -> str:
        return known_topics([value])[0]

    @property
    def pref_key(self) -> str:
        """Stable key for the database, e.g. 'exclude_topic:law'."""
        return f"{self.kind}:{self.value}"


class LanguageSkill(Strict):
    """A language other than English that the student speaks (English is proven by an english_test document)."""
    language: str = Field(pattern=r"^[a-z]+$")   # lowercase, e.g. "french"
    level: CefrLevel


class Profile(Strict):
    student_id: str = Field(pattern=r"^stu-[a-z0-9-]+$")
    fictional: bool = False
    name: str = Field(min_length=1)
    degree: Degree
    grades: Grades
    interests_text: str
    interest_topics: list[str] = Field(min_length=1)
    languages: list[LanguageSkill] = []
    documents: list[Document] = []
    preferences: list[Preference] = []

    @field_validator("interest_topics")
    @classmethod
    def _known_interests(cls, values: list[str]) -> list[str]:
        return known_topics(values)

    @model_validator(mode="after")
    def _one_entry_per_document(self):
        ids = [d.doc_id for d in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError(f"each document may appear once (use doc_id for a second one): {ids}")
        return self


class CycleInput(Strict):
    cycle: int = Field(ge=1)
    date: dt.date                              # the cycle's manual clock
    student_id: str = Field(pattern=r"^stu-[a-z0-9-]+$")
    goal: str
    page_overrides: dict[str, str] = {}        # "programme/page" -> path of a test copy
    new_documents: list[Document] = []
    new_preferences: list[Preference] = []

    @field_validator("page_overrides")
    @classmethod
    def _overrides_exist(cls, overrides: dict[str, str]) -> dict[str, str]:
        for page, path in overrides.items():
            if page.count("/") != 1:
                raise ValueError(f"override key must look like 'programme/page', got '{page}'")
            if not (PROJECT_ROOT / path).is_file():
                raise ValueError(f"override file not found: {path}")
        return overrides


# ---------- Programme side (the target of LLM extraction in M5) ----------

class Course(Strict):
    name: str = Field(min_length=1)            # exactly as scraped
    ects: float | None = Field(default=None, ge=0, le=60)
    semester: str | None = None
    university: str | None = None
    description: str | None = None
    source_url: str
    topics: list[str] = []                     # may be empty, e.g. Academic Writing

    @field_validator("topics")
    @classmethod
    def _known(cls, values: list[str]) -> list[str]:
        return known_topics(values)


class EnglishRequirement(Strict):
    test: str                                  # e.g. "IELTS Academic"
    min_overall: float = Field(ge=0)


class LanguageRequirement(Strict):
    """A language other than English that applicants must know. Language courses offered are not requirements."""
    language: str = Field(pattern=r"^[a-z]+$")   # lowercase, e.g. "french"
    min_level: CefrLevel | None = None           # None = required, but no level stated
    applies_to: Literal["all", "some_tracks"]    # some_tracks: only for a study track, mobility path or semester
    quote: str


class Deadline(Strict):
    label: str                                 # e.g. "scholarship application"
    due: dt.date | None                        # None = the page says it is not published yet
    quote: str                                 # the exact text it came from

    @field_validator("due")
    @classmethod
    def _plausible_year(cls, value: dt.date | None) -> dt.date | None:
        if value is not None and not EARLIEST_YEAR <= value.year <= LATEST_YEAR:
            raise ValueError(f"deadline year {value.year} is outside {EARLIEST_YEAR}-{LATEST_YEAR}")
        return value


class RequiredDocument(Strict):
    doc_type: DocType
    count: int = Field(default=1, ge=1, le=5)


class ProgrammeRecord(Strict):
    programme_id: str
    accepted_degree_fields: list[str] = []
    degree_rule_quote: str | None = None
    degree_rule_vague: bool = False            # e.g. "or equivalent", "related fields"
    english_requirements: list[EnglishRequirement] = []
    other_language_requirements: list[LanguageRequirement] = []
    other_conditions: list[str] = []           # rules code cannot check, e.g. "top 10% of class"
    deadlines: list[Deadline] = []
    required_documents: list[RequiredDocument] = []
    courses: list[Course] = []
    missing_information: list[str] = []        # what the pages do not say
    suspected_instructions: list[str] = []     # page text addressed to AI tools: flagged, never obeyed

    @field_validator("accepted_degree_fields")
    @classmethod
    def _known(cls, values: list[str]) -> list[str]:
        return known_degree_fields(values)


# ---------- Loading files ----------

def load_profile(path: Path | str) -> Profile:
    """Read and validate a profile file such as data/profiles/stu-sara.json."""
    return Profile.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_cycle(path: Path | str) -> CycleInput:
    """Read and validate a cycle file such as data/cycles/cycle2.json."""
    return CycleInput.model_validate_json(Path(path).read_text(encoding="utf-8"))
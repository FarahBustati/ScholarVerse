"""Agent 1: Reader (Profile Analyst).

The only agent that touches raw student text. No tools, no prose: it turns free text into a
validated StudentProfile and records what is missing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from scholarpath.schemas import ProfileExtraction, StudentProfile
from scholarpath.state import ScholarPathState

log = logging.getLogger(__name__)

READER_PROMPT = """You are the Reader agent of ScholarPath, a platform that matches students \
with educational opportunities.

Extract a structured profile from the student's text.
Rules:
- Use only information the student states. Never guess or invent values. Unknown means null \
or an empty list.
- gpa: copy the number as written. gpa_scale: only if the scale is stated or unambiguous \
(e.g. "3.7/4.0" -> 4, "85%" -> 100).
- career_goals: one sentence in the student's own terms.
- constraints: countries, budget in USD, funding need, languages, online/in-person, part-time, \
start timeline. Only what is stated.
- confidence: 0-1, how complete and unambiguous the text is for recommending opportunities."""

TRACKED_FIELDS = [
    "education_level",
    "field_of_study",
    "career_goals",
    "gpa",
    "skills",
    "interests",
    "institution",
    "graduation_year",
]
CRITICAL_FIELDS = ("field_of_study", "career_goals")


def normalize_gpa(gpa: float | None, scale: float | None) -> float | None:
    """Convert a GPA to the 0-4 scale. Infers common scales when none is given."""
    if gpa is None or gpa < 0:
        return None
    if scale is None:
        scale = next((s for s in (4.0, 5.0, 10.0, 20.0, 100.0) if gpa <= s), None)
    if not scale or gpa > scale:
        return None
    return round(gpa / scale * 4.0, 2)


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for item in items:
        clean = item.strip()
        if clean and clean.lower() not in seen:
            seen[clean.lower()] = clean
    return list(seen.values())


def normalize_profile(raw: ProfileExtraction) -> StudentProfile:
    """Rule-based validation applied after the LLM call."""
    year = raw.graduation_year
    if year is not None and not 1950 <= year <= date.today().year + 8:
        year = None
    profile = StudentProfile(
        education_level=raw.education_level,
        field_of_study=(raw.field_of_study or "").strip() or None,
        gpa=normalize_gpa(raw.gpa, raw.gpa_scale),
        institution=(raw.institution or "").strip() or None,
        graduation_year=year,
        skills=_dedupe(raw.skills),
        interests=_dedupe(raw.interests),
        career_goals=(raw.career_goals or "").strip() or None,
        constraints=raw.constraints.model_copy(
            update={
                "countries": _dedupe(raw.constraints.countries),
                "languages": _dedupe(raw.constraints.languages),
            }
        ),
        confidence=max(0.0, min(1.0, raw.confidence)),
    )
    profile.missing_fields = [f for f in TRACKED_FIELDS if getattr(profile, f) in (None, [], "")]
    return profile


def needs_clarification(profile: StudentProfile, threshold: float) -> bool:
    return profile.confidence < threshold or any(f in profile.missing_fields for f in CRITICAL_FIELDS)


def clarification_question(profile: StudentProfile) -> str:
    wanted = [f.replace("_", " ") for f in profile.missing_fields if f in TRACKED_FIELDS[:6]]
    return (
        "To find better matches, please tell me more about: "
        + ", ".join(wanted or ["your background and goals"])
        + ". Also mention any country, budget or online/in-person preferences."
    )


def make_reader(llm: BaseChatModel) -> Callable[[ScholarPathState], dict]:
    extractor = llm.with_structured_output(ProfileExtraction)

    def reader(state: ScholarPathState) -> dict:
        raw = extractor.invoke([SystemMessage(READER_PROMPT), HumanMessage(state["raw_input"])])
        if not isinstance(raw, ProfileExtraction):
            raw = ProfileExtraction.model_validate(raw)
        profile = normalize_profile(raw)
        log.info("reader: confidence=%.2f missing=%s", profile.confidence, profile.missing_fields)
        return {"profile": profile}

    return reader

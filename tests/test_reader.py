from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from scholarpath.agents.reader import (
    make_reader,
    needs_clarification,
    normalize_gpa,
    normalize_profile,
)
from scholarpath.offline_llm import heuristic_profile
from scholarpath.schemas import Constraints, ProfileExtraction
from tests.conftest import ScriptedLLM, example


def test_reader_parses_llm_output_and_records_missing_fields() -> None:
    extraction = ProfileExtraction(
        education_level="undergraduate",
        field_of_study="  Computer Science ",
        gpa=85,
        skills=["Python", "python", "SQL"],
        career_goals="ML engineer",
        constraints=Constraints(countries=["Canada", "canada"]),
        confidence=0.9,
    )
    llm = ScriptedLLM(structured=[extraction])

    update = make_reader(llm)({"raw_input": "student text"})
    profile = update["profile"]

    assert profile.field_of_study == "Computer Science"
    assert profile.gpa == 3.4  # 85 on a 100-point scale
    assert profile.skills == ["Python", "SQL"]
    assert profile.constraints.countries == ["Canada"]
    assert profile.missing_fields == ["interests", "institution", "graduation_year"]
    # The raw text reaches the model unchanged.
    human = [m for m in llm.calls[0] if isinstance(m, HumanMessage)]
    assert human[0].content == "student text"


@pytest.mark.parametrize(
    ("gpa", "scale", "expected"),
    [
        (3.7, 4.0, 3.7),
        (85, None, 3.4),
        (82, 100, 3.28),
        (8.5, 10, 3.4),
        (4.5, 5, 3.6),
        (None, None, None),
        (120, None, None),
    ],
)
def test_normalize_gpa(gpa: float | None, scale: float | None, expected: float | None) -> None:
    assert normalize_gpa(gpa, scale) == expected


def test_implausible_graduation_year_is_dropped() -> None:
    profile = normalize_profile(ProfileExtraction(graduation_year=1800, confidence=0.5))
    assert profile.graduation_year is None
    assert "graduation_year" in profile.missing_fields


def test_needs_clarification() -> None:
    complete = normalize_profile(
        ProfileExtraction(field_of_study="Biology", career_goals="doctor", confidence=0.8)
    )
    vague = normalize_profile(ProfileExtraction(field_of_study="Biology", confidence=0.8))
    unsure = normalize_profile(
        ProfileExtraction(field_of_study="Biology", career_goals="doctor", confidence=0.3)
    )
    assert not needs_clarification(complete, 0.5)
    assert needs_clarification(vague, 0.5)
    assert needs_clarification(unsure, 0.5)


def test_offline_extraction_of_examples() -> None:
    strong = normalize_profile(heuristic_profile(example("student_1.txt")))
    assert strong.education_level == "undergraduate"
    assert strong.field_of_study == "Computer Science"
    assert strong.gpa == 3.7
    assert strong.institution == "University of Toronto"
    assert strong.constraints.needs_funding
    assert strong.constraints.countries == ["Canada", "United Kingdom", "Germany"]

    budget = normalize_profile(heuristic_profile(example("student_3.txt")))
    assert budget.education_level == "graduate"
    assert budget.gpa == 3.28
    assert budget.constraints.max_budget_usd == 500
    assert budget.constraints.study_mode == "online"

    vague = normalize_profile(heuristic_profile(example("student_2.txt")))
    assert vague.education_level == "high_school"
    assert {"field_of_study", "career_goals"} <= set(vague.missing_fields)
    assert vague.confidence < 0.5

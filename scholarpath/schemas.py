"""Typed contracts between agents. Every agent reads and writes these models, never bare dicts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

EducationLevel = Literal["high_school", "undergraduate", "graduate", "professional"]


class Constraints(BaseModel):
    countries: list[str] = Field(default_factory=list, description="Preferred countries")
    max_budget_usd: int | None = Field(None, description="Maximum total cost the student can pay")
    needs_funding: bool = Field(False, description="True if the student needs a scholarship")
    languages: list[str] = Field(default_factory=list, description="Languages the student speaks")
    study_mode: Literal["online", "in_person"] | None = None
    part_time: bool = False
    timeline: str | None = Field(None, description="When the student wants to start")


class ProfileExtraction(BaseModel):
    """What the Reader's LLM returns. Normalized into StudentProfile afterwards."""

    education_level: EducationLevel | None = Field(
        None,
        description=(
            "Current stage: high_school (in or finishing high school), undergraduate "
            "(currently in a bachelor's), graduate (holds a bachelor's or is in a master's/PhD), "
            "professional (working, degree status unclear)"
        ),
    )
    field_of_study: str | None = Field(None, description="Main academic field")
    gpa: float | None = Field(None, description="GPA or grade average exactly as stated")
    gpa_scale: float | None = Field(None, description="Scale of the GPA if stated, e.g. 4, 10, 100")
    institution: str | None = None
    graduation_year: int | None = None
    skills: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    career_goals: str | None = None
    constraints: Constraints = Field(default_factory=Constraints)
    confidence: float = Field(0.5, ge=0, le=1, description="How complete and clear the input is")


class StudentProfile(BaseModel):
    education_level: EducationLevel | None = None
    field_of_study: str | None = None
    gpa: float | None = Field(None, description="Normalized to a 0-4 scale")
    institution: str | None = None
    graduation_year: int | None = None
    skills: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    career_goals: str | None = None
    constraints: Constraints = Field(default_factory=Constraints)
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class OpportunityType(StrEnum):
    DEGREE = "degree"
    SCHOLARSHIP = "scholarship"
    FELLOWSHIP = "fellowship"
    BOOTCAMP = "bootcamp"
    COURSE = "course"


class Opportunity(BaseModel):
    id: str
    name: str
    provider: str
    type: OpportunityType
    url: str
    country: str
    deadline: str | None = None
    cost_or_funding: str
    eligibility_summary: str
    description: str = ""
    source: Literal["catalog", "web"]
    # Attributes used for scoring. Web results usually leave these unknown.
    levels: list[EducationLevel] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    mode: Literal["online", "in_person", "hybrid"] | None = None
    cost_usd: int | None = None
    funded: bool = False
    min_gpa: float | None = None
    language: str | None = None
    # Filled by the Searcher's scorer.
    match_score: int = 0
    match_reasons: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class QueryRecord(BaseModel):
    attempt: int
    tool: str
    query: str
    filters: dict[str, str | int | bool] = Field(default_factory=dict)
    hits: int
    error: str | None = None


class SearchLog(BaseModel):
    attempts: int = 0
    broadened: bool = False
    web_enabled: bool = False
    queries: list[QueryRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

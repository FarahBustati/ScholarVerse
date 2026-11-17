"""Shared LangGraph state."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from scholarpath.schemas import Opportunity, SearchLog, StudentProfile


class ScholarPathState(TypedDict, total=False):
    # Inputs
    raw_input: str
    interactive: bool
    output_dir: str
    # Reader
    profile: StudentProfile
    clarification_rounds: int
    # Searcher
    opportunities: list[Opportunity]
    search_log: SearchLog
    search_attempts: int
    # Reporter
    report: str
    report_path: str
    # Any node may append non-fatal errors.
    errors: Annotated[list[str], operator.add]

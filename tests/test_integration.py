"""End-to-end run against the real Anthropic API. Skipped unless ANTHROPIC_API_KEY is set."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scholarpath.agents.reporter import validate_report
from scholarpath.config import get_settings
from scholarpath.graph import build_graph, run_pipeline
from scholarpath.tools import search_local_catalog
from tests.conftest import example

get_settings()  # loads .env so a key stored there is picked up

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]


def test_real_model_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCHOLARPATH_FAKE_LLM", "0")
    state = run_pipeline(
        example("student_1.txt"),
        graph=build_graph(tools=[search_local_catalog]),
        output_dir=str(tmp_path),
    )
    profile = state["profile"]
    assert profile.field_of_study and "computer" in profile.field_of_study.lower()
    assert state["opportunities"]
    assert validate_report(state["report"], [o.name for o in state["opportunities"]]) == []
    assert Path(state["report_path"]).exists()

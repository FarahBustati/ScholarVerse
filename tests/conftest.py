from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from scholarpath.schemas import Constraints, StudentProfile

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every test runs offline. Empty values also stop a local .env from filling them in."""
    monkeypatch.setenv("SCHOLARPATH_FAKE_LLM", "1")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("SCHOLARPATH_REPORTS_DIR", str(tmp_path / "reports"))


class ScriptedLLM:
    """Stand-in chat model that replays scripted outputs and records every input."""

    def __init__(self, structured: list[Any] | None = None, responses: list[AIMessage] | None = None):
        self.structured = list(structured or [])
        self.responses = list(responses or [])
        self.calls: list[Any] = []
        self.bound_tools: list[str] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        def respond(messages: Any) -> Any:
            self.calls.append(messages)
            return self.structured.pop(0)

        return RunnableLambda(respond)

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedLLM:
        self.bound_tools = [t.name for t in tools]
        return self

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        self.calls.append(list(messages))
        return self.responses.pop(0)


def example(name: str) -> str:
    return (EXAMPLES / name).read_text(encoding="utf-8")


@pytest.fixture
def cs_profile() -> StudentProfile:
    return StudentProfile(
        education_level="undergraduate",
        field_of_study="Computer Science",
        gpa=3.7,
        skills=["Python", "machine learning"],
        interests=["artificial intelligence", "healthcare"],
        career_goals="machine learning engineer working on medical AI",
        constraints=Constraints(countries=["Canada", "Germany"], needs_funding=True),
        confidence=0.9,
    )

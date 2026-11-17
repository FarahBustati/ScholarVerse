from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from langgraph.types import Command

from scholarpath.graph import build_graph, initial_state, route_after_reader, route_after_search, run_pipeline
from scholarpath.offline_llm import OfflineChatModel
from scholarpath.schemas import Opportunity, StudentProfile
from scholarpath.tools import search_local_catalog
from tests.conftest import example


@tool("search_local_catalog")
def empty_catalog(query: str) -> list[dict]:
    """Catalog that never finds anything."""
    return []


def _graph(tools=None):
    return build_graph(OfflineChatModel(), tools=tools or [search_local_catalog])


def test_routing_functions(cs_profile: StudentProfile) -> None:
    vague = StudentProfile(confidence=0.2, missing_fields=["field_of_study", "career_goals"])
    assert route_after_reader({"profile": vague, "interactive": True, "clarification_rounds": 0}) == "clarify"
    assert (
        route_after_reader({"profile": vague, "interactive": True, "clarification_rounds": 1}) == "searcher"
    )
    assert route_after_reader({"profile": vague, "interactive": False}) == "searcher"
    assert route_after_reader({"profile": cs_profile, "interactive": True}) == "searcher"

    strong = Opportunity(
        id="a", name="A", provider="P", type="course", url="u", country="Global",
        cost_or_funding="free", eligibility_summary="all", source="catalog", match_score=80,
    )  # fmt: skip
    assert route_after_search({"opportunities": [strong] * 2, "search_attempts": 1}) == "searcher"
    assert route_after_search({"opportunities": [strong] * 2, "search_attempts": 2}) == "reporter"
    assert route_after_search({"opportunities": [strong] * 3, "search_attempts": 1}) == "reporter"


def test_unclear_profile_interrupts_then_resumes(tmp_path: Path) -> None:
    graph = _graph()
    config = {"configurable": {"thread_id": "clarify-test"}}

    first = graph.invoke(initial_state(example("student_2.txt"), True, str(tmp_path)), config)
    assert "__interrupt__" in first
    assert "field of study" in first["__interrupt__"][0].value["question"]
    assert graph.get_state(config).next == ("clarify",)

    answer = "I want to study biology and become a doctor. I live in Egypt and need a scholarship."
    final = graph.invoke(Command(resume=answer), config)
    assert final["clarification_rounds"] == 1
    assert final["profile"].field_of_study == "biology"
    assert final["profile"].career_goals == "a doctor"
    assert final["profile"].constraints.countries == ["Egypt"]
    assert Path(final["report_path"]).exists()


def test_sparse_results_trigger_exactly_one_broadened_search(tmp_path: Path) -> None:
    visited: list[str] = []
    state = run_pipeline(
        example("student_1.txt"),
        graph=_graph(tools=[empty_catalog]),
        output_dir=str(tmp_path),
        on_update=lambda node, _: visited.append(node),
    )
    assert visited == ["reader", "searcher", "searcher", "reporter"]
    assert state["search_attempts"] == 2
    assert state["search_log"].broadened
    assert state["opportunities"] == []
    assert "## Top Matches" in state["report"]


def test_happy_path_writes_report(tmp_path: Path) -> None:
    visited: list[str] = []
    state = run_pipeline(
        example("student_1.txt"),
        graph=_graph(),
        output_dir=str(tmp_path),
        on_update=lambda node, _: visited.append(node),
    )
    assert visited == ["reader", "searcher", "reporter"]
    strong = [o for o in state["opportunities"] if o.match_score >= 50]
    assert len(strong) >= 3
    report = Path(state["report_path"]).read_text(encoding="utf-8")
    assert report == state["report"]
    assert f"### {state['opportunities'][0].name}" in report


def test_run_pipeline_answers_interrupts_with_callback(tmp_path: Path) -> None:
    questions: list[str] = []

    def ask(question: str) -> str:
        questions.append(question)
        return "I want to study public health and become an epidemiologist."

    state = run_pipeline(
        example("student_2.txt"), graph=_graph(), interactive=True, output_dir=str(tmp_path), ask=ask
    )
    assert len(questions) == 1
    assert state["profile"].field_of_study == "public health"
    assert Path(state["report_path"]).exists()

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from scholarpath.agents.scoring import dedupe, rank, score_opportunity
from scholarpath.agents.searcher import make_searcher
from scholarpath.config import get_settings
from scholarpath.schemas import Constraints, StudentProfile
from scholarpath.tools import default_tools, search_local_catalog
from scholarpath.tools.catalog import load_catalog, search_catalog
from tests.conftest import ScriptedLLM


@tool
def broken_tool(query: str) -> list[dict]:
    """A tool that always fails."""
    raise ConnectionError("service down")


def _catalog_by_id() -> dict:
    return {o.id: o for o in load_catalog(get_settings().catalog_path)}


def test_catalog_search_respects_filters() -> None:
    hits = search_catalog("data analytics", max_cost_usd=500, online_only=True)
    ids = {o.id for o in hits}
    assert {"alx-data-analytics", "google-data-analytics", "ibm-data-analyst"} <= ids
    assert "le-wagon-data" not in ids  # over budget
    assert all(o.mode == "online" for o in hits)


def test_catalog_country_filter_keeps_online_options() -> None:
    hits = search_catalog("machine learning", country="Germany")
    assert all(o.country == "Germany" or o.mode == "online" for o in hits)
    assert any(o.country == "Germany" for o in hits)


def test_scoring_rewards_fit_and_flags_concerns(cs_profile: StudentProfile) -> None:
    catalog = _catalog_by_id()
    funded_ai = score_opportunity(cs_profile, catalog["vector-scholarship-ai"])
    expensive = score_opportunity(cs_profile, catalog["ucl-msc-ml"])
    assert funded_ai.match_score >= 80
    assert any("Funded" in r for r in funded_ai.match_reasons)
    assert expensive.match_score < funded_ai.match_score
    assert any("not funded" in c for c in expensive.concerns)


def test_dedupe_and_top_k(cs_profile: StudentProfile) -> None:
    catalog = list(_catalog_by_id().values())
    duplicate = catalog[0].model_copy(update={"id": "copy", "name": catalog[0].name.upper()})
    assert len(dedupe([catalog[0], duplicate])) == 1
    ranked = rank(cs_profile, catalog + [duplicate], top_k=5)
    assert len(ranked) == 5
    assert [o.match_score for o in ranked] == sorted((o.match_score for o in ranked), reverse=True)


def test_web_search_is_only_enabled_with_a_key(monkeypatch) -> None:
    assert [t.name for t in default_tools()] == ["search_local_catalog"]
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    assert [t.name for t in default_tools()] == ["search_local_catalog", "search_web"]


def test_searcher_runs_tools_survives_failures_and_logs(cs_profile: StudentProfile) -> None:
    llm = ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_local_catalog",
                        "args": {"query": "machine learning", "country": "Canada"},
                        "id": "1",
                    },
                    {"name": "broken_tool", "args": {"query": "anything"}, "id": "2"},
                ],
            ),
            AIMessage(content="DONE"),
        ]
    )
    searcher = make_searcher(llm, tools=[search_local_catalog, broken_tool])
    update = searcher({"profile": cs_profile, "opportunities": []})

    log = update["search_log"]
    assert [q.tool for q in log.queries] == ["search_local_catalog", "broken_tool"]
    assert log.queries[0].filters == {"country": "Canada"}
    assert log.queries[1].error and "service down" in log.queries[1].error
    assert not log.web_enabled and "Web search skipped" in log.notes[0]
    assert update["errors"] == ["searcher: broken_tool failed: ConnectionError: service down"]
    assert update["opportunities"] and update["search_attempts"] == 1
    assert update["opportunities"][0].match_score >= update["opportunities"][-1].match_score


def test_searcher_caps_tool_calls(cs_profile: StudentProfile) -> None:
    calls = [{"name": "search_local_catalog", "args": {"query": f"q{i}"}, "id": str(i)} for i in range(10)]
    llm = ScriptedLLM(responses=[AIMessage(content="", tool_calls=calls)])
    update = make_searcher(llm, tools=[search_local_catalog], max_tool_calls=6)({"profile": cs_profile})
    assert len(update["search_log"].queries) == 6


def test_broadened_search_drops_filters(cs_profile: StudentProfile) -> None:
    call = {
        "name": "search_local_catalog",
        "args": {"query": "AI", "country": "Japan", "online_only": True},
        "id": "1",
    }
    llm = ScriptedLLM(responses=[AIMessage(content="", tool_calls=[call]), AIMessage(content="DONE")])
    update = make_searcher(llm, tools=[search_local_catalog])({"profile": cs_profile, "search_attempts": 1})
    assert update["search_attempts"] == 2
    assert update["search_log"].broadened
    assert update["search_log"].queries[0].filters == {}


def test_profile_constraints_reach_the_model() -> None:
    profile = StudentProfile(field_of_study="Biology", constraints=Constraints(max_budget_usd=100))
    llm = ScriptedLLM(responses=[AIMessage(content="DONE")])
    make_searcher(llm, tools=[search_local_catalog])({"profile": profile})
    assert '"max_budget_usd": 100' in llm.calls[0][1].content
    assert llm.bound_tools == ["search_local_catalog"]

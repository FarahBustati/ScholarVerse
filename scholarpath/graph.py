"""LangGraph orchestration: Reader -> Searcher -> Reporter, with two conditional loops.

    START -> reader --(unclear profile, interactive)--> clarify -> reader
                    \\-> searcher --(too few strong matches)--> searcher (broadened)
                                  \\-> reporter -> END

`clarify` is a human-in-the-loop step (LangGraph interrupt), not an agent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from scholarpath.agents import make_reader, make_reporter, make_searcher
from scholarpath.agents.reader import clarification_question, needs_clarification
from scholarpath.config import (
    CONFIDENCE_THRESHOLD,
    MAX_CLARIFICATION_ROUNDS,
    MAX_SEARCH_ATTEMPTS,
    MIN_STRONG_MATCHES,
    STRONG_MATCH_SCORE,
)
from scholarpath.llm import get_llm
from scholarpath.state import ScholarPathState

# State holds pydantic models, so the checkpointer must be allowed to restore them.
_STATE_MODELS = ("StudentProfile", "Constraints", "Opportunity", "OpportunityType")
_STATE_MODELS += ("SearchLog", "QueryRecord")
CHECKPOINT_TYPES = [("scholarpath.schemas", name) for name in _STATE_MODELS]


def default_checkpointer() -> MemorySaver:
    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))


def route_after_reader(state: ScholarPathState) -> str:
    if (
        state.get("interactive")
        and state.get("clarification_rounds", 0) < MAX_CLARIFICATION_ROUNDS
        and needs_clarification(state["profile"], CONFIDENCE_THRESHOLD)
    ):
        return "clarify"
    return "searcher"


def route_after_search(state: ScholarPathState) -> str:
    strong = [o for o in state.get("opportunities", []) if o.match_score >= STRONG_MATCH_SCORE]
    if len(strong) < MIN_STRONG_MATCHES and state.get("search_attempts", 0) < MAX_SEARCH_ATTEMPTS:
        return "searcher"
    return "reporter"


def clarify(state: ScholarPathState) -> dict:
    profile = state["profile"]
    answer = interrupt(
        {"question": clarification_question(profile), "missing_fields": profile.missing_fields}
    )
    return {
        "raw_input": f"{state['raw_input']}\n\nAdditional details from the student: {answer}",
        "clarification_rounds": state.get("clarification_rounds", 0) + 1,
    }


def build_graph(
    llm: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    llm = llm or get_llm()
    graph = StateGraph(ScholarPathState)
    graph.add_node("reader", make_reader(llm))
    graph.add_node("clarify", clarify)
    graph.add_node("searcher", make_searcher(llm, tools))
    graph.add_node("reporter", make_reporter(llm))

    graph.add_edge(START, "reader")
    graph.add_conditional_edges("reader", route_after_reader, ["clarify", "searcher"])
    graph.add_edge("clarify", "reader")
    graph.add_conditional_edges("searcher", route_after_search, ["searcher", "reporter"])
    graph.add_edge("reporter", END)
    return graph.compile(checkpointer=checkpointer or default_checkpointer())


def initial_state(text: str, interactive: bool = False, output_dir: str | None = None) -> ScholarPathState:
    state: ScholarPathState = {
        "raw_input": text,
        "interactive": interactive,
        "clarification_rounds": 0,
        "search_attempts": 0,
        "opportunities": [],
        "errors": [],
    }
    if output_dir:
        state["output_dir"] = output_dir
    return state


def run_pipeline(
    text: str,
    *,
    graph: CompiledStateGraph | None = None,
    interactive: bool = False,
    output_dir: str | None = None,
    ask: Callable[[str], str] | None = None,
    on_update: Callable[[str, dict[str, Any]], None] | None = None,
) -> ScholarPathState:
    """Run the graph to completion, answering clarification interrupts with `ask`."""
    graph = graph or build_graph()
    config = {"configurable": {"thread_id": str(uuid4())}}
    payload: Any = initial_state(text, interactive, output_dir)
    while True:
        pending = None
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    pending = update[0].value
                elif on_update:
                    on_update(node, update or {})
        if pending is None:
            return graph.get_state(config).values
        if ask is None:
            raise RuntimeError("The graph asked for clarification but no `ask` callback was given.")
        payload = Command(resume=ask(pending["question"]))

"""Agent 2: Searcher (Opportunity Scout).

The only agent with tools. It plans searches from the profile in a bounded tool-calling loop,
then ranks the candidates its tools returned. It never writes text addressed to the student.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from scholarpath.agents.scoring import rank
from scholarpath.config import MAX_TOOL_CALLS, TOP_K
from scholarpath.schemas import Opportunity, QueryRecord, SearchLog
from scholarpath.state import ScholarPathState
from scholarpath.tools import default_tools

log = logging.getLogger(__name__)

SEARCHER_PROMPT = """You are the Searcher agent of ScholarPath. Your job is to find educational \
opportunities (degrees, scholarships, fellowships, bootcamps, courses) for one student.

Use your tools to run 3-5 targeted searches derived from the profile:
- Cover the student's field, career goal and interests with different queries.
- Include funding searches (opportunity_type="scholarship" or "fellowship") when the student \
needs funding.
- Apply filters from the constraints (country, max_cost_usd, online_only) when they are stated.
- If "broaden" is true, earlier searches found too few strong matches: use broader, \
adjacent-field queries and do not use filters.
- You may make at most {max_tool_calls} tool calls in total. Make several calls per turn.

When you have searched enough, reply with the single word DONE. Do not write advice or \
explanations; ranking and reporting are handled by other agents."""

FILTER_KEYS = ("opportunity_type", "country", "max_cost_usd", "online_only")


def _summarize(hits: list[Opportunity]) -> str:
    if not hits:
        return "No results."
    return "\n".join(
        f"- {o.name} | {o.provider} | {o.type.value} | {o.country} | {o.cost_or_funding}" for o in hits
    )


def make_searcher(
    llm: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS,
    top_k: int = TOP_K,
) -> Callable[[ScholarPathState], dict]:
    tools = list(tools) if tools is not None else default_tools()
    by_name = {t.name: t for t in tools}
    model = llm.bind_tools(tools)

    def run_tool(call: dict, attempt: int, broaden: bool) -> tuple[list[Opportunity], QueryRecord]:
        args = dict(call.get("args") or {})
        if broaden:
            for key in FILTER_KEYS:
                args.pop(key, None)
        filters = {k: v for k, v in args.items() if k != "query" and v not in (None, False)}
        record = QueryRecord(
            attempt=attempt, tool=call["name"], query=str(args.get("query", "")), filters=filters, hits=0
        )
        tool = by_name.get(call["name"])
        if tool is None:
            record.error = f"unknown tool {call['name']}"
            return [], record
        try:
            hits = [Opportunity.model_validate(item) for item in tool.invoke(args)]
        except Exception as exc:  # a failing tool must not crash the graph
            record.error = f"{type(exc).__name__}: {exc}"
            log.warning("searcher: %s failed: %s", call["name"], record.error)
            return [], record
        record.hits = len(hits)
        return hits, record

    def searcher(state: ScholarPathState) -> dict:
        profile = state["profile"]
        attempt = state.get("search_attempts", 0) + 1
        broaden = attempt > 1
        search_log = (state.get("search_log") or SearchLog()).model_copy(deep=True)
        search_log.attempts = attempt
        search_log.web_enabled = "search_web" in by_name
        if broaden:
            search_log.broadened = True
            search_log.notes.append("Too few strong matches, so the search was broadened.")
        elif not search_log.web_enabled:
            search_log.notes.append("Web search skipped: TAVILY_API_KEY is not set.")

        context = {
            "profile": profile.model_dump(mode="json"),
            "broaden": broaden,
            "available_tools": list(by_name),
        }
        messages = [
            SystemMessage(SEARCHER_PROMPT.format(max_tool_calls=max_tool_calls)),
            HumanMessage(
                f"Find opportunities for this student.\n```json\n{json.dumps(context, indent=2)}\n```"
            ),
        ]
        candidates = list(state.get("opportunities", []))
        errors: list[str] = []
        calls = 0
        while calls < max_tool_calls:
            reply = model.invoke(messages)
            messages.append(reply)
            if not isinstance(reply, AIMessage) or not reply.tool_calls:
                break
            for call in reply.tool_calls:
                if calls >= max_tool_calls:
                    messages.append(ToolMessage("Tool call limit reached.", tool_call_id=call["id"]))
                    continue
                calls += 1
                hits, record = run_tool(call, attempt, broaden)
                search_log.queries.append(record)
                candidates.extend(hits)
                if record.error:
                    errors.append(f"searcher: {record.tool} failed: {record.error}")
                content = f"error: {record.error}" if record.error else _summarize(hits)
                messages.append(ToolMessage(content, tool_call_id=call["id"]))

        ranked = rank(profile, candidates, top_k)
        log.info("searcher: attempt=%d calls=%d candidates=%d", attempt, calls, len(candidates))
        return {
            "opportunities": ranked,
            "search_log": search_log,
            "search_attempts": attempt,
            "errors": errors,
        }

    return searcher

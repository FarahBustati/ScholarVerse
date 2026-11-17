"""Deterministic offline stand-in for the chat model (SCHOLARPATH_FAKE_LLM=1).

It speaks the same interface the agents use (structured output, tool calling, plain text)
and reads the same inputs a real model gets, so the whole graph runs without network access.
Its answers are rule-based and much weaker than a real model's. It exists for demos and tests.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from scholarpath.schemas import Constraints, ProfileExtraction

SKILLS = [
    "python", "java", "javascript", "sql", "excel", "machine learning", "deep learning",
    "data analysis", "statistics", "tableau", "power bi", "research", "writing", "leadership",
    "public speaking", "matlab", "cad", "project management", "marketing", "lab work",
]  # fmt: skip
INTERESTS = {
    "artificial intelligence": "artificial intelligence", "medical ai": "artificial intelligence",
    "healthcare": "healthcare", "public health": "public health", "medicine": "medicine",
    "science": "science", "biology": "biology", "environment": "environment",
    "climate": "climate", "sustainability": "sustainability", "education": "education",
    "business": "business", "finance": "finance", "entrepreneurship": "entrepreneurship",
    "data analytics": "data analytics", "data science": "data science", "robotics": "robotics",
    "design": "design", "economics": "economics", "helping people": "social impact",
    "renewable energy": "renewable energy", "software": "software",
}  # fmt: skip
COUNTRIES = {
    "canada": "Canada", "united kingdom": "United Kingdom", "uk": "United Kingdom",
    "germany": "Germany", "united states": "United States", "usa": "United States",
    "netherlands": "Netherlands", "australia": "Australia", "japan": "Japan",
    "france": "France", "egypt": "Egypt", "ireland": "Ireland", "sweden": "Sweden",
}  # fmt: skip
LANGUAGES = ["English", "Arabic", "German", "French", "Spanish", "Japanese"]

_FIELD_RE = re.compile(
    r"(?:studying|majoring in|major in|degree in|bachelor'?s (?:degree )?in|"
    r"master'?s (?:degree )?in|studies in|study)\s+([A-Za-z][A-Za-z &]+?)"
    r"(?=\s+(?:at|and|with|from|or|because)\b|[.,;(]|$)"
)
_GOAL_RE = re.compile(
    r"(?:want to|goal is to|hope to|aspire to|plan to|dream is to|would like to|i'd like to)\s+"
    r"(?:become|be|work as|work in|move into)\s+([^.;\n]+)",
    re.IGNORECASE,
)
_BECOME_RE = re.compile(r"\b(?:become|work as)\s+([^.;,\n]+)", re.IGNORECASE)
_GPA_RE = re.compile(
    r"GPA\s*(?:of|is|:)?\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(?:/|out of)\s*([0-9]+(?:\.[0-9]+)?))?",
    re.IGNORECASE,
)
_INSTITUTION_RE = re.compile(
    r"\b(?:at|from)\s+((?:the\s+)?(?:[A-Z][\w.&'-]*\s+)*(?:University|College|Institute|School)"
    r"(?:\s+of(?:\s+[A-Z][\w-]*)+)?)"
)
_YEAR_RE = re.compile(r"(?:graduat(?:e|ed|ing)\s+(?:in\s+)?|class of\s+)(\d{4})", re.IGNORECASE)
_BUDGET_RE = re.compile(r"\$\s?([\d,]+)|([\d,]+)\s*(?:usd|dollars)", re.IGNORECASE)
_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _find_terms(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    return [t for t in terms if re.search(rf"\b{re.escape(t.lower())}\b", low)]


def heuristic_profile(text: str) -> ProfileExtraction:
    """Rule-based profile extraction used by the offline model."""
    low = text.lower()

    if "high school" in low or "secondary school" in low:
        level = "high_school"
    elif re.search(r"\bgraduated\b|\bhave a (?:bachelor|master)|\bhold a\b|bachelor'?s degree", low):
        level = "graduate"
    elif re.search(
        r"undergraduate|university student|\bstudying\b|\b(?:first|second|third|fourth)-year", low
    ):
        level = "undergraduate"
    elif re.search(r"\bwork(?:ing)? (?:full|part)[ -]time\b|professional", low):
        level = "professional"
    else:
        level = None

    field = m.group(1).strip() if (m := _FIELD_RE.search(text)) else None
    m = _GOAL_RE.search(text) or _BECOME_RE.search(text)
    goal = m.group(1).strip() if m else None
    gpa = scale = None
    if m := _GPA_RE.search(text):
        gpa = float(m.group(1))
        scale = float(m.group(2)) if m.group(2) else None
    institution = m.group(1).strip() if (m := _INSTITUTION_RE.search(text)) else None
    year = int(m.group(1)) if (m := _YEAR_RE.search(text)) else None

    interests = sorted({INTERESTS[k] for k in _find_terms(text, list(INTERESTS))})
    countries = list(dict.fromkeys(COUNTRIES[k] for k in _find_terms(text, list(COUNTRIES))))
    budget = None
    if m := _BUDGET_RE.search(text):
        budget = int((m.group(1) or m.group(2)).replace(",", ""))
    mode = "online" if "online" in low else "in_person" if "on campus" in low else None

    constraints = Constraints(
        countries=countries,
        max_budget_usd=budget,
        needs_funding=bool(
            re.search(
                r"full(?:y)? fund|need (?:a )?scholarship|can'?t afford|cannot afford"
                r"|financial aid|need funding",
                low,
            )
        ),
        languages=_find_terms(text, LANGUAGES),
        study_mode=mode,
        part_time="part-time" in low or "part time" in low,
    )
    found = [level, field, goal, gpa, _find_terms(text, SKILLS), interests]
    return ProfileExtraction(
        education_level=level,
        field_of_study=field,
        gpa=gpa,
        gpa_scale=scale,
        institution=institution,
        graduation_year=year,
        skills=_find_terms(text, SKILLS),
        interests=interests,
        career_goals=goal,
        constraints=constraints,
        confidence=round(sum(1 for f in found if f) / len(found), 2),
    )


def _last_human_text(messages: Any) -> str:
    if hasattr(messages, "to_messages"):
        messages = messages.to_messages()
    if isinstance(messages, str):
        return messages
    humans = [m for m in messages if isinstance(m, HumanMessage)]
    return str(humans[-1].content) if humans else ""


def _context(messages: list[BaseMessage]) -> dict[str, Any]:
    match = _JSON_BLOCK_RE.search(_last_human_text(messages))
    return json.loads(match.group(1)) if match else {}


class OfflineChatModel(BaseChatModel):
    tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scholarpath-offline"

    def bind_tools(self, tools: Any, **kwargs: Any) -> OfflineChatModel:  # type: ignore[override]
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.model_copy(update={"tool_names": names})

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:  # type: ignore[override]
        if schema is not ProfileExtraction:
            raise NotImplementedError(f"Offline model has no structured output for {schema}")
        return RunnableLambda(lambda messages: heuristic_profile(_last_human_text(messages)))

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        if self.tool_names:
            message = self._search_turn(messages)
        else:
            message = AIMessage(content=_render_report(_context(messages)))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _search_turn(self, messages: list[BaseMessage]) -> AIMessage:
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="DONE")
        ctx = _context(messages)
        profile, broaden = ctx.get("profile", {}), ctx.get("broaden", False)
        cons = profile.get("constraints", {})
        interests = profile.get("interests", [])
        base = profile.get("field_of_study") or (interests[0] if interests else "scholarship")
        filters: dict[str, Any] = {}
        if not broaden:
            if cons.get("countries"):
                filters["country"] = cons["countries"][0]
            if cons.get("max_budget_usd") is not None:
                filters["max_cost_usd"] = cons["max_budget_usd"]
            if cons.get("study_mode") == "online":
                filters["online_only"] = True

        queries = [base, f"{base} scholarship"]
        if profile.get("career_goals"):
            queries.append(" ".join(profile["career_goals"].split()[:6]))
        if interests:
            queries.append(" ".join(interests[:3]))
        if broaden:
            queries = [base, *interests[:2], "scholarship", "online course"]
        calls = [
            {"name": "search_local_catalog", "args": {"query": q, **filters}, "id": f"call_{i}"}
            for i, q in enumerate(queries[:4])
        ]
        if "search_web" in self.tool_names:
            calls.append(
                {"name": "search_web", "args": {"query": f"{base} scholarships 2027"}, "id": "call_web"}
            )
        return AIMessage(content="", tool_calls=calls)


def _render_report(ctx: dict[str, Any]) -> str:
    profile = ctx.get("profile", {})
    opps = ctx.get("opportunities", [])[:5]
    log = ctx.get("search_log", {})
    cons = profile.get("constraints", {})

    lines = ["# ScholarPath Report", "", "## Profile Summary", ""]
    level = (profile.get("education_level") or "unknown level").replace("_", " ")
    field = profile.get("field_of_study") or "an undecided field"
    lines.append(f"You are at the {level} stage in {field}.")
    if profile.get("career_goals"):
        lines.append(f"Career goal: {profile['career_goals']}.")
    if cons.get("countries"):
        lines.append(f"Preferred countries: {', '.join(cons['countries'])}.")
    if cons.get("needs_funding"):
        lines.append("You need funding, so funded options rank higher.")
    if cons.get("max_budget_usd") is not None:
        lines.append(f"Budget: up to ${cons['max_budget_usd']:,}.")

    lines += ["", "## Top Matches", ""]
    if not opps:
        lines.append("No opportunities matched well enough. Add more detail to your profile and try again.")
    for opp in opps:
        lines += [
            f"### {opp['name']}",
            f"*{opp['provider']} - {opp['type']} - {opp['country']} - match {opp['match_score']}/100*",
            "",
            f"- **Why it fits:** {'; '.join(opp['match_reasons']) or 'General relevance to your profile.'}",
            f"- **Concerns:** {'; '.join(opp['concerns']) or 'None found.'}",
            f"- **Deadline:** {opp.get('deadline') or 'Check the provider site.'}",
            f"- **Next step:** Read the eligibility rules at {opp['url']} and list the documents you need.",
            "",
        ]

    lines += ["## Profile Gaps", ""]
    missing = profile.get("missing_fields", [])
    if missing:
        lines += [
            f"- Add your **{f.replace('_', ' ')}** so matches can be scored more precisely." for f in missing
        ]
    else:
        lines.append("- Your profile is complete enough for confident matching.")

    first = opps[0]["name"] if opps else "your top match"
    lines += [
        "",
        "## 30/60/90-Day Plan",
        "",
        f"- **30 days:** Shortlist 3 options, starting with {first}. "
        "Note every deadline and required document.",
        "- **60 days:** Draft your statement of purpose and CV. Request recommendation letters.",
        "- **90 days:** Submit your first application and prepare for any tests or interviews.",
        "",
        "## Sources and Limitations",
        "",
        f"- Searches run: {len(log.get('queries', []))} across {log.get('attempts', 1)} attempt(s)"
        + (" (broadened)." if log.get("broadened") else "."),
        f"- Web search: {'enabled' if log.get('web_enabled') else 'disabled, local catalog only'}.",
        "- Catalog entries are sample data. Verify deadlines and amounts on each provider's site.",
        "- Written by the offline rule-based model. Set ANTHROPIC_API_KEY for a richer report.",
    ]
    return "\n".join(lines) + "\n"

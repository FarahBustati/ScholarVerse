"""Agent 3: Reporter (Advisor).

The only agent that writes to the student. No tools, no searching: it turns the profile and the
ranked opportunities into a grounded markdown report, and it may only name opportunities that
the Searcher found.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from scholarpath.config import get_settings
from scholarpath.state import ScholarPathState
from scholarpath.text import normalize_name, slugify

log = logging.getLogger(__name__)

REPORTER_PROMPT = """You are the Reporter agent of ScholarPath, a concise and practical student \
advisor. Write a markdown report for the student from the JSON you are given.

Use exactly this structure:

# ScholarPath Report
## Profile Summary
(3-5 sentences about the student's background, goals and constraints)
## Top Matches
(one block per opportunity, best first, up to 5)
### <exact opportunity name from the JSON>
- **Why it fits:** ...
- **Concerns:** ...
- **Deadline:** ...
- **Next step:** one concrete action
## Profile Gaps
(missing information and how adding it would improve the matches)
## 30/60/90-Day Plan
## Sources and Limitations
(what was searched, whether web search was used, and that details must be verified)

Grounding rules:
- Only discuss opportunities in the JSON. Use their names exactly as given in the ### headings.
- Never name any other program, scholarship, university or course.
- Use only "###" headings for opportunities. Do not add other "###" headings.
- Do not invent deadlines, amounts or requirements. Say "check the provider site" instead."""

REQUIRED_SECTIONS = [
    "## Profile Summary",
    "## Top Matches",
    "## Profile Gaps",
    "## 30/60/90-Day Plan",
    "## Sources and Limitations",
]


class ReportGroundingError(RuntimeError):
    """The report failed validation after one regeneration."""


def validate_report(report: str, allowed_names: list[str]) -> list[str]:
    """Return a list of problems. Empty means the report is grounded and complete."""
    problems = [f"missing section '{s}'" for s in REQUIRED_SECTIONS if s not in report]
    allowed = {normalize_name(n) for n in allowed_names}
    named = [h.strip() for h in re.findall(r"^###\s+(.+)$", report, flags=re.MULTILINE)]
    problems += [
        f"'{name}' is not one of the found opportunities"
        for name in named
        if normalize_name(name) not in allowed
    ]
    if allowed_names and not named:
        problems.append("no opportunity headings under Top Matches")
    return problems


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def save_report(report: str, output_dir: Path, slug_source: str | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"{stamp}_{slugify(slug_source or 'student')}.md"
    path.write_text(report, encoding="utf-8")
    return path


def make_reporter(llm: BaseChatModel) -> Callable[[ScholarPathState], dict]:
    def reporter(state: ScholarPathState) -> dict:
        profile = state["profile"]
        opportunities = state.get("opportunities", [])
        search_log = state.get("search_log")
        context = {
            "profile": profile.model_dump(mode="json"),
            "opportunities": [o.model_dump(mode="json", exclude={"description"}) for o in opportunities],
            "search_log": search_log.model_dump(mode="json") if search_log else {},
        }
        messages: list[BaseMessage] = [
            SystemMessage(REPORTER_PROMPT),
            HumanMessage(
                f"Write the report for this student.\n```json\n{json.dumps(context, indent=2)}\n```"
            ),
        ]
        names = [o.name for o in opportunities]

        report = _text(llm.invoke(messages))
        problems = validate_report(report, names)
        if problems:
            log.warning("reporter: regenerating, problems=%s", problems)
            messages += [
                AIMessage(report),
                HumanMessage(
                    "Your report has problems:\n- "
                    + "\n- ".join(problems)
                    + "\nRewrite the full report. Allowed opportunity names:\n- "
                    + "\n- ".join(names or ["(none)"])
                ),
            ]
            report = _text(llm.invoke(messages))
            problems = validate_report(report, names)
            if problems:
                raise ReportGroundingError("Report failed validation twice: " + "; ".join(problems))

        output_dir = Path(state.get("output_dir") or get_settings().reports_dir)
        path = save_report(report, output_dir, profile.field_of_study or profile.career_goals)
        log.info("reporter: wrote %s", path)
        return {"report": report, "report_path": str(path)}

    return reporter

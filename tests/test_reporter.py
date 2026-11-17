from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from scholarpath.agents.reporter import ReportGroundingError, make_reporter, validate_report
from scholarpath.schemas import Opportunity, SearchLog, StudentProfile
from tests.conftest import ScriptedLLM

SECTIONS = (
    "## Profile Summary\nok\n## Top Matches\n{matches}\n## Profile Gaps\nok\n"
    "## 30/60/90-Day Plan\nok\n## Sources and Limitations\nok\n"
)


def _report(*names: str) -> str:
    return "# ScholarPath Report\n" + SECTIONS.format(matches="\n".join(f"### {n}\n- fits" for n in names))


def _state(tmp_path: Path, cs_profile: StudentProfile) -> dict:
    opp = Opportunity(
        id="x", name="Vector Scholarship in Artificial Intelligence", provider="Vector Institute",
        type="scholarship", url="https://vectorinstitute.ai/", country="Canada",
        cost_or_funding="Funded", eligibility_summary="AI master's applicants", source="catalog",
    )  # fmt: skip
    return {
        "profile": cs_profile,
        "opportunities": [opp],
        "search_log": SearchLog(),
        "output_dir": str(tmp_path),
    }


def test_validate_report() -> None:
    allowed = ["Chevening Scholarship"]
    assert validate_report(_report("Chevening Scholarship"), allowed) == []
    assert validate_report(_report("chevening scholarship"), allowed) == []
    problems = validate_report(_report("Made Up Fellowship"), allowed)
    assert problems == ["'Made Up Fellowship' is not one of the found opportunities"]
    assert "missing section '## Profile Gaps'" in validate_report(
        "# ScholarPath Report\n## Top Matches\n", []
    )


def test_reporter_regenerates_once_when_ungrounded(tmp_path: Path, cs_profile: StudentProfile) -> None:
    good = _report("Vector Scholarship in Artificial Intelligence")
    llm = ScriptedLLM(responses=[AIMessage(_report("Imaginary AI Grant")), AIMessage(good)])

    update = make_reporter(llm)(_state(tmp_path, cs_profile))

    assert update["report"] == good
    assert len(llm.calls) == 2
    correction = llm.calls[1][-1].content
    assert (
        "Imaginary AI Grant" in correction and "Vector Scholarship in Artificial Intelligence" in correction
    )
    path = Path(update["report_path"])
    assert path.parent == tmp_path and path.read_text(encoding="utf-8") == good
    assert path.name.endswith("_computer-science.md")


def test_reporter_fails_loudly_after_second_bad_report(tmp_path: Path, cs_profile: StudentProfile) -> None:
    llm = ScriptedLLM(responses=[AIMessage(_report("Fake One")), AIMessage(_report("Fake Two"))])
    with pytest.raises(ReportGroundingError, match="Fake Two"):
        make_reporter(llm)(_state(tmp_path, cs_profile))
    assert not list(tmp_path.glob("*.md"))

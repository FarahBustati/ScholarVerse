"""Optional live web search through Tavily. Only bound to the Searcher when TAVILY_API_KEY is set."""

from __future__ import annotations

import hashlib
from urllib.parse import urlparse

from langchain_core.tools import tool

from scholarpath.config import get_settings
from scholarpath.schemas import Opportunity, OpportunityType

_TYPE_HINTS = [
    (OpportunityType.SCHOLARSHIP, ("scholarship", "grant", "bursary")),
    (OpportunityType.FELLOWSHIP, ("fellowship",)),
    (OpportunityType.BOOTCAMP, ("bootcamp",)),
    (OpportunityType.COURSE, ("course", "certificate", "mooc")),
]


def result_to_opportunity(result: dict) -> Opportunity:
    title = (result.get("title") or "Untitled result").strip()
    url = result.get("url", "")
    text = f"{title} {result.get('content', '')}".lower()
    opp_type = next((t for t, hints in _TYPE_HINTS if any(h in text for h in hints)), OpportunityType.DEGREE)
    return Opportunity(
        id="web-" + hashlib.sha1(url.encode()).hexdigest()[:10],
        name=title[:120],
        provider=urlparse(url).netloc.removeprefix("www.") or "unknown",
        type=opp_type,
        url=url,
        country="Unknown",
        cost_or_funding="See provider site",
        eligibility_summary="See provider site",
        description=(result.get("content") or "")[:500],
        source="web",
        funded=opp_type in (OpportunityType.SCHOLARSHIP, OpportunityType.FELLOWSHIP),
    )


@tool
def search_web(query: str) -> list[dict]:
    """Search the live web for current scholarships, programs and courses. Use for fresh
    deadlines or options the local catalog may not cover.

    Args:
        query: A focused search query, e.g. "fully funded AI master's scholarships Canada".
    """
    from tavily import TavilyClient

    client = TavilyClient(api_key=get_settings().tavily_api_key)
    response = client.search(query=query, max_results=5)
    return [result_to_opportunity(r).model_dump(mode="json") for r in response.get("results", [])]

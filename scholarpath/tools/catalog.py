"""Local opportunity catalog search (data/opportunities.json)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

from scholarpath.config import get_settings
from scholarpath.schemas import Opportunity
from scholarpath.text import keywords

MAX_RESULTS = 10


@lru_cache(maxsize=4)
def load_catalog(path: Path) -> tuple[Opportunity, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(Opportunity(**entry, source="catalog") for entry in data["opportunities"])


def _entry_keywords(opp: Opportunity) -> set[str]:
    return keywords(opp.name, opp.provider, opp.description, opp.type.value, " ".join(opp.fields))


def search_catalog(
    query: str,
    opportunity_type: str | None = None,
    country: str | None = None,
    max_cost_usd: int | None = None,
    online_only: bool = False,
    path: Path | None = None,
) -> list[Opportunity]:
    """Keyword search with optional hard filters. Pure function, used by the tool and tests."""
    query_words = keywords(query)
    results: list[tuple[int, Opportunity]] = []
    for opp in load_catalog(path or get_settings().catalog_path):
        if opportunity_type and opp.type.value != opportunity_type.lower():
            continue
        if country and opp.country.lower() != country.lower() and opp.mode != "online":
            continue
        if max_cost_usd is not None and not opp.funded and (opp.cost_usd or 0) > max_cost_usd:
            continue
        if online_only and opp.mode != "online":
            continue
        overlap = len(query_words & _entry_keywords(opp))
        if overlap:
            results.append((overlap, opp))
    results.sort(key=lambda pair: pair[0], reverse=True)
    return [opp for _, opp in results[:MAX_RESULTS]]


@tool
def search_local_catalog(
    query: str,
    opportunity_type: str | None = None,
    country: str | None = None,
    max_cost_usd: int | None = None,
    online_only: bool = False,
) -> list[dict]:
    """Search ScholarPath's curated catalog of degrees, scholarships, fellowships, bootcamps
    and courses.

    Args:
        query: Keywords such as a field, goal or topic, e.g. "machine learning master's".
        opportunity_type: Optional filter: degree, scholarship, fellowship, bootcamp or course.
        country: Optional country filter. Online opportunities always pass this filter.
        max_cost_usd: Optional maximum cost. Funded opportunities always pass this filter.
        online_only: Only return online opportunities.
    """
    hits = search_catalog(query, opportunity_type, country, max_cost_usd, online_only)
    return [opp.model_dump(mode="json") for opp in hits]

"""Tools available to the Searcher agent. Add a new tool module here and list it in default_tools."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from scholarpath.config import Settings, get_settings
from scholarpath.tools.catalog import search_local_catalog
from scholarpath.tools.web_search import search_web


def default_tools(settings: Settings | None = None) -> list[BaseTool]:
    settings = settings or get_settings()
    tools: list[BaseTool] = [search_local_catalog]
    if settings.tavily_api_key:
        tools.append(search_web)
    return tools


__all__ = ["default_tools", "search_local_catalog", "search_web"]

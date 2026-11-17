"""The three ScholarPath agents. Each module exposes a `make_<agent>(llm, ...)` node factory."""

from scholarpath.agents.reader import make_reader
from scholarpath.agents.reporter import make_reporter
from scholarpath.agents.searcher import make_searcher

__all__ = ["make_reader", "make_searcher", "make_reporter"]

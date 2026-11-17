"""Settings and tunables. Values come from environment variables or a local .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Graph and agent tunables.
CONFIDENCE_THRESHOLD = 0.5  # below this the Reader asks for clarification (interactive mode)
MAX_CLARIFICATION_ROUNDS = 1
MAX_TOOL_CALLS = 6  # per Searcher run
MAX_SEARCH_ATTEMPTS = 2  # first search + one broadened retry
TOP_K = 8
STRONG_MATCH_SCORE = 50
MIN_STRONG_MATCHES = 3


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, existing env vars win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    anthropic_model: str
    tavily_api_key: str | None
    fake_llm: bool
    catalog_path: Path
    reports_dir: Path


def get_settings() -> Settings:
    _load_dotenv(PROJECT_ROOT / ".env")
    env = os.environ
    return Settings(
        anthropic_api_key=env.get("ANTHROPIC_API_KEY") or None,
        anthropic_model=env.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        tavily_api_key=env.get("TAVILY_API_KEY") or None,
        fake_llm=env.get("SCHOLARPATH_FAKE_LLM", "0").lower() in {"1", "true", "yes"},
        catalog_path=Path(env.get("SCHOLARPATH_CATALOG", PROJECT_ROOT / "data" / "opportunities.json")),
        reports_dir=Path(env.get("SCHOLARPATH_REPORTS_DIR", PROJECT_ROOT / "reports")),
    )

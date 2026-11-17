"""LLM factory. Agents receive the model by injection, so tests can pass any stand-in."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from scholarpath.config import Settings, get_settings


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    if settings.fake_llm:
        from scholarpath.offline_llm import OfflineChatModel

        return OfflineChatModel()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env, or set SCHOLARPATH_FAKE_LLM=1 "
            "to run with the offline model."
        )
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )

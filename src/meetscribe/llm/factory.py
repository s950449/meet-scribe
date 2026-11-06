"""Choose the live LLM client or the offline fallback."""

from __future__ import annotations

from collections.abc import Mapping

from meetscribe.config import AppConfig
from meetscribe.llm.budget import content_budget
from meetscribe.llm.client import OpenAIChatClient
from meetscribe.llm.polisher import IdentityPolisher, OpenAIPolisher
from meetscribe.llm.profiles import select_active_llm
from meetscribe.llm.prompts import POLISH_SYSTEM, SUMMARY_MERGE_SYSTEM, SUMMARY_SYSTEM
from meetscribe.llm.summarizer import OpenAISummarizer


def build_polisher(
    config: AppConfig,
    environ: Mapping[str, str] | None = None,
) -> OpenAIPolisher | IdentityPolisher:
    if not config.llm.polish:
        return IdentityPolisher(reason="Polishing is disabled in config")
    return OpenAIPolisher(_client(config, environ), token_budget=_budget(config, POLISH_SYSTEM))


def build_summarizer(
    config: AppConfig,
    environ: Mapping[str, str] | None = None,
) -> OpenAISummarizer:
    return OpenAISummarizer(
        _client(config, environ),
        token_budget=_budget(config, SUMMARY_SYSTEM, SUMMARY_MERGE_SYSTEM),
    )


def _client(config: AppConfig, environ: Mapping[str, str] | None) -> OpenAIChatClient:
    active = select_active_llm(config)
    sampling = active.sampling
    return OpenAIChatClient(
        base_url=config.llm.base_url,
        api_key=config.resolved_api_key(environ),
        model=active.model,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        min_p=sampling.min_p,
        presence_penalty=sampling.presence_penalty,
    )


def _budget(config: AppConfig, *prompts: str) -> int:
    return content_budget(config.llm.context_tokens, config.llm.reserve_tokens, *prompts)

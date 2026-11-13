"""Build LLM clients, or bypass polishing when explicitly disabled."""

from __future__ import annotations

from collections.abc import Mapping

from meetscribe.config import AppConfig, is_local_provider
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
    api_key = config.resolved_api_key(environ)
    active = select_active_llm(config, api_key=api_key)
    settings = config.llm
    if not is_local_provider(settings.provider):
        # Hosted API: send only the sampling fields set in config; retry 429/5xx.
        return OpenAIChatClient(
            base_url=settings.base_url,
            api_key=api_key,
            model=active.model,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=None,
            min_p=None,
            presence_penalty=settings.presence_penalty,
            timeout=settings.timeout,
            local_extras=False,
            max_retries=2,
        )
    sampling = active.sampling
    return OpenAIChatClient(
        base_url=settings.base_url,
        api_key=api_key,
        model=active.model,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        min_p=sampling.min_p,
        presence_penalty=sampling.presence_penalty,
        timeout=settings.timeout,
    )


def _budget(config: AppConfig, *prompts: str) -> int:
    return content_budget(config.llm.context_tokens, config.llm.reserve_tokens, *prompts)

"""Choose the live LLM client or the offline fallback."""

from __future__ import annotations

from collections.abc import Mapping

from meetscribe.config import AppConfig
from meetscribe.llm.client import OpenAIChatClient
from meetscribe.llm.polisher import IdentityPolisher, OpenAIPolisher
from meetscribe.llm.summarizer import ExtractiveSummarizer, OpenAISummarizer


def build_polisher(
    config: AppConfig,
    environ: Mapping[str, str] | None = None,
) -> OpenAIPolisher | IdentityPolisher:
    if not config.llm.polish:
        return IdentityPolisher(reason="Polishing is disabled in config")
    api_key = config.api_key(environ)
    if not api_key:
        return IdentityPolisher(
            reason=(
                f"No API key in {config.llm.api_key_env}. "
                "ASR text is copied through. Set the variable to polish via the OpenAI-compatible endpoint."
            )
        )
    client = OpenAIChatClient(base_url=config.llm.base_url, api_key=api_key, model=config.llm.model)
    return OpenAIPolisher(client)


def build_summarizer(
    config: AppConfig,
    environ: Mapping[str, str] | None = None,
) -> OpenAISummarizer | ExtractiveSummarizer:
    api_key = config.api_key(environ)
    if not api_key:
        return ExtractiveSummarizer()
    client = OpenAIChatClient(base_url=config.llm.base_url, api_key=api_key, model=config.llm.model)
    return OpenAISummarizer(client)

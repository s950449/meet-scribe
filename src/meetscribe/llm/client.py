"""OpenAI-compatible chat client and a small JSON extractor."""

from __future__ import annotations

import json
import re
from typing import Protocol

from meetscribe.errors import LlmError


class ChatClient(Protocol):
    def complete(self, *, system: str, user: str) -> str:
        """Return the assistant message text."""


class OpenAIChatClient:
    """Chat completions against any server that speaks the OpenAI protocol.

    ``base_url`` is the server root that already includes ``/v1``. The default
    is llama.cpp's llama-server. The same client talks to any other
    OpenAI-compatible server. llama-server ignores the model name in
    single-model mode and does not require a key; pass a dummy when the
    environment variable is unset. The key stays in memory.
    """

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, *, system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LlmError("The openai package is not installed. Reinstall meetscribe.") from exc
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=0,
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:
            raise LlmError(f"LLM request to {self.base_url} failed: {exc}") from exc
        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LlmError("LLM returned an empty message")
        return content


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, object]:
    """Parse a JSON object, including fenced blocks and leading prose."""

    cleaned = text.strip()
    cleaned = _FENCE.sub("", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LlmError("LLM response did not contain a JSON object") from None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LlmError(f"LLM response was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LlmError("LLM JSON response must be an object")
    return parsed

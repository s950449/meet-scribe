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

    ``base_url`` is the server root that already includes ``/v1``. Apple
    Silicon defaults to oMLX at ``http://127.0.0.1:8000/v1``. Other machines
    default to llama.cpp's llama-server at ``http://127.0.0.1:8080/v1``.
    llama-server does not check a key. oMLX on a loopback bind can run
    without one; if you set an oMLX API key, put it in the environment
    variable. The client always sends a bearer, using the dummy value
    ``llama-server`` when that variable is unset. The key stays in memory.
    Thinking is turned off with ``chat_template_kwargs.enable_thinking``,
    and any leftover reasoning text is stripped.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        min_p: float = 0.0,
        presence_penalty: float | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty

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
            request: dict[str, object] = {
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if self.presence_penalty is not None:
                request["presence_penalty"] = self.presence_penalty
            extra: dict[str, object] = {
                "top_k": self.top_k,
                "min_p": self.min_p,
                # oMLX and llama-server both honor this nested flag.
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request["extra_body"] = extra
            response = client.chat.completions.create(**request)
        except Exception as exc:
            raise LlmError(f"LLM request to {self.base_url} failed: {exc}") from exc
        if not response.choices:
            raise LlmError("LLM returned no choices")
        return visible_completion(response.choices[0].message)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_GEMMA_THOUGHT = re.compile(r"<\|channel>thought\s*.*?<channel\|>\s*", re.DOTALL)
_THINK_TOKEN = re.compile(r"<\|think\|>\s*")


def strip_thinking(text: str) -> str:
    """Drop reasoning channels so JSON parsing sees only the answer.

    llama-server may leave Qwen ``<think>`` blocks or Gemma 4
    ``<|channel>thought`` blocks in ``message.content`` when reasoning is
    enabled. ``reasoning_content`` is ignored by the caller.
    """

    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _GEMMA_THOUGHT.sub("", cleaned)
    cleaned = _THINK_TOKEN.sub("", cleaned)
    return cleaned.strip()


def visible_completion(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        content = "".join(parts)
    cleaned = strip_thinking(str(content or ""))
    if not cleaned:
        raise LlmError(
            "LLM returned no answer after stripping thinking output. "
            "Start llama-server with --jinja --reasoning off "
            "and --chat-template-kwargs '{\"enable_thinking\":false}'."
        )
    return cleaned


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

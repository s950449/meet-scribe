"""Rough token estimates so long meetings fit in llama-server's context."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def estimate_tokens(text: str) -> int:
    """Estimate tokens for mixed Traditional Chinese and English.

    CJK characters count as one token. Other characters count as about four
    per token. This is deliberately high for Qwen-style models so a chunk
    stays inside ``-c``.
    """

    if not text:
        return 0
    cjk = 0
    other = 0
    for char in text:
        code = ord(char)
        if (
            0x3400 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or 0x3040 <= code <= 0x30FF
        ):
            cjk += 1
        else:
            other += 1
    return cjk + (other + 3) // 4


def content_budget(context_tokens: int, reserve_tokens: int, *prompts: str) -> int:
    """Tokens left for a user message after the system prompt and the reply."""

    prompt_cost = sum(estimate_tokens(prompt) for prompt in prompts)
    return max(256, context_tokens - reserve_tokens - prompt_cost)


def chunk_items(items: Sequence[T], render: Callable[[T], str], budget: int) -> list[list[T]]:
    """Split items so each rendered chunk stays within ``budget`` tokens."""

    if budget < 1:
        raise ValueError("token budget must be positive")
    chunks: list[list[T]] = []
    current: list[T] = []
    used = 0
    for item in items:
        cost = max(1, estimate_tokens(render(item)))
        if current and used + cost > budget:
            chunks.append(current)
            current = []
            used = 0
        current.append(item)
        used += cost
    if current:
        chunks.append(current)
    return chunks

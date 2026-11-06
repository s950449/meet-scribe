"""Rewrite ASR text into Traditional Chinese without changing the facts."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace

from meetscribe.errors import LlmError
from meetscribe.llm.budget import chunk_items
from meetscribe.llm.client import ChatClient, extract_json_object
from meetscribe.llm.prompts import POLISH_SYSTEM
from meetscribe.types import AttributedUtterance, ProgressEvent


class OpenAIPolisher:
    name = "openai"

    def __init__(self, client: ChatClient, *, token_budget: int = 6000) -> None:
        self.client = client
        self.token_budget = token_budget

    def polish(
        self,
        utterances: Sequence[AttributedUtterance],
        *,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> tuple[AttributedUtterance, ...]:
        del on_event
        items = list(utterances)
        polished = list(items)
        chunks = chunk_items(items, _render_utterance, self.token_budget)
        offset = 0
        for chunk in chunks:
            raw = self.client.complete(system=POLISH_SYSTEM, user=_payload(chunk))
            data = extract_json_object(raw)
            rows = data.get("utterances", [])
            if not isinstance(rows, list):
                raise LlmError("Polish response is missing an utterances array")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    index = int(row["index"])
                except (KeyError, TypeError, ValueError):
                    continue
                text = str(row.get("text", "")).strip()
                absolute = offset + index
                if text and 0 <= absolute < len(polished):
                    polished[absolute] = replace(polished[absolute], polished_text=text)
            offset += len(chunk)
        return tuple(
            item if item.polished_text is not None else replace(item, polished_text=item.text)
            for item in polished
        )


class IdentityPolisher:
    """Copy ASR text through when polishing is turned off."""

    name = "identity"

    def __init__(self, *, reason: str) -> None:
        self.reason = reason

    def polish(
        self,
        utterances: Sequence[AttributedUtterance],
        *,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> tuple[AttributedUtterance, ...]:
        del on_event
        return tuple(replace(item, polished_text=item.text) for item in utterances)


def _render_utterance(utterance: AttributedUtterance) -> str:
    """Estimate one JSON row, which is what the request actually sends."""

    return json.dumps(
        {
            "index": 0,
            "speaker": utterance.speaker_id,
            "start": round(utterance.start, 2),
            "text": utterance.text,
        },
        ensure_ascii=False,
    )


def _payload(utterances: Sequence[AttributedUtterance]) -> str:
    rows = [
        {
            "index": index,
            "speaker": utterance.speaker_id,
            "start": round(utterance.start, 2),
            "text": utterance.text,
        }
        for index, utterance in enumerate(utterances)
    ]
    return json.dumps({"utterances": rows}, ensure_ascii=False)

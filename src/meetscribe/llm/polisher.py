"""Rewrite ASR text into Traditional Chinese without changing the facts."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace

from meetscribe.errors import LlmError
from meetscribe.llm.client import ChatClient, extract_json_object
from meetscribe.llm.prompts import POLISH_SYSTEM
from meetscribe.types import AttributedUtterance, ProgressEvent

BATCH_SIZE = 40


class OpenAIPolisher:
    name = "openai"

    def __init__(self, client: ChatClient) -> None:
        self.client = client

    def polish(
        self,
        utterances: Sequence[AttributedUtterance],
        *,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> tuple[AttributedUtterance, ...]:
        del on_event
        items = list(utterances)
        polished = list(items)
        for offset in range(0, len(items), BATCH_SIZE):
            chunk = items[offset : offset + BATCH_SIZE]
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
        return tuple(
            item if item.polished_text is not None else replace(item, polished_text=item.text)
            for item in polished
        )


class IdentityPolisher:
    """Copy ASR text through when polishing is off or no API key is configured."""

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

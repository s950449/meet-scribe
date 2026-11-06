"""Meeting summary through an OpenAI-compatible endpoint, with an offline fallback."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from meetscribe.errors import LlmError
from meetscribe.llm.client import ChatClient, extract_json_object
from meetscribe.llm.prompts import SUMMARY_SYSTEM
from meetscribe.timestamps import format_timestamp
from meetscribe.types import ActionItem, AttributedUtterance, MeetingSummary, ProgressEvent


class OpenAISummarizer:
    name = "openai"

    def __init__(self, client: ChatClient) -> None:
        self.client = client

    def summarize(
        self,
        utterances: Sequence[AttributedUtterance],
        *,
        title: str,
        speaker_map: dict[str, str] | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> MeetingSummary:
        del on_event
        user = json.dumps(
            {
                "title": title,
                "transcript": _transcript(utterances, speaker_map or {}),
            },
            ensure_ascii=False,
        )
        data = extract_json_object(self.client.complete(system=SUMMARY_SYSTEM, user=user))
        return _summary_from_json(data)


class ExtractiveSummarizer:
    """Offline summary used when no API key is configured.

    The overview is marked so a reader can see that no LLM was called.
    """

    name = "extractive"

    def summarize(
        self,
        utterances: Sequence[AttributedUtterance],
        *,
        title: str,
        speaker_map: dict[str, str] | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> MeetingSummary:
        del title, speaker_map, on_event
        texts = [utterance.display_text.strip() for utterance in utterances if utterance.display_text.strip()]
        if not texts:
            overview = "（未呼叫 LLM，逐字稿是空的。）"
            points: tuple[str, ...] = ()
        else:
            overview = "（未呼叫 LLM，以下為抽取式摘要。）" + texts[0][:400]
            points = tuple(texts[:5])
        return MeetingSummary(overview=overview, key_points=points, decisions=(), action_items=())


def _transcript(utterances: Sequence[AttributedUtterance], speaker_map: dict[str, str]) -> str:
    lines: list[str] = []
    for utterance in utterances:
        label = speaker_map.get(utterance.speaker_id, utterance.speaker_id)
        lines.append(f"[{format_timestamp(utterance.start)}] {label}: {utterance.display_text}")
    return "\n".join(lines)


def _summary_from_json(data: dict[str, object]) -> MeetingSummary:
    overview = str(data.get("overview", "")).strip()
    if not overview:
        raise LlmError("LLM summary is missing overview")
    return MeetingSummary(
        overview=overview,
        key_points=_str_tuple(data.get("key_points")),
        decisions=_str_tuple(data.get("decisions")),
        action_items=_action_items(data.get("action_items")),
    )


def _str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _action_items(value: object) -> tuple[ActionItem, ...]:
    if not isinstance(value, list):
        return ()
    items: list[ActionItem] = []
    for row in value:
        if isinstance(row, str) and row.strip():
            items.append(ActionItem(task=row.strip()))
            continue
        if not isinstance(row, dict):
            continue
        task = str(row.get("task", "")).strip()
        if not task:
            continue
        owner = _optional_str(row.get("owner"))
        due = _optional_str(row.get("due"))
        items.append(ActionItem(task=task, owner=owner, due=due))
    return tuple(items)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    return text

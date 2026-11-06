"""JSON sidecar for a meeting. ``summarize`` reads this and rewrites the Markdown."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from meetscribe.errors import MeetscribeError
from meetscribe.types import (
    ActionItem,
    AttributedUtterance,
    MeetingDocument,
    MeetingSummary,
    WordTimestamp,
)

SCHEMA_VERSION = 1


def dump_document(document: MeetingDocument, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = document_to_dict(document)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_document(path: Path) -> MeetingDocument:
    if not path.is_file():
        raise MeetscribeError(f"Meeting JSON not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MeetscribeError(f"Meeting JSON is invalid: {path}") from exc
    if not isinstance(data, dict):
        raise MeetscribeError(f"Meeting JSON must be an object: {path}")
    return document_from_dict(data)


def document_to_dict(document: MeetingDocument) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "title": document.title,
        "started_at": document.started_at.isoformat(timespec="seconds"),
        "summary": {
            "overview": document.summary.overview,
            "key_points": list(document.summary.key_points),
            "decisions": list(document.summary.decisions),
            "action_items": [
                {"task": item.task, "owner": item.owner, "due": item.due}
                for item in document.summary.action_items
            ],
        },
        "utterances": [
            {
                "speaker_id": utterance.speaker_id,
                "text": utterance.text,
                "polished_text": utterance.polished_text,
                "start": round(utterance.start, 3),
                "end": round(utterance.end, 3),
                "track_id": utterance.track_id,
                "overlap": utterance.overlap,
                "words": [
                    {"text": word.text, "start": round(word.start, 3), "end": round(word.end, 3)}
                    for word in utterance.words
                ],
            }
            for utterance in document.utterances
        ],
        "speaker_map": dict(document.speaker_map),
    }


def document_from_dict(data: dict[str, object]) -> MeetingDocument:
    summary_raw = data.get("summary") or {}
    if not isinstance(summary_raw, dict):
        raise MeetscribeError("Meeting JSON summary must be an object")
    utterances_raw = data.get("utterances") or []
    if not isinstance(utterances_raw, list):
        raise MeetscribeError("Meeting JSON utterances must be a list")
    speaker_map = data.get("speaker_map") or {}
    if not isinstance(speaker_map, dict):
        raise MeetscribeError("Meeting JSON speaker_map must be an object")
    started = data.get("started_at")
    if not isinstance(started, str):
        raise MeetscribeError("Meeting JSON is missing started_at")
    return MeetingDocument(
        title=str(data.get("title") or "會議"),
        started_at=datetime.fromisoformat(started),
        summary=_summary(summary_raw),
        utterances=tuple(_utterance(item) for item in utterances_raw if isinstance(item, dict)),
        speaker_map={str(key): str(value) for key, value in speaker_map.items()},
    )


def _summary(data: dict[str, object]) -> MeetingSummary:
    points = data.get("key_points") or []
    decisions = data.get("decisions") or []
    actions = data.get("action_items") or []
    return MeetingSummary(
        overview=str(data.get("overview") or ""),
        key_points=tuple(str(item) for item in points) if isinstance(points, list) else (),
        decisions=tuple(str(item) for item in decisions) if isinstance(decisions, list) else (),
        action_items=tuple(_action(item) for item in actions if isinstance(item, dict))
        if isinstance(actions, list)
        else (),
    )


def _action(data: dict[str, object]) -> ActionItem:
    owner = data.get("owner")
    due = data.get("due")
    return ActionItem(
        task=str(data.get("task") or ""),
        owner=str(owner) if owner else None,
        due=str(due) if due else None,
    )


def _utterance(data: dict[str, object]) -> AttributedUtterance:
    words_raw = data.get("words") or []
    words: list[WordTimestamp] = []
    if isinstance(words_raw, list):
        for word in words_raw:
            if not isinstance(word, dict):
                continue
            words.append(
                WordTimestamp(
                    text=str(word.get("text") or ""),
                    start=float(word.get("start") or 0),
                    end=float(word.get("end") or 0),
                )
            )
    polished = data.get("polished_text")
    return AttributedUtterance(
        speaker_id=str(data.get("speaker_id") or "UNKNOWN"),
        text=str(data.get("text") or ""),
        start=float(data.get("start") or 0),
        end=float(data.get("end") or 0),
        track_id=str(data.get("track_id") or "mic"),
        overlap=bool(data.get("overlap") or False),
        polished_text=str(polished) if isinstance(polished, str) else None,
        words=tuple(words),
    )

"""Versioned, atomic stage checkpoints. Fingerprints contain no configuration values."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from meetscribe import types
from meetscribe.errors import MeetscribeError

T = TypeVar("T")
_CLASSES = {
    name: getattr(types, name)
    for name in (
        "TranscriptSegment",
        "WordTimestamp",
        "SpeakerTurn",
        "AttributedUtterance",
        "MeetingSummary",
        "ActionItem",
    )
}


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(_encode(value), sort_keys=True).encode()).hexdigest()


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _encode(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "_type": type(value).__name__,
            "fields": {field.name: _encode(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    if isinstance(value, dict):
        if "_type" in value:
            cls = _CLASSES[value["_type"]]
            raw_fields = value["fields"]
            if not isinstance(raw_fields, dict):
                raise ValueError("record fields must be an object")
            record = cls(**{key: _decode(item) for key, item in raw_fields.items()})
            _validate_record(record)
            return record
        return {key: _decode(item) for key, item in value.items()}
    return value


class Checkpoint:
    def __init__(self, directory: Path, *, resume: bool) -> None:
        self.path = directory / "checkpoint.json"
        self.entries: dict[str, Any] = {}
        directory.mkdir(parents=True, exist_ok=True)
        self._lock = (directory / ".lock").open("a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                self._lock.write(b"0")
                self._lock.flush()
                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.close()
            raise MeetscribeError(f"Checkpoint directory is already in use: {directory}") from exc
        try:
            self._load(resume)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._lock.close()

    def _load(self, resume: bool) -> None:
        if not resume:
            self._save()
            return
        if not self.path.is_file():
            raise MeetscribeError(f"No checkpoint found: {self.path}")
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data["version"] != 1:
                raise MeetscribeError(f"Unsupported checkpoint version: {self.path}")
            self.entries = data["entries"]
            if not isinstance(self.entries, dict):
                raise ValueError("entries must be an object")
            for name, entry in self.entries.items():
                if not isinstance(entry["key"], str):
                    raise ValueError("invalid fingerprint")
                _validate_stage(name, _decode(entry["value"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise MeetscribeError(f"Invalid checkpoint {self.path}: {exc}") from exc

    def stage(self, name: str, key: str, compute: Callable[[], T]) -> T:
        entry = self.entries.get(name)
        if entry is not None and entry["key"] == key:
            return _decode(entry["value"])
        value = compute()
        self.entries[name] = {"key": key, "value": _encode(value)}
        self._save()
        return value

    def _save(self) -> None:
        atomic_text(
            self.path,
            json.dumps({"version": 1, "entries": self.entries}, ensure_ascii=False, indent=2)
            + "\n",
        )


def _validate_stage(name: str, value: Any) -> None:
    if name == "summary":
        if not isinstance(value, types.MeetingSummary):
            raise ValueError("summary must be a MeetingSummary")
        if not isinstance(value.overview, str):
            raise ValueError("summary overview must be text")
        return
    cls = (
        types.TranscriptSegment
        if name.startswith("asr:")
        else types.SpeakerTurn
        if name == "diarize"
        else types.AttributedUtterance
        if name == "polish"
        else None
    )
    if cls is None or not isinstance(value, tuple):
        raise ValueError(f"invalid stage {name}")
    for item in value:
        if not isinstance(item, cls):
            raise ValueError(f"invalid {name} item")
        if not isinstance(item.start, int | float) or not isinstance(item.end, int | float):
            raise ValueError(f"invalid {name} timestamps")
        if hasattr(item, "words") and not all(
            isinstance(word, types.WordTimestamp) for word in item.words
        ):
            raise ValueError(f"invalid {name} word timestamps")


def _validate_record(record: Any) -> None:
    for field in fields(record):
        name, value = field.name, getattr(record, field.name)
        if name in {"start", "end"}:
            valid = type(value) in {int, float} and math.isfinite(value) and value >= 0
        elif name in {"words", "action_items", "key_points", "decisions"}:
            cls = (
                types.WordTimestamp
                if name == "words"
                else types.ActionItem
                if name == "action_items"
                else str
            )
            valid = isinstance(value, tuple) and all(isinstance(item, cls) for item in value)
        elif name == "overlap":
            valid = isinstance(value, bool)
        elif name in {"polished_text", "language", "owner", "due"} or (
            name == "track_id" and isinstance(record, types.SpeakerTurn)
        ):
            valid = value is None or isinstance(value, str)
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(f"invalid {type(record).__name__}.{name}")
    if hasattr(record, "start") and record.end < record.start:
        raise ValueError("end precedes start")

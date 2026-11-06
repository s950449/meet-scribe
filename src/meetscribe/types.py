"""Shared values passed between capture, ASR, diarization, and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One ASR segment on a single capture track.

    Times are seconds on the capture timebase (t = 0 at the start of the take).
    """

    text: str
    start: float
    end: float
    track_id: str
    words: tuple[WordTimestamp, ...] = ()
    language: str | None = None


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """A diarization turn.

    ``track_id`` is set when the backend already knows which capture track the
    speaker belongs to (mic versus system). ``None`` means the turn may match
    any track. ``overlap`` is true when the diarizer believes another speaker
    is active during this turn.
    """

    speaker_id: str
    start: float
    end: float
    track_id: str | None = None
    overlap: bool = False


@dataclass(frozen=True, slots=True)
class AttributedUtterance:
    """An ASR segment after a speaker has been assigned."""

    speaker_id: str
    text: str
    start: float
    end: float
    track_id: str
    overlap: bool = False
    polished_text: str | None = None
    words: tuple[WordTimestamp, ...] = ()

    @property
    def display_text(self) -> str:
        if self.polished_text is not None:
            return self.polished_text
        return self.text


@dataclass(frozen=True, slots=True)
class AudioTrack:
    track_id: str
    path: Path
    sample_rate: int
    channels: int
    duration_s: float


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """Mic and system audio that share one timebase."""

    tracks: tuple[AudioTrack, ...]
    started_at: datetime
    timebase: str = "capture_start"


@dataclass(frozen=True, slots=True)
class ActionItem:
    task: str
    owner: str | None = None
    due: str | None = None


@dataclass(frozen=True, slots=True)
class MeetingSummary:
    overview: str
    key_points: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    action_items: tuple[ActionItem, ...] = ()


@dataclass
class MeetingDocument:
    title: str
    started_at: datetime
    summary: MeetingSummary
    utterances: tuple[AttributedUtterance, ...]
    speaker_map: dict[str, str] = field(default_factory=dict)

    def label_for(self, speaker_id: str) -> str:
        return self.speaker_map.get(speaker_id, speaker_id)

    def participants(self) -> list[str]:
        labels: list[str] = []
        for utterance in self.utterances:
            label = self.label_for(utterance.speaker_id)
            if label not in labels:
                labels.append(label)
        return labels


@dataclass(frozen=True, slots=True)
class AsrCapabilities:
    streaming: bool
    word_timestamps: bool
    device: str


@dataclass(frozen=True, slots=True)
class DiarizationCapabilities:
    streaming: bool
    overlap: bool
    device: str
    requires_token: bool = False


@dataclass(frozen=True, slots=True)
class CaptureCapabilities:
    mic: bool
    system_audio: bool
    separate_tracks: bool
    device: str


class EventSink(Protocol):
    def emit(self, event: ProgressEvent) -> None:
        """Accept one progress event."""


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One progress record.

    Serialized as one JSON object per line when ``--json-events`` is set.
    Extra fields live under ``data`` so the top-level keys stay stable.
    """

    stage: str
    status: str | None = None
    message: str | None = None
    event: str = "stage"
    track_id: str | None = None
    progress: float | None = None
    data: tuple[tuple[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        body: dict[str, object] = {"event": self.event, "stage": self.stage}
        if self.status is not None:
            body["status"] = self.status
        if self.track_id is not None:
            body["track_id"] = self.track_id
        if self.message is not None:
            body["message"] = self.message
        if self.progress is not None:
            body["progress"] = self.progress
        if self.data:
            body["data"] = {key: value for key, value in self.data}
        return body

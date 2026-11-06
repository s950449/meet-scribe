"""Diarizers that need no model weights."""

from __future__ import annotations

from collections.abc import Callable

from meetscribe.types import (
    AudioTrack,
    DiarizationCapabilities,
    ProgressEvent,
    SpeakerTurn,
)

MIC_SPEAKER = "Me"
REMOTE_SPEAKER = "Remote"
SINGLE_SPEAKER = "SPEAKER_00"


class SingleSpeakerDiarizer:
    """Assign the whole timeline to one anonymous speaker.

    Use this for a single mixed recording until pyannote or sherpa-onnx is wired.
    """

    name = "single-speaker"

    def capabilities(self) -> DiarizationCapabilities:
        return DiarizationCapabilities(
            streaming=False,
            overlap=False,
            device="cpu",
            requires_token=False,
        )

    def diarize(
        self,
        tracks: list[AudioTrack],
        *,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> list[SpeakerTurn]:
        del on_event
        if not tracks:
            return []
        end = max(track.duration_s for track in tracks)
        return [
            SpeakerTurn(
                speaker_id=SINGLE_SPEAKER,
                start=0.0,
                end=end,
                track_id=None,
                overlap=False,
            )
        ]


class TrackDiarizer:
    """Label the mic track as ``Me`` and the system track as ``Remote``.

    Remote participants stay mixed together. Splitting them is the job of
    pyannote or sherpa-onnx, which can later run on ``system.wav`` only while
    this mic label is kept.
    """

    name = "track"

    def __init__(self, labels: dict[str, str] | None = None) -> None:
        self.labels = labels or {"mic": MIC_SPEAKER, "system": REMOTE_SPEAKER}

    def capabilities(self) -> DiarizationCapabilities:
        return DiarizationCapabilities(
            streaming=False,
            overlap=False,
            device="cpu",
            requires_token=False,
        )

    def diarize(
        self,
        tracks: list[AudioTrack],
        *,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> list[SpeakerTurn]:
        del on_event
        turns: list[SpeakerTurn] = []
        for track in tracks:
            speaker_id = self.labels.get(track.track_id, track.track_id)
            turns.append(
                SpeakerTurn(
                    speaker_id=speaker_id,
                    start=0.0,
                    end=track.duration_s,
                    track_id=track.track_id,
                    overlap=False,
                )
            )
        return turns

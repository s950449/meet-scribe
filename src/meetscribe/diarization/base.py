"""Diarization protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from meetscribe.types import AudioTrack, DiarizationCapabilities, ProgressEvent, SpeakerTurn


class DiarizationBackend(Protocol):
    name: str

    def capabilities(self) -> DiarizationCapabilities:
        """Streaming, overlap, device, and whether a token is required."""

    def diarize(
        self,
        tracks: list[AudioTrack],
        *,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> list[SpeakerTurn]:
        """Return speaker turns on the capture timebase."""

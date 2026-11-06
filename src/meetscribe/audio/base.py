"""Capture protocol and the shared resample step."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from meetscribe.audio.io import prepare_track
from meetscribe.events import stage_event
from meetscribe.types import AudioTrack, CaptureCapabilities, CaptureResult, ProgressEvent


class AudioCapture(Protocol):
    """Record or load mic and system audio on one timebase.

    Implementations write ``mic.wav`` and, when system audio exists,
    ``system.wav``. Timestamps elsewhere in the pipeline are seconds from the
    start of this take.
    """

    name: str

    def capabilities(self) -> CaptureCapabilities:
        """Declare tracks and the native audio device."""

    def record(
        self,
        output_dir: Path,
        *,
        duration_s: float | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> CaptureResult:
        """Capture until ``duration_s`` elapses, or load files for file input."""


def normalize_tracks(
    capture: CaptureResult,
    output_dir: Path,
    *,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> CaptureResult:
    """Resample every track to 16 kHz mono PCM."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if on_event is not None:
        on_event(stage_event("normalize", "start", message="Resampling tracks to 16 kHz mono"))
    tracks: list[AudioTrack] = []
    for track in capture.tracks:
        dest = output_dir / f"{track.track_id}.wav"
        prepared = prepare_track(track.path, dest, track.track_id)
        tracks.append(prepared)
        if on_event is not None:
            on_event(
                stage_event(
                    "normalize",
                    "track",
                    track_id=track.track_id,
                    message=f"Normalized {track.track_id} ({prepared.duration_s:.1f}s)",
                    duration_s=round(prepared.duration_s, 3),
                )
            )
    if on_event is not None:
        on_event(stage_event("normalize", "done", message="Audio is 16 kHz mono"))
    return CaptureResult(
        tracks=tuple(tracks),
        started_at=capture.started_at,
        timebase=capture.timebase,
    )

"""Optional chunked transcription while a recording is still running.

Segments emitted here are progress only. ``run --live`` still transcribes the
finished wavs so diarization and the Markdown note see the whole meeting.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from meetscribe.audio.io import TARGET_SAMPLE_RATE, resample_linear, write_wav_mono
from meetscribe.errors import MeetscribeError
from meetscribe.events import log_event, stage_event
from meetscribe.types import EventSink


def make_live_callback(
    asr: object,
    language: str | None,
    sink: EventSink,
    sample_rate: int,
) -> Callable[[str, float, np.ndarray], None]:
    state = {"failed": False}

    def on_chunk(track_id: str, start_s: float, samples: np.ndarray) -> None:
        if state["failed"] or samples.size == 0:
            return
        try:
            audio = resample_linear(np.asarray(samples, dtype=np.float32), sample_rate, TARGET_SAMPLE_RATE)
            with tempfile.TemporaryDirectory(prefix="meetscribe-live-") as tmp:
                path = Path(tmp) / "chunk.wav"
                write_wav_mono(path, audio, TARGET_SAMPLE_RATE)
                segments = asr.transcribe(  # type: ignore[attr-defined]
                    path,
                    track_id=track_id,
                    language=language,
                    on_event=sink.emit,
                )
            for segment in segments:
                absolute = start_s + segment.start
                sink.emit(
                    stage_event(
                        "live",
                        "segment",
                        track_id=track_id,
                        message=f"[{track_id} {absolute:.1f}] {segment.text}",
                        start=round(absolute, 3),
                        text=segment.text,
                    )
                )
        except MeetscribeError as exc:
            state["failed"] = True
            sink.emit(log_event("live", f"Live transcription stopped: {exc}"))

    return on_chunk

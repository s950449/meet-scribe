"""Shared start/stop loop used by the platform recorders."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from datetime import datetime

from meetscribe.audio.devices import AudioDevice
from meetscribe.audio.sync import ChunkCallback, SyncedWavSession
from meetscribe.errors import MeetscribeError
from meetscribe.events import log_event
from meetscribe.types import CaptureResult, ProgressEvent

StreamOpener = Callable[[AudioDevice, str, SyncedWavSession, int], object]


def run_synced_recording(
    output_dir,
    *,
    devices: list[tuple[str, AudioDevice]],
    opener: StreamOpener,
    sample_rate: int,
    duration_s: float | None,
    chunk_seconds: float,
    on_chunk: ChunkCallback | None,
    on_event: Callable[[ProgressEvent], None] | None,
) -> CaptureResult:
    session = SyncedWavSession(
        output_dir=output_dir,
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds if on_chunk is not None else 0.0,
        on_chunk=on_chunk,
    )
    streams: list[object] = []
    body_error: Exception | None = None
    close_error: MeetscribeError | None = None
    try:
        for track_id, device in devices:
            stream = opener(device, track_id, session, sample_rate)
            start = getattr(stream, "start", None)
            if callable(start):
                start()
            streams.append(stream)
        if on_event is not None:
            names = ", ".join(f"{track}={device.name}" for track, device in devices)
            on_event(log_event("capture", f"Recording {names}. Ctrl+C stops."))
        _wait(duration_s)
    except Exception as exc:
        body_error = exc
    for stream in streams:
        close = getattr(stream, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except MeetscribeError as exc:
            close_error = close_error or exc
        except Exception:
            continue
    result = session.finish(datetime.now().astimezone())
    if body_error is not None:
        raise body_error
    if close_error is not None:
        raise close_error
    if on_event is not None:
        for track_id, stats in session.stats().items():
            on_event(
                log_event(
                    "capture",
                    (
                        f"{track_id}: {stats.frames / sample_rate:.2f}s, "
                        f"silence {stats.silence_frames / sample_rate:.3f}s, "
                        f"dropped {stats.dropped_frames / sample_rate:.3f}s"
                    ),
                    track_id=track_id,
                )
            )
    return result


def _wait(duration_s: float | None) -> None:
    if duration_s is not None and duration_s <= 0:
        return
    stop = threading.Event()
    previous = signal.getsignal(signal.SIGINT)

    def _handle(signum: int, frame: object) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    try:
        if duration_s is None:
            while not stop.is_set():
                stop.wait(0.2)
        else:
            stop.wait(duration_s)
    except KeyboardInterrupt:
        return
    finally:
        signal.signal(signal.SIGINT, previous)

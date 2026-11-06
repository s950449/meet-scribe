"""Shared start/stop loop used by the platform recorders."""

from __future__ import annotations

import os
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from meetscribe.audio.control import RecordingControl
from meetscribe.audio.devices import AudioDevice
from meetscribe.audio.sync import ChunkCallback, SyncedWavSession
from meetscribe.errors import MeetscribeError
from meetscribe.events import log_event, stage_event
from meetscribe.types import CaptureResult, ProgressEvent

StreamOpener = Callable[[AudioDevice, str, SyncedWavSession, int], object]


def run_synced_recording(
    output_dir: Path,
    *,
    devices: list[tuple[str, AudioDevice]],
    opener: StreamOpener,
    sample_rate: int,
    duration_s: float | None,
    chunk_seconds: float,
    on_chunk: ChunkCallback | None,
    on_event: Callable[[ProgressEvent], None] | None,
    control: RecordingControl | None = None,
) -> CaptureResult:
    control = control or RecordingControl()
    started_at = datetime.now().astimezone()
    session = SyncedWavSession(
        output_dir=output_dir,
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds if on_chunk is not None else 0.0,
        on_chunk=on_chunk,
        control=control,
    )
    streams: list[object] = []
    body_error: BaseException | None = None
    close_error: MeetscribeError | None = None
    try:
        for track_id, device in devices:
            stream = opener(device, track_id, session, sample_rate)
            streams.append(stream)
            start = getattr(stream, "start", None)
            if callable(start):
                start()
        control.start_timer()
        session.reset_monitor()
        if on_event is not None:
            names = ", ".join(f"{track}={device.name}" for track, device in devices)
            on_event(log_event("capture", f"Recording {names}. Ctrl+C stops."))
        _wait(
            duration_s,
            control=control,
            session=session,
            track_ids=[track for track, _ in devices],
            on_event=on_event,
        )
    except BaseException as exc:
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
    result = session.finish(started_at)
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


@contextmanager
def _keyboard(enabled: bool) -> Iterator[Callable[[], str]]:
    """Restore terminal flags even on interruption; never consume redirected stdin."""
    if not enabled or not sys.stdin.isatty():
        yield lambda: ""
        return
    if os.name == "nt":
        import msvcrt

        console: Any = msvcrt

        def read_console() -> str:
            if not console.kbhit():
                return ""
            key: str = console.getwch()
            if key in {"\x00", "\xe0"}:
                # Extended keys include a scan-code character (e.g. down is P).
                console.getwch()
                return ""
            return key

        yield read_console
        return
    import select
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield (
            lambda: (
                os.read(descriptor, 1).decode("ascii", errors="ignore")
                if select.select([descriptor], [], [], 0)[0]
                else ""
            )
        )
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _wait(
    duration_s: float | None,
    *,
    control: RecordingControl | None = None,
    session: SyncedWavSession | None = None,
    track_ids: list[str] | None = None,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> None:
    control = control or RecordingControl()
    if duration_s is not None and duration_s <= 0:
        return
    previous: Any = None
    main_thread = threading.current_thread() is threading.main_thread()
    if main_thread:
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda signum, frame: control.stop())
    sink = getattr(on_event, "__self__", None)
    keyboard_enabled = main_thread and not getattr(sink, "json_events", False)
    wake = threading.Event()
    next_status = 0.0
    last_paused: bool | None = None
    try:
        with _keyboard(keyboard_enabled) as read_key:
            if keyboard_enabled and sys.stdin.isatty() and on_event is not None:
                on_event(log_event("capture", "p: pause/resume, q: stop, Ctrl+C: stop"))
            while not control.stopped:
                key = read_key().lower()
                if key == "p":
                    control.toggle_pause()
                elif key == "q":
                    control.stop()
                    break
                elapsed = control.elapsed()
                if duration_s is not None and elapsed >= duration_s:
                    break
                paused = control.paused
                if on_event is not None and (elapsed >= next_status or paused != last_paused):
                    state = "paused" if paused else "recording"
                    on_event(
                        stage_event(
                            "capture",
                            state,
                            message=f"{state}: {elapsed:.1f}s recorded",
                            elapsed_s=round(elapsed, 1),
                        )
                    )
                    if not paused and session is not None:
                        for event in session.monitor(track_ids or [], elapsed=elapsed):
                            on_event(event)
                    next_status = elapsed + 1.0
                    last_paused = paused
                wake.wait(0.1)
    except KeyboardInterrupt:
        control.stop()
    finally:
        if main_thread:
            signal.signal(signal.SIGINT, previous)

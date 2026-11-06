"""Place mic and system buffers on one host-clock timeline.

Each buffer is stamped with ``time.monotonic()`` at its start. A device that
runs slow gets silence inserted. A device that runs fast has the overlapping
samples dropped. Both files are padded to the same length on close so the
pipeline's shared timebase holds.
"""

from __future__ import annotations

import json
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from meetscribe.audio.control import RecordingControl
from meetscribe.audio.io import resample_linear
from meetscribe.events import stage_event
from meetscribe.types import AudioTrack, CaptureResult, ProgressEvent

ChunkCallback = Callable[[str, float, np.ndarray], None]


@dataclass
class TrackStats:
    frames: int = 0
    silence_frames: int = 0
    dropped_frames: int = 0


@dataclass
class SyncedWavSession:
    output_dir: Path
    sample_rate: int = 48_000
    chunk_seconds: float = 0.0
    on_chunk: ChunkCallback | None = None
    control: RecordingControl | None = None
    _levels: dict[str, tuple[float, float, float]] = field(default_factory=dict, init=False)
    _warned: dict[str, str] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _t0: float | None = field(default=None, init=False)
    _stats: dict[str, TrackStats] = field(default_factory=dict, init=False)
    _handles: dict[str, wave.Wave_write] = field(default_factory=dict, init=False)
    _pending: dict[str, list[np.ndarray]] = field(default_factory=dict, init=False)
    _pending_count: dict[str, int] = field(default_factory=dict, init=False)
    _emitted: dict[str, int] = field(default_factory=dict, init=False)

    def write(self, track_id: str, host_time: float, samples: np.ndarray, source_rate: int) -> None:
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        if source_rate != self.sample_rate:
            mono = resample_linear(mono, source_rate, self.sample_rate)
        if mono.size == 0:
            return
        if self.control is not None:
            ranges = self.control.retained(host_time, host_time + mono.size / self.sample_rate)
            for left, right, adjusted in ranges:
                first = round((left - host_time) * self.sample_rate)
                last = round((right - host_time) * self.sample_rate)
                if last > first:
                    self._write_active(track_id, adjusted, mono[first:last])
            return
        self._write_active(track_id, host_time, mono)

    def _write_active(self, track_id: str, host_time: float, mono: np.ndarray) -> None:
        callbacks: list[tuple[str, float, np.ndarray]] = []
        with self._lock:
            peak = float(np.max(np.abs(mono)))
            active = self.control.elapsed() if self.control is not None else host_time
            previous = self._levels.get(track_id, (0.0, active, 0.0))
            self._levels[track_id] = (peak, active, active if peak >= 0.001 else previous[2])
            if self._t0 is None:
                self._t0 = host_time
            origin = self._t0
            stats = self._stats.setdefault(track_id, TrackStats())
            target = round((host_time - origin) * self.sample_rate)
            if target < 0:
                target = 0
            if target > stats.frames:
                gap = target - stats.frames
                self._emit_locked(track_id, np.zeros(gap, dtype=np.float32), callbacks)
                stats.silence_frames += gap
            elif target < stats.frames:
                drop = stats.frames - target
                stats.dropped_frames += min(drop, int(mono.size))
                if drop >= mono.size:
                    return
                mono = mono[drop:]
            self._emit_locked(track_id, mono, callbacks)
        for track, start, chunk in callbacks:
            if self.on_chunk is not None:
                try:
                    self.on_chunk(track, start, chunk)
                except Exception:
                    # Live transcription must not tear down the recording threads.
                    continue

    def reset_monitor(self) -> None:
        """Reset observations when the recording duration clock starts."""
        with self._lock:
            self._levels.clear()
            self._warned.clear()

    def monitor(self, track_ids: list[str], *, elapsed: float) -> list[ProgressEvent]:
        """Snapshot levels on the control thread, with one warning per incident."""
        events = []
        with self._lock:
            for track_id in track_ids:
                peak, last, loud = self._levels.get(track_id, (0.0, 0.0, 0.0))
                problem = (
                    "missing"
                    if elapsed - last >= 5.0
                    else ("silent" if elapsed - loud >= 5.0 else "")
                )
                description = "no audio buffers" if problem == "missing" else "audio is silent"
                if problem and self._warned.get(track_id) != problem:
                    events.append(
                        stage_event(
                            "capture",
                            "warning",
                            track_id=track_id,
                            message=f"{track_id}: {description} for 5s",
                            reason=problem,
                        )
                    )
                self._warned[track_id] = problem
                visible_peak = peak if elapsed - last < 1.0 else 0.0
                events.append(
                    stage_event(
                        "capture",
                        "level",
                        track_id=track_id,
                        message=f"{track_id}: level {visible_peak:.3f}",
                        peak=visible_peak,
                        elapsed_s=round(elapsed, 1),
                    )
                )
        return events

    def finish(self, started_at: datetime) -> CaptureResult:
        with self._lock:
            self._flush_pending_locked()
            target = max(stats.frames for stats in self._stats.values()) if self._stats else 0
            for track_id, stats in list(self._stats.items()):
                if stats.frames < target:
                    gap = target - stats.frames
                    self._write_locked(track_id, np.zeros(gap, dtype=np.float32))
                    stats.silence_frames += gap
            tracks = []
            for track_id, stats in self._stats.items():
                handle = self._handles.get(track_id)
                if handle is not None:
                    handle.close()
                path = self.output_dir / f"{track_id}.wav"
                duration = stats.frames / self.sample_rate if self.sample_rate else 0.0
                tracks.append(
                    AudioTrack(
                        track_id=track_id,
                        path=path,
                        sample_rate=self.sample_rate,
                        channels=1,
                        duration_s=duration,
                    )
                )
            self._write_manifest_locked()
        order = {"mic": 0, "system": 1}
        tracks.sort(key=lambda track: (order.get(track.track_id, 9), track.track_id))
        return CaptureResult(tracks=tuple(tracks), started_at=started_at)

    def stats(self) -> dict[str, TrackStats]:
        with self._lock:
            return {
                key: TrackStats(value.frames, value.silence_frames, value.dropped_frames)
                for key, value in self._stats.items()
            }

    def _emit_locked(
        self,
        track_id: str,
        samples: np.ndarray,
        callbacks: list[tuple[str, float, np.ndarray]],
    ) -> None:
        self._write_locked(track_id, samples)
        if self.on_chunk is None or self.chunk_seconds <= 0:
            return
        pending = self._pending.setdefault(track_id, [])
        pending.append(samples.copy())
        self._pending_count[track_id] = self._pending_count.get(track_id, 0) + int(samples.size)
        chunk_n = int(self.chunk_seconds * self.sample_rate)
        while self._pending_count.get(track_id, 0) >= chunk_n:
            chunk = self._take_locked(track_id, chunk_n)
            start = self._emitted.get(track_id, 0) / self.sample_rate
            self._emitted[track_id] = self._emitted.get(track_id, 0) + int(chunk.size)
            callbacks.append((track_id, start, chunk))

    def _flush_pending_locked(self) -> None:
        # Leftover samples shorter than a chunk are not emitted. The final
        # full-file transcription covers them.
        self._pending.clear()
        self._pending_count.clear()

    def _take_locked(self, track_id: str, count: int) -> np.ndarray:
        needed = count
        pieces: list[np.ndarray] = []
        pending = self._pending[track_id]
        while needed > 0 and pending:
            head = pending[0]
            if head.size <= needed:
                pieces.append(head)
                needed -= int(head.size)
                pending.pop(0)
            else:
                pieces.append(head[:needed])
                pending[0] = head[needed:]
                needed = 0
        self._pending_count[track_id] = sum(int(piece.size) for piece in pending)
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces)

    def _write_locked(self, track_id: str, samples: np.ndarray) -> None:
        handle = self._handles.get(track_id)
        if handle is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"{track_id}.wav"
            # Kept open until finish(); a context manager would close it immediately.
            handle = wave.open(str(path), "wb")  # noqa: SIM115
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            self._handles[track_id] = handle
            self._stats.setdefault(track_id, TrackStats())
        clipped = np.clip(samples, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype("<i2")
        handle.writeframes(pcm.tobytes())
        self._stats[track_id].frames += int(samples.size)

    def _write_manifest_locked(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "sample_rate": self.sample_rate,
            "tracks": {
                track_id: {
                    "frames": stats.frames,
                    "silence_frames": stats.silence_frames,
                    "dropped_frames": stats.dropped_frames,
                }
                for track_id, stats in self._stats.items()
            },
        }
        path = self.output_dir / "session.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

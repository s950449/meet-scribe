"""Thread-safe recording controls and a shared pause timeline."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class RecordingControl:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.started = clock()
        self._lock = threading.Lock()
        self._pauses: list[tuple[float, float | None]] = []
        self._stop = threading.Event()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    @property
    def paused(self) -> bool:
        with self._lock:
            return bool(self._pauses and self._pauses[-1][1] is None)

    def stop(self) -> None:
        self._stop.set()

    def toggle_pause(self) -> None:
        with self._lock:
            now = self.clock()
            if self._pauses and self._pauses[-1][1] is None:
                self._pauses[-1] = (self._pauses[-1][0], now)
            else:
                self._pauses.append((now, None))

    def start_timer(self) -> None:
        """Start the duration budget when all capture devices are ready.

        Pause intervals remain on the host timeline so delayed audio buffers
        retain their correct alignment. An injected control keeps its pause and
        stop state.
        """
        with self._lock:
            self.started = self.clock()

    def elapsed(self) -> float:
        now = self.clock()
        with self._lock:
            return max(
                0.0,
                now
                - self.started
                - sum(
                    max(0.0, (end if end is not None else now) - max(start, self.started))
                    for start, end in self._pauses
                ),
            )

    def retained(self, start: float, end: float) -> list[tuple[float, float, float]]:
        """Return retained host ranges and their pause-adjusted start times."""
        with self._lock:
            pauses = list(self._pauses)
        cursor = start
        removed = 0.0
        result = []
        for left, right in pauses:
            stop = float("inf") if right is None else right
            if left >= end:
                break
            if stop <= cursor:
                removed += stop - left
                continue
            if cursor < left:
                result.append((cursor, min(left, end), cursor - removed))
            cursor = max(cursor, stop)
            if right is None:
                return result
            removed += right - left
            if cursor >= end:
                return result
        if cursor < end:
            result.append((cursor, end, cursor - removed))
        return result

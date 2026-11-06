"""ASR protocol. Segments stay on the capture timebase."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from meetscribe.types import AsrCapabilities, ProgressEvent, TranscriptSegment


class AsrBackend(Protocol):
    name: str

    def capabilities(self) -> AsrCapabilities:
        """Streaming, word timestamps, and the compute device."""

    def transcribe(
        self,
        audio: Path,
        *,
        track_id: str,
        language: str | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> list[TranscriptSegment]:
        """Transcribe one 16 kHz mono track."""

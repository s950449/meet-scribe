"""Load existing audio files as mic and system tracks."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from meetscribe.audio.io import probe_wav
from meetscribe.errors import AudioError
from meetscribe.types import AudioTrack, CaptureCapabilities, CaptureResult, ProgressEvent


class FileInput:
    """Ingest already-recorded audio. This is the capture backend that runs today.

    Pass one file as the microphone track, and an optional second file as system
    audio. Both files are assumed to start together. A directory that already
    contains ``mic.wav`` and optional ``system.wav`` is accepted too.
    """

    name = "file"

    def __init__(
        self,
        mic: Path | None = None,
        system: Path | None = None,
        *,
        started_at: datetime | None = None,
    ) -> None:
        self.mic = Path(mic) if mic is not None else None
        self.system = Path(system) if system is not None else None
        self.started_at = started_at

    def capabilities(self) -> CaptureCapabilities:
        return CaptureCapabilities(
            mic=True,
            system_audio=True,
            separate_tracks=True,
            device="file",
        )

    def record(
        self,
        output_dir: Path,
        *,
        duration_s: float | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> CaptureResult:
        del output_dir, duration_s, on_event
        pairs = self._tracks()
        started = self.started_at or _mtime_or_now(pairs[0][1])
        tracks: list[AudioTrack] = []
        for track_id, path in pairs:
            sample_rate, channels, frames = _probe(path)
            duration = frames / sample_rate if sample_rate else 0.0
            tracks.append(
                AudioTrack(
                    track_id=track_id,
                    path=path,
                    sample_rate=sample_rate,
                    channels=channels,
                    duration_s=duration,
                )
            )
        return CaptureResult(tracks=tuple(tracks), started_at=started)

    def _tracks(self) -> list[tuple[str, Path]]:
        if self.mic is None:
            raise AudioError("File input needs a microphone audio path")
        mic, system = _resolve_session(self.mic, self.system)
        if not mic.is_file():
            raise AudioError(f"Audio file not found: {mic}")
        tracks = [("mic", mic)]
        if system is not None:
            if not system.is_file():
                raise AudioError(f"System audio file not found: {system}")
            tracks.append(("system", system))
        return tracks


def _resolve_session(source: Path, system: Path | None) -> tuple[Path, Path | None]:
    if source.is_dir():
        mic = source / "mic.wav"
        if not mic.is_file():
            raise AudioError(f"No mic.wav in session directory {source}")
        session_system = source / "system.wav"
        if system is not None:
            return mic, system
        if session_system.is_file():
            return mic, session_system
        return mic, None
    return source, system


def _probe(path: Path) -> tuple[int, int, int]:
    if path.suffix.lower() == ".wav":
        try:
            return probe_wav(path)
        except AudioError:
            return 0, 0, 0
    return 0, 0, 0


def _mtime_or_now(path: Path) -> datetime:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    except OSError:
        return datetime.now().astimezone()

"""Decode formats unsupported by the native readers using an optional ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from meetscribe.errors import AudioError


def decode_audio(source: Path, destination: Path) -> None:
    """Write a 16 kHz mono PCM WAV, without changing the source file."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise AudioError(
            f"Cannot decode {source.name}. Install ffmpeg and put it on PATH "
            "to import m4a, mp3, and other compressed audio, "
            "or install meetscribe[audio] for formats supported by libsndfile."
        )
    command = [
        executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(source.resolve()),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(destination),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    except OSError as exc:
        raise AudioError(f"Cannot run ffmpeg to decode {source.name}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip()[-2000:] or "No audio could be decoded."
        raise AudioError(
            f"ffmpeg could not decode {source.name}. Check that the file is valid "
            f"and contains an audio stream. {detail}"
        )
    if not destination.is_file():
        raise AudioError(f"ffmpeg produced no audio for {source.name}")

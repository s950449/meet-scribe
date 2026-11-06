from pathlib import Path

import numpy as np
import pytest

from meetscribe.audio.file_input import FileInput
from meetscribe.audio.io import prepare_track, read_audio
from meetscribe.errors import AudioError


def _write_pcm(
    path: Path,
    *,
    rate: int,
    channels: int,
    seconds: float,
    sample_width: int = 2,
) -> None:
    count = int(rate * seconds)
    time = np.arange(count) / rate
    mono = 0.25 * np.sin(2 * np.pi * 440 * time)
    samples = mono if channels == 1 else np.column_stack([mono, mono * 0.5]).reshape(-1)
    if sample_width == 2:
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
        raw = pcm.tobytes()
    elif sample_width == 3:
        scaled = (np.clip(samples, -1, 1) * 8388607).astype(np.int32)
        raw = b"".join(int(value).to_bytes(3, "little", signed=True) for value in scaled)
    else:
        raise AssertionError(sample_width)
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(rate)
        handle.writeframes(raw)


def test_resample_stereo_to_16k_mono(tmp_path: Path) -> None:
    source = tmp_path / "stereo.wav"
    _write_pcm(source, rate=8000, channels=2, seconds=1.0)
    track = prepare_track(source, tmp_path / "mic.wav", "mic")
    samples, rate = read_audio(track.path)
    assert track.sample_rate == 16000
    assert track.channels == 1
    assert rate == 16000
    assert samples.ndim == 1
    assert abs(samples.size - 16000) <= 1
    assert abs(track.duration_s - 1.0) < 0.01


def test_24_bit_wav(tmp_path: Path) -> None:
    source = tmp_path / "wide.wav"
    _write_pcm(source, rate=16000, channels=1, seconds=0.25, sample_width=3)
    samples, rate = read_audio(source)
    assert rate == 16000
    assert samples.size == 4000
    assert float(np.max(np.abs(samples))) < 1.0


def test_file_input_session_directory(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    _write_pcm(session / "mic.wav", rate=16000, channels=1, seconds=0.5)
    _write_pcm(session / "system.wav", rate=16000, channels=1, seconds=0.5)
    capture = FileInput(session).record(tmp_path)
    assert [track.track_id for track in capture.tracks] == ["mic", "system"]


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioError):
        FileInput(tmp_path / "missing.wav").record(tmp_path)

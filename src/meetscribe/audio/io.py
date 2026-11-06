"""WAV reading, linear resample, and 16 kHz mono writing.

WAV PCM is handled with the standard library. Other containers are read with
``soundfile`` when that extra is installed. Compressed meeting exports (m4a,
mp3) need ffmpeg first.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from meetscribe.errors import AudioError
from meetscribe.types import AudioTrack

TARGET_SAMPLE_RATE = 16_000


def probe_wav(path: Path) -> tuple[int, int, int]:
    """Return ``(sample_rate, channels, nframes)`` for a PCM WAV file."""

    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getframerate(), handle.getnchannels(), handle.getnframes()
    except wave.Error as exc:
        raise AudioError(f"Cannot probe WAV file {path}: {exc}") from exc


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return mono float32 samples in ``[-1, 1]`` and the source sample rate."""

    if not path.is_file():
        raise AudioError(f"Audio file not found: {path}")
    if path.suffix.lower() == ".wav":
        try:
            return _read_wav(path)
        except AudioError as wav_error:
            try:
                return _read_soundfile(path)
            except AudioError:
                raise wav_error from None
    return _read_soundfile(path)


def prepare_track(src: Path, dest: Path, track_id: str) -> AudioTrack:
    samples, sample_rate = read_audio(src)
    resampled = resample_linear(samples, sample_rate, TARGET_SAMPLE_RATE)
    write_wav_mono(dest, resampled, TARGET_SAMPLE_RATE)
    duration = float(resampled.size) / TARGET_SAMPLE_RATE if TARGET_SAMPLE_RATE else 0.0
    return AudioTrack(
        track_id=track_id,
        path=dest,
        sample_rate=TARGET_SAMPLE_RATE,
        channels=1,
        duration_s=duration,
    )


def resample_linear(samples: np.ndarray, origin_rate: int, target_rate: int) -> np.ndarray:
    """Linear interpolation. Good enough for a skeleton; ASR stacks resample again."""

    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if origin_rate <= 0:
        raise AudioError(f"Invalid sample rate: {origin_rate}")
    if mono.size == 0 or origin_rate == target_rate:
        return mono
    duration = mono.size / origin_rate
    target_count = max(1, round(duration * target_rate))
    origin_x = np.linspace(0.0, 1.0, num=mono.size, endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_count, endpoint=False)
    return np.interp(target_x, origin_x, mono).astype(np.float32)


def write_wav_mono(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except wave.Error as exc:
        raise AudioError(
            f"Cannot read {path.name}. WAV PCM works out of the box. "
            "Convert m4a/mp3 with ffmpeg, or install meetscribe[audio] for FLAC and OGG."
        ) from exc
    if channels < 1 or sample_rate <= 0:
        raise AudioError(f"WAV file {path} has an empty audio format")
    samples = _pcm_to_float(raw, sample_width)
    if channels > 1:
        usable = samples.size - (samples.size % channels)
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return samples.astype(np.float32), sample_rate


def _pcm_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        return _pcm24_to_float(raw)
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise AudioError(f"Unsupported WAV sample width: {sample_width} bytes")


def _pcm24_to_float(raw: bytes) -> np.ndarray:
    byte_values = np.frombuffer(raw, dtype=np.uint8)
    if byte_values.size % 3 != 0:
        raise AudioError("Truncated 24-bit PCM")
    packed = byte_values.reshape(-1, 3).astype(np.int32)
    values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
    values = np.where(values >= 0x800000, values - 0x1000000, values)
    return values.astype(np.float32) / 8388608.0


def _read_soundfile(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise AudioError(
            f"Cannot read {path.name}. WAV PCM works out of the box. "
            "Install meetscribe[audio] for formats libsndfile supports, "
            "or convert with ffmpeg to 16-bit WAV."
        ) from exc
    data, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return np.asarray(mono, dtype=np.float32), int(sample_rate)

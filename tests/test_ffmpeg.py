import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from meetscribe.audio import io
from meetscribe.audio.io import prepare_track, read_audio, write_wav_mono
from meetscribe.errors import AudioError


def test_fallback_decodes_unicode_path_and_cleans_temp(tmp_path, monkeypatch):
    source = tmp_path / "會議 audio.m4a"
    source.write_bytes(b"original")
    outputs = []
    monkeypatch.setattr(
        io, "_read_soundfile", lambda _: (_ for _ in ()).throw(AudioError("unsupported"))
    )
    monkeypatch.setattr("meetscribe.audio.ffmpeg.shutil.which", lambda _: "/bin/ffmpeg")

    def run(args, **kwargs):
        assert args[args.index("-i") + 1] == str(source.resolve())
        assert args[args.index("-protocol_whitelist") + 1] == "file,pipe"
        assert args.index("-protocol_whitelist") < args.index("-i")
        assert kwargs.get("shell", False) is False
        output = Path(args[-1])
        outputs.append(output)
        write_wav_mono(output, np.full(1600, 0.25), 16000)
        return subprocess.CompletedProcess(args, 0, stderr="")

    monkeypatch.setattr("meetscribe.audio.ffmpeg.subprocess.run", run)
    track = prepare_track(source, tmp_path / "normalized.wav", "system")
    assert track.duration_s == 0.1
    assert track.track_id == "system"
    assert source.read_bytes() == b"original"
    assert outputs and not outputs[0].exists()


def test_missing_ffmpeg_is_actionable(tmp_path, monkeypatch):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"compressed")
    monkeypatch.setattr(
        io, "_read_soundfile", lambda _: (_ for _ in ()).throw(AudioError("unsupported"))
    )
    monkeypatch.setattr("meetscribe.audio.ffmpeg.shutil.which", lambda _: None)
    with pytest.raises(AudioError, match="Install ffmpeg"):
        read_audio(source)


def test_invalid_input_reports_decode_error_and_cleans_temp(tmp_path, monkeypatch):
    source = tmp_path / "broken.m4a"
    source.write_bytes(b"bad")
    outputs = []
    monkeypatch.setattr(
        io, "_read_soundfile", lambda _: (_ for _ in ()).throw(AudioError("unsupported"))
    )
    monkeypatch.setattr("meetscribe.audio.ffmpeg.shutil.which", lambda _: "/bin/ffmpeg")

    def run(args, **kwargs):
        outputs.append(Path(args[-1]))
        return subprocess.CompletedProcess(args, 1, stderr="No audio stream found")

    monkeypatch.setattr("meetscribe.audio.ffmpeg.subprocess.run", run)
    with pytest.raises(AudioError, match="No audio stream found"):
        read_audio(source)
    assert not outputs[0].parent.exists()


def test_pcm_wav_never_needs_ffmpeg(tmp_path, monkeypatch):
    source = tmp_path / "valid.wav"
    write_wav_mono(source, np.ones(100), 16000)
    monkeypatch.setattr(
        "meetscribe.audio.ffmpeg.shutil.which", lambda _: pytest.fail("unneeded ffmpeg")
    )
    assert read_audio(source)[0].size == 100


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_real_m4a_and_mp3_import(tmp_path):
    source = tmp_path / "source.wav"
    write_wav_mono(source, np.sin(np.arange(8000) * 0.1).astype(np.float32) * 0.2, 16000)
    for suffix in ("m4a", "mp3"):
        compressed = tmp_path / f"會議 audio.{suffix}"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), str(compressed)], check=True
        )
        track = prepare_track(compressed, tmp_path / f"{suffix}.wav", "mic")
        assert track.sample_rate == 16000
        assert abs(track.duration_s - 0.5) < 0.1
        assert np.max(np.abs(read_audio(track.path)[0])) > 0.05


def test_dual_compressed_tracks_normalize(tmp_path, monkeypatch):
    from meetscribe.audio.base import normalize_tracks
    from meetscribe.audio.file_input import FileInput

    mic = tmp_path / "mic.m4a"
    system = tmp_path / "system.mp3"
    mic.write_bytes(b"mic")
    system.write_bytes(b"system")
    monkeypatch.setattr(
        io, "_read_soundfile", lambda _: (_ for _ in ()).throw(AudioError("unsupported"))
    )
    monkeypatch.setattr("meetscribe.audio.ffmpeg.shutil.which", lambda _: "/bin/ffmpeg")

    def run(args, **kwargs):
        source = Path(args[args.index("-i") + 1])
        value = 0.1 if source == mic else 0.5
        write_wav_mono(Path(args[-1]), np.full(1600, value), 16000)
        return subprocess.CompletedProcess(args, 0, stderr="")

    monkeypatch.setattr("meetscribe.audio.ffmpeg.subprocess.run", run)
    result = normalize_tracks(FileInput(mic, system).record(tmp_path), tmp_path / "normal")
    assert [track.track_id for track in result.tracks] == ["mic", "system"]
    assert all(track.duration_s == 0.1 for track in result.tracks)
    assert read_audio(result.tracks[0].path)[0].mean() < 0.2
    assert read_audio(result.tracks[1].path)[0].mean() > 0.4


def test_empty_ffmpeg_output_rejected(tmp_path, monkeypatch):
    source = tmp_path / "empty.m4a"
    source.write_bytes(b"header")
    monkeypatch.setattr(
        io, "_read_soundfile", lambda _: (_ for _ in ()).throw(AudioError("unsupported"))
    )
    monkeypatch.setattr("meetscribe.audio.ffmpeg.shutil.which", lambda _: "/bin/ffmpeg")

    def run(args, **kwargs):
        write_wav_mono(Path(args[-1]), np.empty(0), 16000)
        return subprocess.CompletedProcess(args, 0, stderr="")

    monkeypatch.setattr("meetscribe.audio.ffmpeg.subprocess.run", run)
    with pytest.raises(AudioError, match="empty audio"):
        read_audio(source)

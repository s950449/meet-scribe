from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from meetscribe.cli import app

runner = CliRunner()


def test_help_and_version() -> None:
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    for name in ("record", "transcribe", "summarize", "run", "backends"):
        assert name in help_result.stdout
    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0
    assert "0.2.0" in version.stdout


def test_backends_human_and_json() -> None:
    human = runner.invoke(app, ["backends"])
    assert human.exit_code == 0
    assert "faster-whisper" in human.stdout
    assert "mlx-whisper" in human.stdout
    assert "single-speaker" in human.stdout
    assert "recommended_asr:" in human.stdout
    assert "usable=" in human.stdout

    payload = runner.invoke(app, ["--json-events", "backends"])
    assert payload.exit_code == 0
    assert '"name": "openvino"' in payload.stdout
    assert "Hardware" not in payload.stdout


def test_record_uses_platform_stub(tmp_path: Path) -> None:
    result = runner.invoke(app, ["record", "--output-dir", str(tmp_path / "take")])
    text = result.stdout + getattr(result, "stderr", "")
    assert result.exit_code == 1
    lowered = text.lower()
    assert "sounddevice" in lowered or "monitor" in lowered or "pactl" in lowered or "not installed" in lowered


def test_transcribe_reports_missing_asr_extra(tmp_path: Path) -> None:
    path = tmp_path / "mic.wav"
    _silence(path)
    result = runner.invoke(app, ["transcribe", str(path), "--output", str(tmp_path / "note.md")])
    text = result.stdout + getattr(result, "stderr", "")
    assert result.exit_code == 1
    assert "not installed" in text.lower() or "not wired" in text.lower() or "not implemented" in text.lower()


def test_devices_does_not_start_recording() -> None:
    result = runner.invoke(app, ["devices"])
    assert result.exit_code == 0
    assert "Capture devices" in result.stdout


def test_missing_config_is_an_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--config", str(tmp_path / "nope.toml"), "record"])
    text = result.stdout + getattr(result, "stderr", "")
    assert result.exit_code == 1
    assert "not found" in text.lower()


def _silence(path: Path) -> None:
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.zeros(1600, dtype="<i2").tobytes())

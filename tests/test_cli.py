from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from meetscribe.cli import app

runner = CliRunner()


@pytest.mark.parametrize("command", ["run", "transcribe"])
def test_resume_options_reach_pipeline(tmp_path: Path, monkeypatch, command: str) -> None:
    captured = {}

    def fake_run(config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(json_path=tmp_path / "note.json")

    monkeypatch.setattr("meetscribe.cli.run_meeting", fake_run)
    result = runner.invoke(
        app,
        [command, str(tmp_path / "mic.wav"), "--resume", "--work-dir", str(tmp_path / "work")],
    )
    assert result.exit_code == 0, result.output
    assert captured["resume"] is True
    assert captured["work_dir"] == tmp_path / "work"


def test_resume_without_source_does_not_start_recording(monkeypatch) -> None:
    def unexpected_record(*args, **kwargs):
        pytest.fail("resume must never overwrite a recording")

    monkeypatch.setattr("meetscribe.cli.record_audio", unexpected_record)
    result = runner.invoke(app, ["run", "--resume"])
    assert result.exit_code != 0
    assert "source" in result.output.lower()


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
    assert "provider:" in human.stdout
    assert "model:" in human.stdout

    payload = runner.invoke(app, ["--json-events", "backends"])
    assert payload.exit_code == 0
    assert '"name": "openvino"' in payload.stdout
    assert "Hardware" not in payload.stdout


def test_record_uses_platform_stub(tmp_path: Path) -> None:
    result = runner.invoke(app, ["record", "--output-dir", str(tmp_path / "take")])
    text = result.stdout + getattr(result, "stderr", "")
    assert result.exit_code == 1
    lowered = text.lower()
    assert (
        "sounddevice" in lowered
        or "monitor" in lowered
        or "pactl" in lowered
        or "not installed" in lowered
    )


def test_transcribe_reports_missing_asr_extra(tmp_path: Path) -> None:
    path = tmp_path / "mic.wav"
    _silence(path)
    result = runner.invoke(app, ["transcribe", str(path), "--output", str(tmp_path / "note.md")])
    text = result.stdout + getattr(result, "stderr", "")
    assert result.exit_code == 1
    lowered = text.lower()
    assert "not installed" in lowered or "not wired" in lowered or "not implemented" in lowered


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

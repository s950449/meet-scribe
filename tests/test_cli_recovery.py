import json
import shutil
import subprocess

import numpy as np
import pytest
from typer.testing import CliRunner

from meetscribe.audio.io import write_wav_mono
from meetscribe.cli import app
from meetscribe.document import load_document
from meetscribe.errors import MeetscribeError
from meetscribe.pipeline import run_meeting
from meetscribe.types import MeetingSummary, TranscriptSegment


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_compressed_cli_run_can_resume_failed_summary(tmp_path, monkeypatch):
    wav = tmp_path / "source.wav"
    source = tmp_path / "週會 recording.mp3"
    write_wav_mono(wav, np.sin(np.arange(8000) * 0.1) * 0.2, 16000)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav), str(source)], check=True)
    asr_calls = []
    summary_calls = []

    class Asr:
        name = "test-asr"

        def transcribe(self, audio, *, track_id, **kwargs):
            asr_calls.append(track_id)
            return [TranscriptSegment("下週完成 API", 0.0, 0.4, track_id)]

    class Summary:
        def summarize(self, utterances, **kwargs):
            summary_calls.append(True)
            if len(summary_calls) == 1:
                raise MeetscribeError("temporary summary failure")
            return MeetingSummary("下週完成 API。")

    def execute(config, **kwargs):
        return run_meeting(config, asr=Asr(), summarizer=Summary(), **kwargs)

    monkeypatch.setattr("meetscribe.cli.run_meeting", execute)
    arguments = [
        "--json-events",
        "run",
        str(source),
        "--no-polish",
        "-o",
        str(tmp_path / "note.md"),
        "--work-dir",
        str(tmp_path / "work"),
    ]
    runner = CliRunner()
    failed = runner.invoke(app, arguments)
    assert failed.exit_code == 1
    assert "temporary summary failure" in failed.output
    assert load_document(tmp_path / "note.json").utterances[0].text == "下週完成 API"

    resumed = runner.invoke(app, [*arguments, "--resume"])
    assert resumed.exit_code == 0, resumed.output
    assert asr_calls == ["mic"]
    assert len(summary_calls) == 2
    assert load_document(tmp_path / "note.json").summary.overview == "下週完成 API。"
    assert all(isinstance(json.loads(line), dict) for line in resumed.stdout.splitlines())

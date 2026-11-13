from pathlib import Path

import pytest

from meetscribe.agent import agent_payload, save_revision
from meetscribe.config import AppConfig
from meetscribe.document import load_document
from meetscribe.errors import MeetscribeError
from meetscribe.pipeline import run_meeting
from test_pipeline import FakeAsr, _write_silence


def test_agent_mode_never_builds_llm_and_preserves_raw(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Agent mode must not call an LLM")

    monkeypatch.setattr("meetscribe.pipeline.build_polisher", forbidden)
    monkeypatch.setattr("meetscribe.pipeline.build_summarizer", forbidden)
    source = tmp_path / "mic.wav"
    _write_silence(source)
    config = AppConfig()
    config.llm.polish_mode = "agent"
    result = run_meeting(config, mic=source, output=tmp_path / "raw.md", asr=FakeAsr())
    payload = agent_payload(result.json_path)
    assert payload["utterances"][0]["id"] == "u000000"
    revision = save_revision(
        result.json_path, payload["source_revision"], {"u000000": "我們改 API。"}
    )
    assert load_document(result.json_path).utterances[0].polished_text is None
    revised = load_document(Path(revision["json_path"]))
    assert revised.utterances[0].text == "我們改 API"
    assert revised.utterances[0].polished_text == "我們改 API。"
    assert revised.utterances[0].start == result.document.utterances[0].start
    with pytest.raises(MeetscribeError, match="revision"):
        save_revision(result.json_path, "stale", {"u000000": "oops"})
    with pytest.raises(MeetscribeError, match="utterance"):
        save_revision(result.json_path, payload["source_revision"], {"u999999": "oops"})


@pytest.mark.parametrize("command", ["transcribe", "run"])
def test_cli_agent_mode_and_validation(tmp_path, monkeypatch, command):
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from meetscribe.cli import app

    seen = []

    def capture(config, **kwargs):
        seen.append(config.llm.polish_mode)
        return SimpleNamespace(json_path=tmp_path / "note.json")

    monkeypatch.setattr("meetscribe.cli.run_meeting", capture)
    runner = CliRunner()
    result = runner.invoke(app, [command, "audio.wav", "--polish-mode", "agent"])
    assert result.exit_code == 0, result.output
    assert seen == ["agent"]
    result = runner.invoke(app, [command, "audio.wav", "--polish-mode", "invalid"])
    assert result.exit_code == 1
    assert seen == ["agent"]


def test_agent_config_and_none_mode(tmp_path, monkeypatch):
    from meetscribe.config import load_config

    path = tmp_path / "config.toml"
    path.write_text('[llm]\npolish_mode = "agent"\n')
    config = load_config(path, environ={})
    assert config.llm.polish_mode == "agent"
    assert (
        load_config(path, overrides={"polish_mode": "none"}, environ={}).llm.polish_mode == "none"
    )
    config.llm.polish_mode = "none"
    source = tmp_path / "audio.wav"
    _write_silence(source)
    monkeypatch.setattr("meetscribe.pipeline.build_polisher", lambda *a: pytest.fail("LLM called"))
    monkeypatch.setattr(
        "meetscribe.pipeline.build_summarizer", lambda *a: pytest.fail("LLM called")
    )
    result = run_meeting(config, mic=source, output=tmp_path / "raw.md", asr=FakeAsr())
    assert result.document.utterances[0].polished_text is None
    assert not (tmp_path / "raw.agent.json").exists()


@pytest.mark.parametrize("summary", [{"overview": []}, {"action_items": ["bad"]}, {"other": "bad"}])
def test_invalid_summary_does_not_write_revision(tmp_path, summary):
    source = tmp_path / "audio.wav"
    _write_silence(source)
    config = AppConfig()
    config.llm.polish_mode = "agent"
    result = run_meeting(config, mic=source, output=tmp_path / "raw.md", asr=FakeAsr())
    payload = agent_payload(result.json_path)
    with pytest.raises(MeetscribeError):
        save_revision(result.json_path, payload["source_revision"], {}, summary)
    assert not list(tmp_path.glob("raw.agent-*.json"))


def test_apply_agent_cli_roundtrip_and_stale_source(tmp_path):
    import json

    from typer.testing import CliRunner

    from meetscribe.cli import app

    source = tmp_path / "audio.wav"
    _write_silence(source)
    config = AppConfig()
    config.llm.polish_mode = "agent"
    result = run_meeting(config, mic=source, output=tmp_path / "raw.md", asr=FakeAsr())
    payload = agent_payload(result.json_path)
    response = tmp_path / "edits.json"
    response.write_text(
        json.dumps(
            {"source_revision": payload["source_revision"], "edits": {"u000000": "我們改 API。"}}
        )
    )
    runner = CliRunner()
    applied = runner.invoke(app, ["apply-agent", str(result.json_path), str(response)])
    assert applied.exit_code == 0, applied.output
    assert len(list(tmp_path.glob("raw.agent-*.json"))) == 1
    # Retrying is idempotent.
    applied = runner.invoke(app, ["apply-agent", str(result.json_path), str(response)])
    assert applied.exit_code == 0
    assert len(list(tmp_path.glob("raw.agent-*.json"))) == 1
    result.json_path.write_text(result.json_path.read_text() + "\n")
    stale = runner.invoke(app, ["apply-agent", str(result.json_path), str(response)])
    assert stale.exit_code == 1
    assert "revision changed" in stale.output

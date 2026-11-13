from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meetscribe.cli import app
from meetscribe.config import AppConfig, ConfigError, PublishSettings, load_config
from meetscribe.document import dump_document
from meetscribe.errors import MeetscribeError
from meetscribe.publish import build_payload, flush_outbox, pending, publish_note
from meetscribe.types import ActionItem, AttributedUtterance, MeetingDocument, MeetingSummary


class _Hook:
    def __init__(self) -> None:
        self.status = 200
        self.requests: list[tuple[dict[str, str], dict]] = []


@pytest.fixture
def hook() -> Iterator[tuple[_Hook, str]]:
    state = _Hook()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            state.requests.append((dict(self.headers), body))
            self.send_response(state.status)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state, f"http://127.0.0.1:{server.server_port}/webhook/meetscribe"
    server.shutdown()


def _note(tmp_path: Path) -> Path:
    document = MeetingDocument(
        title="週會",
        started_at=datetime(2026, 11, 11, 9, 50),
        summary=MeetingSummary(
            overview="確認發佈時程。",
            key_points=("API 可測試",),
            decisions=("下週一發佈",),
            action_items=(ActionItem(task="寫測試", owner="Alice", due="Friday"),),
        ),
        utterances=(
            AttributedUtterance(speaker_id="Me", text="大家好", start=0.0, end=1.0, track_id="mic"),
            AttributedUtterance(
                speaker_id="Remote", text="你好", start=1.0, end=2.0, track_id="system"
            ),
        ),
        speaker_map={},
    )
    note = tmp_path / "weekly.md"
    note.write_text("# 週會\n", encoding="utf-8")
    dump_document(document, note.with_suffix(".json"))
    return note


def _config(url: str) -> AppConfig:
    return AppConfig(publish=PublishSettings(url=url, timeout=5.0))


def test_payload_carries_summary_and_markdown(tmp_path: Path) -> None:
    payload = build_payload(_note(tmp_path))
    assert payload["source"] == "meetscribe"
    assert payload["title"] == "週會"
    assert payload["participants"] == ["Me", "Remote"]
    assert payload["summary"]["action_items"] == [  # type: ignore[index]
        {"task": "寫測試", "owner": "Alice", "due": "Friday"}
    ]
    assert payload["note"] == {"filename": "weekly.md", "markdown": "# 週會\n"}


def test_publish_sends_token_header(tmp_path: Path, hook, monkeypatch) -> None:
    state, url = hook
    monkeypatch.setenv("MEETSCRIBE_PUBLISH_TOKEN", "s3cret")
    outbox = tmp_path / "outbox"
    outcome = publish_note(_config(url), _note(tmp_path), outbox=outbox)
    assert outcome.sent
    headers, body = state.requests[0]
    assert headers["X-Meetscribe-Token"] == "s3cret"
    assert body["title"] == "週會"
    assert pending(outbox) == []


def test_failure_queues_privately_and_flush_resends(tmp_path: Path, hook, monkeypatch) -> None:
    state, url = hook
    monkeypatch.delenv("MEETSCRIBE_PUBLISH_TOKEN", raising=False)
    outbox = tmp_path / "outbox"
    state.status = 500
    outcome = publish_note(_config(url), _note(tmp_path), outbox=outbox)
    assert not outcome.sent and outcome.queued is not None
    assert "HTTP 500" in (outcome.error or "")
    assert outcome.queued.stat().st_mode & 0o777 == 0o600
    assert "X-Meetscribe-Token" not in state.requests[0][0]

    state.status = 200
    sent, errors = flush_outbox(_config(url), outbox=outbox)
    assert (sent, errors) == (1, [])
    assert pending(outbox) == []


def test_unreachable_webhook_is_queued(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox"
    outcome = publish_note(_config("http://127.0.0.1:9/webhook"), _note(tmp_path), outbox=outbox)
    assert not outcome.sent
    assert "unreachable" in (outcome.error or "")
    assert "/webhook" not in (outcome.error or "")
    assert len(pending(outbox)) == 1


@pytest.mark.parametrize(
    "url",
    ["", "http://WIREGUARD_HOST:5678/webhook/meetscribe", "https://YOUR-TEAM.app.n8n.cloud/x"],
)
def test_unconfigured_or_placeholder_url_is_rejected(tmp_path: Path, url: str) -> None:
    with pytest.raises(MeetscribeError):
        publish_note(_config(url), _note(tmp_path), outbox=tmp_path / "outbox")
    assert pending(tmp_path / "outbox") == []


def test_toml_publish_section_and_secret_rejection(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[publish]\nurl = "https://n8n.test/webhook/m"\nauto = true\ntoken_env = "X_TOKEN"\n',
        encoding="utf-8",
    )
    config = load_config(path, environ={})
    assert config.publish.url == "https://n8n.test/webhook/m"
    assert config.publish.auto is True
    assert config.publish_token({"X_TOKEN": "abc"}) == "abc"

    path.write_text('[publish]\ntoken = "abc"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path, environ={})


def test_run_publish_failure_does_not_fail_the_run(tmp_path: Path, monkeypatch) -> None:
    note = _note(tmp_path)
    monkeypatch.setenv("MEETSCRIBE_OUTBOX", str(tmp_path / "outbox"))
    monkeypatch.setenv("MEETSCRIBE_PUBLISH_URL", "http://127.0.0.1:9/webhook")

    class Result:
        markdown_path = note
        json_path = note.with_suffix(".json")

    monkeypatch.setattr("meetscribe.cli.run_meeting", lambda config, **kwargs: Result())
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(tmp_path / "mic.wav"), "--publish"])
    assert result.exit_code == 0, result.output
    assert "Queued" in result.output
    assert len(list((tmp_path / "outbox").glob("*.json"))) == 1

    result = runner.invoke(app, ["publish", str(note)])
    assert result.exit_code == 1


def test_run_with_placeholder_url_warns_instead_of_failing(tmp_path: Path, monkeypatch) -> None:
    note = _note(tmp_path)
    monkeypatch.setenv("MEETSCRIBE_PUBLISH_URL", "http://WIREGUARD_HOST:5678/webhook/meetscribe")

    class Result:
        markdown_path = note
        json_path = note.with_suffix(".json")

    monkeypatch.setattr("meetscribe.cli.run_meeting", lambda config, **kwargs: Result())
    result = CliRunner().invoke(app, ["run", str(tmp_path / "mic.wav"), "--publish"])
    assert result.exit_code == 0, result.output
    assert "placeholder" in result.output

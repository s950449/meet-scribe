from __future__ import annotations

import json
import threading
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from meetscribe import slack
from meetscribe.config import AppConfig, ConfigError, PublishSettings, load_config
from meetscribe.errors import MeetscribeError
from meetscribe.publish import PENDING_TARGETS, flush_outbox, pending, publish_note
from meetscribe.slack import format_message, validate_targets
from test_publish import _note


class FakeSlack:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str], dict[str, str]]] = []
        self.uploads: list[bytes] = []
        self.fail_channels: set[str] = set()
        self.base = ""

    def handle(self, path: str, headers: dict[str, str], body: bytes) -> dict:
        if path == "/upload":
            self.uploads.append(body)
            return {}
        method = path.removeprefix("/api/")
        fields = {k: v[0] for k, v in urllib.parse.parse_qs(body.decode()).items()}
        self.calls.append((method, fields, headers))
        if headers.get("Authorization") != "Bearer xoxb-test":
            return {"ok": False, "error": "invalid_auth"}
        if method == "conversations.open":
            return {"ok": True, "channel": {"id": "D" + fields["users"][1:]}}
        if method == "files.getUploadURLExternal":
            return {"ok": True, "upload_url": f"{self.base}/upload", "file_id": "F123"}
        if method == "files.completeUploadExternal":
            if fields["channel_id"] in self.fail_channels:
                return {"ok": False, "error": "not_in_channel"}
            return {"ok": True}
        if method == "chat.postMessage":
            return {"ok": True, "ts": "1700000000.000100"}
        return {"ok": False, "error": "unknown_method"}


@pytest.fixture
def fake_slack(monkeypatch) -> Iterator[FakeSlack]:
    state = FakeSlack()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            reply = json.dumps(state.handle(self.path, dict(self.headers), body)).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(slack, "SLACK_API", f"{state.base}/api")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    yield state
    server.shutdown()


def _config(*targets: str) -> AppConfig:
    return AppConfig(publish=PublishSettings(kind="slack", slack_targets=targets, timeout=5.0))


def test_channel_and_dm_get_file_with_summary_comment(tmp_path: Path, fake_slack) -> None:
    outcome = publish_note(_config("C0123ABC", "U0456DEF"), _note(tmp_path), outbox=tmp_path)
    assert outcome.sent, outcome.error
    methods = [call[0] for call in fake_slack.calls]
    assert methods == [
        "files.getUploadURLExternal",
        "files.completeUploadExternal",
        "conversations.open",
        "files.getUploadURLExternal",
        "files.completeUploadExternal",
    ]
    completes = [c for m, c, _ in fake_slack.calls if m == "files.completeUploadExternal"]
    assert [c["channel_id"] for c in completes] == ["C0123ABC", "D0456DEF"]
    assert "*待辦*" in completes[0]["initial_comment"]
    assert json.loads(completes[0]["files"]) == [{"id": "F123", "title": "週會"}]
    assert fake_slack.uploads == ["# 週會\n".encode(), "# 週會\n".encode()]


def test_long_summary_posts_message_and_threads_file(
    tmp_path: Path, fake_slack, monkeypatch
) -> None:
    monkeypatch.setattr(slack, "MAX_COMMENT_CHARS", 10)
    assert publish_note(_config("C0123ABC"), _note(tmp_path), outbox=tmp_path).sent
    methods = [m for m, _, _ in fake_slack.calls]
    assert methods[0] == "chat.postMessage"
    complete = next(c for m, c, _ in fake_slack.calls if m == "files.completeUploadExternal")
    assert complete["thread_ts"] == "1700000000.000100"
    assert "initial_comment" not in complete


def test_partial_failure_requeues_only_unsent_targets(tmp_path: Path, fake_slack) -> None:
    outbox = tmp_path / "outbox"
    fake_slack.fail_channels = {"C0999XYZ"}
    config = _config("C0123ABC", "C0999XYZ")
    outcome = publish_note(config, _note(tmp_path), outbox=outbox)
    assert not outcome.sent
    assert "not_in_channel" in (outcome.error or "")
    queued = json.loads(pending(outbox)[0].read_text(encoding="utf-8"))
    assert queued[PENDING_TARGETS] == ["C0999XYZ"]

    fake_slack.fail_channels = set()
    fake_slack.calls.clear()
    assert flush_outbox(config, outbox=outbox) == (1, [])
    completes = [c for m, c, _ in fake_slack.calls if m == "files.completeUploadExternal"]
    assert [c["channel_id"] for c in completes] == ["C0999XYZ"]
    assert pending(outbox) == []


def test_missing_token_or_bad_targets_are_config_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    with pytest.raises(MeetscribeError, match="SLACK_BOT_TOKEN"):
        publish_note(_config("C0123ABC"), _note(tmp_path), outbox=tmp_path / "o")
    with pytest.raises(MeetscribeError, match="IDs"):
        validate_targets(["#general"])
    with pytest.raises(MeetscribeError, match="empty"):
        validate_targets([])
    assert pending(tmp_path / "o") == []


def test_format_escapes_slack_markup() -> None:
    text = format_message(
        {
            "title": "<!channel> & 會議",
            "started_at": "2026-11-11T09:50:00",
            "participants": ["Me"],
            "summary": {"overview": "", "action_items": [{"task": "a", "owner": None}]},
        }
    )
    assert "<!channel>" not in text
    assert "&lt;!channel&gt; &amp; 會議" in text
    assert "• a" in text


def test_toml_slack_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[publish]\nkind = "slack"\nslack_targets = ["C0123ABC"]\nslack_token_env = "TEST_SLACK"\n',
        encoding="utf-8",
    )
    config = load_config(path, environ={})
    assert config.publish.kind == "slack"
    assert config.publish.slack_targets == ("C0123ABC",)
    assert config.slack_token({"TEST_SLACK": "xoxb-1"}) == "xoxb-1"
    path.write_text('[publish]\nkind = "email"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path, environ={})

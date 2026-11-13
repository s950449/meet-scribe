"""Post a meeting note to Slack with the Web API (bot token, no n8n).

For each target the summary is posted as the comment of an uploaded ``.md``
file. A ``U…``/``W…`` target is a user and gets a direct message through
``conversations.open``. Summaries too long for a file comment are posted as a
message, and the file goes into its thread.

Bot scopes: ``chat:write`` and ``files:write``, plus ``im:write`` for direct
messages. The bot must be a member of each channel.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from meetscribe.errors import MeetscribeError

SLACK_API = "https://slack.com/api"
TARGET_PATTERN = re.compile(r"^[CGDUW][A-Z0-9]{6,}$")
# Keep file comments short; longer summaries go in a message with the file threaded.
MAX_COMMENT_CHARS = 3500


class SlackError(MeetscribeError):
    """Slack rejected a call or could not be reached."""

    def __init__(self, message: str, *, remaining: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.remaining = list(remaining)


def validate_targets(targets: Sequence[str]) -> list[str]:
    cleaned = [target.strip() for target in targets if target.strip()]
    if not cleaned:
        raise MeetscribeError(
            "[publish].slack_targets is empty. Add channel IDs (C…) or user IDs (U…)."
        )
    bad = [target for target in cleaned if not TARGET_PATTERN.match(target)]
    if bad:
        raise MeetscribeError(
            "Slack targets must be IDs, not names: "
            + ", ".join(bad)
            + ". Find a channel ID under channel details → About; a user ID under "
            "profile → ⋮ → Copy member ID."
        )
    return cleaned


def format_message(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [f"*{_esc(payload.get('title'))}*  {_esc(payload.get('started_at'))}"]
    participants = payload.get("participants") or []
    if participants:
        lines.append("參與者：" + "、".join(_esc(item) for item in participants))
    if summary.get("overview"):
        lines += ["", _esc(summary["overview"])]
    for heading, key in (("重點", "key_points"), ("決議", "decisions")):
        items = summary.get(key) or []
        if items:
            lines += ["", f"*{heading}*", *(f"• {_esc(item)}" for item in items)]
    actions = summary.get("action_items") or []
    if actions:
        lines += ["", "*待辦*"]
        for item in actions:
            owner = f"（{_esc(item['owner'])}）" if item.get("owner") else ""
            due = f" — {_esc(item['due'])}" if item.get("due") else ""
            lines.append(f"• {_esc(item.get('task'))}{owner}{due}")
    return "\n".join(lines)


def post_note(
    payload: Mapping[str, Any],
    *,
    token: str,
    targets: Sequence[str],
    timeout: float = 30.0,
) -> None:
    """Send to every target. On failure ``SlackError.remaining`` lists the unsent ones."""

    note = payload.get("note") or {}
    filename = str(note.get("filename") or "meeting.md")
    content = str(note.get("markdown") or "").encode("utf-8")
    title = str(payload.get("title") or filename)
    text = format_message(payload)
    client = _Client(token, timeout)
    pending = list(targets)
    while pending:
        target = pending[0]
        try:
            channel = client.channel_for(target)
            if len(text) <= MAX_COMMENT_CHARS:
                client.upload(channel, filename, title, content, comment=text)
            else:
                ts = client.call("chat.postMessage", {"channel": channel, "text": text})["ts"]
                client.upload(channel, filename, title, content, thread_ts=str(ts))
        except SlackError as exc:
            raise SlackError(f"Slack {target}: {exc}", remaining=pending) from exc
        pending.pop(0)


class _Client:
    def __init__(self, token: str, timeout: float) -> None:
        self.token = token
        self.timeout = timeout

    def channel_for(self, target: str) -> str:
        if target[0] in "UW":
            opened = self.call("conversations.open", {"users": target})
            return str(opened["channel"]["id"])
        return target

    def upload(
        self,
        channel: str,
        filename: str,
        title: str,
        content: bytes,
        *,
        comment: str | None = None,
        thread_ts: str | None = None,
    ) -> None:
        ticket = self.call(
            "files.getUploadURLExternal", {"filename": filename, "length": str(len(content))}
        )
        self._raw_post(str(ticket["upload_url"]), content)
        fields = {
            "files": json.dumps([{"id": ticket["file_id"], "title": title}]),
            "channel_id": channel,
        }
        if comment:
            fields["initial_comment"] = comment
        if thread_ts:
            fields["thread_ts"] = thread_ts
        self.call("files.completeUploadExternal", fields)

    def call(self, method: str, fields: Mapping[str, str]) -> dict[str, Any]:
        body = urllib.parse.urlencode(fields).encode("utf-8")
        request = urllib.request.Request(
            f"{SLACK_API}/{method}",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            },
        )
        raw = self._open(request, method)
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise SlackError(f"{method} returned non-JSON") from exc
        if not isinstance(data, dict) or not data.get("ok"):
            error = data.get("error") if isinstance(data, dict) else "bad response"
            raise SlackError(f"{method} failed: {error}")
        return data

    def _raw_post(self, url: str, content: bytes) -> None:
        if not url.startswith("https://") and not url.startswith(SLACK_API.split("/api")[0]):
            raise SlackError("upload URL is not HTTPS")
        request = urllib.request.Request(
            url,
            data=content,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        self._open(request, "file upload")

    def _open(self, request: urllib.request.Request, what: str) -> bytes:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return bytes(response.read())
        except urllib.error.HTTPError as exc:
            retry = exc.headers.get("Retry-After") if exc.headers else None
            hint = f" (retry after {retry}s)" if retry else ""
            raise SlackError(f"{what} returned HTTP {exc.code}{hint}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise SlackError(f"{what} unreachable: {getattr(exc, 'reason', exc)}") from exc


def _esc(value: object) -> str:
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

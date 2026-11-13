"""Send a finished meeting note to Slack directly, or to a webhook such as n8n.

The payload carries the structured summary from the JSON sidecar and the full
Markdown note, so the receiver can post a Slack message and attach the file
without calling an LLM again. A shared secret is sent as ``X-Meetscribe-Token``
when ``[publish].token_env`` names a set environment variable.

When the webhook cannot be reached (e.g. WireGuard is down), the payload is kept
in a private outbox and resent by ``meetscribe publish --flush``.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from meetscribe.config import AppConfig
from meetscribe.document import load_document
from meetscribe.errors import MeetscribeError
from meetscribe.slack import SlackError, post_note, validate_targets

PAYLOAD_VERSION = 1
PLACEHOLDER_MARKERS = ("WIREGUARD_HOST", "YOUR-TEAM", "example.invalid")


class PublishError(MeetscribeError):
    """The receiver rejected the note or could not be reached."""

    def __init__(self, message: str, *, remaining: Sequence[str] = ()) -> None:
        super().__init__(message)
        # Slack targets not yet delivered; empty for the webhook kind.
        self.remaining = list(remaining)


@dataclass(frozen=True, slots=True)
class PublishOutcome:
    sent: bool
    queued: Path | None = None
    error: str | None = None


def default_outbox(environ: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    env = os.environ if environ is None else environ
    configured = env.get("MEETSCRIBE_OUTBOX", "").strip()
    if configured:
        return Path(configured).expanduser()
    state = env.get("XDG_STATE_HOME", "").strip()
    base = Path(state).expanduser() if state else (home or Path.home()) / ".local" / "state"
    return base / "meetscribe" / "outbox"


def build_payload(markdown_path: Path, json_path: Path | None = None) -> dict[str, object]:
    if not markdown_path.is_file():
        raise MeetscribeError(f"Meeting note not found: {markdown_path}")
    sidecar = json_path or markdown_path.with_suffix(".json")
    document = load_document(sidecar)
    markdown = markdown_path.read_text(encoding="utf-8")
    started = document.started_at.isoformat(timespec="seconds")
    digest = hashlib.sha256(f"{started}\0{document.title}\0{markdown}".encode()).hexdigest()
    summary = document.summary
    return {
        "version": PAYLOAD_VERSION,
        "source": "meetscribe",
        "id": digest[:16],
        "title": document.title,
        "started_at": started,
        "participants": document.participants(),
        "summary": {
            "overview": summary.overview,
            "key_points": list(summary.key_points),
            "decisions": list(summary.decisions),
            "action_items": [
                {"task": item.task, "owner": item.owner, "due": item.due}
                for item in summary.action_items
            ],
        },
        "note": {"filename": markdown_path.name, "markdown": markdown},
    }


PENDING_TARGETS = "_pending_targets"


def send_payload(payload: Mapping[str, object], config: AppConfig) -> None:
    """Deliver one payload with the configured kind. Raises PublishError on failure."""

    check_publish_config(config)
    if config.publish.kind == "slack":
        _send_slack(payload, config)
    else:
        _send_webhook(payload, config)


def check_publish_config(config: AppConfig) -> None:
    """Raise MeetscribeError for settings that a retry cannot fix."""

    if config.publish.kind == "slack":
        validate_targets(config.publish.slack_targets)
        if not config.slack_token():
            raise MeetscribeError(
                f"Set the Slack bot token in ${config.publish.slack_token_env} (xoxb-…)."
            )
    else:
        _publish_url(config)


def _send_webhook(payload: Mapping[str, object], config: AppConfig) -> None:
    url = _publish_url(config)
    headers = {"Content-Type": "application/json; charset=utf-8"}
    token = config.publish_token()
    if token:
        headers["X-Meetscribe-Token"] = token
    clean = {key: value for key, value in payload.items() if key != PENDING_TARGETS}
    body = json.dumps(clean, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=config.publish.timeout) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise PublishError(f"Webhook returned HTTP {exc.code}: {detail}".strip()) from exc
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise PublishError(f"Webhook {_redact(url)} unreachable: {reason}") from exc


def _send_slack(payload: Mapping[str, object], config: AppConfig) -> None:
    queued = payload.get(PENDING_TARGETS)
    targets = (
        [str(item) for item in queued]
        if isinstance(queued, list)
        else validate_targets(config.publish.slack_targets)
    )
    try:
        post_note(
            payload,
            token=config.slack_token() or "",
            targets=targets,
            timeout=config.publish.timeout,
        )
    except SlackError as exc:
        raise PublishError(str(exc), remaining=exc.remaining) from exc


def publish_note(
    config: AppConfig,
    markdown_path: Path,
    *,
    json_path: Path | None = None,
    outbox: Path | None = None,
) -> PublishOutcome:
    """Send one note. On a network, HTTP, or Slack failure the payload is queued."""

    check_publish_config(config)
    payload = build_payload(markdown_path, json_path)
    try:
        send_payload(payload, config)
    except PublishError as exc:
        queued = _enqueue(_with_remaining(payload, exc), outbox or default_outbox())
        return PublishOutcome(sent=False, queued=queued, error=str(exc))
    return PublishOutcome(sent=True)


def flush_outbox(config: AppConfig, *, outbox: Path | None = None) -> tuple[int, list[str]]:
    """Resend queued payloads. Returns the number sent and the errors left."""

    check_publish_config(config)
    directory = outbox or default_outbox()
    if not directory.is_dir():
        return 0, []
    sent = 0
    errors: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: unreadable ({exc})")
            continue
        try:
            send_payload(payload, config)
        except PublishError as exc:
            errors.append(f"{path.name}: {exc}")
            _enqueue(_with_remaining(payload, exc), directory)
            continue
        path.unlink()
        sent += 1
    return sent, errors


def pending(outbox: Path | None = None) -> list[Path]:
    directory = outbox or default_outbox()
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def _with_remaining(payload: Mapping[str, object], exc: PublishError) -> dict[str, object]:
    updated = dict(payload)
    if exc.remaining:
        updated[PENDING_TARGETS] = list(exc.remaining)
    return updated


def _publish_url(config: AppConfig) -> str:
    url = config.publish.url.strip()
    if not url:
        raise MeetscribeError(
            "Publishing is not configured. Set [publish].url in config.toml "
            'or MEETSCRIBE_PUBLISH_URL, or use kind = "slack".'
        )
    if any(marker in url for marker in PLACEHOLDER_MARKERS):
        raise MeetscribeError(
            f"[publish].url still contains a placeholder: {url}. "
            "Replace it with your n8n webhook URL."
        )
    if not url.startswith(("http://", "https://")):
        raise MeetscribeError(f"[publish].url must be an http(s) URL: {url}")
    return url


def _enqueue(payload: Mapping[str, object], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / f"{str(payload.get('started_at', '')).replace(':', '')}-{payload['id']}.json"
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    temporary.replace(path)
    return path


def _redact(url: str) -> str:
    """Drop the webhook path, which may itself act as a secret."""

    scheme, _, rest = url.partition("://")
    host = rest.split("/", 1)[0]
    return f"{scheme}://{host}/…"

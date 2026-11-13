"""Whisper behind an OpenAI-compatible ``/v1/audio/transcriptions`` endpoint.

Built for LocalAI's SYCL whisper.cpp backend (scripts/start-localai.sh), and
works with any server that returns ``verbose_json`` segments. Segment times
are seconds. The audio leaves the process only for ``server_url``; keep that
on loopback for local-only runs.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from meetscribe.asr.common import INITIAL_PROMPT, blank_language, build_segment
from meetscribe.errors import MeetscribeError
from meetscribe.events import log_event
from meetscribe.types import AsrCapabilities, ProgressEvent, TranscriptSegment

DEFAULT_SERVER_URL = "http://127.0.0.1:8080/v1"
# Long tracks are transcribed in one request; allow for a slow first load.
DEFAULT_TIMEOUT = 3600.0

Poster = Callable[[str, dict[str, str], Path, float], dict[str, Any]]


class OpenAIApiWhisperAsr:
    name = "openai-api"

    def __init__(
        self,
        *,
        model: str,
        language: str = "zh",
        server_url: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        poster: Poster | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self.server_url = (server_url.strip() or DEFAULT_SERVER_URL).rstrip("/")
        self.timeout = timeout
        self._poster = poster or _post

    def capabilities(self) -> AsrCapabilities:
        return AsrCapabilities(streaming=False, word_timestamps=False, device="server")

    def transcribe(
        self,
        audio: Path,
        *,
        track_id: str,
        language: str | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> list[TranscriptSegment]:
        chosen = language if language is not None else blank_language(self.language)
        url = f"{self.server_url}/audio/transcriptions"
        fields = {
            "model": self.model,
            "response_format": "verbose_json",
            "prompt": INITIAL_PROMPT,
        }
        if chosen:
            fields["language"] = chosen
        if on_event is not None:
            on_event(log_event("asr", f"POST {url}", model=self.model, track=track_id))
        payload = self._poster(url, fields, audio, self.timeout)
        return parse_verbose_json(payload, track_id, chosen)


def parse_verbose_json(
    payload: dict[str, Any],
    track_id: str,
    language: str | None = None,
) -> list[TranscriptSegment]:
    rows = payload.get("segments")
    segments: list[TranscriptSegment] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            built = build_segment(
                str(row.get("text") or ""),
                _seconds(row.get("start")),
                _seconds(row.get("end")),
                track_id,
                language=language,
            )
            if built is not None:
                segments.append(built)
    if not segments and str(payload.get("text") or "").strip():
        duration = _seconds(payload.get("duration"))
        built = build_segment(str(payload["text"]), 0.0, duration, track_id, language=language)
        if built is not None:
            segments.append(built)
    return segments


def _seconds(value: object) -> float:
    try:
        return max(0.0, float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _post(url: str, fields: dict[str, str], audio: Path, timeout: float) -> dict[str, Any]:
    body, content_type = _multipart(fields, audio)
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": content_type}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise MeetscribeError(f"ASR server returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise MeetscribeError(
            f"ASR request to {url} failed: {exc}. Is scripts/start-localai.sh running?"
        ) from exc
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        text = raw[:200].decode("utf-8", "replace")
        raise MeetscribeError(f"ASR server returned non-JSON: {text}") from exc
    if not isinstance(payload, dict):
        raise MeetscribeError("ASR server returned an unexpected payload")
    return payload


def _multipart(fields: dict[str, str], audio: Path) -> tuple[bytes, str]:
    boundary = f"meetscribe-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    filename = audio.name.replace('"', "_")
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
    )
    parts.append(audio.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"

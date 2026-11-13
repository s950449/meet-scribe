from __future__ import annotations

from pathlib import Path
from typing import Any

from meetscribe.asr.openai_api import OpenAIApiWhisperAsr, parse_verbose_json
from meetscribe.config import AppConfig, apply_overrides
from meetscribe.registry import build_registry

LOCALAI_PAYLOAD = {
    "segments": [
        {"id": 0, "start": 0, "end": 6.98, "text": " 歡迎參加專案會議。", "tokens": [1]},
        {"id": 1, "start": 6.98, "end": 9.5, "text": "API 已經可以測試。"},
        {"id": 2, "start": 9.5, "end": 9.5, "text": "  "},
    ],
    "text": "歡迎參加專案會議。API 已經可以測試。",
    "language": "zh",
    "duration": 9.5,
}


def test_parse_verbose_json_keeps_segments_in_seconds() -> None:
    segments = parse_verbose_json(LOCALAI_PAYLOAD, "mic", "zh")
    assert [(s.text, s.start, s.end) for s in segments] == [
        ("歡迎參加專案會議。", 0.0, 6.98),
        ("API 已經可以測試。", 6.98, 9.5),
    ]
    assert all(s.track_id == "mic" and s.language == "zh" for s in segments)


def test_parse_verbose_json_falls_back_to_text() -> None:
    segments = parse_verbose_json({"text": "只有全文", "duration": 3.2}, "system")
    assert [(s.text, s.start, s.end) for s in segments] == [("只有全文", 0.0, 3.2)]
    assert parse_verbose_json({}, "system") == []


def test_transcribe_posts_model_language_and_prompt(tmp_path: Path) -> None:
    audio = tmp_path / "mic.wav"
    audio.write_bytes(b"RIFF")
    calls: list[tuple[str, dict[str, str], Path]] = []

    def poster(url: str, fields: dict[str, str], path: Path, timeout: float) -> dict[str, Any]:
        calls.append((url, fields, path))
        return LOCALAI_PAYLOAD

    asr = OpenAIApiWhisperAsr(
        model="breeze-asr-25",
        language="zh",
        server_url="http://127.0.0.1:8080/v1/",
        poster=poster,
    )
    segments = asr.transcribe(audio, track_id="mic")
    assert len(segments) == 2
    url, fields, path = calls[0]
    assert url == "http://127.0.0.1:8080/v1/audio/transcriptions"
    assert fields["model"] == "breeze-asr-25"
    assert fields["language"] == "zh"
    assert fields["response_format"] == "verbose_json"
    assert path == audio

    asr.transcribe(audio, track_id="mic", language=None)
    asr_auto = OpenAIApiWhisperAsr(model="m", language="auto", poster=poster)
    asr_auto.transcribe(audio, track_id="mic")
    assert "language" not in calls[-1][1]


def test_registry_builds_openai_api_from_config() -> None:
    config = apply_overrides(
        AppConfig(),
        {
            "asr_backend": "openai-api",
            "asr_model": "breeze-asr-25",
            "asr_server_url": "http://127.0.0.1:9000/v1",
        },
    )
    asr = build_registry().create("asr", "openai-api", config)
    assert isinstance(asr, OpenAIApiWhisperAsr)
    assert asr.model == "breeze-asr-25"
    assert asr.server_url == "http://127.0.0.1:9000/v1"

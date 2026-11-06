"""whisper.cpp with the Vulkan GPU backend.

Two runners are supported. ``whisper-cli`` from a GGML_VULKAN build is preferred
(``MEETSCRIBE_WHISPER_CPP`` or ``whisper-cli`` on PATH). Otherwise the backend
uses pywhispercpp, which is Vulkan only when that package itself was built with
``GGML_VULKAN=1``. Prebuilt wheels are CPU builds of the same code path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from meetscribe.asr.common import INITIAL_PROMPT, blank_language, build_segment, coalesce, join_text
from meetscribe.asr.models import resolve_ggml_file
from meetscribe.errors import BackendNotInstalled, MeetscribeError
from meetscribe.events import log_event
from meetscribe.types import AsrCapabilities, ProgressEvent, TranscriptSegment, WordTimestamp

Runner = Callable[[Sequence[str], Path], None]


class VulkanWhisperAsr:
    name = "vulkan"

    def __init__(
        self,
        *,
        model: str,
        fallback_model: str,
        language: str = "zh",
        model_dir: Path | None = None,
        whisper_cpp_bin: str = "",
        backend_name: str = "vulkan",
        runner: Runner | None = None,
    ) -> None:
        self.name = backend_name
        self.requested_model = model
        self.fallback_model = fallback_model
        self.language = language
        self.model_dir = model_dir
        self.whisper_cpp_bin = whisper_cpp_bin
        self._runner = runner

    def capabilities(self) -> AsrCapabilities:
        return AsrCapabilities(streaming=False, word_timestamps=True, device="vulkan")

    def transcribe(
        self,
        audio: Path,
        *,
        track_id: str,
        language: str | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> list[TranscriptSegment]:
        from meetscribe.asr.models import default_model_dir

        model_dir = self.model_dir or default_model_dir()
        model_path, warning = resolve_ggml_file(
            self.requested_model,
            self.fallback_model,
            model_dir,
        )
        if warning and on_event is not None:
            on_event(log_event("asr", warning, model=str(model_path)))
        chosen = language if language is not None else blank_language(self.language)
        binary = _resolve_binary(self.whisper_cpp_bin)
        if binary is not None or self._runner is not None:
            if on_event is not None:
                on_event(
                    log_event(
                        "asr",
                        f"whisper.cpp binary: {binary or 'injected runner'}",
                        model=str(model_path),
                    )
                )
            return _transcribe_cli(
                binary or "whisper-cli",
                model_path,
                audio,
                track_id,
                chosen,
                runner=self._runner,
            )
        if on_event is not None:
            on_event(
                log_event(
                    "asr",
                    "Using pywhispercpp. GPU inference requires that binding "
                    "to be built with GGML_VULKAN=1. "
                    "A Vulkan whisper-cli takes priority when "
                    "MEETSCRIBE_WHISPER_CPP or whisper-cli is set.",
                    model=str(model_path),
                )
            )
        return _transcribe_binding(model_path, audio, track_id, chosen)


def _resolve_binary(configured: str) -> str | None:
    candidate = configured.strip() or os.environ.get("MEETSCRIBE_WHISPER_CPP", "").strip()
    if candidate and Path(candidate).expanduser().is_file():
        return str(Path(candidate).expanduser())
    found = shutil.which(candidate or "whisper-cli")
    return found


def _transcribe_cli(
    binary: str,
    model_path: Path,
    audio: Path,
    track_id: str,
    language: str | None,
    *,
    runner: Runner | None,
) -> list[TranscriptSegment]:
    with tempfile.TemporaryDirectory(prefix="meetscribe-vulkan-") as tmp:
        out_base = Path(tmp) / "out"
        command = [
            binary,
            "-m",
            str(model_path),
            "-f",
            str(audio),
            "-l",
            language or "auto",
            "-ojf",
            "-of",
            str(out_base),
            "-np",
            "--prompt",
            INITIAL_PROMPT,
            "-ml",
            "1",
            "-sow",
        ]
        if runner is None:
            _run_cli(command, out_base.with_suffix(".json"))
        else:
            runner(command, out_base.with_suffix(".json"))
        payload = json.loads(out_base.with_suffix(".json").read_text(encoding="utf-8"))
    return parse_whisper_cpp_json(payload, track_id)


def _run_cli(command: list[str], output: Path) -> None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise BackendNotInstalled(f"Could not execute {command[0]}: {exc}") from exc
    if completed.returncode != 0 or not output.is_file():
        detail = (completed.stderr or completed.stdout or "").strip()
        raise MeetscribeError(f"whisper-cli failed (exit {completed.returncode}). {detail}".strip())


def _transcribe_binding(
    model_path: Path,
    audio: Path,
    track_id: str,
    language: str | None,
) -> list[TranscriptSegment]:
    try:
        from pywhispercpp.model import Model
    except ImportError as exc:
        raise BackendNotInstalled(
            "Neither whisper-cli nor pywhispercpp is available. "
            "Build whisper.cpp with -DGGML_VULKAN=1 and set MEETSCRIBE_WHISPER_CPP, "
            "or uv sync --extra vulkan."
        ) from exc
    model = Model(str(model_path), print_progress=False, print_realtime=False)
    raw = model.transcribe(
        str(audio),
        language=language or "auto",
        token_timestamps=True,
        max_len=1,
        split_on_word=True,
    )
    segments = [_segment_from_binding(item, track_id) for item in raw]
    return coalesce([item for item in segments if item is not None])


def parse_whisper_cpp_json(payload: dict[str, object], track_id: str) -> list[TranscriptSegment]:
    """Parse ``whisper-cli -oj`` / ``-ojf`` output. Offsets are milliseconds."""

    rows = payload.get("transcription") or []
    if not isinstance(rows, list):
        return []
    segments: list[TranscriptSegment] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start, end = _offset_seconds(row.get("offsets"))
        words = _token_words(row.get("tokens"))
        text = str(row.get("text") or "")
        if not text.strip() and words:
            text = join_text([word.text for word in words])
        built = build_segment(text, start, end, track_id, words)
        if built is not None:
            segments.append(built)
    if segments and _looks_like_words(segments):
        return coalesce(segments)
    return segments


def _segment_from_binding(item: object, track_id: str) -> TranscriptSegment | None:
    text = str(getattr(item, "text", "") or "")
    if hasattr(item, "start") and hasattr(item, "end"):
        start = float(item.start)
        end = float(item.end)
    else:
        # pywhispercpp timestamps are centiseconds.
        start = float(getattr(item, "t0", 0)) * 0.01
        end = float(getattr(item, "t1", 0)) * 0.01
    word = WordTimestamp(text=text.strip(), start=start, end=end) if text.strip() else None
    return build_segment(text, start, end, track_id, (word,) if word else ())


def _offset_seconds(offsets: object) -> tuple[float, float]:
    if not isinstance(offsets, dict):
        return 0.0, 0.0
    start = float(offsets.get("from") or 0.0) / 1000.0
    end = float(offsets.get("to") or 0.0) / 1000.0
    return start, end


def _token_words(tokens: object) -> tuple[WordTimestamp, ...]:
    if not isinstance(tokens, list):
        return ()
    words: list[WordTimestamp] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        text = str(token.get("text") or "").strip()
        if not text or text.startswith("<|") or text.endswith("|>"):
            continue
        start, end = _offset_seconds(token.get("offsets"))
        words.append(WordTimestamp(text=text, start=start, end=end))
    return tuple(words)


def _looks_like_words(segments: list[TranscriptSegment]) -> bool:
    if len(segments) < 2:
        return False
    durations = [item.end - item.start for item in segments]
    return (sum(durations) / len(durations)) < 0.45

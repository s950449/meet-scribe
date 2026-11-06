"""Apple Silicon ASR through mlx-whisper.

The default checkpoint is the public MLX conversion ``schsu/breeze-asr-25-mlx``.
``mlx-community/whisper-large-v3-turbo`` is the fallback. Word timestamps come
from mlx-whisper's cross-attention alignment.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from meetscribe.asr.common import INITIAL_PROMPT, blank_language, build_segment
from meetscribe.asr.models import resolve_mlx_repo
from meetscribe.errors import BackendNotInstalled
from meetscribe.events import log_event
from meetscribe.types import AsrCapabilities, ProgressEvent, TranscriptSegment, WordTimestamp


class MlxWhisperAsr:
    name = "mlx-whisper"

    def __init__(
        self,
        *,
        model: str,
        fallback_model: str,
        language: str = "zh",
        transcribe_fn: Callable[..., dict[str, object]] | None = None,
    ) -> None:
        self.requested_model = model
        self.repo, self.model_warning = resolve_mlx_repo(model, fallback_model)
        self.language = language
        self._transcribe_fn = transcribe_fn

    def capabilities(self) -> AsrCapabilities:
        return AsrCapabilities(streaming=False, word_timestamps=True, device="metal")

    def transcribe(
        self,
        audio: Path,
        *,
        track_id: str,
        language: str | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> list[TranscriptSegment]:
        if self.model_warning and on_event is not None:
            on_event(log_event("asr", self.model_warning, model=self.repo))
        chosen = language if language is not None else blank_language(self.language)
        runner = self._transcribe_fn or _load_transcribe()
        result = runner(
            str(audio),
            path_or_hf_repo=self.repo,
            word_timestamps=True,
            language=chosen,
            initial_prompt=INITIAL_PROMPT,
            verbose=None,
            condition_on_previous_text=False,
        )
        return _segments_from_mlx(result, track_id)


def _load_transcribe() -> Callable[..., dict[str, object]]:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise BackendNotInstalled(
            "mlx-whisper is not installed. On Apple Silicon: uv sync --extra mlx"
        ) from exc
    return mlx_whisper.transcribe


def _segments_from_mlx(result: dict[str, object], track_id: str) -> list[TranscriptSegment]:
    language = result.get("language")
    language_text = str(language) if language else None
    raw_segments = result.get("segments") or []
    if not isinstance(raw_segments, list):
        return []
    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        words = _words(item.get("words"))
        built = build_segment(
            str(item.get("text") or ""),
            float(item.get("start") or 0.0),
            float(item.get("end") or 0.0),
            track_id,
            words,
            language_text,
        )
        if built is not None:
            segments.append(built)
    return segments


def _words(value: object) -> tuple[WordTimestamp, ...]:
    if not isinstance(value, list):
        return ()
    words: list[WordTimestamp] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("word") or item.get("text") or "").strip()
        if not text:
            continue
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or start)
        words.append(WordTimestamp(text=text, start=start, end=end))
    return tuple(words)

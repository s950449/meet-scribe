"""CTranslate2 Whisper via faster-whisper. Optional extra: ``meetscribe[faster-whisper]``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from meetscribe.errors import BackendNotInstalled
from meetscribe.events import log_event
from meetscribe.types import AsrCapabilities, ProgressEvent, TranscriptSegment, WordTimestamp

# Hugging Face ids that are not CTranslate2 checkpoints, mapped to community CT2 builds.
CT2_ALIASES = {
    "openai/whisper-large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "openai/whisper-large-v3": "Systran/faster-whisper-large-v3",
    "openai/whisper-large-v2": "Systran/faster-whisper-large-v2",
    "openai/whisper-medium": "Systran/faster-whisper-medium",
    "openai/whisper-small": "Systran/faster-whisper-small",
}

BUILTIN_CT2_NAMES = frozenset(
    {
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
        "turbo",
        "distil-large-v3",
    }
)

INITIAL_PROMPT = "以下是繁體中文與英文夾雜的會議，技術術語保留英文。"


def resolve_ct2_model(model: str, fallback_model: str) -> tuple[str, str | None]:
    """Pick a CTranslate2 id.

    MediaTek Breeze-ASR-25 is a Transformers checkpoint. faster-whisper cannot
    load it, so this backend substitutes the configured fallback and returns a
    warning the pipeline can show.
    """

    selected = model.strip()
    if not selected:
        selected = fallback_model.strip()
    if selected in CT2_ALIASES:
        return CT2_ALIASES[selected], None
    if (
        selected in BUILTIN_CT2_NAMES
        or "faster-whisper" in selected
        or selected.startswith("Systran/")
    ):
        return selected, None
    local = Path(selected)
    if local.exists():
        return str(local), None

    fallback = fallback_model.strip() or "openai/whisper-large-v3-turbo"
    ct2_fallback = CT2_ALIASES.get(
        fallback,
        fallback if "faster-whisper" in fallback else "large-v3-turbo",
    )
    warning = (
        f"{selected} is not a CTranslate2 checkpoint. faster-whisper will load {ct2_fallback}. "
        "Use the mlx-whisper backend on Apple Silicon for MediaTek-Research/Breeze-ASR-25."
    )
    return ct2_fallback, warning


class FasterWhisperAsr:
    """CPU Whisper. Word timestamps come from faster-whisper's alignment pass."""

    name = "faster-whisper"

    def __init__(
        self,
        *,
        model: str,
        fallback_model: str,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "zh",
    ) -> None:
        self.requested_model = model
        self.model_id, self.model_warning = resolve_ct2_model(model, fallback_model)
        self.device = "cpu" if device in {"", "auto"} else device
        self.compute_type = compute_type
        self.language = language

    def capabilities(self) -> AsrCapabilities:
        return AsrCapabilities(
            streaming=False,
            word_timestamps=True,
            device=self.device,
        )

    def transcribe(
        self,
        audio: Path,
        *,
        track_id: str,
        language: str | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> list[TranscriptSegment]:
        if self.model_warning and on_event is not None:
            on_event(log_event("asr", self.model_warning, model=self.model_id))
        whisper_model = _load_model(self.model_id, self.device, self.compute_type)
        chosen = language if language is not None else _blank_to_none(self.language)
        segments_iter, info = whisper_model.transcribe(
            str(audio),
            language=chosen,
            word_timestamps=True,
            vad_filter=False,
            initial_prompt=INITIAL_PROMPT,
        )
        detected = getattr(info, "language", None)
        results: list[TranscriptSegment] = []
        for segment in segments_iter:
            words = tuple(
                WordTimestamp(text=word.word, start=float(word.start), end=float(word.end))
                for word in (getattr(segment, "words", None) or [])
            )
            text = (segment.text or "").strip()
            if not text:
                continue
            results.append(
                TranscriptSegment(
                    text=text,
                    start=float(segment.start),
                    end=float(segment.end),
                    track_id=track_id,
                    words=words,
                    language=detected,
                )
            )
        return results


def _blank_to_none(language: str) -> str | None:
    if language.strip().lower() in {"", "auto", "detect"}:
        return None
    return language


def _load_model(model_id: str, device: str, compute_type: str) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise BackendNotInstalled(
            "faster-whisper is not installed. Install it with: uv sync --extra faster-whisper"
        ) from exc
    return WhisperModel(model_id, device=device, compute_type=compute_type)

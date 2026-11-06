"""Intel ASR through OpenVINO GenAI ``WhisperPipeline``.

``auto`` selects a GPU device when ``Core.available_devices`` lists one, and
otherwise runs on CPU. The pipeline reads 16 kHz mono audio.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from meetscribe.asr.common import INITIAL_PROMPT, blank_language, build_segment
from meetscribe.asr.models import resolve_openvino_dir
from meetscribe.audio.io import read_audio, resample_linear
from meetscribe.errors import BackendNotInstalled
from meetscribe.events import log_event
from meetscribe.types import AsrCapabilities, ProgressEvent, TranscriptSegment, WordTimestamp

TARGET_RATE = 16_000


def select_openvino_device(requested: str, available: Sequence[str]) -> tuple[str, str | None]:
    """Pick a device string OpenVINO will accept."""

    names = [(device, device.upper()) for device in available]
    want = requested.strip().upper() or "AUTO"
    gpus = [device for device, upper in names if upper == "GPU" or upper.startswith("GPU.")]
    if want in {"", "AUTO"}:
        if gpus:
            return gpus[0], None
        if available:
            return "CPU", "OpenVINO did not list a GPU. Using CPU."
        return "CPU", None
    if want == "GPU" and not gpus:
        return "CPU", "Requested OpenVINO device GPU is not available. Using CPU."
    for device, upper in names:
        if upper == want or upper.startswith(f"{want}."):
            return device, None
    return requested.strip() or "CPU", None


class OpenVINOWhisperAsr:
    name = "openvino"

    def __init__(
        self,
        *,
        model: str,
        fallback_model: str,
        language: str = "zh",
        device: str = "auto",
        model_dir: Path | None = None,
        pipeline_factory: Callable[..., object] | None = None,
        device_lister: Callable[[], list[str]] | None = None,
        audio_loader: Callable[[Path], np.ndarray] | None = None,
    ) -> None:
        self.requested_model = model
        self.fallback_model = fallback_model
        self.language = language
        self.requested_device = device or "auto"
        self.model_dir = model_dir
        self._pipeline_factory = pipeline_factory
        self._device_lister = device_lister
        self._audio_loader = audio_loader

    def capabilities(self) -> AsrCapabilities:
        requested = self.requested_device.strip().upper()
        device = "openvino-gpu" if requested in {"", "AUTO", "GPU"} else "cpu"
        return AsrCapabilities(streaming=False, word_timestamps=True, device=device)

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
        ir_dir, warning = resolve_openvino_dir(self.requested_model, self.fallback_model, model_dir)
        if warning and on_event is not None:
            on_event(log_event("asr", warning, model=str(ir_dir)))
        available = (self._device_lister or list_openvino_devices)()
        device, device_warning = select_openvino_device(self.requested_device, available)
        if device_warning and on_event is not None:
            on_event(log_event("asr", device_warning, device=device))
        if on_event is not None:
            on_event(
                log_event(
                    "asr",
                    f"OpenVINO device {device}, model {ir_dir.name}",
                    device=device,
                )
            )
        factory = self._pipeline_factory or _import_pipeline()
        pipeline = _construct_pipeline(factory, ir_dir, device)
        samples = (self._audio_loader or _load_16k)(audio)
        chosen = language if language is not None else blank_language(self.language)
        result = _generate(pipeline, samples, chosen)
        return _segments_from_result(result, track_id)


def list_openvino_devices() -> list[str]:
    try:
        import openvino as ov
    except ImportError:
        return []
    try:
        return [str(device) for device in ov.Core().available_devices]
    except Exception:
        return []


def _import_pipeline() -> Callable[..., object]:
    try:
        from openvino_genai import WhisperPipeline
    except ImportError as exc:
        raise BackendNotInstalled(
            "openvino-genai is not installed. uv sync --extra openvino"
        ) from exc
    return WhisperPipeline


def _construct_pipeline(factory: Callable[..., object], ir_dir: Path, device: str) -> object:
    try:
        return factory(str(ir_dir), device, word_timestamps=True)
    except TypeError:
        return factory(str(ir_dir), device)


def _generate(pipeline: object, samples: np.ndarray, language: str | None) -> object:
    speech = np.ascontiguousarray(samples, dtype=np.float32)
    kwargs: dict[str, object] = {
        "return_timestamps": True,
        "task": "transcribe",
        "initial_prompt": INITIAL_PROMPT,
    }
    if language:
        kwargs["language"] = language
    generate = getattr(pipeline, "generate", None)
    if not callable(generate):
        raise BackendNotInstalled("openvino-genai WhisperPipeline has no generate()")
    attempts: list[tuple[object, dict[str, object]]] = [
        (speech, kwargs),
        (speech.tolist(), kwargs),
        (speech.tolist(), {key: value for key, value in kwargs.items() if key != "initial_prompt"}),
    ]
    last_error: TypeError | None = None
    for audio, options in attempts:
        try:
            return generate(audio, **options)
        except TypeError as exc:
            last_error = exc
    raise BackendNotInstalled(
        "openvino-genai WhisperPipeline.generate rejected the audio buffer and timestamp options. "
        f"Last error: {last_error}"
    ) from last_error


def _load_16k(path: Path) -> np.ndarray:
    samples, rate = read_audio(path)
    return resample_linear(samples, rate, TARGET_RATE)


def _segments_from_result(result: object, track_id: str) -> list[TranscriptSegment]:
    chunks = getattr(result, "chunks", None) or []
    segments: list[TranscriptSegment] = []
    for chunk in chunks:
        text = str(getattr(chunk, "text", "") or "")
        start = float(getattr(chunk, "start_ts", 0.0) or 0.0)
        end = float(getattr(chunk, "end_ts", start) or start)
        words = _chunk_words(chunk)
        built = build_segment(text, start, end, track_id, words)
        if built is not None:
            segments.append(built)
    if segments:
        return segments
    texts = getattr(result, "texts", None) or []
    if texts:
        built = build_segment(str(texts[0]), 0.0, 0.0, track_id)
        return [built] if built is not None else []
    text = str(result).strip()
    built = build_segment(text, 0.0, 0.0, track_id)
    return [built] if built is not None else []


def _chunk_words(chunk: object) -> tuple[WordTimestamp, ...]:
    raw = getattr(chunk, "words", None)
    if not raw:
        return ()
    words: list[WordTimestamp] = []
    for word in raw:
        text = str(getattr(word, "text", "") or getattr(word, "word", "") or "").strip()
        if not text:
            continue
        start = float(getattr(word, "start_ts", getattr(word, "start", 0.0)) or 0.0)
        end = float(getattr(word, "end_ts", getattr(word, "end", start)) or start)
        words.append(WordTimestamp(text=text, start=start, end=end))
    return tuple(words)

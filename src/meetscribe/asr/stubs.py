"""ASR backends that are specified but not wired to a model runtime yet."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from meetscribe.errors import BackendNotImplemented, BackendNotInstalled
from meetscribe.types import AsrCapabilities, ProgressEvent, TranscriptSegment


class _StubAsr:
    name = "stub"
    device = "cpu"
    module = ""
    install_hint = ""
    planned = "This ASR backend is not implemented yet."

    def __init__(self, *, model: str, fallback_model: str, language: str = "zh") -> None:
        self.model = model
        self.fallback_model = fallback_model
        self.language = language

    def capabilities(self) -> AsrCapabilities:
        return AsrCapabilities(streaming=False, word_timestamps=True, device=self.device)

    def transcribe(
        self,
        audio: Path,
        *,
        track_id: str,
        language: str | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> list[TranscriptSegment]:
        del audio, track_id, language, on_event
        self._lazy_import()
        raise BackendNotImplemented(self.planned)

    def _lazy_import(self) -> None:
        raise NotImplementedError


class MlxWhisperAsr(_StubAsr):
    """Apple Silicon backend. Planned runtime: ``mlx_whisper`` on Metal.

    Default checkpoint: ``MediaTek-Research/Breeze-ASR-25``. Fallback:
    ``openai/whisper-large-v3-turbo``, which mlx-whisper can load directly.
    """

    name = "mlx-whisper"
    device = "metal"
    install_hint = "pip install 'meetscribe[mlx]'"
    planned = (
        "mlx-whisper is not wired up yet. On Apple Silicon it will load "
        "MediaTek-Research/Breeze-ASR-25 through MLX/Metal, falling back to "
        "openai/whisper-large-v3-turbo. Install the extra with: pip install 'meetscribe[mlx]'"
    )

    def _lazy_import(self) -> None:
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as exc:
            raise BackendNotInstalled(
                f"mlx-whisper is not installed. {self.install_hint}"
            ) from exc


class OpenVINOWhisperAsr(_StubAsr):
    """Intel backend. Planned runtime: OpenVINO GenAI ``WhisperPipeline`` on the iGPU.

    Models need an OpenVINO IR export. The Hugging Face Breeze checkpoint has
    to be converted before this backend can select it; until then the planned
    fallback is an OpenVINO Whisper large-v3-turbo export.
    """

    name = "openvino"
    device = "openvino-gpu"
    install_hint = "pip install 'meetscribe[openvino]'"
    planned = (
        "OpenVINO GenAI WhisperPipeline is not wired up yet. "
        "It will target Intel integrated GPUs and keep a large-v3-turbo IR as fallback. "
        "Install the extra with: pip install 'meetscribe[openvino]'"
    )

    def _lazy_import(self) -> None:
        try:
            from openvino_genai import WhisperPipeline  # noqa: F401
        except ImportError as exc:
            raise BackendNotInstalled(
                f"openvino-genai is not installed. {self.install_hint}"
            ) from exc


class WhisperCppAsr(_StubAsr):
    """Portable fallback. Planned runtime: whisper.cpp through pywhispercpp.

    whisper.cpp wants GGML weights, not the Breeze Transformers repo. The
    planned fallback file is a large-v3-turbo GGML build.
    """

    name = "whisper.cpp"
    device = "cpu"
    install_hint = "pip install 'meetscribe[whisper-cpp]'"
    planned = (
        "whisper.cpp via pywhispercpp is not wired up yet. "
        "It will load a GGML large-v3-turbo model as the portable fallback. "
        "Install the extra with: pip install 'meetscribe[whisper-cpp]'"
    )

    def _lazy_import(self) -> None:
        try:
            from pywhispercpp.model import Model  # noqa: F401
        except ImportError as exc:
            raise BackendNotInstalled(
                f"pywhispercpp is not installed. {self.install_hint}"
            ) from exc

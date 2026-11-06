"""Neural diarizers. The pipeline calls them; the model code is still a stub."""

from __future__ import annotations

from collections.abc import Callable

from meetscribe.errors import BackendNotImplemented, BackendNotInstalled
from meetscribe.types import (
    AudioTrack,
    DiarizationCapabilities,
    ProgressEvent,
    SpeakerTurn,
)


class PyannoteDiarizer:
    """Planned: pyannote.audio 4 with ``speaker-diarization-community-1``.

    The model is gated. The pipeline will read the token from the environment
    variable named in config (``HF_TOKEN`` by default) and will not store it.
    Overlap detection is part of that pipeline, so ``overlap`` is declared.
    """

    name = "pyannote"

    def __init__(self, *, hf_token: str | None = None, model: str = "speaker-diarization-community-1") -> None:
        self.hf_token = hf_token
        self.model = model

    def capabilities(self) -> DiarizationCapabilities:
        return DiarizationCapabilities(
            streaming=False,
            overlap=True,
            device="cpu",
            requires_token=True,
        )

    def diarize(
        self,
        tracks: list[AudioTrack],
        *,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> list[SpeakerTurn]:
        del tracks, on_event
        try:
            import pyannote.audio  # noqa: F401
        except ImportError as exc:
            raise BackendNotInstalled(
                "pyannote.audio is not installed. Install it with: pip install 'meetscribe[diarization]'"
            ) from exc
        if not self.hf_token:
            raise BackendNotInstalled(
                "pyannote diarization needs a Hugging Face token. Set HF_TOKEN. "
                "The token is read from the environment and is not stored in config.toml."
            )
        raise BackendNotImplemented(
            "pyannote speaker-diarization-community-1 is not wired up yet. "
            "Use --diarization track or --diarization single-speaker."
        )


class SherpaDiarizer:
    """Planned: sherpa-onnx speaker diarization, fully local, no Hugging Face token."""

    name = "sherpa-onnx"

    def __init__(self, *, model_dir: str | None = None) -> None:
        self.model_dir = model_dir

    def capabilities(self) -> DiarizationCapabilities:
        return DiarizationCapabilities(
            streaming=False,
            overlap=False,
            device="cpu",
            requires_token=False,
        )

    def diarize(
        self,
        tracks: list[AudioTrack],
        *,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> list[SpeakerTurn]:
        del tracks, on_event
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError as exc:
            raise BackendNotInstalled(
                "sherpa-onnx is not installed. Install it with: pip install 'meetscribe[sherpa]'"
            ) from exc
        raise BackendNotImplemented(
            "sherpa-onnx diarization is not wired up yet. "
            "Use --diarization track or --diarization single-speaker."
        )

"""Diarizers that are specified but not wired up yet."""

from __future__ import annotations

from collections.abc import Callable

from meetscribe.errors import BackendNotImplemented, BackendNotInstalled
from meetscribe.types import (
    AudioTrack,
    DiarizationCapabilities,
    ProgressEvent,
    SpeakerTurn,
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

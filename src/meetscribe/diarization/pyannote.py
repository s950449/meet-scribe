"""pyannote speaker-diarization-community-1.

Remote participants share the system track, so that file is diarized and the
microphone track stays a single local speaker named ``Me``. A lone mixed file
is diarized as a whole. Overlapping turns are marked.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any

from meetscribe.diarization.simple import MIC_SPEAKER
from meetscribe.errors import BackendNotInstalled, MeetscribeError
from meetscribe.events import log_event
from meetscribe.types import AudioTrack, DiarizationCapabilities, ProgressEvent, SpeakerTurn

DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"


class PyannoteDiarizer:
    name = "pyannote"

    def __init__(
        self,
        *,
        hf_token: str | None = None,
        model: str = "speaker-diarization-community-1",
        pipeline_loader: Callable[[], object] | None = None,
    ) -> None:
        self.hf_token = hf_token
        self.model = model if "/" in model else f"pyannote/{model}"
        self._pipeline_loader = pipeline_loader

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
        if not tracks:
            return []
        if not self.hf_token and self._pipeline_loader is None:
            raise BackendNotInstalled(
                "pyannote diarization needs a Hugging Face token. Set HF_TOKEN. "
                "Accept the model conditions for pyannote/speaker-diarization-community-1. "
                "The token is read from the environment and is not stored in config.toml."
            )
        pipeline = self._load_pipeline()
        mic_tracks = [track for track in tracks if track.track_id == "mic"]
        other_tracks = [track for track in tracks if track.track_id != "mic"]
        turns: list[SpeakerTurn] = []
        if other_tracks:
            for track in mic_tracks:
                turns.append(
                    SpeakerTurn(
                        speaker_id=MIC_SPEAKER,
                        start=0.0,
                        end=track.duration_s,
                        track_id=track.track_id,
                    )
                )
            targets = other_tracks
        else:
            targets = tracks
        for track in targets:
            if on_event is not None:
                on_event(log_event("diarize", f"pyannote on {track.track_id}", model=self.model))
            output = pipeline(str(track.path))
            turns.extend(_turns_from_output(output, track.track_id))
        return _mark_overlaps(turns)

    def _load_pipeline(self) -> Any:
        if self._pipeline_loader is not None:
            return self._pipeline_loader()
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise BackendNotInstalled(
                "pyannote.audio is not installed. uv sync --extra diarization"
            ) from exc
        try:
            pipeline = Pipeline.from_pretrained(self.model, token=self.hf_token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(self.model, use_auth_token=self.hf_token)
        if pipeline is None:
            raise MeetscribeError(
                f"Could not load {self.model}. "
                "Accept the Hugging Face conditions and check HF_TOKEN."
            )
        return pipeline


def _turns_from_output(output: object, track_id: str) -> list[SpeakerTurn]:
    annotation = getattr(output, "speaker_diarization", output)
    turns: list[SpeakerTurn] = []
    for speech_turn, speaker in _iter_annotation(annotation):
        start = float(getattr(speech_turn, "start", 0.0))
        end = float(getattr(speech_turn, "end", start))
        turns.append(
            SpeakerTurn(
                speaker_id=str(speaker),
                start=start,
                end=end,
                track_id=track_id,
            )
        )
    return turns


def _iter_annotation(annotation: Any) -> Iterable[tuple[Any, str]]:
    itertracks = getattr(annotation, "itertracks", None)
    if callable(itertracks):
        for turn, _track, speaker in itertracks(yield_label=True):
            yield turn, str(speaker)
        return
    if not isinstance(annotation, Iterable):
        return
    for item in annotation:
        if isinstance(item, tuple) and len(item) == 3:
            turn, _track, speaker = item
            yield turn, str(speaker)
        elif isinstance(item, tuple) and len(item) == 2:
            turn, speaker = item
            yield turn, str(speaker)


def _mark_overlaps(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    marked: list[SpeakerTurn] = []
    for turn in turns:
        overlaps = False
        for other in turns:
            if (
                other is turn
                or other.track_id != turn.track_id
                or other.speaker_id == turn.speaker_id
            ):
                continue
            if _overlap(turn.start, turn.end, other.start, other.end) > 0:
                overlaps = True
                break
        marked.append(replace(turn, overlap=turn.overlap or overlaps))
    return marked


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))

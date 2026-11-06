"""Capture, recognize, diarize, polish, summarize, and write a meeting note.

A Qt, GTK, or Electron shell should call these functions and pass its own
``EventSink``. The CLI is one sink that can print JSONL.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from meetscribe.audio.base import normalize_tracks
from meetscribe.audio.file_input import FileInput
from meetscribe.config import AppConfig, language_hint
from meetscribe.document import dump_document, load_document
from meetscribe.errors import AudioError
from meetscribe.events import log_event, stage_event
from meetscribe.hardware import HardwareInfo, detect_hardware, recommended_diarization
from meetscribe.llm.factory import build_polisher, build_summarizer
from meetscribe.llm.polisher import IdentityPolisher
from meetscribe.llm.summarizer import ExtractiveSummarizer
from meetscribe.merge import assign_speakers
from meetscribe.registry import Registry, build_registry
from meetscribe.render.markdown import render_markdown
from meetscribe.types import (
    AttributedUtterance,
    EventSink,
    MeetingDocument,
    MeetingSummary,
    ProgressEvent,
    TranscriptSegment,
)

PENDING_SUMMARY = "（尚未摘要。執行 meetscribe summarize 以產生摘要。）"


@dataclass(frozen=True, slots=True)
class MeetingResult:
    document: MeetingDocument
    markdown_path: Path
    json_path: Path


def record_audio(
    config: AppConfig,
    output_dir: Path,
    *,
    duration_s: float | None = None,
    live: bool = False,
    sink: EventSink | None = None,
    registry: Registry | None = None,
    hardware: HardwareInfo | None = None,
) -> Path:
    """Record mic and system audio onto one host-clock timeline."""

    events = sink or _Null()
    reg = registry or build_registry()
    hw = hardware or detect_hardware()
    name = reg.resolve_capture(config.capture.backend, hw)
    if name == "file":
        raise AudioError(
            "The file backend loads existing audio. "
            "Use `meetscribe transcribe <file>` or select macos, windows, or linux."
        )
    capture = reg.create("capture", name, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    events.emit(
        stage_event(
            "capture",
            "start",
            message=f"Recording with {name}. Ctrl+C stops.",
            backend=name,
            duration_s=duration_s,
        )
    )
    on_chunk = None
    if live:
        from meetscribe.audio.live import make_live_callback

        asr_name = reg.resolve_asr(config.asr.backend, hw)
        asr = reg.create("asr", asr_name, config)
        sample_rate = int(getattr(capture, "sample_rate", config.capture.sample_rate))
        on_chunk = make_live_callback(asr, language_hint(config.asr.language), events, sample_rate)
        events.emit(log_event("live", f"Live transcription with {asr_name} every {config.capture.chunk_seconds:g}s"))
    capture.record(  # type: ignore[attr-defined]
        output_dir,
        duration_s=duration_s,
        on_event=events.emit,
        on_chunk=on_chunk,
    )
    events.emit(stage_event("capture", "done", message=f"Saved audio to {output_dir}"))
    return output_dir


def run_meeting(
    config: AppConfig,
    *,
    mic: Path | None = None,
    system: Path | None = None,
    output: Path | None = None,
    title: str | None = None,
    sink: EventSink | None = None,
    registry: Registry | None = None,
    hardware: HardwareInfo | None = None,
    asr: object | None = None,
    diarizer: object | None = None,
    polisher: object | None = None,
    summarizer: object | None = None,
    do_polish: bool = True,
    do_summary: bool = True,
    keep_audio: bool = False,
    work_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> MeetingResult:
    """Run the file pipeline from audio through Markdown."""

    if mic is None:
        raise AudioError("Pass a microphone recording, or a session directory with mic.wav")
    events = sink or _Null()
    reg = registry or build_registry()
    hw = hardware or detect_hardware()
    owned_work = work_dir is None
    scratch = work_dir if work_dir is not None else Path(tempfile.mkdtemp(prefix="meetscribe-"))
    try:
        captured = FileInput(mic=mic, system=system).record(scratch / "raw")
        events.emit(
            stage_event(
                "capture",
                "done",
                message=f"Loaded {len(captured.tracks)} track(s)",
                tracks=len(captured.tracks),
            )
        )
        normalized = normalize_tracks(captured, scratch / "audio", on_event=events.emit)
        segments = _transcribe(config, normalized.tracks, events, reg, hw, asr)
        turns = _diarize(config, normalized.tracks, events, reg, diarizer)
        events.emit(stage_event("merge", "start", message="Assigning speakers"))
        utterances = assign_speakers(segments, turns)
        events.emit(
            stage_event(
                "merge",
                "done",
                message=f"Assigned speakers to {len(utterances)} utterance(s)",
                utterances=len(utterances),
            )
        )
        polished = _polish(config, utterances, events, polisher, do_polish, environ)
        summary = _summarize(
            config,
            polished,
            events,
            summarizer,
            do_summary,
            title or _title_from_path(mic),
            environ,
        )
        document = MeetingDocument(
            title=title or _title_from_path(mic),
            started_at=captured.started_at,
            summary=summary,
            utterances=tuple(polished),
            speaker_map=dict(config.speaker_map),
        )
        destination = _resolve_output(config, output, document)
        result = _write(document, destination, events)
        if keep_audio:
            _keep_audio(scratch / "audio", result.markdown_path)
        return result
    finally:
        if owned_work:
            shutil.rmtree(scratch, ignore_errors=True)


def summarize_file(
    config: AppConfig,
    json_path: Path,
    *,
    output: Path | None = None,
    sink: EventSink | None = None,
    summarizer: object | None = None,
    environ: Mapping[str, str] | None = None,
) -> MeetingResult:
    """Fill in the summary for a transcript JSON sidecar and rewrite Markdown."""

    events = sink or _Null()
    document = load_document(json_path)
    speaker_map = dict(document.speaker_map)
    speaker_map.update(config.speaker_map)
    document.speaker_map = speaker_map
    events.emit(stage_event("summarize", "start", message="Summarizing transcript"))
    chosen = summarizer or build_summarizer(config, environ)
    if isinstance(chosen, ExtractiveSummarizer):
        events.emit(
            log_event(
                "summarize",
                "Writing an extractive summary without calling the LLM endpoint " f"at {config.llm.base_url}.",
            )
        )
    document.summary = chosen.summarize(  # type: ignore[attr-defined]
        document.utterances,
        title=document.title,
        speaker_map=document.speaker_map,
        on_event=events.emit,
    )
    events.emit(stage_event("summarize", "done", message="Summary ready"))
    destination = output or json_path.with_suffix(".md")
    return _write(document, destination, events)


def _transcribe(config, tracks, events, registry, hardware, asr) -> list[TranscriptSegment]:
    if asr is None:
        name = registry.resolve_asr(config.asr.backend, hardware)
        asr = registry.create("asr", name, config)
        events.emit(log_event("asr", f"ASR backend: {name}", backend=name, model=config.asr.model))
    else:
        events.emit(log_event("asr", f"ASR backend: {getattr(asr, 'name', 'custom')}"))
    language = language_hint(config.asr.language)
    segments: list[TranscriptSegment] = []
    for track in tracks:
        events.emit(
            stage_event(
                "asr",
                "start",
                track_id=track.track_id,
                message=f"Transcribing {track.track_id}",
            )
        )
        track_segments = asr.transcribe(
            track.path,
            track_id=track.track_id,
            language=language,
            on_event=events.emit,
        )
        segments.extend(track_segments)
        events.emit(
            stage_event(
                "asr",
                "done",
                track_id=track.track_id,
                message=f"{track.track_id}: {len(track_segments)} segment(s)",
                segments=len(track_segments),
            )
        )
    return segments


def _diarize(config, tracks, events, registry, diarizer):
    if diarizer is None:
        name = config.diarization.backend
        if name == "auto":
            name = recommended_diarization(len(tracks))
        diarizer = registry.create("diarization", name, config)
    else:
        name = getattr(diarizer, "name", "custom")
    events.emit(stage_event("diarize", "start", message=f"Diarizing with {name}", backend=name))
    turns = diarizer.diarize(list(tracks), on_event=events.emit)
    events.emit(
        stage_event(
            "diarize",
            "done",
            message=f"{len(turns)} speaker turn(s)",
            turns=len(turns),
        )
    )
    return turns


def _polish(config, utterances, events, polisher, do_polish, environ) -> tuple[AttributedUtterance, ...]:
    if not do_polish:
        chosen = IdentityPolisher(reason="Polishing skipped")
    else:
        chosen = polisher or build_polisher(config, environ)
    if isinstance(chosen, IdentityPolisher):
        events.emit(stage_event("polish", "skip", message=chosen.reason))
    else:
        events.emit(
            stage_event(
                "polish",
                "start",
                message=f"Polishing with {_active_model(chosen, config)} at {config.llm.base_url}",
            )
        )
    polished = chosen.polish(utterances, on_event=events.emit)
    if not isinstance(chosen, IdentityPolisher):
        events.emit(stage_event("polish", "done", message="Transcript polished"))
    return tuple(polished)


def _summarize(config, utterances, events, summarizer, do_summary, title, environ) -> MeetingSummary:
    if not do_summary:
        events.emit(stage_event("summarize", "skip", message="Summary skipped"))
        return MeetingSummary(overview=PENDING_SUMMARY)
    chosen = summarizer or build_summarizer(config, environ)
    if isinstance(chosen, ExtractiveSummarizer):
        events.emit(
            log_event(
                "summarize",
                "Writing an extractive summary without calling the LLM endpoint.",
            )
        )
    else:
        events.emit(
            stage_event(
                "summarize",
                "start",
                message=f"Summarizing with {_active_model(chosen, config)} at {config.llm.base_url}",
            )
        )
    summary = chosen.summarize(utterances, title=title, speaker_map=dict(config.speaker_map), on_event=events.emit)
    events.emit(stage_event("summarize", "done", message="Summary ready"))
    return summary


def _active_model(chosen: object, config: AppConfig) -> str:
    client = getattr(chosen, "client", None)
    model = getattr(client, "model", None)
    if isinstance(model, str) and model:
        return model
    return config.llm.model


def _write(document: MeetingDocument, markdown_path: Path, events: EventSink) -> MeetingResult:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(document), encoding="utf-8")
    json_path = markdown_path.with_suffix(".json")
    dump_document(document, json_path)
    events.emit(
        stage_event(
            "write",
            "done",
            message=f"Wrote {markdown_path}",
            markdown=str(markdown_path),
            json=str(json_path),
        )
    )
    return MeetingResult(document=document, markdown_path=markdown_path, json_path=json_path)


def _keep_audio(audio_dir: Path, markdown_path: Path) -> None:
    if not audio_dir.is_dir():
        return
    dest = markdown_path.with_name(markdown_path.stem + ".audio")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(audio_dir, dest)


def _resolve_output(config: AppConfig, output: Path | None, document: MeetingDocument) -> Path:
    if output is not None:
        path = Path(output).expanduser()
        if path.suffix == "":
            return path.with_suffix(".md")
        return path
    stamp = document.started_at.strftime("%Y%m%d-%H%M%S")
    return config.output_dir.expanduser() / f"{stamp}-{_slug(document.title)}.md"


def _title_from_path(path: Path) -> str:
    if path.is_dir():
        return path.name or "會議"
    return path.stem or "會議"


def _slug(title: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", title, flags=re.UNICODE).strip("-_")
    return slug[:80] or "meeting"


class _Null:
    def emit(self, event: ProgressEvent) -> None:
        del event


# Re-export for callers that catch the base error from a pipeline run.
__all__ = ["MeetingResult", "record_audio", "run_meeting", "summarize_file"]

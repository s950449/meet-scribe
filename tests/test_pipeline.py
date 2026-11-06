from datetime import datetime
from pathlib import Path

import numpy as np

from meetscribe.config import AppConfig
from meetscribe.document import load_document
from meetscribe.pipeline import run_meeting, summarize_file
from meetscribe.types import AsrCapabilities, MeetingSummary, ProgressEvent, TranscriptSegment


def _write_silence(path: Path, *, rate: int = 16000, seconds: float = 0.4) -> None:
    import wave

    count = int(rate * seconds)
    pcm = np.zeros(count, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


class MemorySink:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


class FakeAsr:
    name = "fake-asr"

    def capabilities(self) -> AsrCapabilities:
        return AsrCapabilities(streaming=False, word_timestamps=False, device="cpu")

    def transcribe(self, audio: Path, *, track_id: str, language: str | None = None, on_event=None):
        del audio, language, on_event
        if track_id == "system":
            return [TranscriptSegment("gRPC timeout 800ms", 0.1, 0.3, track_id)]
        return [TranscriptSegment("我們改 API", 0.0, 0.2, track_id)]


class FakeSummarizer:
    name = "fake-summary"

    def summarize(self, utterances, *, title: str, speaker_map: dict[str, str], on_event=None) -> MeetingSummary:
        del utterances, on_event
        owner = speaker_map.get("Me", "Me")
        return MeetingSummary(
            overview=f"{title} 由 {owner} 主持。",
            key_points=("保留 timeout。",),
            decisions=("改用 gRPC。",),
            action_items=(),
        )


def test_run_meeting_with_fakes(tmp_path: Path) -> None:
    mic = tmp_path / "mic.wav"
    system = tmp_path / "system.wav"
    _write_silence(mic)
    _write_silence(system)
    config = AppConfig(speaker_map={"Me": "林可安", "Remote": "客戶"}, output_dir=tmp_path)
    sink = MemorySink()
    result = run_meeting(
        config,
        mic=mic,
        system=system,
        output=tmp_path / "notes" / "weekly.md",
        title="支付週會",
        sink=sink,
        asr=FakeAsr(),
        summarizer=FakeSummarizer(),
        do_polish=False,
        environ={},
        work_dir=tmp_path / "work",
    )
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "# 支付週會" in text
    assert "林可安" in text
    assert "客戶" in text
    assert "我們改 API" in text
    assert "gRPC timeout 800ms" in text
    assert "支付週會 由 林可安 主持。" in text
    assert "改用 gRPC。" in text
    loaded = load_document(result.json_path)
    assert loaded.title == "支付週會"
    assert {item.speaker_id for item in loaded.utterances} == {"Me", "Remote"}
    stages = [(event.stage, event.status) for event in sink.events]
    assert ("asr", "done") in stages
    assert ("merge", "done") in stages
    assert ("write", "done") in stages


def test_auto_diarization_uses_single_speaker_for_one_file(tmp_path: Path) -> None:
    mic = tmp_path / "only.wav"
    _write_silence(mic)
    result = run_meeting(
        AppConfig(output_dir=tmp_path),
        mic=mic,
        output=tmp_path / "one.md",
        asr=FakeAsr(),
        do_summary=False,
        do_polish=False,
        environ={},
    )
    assert result.document.utterances[0].speaker_id == "SPEAKER_00"
    assert "尚未摘要" in result.document.summary.overview


def test_summarize_file_rewrites_markdown(tmp_path: Path) -> None:
    mic = tmp_path / "only.wav"
    _write_silence(mic)
    first = run_meeting(
        AppConfig(output_dir=tmp_path),
        mic=mic,
        output=tmp_path / "one.md",
        title="週會",
        asr=FakeAsr(),
        do_summary=False,
        do_polish=False,
        environ={},
    )
    second = summarize_file(
        AppConfig(),
        first.json_path,
        summarizer=FakeSummarizer(),
        environ={},
    )
    assert second.markdown_path == first.markdown_path
    assert "週會 由 Me 主持。" in second.markdown_path.read_text(encoding="utf-8")
    assert load_document(second.json_path).summary.decisions == ("改用 gRPC。",)
    assert isinstance(second.document.started_at, datetime)

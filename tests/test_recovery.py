import json
from dataclasses import replace

import pytest

from meetscribe.config import AppConfig
from meetscribe.document import load_document
from meetscribe.errors import MeetscribeError
from meetscribe.pipeline import run_meeting
from meetscribe.types import MeetingSummary
from test_pipeline import FakeAsr, _write_silence


class CountingAsr(FakeAsr):
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def transcribe(self, audio, *, track_id, **kwargs):
        self.calls.append(track_id)
        if track_id == self.fail:
            raise MeetscribeError("ASR failed")
        return super().transcribe(audio, track_id=track_id, **kwargs)


class Summary:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def summarize(self, utterances, **kwargs):
        self.calls += 1
        if self.fail:
            raise MeetscribeError("summary failed")
        return MeetingSummary("完成")


def setup(tmp_path):
    mic, system = tmp_path / "mic.wav", tmp_path / "system.wav"
    _write_silence(mic)
    _write_silence(system)
    return dict(
        config=AppConfig(),
        mic=mic,
        system=system,
        output=tmp_path / "note.md",
        work_dir=tmp_path / "work",
        do_polish=False,
    )


def test_resume_preserves_first_track_after_second_track_fails(tmp_path):
    args = setup(tmp_path)
    with pytest.raises(MeetscribeError, match="ASR failed"):
        run_meeting(**args, asr=CountingAsr("system"), do_summary=False)
    asr = CountingAsr()
    result = run_meeting(**args, asr=asr, do_summary=False, resume=True)
    assert asr.calls == ["system"]
    assert len(result.document.utterances) == 2


def test_summary_failure_keeps_readable_transcript_and_resume_skips_asr(tmp_path):
    args = setup(tmp_path)
    with pytest.raises(MeetscribeError, match="summary failed"):
        run_meeting(**args, asr=CountingAsr(), summarizer=Summary(True))
    document = load_document(tmp_path / "note.json")
    assert len(document.utterances) == 2
    assert "尚未摘要" in document.summary.overview
    asr = CountingAsr()
    result = run_meeting(**args, asr=asr, summarizer=Summary(), resume=True)
    assert asr.calls == []
    assert result.document.summary.overview == "完成"


def test_changed_asr_settings_invalidate_checkpoint(tmp_path):
    args = setup(tmp_path)
    run_meeting(**args, asr=CountingAsr(), do_summary=False)
    args["config"] = replace(args["config"], asr=replace(args["config"].asr, language="en"))
    asr = CountingAsr()
    run_meeting(**args, asr=asr, do_summary=False, resume=True)
    assert asr.calls == ["mic", "system"]


def test_resume_rejects_missing_corrupt_and_wrong_version(tmp_path):
    args = setup(tmp_path)
    with pytest.raises(MeetscribeError, match="checkpoint"):
        run_meeting(**args, resume=True)
    work = args["work_dir"]
    work.mkdir(exist_ok=True)
    path = work / "checkpoint.json"
    path.write_text("{")
    with pytest.raises(MeetscribeError, match="checkpoint"):
        run_meeting(**args, resume=True)
    path.write_text(json.dumps({"version": 999, "entries": {}}))
    with pytest.raises(MeetscribeError, match="version"):
        run_meeting(**args, resume=True)


def test_normal_run_does_not_reuse_completed_checkpoint(tmp_path):
    args = setup(tmp_path)
    run_meeting(**args, asr=CountingAsr(), do_summary=False)
    asr = CountingAsr()
    run_meeting(**args, asr=asr, do_summary=False)
    assert asr.calls == ["mic", "system"]


class Polisher:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def polish(self, utterances, **kwargs):
        self.calls += 1
        if self.fail:
            raise MeetscribeError("polish failed")
        return tuple(replace(item, polished_text="修訂文字") for item in utterances)


def test_polish_failure_keeps_raw_transcript_and_resumes(tmp_path):
    args = setup(tmp_path)
    args["do_polish"] = True
    with pytest.raises(MeetscribeError, match="polish failed"):
        run_meeting(**args, asr=CountingAsr(), polisher=Polisher(True), do_summary=False)
    assert load_document(tmp_path / "note.json").utterances[0].polished_text is None
    asr = CountingAsr()
    result = run_meeting(**args, asr=asr, polisher=Polisher(), do_summary=False, resume=True)
    assert not asr.calls
    assert result.document.utterances[0].polished_text == "修訂文字"


def test_title_and_speaker_changes_only_repeat_summary(tmp_path):
    args = setup(tmp_path)
    first = Summary()
    run_meeting(**args, asr=CountingAsr(), summarizer=first)
    args["config"].speaker_map["Me"] = "新名字"
    second, asr = Summary(), CountingAsr()
    result = run_meeting(**args, asr=asr, summarizer=second, title="新標題", resume=True)
    assert second.calls == 1
    assert not asr.calls
    assert result.document.title == "新標題"
    assert result.document.label_for("Me") == "新名字"


def test_changed_input_only_retranscribes_changed_track(tmp_path):
    args = setup(tmp_path)
    run_meeting(**args, asr=CountingAsr(), do_summary=False)
    _write_silence(args["system"], seconds=0.8)
    asr = CountingAsr()
    run_meeting(**args, asr=asr, do_summary=False, resume=True)
    assert asr.calls == ["system"]


def test_work_directory_cannot_overwrite_source(tmp_path):
    args = setup(tmp_path)
    audio_dir = args["work_dir"] / "audio"
    audio_dir.mkdir(parents=True)
    args["mic"] = audio_dir / "mic.wav"
    _write_silence(args["mic"])
    before = args["mic"].read_bytes()
    with pytest.raises(MeetscribeError, match="overwrite source"):
        run_meeting(**args, asr=CountingAsr(), do_summary=False)
    assert args["mic"].read_bytes() == before


def test_checkpoint_rejects_concurrent_use_and_invalid_stage(tmp_path):
    from meetscribe.checkpoint import Checkpoint

    work = tmp_path / "work"
    checkpoint = Checkpoint(work, resume=False)
    try:
        with pytest.raises(MeetscribeError, match="already in use"):
            Checkpoint(work, resume=False)
    finally:
        checkpoint.close()
    (work / "checkpoint.json").write_text(
        json.dumps({"version": 1, "entries": {"polish": {"key": "abc", "value": 42}}})
    )
    with pytest.raises(MeetscribeError, match="Invalid checkpoint"):
        Checkpoint(work, resume=True)


@pytest.mark.parametrize(
    "record",
    [
        {"_type": "MeetingSummary", "fields": "bad"},
        {"_type": "MeetingSummary", "fields": {"overview": "ok", "key_points": 4}},
        {"_type": "MeetingSummary", "fields": {"overview": 5}},
    ],
)
def test_malformed_checkpoint_records_are_actionable_errors(tmp_path, record):
    from meetscribe.checkpoint import Checkpoint

    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps({"version": 1, "entries": {"summary": {"key": "abc", "value": record}}})
    )
    with pytest.raises(MeetscribeError, match="Invalid checkpoint"):
        Checkpoint(tmp_path, resume=True)


@pytest.mark.parametrize("alias", ["same", "hardlink"])
def test_output_cannot_overwrite_source(tmp_path, alias):
    args = setup(tmp_path)
    before = args["mic"].read_bytes()
    if alias == "same":
        args["output"] = args["mic"]
    else:
        args["output"] = tmp_path / "linked.md"
        args["output"].hardlink_to(args["mic"])
    with pytest.raises(MeetscribeError, match="Output would overwrite source"):
        run_meeting(**args, asr=CountingAsr(), do_summary=False)
    assert args["mic"].read_bytes() == before


def test_failing_initial_event_sink_releases_checkpoint_lock(tmp_path):
    from meetscribe.checkpoint import Checkpoint

    class BadSink:
        def emit(self, event):
            raise RuntimeError("sink failed")

    args = setup(tmp_path)
    with pytest.raises(RuntimeError, match="sink failed"):
        run_meeting(**args, sink=BadSink())
    checkpoint = Checkpoint(args["work_dir"], resume=True)
    checkpoint.close()


def test_keep_audio_preserves_existing_source_session(tmp_path):
    args = setup(tmp_path)
    source = tmp_path / "note.audio"
    source.mkdir()
    args["mic"].rename(source / "mic.wav")
    args["system"].rename(source / "system.wav")
    args["mic"] = source
    args["system"] = None
    marker = source / "session.json"
    marker.write_text("original recording")
    before = (source / "mic.wav").read_bytes()
    with pytest.raises(MeetscribeError, match="Audio output already exists"):
        run_meeting(**args, asr=CountingAsr(), do_summary=False, keep_audio=True)
    assert marker.read_text() == "original recording"
    assert (source / "mic.wav").read_bytes() == before


def test_keep_audio_cannot_copy_into_checkpoint_tree(tmp_path):
    args = setup(tmp_path)
    args["output"] = args["work_dir"] / "note.md"
    with pytest.raises(MeetscribeError, match="overlaps the work directory"):
        run_meeting(**args, asr=CountingAsr(), do_summary=False, keep_audio=True)
    assert (args["work_dir"] / "checkpoint.json").is_file()

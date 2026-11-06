import importlib.util
from datetime import datetime
from pathlib import Path

from meetscribe.render.markdown import group_utterances, render_markdown
from meetscribe.types import ActionItem, AttributedUtterance, MeetingDocument, MeetingSummary


def _utterance(
    speaker: str,
    text: str,
    start: float,
    end: float,
    *,
    overlap: bool = False,
    polished: str | None = None,
) -> AttributedUtterance:
    return AttributedUtterance(
        speaker_id=speaker,
        text=text,
        polished_text=polished,
        start=start,
        end=end,
        track_id="mic",
        overlap=overlap,
    )


def _document() -> MeetingDocument:
    return MeetingDocument(
        title="週會",
        started_at=datetime.fromisoformat("2026-04-14T15:00:00+08:00"),
        summary=MeetingSummary(
            overview="確認 gRPC 遷移。",
            key_points=("timeout 維持 800ms。",),
            decisions=("用 feature flag 上線。",),
            action_items=(
                ActionItem(task="補 runbook", owner="林可安", due="下週三"),
                ActionItem(task="看 dashboard", owner="陳怡君"),
            ),
        ),
        utterances=(
            _utterance("Me", "raw one", 0.0, 1.2, polished="第一句 API"),
            _utterance("Me", "raw two", 1.4, 3.0, polished="第二句"),
            _utterance("SPEAKER_00", "對方", 4.5, 6.0, overlap=True, polished="對方提到 rollback"),
        ),
        speaker_map={"Me": "林可安", "SPEAKER_00": "陳怡君"},
    )


def test_section_order_and_mapped_labels() -> None:
    text = render_markdown(_document())
    headings = ["# 週會", "## 摘要", "### 重點", "### 決議", "### 待辦", "## 逐字稿"]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "2026-04-14 15:00" in text
    assert "林可安、陳怡君" in text
    assert "確認 gRPC 遷移。" in text
    assert "timeout 維持 800ms。" in text
    assert "用 feature flag 上線。" in text
    assert "- 補 runbook（林可安） — 下週三\n- 看 dashboard（陳怡君）" in text
    assert "### 林可安（00:00:00–00:00:03）" in text
    assert "第一句 API\n第二句" in text
    assert "### 陳怡君（00:00:04–00:00:06） · 重疊" in text
    assert "raw one" not in text


def test_group_breaks_on_speaker_change_and_long_gap() -> None:
    utterances = [
        _utterance("A", "一", 0.0, 1.0),
        _utterance("A", "二", 1.2, 2.0),
        _utterance("B", "三", 2.1, 3.0),
        _utterance("B", "四", 6.0, 7.0),
    ]
    turns = group_utterances(utterances, gap_s=1.0)
    assert [(turn.speaker_id, turn.text) for turn in turns] == [
        ("A", "一\n二"),
        ("B", "三"),
        ("B", "四"),
    ]


def test_empty_summary_lists() -> None:
    document = MeetingDocument(
        title="空白",
        started_at=datetime(2026, 1, 1, 9, 0),
        summary=MeetingSummary(overview="沒有結論。"),
        utterances=(),
    )
    text = render_markdown(document)
    assert "（未標註）" in text
    assert text.count("- （無）") == 3
    assert "（無逐字稿）" in text


def test_checked_in_sample_matches_renderer() -> None:
    script = Path("examples/sample_meeting.py")
    spec = importlib.util.spec_from_file_location("sample_meeting", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = render_markdown(module.build_document())
    assert script.with_name("sample_meeting.md").read_text(encoding="utf-8") == expected

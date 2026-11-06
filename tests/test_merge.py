from meetscribe.merge import UNKNOWN_SPEAKER, assign_speakers
from meetscribe.types import SpeakerTurn, TranscriptSegment, WordTimestamp


def _segment(text: str, start: float, end: float, track: str = "mic") -> TranscriptSegment:
    return TranscriptSegment(
        text=text,
        start=start,
        end=end,
        track_id=track,
        words=(WordTimestamp(text=text, start=start, end=end),),
    )


def test_majority_overlap_picks_the_longer_turn() -> None:
    turns = [
        SpeakerTurn("A", 0.0, 2.0),
        SpeakerTurn("B", 2.0, 10.0),
    ]
    assigned = assign_speakers([_segment("討論 API", 0.0, 10.0)], turns)
    assert assigned[0].speaker_id == "B"
    assert assigned[0].overlap is False
    assert assigned[0].words[0].text == "討論 API"


def test_second_speaker_marks_overlap() -> None:
    turns = [
        SpeakerTurn("A", 0.0, 6.0),
        SpeakerTurn("B", 6.0, 10.0),
    ]
    assigned = assign_speakers([_segment("同時說話", 0.0, 10.0)], turns)
    assert assigned[0].speaker_id == "A"
    assert assigned[0].overlap is True


def test_turn_overlap_flag_is_kept() -> None:
    turns = [SpeakerTurn("A", 0.0, 5.0, overlap=True)]
    assigned = assign_speakers([_segment("嗯", 1.0, 2.0)], turns)
    assert assigned[0].overlap is True


def test_tie_keeps_the_earlier_turn() -> None:
    turns = [
        SpeakerTurn("A", 0.0, 5.0),
        SpeakerTurn("B", 0.0, 5.0),
    ]
    assigned = assign_speakers([_segment("平手", 0.0, 5.0)], turns)
    assert assigned[0].speaker_id == "A"


def test_track_constraint() -> None:
    turns = [
        SpeakerTurn("Me", 0.0, 10.0, track_id="mic"),
        SpeakerTurn("Remote", 0.0, 10.0, track_id="system"),
    ]
    segments = [
        _segment("我這邊", 1.0, 2.0, "mic"),
        _segment("對方", 1.0, 2.0, "system"),
    ]
    assigned = assign_speakers(segments, turns)
    by_track = {item.track_id: item.speaker_id for item in assigned}
    assert by_track == {"mic": "Me", "system": "Remote"}
    assert all(item.overlap is False for item in assigned)


def test_unconstrained_turn_matches_any_track() -> None:
    turns = [SpeakerTurn("SPEAKER_00", 0.0, 4.0, track_id=None)]
    assigned = assign_speakers([_segment("系統", 0.5, 1.5, "system")], turns)
    assert assigned[0].speaker_id == "SPEAKER_00"


def test_no_overlap_is_unknown() -> None:
    turns = [SpeakerTurn("A", 5.0, 6.0)]
    assigned = assign_speakers([_segment("空白", 0.0, 1.0)], turns)
    assert assigned[0].speaker_id == UNKNOWN_SPEAKER
    assert assigned[0].overlap is False


def test_touching_boundary_does_not_count() -> None:
    turns = [SpeakerTurn("A", 2.0, 3.0)]
    assigned = assign_speakers([_segment("邊界", 1.0, 2.0)], turns)
    assert assigned[0].speaker_id == UNKNOWN_SPEAKER


def test_blank_text_is_dropped_and_results_are_sorted() -> None:
    turns = [SpeakerTurn("A", 0.0, 10.0, track_id=None)]
    segments = [
        _segment("後句", 4.0, 5.0, "system"),
        _segment("   ", 0.0, 1.0, "mic"),
        _segment("前句", 1.0, 2.0, "mic"),
    ]
    assigned = assign_speakers(segments, turns)
    assert [item.text for item in assigned] == ["前句", "後句"]

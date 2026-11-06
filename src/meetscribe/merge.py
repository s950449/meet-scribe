"""Assign a speaker to each ASR segment.

The winning turn is the one with the greatest time overlap. Turns that name a
track only match segments from that track, so a mic utterance is not given to
the system-audio speaker. Ties keep the earlier turn. A segment with no
positive overlap becomes ``UNKNOWN``.

``overlap`` on the utterance is set when the chosen turn says so, or when a
second speaker also covers at least ``overlap_ratio`` of the segment.
"""

from __future__ import annotations

from collections.abc import Sequence

from meetscribe.types import AttributedUtterance, SpeakerTurn, TranscriptSegment

UNKNOWN_SPEAKER = "UNKNOWN"


def assign_speakers(
    segments: Sequence[TranscriptSegment],
    turns: Sequence[SpeakerTurn],
    *,
    overlap_ratio: float = 0.3,
) -> list[AttributedUtterance]:
    assigned: list[AttributedUtterance] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        ranked = _ranked_turns(segment, turns)
        if not ranked:
            speaker_id = UNKNOWN_SPEAKER
            is_overlap = False
        else:
            _best_overlap, best_turn = ranked[0]
            speaker_id = best_turn.speaker_id
            duration = max(segment.end - segment.start, 1e-6)
            second_ratio = ranked[1][0] / duration if len(ranked) > 1 else 0.0
            is_overlap = best_turn.overlap or second_ratio >= overlap_ratio
        assigned.append(
            AttributedUtterance(
                speaker_id=speaker_id,
                text=text,
                start=segment.start,
                end=segment.end,
                track_id=segment.track_id,
                overlap=is_overlap,
                words=segment.words,
            )
        )
    assigned.sort(key=lambda utterance: (utterance.start, utterance.track_id, utterance.speaker_id))
    return assigned


def _ranked_turns(
    segment: TranscriptSegment,
    turns: Sequence[SpeakerTurn],
) -> list[tuple[float, SpeakerTurn]]:
    ranked: list[tuple[float, SpeakerTurn]] = []
    for turn in turns:
        if turn.track_id is not None and turn.track_id != segment.track_id:
            continue
        overlap = _overlap(segment.start, segment.end, turn.start, turn.end)
        if overlap > 0:
            ranked.append((overlap, turn))
    # Stable sort: equal overlap and equal turn length keep diarizer order.
    ranked.sort(key=lambda item: (item[0], item[1].end - item[1].start), reverse=True)
    return ranked


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))

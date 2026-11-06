"""Shared ASR prompt and segment shaping."""

from __future__ import annotations

from meetscribe.types import TranscriptSegment, WordTimestamp

INITIAL_PROMPT = "以下是繁體中文與英文夾雜的會議，技術術語保留英文。"


def blank_language(language: str | None) -> str | None:
    if language is None:
        return None
    if language.strip().lower() in {"", "auto", "detect"}:
        return None
    return language.strip()


def build_segment(
    text: str,
    start: float,
    end: float,
    track_id: str,
    words: tuple[WordTimestamp, ...] = (),
    language: str | None = None,
) -> TranscriptSegment | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    if end < start:
        end = start
    return TranscriptSegment(
        text=cleaned,
        start=float(start),
        end=float(end),
        track_id=track_id,
        words=words,
        language=language,
    )


def join_text(parts: list[str]) -> str:
    """Join ASR pieces without inserting spaces between CJK characters."""

    text = ""
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if not text:
            text = piece
            continue
        if _cjk(text[-1]) and _cjk(piece[0]):
            text += piece
        else:
            text += " " + piece
    return text


def coalesce(segments: list[TranscriptSegment], *, gap: float = 0.8) -> list[TranscriptSegment]:
    """Merge adjacent word-sized segments into phrases."""

    if not segments:
        return []
    groups: list[list[TranscriptSegment]] = [[segments[0]]]
    for segment in segments[1:]:
        previous = groups[-1][-1]
        if segment.track_id == previous.track_id and segment.start - previous.end <= gap:
            groups[-1].append(segment)
        else:
            groups.append([segment])
    merged: list[TranscriptSegment] = []
    for group in groups:
        words = tuple(word for item in group for word in item.words)
        built = build_segment(
            join_text([item.text for item in group]),
            group[0].start,
            group[-1].end,
            group[0].track_id,
            words,
            group[0].language,
        )
        if built is not None:
            merged.append(built)
    return merged


def _cjk(char: str) -> bool:
    code = ord(char)
    return 0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF or 0x3040 <= code <= 0x30FF

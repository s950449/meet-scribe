"""Group utterances into speaker turns and render the meeting note."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from meetscribe.timestamps import format_timestamp
from meetscribe.types import ActionItem, AttributedUtterance, MeetingDocument

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
DEFAULT_GAP_S = 1.0


@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    speaker_id: str
    speaker_label: str
    start: str
    end: str
    text: str
    overlap: bool

    @property
    def heading(self) -> str:
        marker = " · 重疊" if self.overlap else ""
        return f"{self.speaker_label}（{self.start}–{self.end}）{marker}"


def group_utterances(
    utterances: tuple[AttributedUtterance, ...] | list[AttributedUtterance],
    speaker_map: dict[str, str] | None = None,
    *,
    gap_s: float = DEFAULT_GAP_S,
) -> list[TranscriptTurn]:
    """Merge consecutive lines from the same speaker when the gap is short."""

    labels = speaker_map or {}
    ordered = sorted(utterances, key=lambda item: (item.start, item.track_id, item.speaker_id))
    grouped: list[list[AttributedUtterance]] = []
    for utterance in ordered:
        if not grouped:
            grouped.append([utterance])
            continue
        previous = grouped[-1][-1]
        same_speaker = previous.speaker_id == utterance.speaker_id
        close_enough = utterance.start - previous.end <= gap_s
        if same_speaker and close_enough:
            grouped[-1].append(utterance)
        else:
            grouped.append([utterance])
    return [_to_turn(chunk, labels) for chunk in grouped]


def render_markdown(document: MeetingDocument, *, gap_s: float = DEFAULT_GAP_S) -> str:
    turns = group_utterances(document.utterances, document.speaker_map, gap_s=gap_s)
    template = _environment().get_template("meeting.md.j2")
    return template.render(
        title=document.title,
        started_at=document.started_at.strftime("%Y-%m-%d %H:%M"),
        participants=document.participants(),
        summary=document.summary,
        action_items=tuple(format_action_item(item) for item in document.summary.action_items),
        turns=turns,
    )


def format_action_item(item: ActionItem) -> str:
    text = item.task
    if item.owner:
        text += f"（{item.owner}）"
    if item.due:
        text += f" — {item.due}"
    return text


def _to_turn(chunk: list[AttributedUtterance], speaker_map: dict[str, str]) -> TranscriptTurn:
    first = chunk[0]
    last = chunk[-1]
    text = "\n".join(item.display_text.strip() for item in chunk if item.display_text.strip())
    return TranscriptTurn(
        speaker_id=first.speaker_id,
        speaker_label=speaker_map.get(first.speaker_id, first.speaker_id),
        start=format_timestamp(first.start),
        end=format_timestamp(last.end),
        text=text,
        overlap=any(item.overlap for item in chunk),
    )


def _environment() -> Environment:
    if not TEMPLATE_DIR.is_dir():
        raise FileNotFoundError(f"Markdown templates are missing: {TEMPLATE_DIR}")
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )

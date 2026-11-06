"""In-process progress events. A GUI subscribes here; the CLI can also print JSONL."""

from __future__ import annotations

import json
import sys
from typing import TextIO

from meetscribe.types import ProgressEvent


def stage_event(
    stage: str,
    status: str,
    *,
    message: str | None = None,
    track_id: str | None = None,
    progress: float | None = None,
    **data: object,
) -> ProgressEvent:
    payload = tuple(sorted(data.items()))
    return ProgressEvent(
        stage=stage,
        status=status,
        message=message,
        event="stage",
        track_id=track_id,
        progress=progress,
        data=payload,
    )


def log_event(stage: str, message: str, **data: object) -> ProgressEvent:
    payload = tuple(sorted(data.items()))
    return ProgressEvent(
        stage=stage,
        message=message,
        event="log",
        data=payload,
    )


class NullSink:
    def emit(self, event: ProgressEvent) -> None:
        del event


class StdioSink:
    """Human lines on a text stream, and optional JSONL on stdout.

    JSONL goes to stdout so a parent process can parse it. Human text goes to
    stderr in that mode, which keeps the stream free of log lines.
    """

    def __init__(
        self,
        *,
        json_events: bool,
        json_stream: TextIO | None = None,
        human_stream: TextIO | None = None,
    ) -> None:
        self.json_events = json_events
        self.json_stream = json_stream if json_stream is not None else sys.stdout
        self.human_stream = human_stream if human_stream is not None else (
            sys.stderr if json_events else sys.stdout
        )

    def emit(self, event: ProgressEvent) -> None:
        if self.json_events:
            print(
                json.dumps(event.to_dict(), ensure_ascii=False),
                file=self.json_stream,
                flush=True,
            )
        if event.message:
            print(event.message, file=self.human_stream, flush=True)

"""Shared timestamp formatting for transcripts and summaries."""


def format_timestamp(seconds: float) -> str:
    """Format seconds on the capture timebase as ``HH:MM:SS``."""

    if seconds < 0:
        seconds = 0.0
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

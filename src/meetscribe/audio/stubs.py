"""Platform capture backends.

Each class records the planned native API and raises until that integration
lands. Lazy imports stay inside ``record`` so importing the CLI does not load
Core Audio, WASAPI, or Pulse.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from meetscribe.errors import BackendNotImplemented
from meetscribe.types import CaptureCapabilities, CaptureResult, ProgressEvent


class _PlannedCapture:
    name = "capture"
    device = "unknown"
    summary = "Capture is not implemented."

    def capabilities(self) -> CaptureCapabilities:
        return CaptureCapabilities(
            mic=True,
            system_audio=True,
            separate_tracks=True,
            device=self.device,
        )

    def record(
        self,
        output_dir: Path,
        *,
        duration_s: float | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> CaptureResult:
        del output_dir, duration_s, on_event
        self._touch_native_api()
        raise BackendNotImplemented(self.summary)

    def _touch_native_api(self) -> None:
        """Import the planned binding. Subclasses override this."""


class MacOSCapture(_PlannedCapture):
    """ScreenCaptureKit for system audio plus a Core Audio tap for the mic.

    The two writers will share a host-time clock and flush ``mic.wav`` and
    ``system.wav``. ScreenCaptureKit needs the screen-recording entitlement.
    """

    name = "macos"
    device = "screencapturekit"
    summary = (
        "macOS capture is not implemented yet. "
        "Planned: ScreenCaptureKit for system audio and a Core Audio process tap "
        "for the microphone, written to mic.wav and system.wav on one timebase. "
        "Transcribe existing files with `meetscribe transcribe <mic.wav> --system <system.wav>`."
    )

    def _touch_native_api(self) -> None:
        # ScreenCaptureKit is Objective-C. The binding (pyobjc) is imported only
        # when recording is attempted, and is not a dependency of this skeleton.
        try:
            import ScreenCaptureKit  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            return


class WindowsCapture(_PlannedCapture):
    """WASAPI loopback for system audio and an eCapture/WASAPI mic client."""

    name = "windows"
    device = "wasapi"
    summary = (
        "Windows capture is not implemented yet. "
        "Planned: WASAPI loopback for system audio and a WASAPI capture client "
        "for the microphone, written to mic.wav and system.wav on one timebase. "
        "Transcribe existing files with `meetscribe transcribe <mic.wav> --system <system.wav>`."
    )

    def _touch_native_api(self) -> None:
        try:
            import comtypes  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            return


class LinuxCapture(_PlannedCapture):
    """PulseAudio or PipeWire monitor source, plus the default mic source."""

    name = "linux"
    device = "pipewire"
    summary = (
        "Linux capture is not implemented yet. "
        "Planned: a PulseAudio/PipeWire monitor source for system audio and the "
        "default source for the microphone, written to mic.wav and system.wav on one timebase. "
        "Transcribe existing files with `meetscribe transcribe <mic.wav> --system <system.wav>`."
    )

    def _touch_native_api(self) -> None:
        try:
            import pulsectl  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            return

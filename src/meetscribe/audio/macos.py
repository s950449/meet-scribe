"""macOS capture: Core Audio mic plus a ScreenCaptureKit system-audio helper.

ScreenCaptureKit's audio callback is a completion-handler API that PyObjC only
wraps awkwardly, and the process that calls it must hold the Screen Recording
permission. A small Swift helper (``scripts/macos/SystemAudioCapture.swift``)
is the capture path: it writes 48 kHz s16le mono PCM to stdout, and this module
stamps each read with ``time.monotonic()``. The microphone stays in-process
through sounddevice so both tracks share that clock. Pipe delay is a few
milliseconds and is absorbed by the same gap/drop alignment as device drift.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from meetscribe.audio.devices import AudioDevice
from meetscribe.audio.record import run_synced_recording
from meetscribe.audio.sync import ChunkCallback, SyncedWavSession
from meetscribe.errors import AudioError, BackendNotInstalled
from meetscribe.types import CaptureCapabilities, CaptureResult, ProgressEvent

SYSTEM_DEVICE = AudioDevice(id="screencapturekit", name="ScreenCaptureKit", kind="monitor")


class MacOSCapture:
    name = "macos"

    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        mic_device: str = "",
        system_device: str = "",
        chunk_seconds: float = 5.0,
        helper: str = "",
        mic_opener: Callable | None = None,
        system_opener: Callable | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.mic_device = mic_device
        self.system_device = system_device
        self.chunk_seconds = chunk_seconds
        self.helper = helper
        self._mic_opener = mic_opener
        self._system_opener = system_opener

    def capabilities(self) -> CaptureCapabilities:
        return CaptureCapabilities(
            mic=True,
            system_audio=True,
            separate_tracks=True,
            device="screencapturekit",
        )

    def record(
        self,
        output_dir,
        *,
        duration_s: float | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
        on_chunk: ChunkCallback | None = None,
    ) -> CaptureResult:
        mic = AudioDevice(id=self.mic_device or "default", name=self.mic_device or "default input", kind="mic")
        system = SYSTEM_DEVICE
        if self.system_device:
            system = AudioDevice(id=self.system_device, name=self.system_device, kind="monitor")

        def opener(device: AudioDevice, track_id: str, session: SyncedWavSession, sample_rate: int):
            if track_id == "system":
                if self._system_opener is not None:
                    return self._system_opener(device, track_id, session, sample_rate)
                return _SystemHelper(self._helper_path(), session, track_id, sample_rate)
            if self._mic_opener is not None:
                return self._mic_opener(device, track_id, session, sample_rate)
            return _MicStream(device, track_id, session, sample_rate)

        return run_synced_recording(
            output_dir,
            devices=[("mic", mic), ("system", system)],
            opener=opener,
            sample_rate=self.sample_rate,
            duration_s=duration_s,
            chunk_seconds=self.chunk_seconds,
            on_chunk=on_chunk,
            on_event=on_event,
        )

    def _helper_path(self) -> str:
        configured = self.helper.strip() or os.environ.get("MEETSCRIBE_SYSTEM_AUDIO_BIN", "").strip()
        if configured and Path(configured).expanduser().is_file():
            return str(Path(configured).expanduser())
        found = shutil.which("meetscribe-system-audio")
        if found:
            return found
        raise AudioError(
            "macOS system audio needs the ScreenCaptureKit helper. From the repo:\n"
            "  swiftc -O -o ~/.local/bin/meetscribe-system-audio scripts/macos/SystemAudioCapture.swift\n"
            "Grant Screen Recording permission to the terminal, then record again. "
            "The microphone still uses Core Audio through sounddevice "
            "(pip install 'meetscribe[capture-macos]')."
        )


class _MicStream:
    def __init__(self, device: AudioDevice, track_id: str, session: SyncedWavSession, sample_rate: int) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise BackendNotInstalled(
                "sounddevice is not installed. On macOS: pip install 'meetscribe[capture-macos]'"
            ) from exc
        resolved: object = None if device.id in {"", "default"} else device.id
        if isinstance(resolved, str) and resolved.isdigit():
            resolved = int(resolved)

        def callback(indata, frames, time_info, status):
            del time_info, status
            host = time.monotonic() - frames / sample_rate
            session.write(track_id, host, indata[:, 0].copy(), sample_rate)

        self._stream = sd.InputStream(
            device=resolved,
            channels=1,
            samplerate=sample_rate,
            dtype="float32",
            callback=callback,
            blocksize=max(1, sample_rate // 10),
        )

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


HELPER_RATE = 48_000


class _SystemHelper:
    """Read s16le mono PCM from the Swift helper. The helper always writes 48 kHz."""

    def __init__(self, binary: str, session: SyncedWavSession, track_id: str, sample_rate: int) -> None:
        del sample_rate
        self._session = session
        self._track_id = track_id
        self._stop = False
        try:
            self._proc = subprocess.Popen(
                [binary],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise AudioError(f"Could not start {binary}: {exc}") from exc
        self._thread = None

    def start(self) -> None:
        import threading

        self._thread = threading.Thread(target=self._read, name="meetscribe-system-audio", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop = True
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)
        stderr = ""
        if self._proc.stderr is not None:
            stderr = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
        frames = self._session.stats().get(self._track_id)
        # SIGTERM is -15. A crash before any PCM is a permission or helper failure.
        if (frames is None or frames.frames == 0) and self._proc.returncode not in (None, 0, -15):
            raise AudioError(stderr or f"ScreenCaptureKit helper exited {self._proc.returncode}")

    def _read(self) -> None:
        assert self._proc.stdout is not None
        frame_bytes = int(HELPER_RATE / 10) * 2
        while not self._stop:
            chunk = self._proc.stdout.read(frame_bytes)
            if not chunk:
                break
            pcm = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
            host = time.monotonic() - pcm.size / HELPER_RATE
            self._session.write(self._track_id, host, pcm, HELPER_RATE)

"""PulseAudio / PipeWire capture: default mic plus the sink monitor."""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable

from meetscribe.audio.devices import AudioDevice, choose_linux_devices, parse_pactl_short
from meetscribe.audio.record import StreamOpener, run_synced_recording
from meetscribe.audio.sync import ChunkCallback, SyncedWavSession
from meetscribe.errors import AudioError, BackendNotInstalled
from meetscribe.types import CaptureCapabilities, CaptureResult, ProgressEvent


class LinuxCapture:
    """Record the microphone and the default sink's ``.monitor`` source.

    PortAudio (via sounddevice) talks to PulseAudio, which on current desktops
    is the PipeWire compatibility layer. Device names come from ``pactl``.
    """

    name = "linux"

    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        mic_device: str = "",
        system_device: str = "",
        chunk_seconds: float = 5.0,
        source_lister: Callable[[], list[AudioDevice]] | None = None,
        stream_opener: StreamOpener | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.mic_device = mic_device
        self.system_device = system_device
        self.chunk_seconds = chunk_seconds
        self._source_lister = source_lister
        self._stream_opener = stream_opener

    def capabilities(self) -> CaptureCapabilities:
        return CaptureCapabilities(
            mic=True,
            system_audio=True,
            separate_tracks=True,
            device="pipewire",
        )

    def list_sources(self) -> list[AudioDevice]:
        if self._source_lister is not None:
            return self._source_lister()
        return list_pulse_sources()

    def record(
        self,
        output_dir,
        *,
        duration_s: float | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
        on_chunk: ChunkCallback | None = None,
    ) -> CaptureResult:
        mic, system = choose_linux_devices(
            self.list_sources(),
            mic_device=self.mic_device,
            system_device=self.system_device,
        )
        opener = self._stream_opener or _open_sounddevice
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


def list_pulse_sources() -> list[AudioDevice]:
    pactl = shutil.which("pactl")
    if pactl is None:
        return _sources_from_sounddevice()
    try:
        completed = subprocess.run(
            [pactl, "list", "short", "sources"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioError(f"pactl failed: {exc}") from exc
    if completed.returncode != 0:
        raise AudioError(completed.stderr.strip() or "pactl list short sources failed")
    sources = parse_pactl_short(completed.stdout)
    if sources:
        return sources
    return _sources_from_sounddevice()


def _sources_from_sounddevice() -> list[AudioDevice]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise BackendNotInstalled(
            "Linux capture needs PulseAudio/PipeWire (`pactl`) or sounddevice. "
            "pip install 'meetscribe[capture-linux]'"
        ) from exc
    devices: list[AudioDevice] = []
    for index, info in enumerate(sd.query_devices()):
        if int(info.get("max_input_channels") or 0) < 1:
            continue
        name = str(info["name"])
        kind = "monitor" if "monitor" in name.lower() else "mic"
        devices.append(AudioDevice(id=str(index), name=name, kind=kind))
    return devices


def _open_sounddevice(device: AudioDevice, track_id: str, session: SyncedWavSession, sample_rate: int):
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise BackendNotInstalled(
            "sounddevice is not installed. pip install 'meetscribe[capture-linux]'"
        ) from exc
    resolved = _resolve_input(sd, device)

    class _Stream:
        def __init__(self) -> None:
            def callback(indata, frames, time_info, status):
                del time_info, status
                samples = indata[:, 0].copy()
                host = time.monotonic() - frames / sample_rate
                session.write(track_id, host, samples, sample_rate)

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

    return _Stream()


def _resolve_input(sd, device: AudioDevice):
    query = device.name
    try:
        queried = sd.query_devices()
    except Exception:
        return query
    for index, info in enumerate(queried):
        if int(info.get("max_input_channels") or 0) < 1:
            continue
        name = str(info["name"])
        if query == str(index) or query == name or query in name or name in query:
            return index
    return query

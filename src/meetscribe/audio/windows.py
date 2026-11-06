"""WASAPI loopback for system audio plus the default capture endpoint."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from meetscribe.audio.devices import AudioDevice, choose_wasapi_devices
from meetscribe.audio.record import StreamOpener, run_synced_recording
from meetscribe.audio.sync import ChunkCallback, SyncedWavSession
from meetscribe.errors import BackendNotInstalled
from meetscribe.types import CaptureCapabilities, CaptureResult, ProgressEvent


class WindowsCapture:
    """Record with PyAudioWPatch, which exposes WASAPI loopback devices.

    Loopback is the render endpoint heard by the speakers, so a meeting in the
    browser lands on ``system.wav`` while the headset mic lands on ``mic.wav``.
    """

    name = "windows"

    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        mic_device: str = "",
        system_device: str = "",
        chunk_seconds: float = 5.0,
        device_lister: Callable[[], tuple[list[AudioDevice], str, str]] | None = None,
        stream_opener: StreamOpener | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.mic_device = mic_device
        self.system_device = system_device
        self.chunk_seconds = chunk_seconds
        self._device_lister = device_lister
        self._stream_opener = stream_opener

    def capabilities(self) -> CaptureCapabilities:
        return CaptureCapabilities(
            mic=True,
            system_audio=True,
            separate_tracks=True,
            device="wasapi",
        )

    def list_devices(self) -> tuple[list[AudioDevice], str, str]:
        if self._device_lister is not None:
            return self._device_lister()
        return list_wasapi_devices()

    def record(
        self,
        output_dir,
        *,
        duration_s: float | None = None,
        on_event: Callable[[ProgressEvent], None] | None = None,
        on_chunk: ChunkCallback | None = None,
    ) -> CaptureResult:
        devices, default_input, default_output = self.list_devices()
        mic, system = choose_wasapi_devices(
            devices,
            mic_device=self.mic_device,
            system_device=self.system_device,
            default_input=default_input,
            default_output=default_output,
        )
        opener = self._stream_opener or _open_wasapi
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


def list_wasapi_devices() -> tuple[list[AudioDevice], str, str]:
    audio = _pyaudio()
    try:
        host = audio.get_host_api_info_by_type(audio.paWASAPI)
        default_input = ""
        default_output = ""
        input_index = host.get("defaultInputDevice")
        output_index = host.get("defaultOutputDevice")
        devices: list[AudioDevice] = []
        count = audio.get_device_count()
        for index in range(count):
            info = audio.get_device_info_by_index(index)
            if info.get("hostApi") != host.get("index"):
                continue
            name = str(info.get("name") or index)
            if info.get("isLoopbackDevice"):
                kind = "loopback"
            elif int(info.get("maxInputChannels") or 0) > 0:
                kind = "mic"
            else:
                kind = "output"
            devices.append(
                AudioDevice(
                    id=str(index),
                    name=name,
                    kind=kind,
                    sample_rate=int(info.get("defaultSampleRate") or 0) or None,
                    channels=int(info.get("maxInputChannels") or info.get("maxOutputChannels") or 0) or None,
                )
            )
            if index == input_index:
                default_input = name
            if index == output_index:
                default_output = name
        if not any(device.kind == "loopback" for device in devices):
            generator = getattr(audio, "get_loopback_device_info_generator", None)
            if callable(generator):
                for info in generator():
                    devices.append(
                        AudioDevice(
                            id=str(info.get("index")),
                            name=str(info.get("name") or "loopback"),
                            kind="loopback",
                            sample_rate=int(info.get("defaultSampleRate") or 0) or None,
                        )
                    )
        return devices, default_input, default_output
    finally:
        audio.terminate()


def _pyaudio():
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise BackendNotInstalled(
            "PyAudioWPatch is not installed. On Windows: uv sync --extra capture-windows"
        ) from exc
    return pyaudio.PyAudio()


def _open_wasapi(device: AudioDevice, track_id: str, session: SyncedWavSession, sample_rate: int):
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise BackendNotInstalled(
            "PyAudioWPatch is not installed. On Windows: uv sync --extra capture-windows"
        ) from exc
    audio = pyaudio.PyAudio()
    channels = device.channels or 1
    rate = device.sample_rate or sample_rate

    class _Stream:
        def __init__(self) -> None:
            def callback(in_data, frame_count, time_info, status):
                del time_info, status
                pcm = np.frombuffer(in_data, dtype="<i2")
                if channels > 1:
                    usable = pcm.size - (pcm.size % channels)
                    pcm = pcm[:usable].reshape(-1, channels).mean(axis=1)
                samples = pcm.astype(np.float32) / 32768.0
                host = time.monotonic() - samples.size / rate
                session.write(track_id, host, samples, rate)
                return (None, pyaudio.paContinue)

            self._audio = audio
            self._stream = audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=int(device.id),
                frames_per_buffer=max(1, rate // 10),
                stream_callback=callback,
            )

        def start(self) -> None:
            self._stream.start_stream()

        def close(self) -> None:
            self._stream.stop_stream()
            self._stream.close()
            self._audio.terminate()

    return _Stream()

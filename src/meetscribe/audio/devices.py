"""Pure device selection for PulseAudio, WASAPI, and listings."""

from __future__ import annotations

from dataclasses import dataclass

from meetscribe.errors import AudioError


@dataclass(frozen=True, slots=True)
class AudioDevice:
    id: str
    name: str
    kind: str  # mic | monitor | output | loopback
    sample_rate: int | None = None
    channels: int | None = None


def parse_pactl_short(text: str) -> list[AudioDevice]:
    """Parse ``pactl list short sources``."""

    devices: list[AudioDevice] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if len(parts) < 2:
            parts = stripped.split()
        if len(parts) < 2:
            continue
        index, name = parts[0], parts[1]
        kind = "monitor" if ".monitor" in name else "mic"
        devices.append(AudioDevice(id=index, name=name, kind=kind))
    return devices


def choose_linux_devices(
    sources: list[AudioDevice],
    *,
    mic_device: str = "",
    system_device: str = "",
) -> tuple[AudioDevice, AudioDevice]:
    mics = [device for device in sources if device.kind != "monitor"]
    monitors = [device for device in sources if device.kind == "monitor"]
    mic = _match(mics or sources, mic_device, "microphone")
    system = _match(monitors, system_device, "system monitor")
    if system is None:
        raise AudioError(
            "No PulseAudio/PipeWire monitor source was found. "
            "System audio is the sink's `.monitor` source. "
            "Run `pactl list short sources` and pass --system-device, "
            "or start the meeting app so a sink exists."
        )
    if mic is None:
        raise AudioError(
            "No microphone source was found. Pass --mic-device or check `pactl list short sources`."
        )
    return mic, system


def choose_wasapi_devices(
    devices: list[AudioDevice],
    *,
    mic_device: str = "",
    system_device: str = "",
    default_input: str = "",
    default_output: str = "",
) -> tuple[AudioDevice, AudioDevice]:
    """Select a capture device and a WASAPI loopback device.

    ``devices`` uses kind ``mic`` for inputs and ``loopback`` for loopback endpoints.
    """

    mics = [device for device in devices if device.kind == "mic"]
    loopbacks = [device for device in devices if device.kind == "loopback"]
    mic = _pick(mics, mic_device, default_input, "microphone")
    if system_device:
        system = _pick(loopbacks, system_device, "", "WASAPI loopback")
    elif default_output:
        named = [device for device in loopbacks if default_output.lower() in device.name.lower()]
        system = named[0] if named else (loopbacks[0] if loopbacks else None)
        if system is None:
            raise AudioError(
                "No WASAPI loopback device was found. Install PyAudioWPatch and check that an output device is active."
            )
    else:
        system = _pick(loopbacks, "", "", "WASAPI loopback")
    return mic, system


def _pick(devices: list[AudioDevice], query: str, fallback: str, label: str) -> AudioDevice:
    if query:
        matched = _match(devices, query, label)
        if matched is None:
            raise AudioError(f"No {label} was found. Run `meetscribe devices`.")
        return matched
    if fallback:
        matched = _find(devices, fallback)
        if matched is not None:
            return matched
    if devices:
        return devices[0]
    raise AudioError(f"No {label} was found. Run `meetscribe devices`.")


def _find(devices: list[AudioDevice], query: str) -> AudioDevice | None:
    needle = query.lower()
    for device in devices:
        if device.id == query or device.name == query:
            return device
    for device in devices:
        if needle in device.name.lower() or needle in device.id.lower():
            return device
    return None


def _match(devices: list[AudioDevice], query: str, label: str) -> AudioDevice | None:
    if not devices:
        return None
    if not query:
        return devices[0]
    needle = query.lower()
    for device in devices:
        if device.id == query or device.name == query:
            return device
    for device in devices:
        if needle in device.name.lower() or needle in device.id.lower():
            return device
    raise AudioError(
        f"No {label} matched {query!r}. Known: " + ", ".join(device.name for device in devices)
    )

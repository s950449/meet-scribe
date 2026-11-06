"""Platform facts used to pick default backends.

Detection is best-effort. Intel integrated GPU detection reads Linux sysfs and
does not load a GPU driver. ``None`` means the probe could not decide.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    system: str
    machine: str
    apple_silicon: bool
    intel_cpu: bool
    intel_igpu: bool | None
    vulkan_gpu: bool = False


def recommended_asr(info: HardwareInfo) -> str:
    """Default ASR backend name for this machine.

    Apple Silicon uses MLX. Intel CPUs with an integrated GPU use OpenVINO.
    Everything else stays on the CTranslate2 CPU backend.
    """

    if info.apple_silicon:
        return "mlx-whisper"
    if info.intel_igpu:
        return "openvino"
    return "faster-whisper"


def recommended_capture(info: HardwareInfo) -> str:
    return {
        "Darwin": "macos",
        "Windows": "windows",
        "Linux": "linux",
    }.get(info.system, "file")


def recommended_diarization(track_count: int) -> str:
    """Two tracks can separate the local mic from everyone else.

    A single mixed file has no track cue, so the honest local default is one
    speaker until a neural diarizer is selected.
    """

    if track_count > 1:
        return "track"
    return "single-speaker"


def detect_intel_igpu(sysfs: Path | None = None) -> bool | None:
    """Return True when a DRM device advertises PCI vendor 0x8086.

    Returns None when sysfs is absent (macOS, Windows, or a container without
    /sys) so callers can avoid treating "unknown" as "no iGPU".
    """

    root = sysfs if sysfs is not None else Path("/sys/class/drm")
    if not root.is_dir():
        return None
    cards = [path for path in root.glob("card*") if path.name[4:].isdigit()]
    if not cards:
        return False
    for card in cards:
        vendor_file = card / "device" / "vendor"
        try:
            vendor = vendor_file.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if vendor in {"0x8086", "8086"}:
            return True
    return False


def detect_vulkan_loader() -> bool:
    """Return True when a Vulkan loader or ICD is visible. This is not a GPU benchmark."""

    if shutil.which("vulkaninfo"):
        return True
    icd = Path("/usr/share/vulkan/icd.d")
    return bool(icd.is_dir() and any(icd.glob("*.json")))


def detect_hardware() -> HardwareInfo:
    system = platform.system()
    machine = platform.machine().lower()
    processor = platform.processor().lower()
    apple_silicon = system == "Darwin" and machine in {"arm64", "aarch64"}
    intel_cpu = machine in {"x86_64", "amd64", "i386", "x86", "i686"} or "intel" in processor
    intel_igpu = detect_intel_igpu() if system == "Linux" else None
    return HardwareInfo(
        system=system,
        machine=machine,
        apple_silicon=apple_silicon,
        intel_cpu=intel_cpu,
        intel_igpu=intel_igpu,
        vulkan_gpu=detect_vulkan_loader(),
    )

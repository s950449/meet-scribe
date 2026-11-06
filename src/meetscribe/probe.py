"""Decide whether a registered backend can run on this machine.

``implemented`` means the code path exists. ``usable`` also requires the
platform and the optional dependency. Reasons are shown by ``meetscribe backends``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from meetscribe.hardware import HardwareInfo, detect_vulkan_loader
from meetscribe.registry import BackendSpec


@dataclass(frozen=True, slots=True)
class BackendStatus:
    implemented: bool
    dependency_present: bool
    platform_ok: bool
    usable: bool
    reason: str


def platform_matches(spec: BackendSpec, hardware: HardwareInfo) -> bool:
    if "any" in spec.platforms:
        return True
    tags = set(spec.platforms)
    system = hardware.system.lower()
    if hardware.apple_silicon and "darwin-arm64" in tags:
        return True
    if system == "darwin" and "darwin" in tags:
        return True
    if system == "linux" and "linux" in tags:
        return True
    if system == "windows" and "windows" in tags:
        return True
    return False


def module_available(name: str | None) -> bool:
    if not name:
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def vulkan_loader_present() -> bool:
    return detect_vulkan_loader()


def whisper_cli_path() -> str | None:
    configured = os.environ.get("MEETSCRIBE_WHISPER_CPP", "").strip()
    if configured and Path(configured).expanduser().is_file():
        return str(Path(configured).expanduser())
    return shutil.which("whisper-cli")


def describe_backend(spec: BackendSpec, hardware: HardwareInfo) -> BackendStatus:
    if not spec.implemented:
        return BackendStatus(False, spec.dependency_present(), platform_matches(spec, hardware), False, "not implemented")
    platform_ok = platform_matches(spec, hardware)
    if not platform_ok:
        if spec.name == "mlx-whisper":
            reason = "requires Apple Silicon"
        else:
            reason = f"not usable on {hardware.system} {hardware.machine}"
        return BackendStatus(True, spec.dependency_present(), False, False, reason)
    reason = _runtime_reason(spec, hardware)
    dependency = spec.dependency_present() or _extra_runtime(spec)
    usable = _is_usable(spec, hardware, dependency)
    return BackendStatus(True, dependency, True, usable, reason)


def _extra_runtime(spec: BackendSpec) -> bool:
    if spec.name in {"vulkan", "whisper.cpp"}:
        return whisper_cli_path() is not None
    if spec.name == "macos":
        return shutil.which("meetscribe-system-audio") is not None or shutil.which("swiftc") is not None
    return False


def _is_usable(spec: BackendSpec, hardware: HardwareInfo, dependency: bool) -> bool:
    if spec.name == "mlx-whisper":
        return hardware.apple_silicon and module_available("mlx_whisper")
    if spec.name in {"vulkan", "whisper.cpp"}:
        return dependency
    if spec.name == "openvino":
        return module_available("openvino_genai")
    if spec.name == "pyannote":
        return module_available("pyannote.audio")
    if spec.name == "linux":
        return module_available("sounddevice")
    if spec.name == "windows":
        return module_available("pyaudiowpatch")
    if spec.name == "macos":
        return module_available("sounddevice") and shutil.which("meetscribe-system-audio") is not None
    if spec.module is None:
        return True
    return dependency


def _runtime_reason(spec: BackendSpec, hardware: HardwareInfo) -> str:
    if spec.name == "mlx-whisper":
        if not hardware.apple_silicon:
            return "requires Apple Silicon"
        if not module_available("mlx_whisper"):
            return "pip install 'meetscribe[mlx]'"
        return "mlx-whisper importable; default weights schsu/breeze-asr-25-mlx"
    if spec.name in {"vulkan", "whisper.cpp"}:
        binary = whisper_cli_path()
        binding = module_available("pywhispercpp")
        vulkan = vulkan_loader_present()
        if binary and vulkan:
            return f"whisper-cli at {binary}; Vulkan ICD present"
        if binary:
            return f"whisper-cli at {binary}; Vulkan ICD not detected, binary may be CPU-only"
        if binding and vulkan:
            return "pywhispercpp importable; Vulkan ICD present (GPU only if built with GGML_VULKAN=1)"
        if binding:
            return "pywhispercpp importable; no Vulkan ICD, the stock wheel runs on CPU"
        return "build whisper.cpp with -DGGML_VULKAN=1 or pip install 'meetscribe[vulkan]'"
    if spec.name == "openvino":
        if not module_available("openvino_genai"):
            return "pip install 'meetscribe[openvino]'"
        devices = _openvino_devices()
        if any(device.upper().startswith("GPU") for device in devices):
            return "OpenVINO devices: " + ", ".join(devices)
        if devices:
            return "OpenVINO devices: " + ", ".join(devices) + " (no GPU listed, CPU fallback works)"
        return "openvino-genai importable"
    if spec.name == "pyannote":
        if not module_available("pyannote.audio"):
            return "pip install 'meetscribe[diarization]'"
        return "pyannote importable; set HF_TOKEN and accept speaker-diarization-community-1"
    if spec.name == "linux":
        if not module_available("sounddevice"):
            return "pip install 'meetscribe[capture-linux]'"
        if shutil.which("pactl") is None:
            return "sounddevice importable; pactl was not found"
        return "sounddevice importable; pactl found"
    if spec.name == "windows":
        if not module_available("pyaudiowpatch"):
            return "pip install 'meetscribe[capture-windows]'"
        return "PyAudioWPatch importable"
    if spec.name == "macos":
        if not module_available("sounddevice"):
            return "pip install 'meetscribe[capture-macos]'"
        if shutil.which("meetscribe-system-audio"):
            return "sounddevice and meetscribe-system-audio found"
        if shutil.which("swiftc"):
            return "sounddevice found; compile scripts/macos/SystemAudioCapture.swift"
        return "sounddevice found; swiftc is missing, so system audio cannot be built"
    if spec.name == "faster-whisper" and not module_available("faster_whisper"):
        return "pip install 'meetscribe[faster-whisper]'"
    if spec.dependency_present():
        return "ready"
    if spec.extra:
        return f"pip install 'meetscribe[{spec.extra}]'"
    return "ready"


def _openvino_devices() -> list[str]:
    if not module_available("openvino"):
        return []
    try:
        import openvino as ov

        return [str(device) for device in ov.Core().available_devices]
    except Exception:
        return []

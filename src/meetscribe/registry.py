"""Backend registry. Names in the config file are resolved here."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from meetscribe.config import AppConfig
from meetscribe.errors import UnknownBackend
from meetscribe.hardware import HardwareInfo, recommended_asr, recommended_capture

Factory = Callable[[AppConfig], object]


@dataclass(frozen=True, slots=True)
class BackendSpec:
    kind: str
    name: str
    description: str
    device: str
    implemented: bool
    factory: Factory
    module: str | None = None
    extra: str | None = None
    platforms: tuple[str, ...] = ("any",)

    def dependency_present(self) -> bool:
        if self.module is None:
            return True
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False


class Registry:
    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], BackendSpec] = {}
        self._order: list[tuple[str, str]] = []

    def register(self, spec: BackendSpec) -> None:
        key = (spec.kind, spec.name)
        if key not in self._specs:
            self._order.append(key)
        self._specs[key] = spec

    def names(self, kind: str) -> list[str]:
        return [name for registered_kind, name in self._order if registered_kind == kind]

    def get(self, kind: str, name: str) -> BackendSpec:
        try:
            return self._specs[(kind, name)]
        except KeyError as exc:
            known = ", ".join(self.names(kind)) or "(none)"
            raise UnknownBackend(f"Unknown {kind} backend {name!r}. Known: {known}") from exc

    def create(self, kind: str, name: str, config: AppConfig) -> object:
        return self.get(kind, name).factory(config)

    def list(self, kind: str | None = None) -> list[BackendSpec]:
        specs: list[BackendSpec] = []
        for registered_kind, name in self._order:
            if kind is not None and registered_kind != kind:
                continue
            specs.append(self._specs[(registered_kind, name)])
        return specs

    def resolve_asr(self, name: str, hardware: HardwareInfo) -> str:
        if name != "auto":
            return name
        return recommended_asr(hardware)

    def resolve_capture(self, name: str, hardware: HardwareInfo) -> str:
        if name != "auto":
            return name
        return recommended_capture(hardware)


def build_registry() -> Registry:
    registry = Registry()
    for spec in _asr_specs() + _diarization_specs() + _capture_specs():
        registry.register(spec)
    return registry


def _asr_specs() -> list[BackendSpec]:
    return [
        BackendSpec(
            kind="asr",
            name="faster-whisper",
            description="CTranslate2 Whisper on CPU. Uses the fallback model when the checkpoint is not CT2.",
            device="cpu",
            implemented=True,
            factory=_faster_whisper,
            module="faster_whisper",
            extra="faster-whisper",
            platforms=("linux", "windows", "darwin"),
        ),
        BackendSpec(
            kind="asr",
            name="mlx-whisper",
            description="MLX Whisper on Apple Silicon. Default weights: schsu/breeze-asr-25-mlx.",
            device="metal",
            implemented=True,
            factory=_mlx_whisper,
            module="mlx_whisper",
            extra="mlx",
            platforms=("darwin-arm64",),
        ),
        BackendSpec(
            kind="asr",
            name="openvino",
            description="OpenVINO GenAI WhisperPipeline. GPU when listed, otherwise CPU.",
            device="openvino-gpu",
            implemented=True,
            factory=_openvino,
            module="openvino_genai",
            extra="openvino",
            platforms=("linux", "windows"),
        ),
        BackendSpec(
            kind="asr",
            name="vulkan",
            description="whisper.cpp built with GGML_VULKAN. Falls back to pywhispercpp.",
            device="vulkan",
            implemented=True,
            factory=_vulkan,
            module="pywhispercpp",
            extra="vulkan",
            platforms=("linux", "windows", "darwin"),
        ),
        BackendSpec(
            kind="asr",
            name="whisper.cpp",
            description="Alias of vulkan: whisper-cli if set, otherwise pywhispercpp.",
            device="vulkan",
            implemented=True,
            factory=_whisper_cpp,
            module="pywhispercpp",
            extra="whisper-cpp",
            platforms=("linux", "windows", "darwin"),
        ),
    ]


def _diarization_specs() -> list[BackendSpec]:
    return [
        BackendSpec(
            kind="diarization",
            name="track",
            description="Mic track is Me, system track is Remote. No neural model.",
            device="cpu",
            implemented=True,
            factory=_track,
            platforms=("any",),
        ),
        BackendSpec(
            kind="diarization",
            name="single-speaker",
            description="One anonymous speaker for the whole timeline.",
            device="cpu",
            implemented=True,
            factory=_single_speaker,
            platforms=("any",),
        ),
        BackendSpec(
            kind="diarization",
            name="pyannote",
            description="pyannote speaker-diarization-community-1 on the system track. Needs HF_TOKEN.",
            device="cpu",
            implemented=True,
            factory=_pyannote,
            module="pyannote.audio",
            extra="diarization",
            platforms=("linux", "windows", "darwin"),
        ),
        BackendSpec(
            kind="diarization",
            name="sherpa-onnx",
            description="sherpa-onnx speaker diarization. Local, no token.",
            device="cpu",
            implemented=False,
            factory=_sherpa,
            module="sherpa_onnx",
            extra="sherpa",
            platforms=("linux", "windows", "darwin"),
        ),
    ]


def _capture_specs() -> list[BackendSpec]:
    return [
        BackendSpec(
            kind="capture",
            name="file",
            description="Load existing audio files as mic and system tracks.",
            device="file",
            implemented=True,
            factory=_file_input,
            platforms=("any",),
        ),
        BackendSpec(
            kind="capture",
            name="macos",
            description="ScreenCaptureKit helper for system audio plus a Core Audio mic.",
            device="screencapturekit",
            implemented=True,
            factory=_macos_capture,
            module="sounddevice",
            extra="capture-macos",
            platforms=("darwin",),
        ),
        BackendSpec(
            kind="capture",
            name="windows",
            description="WASAPI loopback for system audio plus the default microphone.",
            device="wasapi",
            implemented=True,
            factory=_windows_capture,
            module="pyaudiowpatch",
            extra="capture-windows",
            platforms=("windows",),
        ),
        BackendSpec(
            kind="capture",
            name="linux",
            description="PulseAudio/PipeWire monitor source plus the default microphone.",
            device="pipewire",
            implemented=True,
            factory=_linux_capture,
            module="sounddevice",
            extra="capture-linux",
            platforms=("linux",),
        ),
    ]


def _faster_whisper(config: AppConfig) -> object:
    from meetscribe.asr.faster_whisper import FasterWhisperAsr

    return FasterWhisperAsr(
        model=config.asr.model,
        fallback_model=config.asr.fallback_model,
        device=config.asr.device,
        language=config.asr.language,
    )


def _model_dir(config: AppConfig):
    from meetscribe.asr.models import default_model_dir

    if config.asr.model_dir:
        return Path(config.asr.model_dir).expanduser()
    return default_model_dir()


def _mlx_whisper(config: AppConfig) -> object:
    from meetscribe.asr.mlx_whisper import MlxWhisperAsr

    return MlxWhisperAsr(
        model=config.asr.model,
        fallback_model=config.asr.fallback_model,
        language=config.asr.language,
    )


def _openvino(config: AppConfig) -> object:
    from meetscribe.asr.openvino_backend import OpenVINOWhisperAsr

    return OpenVINOWhisperAsr(
        model=config.asr.model,
        fallback_model=config.asr.fallback_model,
        language=config.asr.language,
        device=config.asr.openvino_device or config.asr.device,
        model_dir=_model_dir(config),
    )


def _vulkan(config: AppConfig) -> object:
    from meetscribe.asr.vulkan import VulkanWhisperAsr

    return VulkanWhisperAsr(
        model=config.asr.model,
        fallback_model=config.asr.fallback_model,
        language=config.asr.language,
        model_dir=_model_dir(config),
        whisper_cpp_bin=config.asr.whisper_cpp_bin,
        backend_name="vulkan",
    )


def _whisper_cpp(config: AppConfig) -> object:
    from meetscribe.asr.vulkan import VulkanWhisperAsr

    return VulkanWhisperAsr(
        model=config.asr.model,
        fallback_model=config.asr.fallback_model,
        language=config.asr.language,
        model_dir=_model_dir(config),
        whisper_cpp_bin=config.asr.whisper_cpp_bin,
        backend_name="whisper.cpp",
    )


def _track(config: AppConfig) -> object:
    del config
    from meetscribe.diarization.simple import TrackDiarizer

    return TrackDiarizer()


def _single_speaker(config: AppConfig) -> object:
    del config
    from meetscribe.diarization.simple import SingleSpeakerDiarizer

    return SingleSpeakerDiarizer()


def _pyannote(config: AppConfig) -> object:
    from meetscribe.diarization.pyannote import PyannoteDiarizer

    return PyannoteDiarizer(hf_token=config.hf_token())


def _sherpa(config: AppConfig) -> object:
    del config
    from meetscribe.diarization.stubs import SherpaDiarizer

    return SherpaDiarizer()


def _file_input(config: AppConfig) -> object:
    del config
    from meetscribe.audio.file_input import FileInput

    return FileInput()


def _capture_kwargs(config: AppConfig) -> dict[str, object]:
    return {
        "sample_rate": config.capture.sample_rate,
        "mic_device": config.capture.mic_device,
        "system_device": config.capture.system_device,
        "chunk_seconds": config.capture.chunk_seconds,
    }


def _macos_capture(config: AppConfig) -> object:
    from meetscribe.audio.macos import MacOSCapture

    return MacOSCapture(**_capture_kwargs(config))


def _windows_capture(config: AppConfig) -> object:
    from meetscribe.audio.windows import WindowsCapture

    return WindowsCapture(**_capture_kwargs(config))


def _linux_capture(config: AppConfig) -> object:
    from meetscribe.audio.linux import LinuxCapture

    return LinuxCapture(**_capture_kwargs(config))

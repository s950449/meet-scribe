from pathlib import Path

import pytest

from meetscribe.asr.faster_whisper import resolve_ct2_model
from meetscribe.asr.mlx_whisper import MlxWhisperAsr
from meetscribe.asr.openvino_backend import OpenVINOWhisperAsr
from meetscribe.asr.vulkan import VulkanWhisperAsr
from meetscribe.config import AppConfig
from meetscribe.diarization.pyannote import PyannoteDiarizer
from meetscribe.diarization.simple import SingleSpeakerDiarizer, TrackDiarizer
from meetscribe.errors import BackendNotInstalled, ModelMissing, UnknownBackend
from meetscribe.hardware import (
    HardwareInfo,
    detect_intel_igpu,
    recommended_asr,
    recommended_capture,
    recommended_diarization,
)
from meetscribe.registry import BackendSpec, Registry, build_registry
from meetscribe.types import AudioTrack


def _hardware(**overrides: object) -> HardwareInfo:
    data = {
        "system": "Linux",
        "machine": "x86_64",
        "apple_silicon": False,
        "intel_cpu": True,
        "intel_igpu": False,
    }
    data.update(overrides)
    return HardwareInfo(**data)  # type: ignore[arg-type]


def test_registry_lists_planned_backends() -> None:
    registry = build_registry()
    assert registry.names("asr") == [
        "faster-whisper",
        "mlx-whisper",
        "openvino",
        "vulkan",
        "whisper.cpp",
    ]
    assert registry.names("diarization") == ["track", "single-speaker", "pyannote", "sherpa-onnx"]
    assert registry.names("capture") == ["file", "macos", "windows", "linux"]
    implemented = {spec.name for spec in registry.list() if spec.implemented}
    assert {
        "faster-whisper",
        "mlx-whisper",
        "openvino",
        "vulkan",
        "whisper.cpp",
        "pyannote",
        "track",
        "single-speaker",
        "file",
        "macos",
        "windows",
        "linux",
    } <= implemented
    stubs = {spec.name for spec in registry.list() if not spec.implemented}
    assert stubs == {"sherpa-onnx"}


def test_unknown_backend() -> None:
    registry = build_registry()
    with pytest.raises(UnknownBackend, match="faster-whisper"):
        registry.create("asr", "nope", AppConfig())


def test_auto_resolution() -> None:
    registry = build_registry()
    apple = _hardware(
        system="Darwin",
        machine="arm64",
        apple_silicon=True,
        intel_cpu=False,
        intel_igpu=None,
    )
    intel = _hardware(system="Linux", intel_igpu=True)
    cpu = _hardware(system="Linux", intel_igpu=False)
    assert registry.resolve_asr("auto", apple) == "mlx-whisper"
    assert registry.resolve_asr("auto", intel) == "openvino"
    assert registry.resolve_asr("auto", cpu) == "faster-whisper"
    assert registry.resolve_asr("whisper.cpp", apple) == "whisper.cpp"
    assert registry.resolve_capture("auto", apple) == "macos"
    assert registry.resolve_capture("auto", _hardware(system="Windows")) == "windows"
    assert recommended_diarization(2) == "track"
    assert recommended_diarization(1) == "single-speaker"
    assert recommended_asr(cpu) == "faster-whisper"
    assert recommended_capture(_hardware(system="Linux")) == "linux"


def test_create_real_and_stub_backends() -> None:
    registry = build_registry()
    config = AppConfig()
    asr = registry.create("asr", "faster-whisper", config)
    assert asr.name == "faster-whisper"
    assert asr.capabilities().word_timestamps is True
    assert asr.capabilities().device == "cpu"
    mlx = registry.create("asr", "mlx-whisper", config)
    assert mlx.capabilities().device == "metal"
    assert registry.create("diarization", "track", config).name == "track"
    assert registry.get("asr", "faster-whisper").dependency_present() is False


def test_custom_backend_registration() -> None:
    registry = Registry()

    def factory(config: AppConfig) -> str:
        del config
        return "fake-backend"

    registry.register(
        BackendSpec(
            kind="asr",
            name="fake",
            description="test double",
            device="cpu",
            implemented=True,
            factory=factory,
        )
    )
    assert registry.create("asr", "fake", AppConfig()) == "fake-backend"


def test_ct2_model_resolution() -> None:
    model, warning = resolve_ct2_model(
        "MediaTek-Research/Breeze-ASR-25",
        "openai/whisper-large-v3-turbo",
    )
    assert model == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert warning is not None and "Breeze-ASR-25" in warning
    mapped, mapped_warning = resolve_ct2_model("openai/whisper-large-v3-turbo", "tiny")
    assert mapped == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert mapped_warning is None
    assert resolve_ct2_model("tiny", "tiny") == ("tiny", None)


def test_simple_diarizers(tmp_path: Path) -> None:
    mic = AudioTrack("mic", tmp_path / "mic.wav", 16000, 1, 3.0)
    system = AudioTrack("system", tmp_path / "system.wav", 16000, 1, 5.0)
    single = SingleSpeakerDiarizer().diarize([mic])
    assert single[0].speaker_id == "SPEAKER_00"
    assert single[0].end == 3.0
    turns = TrackDiarizer().diarize([mic, system])
    assert [(turn.speaker_id, turn.track_id, turn.end) for turn in turns] == [
        ("Me", "mic", 3.0),
        ("Remote", "system", 5.0),
    ]


def test_optional_backends_fail_clearly_without_weights_or_packages(tmp_path: Path) -> None:
    track = AudioTrack("mic", tmp_path / "mic.wav", 16000, 1, 1.0)
    with pytest.raises(BackendNotInstalled):
        MlxWhisperAsr(
            model="MediaTek-Research/Breeze-ASR-25",
            fallback_model="openai/whisper-large-v3-turbo",
        ).transcribe(tmp_path / "mic.wav", track_id="mic")
    with pytest.raises(ModelMissing):
        OpenVINOWhisperAsr(model="m", fallback_model="f", model_dir=tmp_path).transcribe(
            tmp_path / "mic.wav",
            track_id="mic",
        )
    with pytest.raises(ModelMissing):
        VulkanWhisperAsr(model="m", fallback_model="f", model_dir=tmp_path).transcribe(
            tmp_path / "mic.wav",
            track_id="mic",
        )
    with pytest.raises(BackendNotInstalled):
        PyannoteDiarizer(hf_token=None).diarize([track])


def test_pyannote_without_the_package_reports_the_extra(tmp_path: Path) -> None:
    track = AudioTrack("mic", tmp_path / "mic.wav", 16000, 1, 1.0)
    with pytest.raises(BackendNotInstalled, match="pyannote"):
        PyannoteDiarizer(hf_token="hf-test").diarize([track])


def test_intel_igpu_sysfs(tmp_path: Path) -> None:
    intel = tmp_path / "card0" / "device"
    other = tmp_path / "card1" / "device"
    intel.mkdir(parents=True)
    other.mkdir(parents=True)
    (intel / "vendor").write_text("0x8086\n", encoding="utf-8")
    (other / "vendor").write_text("0x10de\n", encoding="utf-8")
    assert detect_intel_igpu(tmp_path) is True
    (intel / "vendor").write_text("0x10de\n", encoding="utf-8")
    assert detect_intel_igpu(tmp_path) is False
    assert detect_intel_igpu(tmp_path / "missing") is None

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from meetscribe.asr.mlx_whisper import MlxWhisperAsr
from meetscribe.asr.models import (
    BREEZE,
    TURBO,
    resolve_ggml_file,
    resolve_mlx_repo,
    resolve_openvino_dir,
)
from meetscribe.asr.openvino_backend import OpenVINOWhisperAsr, select_openvino_device
from meetscribe.asr.vulkan import VulkanWhisperAsr, parse_whisper_cpp_json
from meetscribe.audio.devices import (
    AudioDevice,
    choose_linux_devices,
    choose_wasapi_devices,
    parse_pactl_short,
)
from meetscribe.audio.linux import LinuxCapture
from meetscribe.audio.live import make_live_callback
from meetscribe.audio.sync import SyncedWavSession
from meetscribe.diarization.pyannote import PyannoteDiarizer
from meetscribe.errors import AudioError, BackendNotInstalled, ModelMissing
from meetscribe.events import stage_event
from meetscribe.hardware import HardwareInfo
from meetscribe.probe import describe_backend
from meetscribe.registry import build_registry
from meetscribe.types import AudioTrack, ProgressEvent, TranscriptSegment


class _Sink:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


def test_mlx_repo_resolution() -> None:
    repo, warning = resolve_mlx_repo(BREEZE, TURBO)
    assert repo == "schsu/breeze-asr-25-mlx"
    assert warning is None
    turbo, turbo_warning = resolve_mlx_repo(TURBO, TURBO)
    assert turbo == "mlx-community/whisper-large-v3-turbo"
    assert turbo_warning is None
    fallback, fallback_warning = resolve_mlx_repo("acme/custom-whisper", TURBO)
    assert fallback == "mlx-community/whisper-large-v3-turbo"
    assert fallback_warning is not None


def test_mlx_transcribe_parses_words(tmp_path: Path) -> None:
    audio = tmp_path / "mic.wav"
    audio.write_bytes(b"")

    def fake(path: str, **kwargs: object) -> dict[str, object]:
        assert kwargs["path_or_hf_repo"] == "schsu/breeze-asr-25-mlx"
        assert kwargs["word_timestamps"] is True
        return {
            "language": "zh",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "text": "你好 API",
                    "words": [
                        {"word": "你好", "start": 0.0, "end": 0.4},
                        {"word": "API", "start": 0.5, "end": 1.1},
                    ],
                }
            ],
        }

    segments = MlxWhisperAsr(model=BREEZE, fallback_model=TURBO, transcribe_fn=fake).transcribe(
        audio, track_id="mic"
    )
    assert segments[0].text == "你好 API"
    assert segments[0].words[1].text == "API"
    assert segments[0].language == "zh"


def test_ggml_and_openvino_resolution(tmp_path: Path) -> None:
    turbo = tmp_path / "ggml-large-v3-turbo.bin"
    turbo.write_bytes(b"ggml")
    path, warning = resolve_ggml_file(BREEZE, TURBO, tmp_path)
    assert path == turbo
    assert warning is not None and "Breeze" in warning
    breeze = tmp_path / "ggml-breeze-asr-25.bin"
    breeze.write_bytes(b"ggml")
    exact, exact_warning = resolve_ggml_file(BREEZE, TURBO, tmp_path)
    assert exact == breeze
    assert exact_warning is None

    ov = tmp_path / "whisper-large-v3-turbo-ov"
    ov.mkdir()
    (ov / "generation_config.json").write_text("{}", encoding="utf-8")
    resolved, ov_warning = resolve_openvino_dir(BREEZE, TURBO, tmp_path)
    assert resolved == ov
    assert ov_warning is not None
    with pytest.raises(ModelMissing):
        resolve_openvino_dir("openai/whisper-tiny", TURBO, tmp_path / "missing")


def test_openvino_device_selection_and_fake_pipeline(tmp_path: Path) -> None:
    assert select_openvino_device("auto", ["CPU", "GPU"]) == ("GPU", None)
    cpu, warning = select_openvino_device("GPU", ["CPU"])
    assert cpu == "CPU"
    assert warning is not None

    model_dir = tmp_path / "models"
    ir = model_dir / "breeze-asr-25-ov"
    ir.mkdir(parents=True)
    (ir / "generation_config.json").write_text("{}", encoding="utf-8")

    class Chunk:
        def __init__(self, start: float, end: float, text: str) -> None:
            self.start_ts = start
            self.end_ts = end
            self.text = text

    class Result:
        def __init__(self) -> None:
            self.chunks = [Chunk(0.0, 1.5, " 你好 API")]
            self.texts = ["你好 API"]

    seen: dict[str, object] = {}

    def factory(path: str, device: str, word_timestamps: bool = False) -> object:
        seen["path"] = path
        seen["device"] = device
        seen["words"] = word_timestamps

        class Pipe:
            def generate(self, audio: object, **kwargs: object) -> Result:
                seen["language"] = kwargs.get("language")
                assert len(audio) > 0  # type: ignore[arg-type]
                return Result()

        return Pipe()

    asr = OpenVINOWhisperAsr(
        model=BREEZE,
        fallback_model=TURBO,
        device="auto",
        model_dir=model_dir,
        pipeline_factory=factory,
        device_lister=lambda: ["CPU", "GPU.0"],
        audio_loader=lambda path: np.ones(160, dtype=np.float32),
    )
    segments = asr.transcribe(tmp_path / "mic.wav", track_id="system", language="zh")
    assert seen["device"] == "GPU.0"
    assert seen["words"] is True
    assert segments[0].text == "你好 API"
    assert segments[0].track_id == "system"


def test_whisper_cpp_json_and_runner(tmp_path: Path) -> None:
    payload = {
        "transcription": [
            {
                "text": " 你好",
                "offsets": {"from": 0, "to": 400},
                "tokens": [{"text": " 你好", "offsets": {"from": 0, "to": 400}}],
            },
            {
                "text": " API",
                "offsets": {"from": 500, "to": 900},
                "tokens": [{"text": " API", "offsets": {"from": 500, "to": 900}}],
            },
        ]
    }
    segments = parse_whisper_cpp_json(payload, "mic")
    assert len(segments) == 1
    assert "API" in segments[0].text
    assert segments[0].start == 0.0
    assert segments[0].end == 0.9

    model = tmp_path / "ggml-large-v3-turbo.bin"
    model.write_bytes(b"ggml")

    def runner(command: list[str], output: Path) -> None:
        assert Path(command[0]).name == "whisper-cli"
        assert "-ojf" in command
        output.write_text(json.dumps(payload), encoding="utf-8")

    asr = VulkanWhisperAsr(
        model=str(model),
        fallback_model=TURBO,
        model_dir=tmp_path,
        runner=runner,
    )
    assert asr.transcribe(tmp_path / "in.wav", track_id="mic")[0].track_id == "mic"


def test_synced_session_corrects_drift_and_chunks(tmp_path: Path) -> None:
    chunks: list[tuple[str, float, int]] = []
    session = SyncedWavSession(
        tmp_path,
        sample_rate=1000,
        chunk_seconds=0.05,
        on_chunk=lambda track, start, samples: chunks.append((track, start, int(samples.size))),
    )
    session.write("mic", 10.0, np.ones(100, dtype=np.float32), 1000)
    session.write("mic", 10.3, np.ones(100, dtype=np.float32), 1000)
    session.write("system", 10.0, np.ones(50, dtype=np.float32), 1000)
    session.write("system", 10.0, np.ones(80, dtype=np.float32), 1000)
    result = session.finish(datetime.now().astimezone())
    stats = session.stats()
    assert stats["mic"].silence_frames == 200
    assert stats["system"].dropped_frames == 50
    durations = {track.track_id: track.duration_s for track in result.tracks}
    assert durations["mic"] == durations["system"] == 0.4
    assert (tmp_path / "session.json").is_file()
    assert ("mic", 0.0, 50) in [(item[0], item[1], item[2]) for item in chunks]


def test_linux_device_choice_and_fake_record(tmp_path: Path) -> None:
    text = (
        "0\talsa_output.pci.analog-stereo.monitor\tmodule\n"
        "1\talsa_input.pci.analog-stereo\tmodule\n"
    )
    sources = parse_pactl_short(text)
    mic, system = choose_linux_devices(sources)
    assert mic.kind == "mic"
    assert system.kind == "monitor"
    with pytest.raises(AudioError):
        choose_linux_devices(sources, system_device="missing")

    opened: list[str] = []

    def opener(device: AudioDevice, track_id: str, session: SyncedWavSession, sample_rate: int):
        opened.append(track_id)
        session.write(track_id, 0.0, np.ones(sample_rate // 10, dtype=np.float32), sample_rate)

        class Stream:
            def start(self) -> None:
                return None

            def close(self) -> None:
                return None

        return Stream()

    capture = LinuxCapture(sample_rate=16000, source_lister=lambda: sources, stream_opener=opener)
    result = capture.record(tmp_path, duration_s=0)
    assert opened == ["mic", "system"]
    assert {track.track_id for track in result.tracks} == {"mic", "system"}


def test_wasapi_choice() -> None:
    devices = [
        AudioDevice("1", "Headset", "mic"),
        AudioDevice("2", "Speakers", "output"),
        AudioDevice("3", "Speakers (loopback)", "loopback"),
        AudioDevice("4", "HDMI (loopback)", "loopback"),
    ]
    mic, system = choose_wasapi_devices(devices, default_output="Speakers")
    assert mic.name == "Headset"
    assert system.name == "Speakers (loopback)"


def test_pyannote_keeps_mic_and_splits_system(tmp_path: Path) -> None:
    class Turn:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class Annotation:
        def __iter__(self):
            return iter([(Turn(0.0, 1.0), "SPEAKER_00"), (Turn(0.8, 2.0), "SPEAKER_01")])

    class Output:
        speaker_diarization = Annotation()

    diarizer = PyannoteDiarizer(hf_token=None, pipeline_loader=lambda: lambda path: Output())
    tracks = [
        AudioTrack("mic", tmp_path / "mic.wav", 16000, 1, 3.0),
        AudioTrack("system", tmp_path / "system.wav", 16000, 1, 3.0),
    ]
    turns = diarizer.diarize(tracks)
    assert turns[0].speaker_id == "Me"
    assert turns[0].track_id == "mic"
    remote = [turn for turn in turns if turn.track_id == "system"]
    assert {turn.speaker_id for turn in remote} == {"SPEAKER_00", "SPEAKER_01"}
    assert all(turn.overlap for turn in remote)
    with pytest.raises(BackendNotInstalled):
        PyannoteDiarizer(hf_token=None).diarize(tracks[:1])


def test_live_callback_offsets_segments(tmp_path: Path) -> None:
    class FakeAsr:
        def transcribe(
            self,
            audio: Path,
            *,
            track_id: str,
            language: str | None = None,
            on_event=None,
        ):
            del audio, language, on_event
            return [TranscriptSegment("你好", 0.2, 0.8, track_id)]

    sink = _Sink()
    callback = make_live_callback(FakeAsr(), "zh", sink, 16000)
    callback("system", 5.0, np.zeros(1600, dtype=np.float32))
    messages = [event.message for event in sink.events if event.stage == "live"]
    assert any(message and "5.2" in message and "你好" in message for message in messages)
    stage_event("live", "segment")


def test_usability_on_linux_cpu() -> None:
    hardware = HardwareInfo(
        system="Linux",
        machine="x86_64",
        apple_silicon=False,
        intel_cpu=True,
        intel_igpu=False,
        vulkan_gpu=False,
    )
    registry = build_registry()
    by_name = {spec.name: describe_backend(spec, hardware) for spec in registry.list()}
    assert by_name["mlx-whisper"].usable is False
    assert "Apple Silicon" in by_name["mlx-whisper"].reason
    assert by_name["macos"].usable is False
    assert by_name["track"].usable is True
    assert by_name["sherpa-onnx"].usable is False
    assert by_name["faster-whisper"].implemented is True

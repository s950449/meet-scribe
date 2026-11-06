"""TOML config, environment secrets, and CLI overrides.

Precedence, from lowest to highest: built-in defaults, the TOML file,
selected environment variables, then CLI overrides. API keys and Hugging Face
tokens are never read from the TOML file.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from meetscribe.errors import ConfigError

DEFAULT_ASR_MODEL = "MediaTek-Research/Breeze-ASR-25"
DEFAULT_FALLBACK_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LLM_MODEL = "gemma-4-12b-it"
DEFAULT_LLM_FALLBACK_MODEL = "qwen3.5-9b"
# Sent only when OPENAI_API_KEY is unset. llama-server does not check it.
DUMMY_API_KEY = "llama-server"
SECRET_KEYS = frozenset({"api_key", "hf_token", "token", "openai_api_key"})


@dataclass
class AsrSettings:
    backend: str = "auto"
    model: str = DEFAULT_ASR_MODEL
    fallback_model: str = DEFAULT_FALLBACK_MODEL
    language: str = "zh"
    device: str = "auto"
    openvino_device: str = "auto"
    model_dir: str = ""
    whisper_cpp_bin: str = ""


@dataclass
class DiarizationSettings:
    backend: str = "auto"
    hf_token_env: str = "HF_TOKEN"


@dataclass(frozen=True, slots=True)
class SamplingProfile:
    temperature: float
    top_p: float
    top_k: int
    min_p: float = 0.0
    presence_penalty: float | None = None


def default_sampling_profiles() -> dict[str, SamplingProfile]:
    """Unsloth card values. Gemma is the default model; Qwen is the smaller GGUF."""

    return {
        "gemma-4-12b-it": SamplingProfile(
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            min_p=0.0,
            presence_penalty=None,
        ),
        "qwen3.5-9b": SamplingProfile(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
        ),
    }


@dataclass
class LlmSettings:
    base_url: str = DEFAULT_LLM_BASE_URL
    api_key_env: str = "OPENAI_API_KEY"
    model: str = DEFAULT_LLM_MODEL
    fallback_model: str = DEFAULT_LLM_FALLBACK_MODEL
    polish: bool = True
    # Practical KV-cache size for an iGPU. Both suggested models natively
    # support 256K; raise this together with llama-server -c when memory allows.
    context_tokens: int = 16384
    # Room for the JSON completion. Not sent to the server.
    reserve_tokens: int = 2048
    # None means "use the matched profile". A value set in TOML or on the CLI overrides it.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    profiles: dict[str, SamplingProfile] = field(default_factory=default_sampling_profiles)
    # Set when --llm-model (or MEETSCRIBE_LLM_MODEL) names the model explicitly.
    model_locked: bool = False


@dataclass
class CaptureSettings:
    backend: str = "auto"
    sample_rate: int = 48_000
    mic_device: str = ""
    system_device: str = ""
    chunk_seconds: float = 5.0


@dataclass
class AppConfig:
    asr: AsrSettings = field(default_factory=AsrSettings)
    diarization: DiarizationSettings = field(default_factory=DiarizationSettings)
    llm: LlmSettings = field(default_factory=LlmSettings)
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    speaker_map: dict[str, str] = field(default_factory=dict)
    output_dir: Path = field(default_factory=Path.cwd)

    def api_key(self, environ: Mapping[str, str] | None = None) -> str | None:
        env = os.environ if environ is None else environ
        value = env.get(self.llm.api_key_env, "").strip()
        return value or None

    def resolved_api_key(self, environ: Mapping[str, str] | None = None) -> str:
        """Key for the OpenAI client. llama-server accepts the dummy value."""

        return self.api_key(environ) or DUMMY_API_KEY

    def hf_token(self, environ: Mapping[str, str] | None = None) -> str | None:
        env = os.environ if environ is None else environ
        value = env.get(self.diarization.hf_token_env, "").strip()
        return value or None


def default_config_path(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    env = os.environ if environ is None else environ
    home_dir = Path.home() if home is None else home
    xdg = env.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return Path(xdg).expanduser() / "meetscribe" / "config.toml"
    return home_dir / ".config" / "meetscribe" / "config.toml"


def locate_config_file(
    explicit: Path | None,
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> tuple[Path | None, bool]:
    """Return ``(path, required)``.

    An explicit ``--config`` or ``MEETSCRIBE_CONFIG`` path is required to exist.
    The XDG default is optional.
    """

    env = os.environ if environ is None else environ
    if explicit is not None:
        return Path(explicit).expanduser(), True
    configured = env.get("MEETSCRIBE_CONFIG", "").strip()
    if configured:
        return Path(configured).expanduser(), True
    return default_config_path(environ=env, home=home), False


def load_config(
    path: Path | None = None,
    *,
    overrides: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> AppConfig:
    env = os.environ if environ is None else environ
    config = AppConfig()
    file_path, required = locate_config_file(path, environ=env, home=home)
    if file_path is not None and file_path.is_file():
        config = _merge_toml(config, _read_toml(file_path))
    elif file_path is not None and required:
        raise ConfigError(f"Config file not found: {file_path}")

    config = _apply_env(config, env)
    if overrides:
        config = apply_overrides(config, overrides)
    return config


def apply_overrides(config: AppConfig, overrides: Mapping[str, object]) -> AppConfig:
    asr = config.asr
    diar = config.diarization
    llm = config.llm
    capture = config.capture
    speaker_map = dict(config.speaker_map)
    output_dir = config.output_dir

    if "asr_backend" in overrides and overrides["asr_backend"] is not None:
        asr = replace(asr, backend=str(overrides["asr_backend"]))
    if "asr_model" in overrides and overrides["asr_model"] is not None:
        asr = replace(asr, model=str(overrides["asr_model"]))
    if "asr_fallback_model" in overrides and overrides["asr_fallback_model"] is not None:
        asr = replace(asr, fallback_model=str(overrides["asr_fallback_model"]))
    if "asr_device" in overrides and overrides["asr_device"] is not None:
        asr = replace(asr, device=str(overrides["asr_device"]))
    if "language" in overrides and overrides["language"] is not None:
        asr = replace(asr, language=str(overrides["language"]))
    if "openvino_device" in overrides and overrides["openvino_device"] is not None:
        asr = replace(asr, openvino_device=str(overrides["openvino_device"]))
    if "model_dir" in overrides and overrides["model_dir"] is not None:
        asr = replace(asr, model_dir=str(overrides["model_dir"]))
    if "whisper_cpp_bin" in overrides and overrides["whisper_cpp_bin"] is not None:
        asr = replace(asr, whisper_cpp_bin=str(overrides["whisper_cpp_bin"]))
    if "diarization_backend" in overrides and overrides["diarization_backend"] is not None:
        diar = replace(diar, backend=str(overrides["diarization_backend"]))
    if "llm_base_url" in overrides and overrides["llm_base_url"] is not None:
        llm = replace(llm, base_url=str(overrides["llm_base_url"]))
    if "llm_model" in overrides and overrides["llm_model"] is not None:
        llm = replace(llm, model=str(overrides["llm_model"]), model_locked=True)
    if "llm_polish" in overrides and overrides["llm_polish"] is not None:
        llm = replace(llm, polish=bool(overrides["llm_polish"]))
    if "context_tokens" in overrides and overrides["context_tokens"] is not None:
        llm = replace(llm, context_tokens=int(overrides["context_tokens"]))
    if "reserve_tokens" in overrides and overrides["reserve_tokens"] is not None:
        llm = replace(llm, reserve_tokens=int(overrides["reserve_tokens"]))
    if "temperature" in overrides and overrides["temperature"] is not None:
        llm = replace(llm, temperature=float(overrides["temperature"]))
    if "top_p" in overrides and overrides["top_p"] is not None:
        llm = replace(llm, top_p=float(overrides["top_p"]))
    if "top_k" in overrides and overrides["top_k"] is not None:
        llm = replace(llm, top_k=int(overrides["top_k"]))
    if "min_p" in overrides and overrides["min_p"] is not None:
        llm = replace(llm, min_p=float(overrides["min_p"]))
    if "presence_penalty" in overrides and overrides["presence_penalty"] is not None:
        llm = replace(llm, presence_penalty=float(overrides["presence_penalty"]))
    if "capture_backend" in overrides and overrides["capture_backend"] is not None:
        capture = replace(capture, backend=str(overrides["capture_backend"]))
    if "mic_device" in overrides and overrides["mic_device"] is not None:
        capture = replace(capture, mic_device=str(overrides["mic_device"]))
    if "system_device" in overrides and overrides["system_device"] is not None:
        capture = replace(capture, system_device=str(overrides["system_device"]))
    if "chunk_seconds" in overrides and overrides["chunk_seconds"] is not None:
        capture = replace(capture, chunk_seconds=float(overrides["chunk_seconds"]))
    if "sample_rate" in overrides and overrides["sample_rate"] is not None:
        capture = replace(capture, sample_rate=int(overrides["sample_rate"]))
    if "output_dir" in overrides and overrides["output_dir"] is not None:
        output_dir = Path(str(overrides["output_dir"])).expanduser()
    if "speaker_map" in overrides and overrides["speaker_map"] is not None:
        incoming = overrides["speaker_map"]
        if not isinstance(incoming, Mapping):
            raise ConfigError("speaker_map override must be a mapping")
        for key, value in incoming.items():
            speaker_map[str(key)] = str(value)

    return AppConfig(
        asr=asr,
        diarization=diar,
        llm=llm,
        capture=capture,
        speaker_map=speaker_map,
        output_dir=output_dir,
    )


def parse_speaker_map(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"Speaker mapping must look like id=name, got {item!r}")
        speaker_id, name = item.split("=", 1)
        speaker_id = speaker_id.strip()
        name = name.strip()
        if not speaker_id or not name:
            raise ConfigError(f"Speaker mapping must look like id=name, got {item!r}")
        mapping[speaker_id] = name
    return mapping


def language_hint(language: str) -> str | None:
    """Whisper language code. ``auto`` leaves detection to the model."""

    if language.strip().lower() in {"", "auto", "detect"}:
        return None
    return language.strip()


def _read_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a table: {path}")
    return data


def _merge_toml(config: AppConfig, data: Mapping[str, object]) -> AppConfig:
    _reject_secrets(data)
    asr = config.asr
    diar = config.diarization
    llm = config.llm
    capture = config.capture
    speaker_map = dict(config.speaker_map)
    output_dir = config.output_dir

    if "output_dir" in data:
        output_dir = Path(str(data["output_dir"])).expanduser()

    asr_table = _table(data.get("asr"), "asr")
    if "backend" in asr_table:
        asr = replace(asr, backend=str(asr_table["backend"]))
    if "model" in asr_table:
        asr = replace(asr, model=str(asr_table["model"]))
    if "fallback_model" in asr_table:
        asr = replace(asr, fallback_model=str(asr_table["fallback_model"]))
    if "language" in asr_table:
        asr = replace(asr, language=str(asr_table["language"]))
    if "device" in asr_table:
        asr = replace(asr, device=str(asr_table["device"]))
    if "openvino_device" in asr_table:
        asr = replace(asr, openvino_device=str(asr_table["openvino_device"]))
    if "model_dir" in asr_table:
        asr = replace(asr, model_dir=str(asr_table["model_dir"]))
    if "whisper_cpp_bin" in asr_table:
        asr = replace(asr, whisper_cpp_bin=str(asr_table["whisper_cpp_bin"]))

    diar_table = _table(data.get("diarization"), "diarization")
    if "backend" in diar_table:
        diar = replace(diar, backend=str(diar_table["backend"]))
    if "hf_token_env" in diar_table:
        diar = replace(diar, hf_token_env=str(diar_table["hf_token_env"]))

    llm_table = _table(data.get("llm"), "llm")
    if "base_url" in llm_table:
        llm = replace(llm, base_url=str(llm_table["base_url"]))
    if "api_key_env" in llm_table:
        llm = replace(llm, api_key_env=str(llm_table["api_key_env"]))
    if "model" in llm_table:
        llm = replace(llm, model=str(llm_table["model"]))
    if "fallback_model" in llm_table:
        llm = replace(llm, fallback_model=str(llm_table["fallback_model"]))
    if "polish" in llm_table:
        polish = llm_table["polish"]
        if not isinstance(polish, bool):
            raise ConfigError("[llm].polish must be a boolean")
        llm = replace(llm, polish=polish)
    if "context_tokens" in llm_table:
        llm = replace(llm, context_tokens=int(llm_table["context_tokens"]))
    if "reserve_tokens" in llm_table:
        llm = replace(llm, reserve_tokens=int(llm_table["reserve_tokens"]))
    if "temperature" in llm_table:
        llm = replace(llm, temperature=float(llm_table["temperature"]))
    if "top_p" in llm_table:
        llm = replace(llm, top_p=float(llm_table["top_p"]))
    if "top_k" in llm_table:
        llm = replace(llm, top_k=int(llm_table["top_k"]))
    if "min_p" in llm_table:
        llm = replace(llm, min_p=float(llm_table["min_p"]))
    if "presence_penalty" in llm_table:
        llm = replace(llm, presence_penalty=float(llm_table["presence_penalty"]))
    profiles = _merge_profile_table(llm.profiles, _table(llm_table.get("profiles"), "llm.profiles"))
    llm = replace(llm, profiles=profiles)

    capture_table = _table(data.get("capture"), "capture")
    if "backend" in capture_table:
        capture = replace(capture, backend=str(capture_table["backend"]))
    if "sample_rate" in capture_table:
        capture = replace(capture, sample_rate=int(capture_table["sample_rate"]))
    if "mic_device" in capture_table:
        capture = replace(capture, mic_device=str(capture_table["mic_device"]))
    if "system_device" in capture_table:
        capture = replace(capture, system_device=str(capture_table["system_device"]))
    if "chunk_seconds" in capture_table:
        capture = replace(capture, chunk_seconds=float(capture_table["chunk_seconds"]))

    speakers = _table(data.get("speakers"), "speakers")
    for key, value in speakers.items():
        speaker_map[str(key)] = str(value)

    return AppConfig(
        asr=asr,
        diarization=diar,
        llm=llm,
        capture=capture,
        speaker_map=speaker_map,
        output_dir=output_dir,
    )


def _apply_env(config: AppConfig, env: Mapping[str, str]) -> AppConfig:
    overrides: dict[str, object] = {}
    if env.get("MEETSCRIBE_ASR_BACKEND"):
        overrides["asr_backend"] = env["MEETSCRIBE_ASR_BACKEND"]
    if env.get("MEETSCRIBE_ASR_MODEL"):
        overrides["asr_model"] = env["MEETSCRIBE_ASR_MODEL"]
    if env.get("MEETSCRIBE_DIARIZATION"):
        overrides["diarization_backend"] = env["MEETSCRIBE_DIARIZATION"]
    if env.get("OPENAI_BASE_URL"):
        overrides["llm_base_url"] = env["OPENAI_BASE_URL"]
    if env.get("MEETSCRIBE_LLM_MODEL"):
        overrides["llm_model"] = env["MEETSCRIBE_LLM_MODEL"]
    if env.get("MEETSCRIBE_MODEL_DIR"):
        overrides["model_dir"] = env["MEETSCRIBE_MODEL_DIR"]
    if env.get("MEETSCRIBE_WHISPER_CPP"):
        overrides["whisper_cpp_bin"] = env["MEETSCRIBE_WHISPER_CPP"]
    if env.get("MEETSCRIBE_OPENVINO_DEVICE"):
        overrides["openvino_device"] = env["MEETSCRIBE_OPENVINO_DEVICE"]
    if env.get("MEETSCRIBE_MIC_DEVICE"):
        overrides["mic_device"] = env["MEETSCRIBE_MIC_DEVICE"]
    if env.get("MEETSCRIBE_SYSTEM_DEVICE"):
        overrides["system_device"] = env["MEETSCRIBE_SYSTEM_DEVICE"]
    if not overrides:
        return config
    return apply_overrides(config, overrides)


def _merge_profile_table(
    current: dict[str, SamplingProfile],
    table: Mapping[str, object],
) -> dict[str, SamplingProfile]:
    from meetscribe.llm.profiles import profile_from_table

    merged = dict(current)
    for name, raw in table.items():
        if not isinstance(raw, Mapping):
            raise ConfigError(f"[llm.profiles.{name}] must be a table")
        merged[str(name)] = profile_from_table(raw, merged.get(str(name)))
    return merged


def _table(value: object, section: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{section}] must be a table")
    return value


def _reject_secrets(data: Mapping[str, object]) -> None:
    for section_name in ("llm", "diarization", "asr", "capture"):
        section = data.get(section_name)
        if not isinstance(section, Mapping):
            continue
        leaked = SECRET_KEYS.intersection(section)
        if leaked:
            keys = ", ".join(sorted(leaked))
            raise ConfigError(
                f"Do not put {keys} in config.toml. "
                "Set OPENAI_API_KEY or HF_TOKEN in the environment."
            )
    if SECRET_KEYS.intersection(data):
        raise ConfigError(
            "Do not put secrets in config.toml. Set them in the environment."
        )

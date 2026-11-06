from pathlib import Path

import pytest

from meetscribe.config import apply_overrides, load_config, parse_speaker_map
from meetscribe.errors import ConfigError


def test_missing_default_file_uses_builtin_defaults(tmp_path: Path) -> None:
    config = load_config(environ={}, home=tmp_path)
    assert config.asr.backend == "auto"
    assert config.asr.model == "MediaTek-Research/Breeze-ASR-25"
    assert config.asr.fallback_model == "openai/whisper-large-v3-turbo"
    assert config.asr.language == "zh"
    assert config.diarization.backend == "auto"
    assert config.llm.base_url == "http://127.0.0.1:8080/v1"
    assert config.llm.model == "qwen2.5-14b-instruct"
    assert config.llm.context_tokens == 8192
    assert config.llm.polish is True
    assert config.api_key({}) is None
    assert config.resolved_api_key({}) == "llama-server"


def test_toml_env_and_cli_precedence(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
output_dir = "~/Meetings"

[asr]
backend = "faster-whisper"
model = "tiny"
language = "zh"

[diarization]
backend = "track"
hf_token_env = "MY_HF_TOKEN"

[llm]
base_url = "http://toml/v1"
api_key_env = "MY_LLM_KEY"
model = "toml-model"
polish = true

[capture]
backend = "file"

[speakers]
Me = "我"
SPEAKER_00 = "舊名"
""".strip(),
        encoding="utf-8",
    )
    config = load_config(
        path,
        environ={
            "OPENAI_BASE_URL": "http://env/v1",
            "MEETSCRIBE_ASR_MODEL": "from-env",
            "MY_LLM_KEY": "sk-from-env",
            "MY_HF_TOKEN": "hf-from-env",
        },
        overrides={
            "llm_base_url": "http://cli/v1",
            "asr_backend": "whisper.cpp",
            "speaker_map": {"SPEAKER_00": "陳怡君"},
            "llm_polish": False,
        },
    )
    assert config.asr.backend == "whisper.cpp"
    assert config.asr.model == "from-env"
    assert config.llm.base_url == "http://cli/v1"
    assert config.llm.model == "toml-model"
    assert config.llm.polish is False
    assert config.diarization.backend == "track"
    assert config.capture.backend == "file"
    assert config.output_dir == Path("~/Meetings").expanduser()
    assert config.speaker_map == {"Me": "我", "SPEAKER_00": "陳怡君"}
    assert config.api_key({"MY_LLM_KEY": "sk-from-env"}) == "sk-from-env"
    assert config.hf_token({"MY_HF_TOKEN": "hf-from-env"}) == "hf-from-env"
    assert "sk-from-env" not in repr(config)


def test_secrets_in_toml_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[llm]
api_key = "sk-should-not-load"
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="environment"):
        load_config(path, environ={})


def test_explicit_config_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.toml", environ={})


def test_xdg_config_home(tmp_path: Path) -> None:
    config_dir = tmp_path / "xdg"
    path = config_dir / "meetscribe" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text('[asr]\nbackend = "openvino"\n', encoding="utf-8")
    config = load_config(environ={"XDG_CONFIG_HOME": str(config_dir)}, home=tmp_path / "home")
    assert config.asr.backend == "openvino"


def test_invalid_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("asr = [\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(path, environ={})


def test_speaker_map_parser() -> None:
    assert parse_speaker_map(["Me=林可安", "SPEAKER_00=陳怡君"]) == {
        "Me": "林可安",
        "SPEAKER_00": "陳怡君",
    }
    with pytest.raises(ConfigError):
        parse_speaker_map(["林可安"])


def test_apply_overrides_ignores_empty() -> None:
    config = load_config(environ={}, home=Path("/tmp/meetscribe-empty-home"))
    updated = apply_overrides(config, {"asr_backend": None, "language": "en"})
    assert updated.asr.backend == "auto"
    assert updated.asr.language == "en"

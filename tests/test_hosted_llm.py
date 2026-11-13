from __future__ import annotations

from pathlib import Path

import pytest

from meetscribe.config import AppConfig, ConfigError, load_config
from meetscribe.llm.factory import build_polisher, build_summarizer
from meetscribe.llm.profiles import select_active_llm


class _Recorder:
    def __init__(self) -> None:
        self.client_kwargs: dict[str, object] = {}
        self.request: dict[str, object] = {}


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()

    class _Message:
        content = '{"utterances": []}'

    class _Choice:
        message = _Message()

    class _Response:
        def __init__(self) -> None:
            self.choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs: object) -> _Response:
            recorder.request = kwargs
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            recorder.client_kwargs = kwargs
            self.chat = _Chat()

    monkeypatch.setattr("openai.OpenAI", _Client)
    return recorder


def _hosted(tmp_path: Path, extra: str = "") -> AppConfig:
    path = tmp_path / "config.toml"
    path.write_text(
        "[llm]\n"
        'provider = "openai-compatible"\n'
        'base_url = "https://llm.example.test/v1"\n'
        'model = "vendor/chat-model"\n'
        'api_key_env = "HOSTED_KEY"\n' + extra,
        encoding="utf-8",
    )
    return load_config(path, environ={})


def test_hosted_request_omits_local_only_fields(tmp_path: Path, fake_openai) -> None:
    config = _hosted(tmp_path)
    polisher = build_polisher(config, environ={"HOSTED_KEY": "sk-test"})
    polisher.client.complete(system="系統", user="使用者")
    request = fake_openai.request
    assert request["model"] == "vendor/chat-model"
    for field in ("extra_body", "temperature", "top_p", "presence_penalty"):
        assert field not in request
    assert fake_openai.client_kwargs["api_key"] == "sk-test"
    assert fake_openai.client_kwargs["max_retries"] == 2
    assert fake_openai.client_kwargs["base_url"] == "https://llm.example.test/v1"


def test_hosted_request_sends_only_configured_sampling(tmp_path: Path, fake_openai) -> None:
    config = _hosted(tmp_path, "temperature = 0.3\n")
    build_summarizer(config, environ={"HOSTED_KEY": "k"}).client.complete(system="s", user="u")
    assert fake_openai.request["temperature"] == 0.3
    assert "top_p" not in fake_openai.request
    assert "extra_body" not in fake_openai.request


def test_hosted_provider_never_probes_model_list(tmp_path: Path, monkeypatch) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("a hosted /v1/models must not pick the model")

    monkeypatch.setattr("meetscribe.llm.profiles.fetch_loaded_model", unexpected)
    active = select_active_llm(_hosted(tmp_path))
    assert (active.model, active.source) == ("vendor/chat-model", "config")


def test_hosted_provider_requires_base_url_and_model(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[llm]\nprovider = "openai-compatible"\n', encoding="utf-8")
    config = load_config(path, environ={})
    assert (config.llm.base_url, config.llm.model) == ("", "")
    with pytest.raises(ConfigError, match="base_url"):
        select_active_llm(config)


@pytest.mark.parametrize("alias", ["openai", "remote", "cloud", "OpenAI-Compatible"])
def test_provider_aliases(tmp_path: Path, alias: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'[llm]\nprovider = "{alias}"\nmodel = "m"\n', encoding="utf-8")
    assert load_config(path, environ={}).llm.provider == "openai-compatible"


def test_api_key_cannot_live_in_config(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[llm]\nprovider = "openai"\napi_key = "sk-leak"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path, environ={})


def test_local_provider_still_sends_llama_extras(fake_openai, monkeypatch) -> None:
    monkeypatch.setattr("meetscribe.llm.profiles.fetch_loaded_model", lambda *a, **k: None)
    build_polisher(AppConfig(), environ={}).client.complete(system="s", user="u")
    assert fake_openai.request["extra_body"]["chat_template_kwargs"] == {  # type: ignore[index]
        "enable_thinking": False
    }
    assert fake_openai.client_kwargs["max_retries"] == 0


def test_placeholder_values_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[llm]\nprovider = "openai-compatible"\n'
        'base_url = "https://YOUR-PROVIDER.example/v1"\nmodel = "YOUR-MODEL-ID"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="placeholder"):
        select_active_llm(load_config(path, environ={}))

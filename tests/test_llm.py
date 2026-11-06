import platform

import pytest

from meetscribe.config import AppConfig, apply_overrides, llm_provider_for
from meetscribe.errors import LlmError
from meetscribe.llm.client import extract_json_object, strip_thinking
from meetscribe.llm.budget import chunk_items, estimate_tokens
from meetscribe.llm.factory import build_polisher, build_summarizer
from meetscribe.llm.polisher import OpenAIPolisher
from meetscribe.llm.prompts import POLISH_SYSTEM, SUMMARY_MERGE_SYSTEM, SUMMARY_SYSTEM
from meetscribe.llm.summarizer import OpenAISummarizer
from meetscribe.types import AttributedUtterance


class FakeChat:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.payload


def _utterance(text: str, index: int = 0) -> AttributedUtterance:
    return AttributedUtterance(
        speaker_id="Me",
        text=text,
        start=float(index),
        end=float(index) + 0.8,
        track_id="mic",
    )


def test_extract_json_from_fence_and_prose() -> None:
    fenced = '```json\n{"overview": "好"}\n```'
    assert extract_json_object(fenced) == {"overview": "好"}
    prose = '結果如下：\n{"utterances": [{"index": 0, "text": "API"}]}\n謝謝'
    assert extract_json_object(prose)["utterances"][0]["text"] == "API"
    with pytest.raises(LlmError):
        extract_json_object("沒有 JSON")


def test_polisher_rewrites_by_index_and_keeps_gaps() -> None:
    chat = FakeChat('{"utterances": [{"index": 1, "text": "第二句 API"}]}')
    polished = OpenAIPolisher(chat).polish([_utterance("第一句", 0), _utterance("第二句", 1)])
    assert polished[0].polished_text == "第一句"
    assert polished[1].polished_text == "第二句 API"
    assert polished[0].text == "第一句"
    assert "繁體中文" in chat.calls[0][0]


def test_summarizer_parses_action_items() -> None:
    payload = """
    {
      "overview": "決定用 feature flag。",
      "key_points": ["timeout 800ms"],
      "decisions": ["不上不可逆 migration"],
      "action_items": [
        {"task": "補 runbook", "owner": "林可安", "due": "下週三"},
        {"task": "看 dashboard", "owner": null, "due": "null"}
      ]
    }
    """
    summary = OpenAISummarizer(FakeChat(payload)).summarize(
        [_utterance("我們談 API")],
        title="週會",
        speaker_map={"Me": "林可安"},
    )
    assert summary.overview == "決定用 feature flag。"
    assert summary.key_points == ("timeout 800ms",)
    assert summary.action_items[0].owner == "林可安"
    assert summary.action_items[1].owner is None
    assert summary.action_items[1].due is None


def test_factory_uses_llama_server_without_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("meetscribe.llm.profiles.fetch_loaded_model", lambda *args, **kwargs: None)
    config = AppConfig()
    polisher = build_polisher(config, environ={})
    summarizer = build_summarizer(config, environ={})
    assert isinstance(polisher, OpenAIPolisher)
    assert isinstance(summarizer, OpenAISummarizer)
    assert polisher.client.api_key == "llama-server"
    assert config.llm.provider == llm_provider_for(platform.system(), platform.machine())
    assert polisher.client.base_url == config.llm.base_url
    assert polisher.client.model == config.llm.model
    assert polisher.client.temperature == 1.0
    assert polisher.client.top_p == 0.95
    assert polisher.client.top_k == 64
    assert polisher.client.presence_penalty is None


def test_sampling_profile_follows_the_loaded_model() -> None:
    from meetscribe.llm.profiles import select_active_llm

    config = AppConfig()
    qwen = select_active_llm(config, detected="unsloth/Qwen3.5-9B-GGUF", probe=False)
    assert qwen.source == "server"
    assert qwen.profile == "qwen3.5-9b"
    assert qwen.sampling.temperature == 0.7
    assert qwen.sampling.top_p == 0.8
    assert qwen.sampling.top_k == 20
    assert qwen.sampling.presence_penalty == 1.5

    locked = select_active_llm(
        apply_overrides(config, {"llm_model": "gemma-4-12b-it"}),
        detected="qwen3.5-9b",
        probe=False,
    )
    assert locked.source == "cli"
    assert locked.model == "gemma-4-12b-it"
    assert locked.sampling.temperature == 1.0
    assert locked.sampling.presence_penalty is None

    overridden = select_active_llm(
        apply_overrides(config, {"temperature": 0.2}),
        detected=None,
        probe=False,
    )
    assert overridden.source == "config"
    assert overridden.sampling.temperature == 0.2
    assert overridden.sampling.top_k == 64

    mlx_qwen = select_active_llm(config, detected="mlx-community/Qwen3.5-9B-MLX-4bit", probe=False)
    assert mlx_qwen.profile == "mlx-community/Qwen3.5-9B-MLX-4bit"
    assert mlx_qwen.sampling.presence_penalty == 1.5
    mlx_gemma = select_active_llm(config, detected="gemma-4-12B-it-4bit", probe=False)
    assert mlx_gemma.profile == "gemma-4-12B-it-4bit"
    assert mlx_gemma.sampling.temperature == 1.0
    assert mlx_gemma.sampling.top_k == 64
    assert mlx_gemma.sampling.presence_penalty is None


def test_request_disables_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    from meetscribe.llm.client import OpenAIChatClient

    captured: dict[str, object] = {}

    class _Message:
        content = '{"overview": "好"}'

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs: object) -> _Response:
            captured.update(kwargs)
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.chat = _Chat()

    monkeypatch.setattr("openai.OpenAI", _Client)
    text = OpenAIChatClient(
        base_url="http://127.0.0.1:8000/v1",
        api_key="llama-server",
        model="gemma-4-12B-it-4bit",
        top_k=64,
        min_p=0.0,
    ).complete(system="系統", user="使用者")
    assert text == '{"overview": "好"}'
    extra = captured["extra_body"]
    assert extra == {
        "top_k": 64,
        "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_strip_thinking_blocks() -> None:
    qwen = "<think>\n先想一下\n</think>\n{\"overview\": \"好\"}"
    assert strip_thinking(qwen) == '{"overview": "好"}'
    gemma = "<|channel>thought\n內部推理\n<channel|>{\"utterances\": []}"
    assert strip_thinking(gemma) == '{"utterances": []}'
    empty = "<|channel>thought\n<channel|>答案"
    assert strip_thinking(empty) == "答案"


def test_prompts_keep_english_terms_and_traditional_chinese() -> None:
    for prompt in (POLISH_SYSTEM, SUMMARY_SYSTEM, SUMMARY_MERGE_SYSTEM):
        assert "繁體中文" in prompt
        assert "英文" in prompt
        assert "zh-TW" in prompt


def test_polish_and_summary_split_on_token_budget() -> None:
    assert estimate_tokens("API") < estimate_tokens("會議摘要")
    chat = FakeChat('{"utterances": []}')
    utterances = [_utterance("甲" * 40, index) for index in range(3)]
    OpenAIPolisher(chat, token_budget=30).polish(utterances)
    assert len(chat.calls) == 3

    summary_chat = FakeChat("")
    calls: list[str] = []

    def complete(*, system: str, user: str) -> str:
        calls.append(system)
        return '{"overview": "分段", "key_points": ["一"], "decisions": [], "action_items": []}'

    summary_chat.complete = complete  # type: ignore[method-assign]
    long_lines = [f"第{index}句 " + ("討論 API " * 30) for index in range(4)]
    chunks = chunk_items(long_lines, lambda line: line, 80)
    assert len(chunks) > 1
    OpenAISummarizer(summary_chat, token_budget=80).summarize(
        [_utterance("討論 API " * 30, index) for index in range(4)],
        title="週會",
    )
    assert calls[0] == SUMMARY_SYSTEM
    assert calls[-1] == SUMMARY_MERGE_SYSTEM
    assert len(calls) >= 3

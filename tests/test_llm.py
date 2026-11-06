import pytest

from meetscribe.config import AppConfig
from meetscribe.errors import LlmError
from meetscribe.llm.client import extract_json_object
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


def test_factory_uses_llama_server_without_an_api_key() -> None:
    config = AppConfig()
    polisher = build_polisher(config, environ={})
    summarizer = build_summarizer(config, environ={})
    assert isinstance(polisher, OpenAIPolisher)
    assert isinstance(summarizer, OpenAISummarizer)
    assert polisher.client.api_key == "llama-server"
    assert polisher.client.base_url == "http://127.0.0.1:8080/v1"
    assert polisher.client.model == "qwen2.5-14b-instruct"


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

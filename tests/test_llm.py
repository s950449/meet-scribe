import pytest

from meetscribe.config import AppConfig
from meetscribe.errors import LlmError
from meetscribe.llm.client import extract_json_object
from meetscribe.llm.factory import build_polisher, build_summarizer
from meetscribe.llm.polisher import IdentityPolisher, OpenAIPolisher
from meetscribe.llm.prompts import POLISH_SYSTEM, SUMMARY_SYSTEM
from meetscribe.llm.summarizer import ExtractiveSummarizer, OpenAISummarizer
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


def test_factory_falls_back_without_api_key() -> None:
    config = AppConfig()
    polisher = build_polisher(config, environ={})
    summarizer = build_summarizer(config, environ={})
    assert isinstance(polisher, IdentityPolisher)
    assert isinstance(summarizer, ExtractiveSummarizer)
    assert "OPENAI_API_KEY" in polisher.reason


def test_prompts_keep_english_terms_and_traditional_chinese() -> None:
    for prompt in (POLISH_SYSTEM, SUMMARY_SYSTEM):
        assert "繁體中文" in prompt
        assert "英文" in prompt
        assert "zh-TW" in prompt

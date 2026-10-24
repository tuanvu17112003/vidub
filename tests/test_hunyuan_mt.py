"""Tests for the Hunyuan HY-MT1.5-7B MT engine (M5-T1).

No live Ollama is needed: the HTTP layer is stubbed by overriding ``_chat`` (or
``_post``). One live smoke test runs only when ``HUNYUAN_LIVE=1`` is set.
"""

from __future__ import annotations

import os

import pytest

from vidub.config import (
    DEFAULT_HUNYUAN_HOST,
    DEFAULT_HUNYUAN_MODEL,
    JobConfig,
    check_mt_engine,
)
from vidub.engines.mt.base import BaseTranslator, TranslationError
from vidub.engines.mt.hunyuan_mt import (
    FULL_LANG_NAMES,
    HunyuanMTTranslator,
    _budget_clause,
    _lang_full,
    _template_contextual,
    _template_direct,
    _template_fusion,
    _template_terminology,
)


class StubChatHunyuan(HunyuanMTTranslator):
    """HunyuanTranslator whose ``_chat`` is scripted (never touches the network)."""

    def __init__(self, respond, **kwargs) -> None:  # noqa: ANN001 - test double
        super().__init__(**kwargs)
        self._respond = respond
        self.calls: list[tuple[str, float]] = []

    def _chat(self, user: str, temperature: float = 0.7) -> str:
        self.calls.append((user, temperature))
        return self._respond(user, temperature)


class FakeFallback(BaseTranslator):
    """Deterministic fallback translator that records what it was asked."""

    def __init__(self, value: str = "Xin chào thế giới") -> None:
        self.value = value
        self.calls: list[list[str]] = []

    def translate(self, texts, src_lang, tgt_lang):  # noqa: ANN001
        self.calls.append(list(texts))
        return [self.value for _ in texts]


# ------------------------------------------------------------------ templates
def test_template_direct_exact() -> None:
    assert (
        _template_direct("Vietnamese", "Hello")
        == "Translate the following segment into Vietnamese, without additional "
        "explanation.\n\nHello"
    )


def test_template_contextual_exact() -> None:
    assert (
        _template_contextual("Vietnamese", "prev", "Hello", "next")
        == "prev\nnext\nBased on the information above, translate the following "
        "text into Vietnamese. Note: do not translate the text above, and do "
        "not add any explanation:\nHello"
    )


def test_template_contextual_single_neighbour() -> None:
    assert (
        _template_contextual("Vietnamese", "prev", "Hello", "")
        == "prev\nBased on the information above, translate the following text "
        "into Vietnamese. Note: do not translate the text above, and do not add "
        "any explanation:\nHello"
    )


def test_template_contextual_no_neighbours_equals_direct() -> None:
    assert _template_contextual("Vietnamese", "", "Hello", "") == _template_direct(
        "Vietnamese", "Hello"
    )


def test_template_terminology_exact() -> None:
    assert (
        _template_terminology("Vietnamese", {"GPU": "bộ xử lý đồ họa"}, "Hello")
        == "Refer to the following translations:\n"
        "GPU should be translated as bộ xử lý đồ họa\n\n"
        "Translate the following text into Vietnamese. Only output the "
        "translation, without any additional explanation:\nHello"
    )


def test_lang_full_known_and_unknown() -> None:
    assert _lang_full("vi") == "Vietnamese"
    assert _lang_full("EN") == "English"
    assert _lang_full("xx") == "xx"
    assert FULL_LANG_NAMES["ja"] == "Japanese"


# ------------------------------------------------------- budget templates
def test_budget_clause_exact_wording() -> None:
    assert _budget_clause(42) == (
        " At most 42 characters, keep the meaning exactly, and output only the "
        "translation."
    )


def test_template_direct_with_budget_inserts_clause_before_source() -> None:
    assert _template_direct("Vietnamese", "Hello", 42) == (
        "Translate the following segment into Vietnamese, without additional "
        "explanation. At most 42 characters, keep the meaning exactly, and "
        "output only the translation.\n\nHello"
    )
    # no budget -> the official template, unchanged
    assert _template_direct("Vietnamese", "Hello") == (
        "Translate the following segment into Vietnamese, without additional "
        "explanation.\n\nHello"
    )


def test_template_contextual_with_budget() -> None:
    prompt = _template_contextual("Vietnamese", "prev", "Hello", "next", 55)
    assert "At most 55 characters, keep the meaning exactly" in prompt
    assert prompt.endswith(":\nHello")
    assert prompt != _template_contextual("Vietnamese", "prev", "Hello", "next")
    # no neighbours -> still exactly the direct template (now with the budget)
    assert _template_contextual("Vietnamese", "", "Hello", "", 42) == _template_direct(
        "Vietnamese", "Hello", 42
    )


def test_template_terminology_with_budget() -> None:
    prompt = _template_terminology(
        "Vietnamese", {"GPU": "bộ xử lý đồ họa"}, "Hello", 33
    )
    assert "At most 33 characters, keep the meaning exactly" in prompt
    assert prompt.endswith(":\nHello")
    assert "GPU should be translated as bộ xử lý đồ họa" in prompt


# --------------------------------------------------------- M5-T2 refine API
def test_template_fusion_exact() -> None:
    assert _template_fusion(
        "English", "Vietnamese", "Hello world", ["Xin chào", "Chào thế giới"]
    ) == (
        "Analyze the following multiple Vietnamese translations of the English "
        "segment surrounded in triple backticks and generate a single refined "
        "Vietnamese translation. Only output the refined translation, do not "
        "explain.\n\nThe English segment:\n```Hello world```\n\nThe multiple "
        "Vietnamese translations:\n"
        "1. ```Xin chào```\n2. ```Chào thế giới```"
    )


def test_sample_candidates_dedupes_and_caps() -> None:
    responses = iter(["Xin chào", "Xin chào", "Tạm biệt", "Khác"])
    engine = StubChatHunyuan(lambda user, temperature: next(responses))

    out = engine.sample_candidates("Hello world", "en", "vi", 4)

    assert out == ["Xin chào", "Tạm biệt", "Khác"]
    assert len(engine.calls) == 4  # exactly k chat calls
    assert all(temperature == 0.7 for _user, temperature in engine.calls)
    assert engine.calls[0][0] == _template_direct("Vietnamese", "Hello world")


def test_sample_candidates_empty_source_or_zero_k() -> None:
    engine = StubChatHunyuan(lambda user, temperature: "Xin chào")

    assert engine.sample_candidates("   ", "en", "vi", 3) == []
    assert engine.sample_candidates("Hello", "en", "vi", 0) == []
    assert engine.calls == []


def test_fuse_uses_fusion_template_and_strips() -> None:
    engine = StubChatHunyuan(lambda user, temperature: "  Bản dịch gộp  ")

    out = engine.fuse("Hello world", ["Xin chào", "Chào thế giới"], "en", "vi")

    assert out == "Bản dịch gộp"
    assert engine.calls[0][0] == _template_fusion(
        "English", "Vietnamese", "Hello world", ["Xin chào", "Chào thế giới"]
    )


# ------------------------------------------------------------------ HTTP layer
def test_chat_payload_plain_text_no_system_no_format(monkeypatch) -> None:  # noqa: ANN001
    engine = HunyuanMTTranslator(model="hunyuan-test", host="http://host")
    captured: dict = {}

    def fake_post(path, payload):  # noqa: ANN001 - test double
        captured["path"] = path
        captured["payload"] = payload
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(engine, "_post", fake_post)
    engine._chat("xin chào")

    assert captured["path"] == "/api/chat"
    payload = captured["payload"]
    assert payload["model"] == "hunyuan-test"
    assert payload["stream"] is False
    assert "format" not in payload  # translation specialists take plain text
    assert payload["messages"] == [{"role": "user", "content": "xin chào"}]
    assert all(message["role"] != "system" for message in payload["messages"])
    assert payload["options"] == {
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.6,
        "repeat_penalty": 1.05,
    }


def test_post_connection_error_is_actionable(monkeypatch) -> None:  # noqa: ANN001
    import urllib.error

    engine = HunyuanMTTranslator(host="http://127.0.0.1:11434")

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003 - test double
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(
        "vidub.engines.mt.hunyuan_mt.urllib.request.urlopen", boom
    )
    with pytest.raises(TranslationError) as excinfo:
        engine._post("/api/chat", {})
    message = str(excinfo.value)
    assert "ollama serve" in message
    assert "http://127.0.0.1:11434" in message


# ------------------------------------------------------------------ happy path
def test_happy_path_returns_translation_and_counts() -> None:
    def respond(user, temperature):  # noqa: ANN001 - test double
        return "Xin chào thế giới"

    engine = StubChatHunyuan(respond)
    out = engine.translate(["Hello world"], "en", "vi")

    assert out == ["Xin chào thế giới"]
    assert engine.stats == {"llm_used": 1, "fallback_used": 0, "validator_retries": 0}
    assert len(engine.calls) == 1
    assert engine.calls[0][1] == 0.7  # default temperature
    # the direct template is what was sent
    assert engine.calls[0][0] == _template_direct("Vietnamese", "Hello world")


# --------------------------------------------------------- budget plumbing
def test_translate_no_budgets_prompt_is_byte_identical() -> None:
    engine = StubChatHunyuan(lambda user, temperature: "Xin chào thế giới")

    engine.translate(["Hello world"], "en", "vi")

    assert engine.calls[0][0] == _template_direct("Vietnamese", "Hello world")


def test_translate_with_budgets_states_the_number() -> None:
    engine = StubChatHunyuan(lambda user, temperature: "Xin chào thế giới")

    out = engine.translate(["Hello world"], "en", "vi", budgets=[40])

    assert out == ["Xin chào thế giới"]
    assert "At most 40 characters, keep the meaning exactly" in engine.calls[0][0]


def test_translate_contextual_with_per_segment_budgets() -> None:
    engine = StubChatHunyuan(lambda user, temperature: "Xin chào thế giới")

    engine.translate_contextual(["Hello world", "Second line"], "en", "vi", budgets=[40, 60])

    assert "At most 40 characters" in engine.calls[0][0]
    assert "At most 60 characters" in engine.calls[1][0]
    # both still use the official contextual template shape
    assert "Based on the information above" in engine.calls[0][0]


def test_sample_candidates_with_budget() -> None:
    engine = StubChatHunyuan(lambda user, temperature: "Xin chào")

    engine.sample_candidates("Hello world", "en", "vi", 1, budget_chars=25)

    assert "At most 25 characters" in engine.calls[0][0]


def test_budgets_length_mismatch_raises() -> None:
    engine = StubChatHunyuan(lambda user, temperature: "Xin chào")

    with pytest.raises(ValueError):
        engine.translate(["a", "b"], "en", "vi", budgets=[10])
    with pytest.raises(ValueError):
        engine.translate_contextual(["a"], "en", "vi", budgets=[1, 2])


# --------------------------------------------------------- validator gating
def test_validator_failure_retries_once_at_03_then_accepts() -> None:
    bad = "hello there friend"  # English -> fails the Vietnamese lang check

    def respond(user, temperature):  # noqa: ANN001 - test double
        return bad if temperature == 0.7 else "Xin chào các bạn"

    engine = StubChatHunyuan(respond)
    out = engine.translate(["Hello everyone friend"], "en", "vi")

    assert out == ["Xin chào các bạn"]
    assert engine.stats == {"llm_used": 1, "fallback_used": 0, "validator_retries": 1}
    assert len(engine.calls) == 2
    assert engine.calls[1][1] == 0.3  # the retry path is cooler
    assert "failed validation" in engine.calls[1][0]


def test_double_failure_uses_envit5_fallback() -> None:
    bad = "hello there friend"

    def respond(user, temperature):  # noqa: ANN001 - test double
        return bad

    fallback = FakeFallback("Xin chào các bạn")
    engine = StubChatHunyuan(respond, fallback=fallback)
    out = engine.translate(["Hello everyone friend"], "en", "vi")

    assert out == ["Xin chào các bạn"]
    assert engine.stats == {"llm_used": 0, "fallback_used": 1, "validator_retries": 1}
    assert fallback.calls == [["Hello everyone friend"]]


def test_fallback_is_lazy_when_none_provided(monkeypatch) -> None:  # noqa: ANN001
    created: list[object] = []

    class FakeEnvit5(BaseTranslator):
        def __init__(self) -> None:
            created.append(self)

        def translate(self, texts, src_lang, tgt_lang):  # noqa: ANN001
            return ["Xin chào các bạn" for _ in texts]

    monkeypatch.setattr(
        "vidub.engines.mt.envit5_mt.Envit5Translator", FakeEnvit5
    )

    def respond(user, temperature):  # noqa: ANN001 - test double
        return "hello there friend"

    engine = StubChatHunyuan(respond)
    assert created == []  # not constructed until the fallback is needed
    out = engine.translate(["Hello everyone friend"], "en", "vi")
    assert out == ["Xin chào các bạn"]
    assert len(created) == 1
    assert engine.stats["fallback_used"] == 1


# --------------------------------------------------------------- empty source
def test_empty_source_returns_empty_without_chat() -> None:
    def respond(user, temperature):  # noqa: ANN001 - test double
        raise AssertionError("no chat call expected for an empty source")

    engine = StubChatHunyuan(respond)
    out = engine.translate(["", "   "], "en", "vi")

    assert out == ["", ""]
    assert engine.calls == []
    assert engine.stats == {"llm_used": 0, "fallback_used": 0, "validator_retries": 0}


# --------------------------------------------------------- contextual mode
def test_contextual_uses_neighbours_and_terminology() -> None:
    def respond(user, temperature):  # noqa: ANN001 - test double
        return "Xin chào thế giới"

    engine = StubChatHunyuan(respond)
    engine.translate_contextual(
        ["Hello world", "Second line"], "en", "vi"
    )

    # first segment sees the following neighbour, second sees the previous one
    assert "Second line" in engine.calls[0][0]
    assert "Based on the information above" in engine.calls[0][0]
    assert "Hello world" in engine.calls[1][0]

    # a glossary switches to the official terminology template
    glossary_engine = StubChatHunyuan(respond)
    glossary_engine.glossary = {"GPU": "bộ xử lý đồ họa"}
    glossary_engine.translate_contextual(["The GPU is fast"], "en", "vi")
    assert "Refer to the following translations:" in glossary_engine.calls[0][0]
    assert "GPU should be translated as bộ xử lý đồ họa" in glossary_engine.calls[0][0]


# --------------------------------------------------------------- config + build
def test_check_mt_engine_hunyuan() -> None:
    assert check_mt_engine("hunyuan") == "hunyuan"
    assert check_mt_engine("HUNYUAN") == "hunyuan"


def test_build_translator_maps_hunyuan_with_cfg() -> None:
    from vidub.manifest import JobManifest
    from vidub.stages.translate import build_translator

    manifest = JobManifest(mt_engine="hunyuan")
    cfg = JobConfig(
        mt_engine="hunyuan",
        hunyuan_model="foo",
        hunyuan_host="http://example:1234",
    )

    engine = build_translator(manifest, cfg)

    assert isinstance(engine, HunyuanMTTranslator)
    assert engine.model == "foo"
    assert engine.host == "http://example:1234"


def test_build_translator_hunyuan_defaults_without_cfg() -> None:
    from vidub.manifest import JobManifest
    from vidub.stages.translate import build_translator

    assert isinstance(
        build_translator(JobManifest(mt_engine="hunyuan")), HunyuanMTTranslator
    )


# ------------------------------------------------------------------ CLI parse
def test_cli_hunyuan_flags_reach_config() -> None:
    from vidub.cli import _build_cfg, build_parser

    parser = build_parser()
    cfg = _build_cfg(
        parser.parse_args(
            [
                "run",
                "x.mp4",
                "--mt-engine",
                "hunyuan",
                "--hunyuan-model",
                "foo",
                "--hunyuan-host",
                "http://x",
            ]
        )
    )

    assert cfg.mt_engine == "hunyuan"
    assert cfg.hunyuan_model == "foo"
    assert cfg.hunyuan_host == "http://x"


def test_cli_hunyuan_defaults() -> None:
    from vidub.cli import _build_cfg, build_parser

    cfg = _build_cfg(build_parser().parse_args(["run", "x.mp4"]))

    assert cfg.hunyuan_model == DEFAULT_HUNYUAN_MODEL
    assert cfg.hunyuan_host == DEFAULT_HUNYUAN_HOST


# --------------------------------------------------------------- live smoke
@pytest.mark.skipif(
    os.environ.get("HUNYUAN_LIVE") != "1",
    reason="set HUNYUAN_LIVE=1 to run against a live Hunyuan/hunyuan server",
)
def test_live_hunyuan_translation() -> None:
    engine = HunyuanMTTranslator(host=DEFAULT_HUNYUAN_HOST, model=DEFAULT_HUNYUAN_MODEL)
    try:
        out = engine.translate(
            ["Hello world.", "The GPU costs 300 dollars."], "en", "vi"
        )
    except TranslationError as exc:
        pytest.skip(f"Hunyuan server unreachable: {exc}")

    assert len(out) == 2
    assert out[0].strip()
    assert out[1].strip()
    assert "300" in out[1]

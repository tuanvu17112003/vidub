"""Tests for the Ollama LLM MT engine (M4-T5).

No live Ollama is needed: the HTTP layer is stubbed by overriding ``_chat`` (or
monkeypatching ``urllib.request.urlopen``). One live smoke test runs only when
``OLLAMA_LIVE=1`` is set.
"""

from __future__ import annotations

import json
import os
import urllib.error
from pathlib import Path

import pytest

from vidub.config import JobConfig
from vidub.engines.mt.base import BaseTranslator, TranslationError
from vidub.engines.mt.ollama_mt import (
    OllamaTranslator,
    _lang_ok,
    _length_ok,
    _numbers_ok,
    _parse_critiques,
    _parse_translations,
    _sys_improve,
    _sys_reflect,
    _sys_translate,
)
from vidub.manifest import JobManifest
from vidub.stages.translate import SCRIPT_NAME, build_translator, translate

EN_VI_TRANSLATE_SYSTEM = _sys_translate("en", "vi")
EN_VI_REFLECT_SYSTEM = _sys_reflect("en", "vi")
EN_VI_IMPROVE_SYSTEM = _sys_improve("en", "vi")


class StubChatOllama(OllamaTranslator):
    """OllamaTranslator whose ``_chat`` is scripted (never touches the network)."""

    def __init__(self, respond, **kwargs) -> None:  # noqa: ANN001 - test double
        super().__init__(**kwargs)
        self._respond = respond
        self.calls: list[tuple[str, str, bool]] = []

    def _chat(self, system: str, user: str, json_mode: bool = False) -> str:
        self.calls.append((system, user, json_mode))
        return self._respond(system, user, json_mode)

    def translate_calls(self) -> list[tuple[str, str, bool]]:
        return [call for call in self.calls if call[0] == EN_VI_TRANSLATE_SYSTEM]

    def reflect_calls(self) -> list[tuple[str, str, bool]]:
        return [call for call in self.calls if call[0] == EN_VI_REFLECT_SYSTEM]

    def improve_calls(self) -> list[tuple[str, str, bool]]:
        return [call for call in self.calls if call[0] == EN_VI_IMPROVE_SYSTEM]


class FakeFallback(BaseTranslator):
    """Deterministic fallback translator that records what it was asked."""

    def __init__(self, value: str = "Xin chào thế giới") -> None:
        self.value = value
        self.calls: list[list[str]] = []

    def translate(self, texts, src_lang, tgt_lang):  # noqa: ANN001
        self.calls.append(list(texts))
        return [self.value for _ in texts]


# ------------------------------------------------------------------ validators
def test_lang_validator_matrix() -> None:
    assert _lang_ok("Xin chào các bạn", "vi", "en")
    assert not _lang_ok("hello there friend", "vi", "en")
    # a Vietnamese source already yields Vietnamese-looking text -> skipped
    assert _lang_ok("hello there", "vi", "vi")
    # any non-vi target passes (no cheaper heuristic)
    assert _lang_ok("hello there", "en", "en")


def test_number_validator_matrix() -> None:
    assert _numbers_ok("The GPU needs 16 GB", "GPU cần 16 GB")
    assert _numbers_ok("no digits here", "không có số")
    assert not _numbers_ok("The GPU needs 16 GB", "GPU cần mười sáu GB")
    # a completely different number never satisfies the source digits
    assert not _numbers_ok("The GPU needs 16 GB", "GPU cần 24 GB")


def test_length_validator_matrix() -> None:
    assert _length_ok("one two", "một hai ba")  # 3 <= 2.2 * 2
    assert _length_ok("", "một")  # empty source -> max(0, 1) = 1
    assert not _length_ok("one", "a b c d e")  # 5 > 2.2 * 1


# ------------------------------------------------------------------ JSON parsing
def test_parse_translations_edges() -> None:
    assert _parse_translations("garbage, not json", {0}) == {}
    good = json.dumps({"translations": [{"index": 0, "text": "ok"}]})
    assert _parse_translations(good, {0, 1}) == {0: "ok"}
    assert _parse_translations(json.dumps({"translations": []}), {0}) == {}
    # missing index / non-numeric index / blank text are all dropped
    assert _parse_translations(
        json.dumps({"translations": [{"text": "no index"}]}), {0}
    ) == {}
    assert _parse_translations(
        json.dumps({"translations": [{"index": "x", "text": "bad"}]}), {0}
    ) == {}
    assert _parse_translations(
        json.dumps({"translations": [{"index": 0, "text": "   "}]}), {0}
    ) == {}
    # markdown-fenced / prose-wrapped JSON is still found
    fenced = 'Sure:\n```json\n{"translations": [{"index": 1, "text": "xin chào"}]}\n```'
    assert _parse_translations(fenced, {1}) == {1: "xin chào"}


def test_parse_critiques_only_major() -> None:
    raw = json.dumps(
        {
            "critiques": [
                {"index": 0, "issues": ["meaning changed"], "severity": "major"},
                {"index": 1, "issues": ["word choice"], "severity": "minor"},
                {"index": 2, "issues": ["x"], "severity": "MAJOR"},
            ]
        }
    )
    assert _parse_critiques(raw, {0, 1, 2}) == {0: ["meaning changed"], 2: ["x"]}
    assert _parse_critiques("nope", {0}) == {}


# ------------------------------------------------------------------ HTTP layer
def test_post_connection_error_is_actionable(monkeypatch) -> None:  # noqa: ANN001
    engine = OllamaTranslator(host="http://127.0.0.1:11434")

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003 - test double
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(
        "vidub.engines.mt.ollama_mt.urllib.request.urlopen", boom
    )
    with pytest.raises(TranslationError) as excinfo:
        engine._post("/api/chat", {})
    message = str(excinfo.value)
    assert "ollama serve" in message
    assert "http://127.0.0.1:11434" in message


def test_chat_sends_json_mode_and_temperature(monkeypatch) -> None:  # noqa: ANN001
    engine = OllamaTranslator(model="qwen2.5:7b", host="http://host")
    captured: dict = {}

    def fake_post(path, payload):  # noqa: ANN001 - test double
        captured["path"] = path
        captured["payload"] = payload
        return {"message": {"content": "{}"}}

    monkeypatch.setattr(engine, "_post", fake_post)
    engine._chat("system", "user", json_mode=True)

    assert captured["path"] == "/api/chat"
    assert captured["payload"]["model"] == "qwen2.5:7b"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["options"] == {"temperature": 0.2}


# -------------------------------------------------------------- initial pass
def test_initial_pass_parses_translations_and_uses_context() -> None:
    def respond(system, user, json_mode):  # noqa: ANN001 - test double
        assert json_mode
        assert "PREVIOUS (context only)" in user
        assert "NEXT (context only)" in user
        return json.dumps(
            {
                "translations": [
                    {"index": 0, "text": "Xin chào các bạn"},
                    {"index": 1, "text": "Thời tiết hôm nay đẹp"},
                ]
            }
        )

    engine = StubChatOllama(respond)
    out = engine.translate(
        ["Hello everyone", "The weather is nice today"], "en", "vi"
    )

    assert out == ["Xin chào các bạn", "Thời tiết hôm nay đẹp"]
    assert engine.stats == {"llm_used": 2, "fallback_used": 0, "validator_retries": 0}
    # plain translate skips reflect/improve entirely
    assert engine.reflect_calls() == []
    assert engine.improve_calls() == []
    # one batch call for two segments
    assert len(engine.translate_calls()) == 1


def test_initial_pass_retries_missing_and_malformed_individually() -> None:
    mapping = {
        "Hello everyone": {"index": 0, "text": "Xin chào các bạn"},
        "The weather is nice today": {"index": 1, "text": "Thời tiết hôm nay đẹp"},
    }

    def respond(system, user, json_mode):  # noqa: ANN001 - test double
        if user.count("TRANSLATE THIS") > 1:
            return "this is not JSON"  # malformed batch -> per-segment retry
        for source, item in mapping.items():
            if f"TRANSLATE THIS: {source}" in user:
                return json.dumps({"translations": [item]})
        return "{}"

    engine = StubChatOllama(respond)
    out = engine.translate(list(mapping), "en", "vi")

    assert out == ["Xin chào các bạn", "Thời tiết hôm nay đẹp"]
    # 1 malformed batch + 1 individual retry per segment
    assert len(engine.translate_calls()) == 3


# ------------------------------------------------------- validator gating
def test_validator_failure_retries_once_then_falls_back() -> None:
    bad = "hello there friend"  # no Vietnamese diacritics -> lang validator fails

    def respond(system, user, json_mode):  # noqa: ANN001 - test double
        return json.dumps({"translations": [{"index": 0, "text": bad}]})

    fallback = FakeFallback("Xin chào các bạn")
    engine = StubChatOllama(respond, fallback=fallback)
    out = engine.translate(["Hello everyone"], "en", "vi")

    assert out == ["Xin chào các bạn"]
    assert engine.stats == {"llm_used": 0, "fallback_used": 1, "validator_retries": 1}
    # exactly two translate calls: the initial pass + ONE targeted retry
    assert len(engine.translate_calls()) == 2
    assert "failed validation" in engine.translate_calls()[1][1]
    assert fallback.calls == [["Hello everyone"]]


def test_validator_failure_recovers_on_targeted_retry() -> None:
    def respond(system, user, json_mode):  # noqa: ANN001 - test double
        if "failed validation" in user:
            return json.dumps(
                {"translations": [{"index": 0, "text": "Xin chào các bạn"}]}
            )
        return json.dumps(
            {"translations": [{"index": 0, "text": "hello there friend"}]}
        )

    engine = StubChatOllama(respond)
    out = engine.translate(["Hello everyone"], "en", "vi")

    assert out == ["Xin chào các bạn"]
    assert engine.stats == {"llm_used": 1, "fallback_used": 0, "validator_retries": 1}
    assert len(engine.translate_calls()) == 2


def test_fallback_is_lazy_when_none_provided(monkeypatch) -> None:  # noqa: ANN001
    """fallback=None -> an Envit5Translator is constructed on first use only."""
    created: list[object] = []

    class FakeEnvit5(BaseTranslator):
        def __init__(self) -> None:
            created.append(self)

        def translate(self, texts, src_lang, tgt_lang):  # noqa: ANN001
            return ["Xin chào các bạn" for _ in texts]

    monkeypatch.setattr(
        "vidub.engines.mt.envit5_mt.Envit5Translator", FakeEnvit5
    )

    def respond(system, user, json_mode):  # noqa: ANN001 - test double
        return json.dumps(
            {"translations": [{"index": 0, "text": "hello there friend"}]}
        )

    engine = StubChatOllama(respond)
    assert created == []  # not constructed until the fallback is needed
    out = engine.translate(["Hello everyone"], "en", "vi")
    assert out == ["Xin chào các bạn"]
    assert len(created) == 1
    assert engine.stats["fallback_used"] == 1


# ------------------------------------------------------- reflect / improve
def test_reflect_escalates_only_major_issues() -> None:
    def respond(system, user, json_mode):  # noqa: ANN001 - test double
        if system == EN_VI_TRANSLATE_SYSTEM:
            return json.dumps(
                {
                    "translations": [
                        {"index": 0, "text": "Xin chào các bạn"},
                        {"index": 1, "text": "Thời tiết hôm nay đẹp"},
                    ]
                }
            )
        if system == EN_VI_REFLECT_SYSTEM:
            return json.dumps(
                {
                    "critiques": [
                        {"index": 0, "issues": ["meaning changed"], "severity": "major"},
                        {"index": 1, "issues": ["word choice"], "severity": "minor"},
                    ]
                }
            )
        if system == EN_VI_IMPROVE_SYSTEM:
            return json.dumps(
                {"translations": [{"index": 0, "text": "Chào mọi người nhé"}]}
            )
        return "{}"

    engine = StubChatOllama(respond)
    out = engine.translate_contextual(
        ["Hello everyone", "The weather is nice today"], "en", "vi"
    )

    assert out == ["Chào mọi người nhé", "Thời tiết hôm nay đẹp"]
    assert len(engine.reflect_calls()) == 1
    # the improve pass only lists the escalated (major) segment
    improve_user = engine.improve_calls()[0][1]
    assert "Segment 0:" in improve_user
    assert "Segment 1:" not in improve_user
    assert "ISSUES: meaning changed" in improve_user


def test_reflect_parse_failure_keeps_all_drafts() -> None:
    def respond(system, user, json_mode):  # noqa: ANN001 - test double
        if system == EN_VI_TRANSLATE_SYSTEM:
            return json.dumps(
                {"translations": [{"index": 0, "text": "Xin chào các bạn"}]}
            )
        return "not json"  # reflect parse failure

    engine = StubChatOllama(respond)
    out = engine.translate_contextual(["Hello everyone"], "en", "vi")

    assert out == ["Xin chào các bạn"]
    assert engine.improve_calls() == []


def test_contextual_runs_reflect_but_plain_translate_does_not() -> None:
    def respond(system, user, json_mode):  # noqa: ANN001 - test double
        if system == EN_VI_REFLECT_SYSTEM:
            return json.dumps({"critiques": []})
        return json.dumps(
            {"translations": [{"index": 0, "text": "Xin chào các bạn"}]}
        )

    contextual = StubChatOllama(respond)
    contextual.translate_contextual(["Hello everyone"], "en", "vi")
    assert len(contextual.reflect_calls()) == 1

    plain = StubChatOllama(respond)
    plain.translate(["Hello everyone"], "en", "vi")
    assert plain.reflect_calls() == []


# ------------------------------------------------------------------ batching
def test_batches_of_eight_make_two_calls() -> None:
    def respond(system, user, json_mode):  # noqa: ANN001 - test double
        count = user.count("TRANSLATE THIS")
        items = [
            {"index": i, "text": f"Xin chào {i}"}
            for i in range(9)
            if f"Segment {i}:" in user
        ]
        assert len(items) == count
        return json.dumps({"translations": items})

    engine = StubChatOllama(respond)
    out = engine.translate([f"Hello {i}" for i in range(9)], "en", "vi")

    assert len(out) == 9
    assert len(engine.translate_calls()) == 2  # 8 + 1


# ------------------------------------------------------- base default + build
def test_base_default_translate_contextual_delegates() -> None:
    class Recording(BaseTranslator):
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def translate(self, texts, src_lang, tgt_lang):  # noqa: ANN001
            self.calls.append(list(texts))
            return [f"x{text}" for text in texts]

        def translate_contextual(self, texts, src_lang, tgt_lang):  # noqa: ANN001
            return super().translate_contextual(texts, src_lang, tgt_lang)

    engine = Recording()
    assert engine.translate_contextual(["a", "b"], "en", "vi") == ["xa", "xb"]
    assert engine.calls == [["a", "b"]]


def test_base_translate_contextual_carries_budgets() -> None:
    class Recording(BaseTranslator):
        def __init__(self) -> None:
            self.seen: list = []

        def translate(self, texts, src_lang, tgt_lang, *, budgets=None):  # noqa: ANN001
            self.seen.append(budgets)
            return list(texts)

    engine = Recording()
    assert engine.translate_contextual(["a", "b"], "en", "vi", budgets=[5, 6]) == [
        "a",
        "b",
    ]
    assert engine.seen == [[5, 6]]


def test_base_budgets_length_mismatch_raises() -> None:
    class Recording(BaseTranslator):
        def translate(self, texts, src_lang, tgt_lang, *, budgets=None):  # noqa: ANN001
            return list(texts)

    engine = Recording()
    with pytest.raises(ValueError):
        engine.translate_contextual(["a", "b"], "en", "vi", budgets=[1])


# --------------------------------------------------------- budget prompts
def test_translate_user_without_budgets_unchanged() -> None:
    engine = StubChatOllama(lambda system, user, json_mode: "{}")

    user = engine._translate_user(["Hello everyone"], [0])

    assert user == (
        "Translate every segment marked TRANSLATE (1 of them). Return strict "
        "JSON with one entry per marked segment.\n\n"
        "Segment 0:\n  TRANSLATE THIS: Hello everyone"
    )


def test_translate_user_with_budgets_states_limits() -> None:
    engine = StubChatOllama(lambda system, user, json_mode: "{}")

    user = engine._translate_user(
        ["Hello everyone", "The weather is nice today"], [0, 1], [42, 77]
    )

    assert "Character limit per segment" in user
    assert "Segment 0: at most 42 characters" in user
    assert "Segment 1: at most 77 characters" in user
    # the limit is stated before the segments it applies to
    assert user.index("at most 42 characters") < user.index("TRANSLATE THIS")


def test_translate_with_budgets_threads_limit_and_keeps_system() -> None:
    def respond(system, user, json_mode):  # noqa: ANN001 - test double
        assert system == EN_VI_TRANSLATE_SYSTEM
        return json.dumps(
            {"translations": [{"index": 0, "text": "Xin chào các bạn"}]}
        )

    engine = StubChatOllama(respond)
    out = engine.translate(["Hello everyone"], "en", "vi", budgets=[35])

    assert out == ["Xin chào các bạn"]
    assert "Segment 0: at most 35 characters" in engine.translate_calls()[0][1]


def test_retry_keeps_the_budget_limit() -> None:
    bad = "hello there friend"  # fails the lang validator

    def respond(system, user, json_mode):  # noqa: ANN001 - test double
        return json.dumps({"translations": [{"index": 0, "text": bad}]})

    engine = StubChatOllama(respond, fallback=FakeFallback())
    engine.translate(["Hello everyone"], "en", "vi", budgets=[35])

    # the initial pass + the one targeted retry both carry the limit
    assert len(engine.translate_calls()) == 2
    for _system, user, _json_mode in engine.translate_calls():
        assert "Segment 0: at most 35 characters" in user


def test_budgets_length_mismatch_raises() -> None:
    engine = StubChatOllama(lambda system, user, json_mode: "{}")

    with pytest.raises(ValueError):
        engine.translate(["a", "b"], "en", "vi", budgets=[10])
    with pytest.raises(ValueError):
        engine.translate_contextual(["a"], "en", "vi", budgets=[1, 2])


def test_build_translator_maps_ollama_with_cfg() -> None:
    manifest = JobManifest(mt_engine="ollama")
    cfg = JobConfig(
        mt_engine="ollama",
        ollama_model="llama3:8b",
        ollama_host="http://example:1234",
    )

    engine = build_translator(manifest, cfg)

    assert isinstance(engine, OllamaTranslator)
    assert engine.model == "llama3:8b"
    assert engine.host == "http://example:1234"


def test_build_translator_envit5_default_unchanged() -> None:
    from vidub.engines.mt.envit5_mt import Envit5Translator

    assert isinstance(
        build_translator(JobManifest(mt_engine="envit5")), Envit5Translator
    )
    with pytest.raises(TranslationError):
        build_translator(JobManifest(mt_engine="definitely-not-an-engine"))


# ------------------------------------------------------------------ doctor
def test_doctor_ollama_line_reachable(monkeypatch, capsys) -> None:  # noqa: ANN001
    import vidub.cli as cli

    class FakeResponse:
        def __init__(self, payload) -> None:  # noqa: ANN001
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *exc):  # noqa: ANN002
            return False

    def fake_urlopen(url, timeout=None):  # noqa: ANN001
        if url.endswith("/api/version"):
            return FakeResponse({"version": "0.34.4"})
        return FakeResponse({"models": [{"name": "qwen2.5:7b"}]})

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)

    rc = cli.main(["doctor"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "ollama: 0.34.4 at http://127.0.0.1:11434 (model qwen2.5:7b: present)" in out


def test_doctor_ollama_line_unreachable(monkeypatch, capsys) -> None:  # noqa: ANN001
    import vidub.cli as cli

    def fake_urlopen(url, timeout=None):  # noqa: ANN001 - test double
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)

    rc = cli.main(["doctor"])
    out = capsys.readouterr().out

    assert rc == 0  # the doctor must never fail on a missing server
    assert "ollama: not reachable (start it with 'ollama serve')" in out
    assert "doctor complete" in out


# --------------------------------------------------- translate-stage integration
def _make_job(
    tmp_path: Path, segments: list[dict]
) -> tuple[Path, Path, JobConfig]:
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "hash1"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False), encoding="utf-8"
    )
    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="hash1",
        cache_dir=str(cache_dir),
        src_lang="en",
        tgt_lang="vi",
        paths={"segments": "segments.json"},
        stages={"transcribe": "done"},
    ).save(job_path)
    return job_path, artifacts_dir, JobConfig(cache_dir=str(cache_dir))


class ContextualStub(BaseTranslator):
    """Plain ``translate`` must never be used for the main batch."""

    def __init__(self) -> None:
        self.contextual_calls: list[list[str]] = []

    def translate(self, texts, src_lang, tgt_lang):  # noqa: ANN001
        raise AssertionError("the stage must use translate_contextual")

    def translate_contextual(self, texts, src_lang, tgt_lang):  # noqa: ANN001
        self.contextual_calls.append(list(texts))
        return ["Xin chào các bạn" for _ in texts]


def test_translate_stage_uses_contextual_output(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path,
        [{"start": 0.0, "end": 2.0, "text": "Hello world", "speaker": "spk0"}],
    )
    cfg.qc_translation = False  # isolate the engine call from the QC round-trip
    engine = ContextualStub()

    translate(str(job_path), cfg, translator=engine)

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0]["vi"] == "Xin chào các bạn"
    assert engine.contextual_calls == [["Hello world"]]


# --------------------------------------------------------------- live smoke
@pytest.mark.skipif(
    os.environ.get("OLLAMA_LIVE") != "1",
    reason="set OLLAMA_LIVE=1 to run against a live Ollama server",
)
def test_live_ollama_translation() -> None:
    engine = OllamaTranslator()
    out = engine.translate_contextual(
        ["The GPU needs more VRAM.", "There are 16 GB of VRAM."], "en", "vi"
    )

    assert len(out) == 2
    assert _lang_ok(out[0], "vi", "en")
    assert _numbers_ok("There are 16 GB of VRAM.", out[1])
    assert engine.stats["llm_used"] >= 1

"""Regression cases from the two user exports; no model or network required."""
from pathlib import Path

import pytest

from vidub.config import JobConfig
from vidub.pipeline import mt_signature
from vidub.utils.english_tokens import grammar_leaks
from vidub.utils.glossary import load_glossary
from vidub.utils.tqc import correct_rows, extract_entities, score_rows


@pytest.mark.parametrize("token", ["I've", "I’m", "I'll", "We're", "They'd", "Isn't", "Won't", "Let's", "Can't", "Don't"])
def test_grammar_is_not_an_entity(token: str) -> None:
    assert extract_entities(f"and {token} explain this today.") == []
    assert grammar_leaks(f"và {token} giải thích điều này.") == [token]


def test_acronym_possessive_and_apostrophized_names() -> None:
    assert extract_entities("we discuss AI's output with O'Reilly and McDonald's.") == [
        "AI", "O'Reilly", "McDonald's",
    ]
    assert grammar_leaks("AI's trả lời O'Reilly và McDonald's.") == ["AI's"]
    assert grammar_leaks("ví dụ là I'm here.", ["I'm here"]) == []
    assert grammar_leaks("I'm here và I'm there.", ["I'm here"]) == ["I'm"]
    assert extract_entities("we use GPUs and URLs with O'Reilly.") == ["GPU", "URL", "O'Reilly"]


def test_leak_is_flagged_and_full_translation_retry_can_resolve_it() -> None:
    rows = [{"en": "and I've explained it already.", "vi": "và I've giải thích điều này rồi."}]
    labels, _ = score_rows(rows, {})
    assert labels[0]["label"] == "unnatural"

    class Translator:
        def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
            assert texts == [rows[0]["en"]]
            return ["và tôi đã giải thích điều này rồi."]

    repaired, actions = correct_rows(rows, labels, {}, Translator())
    assert repaired[0]["vi"] == "và tôi đã giải thích điều này rồi."
    assert actions[0]["resolved"]


def test_entity_repair_rejects_new_grammar_leak() -> None:
    rows = [{"en": "we use Acme and I'll explain the tool today.",
             "vi": "chúng tôi dùng công cụ và tôi sẽ giải thích hôm nay."}]
    original = rows[0]["vi"]
    labels, _ = score_rows(rows, {})

    class Echo:
        def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
            return texts

    repaired, actions = correct_rows(rows, labels, {}, Echo())
    assert repaired[0]["vi"] == original
    assert actions[0]["action"] == "splice-entity-rejected"
    assert not actions[0]["resolved"]


def test_retry_rejects_new_completeness_risk() -> None:
    rows = [{"en": "and I've explained how this system works in detail today.",
             "vi": "và I've giải thích cách hệ thống này hoạt động chi tiết hôm nay."}]
    original = rows[0]["vi"]
    labels, _ = score_rows(rows, {})

    class Short:
        def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
            return ["tôi đã giải thích rồi."]

    repaired, actions = correct_rows(rows, labels, {}, Short())
    assert repaired[0]["vi"] == original
    assert actions[0]["action"] == "retranslate-rejected"


def test_language_and_explicit_glossary_exemptions() -> None:
    rows = [{"en": "and I'm here today.", "vi": "ví dụ I'm here hôm nay."}]
    assert score_rows(rows, {"I'm here": "I'm here"})[0][0]["label"] == "ok"
    assert score_rows(rows, {}, tgt_lang="en")[0][0]["label"] == "ok"
    rows[0]["passthrough"] = True
    assert score_rows(rows, {})[0][0]["label"] == "ok"


def test_glossary_content_changes_signature_at_same_path(tmp_path: Path) -> None:
    path = tmp_path / "terms.tsv"
    path.write_text("code\tmã nguồn\n", encoding="utf-8")
    cfg = JobConfig(glossary_path=str(path))
    first = mt_signature(cfg)
    assert first == mt_signature(cfg)
    path.write_text("code\tmã lập trình\n", encoding="utf-8")
    assert first != mt_signature(cfg)
    assert first != mt_signature(JobConfig())


def test_ai_glossary_parses_and_keeps_technical_sense() -> None:
    mapping = load_glossary(Path(__file__).resolve().parents[1] / "samples/glossary-ai-tutorial.tsv")
    assert mapping["transformer"] == "Transformer"
    assert mapping["code"] == "mã nguồn"

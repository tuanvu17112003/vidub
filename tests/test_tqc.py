"""Tests for the Level-2 translation QC battery (M4-T3, QUALITY-ROADMAP.md §3.3).

All fast/pure: stub translators, synthetic rows, no models/network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vidub.engines.mt.base import BaseTranslator
from vidub.utils.tqc import (
    ENTITY_STOPLIST,
    correct_rows,
    digits_of,
    extract_entities,
    extract_numbers,
    run_battery,
    score_rows,
    token_f1,
    vi_number_form,
)


# ------------------------------------------------------------------ extractors
def test_extract_numbers_separators_currency_percent() -> None:
    assert extract_numbers("It cost $30,000 in 1999, up 2.5% since.") == [
        "$30,000",
        "1999",
        "2.5%",
    ]


def test_digits_of_strips_non_digits() -> None:
    assert digits_of("30,000") == "30000"
    assert digits_of("$50%") == "50"
    assert digits_of("1999") == "1999"


def test_vi_number_form_swaps_separators() -> None:
    assert vi_number_form("30,000") == "30.000"
    assert vi_number_form("2.5") == "2,5"
    assert vi_number_form("1999") == "1999"
    assert vi_number_form("50%") == "50%"


def test_extract_entities_skips_initial_and_stoplist() -> None:
    text = "we use Vigib daily, and their AI Corp is here. The Acme team too!"
    entities = extract_entities(text)
    assert "Vigib" in entities
    assert "AI" in entities and "Corp" in entities and "Acme" in entities
    # "The" is a stoplist word; "Vigib" at the text start would not be an entity
    assert "The" not in entities
    assert extract_entities("Vigib will dub it") == []  # sentence-initial
    # length-1 tokens and stoplist words are excluded
    assert "I" not in entities and "And" not in entities
    assert "the" in ENTITY_STOPLIST


def test_token_f1_bounds() -> None:
    assert token_f1("hello world", "hello world") == 1.0
    assert token_f1("", "") == 1.0
    assert token_f1("hello", "world") == 0.0
    assert 0.0 < token_f1("hello there world", "hello world") < 1.0


# ------------------------------------------------------------------ scoring
def _row(en: str, vi: str, **extra) -> dict:
    row = {"start": 0.0, "end": 2.0, "speaker": "spk0", "en": en, "vi": vi,
           "truncated": False}
    row.update(extra)
    return row


def test_score_ok_row() -> None:
    rows = [_row("Hello world", "Xin chào thế giới")]
    results, summary = score_rows(rows, {})
    assert results[0]["label"] == "ok"
    assert summary["counts"] == {"ok": 1}


def test_passthrough_row_is_skipped_and_recorded_ok() -> None:
    rows = [_row("Bạn có thể", "Bạn có thể", passthrough=True)]
    results, _summary = score_rows(rows, {})
    assert results[0]["label"] == "ok"
    assert results[0]["signals"]["passthrough"] == ["passthrough"]


def test_number_found_despite_separator_difference() -> None:
    # source "30,000" vs vi "30.000" -> the digits match, no error
    results, _ = score_rows([_row("It costs 30,000 dollars", "Giá là 30.000 đô la")], {})
    assert results[0]["label"] == "ok"


def test_number_error_when_digits_missing() -> None:
    results, _ = score_rows([_row("It costs 30,000 dollars", "Giá là đô la")], {})
    assert results[0]["label"] == "number-error"
    assert "30,000" in results[0]["signals"]["number-error"][0]


def test_entity_error_for_dropped_latin_entity() -> None:
    results, _ = score_rows([_row("we use Vigib daily", "chúng tôi dùng hằng ngày")], {})
    assert results[0]["label"] == "entity-error"
    assert "Vigib" in results[0]["signals"]["entity-error"][0]


def test_entity_mapped_by_glossary_is_not_a_mismatch() -> None:
    # source "Vigib" is glossed to "ViDub": the target differs, so no error
    results, _ = score_rows(
        [_row("we use Vigib daily", "chúng tôi dùng ViDub")], {"Vigib": "ViDub"}
    )
    assert "entity-error" not in results[0]["signals"]


def test_terminology_error_when_target_absent() -> None:
    rows = [_row("and their AI assistant.", "và cỗ máy của họ.")]
    results, _ = score_rows(rows, {"AI assistant": "trợ lý AI"})
    assert results[0]["label"] == "terminology-error"
    assert "trợ lý AI" in results[0]["signals"]["terminology-error"][0]


def test_terminology_error_does_not_double_as_entity_error() -> None:
    # "AI" is a sentence-initial fragment of the glossed term, not an entity.
    rows = [_row("and their AI assistant.", "và cỗ máy của họ.")]
    results, _ = score_rows(rows, {"AI assistant": "trợ lý AI"})
    assert "entity-error" not in results[0]["signals"]


def test_terminology_ok_when_target_present() -> None:
    rows = [_row("the AI assistant is here", "trợ lý AI ở đây.")]
    results, _ = score_rows(rows, {"AI assistant": "trợ lý AI"})
    assert results[0]["label"] != "terminology-error"


def test_omission_for_empty_translation() -> None:
    results, _ = score_rows([_row("Hello world today", "")], {})
    assert results[0]["label"] == "omission"
    assert "empty" in results[0]["signals"]["omission"][0]


def test_omission_for_short_ratio() -> None:
    long_source = "one two three four five six seven eight nine ten"
    results, _ = score_rows([_row(long_source, "ngắn")], {})
    assert results[0]["label"] == "omission"
    ratio = 1 / 10  # < LENGTH_RATIO_MIN
    assert f"length_ratio={ratio:.2f}" in results[0]["signals"]["omission"][0]


def test_addition_for_overlong_untruncated_row() -> None:
    results, _ = score_rows([_row("hello", "một hai ba bốn năm sáu bảy tám")], {})
    assert results[0]["label"] == "addition"


def test_addition_not_flagged_when_row_is_truncated() -> None:
    results, _ = score_rows(
        [_row("hello", "một hai ba bốn năm sáu bảy tám", truncated=True)], {}
    )
    assert "addition" not in results[0]["signals"]


def test_unnatural_from_low_confidence_flag() -> None:
    results, _ = score_rows([_row("Hello world", "hello there", low_confidence=True)], {})
    assert results[0]["label"] == "unnatural"


def test_precedence_number_over_entity() -> None:
    # this row drops both a number and an entity -> number-error wins
    results, _ = score_rows(
        [_row("Vigib costs 30,000 dollars", "Nó đắt")], {}
    )
    assert results[0]["label"] == "number-error"


def test_consistency_divergence_flags_the_later_row() -> None:
    phrase = "the weather is nice"
    rows = [
        _row(f"{phrase} today", "thời tiết hôm nay đẹp"),
        _row(f"{phrase} tomorrow", "hoàn toàn khác biệt"),
    ]
    results, _ = score_rows(rows, {})
    assert "inconsistent" in results[1]["signals"]
    assert "inconsistent" not in results[0]["signals"]


def test_consistency_agreement_is_not_flagged() -> None:
    phrase = "the weather is nice"
    rows = [
        _row(f"{phrase} today", "thời tiết hôm nay đẹp"),
        _row(f"{phrase} tomorrow", "thời tiết ngày mai đẹp"),
    ]
    results, _ = score_rows(rows, {})
    assert all("inconsistent" not in result["signals"] for result in results)


class _BackTranslator(BaseTranslator):
    """Returns a fixed back-translation for every input."""

    def __init__(self, value: str) -> None:
        self.value = value

    def translate(self, texts, src_lang, tgt_lang):
        return [self.value for _ in texts]


def test_roundtrip_flag_with_stub_engine() -> None:
    rows = [_row("Hello world", "Xin chào thế giới")]
    results, summary = score_rows(rows, {}, engine=_BackTranslator("totally different"))
    assert summary["roundtrip"] == "ok"
    assert results[0]["label"] == "mistranslation-risk"


def test_roundtrip_skipped_for_non_en_source() -> None:
    rows = [_row("Bonjour monde", "Xin chào")]
    results, summary = score_rows(rows, {}, engine=_BackTranslator("x"), src_lang="fr")
    assert summary["roundtrip"] == "skipped (engine covers en<->vi only)"
    assert "mistranslation-risk" not in results[0]["signals"]


def test_roundtrip_disabled_without_engine() -> None:
    _results, summary = score_rows([_row("Hi", "Xin chào")], {})
    assert summary["roundtrip"] == "disabled"


# ---------------------------------------------------------------- corrections
class _EchoTranslator(BaseTranslator):
    """Returns the input unchanged (used as the splice's translate_fn)."""

    def translate(self, texts, src_lang, tgt_lang):
        return list(texts)


def test_entity_error_splice_resolves_dropped_entity(tmp_path: Path) -> None:
    rows = [_row("we use Vigib daily", "chúng tôi dùng hằng ngày")]
    results, _ = score_rows(rows, {})
    assert results[0]["label"] == "entity-error"
    assert "Vigib" in results[0]["signals"]["entity-error"][0]

    # An echo engine forces the latin entity in via the existing splice helper.
    rows, corrections = correct_rows(rows, results, {}, _EchoTranslator(), "en", "vi")
    assert corrections[0]["action"] == "splice-entity"
    assert "Vigib" in rows[0]["vi"]
    assert corrections[0]["resolved"] is True


def test_number_splice_fires_on_digit_free_vi() -> None:
    rows = [_row("It costs 30,000 dollars", "Giá là đô la")]
    results, _ = score_rows(rows, {})
    rows, corrections = correct_rows(rows, results, {}, _EchoTranslator(), "en", "vi")
    assert corrections[0]["action"] == "splice-number"
    assert "30.000" in rows[0]["vi"]


def test_number_splice_skipped_when_vi_has_digits() -> None:
    rows = [_row("It costs 30,000 dollars", "Giá là 5 đô la")]
    results, _ = score_rows(rows, {})
    rows, corrections = correct_rows(rows, results, {}, _EchoTranslator(), "en", "vi")
    assert corrections[0]["action"] == "review-number"
    assert corrections[0]["resolved"] is False


def test_terminology_error_path_enforces_glossary() -> None:
    glossary = {"AI assistant": "trợ lý AI"}
    rows = [_row("and their AI assistant.", "và cỗ máy của họ.")]
    results, _ = score_rows(rows, glossary)
    rows, corrections = correct_rows(
        rows, results, glossary, _EchoTranslator(), "en", "vi"
    )
    assert corrections[0]["action"] == "enforce-glossary"
    assert "trợ lý AI" in rows[0]["vi"]


def test_correction_is_at_most_one_retry_per_row() -> None:
    class _Counting(BaseTranslator):
        def __init__(self) -> None:
            self.calls = 0

        def translate(self, texts, src_lang, tgt_lang):
            self.calls += 1
            return ["hoàn toàn khác biệt" for _ in texts]

    rows = [_row("Hello world", "Xin chào thế giới")]
    results, _ = score_rows(rows, {}, engine=_BackTranslator("totally different"))
    engine = _Counting()
    _rows, corrections = correct_rows(rows, results, {}, engine, "en", "vi")

    assert engine.calls <= 1  # one retry max
    assert corrections[0]["retried"] is True


def test_passthrough_rows_are_never_modified() -> None:
    rows = [_row("Bạn có thể", "Bạn có thể", passthrough=True)]
    before = str(rows[0]["vi"])
    results, _ = score_rows(rows, {}, engine=_BackTranslator("x"))
    rows, corrections = correct_rows(rows, results, {}, _EchoTranslator(), "en", "vi")
    assert rows[0]["vi"] == before
    assert corrections == []


# -------------------------------------------------------------------- verdicts
def test_verdict_fail_for_digit_free_number_error() -> None:
    # vi has NO digits at all -> the number was wholly dropped -> content-critical
    rows = [_row("It costs 30,000 dollars", "Giá là đô la")]
    _rows, artifact = run_battery(rows, {}, None, "en", "vi")
    assert artifact["verdict"] == "fail"


def test_verdict_review_for_digit_present_number_error() -> None:
    # vi keeps a digit but the source number is missing -> surfaced, not fatal
    rows = [_row("It costs 30,000 dollars", "Giá là 5 đô la")]
    _rows, artifact = run_battery(rows, {}, None, "en", "vi")
    assert artifact["counts"].get("number-error") == 1
    assert artifact["verdict"] == "review"


def test_verdict_review_for_real_segment_14_shape() -> None:
    # The production seg-14 shape: vi has digits, one fragment number dropped.
    rows = [_row("It costs 30,000 in 1999", "Giá là 30.000 đô la")]
    _rows, artifact = run_battery(rows, {}, None, "en", "vi")
    assert artifact["counts"].get("number-error") == 1
    assert artifact["verdict"] == "review"


def test_verdict_review_for_unresolved_terminology_error() -> None:
    glossary = {"AI assistant": "trợ lý AI"}
    rows = [_row("and their AI assistant.", "và cỗ máy của họ.")]
    _rows, artifact = run_battery(rows, glossary, None, "en", "vi")
    assert artifact["counts"].get("terminology-error") == 1
    assert artifact["verdict"] == "review"


def test_verdict_review_for_unresolved_entity_error() -> None:
    rows = [_row("we use Vigib daily", "chúng tôi dùng hằng ngày")]
    _rows, artifact = run_battery(rows, {}, None, "en", "vi")
    assert artifact["counts"].get("entity-error") == 1
    assert artifact["verdict"] == "review"


def test_verdict_review_for_only_mistranslation_risk() -> None:
    rows = [_row("Hello world", "Xin chào thế giới")]
    _rows, artifact = run_battery(rows, {}, _BackTranslator("totally different"), "en", "vi")
    assert artifact["verdict"] == "review"
    assert artifact["counts"] == {"mistranslation-risk": 1}


def test_verdict_pass_for_clean_rows() -> None:
    rows = [_row("Hello world", "Xin chào thế giới")]
    _rows, artifact = run_battery(rows, {}, None, "en", "vi")
    assert artifact["verdict"] == "pass"
    assert artifact["counts"] == {"ok": 1}


def test_verdict_fail_for_unresolved_omission() -> None:
    rows = [_row("Hello world today", "")]
    _rows, artifact = run_battery(rows, {}, None, "en", "vi")
    assert artifact["verdict"] == "fail"


def test_run_battery_artifact_shape() -> None:
    rows = [_row("Vigib will dub it", "Nó sẽ lồng tiếng")]
    _rows, artifact = run_battery(rows, {"Vigib": "ViDub"}, _EchoTranslator(), "en", "vi")
    assert set(artifact) == {"verdict", "counts", "roundtrip", "segments"}
    entry = artifact["segments"][0]
    assert {"index", "label", "signals", "retried", "resolved", "action"} <= set(entry)


def test_run_battery_writes_review_packet(tmp_path: Path) -> None:
    rows = [_row("It costs 30,000 dollars", "Giá là 5 đô la")]  # digit-present: no fix
    _rows, artifact = run_battery(rows, {}, None, "en", "vi", out_dir=tmp_path)
    assert artifact["verdict"] == "review"  # M4-T9-C1: not content-critical
    packet = tmp_path / "review" / "translation" / "seg-0.txt"
    assert packet.exists()
    content = packet.read_text(encoding="utf-8")
    assert "label: number-error" in content


def test_run_battery_no_packet_for_clean_rows(tmp_path: Path) -> None:
    rows = [_row("Hello world", "Xin chào thế giới")]
    _rows, artifact = run_battery(rows, {}, None, "en", "vi", out_dir=tmp_path)
    assert artifact["verdict"] == "pass"
    assert not (tmp_path / "review" / "translation" / "seg-0.txt").exists()


# ------------------------------------------------------- M5-T2 refiner hook
def test_run_battery_without_refiner_has_no_refine_key() -> None:
    rows = [_row("Hello world", "Xin chào thế giới")]
    _rows, artifact = run_battery(rows, {}, None, "en", "vi")
    assert "refine" not in artifact
    assert set(artifact) == {"verdict", "counts", "roundtrip", "segments"}


def test_run_battery_refiner_improves_row_and_records_summary(tmp_path: Path) -> None:
    rows = [_row("we use Vigib daily", "chúng tôi dùng hằng ngày")]  # entity-error

    def refiner(rows, entries):  # noqa: ANN001 - test double
        rows[0]["vi"] = "chúng tôi dùng Vigib hằng ngày"
        entries[0]["label"] = "ok"
        entries[0]["signals"] = {}
        return rows, entries, {"attempted": 1, "improved": 1, "fusion_used": False}

    _rows, artifact = run_battery(
        rows, {}, None, "en", "vi", refiner=refiner, out_dir=tmp_path
    )

    assert artifact["refine"] == {"attempted": 1, "improved": 1, "fusion_used": False}
    # counts/verdict/packets all see the refined label
    assert artifact["counts"] == {"ok": 1}
    assert artifact["verdict"] == "pass"
    assert not (tmp_path / "review" / "translation" / "seg-0.txt").exists()


def test_run_battery_refiner_exception_propagates() -> None:
    rows = [_row("we use Vigib daily", "chúng tôi dùng hằng ngày")]

    class _Boom(RuntimeError):
        pass

    def refiner(rows, entries):  # noqa: ANN001 - test double
        raise _Boom("refine exploded")

    # Contract: run_battery does NOT catch refiner errors (the stage guards).
    with pytest.raises(_Boom):
        run_battery(rows, {}, None, "en", "vi", refiner=refiner)

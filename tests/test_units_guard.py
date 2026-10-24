"""Tests for the SA-1-T2 unit completeness guard (``vidub.utils.units_guard``).

Pure ``guard_signals`` / ``guard_rank`` cases (no engine) plus the orchestration
outcomes (``resampled`` / ``fallback`` / ``flagged`` / ``decontext``) against
stub engines. The Q-1-T1 section adds the neighbour-contamination signals,
calibrated on the User's real hunyuan job. No network, no model.
"""

from __future__ import annotations

from vidub.engines.mt.base import BaseTranslator
from vidub.utils.units_guard import (
    ADDITION_SENTENCE_COVERAGE,
    CONTAMINATION_OVERLAP,
    MIN_WORDS_FOR_SENTENCE_LOSS,
    RESAMPLE_K,
    UNIT_COVERAGE_MAX,
    UNIT_COVERAGE_MIN,
    contamination_signals,
    count_sentences,
    guard_rank,
    guard_signals,
    guard_signals_with_context,
    guard_units,
)

# 40 source words in 3 sentences (the User's calibration shape).
SOURCE_40_3 = (
    "a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14. "
    "b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14. "
    "c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12."
)
# 29 target words in 1 sentence -> coverage 29/40 == 0.725, sentences 3 -> 1.
VI_29_1 = " ".join(f"t{i}" for i in range(28)) + " cuối."
VI_40_3 = (
    " ".join(f"x{i}" for i in range(14))
    + ". "
    + " ".join(f"y{i}" for i in range(13))
    + ". "
    + " ".join(f"z{i}" for i in range(13))
    + "."
)

SOURCE = "The weather is sunny today and tomorrow morning"  # 8 words, 0 terminals
SHORT = "ngắn"  # 1 word
COMPLETE = "một hai ba bốn năm sáu bảy tám"  # 8 words


# ------------------------------------------------------------------ pure logic
def test_count_sentences_counts_terminal_characters() -> None:
    assert count_sentences("") == 0
    assert count_sentences("   ") == 0
    assert count_sentences("One. Two! Three?") == 3
    assert count_sentences("No terminal here") == 0
    assert count_sentences("Ellipsis…") == 1


def test_guard_signals_empty_translation_is_omission_risk() -> None:
    signals = guard_signals("Some source text here", "")
    assert signals["words_out"] == 0
    assert signals["coverage"] == 0.0
    assert signals["flags"] == ["omission-risk"]


def test_guard_signals_user_calibration_half_dropped() -> None:
    # exactly the evidence from the task: 29/40 coverage, 3 source sentences -> 1
    signals = guard_signals(SOURCE_40_3, VI_29_1)
    assert signals["words_in"] == 40
    assert signals["words_out"] == 29
    assert signals["coverage"] == 29 / 40 == 0.725
    assert signals["coverage"] < UNIT_COVERAGE_MIN
    assert signals["sentences_in"] == 3
    assert signals["sentences_out"] == 1
    assert signals["flags"] == ["omission-risk", "sentence-loss"]


def test_guard_signals_faithful_translation_is_clean() -> None:
    signals = guard_signals(SOURCE_40_3, VI_40_3)
    assert signals["coverage"] == 1.0
    assert signals["sentences_in"] == signals["sentences_out"] == 3
    assert signals["flags"] == []


def test_guard_signals_verbose_addition() -> None:
    signals = guard_signals("hello world", " ".join(f"t{i}" for i in range(20)) + ".")
    assert signals["coverage"] > UNIT_COVERAGE_MAX
    assert signals["flags"] == ["addition-risk"]


def test_guard_signals_single_sentence_unit_has_no_sentence_loss() -> None:
    clean = guard_signals("This is one single sentence.", "Đây là một câu đơn.")
    assert clean["flags"] == []
    short = guard_signals("This is one single sentence.", "Đây là.")
    # a single-sentence source can never trip the sentence-loss signal
    assert short["flags"] == ["omission-risk"]
    assert short["sentences_in"] == 1
    assert MIN_WORDS_FOR_SENTENCE_LOSS == 8


def test_guard_rank_orders_by_flags_then_coverage() -> None:
    clean = {"coverage": 1.0, "flags": []}
    near = {"coverage": 1.05, "flags": []}
    far = {"coverage": 2.0, "flags": []}
    one = {"coverage": 1.0, "flags": ["addition-risk"]}
    two = {"coverage": 1.0, "flags": ["omission-risk", "sentence-loss"]}

    assert guard_rank(clean) < guard_rank(one) < guard_rank(two)
    assert guard_rank(near) < guard_rank(far)


# --------------------------------------------------------------- orchestration
class _StubEngine(BaseTranslator):
    """Fixed translation; optional ``sample_candidates`` output."""

    def __init__(self, translation: str, samples: list[str] | None = None) -> None:
        self.translation = translation
        self.samples = samples
        self.calls: list[list[str]] = []

    def translate(self, texts, src_lang, tgt_lang):
        self.calls.append(list(texts))
        return [self.translation for _ in texts]

    def sample_candidates(self, text, src_lang, tgt_lang, k):
        return list(self.samples or [])[:k]


class _RaisingSampler(_StubEngine):
    """A sampling engine that raises: the candidate must only be dropped."""

    def sample_candidates(self, text, src_lang, tgt_lang, k):
        raise RuntimeError("boom")


def test_guard_units_resampled_outcome() -> None:
    engine = _StubEngine(SHORT, [COMPLETE])
    chosen, records = guard_units([SOURCE], [SHORT], engine, "en", "vi")

    assert chosen == [COMPLETE]
    record = records[0]
    assert record["action"] == "resampled"
    assert record["candidates"] == 2
    assert record["flags"] == ["omission-risk"]
    assert record["indices"] == [0]
    assert set(record) == {
        "indices", "words_in", "words_out", "coverage",
        "sentences_in", "sentences_out", "flags", "action", "candidates",
    }
    assert RESAMPLE_K == 3


def test_guard_units_fallback_outcome() -> None:
    engine = _StubEngine(SHORT, [])
    fallback = _StubEngine(COMPLETE)
    chosen, records = guard_units(
        [SOURCE], [SHORT], engine, "en", "vi", fallback_engine=fallback
    )

    assert chosen == [COMPLETE]
    assert records[0]["action"] == "fallback"
    assert records[0]["candidates"] == 2
    assert fallback.calls == [[SOURCE]]


def test_guard_units_flagged_outcome_when_nothing_fixes_it() -> None:
    engine = _StubEngine(SHORT, [])
    chosen, records = guard_units([SOURCE], [SHORT], engine, "en", "vi")

    assert chosen == [SHORT]
    assert records[0]["action"] == "flagged"
    assert records[0]["candidates"] == 1
    assert records[0]["flags"] == ["omission-risk"]


def test_guard_units_never_raises_on_a_broken_engine() -> None:
    # sampling raises -> only that candidate is dropped, the guard still returns
    engine = _RaisingSampler(SHORT, [COMPLETE])
    chosen, records = guard_units([SOURCE], [SHORT], engine, "en", "vi")
    assert chosen == [SHORT]
    assert records[0]["action"] == "flagged"

    # a raising fallback engine is dropped the same way
    chosen, records = guard_units(
        [SOURCE],
        [SHORT],
        _StubEngine(SHORT, []),
        "en",
        "vi",
        fallback_engine=_RaisingTranslateEngine(),
    )
    assert chosen == [SHORT]
    assert records[0]["action"] == "flagged"


def test_guard_units_clean_unit_is_untouched() -> None:
    engine = _StubEngine(COMPLETE)
    chosen, records = guard_units([SOURCE], [COMPLETE], engine, "en", "vi")
    assert chosen == [COMPLETE]
    assert records[0]["action"] == "none"
    assert records[0]["candidates"] == 1
    assert records[0]["flags"] == []


class _RaisingTranslateEngine(BaseTranslator):
    """A fallback engine whose ``translate`` raises."""

    def translate(self, texts, src_lang, tgt_lang):
        raise RuntimeError("fallback boom")


# ------------------------------------------------------- Q-1-T1 contamination
# The User's REAL strings from the contaminated hunyuan job
# ``326ea9c45ee94afbfe3186be9b798ee8a05fbf63``.  Unit 24's vi swallowed unit
# 23's whole vi and ~70% of unit 25's; unit 25 in turn contains unit 24's.
U22_VI = "Bạn có một chiếc card đồ họa dung lượng 8 gigabyte phải không?"
U23_VI = "Thật không may cho bạn đấy."
U24_SOURCE = (
    "Sure, you could do some fancy offloading where you constantly swap "
    "weights in and out of memory."
)
U24_VI = (
    "Thật không may cho bạn đấy. Nhưng việc đó cũng giống như cố gắng nấu bữa "
    "tối bằng cách chạy đi mua từng nguyên liệu riêng lẻ vậy."
)
U25_SOURCE = (
    "But that's like trying to cook dinner by running to the grocery store "
    "for each ingredient."
)
U25_VI = (
    "Nhưng đó giống như cố gắng nấu bữa tối bằng cách liên tục chạy đến cửa "
    "hàng tạp hóa để mua từng nguyên liệu riêng lẻ vậy."
)


def test_contamination_constants() -> None:
    assert CONTAMINATION_OVERLAP == 0.5
    assert ADDITION_SENTENCE_COVERAGE == 1.4


def test_contamination_signals_user_unit_24_context_bleed() -> None:
    """Unit 24 contains 100% of unit 23 and >=50% of unit 25: a context bleed."""
    signals = contamination_signals(U24_VI, U23_VI, U25_VI)
    assert signals["overlap_prev"] == 1.0
    assert signals["overlap_next"] >= CONTAMINATION_OVERLAP
    assert "context-bleed" in signals["flags"]


def test_contamination_signals_user_unit_25_context_bleed() -> None:
    signals = contamination_signals(U25_VI, U24_VI, "")
    assert signals["overlap_prev"] >= CONTAMINATION_OVERLAP
    assert "context-bleed" in signals["flags"]


def test_contamination_signals_clean_pair_stays_silent() -> None:
    # units 22/23 are neighbours in the same job and share almost no content
    assert contamination_signals(U23_VI, U22_VI, "")["flags"] == []
    assert contamination_signals(U22_VI, "", U23_VI)["flags"] == []
    # an unrelated neighbour never flags either
    assert (
        contamination_signals(
            U24_VI, "An unrelated sentence about the weather today.", ""
        )["flags"]
        == []
    )


def test_contamination_signals_sentence_addition() -> None:
    source = "alpha beta. gamma delta."  # 2 sentences
    vi = (
        "một hai ba bốn năm sáu bảy tám chín mười. "
        "mười một mười hai. mười ba mười bốn."
    )  # 3 sentences, coverage >> 1.4
    signals = contamination_signals(vi, "", "", source=source)
    assert signals["flags"] == ["sentence-addition"]
    # without the source the sentence counts are unknown: no addition flag
    assert contamination_signals(vi, "", "")["flags"] == []


def test_contamination_signals_verbatim_repeat_is_flagged_on_purpose() -> None:
    """A genuine repeated slogan trips ``context-bleed`` too; this is acceptable.

    Containment cannot tell a swallowed neighbour from a phrase the speaker
    legitimately repeats in two consecutive units.  The code flags it; then
    ``guard_units`` asks for the direct candidate and, because the repeat is
    genuine the direct text repeats as well, the original wins the tie and the
    unit is recorded ``flagged``.  The text is never degraded — only the label is
    conservative, and the repair costs one harmless direct engine call.  That is
    the accepted trade: never miss a real bleed.
    """
    slogan = "Chúng tôi luôn đổi mới vì khách hàng."
    signals = contamination_signals(slogan, slogan, slogan)
    assert signals["overlap_prev"] == 1.0
    assert signals["overlap_next"] == 1.0
    assert "context-bleed" in signals["flags"]


def test_guard_signals_with_context_merges_the_contamination_flags() -> None:
    signals = guard_signals_with_context(U24_SOURCE, U24_VI, U23_VI, U25_VI)
    assert {"context-bleed", "sentence-addition"} <= set(signals["flags"])
    assert signals["overlap_prev"] == 1.0
    # the plain guard never sees the neighbours: source-completeness only
    assert guard_signals(U24_SOURCE, U24_VI)["flags"] == []


def test_guard_rank_prefers_a_bleed_over_an_omission() -> None:
    clean = {"coverage": 1.0, "flags": []}
    bleed = {"coverage": 1.0, "flags": ["context-bleed"]}
    omit = {"coverage": 1.0, "flags": ["omission-risk"]}
    assert guard_rank(clean) < guard_rank(bleed) < guard_rank(omit)


# Synthetic strings keep the orchestration deterministic and independent of the
# real job's exact token counts.
_BLEED_PREV = "Chúng ta hãy bắt đầu với phần đầu tiên của bài học."
_BLEED_NEXT = "Sau đó chúng ta sẽ xem xét ví dụ tiếp theo."
_BLEED_SOURCE = "And here is the content of this unit."
_BLEED_CLEAN = "Và đây là nội dung của đơn vị này."
_BLEED_CONTAM = f"{_BLEED_PREV} Và đây là nội dung của đơn vị này {_BLEED_NEXT}"


class _BleedStub(BaseTranslator):
    """``translate`` supplies the direct (decontext) candidate; samples optional."""

    def __init__(self, direct: str, samples: list[str] | None = None) -> None:
        self.direct = direct
        self.samples = samples or []
        self.calls: list[list[str]] = []

    def translate(self, texts, src_lang, tgt_lang):
        self.calls.append(list(texts))
        return [self.direct for _ in texts]

    def sample_candidates(self, text, src_lang, tgt_lang, k):
        return list(self.samples)[:k]


def test_guard_units_decontext_repairs_a_context_bleed() -> None:
    engine = _BleedStub(_BLEED_CLEAN)
    chosen, records = guard_units(
        [_BLEED_SOURCE],
        [_BLEED_CONTAM],
        engine,
        "en",
        "vi",
        neighbours=[(_BLEED_PREV, _BLEED_NEXT)],
    )

    assert chosen == [_BLEED_CLEAN]
    record = records[0]
    assert record["action"] == "decontext"
    assert record["candidates"] == 2
    assert "context-bleed" in record["flags"]
    assert record["overlap_prev"] == 1.0
    assert record["overlap_next"] == 1.0
    assert record["overlap_prev_after"] < CONTAMINATION_OVERLAP
    assert record["overlap_next_after"] < CONTAMINATION_OVERLAP
    assert engine.calls == [[_BLEED_SOURCE]]  # exactly the direct call


def test_guard_units_flags_a_bleed_the_direct_template_cannot_fix() -> None:
    # the "direct" output still reproduces the previous unit: nothing fixed it
    engine = _BleedStub(f"{_BLEED_PREV} và đây là nội dung của đơn vị này.")
    chosen, records = guard_units(
        [_BLEED_SOURCE],
        [_BLEED_CONTAM],
        engine,
        "en",
        "vi",
        neighbours=[(_BLEED_PREV, _BLEED_NEXT)],
    )

    assert records[0]["action"] == "flagged"
    after = contamination_signals(
        chosen[0], _BLEED_PREV, _BLEED_NEXT, source=_BLEED_SOURCE
    )
    assert "context-bleed" in after["flags"]
    assert after["overlap_prev"] >= CONTAMINATION_OVERLAP


def test_guard_units_keeps_the_original_when_the_direct_call_is_identical() -> None:
    engine = _BleedStub(_BLEED_CONTAM)  # byte-identical to the original
    chosen, records = guard_units(
        [_BLEED_SOURCE],
        [_BLEED_CONTAM],
        engine,
        "en",
        "vi",
        neighbours=[(_BLEED_PREV, _BLEED_NEXT)],
    )
    assert chosen == [_BLEED_CONTAM]
    assert records[0]["candidates"] == 1
    assert records[0]["action"] == "flagged"


def test_guard_units_clean_run_makes_no_extra_engine_calls() -> None:
    engine = _BleedStub("bất kỳ")
    chosen, records = guard_units(
        [_BLEED_SOURCE],
        [_BLEED_CLEAN],
        engine,
        "en",
        "vi",
        neighbours=[(_BLEED_PREV, _BLEED_NEXT)],
    )

    assert chosen == [_BLEED_CLEAN]
    assert records[0]["action"] == "none"
    assert records[0]["flags"] == []
    assert engine.calls == []  # no direct/resample/fallback call at all


# ------------------------------------------------- ISO-1-T3-C1 min_coverage
def test_guard_signals_min_coverage_is_additive() -> None:
    # 29/40 == 0.725: flagged at the default 0.75 floor, clean at the lowered 0.65
    default = guard_signals(SOURCE_40_3, VI_29_1)
    assert "omission-risk" in default["flags"]
    lowered = guard_signals(SOURCE_40_3, VI_29_1, min_coverage=0.65)
    assert "omission-risk" not in lowered["flags"]
    # sentence-loss is independent of the coverage floor and still present here
    assert "sentence-loss" in lowered["flags"]
    # the default is unchanged (0.75): the same call without the argument matches
    assert guard_signals(SOURCE_40_3, VI_29_1) == default


def test_guard_signals_with_context_forwards_min_coverage() -> None:
    wrapped_default = guard_signals_with_context(SOURCE_40_3, VI_29_1)
    assert "omission-risk" in wrapped_default["flags"]
    wrapped_lowered = guard_signals_with_context(
        SOURCE_40_3, VI_29_1, min_coverage=0.65
    )
    assert "omission-risk" not in wrapped_lowered["flags"]

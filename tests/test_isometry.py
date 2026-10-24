"""Tests for the pure isometric length policy (``vidub.utils.isometry``, ISO-1-T1).

Everything here is pure policy: no engine, no network, no model. The acceptance
cases use the *real* calibration strings from ``tests/test_units_guard.py``
(source 40 words / 3 sentences) so the omission-floor rejection is exercised with
numbers the rest of the suite already trusts. A dedicated section asserts the
module aliases the shared ``fit`` / ``units_guard`` constants instead of
re-declaring them.
"""

from __future__ import annotations

from vidub.utils import fit, units_guard
from vidub.utils.isometry import (
    BUDGET_HEADROOM,
    COVERAGE_FLOOR,
    MAX_BUDGET_CHARS,
    MIN_BUDGET_CHARS,
    NATURAL_CHARS_PER_SECOND,
    SHORT_COVERAGE_MIN,
    accept_shrunk,
    budget_report,
    char_count,
    coverage_floor,
    over_budget,
    projected_stretch,
    select_over_budget,
    unit_budget,
)


# ------------------------------------------------------------------ fixtures
def _tokens(prefix: str, count: int) -> str:
    """``count`` distinct alphabetic tokens ``prefix + letter`` (no digits)."""
    return " ".join(f"{prefix}{chr(ord('a') + i)}" for i in range(count))


# 40 source words in 3 sentences, digit-free so ``_validate`` isolates the guard.
SOURCE_40_3 = (
    _tokens("s", 14) + ". " + _tokens("t", 14) + ". " + _tokens("u", 12) + "."
)
# 29 Vietnamese words in 1 sentence -> coverage 29/40 == 0.725, sentences 3 -> 1.
VI_29_1 = " ".join(["từ"] * 28) + " cuối."


# ------------------------------------------------------------- constant reuse
def test_constants_alias_fit_and_guard() -> None:
    # Same objects (and values): the calibration is imported, never re-declared.
    assert NATURAL_CHARS_PER_SECOND == fit.CONCISE_CHARS_PER_SECOND
    assert NATURAL_CHARS_PER_SECOND is fit.CONCISE_CHARS_PER_SECOND
    assert MIN_BUDGET_CHARS == fit.MIN_CONCISE_CHARS
    assert MIN_BUDGET_CHARS is fit.MIN_CONCISE_CHARS
    assert COVERAGE_FLOOR == units_guard.UNIT_COVERAGE_MIN
    assert COVERAGE_FLOOR is units_guard.UNIT_COVERAGE_MIN
    assert BUDGET_HEADROOM == 0.9


# ---------------------------------------------------------------- unit_budget
def test_unit_budget_formula() -> None:
    # window * cps * headroom == 2.0 * 20.0 * 0.9 == 36
    assert unit_budget(0.0, 2.0) == 36
    assert unit_budget(1.0, 3.0) == 36


def test_unit_budget_tiny_window_hits_floor() -> None:
    # 0.1 * 20 * 0.9 == 1.8 -> 2, floored to MIN_BUDGET_CHARS
    assert unit_budget(0.0, 0.1) == MIN_BUDGET_CHARS
    # a zero/negative-length window still gets the floor, never a sub-word budget
    assert unit_budget(5.0, 5.0) == MIN_BUDGET_CHARS
    assert unit_budget(5.0, 4.0) == MIN_BUDGET_CHARS


def test_unit_budget_huge_window_hits_cap() -> None:
    assert unit_budget(0.0, 10_000.0) == MAX_BUDGET_CHARS
    assert MAX_BUDGET_CHARS > MIN_BUDGET_CHARS  # documented ceiling is real


def test_unit_budget_headroom_applied() -> None:
    assert unit_budget(0.0, 2.0, headroom=1.0) == 40
    assert unit_budget(0.0, 2.0, headroom=0.5) == 20
    assert unit_budget(0.0, 2.0, headroom=BUDGET_HEADROOM) == 36


def test_unit_budget_following_silence_share() -> None:
    # window 2 s, gap to next start 4 s -> silence 2 s
    assert unit_budget(0.0, 2.0, 4.0) == 36  # default borrows none
    assert unit_budget(0.0, 2.0, 4.0, include_following_silence=1.0) == 72
    assert unit_budget(0.0, 2.0, 4.0, include_following_silence=0.5) == 54
    # the share is clamped into [0, 1]
    assert unit_budget(0.0, 2.0, 4.0, include_following_silence=9.0) == 72
    assert unit_budget(0.0, 2.0, 4.0, include_following_silence=-1.0) == 36
    # no next unit -> no silence to borrow; an overlapping next start -> none
    assert unit_budget(0.0, 2.0, None, include_following_silence=1.0) == 36
    assert unit_budget(0.0, 2.0, 1.0, include_following_silence=1.0) == 36


def test_unit_budget_custom_pace() -> None:
    assert unit_budget(0.0, 2.0, chars_per_second=10.0, headroom=1.0) == 20


# ------------------------------------------------------------- char_count etc
def test_char_count_counts_non_whitespace() -> None:
    assert char_count("a b\tc\nd") == 4
    assert char_count("") == 0
    assert char_count("   \n\t") == 0
    # diacritics are characters too
    assert char_count("từ") == 2


def test_over_budget_boundary_is_inclusive() -> None:
    assert over_budget("abc", 3) is False  # exactly at budget is not over
    assert over_budget("abcd", 3) is True
    assert over_budget("", 0) is False
    assert over_budget("a", 0) is True  # whitespace-free length 1 > 0


def test_projected_stretch_monotonic_and_clamped() -> None:
    assert projected_stretch("", 10) == 1.0
    assert projected_stretch("x" * 5, 10) == 1.0
    assert projected_stretch("x" * 10, 10) == 1.0
    assert projected_stretch("x" * 15, 10) == 1.5
    # non-decreasing in text length
    assert (
        projected_stretch("x" * 80, 100)
        <= projected_stretch("x" * 110, 100)
        <= projected_stretch("x" * 150, 100)
    )
    # a non-positive budget has no meaningful ratio
    assert projected_stretch("", 0) == 1.0
    assert projected_stretch("x", 0) == float("inf")


# ------------------------------------------------------------- budget_report
def test_budget_report_shape_and_flags() -> None:
    report = budget_report(SOURCE_40_3, VI_29_1, 0.0, 2.0)
    assert set(report) == {
        "budget_chars",
        "chars",
        "over_budget",
        "projected_stretch",
        "coverage",
        "flags",
    }
    assert report["budget_chars"] == 36
    assert report["chars"] == char_count(VI_29_1)
    assert report["coverage"] == 29 / 40
    assert "omission-risk" in report["flags"]
    assert "sentence-loss" in report["flags"]


def test_budget_report_forwards_unit_budget_kwargs() -> None:
    assert budget_report("s", "vi", 0.0, 2.0, headroom=1.0)["budget_chars"] == 40


# ---------------------------------------------------------- select_over_budget
def test_select_over_budget_preserves_order() -> None:
    sources = ["s1 word here", "another source line", "third source line"]
    translations = [
        "một bản dịch siêu dài không thể nào vừa nổi cửa sổ này đâu",
        "ok",
        "thêm một bản dịch dài dòng khác nữa cho chắc chắn",
    ]
    starts = [0.0, 10.0, 20.0]
    ends = [1.0, 11.0, 21.0]
    next_starts = [10.0, 20.0, None]
    records = select_over_budget(sources, translations, starts, ends, next_starts)
    assert [record["index"] for record in records] == [0, 2]
    assert all(record["over_budget"] for record in records)


def test_select_over_budget_empty_next_starts() -> None:
    records = select_over_budget(
        ["a b c"],
        ["một bản dịch dài dòng hơn hẳn ngân sách nhỏ này"],
        [0.0],
        [0.5],
    )
    assert [record["index"] for record in records] == [0]


# ------------------------------------------------------------- accept_shrunk
def test_accept_shrunk_accepts_shorter_fit() -> None:
    source = "The quick brown fox jumps over the lazy dog"
    previous_vi = "Đây là một bản dịch tiếng Việt khá là dài dòng và lan man cho đơn vị này"
    candidate = "Con cáo nâu nhanh nhẹn nhảy qua chú chó lười"
    budget = 30
    assert char_count(candidate) <= char_count(previous_vi)
    assert projected_stretch(candidate, budget) < projected_stretch(previous_vi, budget)
    assert accept_shrunk(source, candidate, budget, previous_vi=previous_vi) is True


def test_accept_shrunk_rejects_below_omission_floor() -> None:
    # (a) 29/40 words -> omission-risk + sentence-loss from the real guard.
    assert accept_shrunk(SOURCE_40_3, VI_29_1, 30) is False
    flags = units_guard.guard_signals(SOURCE_40_3, VI_29_1)["flags"]
    assert "omission-risk" in flags and "sentence-loss" in flags


def test_accept_shrunk_rejects_sentence_loss_above_omission_floor() -> None:
    # 31/40 words -> coverage 0.775 (above the 0.75 floor) but the 3 source
    # sentences collapse to 1 -> ``sentence-loss`` is the only floor that fires.
    candidate = " ".join(["từ"] * 30) + " cuối."
    signals = units_guard.guard_signals(SOURCE_40_3, candidate)
    assert signals["coverage"] >= units_guard.UNIT_COVERAGE_MIN
    assert "omission-risk" not in signals["flags"]
    assert "sentence-loss" in signals["flags"]
    assert accept_shrunk(SOURCE_40_3, candidate, 30) is False


def test_accept_shrunk_rejects_non_vietnamese_candidate() -> None:
    # The engine language validator rejects an English-looking candidate.
    assert accept_shrunk("hello world", "this is an english sentence", 30) is False


def test_accept_shrunk_rejects_context_bleed() -> None:
    # (b) the candidate reproduces the previous unit's wording.
    source = "the weather is nice today"
    previous_vi = "hôm nay thời tiết rất đẹp và mát mẻ nữa"
    candidate = "hôm nay thời tiết rất đẹp"
    budget = char_count(candidate)  # otherwise the candidate is a genuine shrink
    assert projected_stretch(candidate, budget) < projected_stretch(previous_vi, budget)
    assert accept_shrunk(source, candidate, budget, previous_vi=previous_vi) is False


def test_accept_shrunk_rejects_empty() -> None:
    # (c) empty and whitespace-only candidates are never accepted.
    assert accept_shrunk("hello world", "", 20) is False
    assert accept_shrunk("hello world", "   ", 20) is False


def test_accept_shrunk_rejects_dropped_number() -> None:
    # (d) the source number "16" is gone, so the validator rejects the candidate.
    source = "Model 3 has 16 GB of RAM"
    candidate = "Mô hình ba có mười sáu GB bộ nhớ"
    assert accept_shrunk(source, candidate, 30) is False


def test_accept_shrunk_rejects_longer_candidate() -> None:
    # (e1) a longer candidate is never a shrink.
    source = "hello world"
    previous_vi = "xin chào"
    candidate = "bây giờ chúng ta bắt đầu nhé"
    assert char_count(candidate) > char_count(previous_vi)
    assert accept_shrunk(
        source, candidate, 10, previous_vi=previous_vi
    ) is False


def test_accept_shrunk_rejects_same_length_candidate() -> None:
    # (e2) same length -> same projected stretch -> no gain -> rejected.
    source = "hello world"
    previous_vi = "bây giờ"
    candidate = "mau lên"
    budget = 10
    assert char_count(candidate) == char_count(previous_vi)
    assert projected_stretch(candidate, budget) == projected_stretch(previous_vi, budget)
    assert accept_shrunk(
        source, candidate, budget, previous_vi=previous_vi
    ) is False


def test_accept_shrunk_skips_relative_checks_without_previous() -> None:
    # No baseline: a valid candidate is accepted without the shrink comparison.
    source = "hello world"
    candidate = "xin chào thế giới"
    assert accept_shrunk(source, candidate, 30) is True


# ---------------------------------------------------------------- determinism
def test_determinism_same_input_same_output() -> None:
    first = budget_report(SOURCE_40_3, VI_29_1, 0.0, 2.0)
    second = budget_report(SOURCE_40_3, VI_29_1, 0.0, 2.0)
    assert first == second

    source = "The quick brown fox jumps over the lazy dog"
    previous_vi = "Đây là một bản dịch tiếng Việt khá là dài dòng và lan man cho đơn vị này"
    candidate = "Con cáo nâu nhanh nhẹn nhảy qua chú chó lười"
    assert accept_shrunk(source, candidate, 30, previous_vi=previous_vi) is True
    assert accept_shrunk(source, candidate, 30, previous_vi=previous_vi) is True


# ------------------------------------------------- ISO-1-T3-C1 adaptive floor
def test_coverage_floor_tiers_on_the_window_threshold() -> None:
    # a normal window keeps the shared 0.75 floor
    assert coverage_floor(2.2) == units_guard.UNIT_COVERAGE_MIN == COVERAGE_FLOOR
    # a window strictly below SHORT_WINDOW_MAX (1.80) drops to 0.65
    assert coverage_floor(1.6) == SHORT_COVERAGE_MIN == 0.65
    assert coverage_floor(fit.SHORT_WINDOW_MAX) == units_guard.UNIT_COVERAGE_MIN
    assert coverage_floor(fit.SHORT_WINDOW_MAX - 1e-9) == SHORT_COVERAGE_MIN
    # the threshold/values are parameters (the config knobs), so they compose
    assert coverage_floor(1.6, short_window=1.0) == units_guard.UNIT_COVERAGE_MIN
    assert coverage_floor(0.9, short_window=1.0, normal=0.7, short=0.5) == 0.5


# 20 source words, no terminal (so `sentence-loss` can never fire); the candidate
# keeps 14 words -> coverage 0.70, between the 0.65 and 0.75 floors.
FLOOR_SOURCE = " ".join(f"s{chr(ord('a') + index)}" for index in range(20))
FLOOR_CANDIDATE = " ".join(["từ"] * 14)
FLOOR_PREVIOUS = " ".join(["từ"] * 25)


def test_accept_shrunk_uses_the_lowered_floor_only_below_the_threshold() -> None:
    # a 1.6 s window: floor 0.65, so the 0.70-coverage candidate passes ...
    assert (
        accept_shrunk(
            FLOOR_SOURCE,
            FLOOR_CANDIDATE,
            30,
            previous_vi=FLOOR_PREVIOUS,
            window=1.6,
        )
        is True
    )
    # ... and the exact same candidate at a 2.2 s window (floor 0.75) is rejected
    assert (
        accept_shrunk(
            FLOOR_SOURCE,
            FLOOR_CANDIDATE,
            30,
            previous_vi=FLOOR_PREVIOUS,
            window=2.2,
        )
        is False
    )
    # without a window the default floor (0.75) applies
    assert (
        accept_shrunk(
            FLOOR_SOURCE, FLOOR_CANDIDATE, 30, previous_vi=FLOOR_PREVIOUS
        )
        is False
    )


# ------------------------------------------- ISO-1-C1-FIX configured floor
# 25 source words, no terminal punctuation (so ``sentence-loss`` can never fire);
# the candidate keeps 18 -> coverage 0.720, strictly between the configured 0.70
# short floor and the normal 0.75 floor. Distinct digit-bearing tokens keep the
# containment guard empty (they share no content tokens with the previous text).
CFG_SOURCE = " ".join(f"s{chr(ord('a') + index)}" for index in range(25))
CFG_CANDIDATE = " ".join(f"bò{index}" for index in range(18))
CFG_PREVIOUS = " ".join(f"mã{index}" for index in range(30))


def test_accept_shrunk_uses_the_configured_short_floor_knobs() -> None:
    """The configured threshold/coverage must change what is accepted.

    Fails with the pre-fix ``accept_shrunk`` (which only took ``window`` and
    re-derived the floor from the module defaults, so ``short_window`` /
    ``short_coverage`` were unknown and the knobs had no effect on acceptance).
    """
    # 2.2 s < configured 2.5 s -> the configured 0.70 floor applies -> accepted.
    assert (
        accept_shrunk(
            CFG_SOURCE,
            CFG_CANDIDATE,
            30,
            previous_vi=CFG_PREVIOUS,
            window=2.2,
            short_window=2.5,
            short_coverage=0.70,
        )
        is True
    )
    # The exact same 2.2 s unit under the module defaults (threshold 1.80 s) is a
    # normal window -> floor 0.75 -> the 0.72-coverage candidate is rejected.
    assert (
        accept_shrunk(
            CFG_SOURCE, CFG_CANDIDATE, 30, previous_vi=CFG_PREVIOUS, window=2.2
        )
        is False
    )
    # A pre-resolved floor (what the stage passes) is authoritative in both
    # directions, regardless of the window.
    assert (
        accept_shrunk(
            CFG_SOURCE,
            CFG_CANDIDATE,
            30,
            previous_vi=CFG_PREVIOUS,
            window=2.2,
            floor=0.70,
        )
        is True
    )
    assert (
        accept_shrunk(
            CFG_SOURCE,
            CFG_CANDIDATE,
            30,
            previous_vi=CFG_PREVIOUS,
            window=2.2,
            floor=0.75,
        )
        is False
    )
    # The configured threshold/floor are exactly what ``coverage_floor`` yields.
    assert coverage_floor(2.2, short_window=2.5, short=0.70) == 0.70
    assert coverage_floor(2.2) == units_guard.UNIT_COVERAGE_MIN == 0.75

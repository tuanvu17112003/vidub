"""Tests for Vietnamese subtitle standards QC (M4-T2, QUALITY-ROADMAP.md G3).

All pure/fast: no ffmpeg, no models. Covers the Netflix-derived profile
measurements, line wrapping, cue re-splitting (span contiguity/bounds), and the
row-level QC transform used by the export stage.
"""

from __future__ import annotations

import pytest

from vidub.utils import subqc
from vidub.utils.subqc import (
    CPS_MAX,
    DUR_MAX,
    DUR_MIN,
    LINE_MAX,
    LINES_MAX,
    measure,
    qc_rows,
    split_cue,
    wrap_text,
)


# ------------------------------------------------------------------- constants
def test_constants_match_the_netflix_vi_profile() -> None:
    assert CPS_MAX == 17.0
    assert DUR_MIN == pytest.approx(5 / 6)
    assert DUR_MAX == 7.0
    assert LINE_MAX == 42
    assert LINES_MAX == 2
    assert subqc.CPS_MAX == 17.0


# --------------------------------------------------------------------- measure
def test_measure_clean_cue_has_no_violations() -> None:
    result = measure("Xin chào", 0.0, 1.0)
    assert result["cps"] == pytest.approx(8.0)
    assert result["duration"] == pytest.approx(1.0)
    assert result["lines"] == ["Xin chào"]
    assert result["max_line_len"] == 8
    assert result["violations"] == []


def test_measure_cps_violation() -> None:
    result = measure("a" * 30, 0.0, 1.0)
    assert "cps=30.0>17.0" in result["violations"]


def test_measure_duration_min_violation() -> None:
    result = measure("hi", 0.0, 0.4)
    assert "duration=0.4<0.833" in result["violations"]


def test_measure_duration_max_violation() -> None:
    result = measure("hello world", 0.0, 8.0)
    assert "duration=8.0>7.0" in result["violations"]


def test_measure_line_length_violation() -> None:
    result = measure("a" * 50, 0.0, 10.0)
    assert "line_len=50>42" in result["violations"]


def test_measure_lines_violation() -> None:
    result = measure("a\nb\nc", 0.0, 10.0)
    assert "lines=3>2" in result["violations"]


def test_measure_zero_duration_does_not_crash() -> None:
    result = measure("ab", 1.0, 1.0)
    assert result["duration"] == 0.0
    assert any(v.startswith("cps=") for v in result["violations"])


def test_measure_normalizes_crlf_lines() -> None:
    result = measure("a\r\nb", 0.0, 1.0)
    assert result["lines"] == ["a", "b"]


# ------------------------------------------------------------------ wrap_text
def test_wrap_text_short_line_is_unchanged() -> None:
    assert wrap_text("Xin chào") == "Xin chào"


def test_wrap_text_already_within_limits_is_unchanged() -> None:
    text = "a" * 40 + "\n" + "b" * 40
    assert wrap_text(text) == text


def test_wrap_text_long_line_becomes_two_fitting_lines() -> None:
    text = (
        "Đây là một câu rất dài cần phải được tự động ngắt "
        "thành hai dòng cho đúng chuẩn."
    )
    wrapped = wrap_text(text)
    lines = wrapped.split("\n")
    assert len(lines) == 2
    assert all(len(line) <= LINE_MAX for line in lines)
    assert " ".join(lines) == text  # no characters lost


def test_wrap_text_unwrappable_is_returned_unchanged() -> None:
    text = "x" * 50  # a single word longer than a line: cannot wrap
    assert wrap_text(text) == text


# ------------------------------------------------------------------- split_cue
def test_split_cue_at_sentence_boundary() -> None:
    text = "Câu một đây nhé. Câu hai đây nhé."
    parts = split_cue(text, 0.0, 0.6)

    assert len(parts) >= 2
    assert parts[0][0] == 0.0
    assert parts[-1][1] == pytest.approx(0.6)
    # contiguous spans, all inside the parent, durations sum to the parent
    for start, end, part_text in parts:
        assert 0.0 <= start <= end <= 0.6 + 1e-9
        assert part_text
    for first, second in zip(parts, parts[1:]):
        assert first[1] == pytest.approx(second[0])
    assert sum(end - start for start, end, _ in parts) == pytest.approx(0.6)
    # the sentence ender is honoured (some part ends with the period)
    assert any(part[2].endswith(".") for part in parts)


def test_split_cue_stops_on_a_single_clause() -> None:
    # No punctuation: exactly one space split, then stop (no word explosion).
    parts = split_cue("một hai ba bốn năm sáu", 0.0, 0.1)
    assert len(parts) == 2
    assert all(0.0 <= start <= end <= 0.1 + 1e-9 for start, end, _ in parts)
    assert all(part[2] for part in parts)


def test_split_cue_under_cps_returns_single_unchanged() -> None:
    parts = split_cue("Xin chào", 0.0, 5.0)
    assert parts == [(0.0, 5.0, "Xin chào")]


# --------------------------------------------------------- word-edge split (T4)
# 10 words, one uneven pause before word 5: the proportional split would cut at
# 2.5 s, the ASR word edge is at 3.0 s.
_WORD_TEXT = " ".join(["aaaaaaaa"] * 10)
_WORD_TIMES = [
    [0.0, 0.4, "w0"],
    [0.4, 0.8, "w1"],
    [0.8, 1.2, "w2"],
    [1.2, 1.6, "w3"],
    [1.6, 2.0, "w4"],
    [3.0, 3.4, "w5"],
    [3.4, 3.8, "w6"],
    [3.8, 4.2, "w7"],
    [4.2, 4.6, "w8"],
    [4.6, 5.0, "w9"],
]


def test_split_cue_uses_word_edges_and_windows_sum() -> None:
    parts = split_cue(_WORD_TEXT, 0.0, 5.0, _WORD_TIMES)

    assert len(parts) == 2
    # the boundary lands on the real ASR word edge (word 5 starts at 3.0 s)
    assert parts[0][0] == pytest.approx(0.0)
    assert parts[0][1] == pytest.approx(3.0)
    assert parts[1][0] == pytest.approx(3.0)
    assert parts[1][1] == pytest.approx(5.0)
    # windows are contiguous and sum to the original span
    assert sum(end - start for start, end, _ in parts) == pytest.approx(5.0)
    # no text is lost
    assert " ".join(part[2] for part in parts) == _WORD_TEXT


def test_split_cue_without_words_uses_proportional_allocation() -> None:
    parts = split_cue(_WORD_TEXT, 0.0, 5.0)

    assert len(parts) == 2
    # no word timeline -> the historical proportional cut (2.5 s here)
    assert parts[0][1] == pytest.approx(2.5)
    assert sum(end - start for start, end, _ in parts) == pytest.approx(5.0)


def test_qc_rows_splits_on_word_edges_when_the_row_carries_them() -> None:
    row = {"start": 0.0, "end": 5.0, "vi": _WORD_TEXT, "align_words": _WORD_TIMES}
    out, stats = qc_rows([row], True)

    assert stats["split"] == 1
    assert len(out) == 2
    assert out[0]["start"] == pytest.approx(0.0)
    assert 2.75 <= out[0]["end"] <= 4.5  # bounded around the 3.0 s word edge
    assert out[1]["start"] == out[0]["end"]
    assert 5.0 <= out[-1]["end"] <= 6.5
    assert stats["violations"]["cps"] == 0


def test_qc_rows_without_word_timings_is_unchanged() -> None:
    row = {"start": 0.0, "end": 5.0, "vi": _WORD_TEXT}
    out, stats = qc_rows([row], True)

    assert stats["split"] == 1
    # Proportional reference, bounded scheduling, no invented word timeline.
    assert 2.25 <= out[0]["end"] <= 4.0
    assert stats["violations"]["cps"] == 0
    assert all("align_words" not in result for result in out)


def test_qc_rows_ignores_malformed_word_timings() -> None:
    row = {"start": 0.0, "end": 5.0, "vi": _WORD_TEXT, "align_words": [[0.0, "x"]]}
    out, stats = qc_rows([row], True)

    # A malformed timeline degrades to the same proportional reference.
    assert stats["split"] == 1
    expected, _ = qc_rows([{**row, "align_words": None}], True)
    assert [(r["start"], r["end"], r["vi"]) for r in out] == [(r["start"], r["end"], r["vi"]) for r in expected]


def test_split_cue_proportional_time_preserves_cps() -> None:
    # Splitting keeps the whole text and the whole span, so proportional time
    # allocation keeps each part's CPS near the parent's: an over-CPS cue stays
    # over CPS (the number of over-CPS cues grows, the speed does not drop).
    text = "alpha beta gamma delta epsilon zeta eta theta"
    parent = measure(text, 0.0, 1.0)["cps"]
    parts = split_cue(text, 0.0, 1.0)
    assert len(parts) >= 2
    for start, end, part_text in parts:
        assert measure(part_text, start, end)["cps"] >= parent * 0.8


# -------------------------------------------------------------------- qc_rows
def test_qc_rows_disabled_is_passthrough() -> None:
    rows = [{"start": 0.0, "end": 1.0, "vi": "Xin chào", "speaker": "spk0"}]
    out, stats = qc_rows(rows, False)
    assert out is rows
    assert stats["enabled"] is False
    assert stats["cues_in"] == 0 and stats["cues_out"] == 0
    assert stats["split"] == 0


def test_qc_rows_clean_row_is_unchanged() -> None:
    rows = [
        {"start": 0.0, "end": 2.0, "speaker": "spk0", "vi": "Xin chào các bạn",
         "truncated": False}
    ]
    out, stats = qc_rows(rows, True)
    assert out == rows
    assert stats["enabled"] is True
    assert stats["cues_out"] == 1
    assert stats["split"] == 0
    assert stats["violations"] == {"cps": 0, "duration": 0, "line_length": 0, "lines": 0}


def test_qc_rows_borrows_silence_for_high_cps_and_preserves_other_keys() -> None:
    row = {
        "start": 0.0,
        "end": 0.6,
        "speaker": "spk0",
        "en": "source",
        "vi": "Câu một đây nhé. Câu hai đây nhé.",
        "truncated": False,
        "custom": 7,
    }
    out, stats = qc_rows([row], True)

    assert stats["cues_in"] == 1
    assert stats["cues_out"] == len(out) == 1
    assert stats["split"] == 0
    assert stats["violations"]["cps"] == 0
    assert stats["cps_max_observed"] <= CPS_MAX
    for result in out:
        assert result["speaker"] == "spk0"
        assert result["en"] == "source"
        assert result["truncated"] is False
        assert result["custom"] == 7
        assert 0.0 <= result["start"] <= result["end"] <= 2.1


def test_qc_rows_wraps_long_line_without_splitting() -> None:
    text = (
        "Đây là một câu rất dài cần phải được tự động ngắt "
        "thành hai dòng cho đúng chuẩn."
    )
    row = {"start": 0.0, "end": 6.0, "speaker": "spk0", "vi": text,
           "truncated": False}
    out, stats = qc_rows([row], True)

    assert len(out) == 1
    assert "\n" in out[0]["vi"]
    assert max(len(line) for line in out[0]["vi"].split("\n")) <= LINE_MAX
    assert stats["split"] == 0
    assert stats["violations"]["line_length"] == 0


def test_qc_rows_empty_rows_are_zeroed() -> None:
    out, stats = qc_rows([], True)
    assert out == []
    assert stats["enabled"] is True
    assert stats["cues_in"] == 0 and stats["cues_out"] == 0
    assert stats["cps_max_observed"] == 0.0

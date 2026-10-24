"""Tests for the pure alignment allocators (vidub/utils/align.py).

All pure/fast: no I/O, no engine.  Covers the weight -> fraction arithmetic, the
word snapping, the span mapping (monotone, complete coverage, never inverted),
the exact phrase splitting and the ``allocate_spans`` degenerate cases plus
determinism.
"""

from __future__ import annotations

import pytest

from vidub.utils.align import (
    allocate_spans,
    phrase_pieces,
    share_fractions,
    snap_word_cuts,
    time_ranges,
)


# --------------------------------------------------------- share_fractions
def test_share_fractions_empty() -> None:
    assert share_fractions([]) == []


def test_share_fractions_equal_weights_are_cumulative() -> None:
    assert share_fractions([1.0, 1.0, 1.0]) == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert share_fractions([1.0, 1.0]) == pytest.approx([0.5, 1.0])


def test_share_fractions_all_non_positive_is_equal() -> None:
    assert share_fractions([0.0, 0.0]) == pytest.approx([0.5, 1.0])
    assert share_fractions([-1.0, 0.0, -2.0]) == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_share_fractions_are_proportional_and_end_at_one() -> None:
    fractions = share_fractions([1.0, 3.0])
    assert fractions == pytest.approx([0.25, 1.0])
    assert fractions[-1] == 1.0


# ---------------------------------------------------------- snap_word_cuts
def test_snap_word_cuts_is_monotone_and_clamped() -> None:
    cuts = snap_word_cuts([0.0, 0.34, 0.5, 1.0], 10)
    assert cuts == [0, 3, 5, 10]
    assert cuts == sorted(cuts)


def test_snap_word_cuts_more_buckets_than_words_stays_monotone() -> None:
    assert snap_word_cuts([0.2, 0.4, 0.6, 0.8, 1.0], 2) == [0, 1, 1, 2, 2]


def test_snap_word_cuts_empty_and_zero_words() -> None:
    assert snap_word_cuts([], 5) == []
    assert snap_word_cuts([0.5, 1.0], 0) == [0, 0]


# ------------------------------------------------------------ time_ranges
def test_time_ranges_monotone_and_complete() -> None:
    times = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    ranges = time_ranges([0.5, 1.0], times)

    assert ranges[0][0] == times[0][0]
    assert ranges[-1][1] == times[-1][1]
    assert all(start <= end for start, end in ranges)
    for (start, end), (next_start, _next_end) in zip(ranges, ranges[1:]):
        assert end <= next_start


def test_time_ranges_never_inverted() -> None:
    times = [(0.0, 0.4), (0.4, 0.9), (0.9, 1.0)]
    ranges = time_ranges([1 / 3, 2 / 3, 1.0], times)

    assert all(start <= end for start, end in ranges)
    assert ranges[0][0] == 0.0
    assert ranges[-1][1] == 1.0
    assert ranges == sorted(ranges)


def test_time_ranges_pins_edges_even_on_a_degenerate_last_fraction() -> None:
    times = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    ranges = time_ranges([0.4, 0.8], times)

    assert ranges[0][0] == 0.0
    assert ranges[-1][1] == 3.0
    assert all(start <= end for start, end in ranges)


def test_time_ranges_empty() -> None:
    assert time_ranges([1.0], []) == []
    assert time_ranges([], [(0.0, 1.0)]) == []


# ----------------------------------------------------------- phrase_pieces
def test_phrase_pieces_reassembles_exactly() -> None:
    text = "  Hello   there, my friend.  "
    pieces = phrase_pieces(text, [0.5, 1.0])

    assert len(pieces) == 2
    assert "".join(pieces) == text


def test_phrase_pieces_never_splits_a_token() -> None:
    text = "one two three four five"
    pieces = phrase_pieces(text, [0.2, 0.4, 0.6, 0.8, 1.0])

    for token in text.split():
        assert sum(piece.count(token) for piece in pieces) == 1
    assert "".join(pieces) == text


def test_phrase_pieces_empty_single_and_whitespace() -> None:
    assert phrase_pieces("abc", []) == []
    assert phrase_pieces("abc def", [1.0]) == ["abc def"]
    whitespace = phrase_pieces("   ", [0.5, 0.5])
    assert whitespace == ["   ", ""]
    assert "".join(whitespace) == "   "


# ---------------------------------------------------------- allocate_spans
def test_allocate_spans_happy_path() -> None:
    times = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    spans = allocate_spans([1.0, 1.0], times)

    assert spans is not None
    assert len(spans) == 2
    assert spans[0][0] == 0.0
    assert spans[-1][1] == 3.0
    assert all(start <= end for start, end in spans)


def test_allocate_spans_none_when_fewer_words_than_buckets() -> None:
    times = [(0.0, 1.0), (1.0, 2.0)]
    assert allocate_spans([1.0, 1.0, 1.0], times) is None


def test_allocate_spans_none_on_empty_inputs() -> None:
    assert allocate_spans([], [(0.0, 1.0)]) is None
    assert allocate_spans([1.0], []) is None


def test_allocate_spans_single_bucket() -> None:
    times = [(0.0, 1.0), (1.0, 2.5)]
    assert allocate_spans([1.0], times) == [(0.0, 2.5)]


def test_allocate_spans_is_deterministic() -> None:
    times = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    first = allocate_spans([1.0, 2.0, 1.0], times)
    second = allocate_spans([1.0, 2.0, 1.0], times)

    assert first == second
    assert first is not None

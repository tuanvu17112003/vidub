"""Tests for sentence-level translation units (vidub/utils/sentences.py).

All pure/fast: no ffmpeg, no models, no I/O beyond reading the committed
screenshot cache.  Covers sentence splitting, sentence-scoped grouping (a unit
ends only at a completed sentence or a speaker change; the hard ceiling and its
largest-gap pause cut are the sole size guard), the share arithmetic, and the
``split_translation`` invariants (count, non-emptiness, digit safety and the
reassembly property).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidub.utils.sentences import (
    CLAUSE_CHARS,
    HARD_UNIT_SECONDS,
    HARD_UNIT_WORDS,
    PAUSE_CUT_GAP_S,
    SENTENCE_GAP_S,
    TERMINAL_CHARS,
    ends_sentence,
    group_units,
    split_into_sentences,
    split_translation,
    unit_cut_stats,
    unit_share_summary,
)

SCREENSHOT = (
    Path(__file__).resolve().parents[1]
    / ".vidub-cache"
    / "53a7f0d67f5dc6d3853bb388f1bfca112252d512"
    / "segments.json"
)


# ------------------------------------------------------------------- helpers
def _row(text: str, start: float, end: float, speaker: str = "spk0") -> dict:
    """One ASR-row-shaped dict (the translate stage's segment shape)."""
    return {"text": text, "start": start, "end": end, "speaker": speaker}


def _frag(words: int) -> str:
    """A punctuation-free fragment of ``words`` words (never ends a sentence)."""
    return " ".join(f"w{i}" for i in range(words))


def _nonspace(text: str) -> str:
    return "".join(text.split())


# ------------------------------------------------------------------ constants
def test_constants_match_the_documented_profile() -> None:
    assert SENTENCE_GAP_S == 0.45
    assert HARD_UNIT_WORDS == 150
    assert HARD_UNIT_SECONDS == 45.0
    assert PAUSE_CUT_GAP_S == 0.6
    assert TERMINAL_CHARS == ".!?…"
    assert CLAUSE_CHARS == ",;:—"


# ------------------------------------------------------------- ends_sentence
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello.", True),
        ("Hello!", True),
        ("Really?", True),
        ("Wait…", True),
        ('He said "stop."', True),
        ("Done.”", True),
        ("Done.’", True),
        ("List (1).", True),
        ("Hello,", False),
        ("Model 3", False),
        ("", False),
        ("   ", False),
        ("no terminal here", False),
        ("word )", False),
    ],
)
def test_ends_sentence_table(text: str, expected: bool) -> None:
    assert ends_sentence(text) is expected


# ---------------------------------------------------- split_into_sentences
def _word(word: str, start: float, end: float) -> dict:
    return {"start": start, "end": end, "word": word}


def test_split_into_sentences_locks_word_edges() -> None:
    row = {
        "text": "Hello there. How are you?",
        "start": 0.0,
        "end": 2.5,
        "speaker": "spk0",
        "words": [
            _word("Hello", 0.0, 0.5),
            _word("there.", 0.5, 1.0),
            _word("How", 1.0, 1.5),
            _word("are", 1.5, 2.0),
            _word("you?", 2.0, 2.5),
        ],
    }
    pieces = split_into_sentences([row])

    assert [piece["text"] for piece in pieces] == ["Hello there.", "How are you?"]
    assert (pieces[0]["start"], pieces[0]["end"]) == (0.0, 1.0)
    assert (pieces[1]["start"], pieces[1]["end"]) == (1.0, 2.5)
    assert [word["word"] for word in pieces[0]["words"]] == ["Hello", "there."]
    assert [word["word"] for word in pieces[1]["words"]] == ["How", "are", "you?"]
    assert pieces[0]["speaker"] == "spk0"


def test_split_into_sentences_interpolates_without_words() -> None:
    row = _row("First one. Second one.", 0.0, 10.0)
    pieces = split_into_sentences([row])

    assert [piece["text"] for piece in pieces] == ["First one.", "Second one."]
    first, second = pieces
    # Character-fraction interpolation across the row's own timeline.
    assert first["start"] == pytest.approx(0.0)
    assert first["end"] == pytest.approx(second["start"])
    assert second["end"] == pytest.approx(10.0)
    assert 0.0 <= first["start"] <= first["end"] <= second["end"] <= 10.0


def test_split_into_sentences_identity_for_single_sentence() -> None:
    row = _row("Only one sentence here.", 0.0, 1.0)
    assert split_into_sentences([row])[0] is row


def test_split_into_sentences_identity_for_empty_text() -> None:
    row = _row("   ", 0.0, 1.0)
    assert split_into_sentences([row])[0] is row


def test_split_into_sentences_is_deterministic() -> None:
    row = _row("One. Two. Three.", 0.0, 3.0)
    assert split_into_sentences([row]) == split_into_sentences([row])


# ---------------------------------------------------------- safe-cut logic
def test_merge_when_previous_text_does_not_end_a_sentence() -> None:
    rows = [
        _row("fragment one two", 0.0, 1.0),
        _row("three four five", 1.05, 2.0),  # gap 0.05 < SENTENCE_GAP_S
    ]
    assert [unit.indices for unit in group_units(rows)] == [(0, 1)]


def test_cut_when_previous_text_ends_a_sentence() -> None:
    rows = [
        _row("Complete sentence.", 0.0, 1.0),
        _row("next fragment", 1.05, 2.0),
    ]
    assert [unit.indices for unit in group_units(rows)] == [(0,), (1,)]


def test_a_silence_gap_is_safe_but_never_ends_a_unit() -> None:
    # gap 0.5 >= SENTENCE_GAP_S is a *safe cut point*, but under the new rules a
    # gap can never end a unit: the two fragments stay one unit.
    rows = [
        _row("fragment with no end", 0.0, 1.0),
        _row("more words here", 1.5, 2.5),
    ]
    assert [unit.indices for unit in group_units(rows)] == [(0, 1)]


def test_even_a_huge_gap_does_not_end_a_unit() -> None:
    rows = [
        _row("fragment with no end", 0.0, 1.0),
        _row("more words here", 9.0, 10.0),  # gap 8.0 s
    ]
    assert [unit.indices for unit in group_units(rows)] == [(0, 1)]


def test_never_merge_across_a_speaker_change() -> None:
    rows = [
        _row("fragment with no end", 0.0, 1.0, "spk0"),
        _row("more words here", 1.0, 2.0, "spk1"),  # gap 0, no terminal
    ]
    assert [unit.indices for unit in group_units(rows)] == [(0,), (1,)]


def test_empty_rows_yield_empty_units_list() -> None:
    assert group_units([]) == []


def test_single_row_yields_one_unit() -> None:
    units = group_units([_row("just one row", 0.0, 1.0)])
    assert [unit.indices for unit in units] == [(0,)]
    assert units[0].source == "just one row"


# --------------------------------------------------- sentence-scoped units
def test_empty_row_is_safe_but_does_not_end_a_unit() -> None:
    # The empty row (index 1) does not end a unit; the run is one complete
    # sentence and stays whole (no mid-sentence cut).
    rows = [
        _row(_frag(30), 0.0, 1.0),
        _row("", 1.0, 1.0),
        _row(_frag(30), 1.0, 2.0),
        _row(_frag(30), 2.0, 3.0),
    ]
    units = group_units(rows)

    assert [unit.indices for unit in units] == [(0, 1, 2, 3)]
    assert unit_cut_stats(units, rows) == {
        "units": 1,
        "sentence_cuts": 0,
        "pause_cuts": 0,
        "speaker_cuts": 0,
        "mid_sentence_cuts": 0,
        "oversized": 0,
    }


def test_a_sentence_under_the_hard_ceiling_is_never_split() -> None:
    # No terminal punctuation anywhere: one sentence of 90 words.  There is no
    # target cap any more, so it is simply one unit (and not "oversized").
    rows = [
        _row(_frag(30), 0.0, 1.0),
        _row(_frag(30), 1.0, 2.0),
        _row(_frag(30), 2.0, 3.0),
    ]
    units = group_units(rows)

    assert [unit.indices for unit in units] == [(0, 1, 2)]
    assert len(units[0].source.split()) == 90
    stats = unit_cut_stats(units, rows)
    assert stats["oversized"] == 0
    assert stats["mid_sentence_cuts"] == 0


def test_a_fragment_run_closes_at_the_sentence_that_completes_it() -> None:
    # A fragment run plus the row that completes the sentence is ONE unit ending
    # at that sentence boundary; no size cap is involved.
    rows = [
        _row(_frag(20), 0.0, 1.0),
        _row(_frag(20) + " done.", 1.0, 2.0),
        _row("next fragment", 2.0, 3.0),
    ]
    units = group_units(rows)

    assert [unit.indices for unit in units] == [(0, 1), (2,)]
    assert units[0].source.endswith("done.")
    assert unit_cut_stats(units, rows)["oversized"] == 0


def test_a_sentence_over_the_hard_ceiling_by_time_stays_whole() -> None:
    rows = [
        _row("alpha beta", 0.0, 30.0),
        _row("gamma delta", 30.0, 60.0),  # 60.0 s span > HARD_UNIT_SECONDS
    ]
    units = group_units(rows)

    assert [unit.indices for unit in units] == [(0, 1)]
    assert (units[0].end - units[0].start) > HARD_UNIT_SECONDS
    assert unit_cut_stats(units, rows)["oversized"] == 1


def test_hard_ceiling_uses_the_largest_pause_gap() -> None:
    # 160 words > HARD_UNIT_WORDS and no sentence boundary: the hard ceiling
    # closes at the largest internal gap >= PAUSE_CUT_GAP_S (the only cut).
    rows = [
        _row(_frag(80), 0.0, 1.0),
        _row(_frag(80), 5.0, 6.0),  # gap 4.0 >= PAUSE_CUT_GAP_S
    ]
    units = group_units(rows)

    assert [unit.indices for unit in units] == [(0,), (1,)]
    stats = unit_cut_stats(units, rows)
    assert stats["pause_cuts"] == 1
    assert stats["mid_sentence_cuts"] == 0


def test_hard_ceiling_keeps_whole_when_no_pause_gap_exists() -> None:
    # 200 words > HARD_UNIT_WORDS but every gap is 0 (< PAUSE_CUT_GAP_S): the
    # single sentence stays whole and is counted oversized rather than cut.
    rows = [_row(_frag(25), float(i), float(i) + 1.0) for i in range(8)]
    units = group_units(rows)

    assert len(units) == 1
    assert units[0].indices == tuple(range(8))
    assert len(units[0].source.split()) > HARD_UNIT_WORDS
    stats = unit_cut_stats(units, rows)
    assert stats["pause_cuts"] == 0
    assert stats["oversized"] == 1
    assert stats["mid_sentence_cuts"] == 0


def test_a_single_row_over_the_hard_ceiling_is_its_own_unit() -> None:
    # A single member cannot be split, so the one-row unit over the hard ceiling
    # is left whole (counted oversized) rather than dropping content.
    rows = [_row(_frag(HARD_UNIT_WORDS + 5), 0.0, 1.0)]
    units = group_units(rows)

    assert [unit.indices for unit in units] == [(0,)]
    assert len(units[0].source.split()) > HARD_UNIT_WORDS
    assert unit_cut_stats(units, rows)["oversized"] == 1


# ------------------------------------------------------------------- shares
def test_shares_sum_to_one_and_empty_member_is_zero() -> None:
    rows = [
        _row("aa bb", 0.0, 1.0),
        _row("", 1.0, 1.0),  # absorbed empty member
    ]
    units = group_units(rows)
    assert len(units) == 1
    unit = units[0]
    assert unit.shares[0] == pytest.approx(1.0)
    assert unit.shares[1] == 0.0
    assert sum(unit.shares) == pytest.approx(1.0)


def test_all_empty_members_get_equal_shares() -> None:
    # Two empty rows are absorbed into one unit; with no characters at all the
    # shares fall back to equal.
    units = group_units([_row("", 0.0, 1.0), _row("   ", 1.0, 2.0)])

    assert len(units) == 1
    unit = units[0]
    assert unit.shares == (0.5, 0.5)
    assert unit.source == ""


def test_unit_source_joins_members_with_a_single_space() -> None:
    rows = [
        _row("actually running", 0.0, 1.0),
        _row("them, that's it", 1.0, 2.0),
    ]
    unit = group_units(rows)[0]
    assert unit.source == "actually running them, that's it"
    assert "  " not in unit.source


# ---------------------------------------------------- screenshot regression
def test_group_units_keeps_the_screenshot_sentence_pair_together() -> None:
    if not SCREENSHOT.exists():
        pytest.skip("screenshot cache not present")
    rows = json.loads(SCREENSHOT.read_text(encoding="utf-8"))

    units = group_units(rows)
    holder = next(
        (unit for unit in units if {1, 2, 3} <= set(unit.indices)), None
    )
    assert holder is not None
    assert "actually running them" in holder.source
    assert holder.end >= holder.start
    assert len(holder.shares) == len(holder.indices)
    assert sum(holder.shares) == pytest.approx(1.0)
    # The old cap-forced mid-sentence cut between rows 3 and 4 is gone.
    assert unit_cut_stats(units, rows)["mid_sentence_cuts"] == 0


# --------------------------------------------------------- split_translation
@pytest.mark.parametrize(
    ("text", "shares"),
    [
        ("Giá là 300 đô la cho GPU mới", [0.5, 0.5]),
        ("Sure, it'll respond, but you'll hate it.", [0.4, 0.6]),
        ("one two three four five six seven eight", [1 / 3, 1 / 3, 1 / 3]),
        ("Hello there, my friend.", [0.5, 0.5]),
        ("Đây là một câu dài hơn một chút nữa nhé", [0.25, 0.25, 0.25, 0.25]),
        ("Xinchao", [0.5, 0.5]),
        ("Xinchao ban", [1 / 3, 1 / 3, 1 / 3]),
        ("a b c d e", [0.2, 0.3, 0.5]),
    ],
)
def test_split_translation_reassembly_and_non_empty(
    text: str, shares: list[float]
) -> None:
    pieces = split_translation(text, shares)

    assert len(pieces) == len(shares)
    assert all(piece.strip() for piece in pieces), pieces
    # Reassembly: same non-whitespace characters in the same order.
    assert _nonspace(" ".join(pieces)) == _nonspace(text)


def test_split_translation_single_share() -> None:
    assert split_translation("  Hello world.  ", [1.0]) == ["Hello world."]


def test_split_translation_keeps_digit_token_intact() -> None:
    text = "Giá là 300 đô la cho GPU mới"
    pieces = split_translation(text, [0.5, 0.5])

    assert any("300" in piece for piece in pieces)
    assert sum(piece.count("300") for piece in pieces) == 1


def test_split_translation_keeps_number_across_spaces_together() -> None:
    text = "Version 3.5 costs 30 000 đồng cho bản mới"
    pieces = split_translation(text, [0.34, 0.33, 0.33])
    joined = " ".join(pieces)

    assert "3.5" in joined
    assert any("30 000" in piece for piece in pieces)


def test_split_translation_prefers_a_clause_boundary() -> None:
    text = "First part here, second part there, third part somewhere"
    pieces = split_translation(text, [0.5, 0.5])
    # A comma boundary is preferred over an arbitrary mid-clause word boundary.
    assert pieces[0].endswith(",") or pieces[1].startswith("second")


def test_split_translation_is_deterministic() -> None:
    text = "một hai ba bốn năm sáu bảy tám chín mười"
    assert split_translation(text, [0.5, 0.5]) == split_translation(text, [0.5, 0.5])


# ------------------------------------------- sparse path + exact guarantees
def test_split_translation_sparse_never_breaks_a_digit_token() -> None:
    # Two tokens, three members: split the non-digit "xy" and keep "300" whole.
    pieces = split_translation("xy 300", [1 / 3, 1 / 3, 1 / 3])

    assert len(pieces) == 3
    assert all(piece.strip() for piece in pieces), pieces
    assert sum(piece.count("300") for piece in pieces) == 1
    assert "300" in pieces
    assert not any(piece in ("3", "00") for piece in pieces)
    assert _nonspace(" ".join(pieces)) == _nonspace("xy 300")


def test_split_translation_single_digit_token_stays_intact_and_pads_empty() -> None:
    # One token, two members, and the only token contains a digit: it is kept
    # whole and the other piece is the documented unavoidable empty.
    pieces = split_translation("300", [0.5, 0.5])

    assert len(pieces) == 2
    assert pieces == ["300", ""]
    assert sum(piece.count("300") for piece in pieces) == 1
    assert _nonspace(" ".join(pieces)) == "300"


def test_split_translation_sparse_unavoidable_empty_pieces() -> None:
    # Fewer non-whitespace characters than members -> padded empty piece.
    pieces = split_translation("ab", [1 / 3, 1 / 3, 1 / 3])
    assert len(pieces) == 3
    assert pieces == ["a", "b", ""]
    assert _nonspace(" ".join(pieces)) == "ab"

    # Empty / whitespace-only translations have nothing to distribute.
    for text in ("", "   "):
        empty = split_translation(text, [0.5, 0.5])
        assert len(empty) == 2
        assert empty == ["", ""]
        assert _nonspace(" ".join(empty)) == _nonspace(text)


def test_split_translation_adjacent_number_is_documented_fallback() -> None:
    # The only cut sits between the two digit-bearing tokens, so it is used
    # (adjacency avoided only when an alternative cut exists).
    pieces = split_translation("30 000", [0.5, 0.5])

    assert pieces == ["30", "000"]
    assert _nonspace(" ".join(pieces)) == "30000"


# ----------------------------------------------------------- unit_cut_stats
def test_unit_cut_stats_classifies_a_speaker_cut() -> None:
    rows = [
        _row("complete one.", 0.0, 1.0, "spk0"),
        _row("next fragment", 1.0, 2.0, "spk1"),
        _row("more words", 2.0, 3.0, "spk1"),
    ]
    units = group_units(rows)

    assert [unit.indices for unit in units] == [(0,), (1, 2)]
    stats = unit_cut_stats(units, rows)
    assert stats["speaker_cuts"] == 1
    assert stats["mid_sentence_cuts"] == 0


def test_unit_cut_stats_mid_sentence_cuts_is_zero_on_a_fragment_run() -> None:
    rows = [
        _row(_frag(10), 0.0, 1.0),
        _row(_frag(10), 1.0, 2.0),
        _row(_frag(10) + " done.", 2.0, 3.0),
    ]
    units = group_units(rows)

    assert unit_cut_stats(units, rows)["mid_sentence_cuts"] == 0


def test_unit_cut_stats_requires_rows() -> None:
    rows = [
        _row("first fragment.", 0.0, 1.0),
        _row("second fragment.", 1.0, 2.0),
    ]
    units = group_units(rows)

    # Omitting rows entirely and passing None both raise, with a message that
    # explains why the rows are needed.
    with pytest.raises(TypeError, match="requires the ASR rows"):
        unit_cut_stats(units)
    with pytest.raises(TypeError, match="requires the ASR rows"):
        unit_cut_stats(units, None)


# ------------------------------------------------------------ determinism
def test_group_units_is_deterministic() -> None:
    rows = [
        _row("alpha beta", 0.0, 1.0),
        _row("gamma delta.", 1.0, 2.0),
        _row("epsilon zeta", 2.0, 3.0),
    ]
    assert group_units(rows) == group_units(rows)


# ------------------------------------------------------- unit_share_summary
def test_unit_share_summary_arithmetic() -> None:
    assert unit_share_summary([]) == {
        "units": 0,
        "segments": 0,
        "merged_segments": 0,
    }

    one = group_units([_row("hello world.", 0.0, 1.0)])
    assert unit_share_summary(one) == {
        "units": 1,
        "segments": 1,
        "merged_segments": 0,
    }

    merged = group_units(
        [
            _row("alpha beta", 0.0, 0.5),
            _row("gamma delta", 0.5, 1.0),
        ]
    )
    assert unit_share_summary(merged) == {
        "units": 1,
        "segments": 2,
        "merged_segments": 1,
    }

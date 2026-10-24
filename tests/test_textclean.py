"""Tests for hesitation/repetition + filler cleanup (M3-C1)."""

from __future__ import annotations

from vidub.utils.textclean import FILLERS, clean_hesitation, strip_fillers


# --------------------------------------------------------------------- fillers
def test_strip_fillers_removes_phrases_and_counts() -> None:
    out, n = strip_fillers("You know, I mean, this is basically fine")

    assert out == "this is fine"
    assert n == 3


def test_strip_fillers_removes_parenthetical() -> None:
    out, n = strip_fillers("This is (a note) fine")

    assert out == "This is fine"
    assert n == 1


def test_strip_fillers_collapses_double_spaces() -> None:
    out, n = strip_fillers("a  b")

    assert out == "a b"
    assert n == 0


def test_strip_fillers_extended_list() -> None:
    out, n = strip_fillers("But also, I just really like it")

    assert out == "I like it"
    assert n == 3


def test_strip_fillers_noop_without_fillers() -> None:
    assert strip_fillers("Hello world.") == ("Hello world.", 0)
    assert strip_fillers("") == ("", 0)


def test_fillers_constant_contains_ruled_entries() -> None:
    for entry in ("you know", "I mean", "but also", "just", "really", "and then"):
        assert entry in FILLERS


# ------------------------------------------------------------------ hesitation
def test_clean_hesitation_collapses_duplicate_words() -> None:
    assert clean_hesitation("một một") == ("một", 1)


def test_clean_hesitation_ellipsis_variant() -> None:
    out, n = clean_hesitation("một... một")

    assert out == "một"
    assert n == 2


def test_clean_hesitation_case_insensitive() -> None:
    assert clean_hesitation("Tôi tôi vừa")[0] == "Tôi vừa"


def test_clean_hesitation_collapses_comma_variant() -> None:
    assert clean_hesitation("nhiều, nhiều hoạt động")[0] == "nhiều, hoạt động"


def test_clean_hesitation_trailing_ellipsis() -> None:
    assert clean_hesitation("Xin chào...")[0] == "Xin chào"


def test_clean_hesitation_unicode_ellipsis() -> None:
    assert clean_hesitation("a… b")[0] == "a, b"


def test_clean_hesitation_keeps_clean_text_identical() -> None:
    text = "Đây là câu bình thường."

    assert clean_hesitation(text) == (text, 0)


def test_clean_hesitation_empty() -> None:
    assert clean_hesitation("") == ("", 0)

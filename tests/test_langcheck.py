"""Unit tests for the shared Vietnamese-language heuristic (M2-C1).

Pure-stdlib: no model, no ffmpeg. Covers the diacritic ratio, the mismatch
threshold edge, and the ``src_lang == "vi"`` escape hatch.
"""

from __future__ import annotations

import unicodedata

from vidub.utils.langcheck import (
    VN_CHARS,
    VN_MISMATCH_THRESHOLD,
    detect_mismatch,
    vn_diacritic_ratio,
)


def test_vn_chars_covers_base_letters_and_tones_both_cases() -> None:
    for char in "ăâêôơưđĂÂÊÔƠƯĐáàảãạắằẳẵặấầẩẫậ":
        assert char in VN_CHARS, char
    for char in "ÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÝỲỶỸỴ":
        assert char in VN_CHARS, char
    # plain ASCII letters are not Vietnamese markers
    assert "a" not in VN_CHARS and "e" not in VN_CHARS


def test_vn_diacritic_ratio_counts_alphabetic_words() -> None:
    assert vn_diacritic_ratio("") == 0.0
    assert vn_diacritic_ratio("   ") == 0.0
    assert vn_diacritic_ratio("hello world") == 0.0
    # xin | chào | bạn -> 2 of 3 carry diacritics
    assert vn_diacritic_ratio("xin chào bạn") == 2 / 3
    # uppercase Đ counts too
    assert vn_diacritic_ratio("Đây là clip") == 2 / 3
    # punctuation / digits are not part of a word
    assert vn_diacritic_ratio("Xin chào, bạn! 123") == 2 / 3


def test_vn_diacritic_ratio_normalises_nfd() -> None:
    decomposed = unicodedata.normalize("NFD", "chào")
    assert decomposed != "chào"
    assert vn_diacritic_ratio(decomposed) == 1.0


def test_detect_mismatch_none_for_english() -> None:
    assert detect_mismatch("This is the demonstration clip", "en") is None
    assert detect_mismatch("", "en") is None


def test_detect_mismatch_flags_vietnamese_for_non_vi_source() -> None:
    text = "thì tuấn triều tư trong văn thì trước triều tiên."
    message = detect_mismatch(text, "en")
    assert message is not None
    assert "Vietnamese" in message
    assert "--src-lang" in message


def test_detect_mismatch_ignores_vietnamese_source() -> None:
    text = "thì tuấn triều tư trong văn thì trước triều tiên."
    assert detect_mismatch(text, "vi") is None
    assert detect_mismatch(text, "VI") is None
    assert detect_mismatch(text, "vi-VN") is None


def test_detect_mismatch_threshold_edge() -> None:
    # 10 alphabetic words, 3 with diacritics -> exactly 0.30 -> flagged (>=).
    at_threshold = "a b c d e f g chào bạn đây"
    assert vn_diacritic_ratio(at_threshold) == VN_MISMATCH_THRESHOLD
    assert detect_mismatch(at_threshold, "en") is not None

    # 10 words, 2 with diacritics -> 0.20 -> below the threshold -> clean.
    below = "a b c d e f g h chào bạn"
    assert vn_diacritic_ratio(below) < VN_MISMATCH_THRESHOLD
    assert detect_mismatch(below, "en") is None

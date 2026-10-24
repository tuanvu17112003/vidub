"""Tests for the TTS pronunciation lexicon (M3-C3).

Fast in-process tests: no model is loaded. The pure helpers are exercised
directly; the wiring into the speak and timing stages is covered by
``tests/test_speak.py`` and ``tests/test_timing.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from vidub.utils.pronounce import (
    apply_pronunciation,
    load_pronunciation,
    parse_pronunciation,
)


# ----------------------------------------------------------------------- loader
def test_load_pronunciation_valid_rows_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "p.tsv"
    path.write_text(
        "# a comment\n"
        "\n"
        "VRAM\tVê-ram\n"
        "GPU\tGờ-Pê-U\n",
        encoding="utf-8",
    )

    assert load_pronunciation(path) == {"VRAM": "Vê-ram", "GPU": "Gờ-Pê-U"}


def test_parse_pronunciation_reports_bad_rows() -> None:
    mapping, problems = parse_pronunciation(
        "ok\tfine\n"
        "missing-tab\n"
        "\tno-written\n"
        "empty-spoken\t\n"
    )

    assert mapping == {"ok": "fine"}
    assert len(problems) == 3
    assert all("line" in problem for problem in problems)


def test_load_pronunciation_skips_bad_rows_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "p.tsv"
    path.write_text("good\tgreat\nbroken-row\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="vidub.utils.pronounce"):
        mapping = load_pronunciation(path)

    assert mapping == {"good": "great"}
    assert any("skipped" in record.getMessage() for record in caplog.records)


def test_load_pronunciation_missing_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        load_pronunciation(tmp_path / "nope.tsv")

    assert "pronunciation" in str(excinfo.value)


def test_shipped_example_loads_phonetic_forms() -> None:
    root = Path(__file__).resolve().parents[1]
    mapping = load_pronunciation(root / "samples" / "pronounce.example.tsv")

    assert mapping["VRAM"] == "Vê-ram"
    assert mapping["GPU"] == "Gờ-Pê-U"
    assert mapping["ChatGPT"] == "Chét-Gê-Pê-Tê"
    # an identity row would be pointless for speech, so it is not shipped
    assert "AI" not in mapping


# ------------------------------------------------------------------------ apply
def test_apply_pronunciation_case_insensitive_whole_word() -> None:
    text = "The VRAM is not VRAMish, but vram is fine."
    mapping = {"VRAM": "Vê-ram"}

    replaced, count = apply_pronunciation(text, mapping)

    assert replaced == "The Vê-ram is not VRAMish, but Vê-ram is fine."
    assert count == 2


def test_apply_pronunciation_longest_term_first() -> None:
    mapping = {"GPT": "Gờ-Pê-Tê", "ChatGPT": "Chét-Gê-Pê-Tê"}

    replaced, count = apply_pronunciation("ChatGPT beats GPT", mapping)

    assert replaced == "Chét-Gê-Pê-Tê beats Gờ-Pê-Tê"
    assert count == 2


def test_apply_pronunciation_word_boundary() -> None:
    replaced, count = apply_pronunciation(
        "VRAMs and xVRAM stay", {"VRAM": "Vê-ram"}
    )

    # embedded and plural occurrences are not whole-word matches
    assert replaced == "VRAMs and xVRAM stay"
    assert count == 0


def test_apply_pronunciation_no_match_unchanged() -> None:
    assert apply_pronunciation("nothing to see", {"VRAM": "Vê-ram"}) == (
        "nothing to see",
        0,
    )
    assert apply_pronunciation("anything", {}) == ("anything", 0)
    assert apply_pronunciation("", {"VRAM": "Vê-ram"}) == ("", 0)

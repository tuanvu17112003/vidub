"""Tests for opt-in punctuation/casing restoration (M4-T6).

No model is downloaded: the token-classification model is stubbed at the
``PunctRestorer`` boundary (or its ``_predict_marks`` / ``_predict_window``
internals), so the density guard, merge logic and casing are all unit-testable.
"""

from __future__ import annotations

import pytest

from vidub.manifest import Segment
from vidub.utils import punct as punct_mod
from vidub.utils.punct import PUNCT_DENSITY_MIN, PunctRestorer, restore_segments


# ------------------------------------------------------------------- density
def test_punct_density_values() -> None:
    assert punct_mod._punct_density("") == 0.0
    assert punct_mod._punct_density("   ") == 0.0
    assert punct_mod._punct_density("hello world") == 0.0
    assert punct_mod._punct_density("hello world.") == 0.5
    assert punct_mod._punct_density("Hi. Ok?") == 1.0
    assert PUNCT_DENSITY_MIN == 0.1


def test_label_to_mark_mapping() -> None:
    assert punct_mod._label_to_mark("0") == ""
    assert punct_mod._label_to_mark("") == ""
    assert punct_mod._label_to_mark("O") == ""
    assert punct_mod._label_to_mark(".") == "."
    assert punct_mod._label_to_mark(",") == ","
    assert punct_mod._label_to_mark("?") == "?"
    assert punct_mod._label_to_mark(":") == ":"
    assert punct_mod._label_to_mark("-") == "-"
    assert punct_mod._label_to_mark("PERIOD") == "."
    assert punct_mod._label_to_mark("comma") == ","
    assert punct_mod._label_to_mark("B-PERIOD") == "."
    assert punct_mod._label_to_mark("none") == ""
    assert punct_mod._label_to_mark("nonsense") == ""


# ---------------------------------------------------------- restore_segments
def test_restore_segments_density_guard_and_shared_restorer(monkeypatch) -> None:
    created: list = []

    class FakeRestorer:
        def __init__(self) -> None:
            created.append(self)
            self.calls: list[str] = []

        def restore(self, text: str) -> tuple[str, int]:
            self.calls.append(text)
            return text + ".", 1

    monkeypatch.setattr(punct_mod, "PunctRestorer", FakeRestorer)
    segments = [
        Segment(start=0.0, end=1.0, text="Already punctuated."),
        Segment(start=1.0, end=2.0, text="no punctuation here"),
        Segment(start=2.0, end=3.0, text="second one too"),
    ]

    total = restore_segments(segments, True)

    assert total == 2
    # a punctuated segment is never re-written
    assert segments[0].text == "Already punctuated."
    assert segments[1].text == "no punctuation here."
    assert segments[2].text == "second one too."
    # exactly one shared restorer for the whole call
    assert len(created) == 1
    assert created[0].calls == ["no punctuation here", "second one too"]


def test_restore_segments_disabled_is_noop(monkeypatch) -> None:
    def boom(*args, **kwargs):  # noqa: ANN002, ANN003 - test double
        raise AssertionError("the restorer must not be constructed when disabled")

    monkeypatch.setattr(punct_mod, "PunctRestorer", boom)
    segments = [Segment(start=0.0, end=1.0, text="no punctuation here")]

    assert restore_segments(segments, False) == 0
    assert segments[0].text == "no punctuation here"
    # empty input never constructs a restorer either
    assert restore_segments([], True) == 0


# -------------------------------------------------------------- restore logic
def test_restore_inserts_marks_and_capitalizes(monkeypatch) -> None:
    restorer = PunctRestorer()
    monkeypatch.setattr(
        restorer, "_predict_marks", lambda words: ["", ".", "", "?", ""]
    )

    text, count = restorer.restore("hello there general kenobi now")

    assert count == 2
    # first word capitalised; new sentence starts after '.' and '?'
    assert text == "Hello there. General kenobi? Now"


def test_restore_appends_every_predicted_mark(monkeypatch) -> None:
    restorer = PunctRestorer()
    monkeypatch.setattr(
        restorer, "_predict_marks", lambda words: [",", "", ":", "-"]
    )

    text, count = restorer.restore("alpha beta gamma delta")

    assert count == 3
    assert text == "Alpha, beta gamma: delta-"


def test_restore_empty_text_returns_empty() -> None:
    restorer = PunctRestorer()
    assert restorer.restore("") == ("", 0)
    assert restorer.restore("   ") == ("", 0)


# ------------------------------------------------------------- window merging
def test_windows_overlap_and_cover_every_token() -> None:
    assert punct_mod._windows(0) == []
    assert punct_mod._windows(5) == [(0, 5)]

    windows = punct_mod._windows(300)
    assert windows[0] == (0, 128)
    assert windows[1] == (112, 240)
    assert windows[2] == (224, 300)

    covered: set[int] = set()
    for start, end in windows:
        covered.update(range(start, end))
    assert covered == set(range(300))


def test_predict_marks_merges_overlapping_windows_first_wins(monkeypatch) -> None:
    restorer = PunctRestorer()
    restorer._model = object()  # bypass the lazy model load
    restorer._labels = {0: "0", 1: ".", 2: ","}

    monkeypatch.setattr(
        restorer,
        "_tokenize_words",
        lambda words: ([0, 1, 2, 3, 4], [1, 1, 1, 1, 1], [0, 0, 1, 2, 3]),
    )

    def fake_predict(input_ids, attention_mask):  # noqa: ANN001 - test double
        if input_ids == [0, 1, 2]:
            return [0, 0, 2]  # token 2 (word 1) -> ","
        if input_ids == [2, 3, 4]:
            return [1, 0, 0]  # token 2 (word 1) -> "." but must NOT win
        raise AssertionError(input_ids)

    monkeypatch.setattr(restorer, "_predict_window", fake_predict)
    monkeypatch.setattr(punct_mod, "_windows", lambda n: [(0, 3), (2, 5)])

    first = restorer._predict_marks(["a", "b", "c", "d"])
    second = restorer._predict_marks(["a", "b", "c", "d"])

    # first window wins at the overlap -> word 1 is "," not "."
    assert first == ["", ",", "", ""]
    assert second == first  # deterministic


# ------------------------------------------------------------- load failure
def test_model_load_failure_is_actionable(monkeypatch) -> None:
    transformers = pytest.importorskip("transformers")

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003 - test double
        raise OSError("network down")

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(boom)
    )
    monkeypatch.setattr(
        transformers.AutoModelForTokenClassification,
        "from_pretrained",
        staticmethod(boom),
    )

    restorer = PunctRestorer()
    with pytest.raises(RuntimeError) as excinfo:
        restorer._ensure_loaded()

    message = str(excinfo.value)
    assert "fullstop-punctuation" in message
    assert "--restore-punct" in message


# ------------------------------------------------------------------- CLI flag
def test_cli_restore_punct_flag_reaches_config() -> None:
    from vidub.cli import _build_cfg, build_parser

    parser = build_parser()
    assert _build_cfg(parser.parse_args(["run", "x.mp4"])).restore_punct is False
    assert (
        _build_cfg(
            parser.parse_args(["run", "x.mp4", "--restore-punct"])
        ).restore_punct
        is True
    )

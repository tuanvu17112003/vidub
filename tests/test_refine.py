"""Tests for the M5-T2 translation refine pass.

Fake generator / fusion / envit5 objects only -- no model and no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from vidub.engines.mt.base import BaseTranslator
from vidub.manifest import JobManifest
from vidub.utils.refine import REFINE_LABELS, _best_candidate, make_refiner


class FakeGenerator:
    """``sample_candidates`` returns the configured list, capped at ``k``."""

    def __init__(self, candidates: list[str]) -> None:
        self.candidates = list(candidates)
        self.calls: list[tuple[str, str, str, int]] = []

    def sample_candidates(self, text, src_lang, tgt_lang, k):  # noqa: ANN001
        self.calls.append((text, src_lang, tgt_lang, k))
        return [candidate for candidate in self.candidates[:k]]


class FakeFusion:
    """``fuse`` returns a fixed string and records every call."""

    def __init__(self, value: str) -> None:
        self.value = value
        self.calls: list[tuple[str, list[str], str, str]] = []

    def fuse(self, source, candidates, src_lang, tgt_lang):  # noqa: ANN001
        self.calls.append((source, list(candidates), src_lang, tgt_lang))
        return self.value


class FakeEnvit5:
    """Minimal MT engine used as the envit5 candidate source."""

    def __init__(self, value: str) -> None:
        self.value = value
        self.calls: list[list[str]] = []

    def translate(self, texts, src_lang, tgt_lang):  # noqa: ANN001
        self.calls.append(list(texts))
        return [self.value for _ in texts]


def _row(en: str, vi: str, **extra) -> dict:
    row = {
        "start": 0.0,
        "end": 2.0,
        "speaker": "spk0",
        "en": en,
        "vi": vi,
        "truncated": False,
    }
    row.update(extra)
    return row


def _entry(index: int, label: str) -> dict:
    return {"index": index, "label": label, "signals": {}}


# ------------------------------------------------------------- best-of-K
def test_best_of_k_prefers_number_survival_over_length() -> None:
    source = "the machine costs 300 dollars"
    current = "cỗ máy này giá 300 đô la mỹ và rất tốt"  # ratio 2.2 -> worst fit
    no_number = "chiếc máy này giá ba trăm đô"  # ratio 1.2 but drops 300
    numbered = "chiếc máy này giá 300 đô"  # ratio 1.2 and keeps 300

    rows = [_row(source, current, low_confidence=True)]
    entries = [_entry(0, "unnatural")]
    generator = FakeGenerator([no_number, numbered])

    refiner = make_refiner(generator)
    rows, entries, summary = refiner(rows, entries)

    assert rows[0]["vi"] == numbered
    assert "low_confidence" not in rows[0]  # stale flag cleared on accept
    assert summary["attempted"] == 1
    assert summary["improved"] == 1
    assert summary["fusion_used"] is False
    assert generator.calls == [(source, "en", "vi", 3)]
    assert entries[0]["label"] == "ok"
    assert summary["outcomes"][0]["action"] == "refine-best"
    assert summary["outcomes"][0] == {
        "index": 0,
        "action": "refine-best",
        "before": "unnatural",
        "after": "ok",
    }


def test_best_candidate_unit_prefers_numbers_then_length() -> None:
    # numbered but long beats a short candidate that drops the number
    assert (
        _best_candidate("it costs 300 dollars", ["ba trăm đô", "giá 300 đô la"])
        == "giá 300 đô la"
    )
    # without numbers the closest length ratio wins, ties go to the earliest
    assert _best_candidate("one two three", ["a", "b", "c d e"]) == "c d e"
    assert _best_candidate("one two three", ["aa bb", "cc dd"]) == "aa bb"


# ------------------------------------------------------------- envit5 candidate
def test_envit5_candidate_can_win() -> None:
    source = "the machine costs 300 dollars"
    current = "cỗ máy này giá 300 đô la mỹ và rất tốt"
    envit5_value = "chiếc máy này giá 300 đô"

    rows = [_row(source, current, low_confidence=True)]
    entries = [_entry(0, "unnatural")]
    generator = FakeGenerator([])
    envit5 = FakeEnvit5(envit5_value)

    refiner = make_refiner(generator, envit5=envit5)
    rows, entries, summary = refiner(rows, entries)

    assert envit5.calls == [[source]]
    assert rows[0]["vi"] == envit5_value
    assert summary["improved"] == 1


# ------------------------------------------------------------- fusion path
def test_fusion_path_used_when_fusion_provided() -> None:
    source = "we use Vigib daily"
    current = "chúng tôi dùng hằng ngày"  # entity-error: Vigib dropped
    fused = "chúng tôi dùng Vigib hằng ngày"

    rows = [_row(source, current)]
    entries = [_entry(0, "entity-error")]
    generator = FakeGenerator(["chúng tôi dùng mỗi ngày"])
    fusion = FakeFusion(fused)

    refiner = make_refiner(generator, fusion=fusion)
    rows, entries, summary = refiner(rows, entries)

    assert rows[0]["vi"] == fused
    assert entries[0]["label"] == "ok"
    assert summary["fusion_used"] is True
    assert summary["improved"] == 1
    assert len(fusion.calls) == 1
    assert fusion.calls[0][0] == source
    assert len(fusion.calls[0][1]) == 2  # generator candidate + current vi
    assert summary["outcomes"][0]["action"] == "refine-fusion"


# ------------------------------------------------------- strict-better rule
def test_same_label_candidate_is_never_accepted() -> None:
    source = "we use Vigib daily"
    current = "chúng tôi dùng hằng ngày"  # entity-error
    still_bad = "chúng tôi dùng mỗi ngày"  # also drops Vigib -> same label

    rows = [_row(source, current)]
    entries = [_entry(0, "entity-error")]
    generator = FakeGenerator([still_bad])

    refiner = make_refiner(generator)
    rows, entries, summary = refiner(rows, entries)

    assert rows[0]["vi"] == current  # original preserved
    assert entries[0]["label"] == "entity-error"
    assert summary["improved"] == 0
    assert summary["outcomes"][0]["action"] == "refine-skipped"
    assert summary["outcomes"][0]["after"] == "entity-error"


# ------------------------------------------------------- validator gate
def test_invalid_english_winner_is_rejected() -> None:
    source = "the weather is nice today"
    current = "thời tiết hôm nay đẹp trời"

    rows = [_row(source, current, low_confidence=True)]
    entries = [_entry(0, "unnatural")]
    generator = FakeGenerator(["thời tiết hôm nay rất đẹp"])
    fusion = FakeFusion("hello there friend")  # English -> lang validator fails

    refiner = make_refiner(generator, fusion=fusion)
    rows, entries, summary = refiner(rows, entries)

    assert rows[0]["vi"] == current  # untouched
    assert summary["improved"] == 0
    assert summary["outcomes"][0]["action"] == "refine-invalid"


def test_rejected_candidate_keeps_row_low_confidence() -> None:
    source = "the weather is nice today"
    current = "thời tiết hôm nay đẹp trời"

    rows = [_row(source, current, low_confidence=True)]
    entries = [_entry(0, "unnatural")]
    generator = FakeGenerator(["thời tiết hôm nay rất đẹp"])
    fusion = FakeFusion("hello there friend")  # invalid -> rejected

    refiner = make_refiner(generator, fusion=fusion)
    rows, entries, summary = refiner(rows, entries)

    assert rows[0]["vi"] == current
    assert rows[0]["low_confidence"] is True  # untouched on rejection
    assert summary["improved"] == 0


# ------------------------------------------------------- filtering
def test_non_refine_labels_are_untouched() -> None:
    rows = [_row("a b c", "x y z"), _row("one two", "")]
    entries = [_entry(0, "ok"), _entry(1, "omission")]
    generator = FakeGenerator(["candidate"])

    refiner = make_refiner(generator)
    rows, entries, summary = refiner(rows, entries)

    assert generator.calls == []
    assert summary == {
        "attempted": 0,
        "improved": 0,
        "fusion_used": False,
        "outcomes": [],
    }
    assert rows[0]["vi"] == "x y z"
    assert rows[1]["vi"] == ""


def test_passthrough_rows_are_never_refined() -> None:
    rows = [_row("Bạn có thể", "Bạn có thể", passthrough=True)]
    entries = [_entry(0, "unnatural")]
    generator = FakeGenerator(["khác"])

    refiner = make_refiner(generator)
    rows, entries, summary = refiner(rows, entries)

    assert generator.calls == []
    assert rows[0]["vi"] == "Bạn có thể"
    assert summary["attempted"] == 0


def test_refine_budget_caps_attempts_and_is_recorded() -> None:
    """At most ``max_units`` flagged units are refined; the rest are recorded."""
    rows = [
        _row("source one two", "bản dịch một"),
        _row("source three four", "bản dịch ba"),
        _row("source five six", "bản dịch năm"),
    ]
    entries = [_entry(0, "unnatural"), _entry(1, "unnatural"), _entry(2, "unnatural")]
    generator = FakeGenerator([])

    refiner = make_refiner(generator, max_units=1)
    rows, entries, summary = refiner(rows, entries)

    assert summary["attempted"] == 1
    assert summary["skipped_budget"] == 2
    assert summary["max_units"] == 1
    actions = [outcome["action"] for outcome in summary["outcomes"]]
    assert actions.count("refine-budget") == 2
    # the two budget-skipped rows are left exactly as they were
    assert rows[1]["vi"] == "bản dịch ba"
    assert rows[2]["vi"] == "bản dịch năm"
    # only the one refined unit was sent to the generator
    assert len(generator.calls) == 1


def test_refine_labels_constant() -> None:
    assert set(REFINE_LABELS) == {
        "mistranslation-risk",
        "inconsistent",
        "unnatural",
        "entity-error",
        "terminology-error",
    }


# ------------------------------------------------------- config + CLI
def test_refine_config_defaults() -> None:
    from vidub.config import JobConfig

    cfg = JobConfig()
    # Tri-state: None = auto -> the stage refines hunyuan only (see translate).
    assert cfg.hunyuan_refine is None
    assert cfg.fusion_model == ""
    assert cfg.hunyuan_samples == 3
    assert cfg.hunyuan_refine_max_units == 40


def test_cli_refine_flags_map_to_config() -> None:
    from vidub.cli import _build_cfg, build_parser

    parser = build_parser()
    cfg = _build_cfg(
        parser.parse_args(
            [
                "run",
                "x.mp4",
                "--hunyuan-refine",
                "--fusion-model",
                "foo",
                "--hunyuan-samples",
                "5",
            ]
        )
    )
    assert cfg.hunyuan_refine is True
    assert cfg.fusion_model == "foo"
    assert cfg.hunyuan_samples == 5

    default = _build_cfg(parser.parse_args(["run", "x.mp4"]))
    assert default.hunyuan_refine is None  # auto (on for hunyuan)
    assert default.fusion_model == ""
    assert default.hunyuan_samples == 3
    assert default.hunyuan_refine_max_units == 40

    off = _build_cfg(parser.parse_args(["run", "x.mp4", "--no-hunyuan-refine"]))
    assert off.hunyuan_refine is False

    capped = _build_cfg(
        parser.parse_args(["run", "x.mp4", "--hunyuan-refine-max-units", "7"])
    )
    assert capped.hunyuan_refine_max_units == 7


# ------------------------------------------------- stage error fallback (B)
class _FakeTranslator(BaseTranslator):
    def translate(self, texts, src_lang, tgt_lang):  # noqa: ANN001
        return ["Xin chào thế giới" for _ in texts]


def _make_job(tmp_path: Path) -> tuple[Path, Path]:
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "hash1"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "segments.json").write_text(
        json.dumps(
            [{"start": 0.0, "end": 2.0, "text": "Hello world", "speaker": "spk0"}]
        ),
        encoding="utf-8",
    )
    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="hash1",
        cache_dir=str(cache_dir),
        src_lang="en",
        tgt_lang="vi",
        # The refine gate keys on the hunyuan engine, so the refine-failure
        # contract must be exercised on a hunyuan job.
        mt_engine="hunyuan",
        paths={"segments": "segments.json"},
        stages={"transcribe": "done"},
    ).save(job_path)
    return job_path, artifacts_dir


def test_refiner_exception_reruns_battery_on_pristine_snapshot(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    import vidub.stages.translate as translate_mod
    from vidub.config import JobConfig

    job_path, artifacts_dir = _make_job(tmp_path)
    cfg = JobConfig(cache_dir=str(tmp_path / "cache"), hunyuan_refine=True)

    def mutate_then_raise(rows, entries):  # noqa: ANN001 - test double
        calls["n"] += 1
        rows[0]["vi"] = "SENTINEL-XYZ"
        raise RuntimeError("refine exploded")

    calls = {"n": 0}
    from vidub.stages.translate import qc as translate_qc

    monkeypatch.setattr(
        translate_qc,
        "_build_refiner",
        lambda cfg, glossary, manifest: mutate_then_raise,
    )

    translate_mod.translate(str(job_path), cfg, translator=_FakeTranslator())

    # Non-vacuity guard: the fake refiner must have actually run (a patch on
    # the wrong module would leave it at 0 and the assertions below trivial).
    assert calls["n"] == 1

    script = json.loads((artifacts_dir / "script.json").read_text(encoding="utf-8"))
    # The partial refine mutation was discarded: the fallback re-ran the battery
    # from the pristine pre-battery snapshot, so no sentinel survives.
    assert "SENTINEL-XYZ" not in script[0]["vi"]
    assert script[0]["vi"] == "Xin chào thế giới"

    artifact = json.loads(
        (artifacts_dir / "translation_qc.json").read_text(encoding="utf-8")
    )
    assert "refine" not in artifact  # the fallback re-run had no refiner

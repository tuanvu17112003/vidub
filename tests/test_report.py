"""Tests for report.json metadata + the resolved ASR device (M2-T6).

Fast in-process tests: the ASR engine is stubbed (no model load) and the report
is assembled with :func:`vidub.stages.export.build_report` directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidub import __version__
from vidub.config import JobConfig
from vidub.manifest import JobManifest, Segment
from vidub.stages import transcribe as transcribe_mod
from vidub.stages.export import build_report

META_KEYS = {
    "tool_version",
    "source_url",
    "asr_model",
    "asr_device",
    "sep_device",
    "mt_engine",
    "tts_engine",
    "mode",
    "cloned_speakers",
    "total_seconds",
}


# --------------------------------------------------------------- ASR device
class _StubASR:
    """Fake engine recording which device it resolved to."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    def transcribe(self, wav_path, language=None, word_timestamps=True):
        return [Segment(start=0.0, end=1.0, text="hello")]


class _DeviceLessASR:
    """Engine that exposes no ``device`` attribute at all."""

    def transcribe(self, wav_path, language=None, word_timestamps=True):
        return []


def _stub_job(tmp_path: Path) -> tuple[Path, JobConfig]:
    cache = tmp_path / "cache"
    artifacts = cache / "h1"
    artifacts.mkdir(parents=True)
    (artifacts / "audio.wav").write_bytes(b"RIFF")
    job = artifacts / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="h1",
        cache_dir=str(cache),
        asr_model="tiny",
        paths={"audio": "audio.wav"},
        stages={"ingest": "done"},
    ).save(job)
    return job, JobConfig(cache_dir=str(cache))


def test_transcribe_records_resolved_asr_device(monkeypatch, tmp_path) -> None:
    job, cfg = _stub_job(tmp_path)
    monkeypatch.setattr(
        transcribe_mod, "build_asr", lambda model, device="auto": _StubASR("cuda")
    )

    manifest = transcribe_mod.transcribe(str(job), cfg)

    assert manifest.asr_device == "cuda"
    # and it is persisted in job.json for later (skipped) runs
    assert JobManifest.load(job).asr_device == "cuda"


def test_transcribe_asr_device_is_empty_when_unknown(monkeypatch, tmp_path) -> None:
    job, cfg = _stub_job(tmp_path)
    monkeypatch.setattr(
        transcribe_mod, "build_asr", lambda model, device="auto": _DeviceLessASR()
    )

    manifest = transcribe_mod.transcribe(str(job), cfg)

    assert manifest.asr_device == ""


# ------------------------------------------------------------- report meta
def _manifest(tmp_path: Path, **overrides) -> JobManifest:
    fields = dict(
        input_path="x/demo_en.mp4",
        video_hash="h",
        cache_dir=str(tmp_path),
        mode="dub",
        asr_model="vinai/PhoWhisper-small",
        asr_device="cpu",
        sep_device="cuda",
        mt_engine="envit5",
        tts_engine="piper",
        duration=4.0,
        stages={
            name: "done"
            for name in (
                "ingest",
                "transcribe",
                "translate",
                "speak",
                "timing",
                "mix",
                "export",
            )
        },
    )
    fields.update(overrides)
    return JobManifest(**fields)


def test_build_report_meta_shape_and_values(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    timings = {"ingest": 0.5, "transcribe": 1.25, "export": 0.25}

    report = build_report(manifest, timings)

    meta = report["meta"]
    assert set(meta) == META_KEYS
    assert meta["tool_version"] == __version__ == "0.11.0"
    assert meta["source_url"] == ""
    assert meta["asr_model"] == "vinai/PhoWhisper-small"
    assert meta["asr_device"] == "cpu"
    assert meta["sep_device"] == "cuda"
    assert meta["mt_engine"] == "envit5"
    assert meta["tts_engine"] == "piper"
    assert meta["mode"] == "dub"
    assert meta["cloned_speakers"] == 0
    assert meta["total_seconds"] == pytest.approx(2.0)


def test_build_report_meta_defaults_without_timings(tmp_path) -> None:
    report = build_report(_manifest(tmp_path, asr_device=""))

    assert report["meta"]["asr_device"] == ""
    assert report["meta"]["sep_device"] == "cuda"
    assert report["meta"]["total_seconds"] == 0.0


# --------------------------------------------------- source_url + cloning (M3)
def test_build_report_source_url_from_manifest(tmp_path) -> None:
    """A URL input surfaces in ``meta`` and at the top level (M3-T6)."""
    manifest = _manifest(tmp_path, source_url="https://example.com/clip.mp4")

    report = build_report(manifest)

    assert report["meta"]["source_url"] == "https://example.com/clip.mp4"
    assert report["source_url"] == "https://example.com/clip.mp4"


def test_build_report_source_url_empty_for_local_file(tmp_path) -> None:
    report = build_report(_manifest(tmp_path))

    assert report["meta"]["source_url"] == ""
    assert report["source_url"] == ""


def test_build_report_cloned_speakers_counts_non_fallback(tmp_path) -> None:
    """``meta.cloned_speakers`` counts references.json rows that are not fallbacks."""
    manifest = _manifest(tmp_path)
    artifacts = manifest.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "references.json").write_text(
        json.dumps(
            [
                {"speaker": "spk0", "file": "references/spk0.wav", "fallback": False},
                {"speaker": "spk1", "file": "references/spk1.wav", "fallback": False},
                {"speaker": "spk2", "file": "", "fallback": True},
            ]
        ),
        encoding="utf-8",
    )

    report = build_report(manifest)

    assert report["meta"]["cloned_speakers"] == 2


def test_build_report_cloned_speakers_is_zero_without_sidecar(tmp_path) -> None:
    report = build_report(_manifest(tmp_path))

    assert report["meta"]["cloned_speakers"] == 0


# --------------------------------------------------- M3-C1 pacing + cleanup
def test_build_report_pacing_and_cleanup_default_zero(tmp_path) -> None:
    report = build_report(_manifest(tmp_path))

    assert report["pacing"] == {
        "stretched": 0,
        "retranslated": 0,
        "clipped": 0,
        "smart_trims": 0,
        "clips": [],
        "max_stretch": 1.0,
        "max_neighbor_delta": 0.0,
    }
    assert report["cleanup"] == {
        "hesitations": 0,
        "filler_removals": 0,
        "style_fixes": 0,
    }
    assert report["glossary"]["spliced"] == 0
    assert report["glossary"]["enforced"] == 0


# ------------------------------------------------- M3-C3 pronunciation block
def test_build_report_pronunciation_defaults_to_zero(tmp_path) -> None:
    report = build_report(_manifest(tmp_path))

    assert report["pronunciation"] == {"replacements": 0}


def test_build_report_reads_pronunciation_sidecar(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    artifacts = manifest.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "pronunciation.json").write_text(
        json.dumps({"replacements": 3}), encoding="utf-8"
    )

    report = build_report(manifest)

    assert report["pronunciation"] == {"replacements": 3}


def test_build_report_reads_timing_and_glossary_sidecars(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    artifacts = manifest.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "timing.json").write_text(
        json.dumps(
            {
                "stretched": 3,
                "retranslated": 1,
                "clipped": 1,
                "smart_trims": 2,
                "clips": [
                    {
                        "idx": 4,
                        "start": 10.0,
                        "seconds_trimmed": 0.2,
                        "reason": "x",
                        "smart_trim": False,
                    }
                ],
                "max_stretch": 1.2,
                "max_neighbor_delta": 0.18,
                "filler_removals": 2,
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "glossary.json").write_text(
        json.dumps(
            {
                "terms": 1,
                "replacements": 0,
                "spliced": 0,
                "enforced": 4,
                "hesitations": 5,
                "style_fixes": 2,
            }
        ),
        encoding="utf-8",
    )

    report = build_report(manifest)

    assert report["pacing"]["stretched"] == 3
    assert report["pacing"]["clipped"] == 1
    assert report["pacing"]["smart_trims"] == 2
    assert report["pacing"]["clips"][0]["idx"] == 4
    assert report["pacing"]["max_stretch"] == 1.2
    assert report["pacing"]["max_neighbor_delta"] == 0.18
    assert report["cleanup"] == {
        "hesitations": 5,
        "filler_removals": 2,
        "style_fixes": 2,
    }
    assert report["glossary"]["enforced"] == 4


# ------------------------------------------------------------- FIT-1 rate fit
def test_build_report_mirrors_rate_fit_block(tmp_path) -> None:
    """FIT-1: the timing sidecar's ``rate_fit`` block is mirrored additively."""
    manifest = _manifest(tmp_path)
    artifacts = manifest.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    rate_fit = {
        "resynthesized": 2,
        "kept": 1,
        "failed": 0,
        "global_length_scale": 1.05,
        "recentered": 2,
    }
    (artifacts / "timing.json").write_text(
        json.dumps({"stretched": 1, "rate_fit": rate_fit}), encoding="utf-8"
    )

    report = build_report(manifest)

    assert report["pacing"]["rate_fit"] == rate_fit
    assert report["pacing"]["stretched"] == 1


def test_build_report_omits_rate_fit_without_sidecar(tmp_path) -> None:
    """An old/opt-out timing sidecar keeps the exact original pacing shape."""
    report = build_report(_manifest(tmp_path))

    assert "rate_fit" not in report["pacing"]


# ------------------------------------------- UA-1 per-unit rows (sentence units)
def test_build_report_renders_per_unit_pacing_and_rows(tmp_path) -> None:
    """UA-1: a per-unit timing sidecar + member-carrying ``timed.json`` builds.

    Sentence units change the row cardinality (one row per unit) and add
    ``members``; the report's pacing block is a straight mirror of the timing
    sidecar counters, so per-unit counts must render unchanged. This is the
    report-side regression for the UA-1 unit rows (the ``members`` field and the
    clipped row's index are unit-scoped).
    """
    manifest = _manifest(tmp_path)
    artifacts = manifest.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    timing = {
        "stretched": 2,
        "retranslated": 0,
        "clipped": 1,
        "smart_trims": 0,
        "clips": [
            {
                "idx": 1,
                "start": 3.0,
                "seconds_trimmed": 0.2,
                "reason": "sentence unit: no re-translation retry",
                "smart_trim": False,
            }
        ],
        "max_stretch": 1.2,
        "max_neighbor_delta": 0.05,
        "filler_removals": 0,
        "rate_fit": {
            "resynthesized": 1,
            "kept": 0,
            "failed": 0,
            "global_length_scale": 1.1,
            "recentered": 0,
        },
    }
    (artifacts / "timing.json").write_text(json.dumps(timing), encoding="utf-8")

    # Per-unit timed.json rows: the UA-1 shape carries ``members`` and the row's
    # ``idx`` is the UNIT index, not a script-segment index.
    (artifacts / "timed.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "members": [0, 1],
                    "file": "speech/unit_0000.wav",
                    "start": 0.0,
                    "end": 2.0,
                    "slot_start": 0.0,
                    "window": 2.0,
                    "stretch": 1.0,
                    "action": "asis",
                },
                {
                    "idx": 1,
                    "members": [2],
                    "file": "speech/unit_0001.wav",
                    "start": 3.0,
                    "end": 4.0,
                    "slot_start": 3.0,
                    "window": 1.0,
                    "stretch": 1.2,
                    "action": "clip",
                    "clipped": True,
                },
            ]
        ),
        encoding="utf-8",
    )

    report = build_report(manifest)

    # per-unit counters are mirrored verbatim (counts are unit rows)
    assert report["pacing"]["stretched"] == 2
    assert report["pacing"]["clipped"] == 1
    assert report["pacing"]["clips"][0]["idx"] == 1
    assert report["pacing"]["max_stretch"] == 1.2
    assert report["pacing"]["rate_fit"] == timing["rate_fit"]
    # the report still renders every expected block with the timing stage done
    assert report["stages"]["timing"]["status"] == "done"
    assert set(report) >= {"pacing", "cleanup", "stages", "meta", "outputs"}

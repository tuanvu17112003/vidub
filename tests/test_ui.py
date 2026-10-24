"""Tests for the local Gradio UI (M3-T5).

Gradio is an optional extra, so the whole module is skipped when it is missing.
The app is *built* (never launched) in-process; the one server test runs in a
subprocess and is skip-safe.
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

import vidub.pipeline as pipeline
import vidub.ui as ui
from vidub.config import (
    DEFAULT_ASR_MODEL,
    DEFAULT_SUB_FONT_SIZE,
    DEFAULT_SUB_POSITION,
    JobConfig,
)
from vidub.manifest import JobManifest
from vidub.pipeline import PipelineCancelled


# ------------------------------------------------------------------ build_app
def test_build_app_returns_app_and_state_without_launching() -> None:
    app, state = ui.build_app(JobConfig())

    assert isinstance(app, gr.Blocks)
    assert isinstance(state, dict)
    assert state["status"] == "idle"
    assert state["report"] is None
    assert state["qc_summary"] == ""


def test_snapshot_has_one_value_per_output() -> None:
    out = ui.snapshot(ui.new_state())

    assert len(out) == 8
    assert isinstance(out[0], str)  # status markdown
    assert out[1] == []  # stages table
    assert out[2] == []  # script table
    assert out[7] == ""  # Quality results panel


# --------------------------------------------------------------- form mapping
def test_form_to_config_maps_fields(tmp_path: Path) -> None:
    glossary = tmp_path / "g.tsv"

    cfg = ui.form_to_config(
        mode="voice-over",
        asr_model="cpu-base",
        tts_engine="piper",
        diarize=True,
        glossary_file=str(glossary),
        loudness_target=-14.0,
        loudness_tp=-1.0,
        loudness_lra=9.0,
        cache_dir="mycache",
        mt_engine="hunyuan",
        hunyuan_refine=True,
        hunyuan_samples=5,
        fusion_model="x",
        sentence_units=False,
        restore_punct=True,
        qc_translation=False,
        tts_qc_mos_min=2.5,
    )

    assert cfg.mode == "voice-over"
    assert cfg.asr_model == "cpu-base"
    assert cfg.tts_engine == "piper"
    assert cfg.diarize is True
    assert cfg.glossary_path == str(glossary)
    assert cfg.loudness_target_i == -14.0
    assert cfg.loudness_tp == -1.0
    assert cfg.loudness_lra == 9.0
    assert cfg.cache_dir == "mycache"
    # M4/M5 engine + QC fields.
    assert cfg.mt_engine == "hunyuan"
    assert cfg.hunyuan_refine is True
    assert cfg.hunyuan_samples == 5
    assert cfg.fusion_model == "x"
    assert cfg.sentence_units is False
    assert cfg.restore_punct is True
    assert cfg.qc_translation is False
    assert cfg.tts_qc_mos_min == 2.5


def test_form_to_config_rejects_unknown_mt_engine() -> None:
    with pytest.raises(ValueError):
        ui.form_to_config(mt_engine="definitely-not")


def test_form_to_config_defaults() -> None:
    cfg = ui.form_to_config()

    assert cfg.mode == "dub"
    assert cfg.asr_model == DEFAULT_ASR_MODEL
    assert cfg.tts_engine == "piper"
    assert cfg.sentence_units is True
    assert cfg.cache_dir == JobConfig().cache_dir
    assert cfg.loudness_target_i == JobConfig().loudness_target_i


def test_form_to_config_exposes_voxcpm2_as_an_opt_in_tts_engine() -> None:
    assert "voxcpm2" in ui.TTS_ENGINE_CHOICES
    cfg = ui.form_to_config(tts_engine="voxcpm2")
    assert cfg.tts_engine == "voxcpm2"


def test_form_to_config_maps_voxcpm2_voice_reference(tmp_path: Path) -> None:
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"reference audio")

    cfg = ui.form_to_config(tts_engine="voxcpm2", ref_voice=str(reference))

    assert cfg.ref_voice == str(reference)


def test_form_to_config_maps_voxcpm2_voice_profile() -> None:
    cfg = ui.form_to_config(tts_engine="voxcpm2", voxcpm2_voice="vi_male_calm")

    assert cfg.voxcpm2_voice == "vi_male_calm"


def test_handle_run_passes_voice_reference_to_pipeline(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"reference audio")
    captured: list[JobConfig] = []
    monkeypatch.setattr(
        ui, "run_pipeline", lambda _video, cfg, **_kwargs: captured.append(cfg) or {}
    )
    state = ui.new_state()

    ui._handle_run(
        state,
        str(video),
        "",
        "dub",
        DEFAULT_SUB_POSITION,
        DEFAULT_SUB_FONT_SIZE,
        False,
        DEFAULT_ASR_MODEL,
        "voxcpm2",
        False,
        False,
        False,
        None,
        None,
        None,
        None,
        str(tmp_path),
        voxcpm2_voice="vi_male_calm",
        ref_voice_path=str(reference),
    )
    thread = state["thread"]
    if thread is not None:
        thread.join()

    assert captured[0].ref_voice == str(reference)
    assert captured[0].voxcpm2_voice == "vi_male_calm"


def test_form_to_config_rejects_bad_combo() -> None:
    with pytest.raises(ValueError):
        ui.form_to_config(asr_model="definitely-not-a-model")

    with pytest.raises(ValueError, match="clone-speakers"):
        ui.form_to_config(
            clone_speakers=True, tts_engine="piper", allow_nc_models=True
        )


# ------------------------------------------- subtitles-only UI (SU-1-T3)
def test_subtitles_only_mode_label_list() -> None:
    """The dropdown carries the friendly label while keeping the mode value."""
    assert ("Subtitles only (no TTS)", "subtitles-only") in ui.MODE_CHOICES_LABELED
    assert [value for _label, value in ui.MODE_CHOICES_LABELED] == ui.MODE_CHOICES
    assert ui.MODE_CHOICES == ["dub", "voice-over", "subtitles-only"]


def test_new_state_carries_the_subtitle_style() -> None:
    state = ui.new_state()
    assert state["mode"] == "dub"
    assert state["sub_position"] == DEFAULT_SUB_POSITION
    assert state["sub_font_size"] == DEFAULT_SUB_FONT_SIZE
    assert state["burn"] is False


def test_form_to_config_maps_subtitle_style() -> None:
    cfg = ui.form_to_config(
        mode="subtitles-only", sub_position="top", sub_font_size=30
    )
    assert cfg.mode == "subtitles-only"
    assert cfg.sub_position == "top"
    assert cfg.sub_font_size == 30
    assert cfg.sub_margin_v == 24  # per-position default for top

    center = ui.form_to_config(mode="subtitles", sub_position="center")
    assert center.mode == "subtitles-only"  # alias normalised
    assert center.sub_position == "center"
    assert center.sub_margin_v == 0  # per-position default for center


def test_form_to_config_subtitles_only_skips_tts_guard() -> None:
    """An unusable TTS config must not block a subtitles-only UI run."""
    cfg = ui.form_to_config(
        mode="subtitles-only", tts_engine="vixtts", allow_nc_models=False
    )
    assert cfg.mode == "subtitles-only"
    # dub keeps the license guard.
    with pytest.raises(ValueError):
        ui.form_to_config(mode="dub", tts_engine="vixtts", allow_nc_models=False)


def test_form_to_config_subtitles_only_clone_speakers_errors() -> None:
    with pytest.raises(ValueError, match="subtitles-only"):
        ui.form_to_config(
            mode="subtitles-only",
            clone_speakers=True,
            tts_engine="vixtts",
            allow_nc_models=True,
        )


def test_resolve_input_prefers_url_and_requires_something() -> None:
    assert ui.resolve_input("/tmp/a.mp4", "https://x/y") == "https://x/y"
    assert ui.resolve_input("/tmp/a.mp4", "") == "/tmp/a.mp4"
    with pytest.raises(ValueError):
        ui.resolve_input(None, "")


# ------------------------------------------------------------------- cancel
def test_run_pipeline_cancel_before_any_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(pipeline.ingest_mod, "video_hash", lambda video: "hash1")
    monkeypatch.setattr(
        pipeline.ingest_mod, "ingest", lambda *a, **k: calls.append("ingest")
    )

    with pytest.raises(PipelineCancelled):
        pipeline.run_pipeline(
            "video.mp4", JobConfig(cache_dir=str(tmp_path)), cancel_cb=lambda: True
        )

    assert calls == []


def test_run_pipeline_cancel_between_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(pipeline.ingest_mod, "video_hash", lambda video: "hash1")

    def fake_ingest(video: str, cfg: JobConfig) -> JobManifest:
        calls.append("ingest")
        return JobManifest(
            cache_dir=str(tmp_path),
            video_hash="hash1",
            stages={"ingest": "done"},
        )

    monkeypatch.setattr(pipeline.ingest_mod, "ingest", fake_ingest)

    # False on the first boundary (ingest runs), True on the next -> stop.
    flags = iter([False, True])
    with pytest.raises(PipelineCancelled):
        pipeline.run_pipeline(
            "video.mp4",
            JobConfig(cache_dir=str(tmp_path)),
            cancel_cb=lambda: next(flags),
        )

    assert calls == ["ingest"]


def test_cancel_sets_the_flag() -> None:
    state = ui.new_state()
    state["status"] = "running"

    ui.cancel(state)

    assert state["cancel"] is True
    assert state["status"] == "running"  # the worker flips it to cancelled


def test_cmd_ui_reports_missing_gradio(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without the extra, `vidub ui` exits 1 with the install hint."""
    import argparse

    import vidub.cli as cli

    monkeypatch.setitem(sys.modules, "vidub.ui", None)

    rc = cli.cmd_ui(argparse.Namespace(host="127.0.0.1", port=7860))

    assert rc == 1
    assert 'pip install -e ".[ui]"' in capsys.readouterr().err


def test_worker_reports_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")  # so video_hash() can stat it
    state = ui.new_state()
    state["status"] = "running"
    state["cancel"] = True

    def fake_run_pipeline(video_arg, cfg, **kwargs):
        if kwargs["cancel_cb"]():
            raise PipelineCancelled("cancelled")
        return {}

    monkeypatch.setattr(ui, "run_pipeline", fake_run_pipeline)

    ui._worker(state, str(video), JobConfig(cache_dir=str(tmp_path)))

    assert state["status"] == "cancelled"


# ----------------------------------------------------- force + QC panel (UI-1-T2)
def test_handle_run_passes_force_stages(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")  # so video_hash() can stat it
    captured: list = []

    def fake_run_pipeline(video_arg, cfg, **kwargs):
        captured.append(kwargs.get("force"))
        return {}

    monkeypatch.setattr(ui, "run_pipeline", fake_run_pipeline)

    def run_once(force_stages):
        state = ui.new_state()
        ui._handle_run(
            state,
            str(video),
            "",
            "dub",
            DEFAULT_SUB_POSITION,
            DEFAULT_SUB_FONT_SIZE,
            False,
            DEFAULT_ASR_MODEL,
            "piper",
            False,
            False,
            False,
            None,
            None,
            None,
            None,
            str(tmp_path),
            force_stages=force_stages,
        )
        state["thread"].join()
        return state

    run_once(["translate"])
    assert captured[-1] == ["translate"]

    run_once(None)
    assert captured[-1] in (None, [])

    run_once([])
    assert captured[-1] in (None, [])


def test_handle_run_passes_burn_and_style(monkeypatch, tmp_path: Path) -> None:
    """The burn checkbox + subtitle style reach ``run_pipeline``/``JobConfig``."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")  # so video_hash() can stat it
    captured: list[dict] = []

    def fake_run_pipeline(video_arg, cfg, **kwargs):
        captured.append({"cfg": cfg, "kwargs": kwargs})
        return {}

    monkeypatch.setattr(ui, "run_pipeline", fake_run_pipeline)
    state = ui.new_state()

    ui._handle_run(
        state,
        str(video),
        "",
        "subtitles-only",
        "top",
        30,
        True,
        DEFAULT_ASR_MODEL,
        "piper",
        False,
        False,
        False,
        None,
        None,
        None,
        None,
        str(tmp_path),
    )
    state["thread"].join()

    assert captured[-1]["kwargs"]["burn"] is True
    cfg = captured[-1]["cfg"]
    assert cfg.mode == "subtitles-only"
    assert cfg.sub_position == "top"
    assert cfg.sub_font_size == 30
    # the run state mirrors the chosen style, and only the mode's stages run
    assert state["mode"] == "subtitles-only"
    assert state["sub_position"] == "top"
    assert state["sub_font_size"] == 30
    assert state["burn"] is True
    assert state["expected_stages"] == [
        "ingest", "transcribe", "translate", "export",
    ]


def test_qc_summary_shows_subtitles_only_mode(tmp_path: Path) -> None:
    report = {
        "mode": "subtitles-only",
        "engines": {"tts": "skipped (subtitles-only)"},
        "quality": {"verdict": "pass", "levels": {}},
        "stages": {
            "speak": {"status": "skipped"},
            "timing": {"status": "skipped"},
            "mix": {"status": "skipped"},
        },
    }

    summary = ui._qc_summary(tmp_path, report)

    assert "**Mode:** `subtitles-only`" in summary
    assert "`speak`, `timing`, `mix`" in summary
    assert "TTS" in summary


def test_qc_summary_old_artifact_keeps_the_previous_shape(tmp_path: Path) -> None:
    """A report without ``mode`` (old artifacts) gains no new block."""
    summary = ui._qc_summary(tmp_path, {"quality": {"verdict": "pass"}})

    assert "Mode:" not in summary


def test_qc_summary_renders_artifacts(tmp_path: Path) -> None:
    (tmp_path / "transcript_qc.json").write_text(
        json.dumps(
            {
                "verdict": "pass",
                "counts": {"ok": 74, "hallucination": 0, "garbled": 1},
                "flagged_ratio_unresolved": 0.25,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "translation_qc.json").write_text(
        json.dumps(
            {
                "verdict": "review",
                "counts": {"ok": 73, "number-error": 1},
                "refine": {"attempted": 7, "improved": 5, "fusion_used": True},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tts_qc.json").write_text(
        json.dumps(
            {
                "mos_mean": 3.94,
                "mos_min": 3.0,
                "scored": 74,
                "low_mos": 0,
                "retried": 2,
                "resolved": 1,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "subtitles_qc.json").write_text(
        json.dumps(
            {
                "cues_in": 74,
                "cues_out": 136,
                "split": 33,
                "cps_max_observed": 32.14,
                "violations": {"cps": 66, "duration": 53},
            }
        ),
        encoding="utf-8",
    )
    report = {
        "quality": {
            "verdict": "review",
            "levels": {
                "transcript": "pass",
                "translation": "review",
                "subtitles": "review",
            },
        }
    }

    text = ui._qc_summary(tmp_path, report)

    assert "**Verdict:** `review`" in text
    assert "transcript=pass" in text
    assert "flagged ratio unresolved: 0.25" in text
    assert "garbled=1" in text  # non-zero transcript count
    assert "unresolved flags (battery): 1" in text
    assert "refine: attempted 7, improved 5, fusion used yes" in text
    assert "rejected 2" in text
    assert "mos_mean: 3.94" in text
    assert "cues_in: 74" in text
    assert "violations: cps=66, duration=53" in text


def test_qc_summary_degrades_without_artifacts(tmp_path: Path) -> None:
    # No files at all: the verdict and every section say "not run".
    text = ui._qc_summary(tmp_path, None)
    assert "### Quality results" in text
    assert "**Verdict:** not run" in text
    assert text.count("not run") >= 5  # verdict + 4 sections

    # A translation QC file without a refine block reports "refine: off".
    (tmp_path / "translation_qc.json").write_text(
        json.dumps({"verdict": "pass", "counts": {"ok": 3}}), encoding="utf-8"
    )
    text = ui._qc_summary(tmp_path, None)
    assert "refine: off" in text


# ------------------------------------------------ presets + pre-flight (UI-1-T3)
def test_preset_hunyuan_updates() -> None:
    out = ui.preset_hunyuan()

    assert out[0]["value"] == "hunyuan"
    assert out[1]["value"] == ui.DEFAULT_HUNYUAN_MODEL
    assert out[2]["value"] == ui.DEFAULT_HUNYUAN_HOST
    assert out[3]["value"] is False
    assert out[4]["value"] == ""
    assert isinstance(out[5]["value"], str)


def test_preset_refine_updates() -> None:
    out = ui.preset_refine()

    assert out[0]["value"] is True
    assert out[1]["value"] == 3
    assert out[2]["value"] == ""
    assert isinstance(out[3]["value"], str)


def test_preset_fusion_is_evaluation_only() -> None:
    out = ui.preset_fusion()

    assert out[0]["value"] is True
    assert out[1]["value"] == ui.FUSION_MODEL
    hint = out[2]["value"].lower()
    assert "evaluation only" in hint
    assert "regression" in hint


def test_preset_defaults_restores_envit5() -> None:
    out = ui.preset_defaults()

    assert out[0]["value"] == "envit5"
    assert out[1]["value"] is False
    assert out[2]["value"] == ""
    assert isinstance(out[3]["value"], str)


def test_preset_force_full_rerun_ticks_every_stage() -> None:
    out = ui.preset_force_all()

    expected = {*pipeline.STAGES, "diarize", "references"}
    assert set(out[0]["value"]) == expected


def test_ollama_ready_units(monkeypatch: pytest.MonkeyPatch) -> None:
    host = "http://127.0.0.1:11434"

    # Exact tag hit.
    monkeypatch.setattr(
        ui,
        "_http_get_json",
        lambda url, timeout: {
            "models": [{"name": "hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K"}]
        },
    )
    ok, message = ui._ollama_ready(host, "hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K")
    assert ok is True
    assert "available" in message

    # Base-name-without-tag hit.
    monkeypatch.setattr(
        ui, "_http_get_json", lambda url, timeout: {"models": [{"name": "qwen2.5:7b"}]}
    )
    ok, _ = ui._ollama_ready(host, "qwen2.5")
    assert ok is True

    # Model missing -> names the model.
    monkeypatch.setattr(
        ui, "_http_get_json", lambda url, timeout: {"models": [{"name": "other:1"}]}
    )
    ok, message = ui._ollama_ready(host, "wanted:9b")
    assert ok is False
    assert "wanted:9b" in message

    # Connection error -> the serve fix.
    def boom(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(ui, "_http_get_json", boom)
    ok, message = ui._ollama_ready(host, "wanted:9b")
    assert ok is False
    assert "ollama serve" in message


def _run_form(monkeypatch, tmp_path, **overrides):
    """Call ``_handle_run`` with a real temp video and a recording pipeline."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")  # so video_hash() can stat it
    started: list = []

    def fake_run_pipeline(video_arg, cfg, **kwargs):
        started.append(True)
        return {}

    monkeypatch.setattr(ui, "run_pipeline", fake_run_pipeline)
    state = ui.new_state()
    base = {
        "mode": "dub",
        "sub_position": DEFAULT_SUB_POSITION,
        "sub_font_size": DEFAULT_SUB_FONT_SIZE,
        "burn_subtitles": False,
        "asr_model": DEFAULT_ASR_MODEL,
        "tts_engine": "piper",
        "allow_nc_models": False,
        "diarize": False,
        "clone_speakers": False,
        "glossary_file": None,
        "loudness_target": None,
        "loudness_tp": None,
        "loudness_lra": None,
        "cache_dir": str(tmp_path),
    }
    base.update(overrides)
    ui._handle_run(state, str(video), "", **base)
    return state, started


def test_preflight_blocks_hunyuan_when_ollama_down(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ui, "_ollama_ready", lambda host, model: (False, "no server"))
    state, started = _run_form(monkeypatch, tmp_path, mt_engine="hunyuan")

    assert state["status"] == "error"
    assert "cannot start" in state["message"]
    assert state["thread"] is None
    assert started == []


def test_preflight_blocks_refine_when_ollama_down(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ui, "_ollama_ready", lambda host, model: (False, "no server"))
    state, started = _run_form(
        monkeypatch, tmp_path, mt_engine="envit5", hunyuan_refine=True
    )

    assert state["status"] == "error"
    assert state["thread"] is None
    assert started == []


def test_preflight_skips_probe_for_envit5_without_refine(monkeypatch, tmp_path) -> None:
    def explode(host, model):
        raise AssertionError("must not probe for envit5 + refine off")

    monkeypatch.setattr(ui, "_ollama_ready", explode)
    state, started = _run_form(
        monkeypatch, tmp_path, mt_engine="envit5", hunyuan_refine=False
    )

    state["thread"].join()
    assert started == [True]
    assert state["status"] == "done"


def test_form_to_config_maps_m3_fields(tmp_path: Path) -> None:
    pron = tmp_path / "p.tsv"

    cfg = ui.form_to_config(
        src_lang="ja",
        tgt_lang="vi",
        allow_lang_mismatch=True,
        pronounce_path=str(pron),
        tail_pad=3.5,
    )

    assert cfg.src_lang == "ja"
    assert cfg.tgt_lang == "vi"
    assert cfg.allow_lang_mismatch is True
    assert cfg.pronounce_path == str(pron)
    assert cfg.tail_pad == 3.5


def test_form_to_config_rejects_unsupported_lang() -> None:
    with pytest.raises(ValueError):
        ui.form_to_config(tgt_lang="fr", tts_engine="piper")


# ------------------------------------------------------ event-layer hardening
class _FakeThread:
    """Minimal stand-in for ``threading.Thread`` (only ``is_alive`` is used)."""

    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def test_format_elapsed_contract() -> None:
    """Elapsed rendering is ``M:SS`` below an hour, ``H:MM:SS`` above."""
    assert ui._format_elapsed(7) == "0:07"
    assert ui._format_elapsed(192) == "3:12"
    assert ui._format_elapsed(3599) == "59:59"
    assert ui._format_elapsed(3723) == "1:02:03"


def test_status_text_running_shows_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ui.time, "monotonic", lambda: 1000.0)
    state = ui.new_state()
    state.update(
        status="running",
        started_at=1000.0 - 192,  # 3m 12s ago
        expected_stages=["translate"],
        stages={},
    )

    text = ui._status_text(state)

    assert "**Running** — stage `translate` (elapsed 3:12)" in text


def test_status_text_running_elapsed_for_active_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ui.time, "monotonic", lambda: 1000.0)
    state = ui.new_state()
    state.update(status="running", started_at=1000.0 - 7, stage="download")

    assert "(elapsed 0:07)" in ui._status_text(state)


# --------------------------------------------------------- never-raise guards
def test_safe_returns_fallback_and_never_raises() -> None:
    def boom() -> tuple:
        raise RuntimeError("unexpected")

    assert ui._safe(boom, fallback=lambda: ("degraded",))() == ("degraded",)
    assert ui._safe(boom)() is None  # no fallback -> None, still no raise

    def bad_fallback() -> tuple:
        raise RuntimeError("fallback also fails")

    assert ui._safe(boom, fallback=bad_fallback)() is None


def test_safe_passes_arguments_through() -> None:
    assert ui._safe(lambda a, b: a + b)(2, 3) == 5


def test_poll_survives_collect_error(monkeypatch: pytest.MonkeyPatch) -> None:
    state = ui.new_state()
    state.update(status="running", manifest_path="x/job.json")

    def boom(_state) -> None:
        raise RuntimeError("collect exploded")

    monkeypatch.setattr(ui, "_collect", boom)

    out = ui.poll(state)

    assert len(out) == 8
    assert isinstance(out[0], str)


def test_poll_survives_qc_summary_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = JobManifest(
        cache_dir=str(tmp_path), video_hash="h1", stages={"transcribe": "done"}
    )
    mpath = manifest.save(tmp_path / "job.json")
    state = ui.new_state()
    state.update(status="running", manifest_path=str(mpath))

    def boom(*_args, **_kwargs) -> str:
        raise RuntimeError("qc summary exploded")

    monkeypatch.setattr(ui, "_qc_summary", boom)

    out = ui.poll(state)

    assert len(out) == 8
    assert isinstance(out[0], str)


def test_handle_run_bad_input_still_reports_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberate ValueError validation must still reach the UI as an error."""
    monkeypatch.setattr(ui, "_ollama_ready", lambda host, model: (True, "ok"))
    state = ui.new_state()

    out = ui._handle_run(
        state,
        "",
        "",
        "dub",
        DEFAULT_SUB_POSITION,
        DEFAULT_SUB_FONT_SIZE,
        False,
        DEFAULT_ASR_MODEL,
        "piper",
        False,
        False,
        False,
        None,
        None,
        None,
        None,
        None,
    )

    assert state["status"] == "error"
    assert state["message"].startswith("error:")
    assert len(out) == 8


# --------------------------------------------------- live-thread run gating
def test_handle_run_refuses_while_thread_alive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = ui.new_state()
    alive = _FakeThread(True)
    state.update(status="running", thread=alive, started_at=time.monotonic() - 192)
    called: list = []
    monkeypatch.setattr(ui, "run_pipeline", lambda *a, **k: called.append(1) or {})

    out = ui._handle_run(
        state,
        str(tmp_path / "video.mp4"),
        "",
        "dub",
        DEFAULT_SUB_POSITION,
        DEFAULT_SUB_FONT_SIZE,
        False,
        DEFAULT_ASR_MODEL,
        "piper",
        False,
        False,
        False,
        None,
        None,
        None,
        None,
        str(tmp_path),
    )

    assert "already in progress" in state["message"]
    assert "elapsed" in state["message"]
    assert state["thread"] is alive  # untouched while it is alive
    assert called == []  # no worker was started
    assert len(out) == 8


def test_handle_run_proceeds_when_thread_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")  # so video_hash() can stat it
    state = ui.new_state()
    stale = _FakeThread(False)
    state["thread"] = stale
    started: list = []
    monkeypatch.setattr(ui, "run_pipeline", lambda *a, **k: started.append(1) or {})

    ui._handle_run(
        state,
        str(video),
        "",
        "dub",
        DEFAULT_SUB_POSITION,
        DEFAULT_SUB_FONT_SIZE,
        False,
        DEFAULT_ASR_MODEL,
        "piper",
        False,
        False,
        False,
        None,
        None,
        None,
        None,
        str(tmp_path),
    )

    assert state["thread"] is not stale  # stale reference cleared + replaced
    state["thread"].join()
    assert started == [1]
    assert state["status"] == "done"
    assert state["started_at"] is None  # cleared after the run ends


# --------------------------------------------------- socket-reset noise filter
def test_noise_filter_swallows_reset_but_reraises_other(
    capsys: pytest.CaptureFixture[str],
) -> None:
    if sys.platform != "win32":
        pytest.skip("proactor transport is Windows-only")

    def reset(self, exc):
        raise ConnectionResetError(10054, "forcibly closed")

    quiet = ui._silence_connection_reset(reset)
    quiet(object(), None)  # must not raise ...
    assert capsys.readouterr().err == ""  # ... and print nothing

    def boom(self, exc):
        raise RuntimeError("real failure")

    loud = ui._silence_connection_reset(boom)
    with pytest.raises(RuntimeError):
        loud(object(), None)


def test_install_socket_noise_filter_patches_once() -> None:
    if sys.platform != "win32":
        pytest.skip("proactor transport is Windows-only")

    from asyncio.proactor_events import _ProactorBasePipeTransport

    original = _ProactorBasePipeTransport._call_connection_lost
    try:
        assert ui._install_socket_noise_filter() is True
        patched = _ProactorBasePipeTransport._call_connection_lost
        assert getattr(patched, "_vidub_quiet", False)
        assert patched is not original

        # Idempotent: a second install must not wrap the wrapper again.
        assert ui._install_socket_noise_filter() is True
        assert _ProactorBasePipeTransport._call_connection_lost is patched
    finally:
        _ProactorBasePipeTransport._call_connection_lost = original


# --------------------------------------------------------------- server smoke
def test_ui_server_smoke(tmp_path: Path) -> None:
    """Launch ``vidub ui`` in a subprocess and GET the page (skip-safe)."""
    port = 7899
    proc = subprocess.Popen(
        [sys.executable, "-m", "vidub.cli", "ui", "--port", str(port)],
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    reachable = False
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/", timeout=2
                ) as response:
                    if response.status == 200:
                        reachable = True
                        break
            except Exception:  # noqa: BLE001 - not up yet
                time.sleep(0.5)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - force-kill a stuck server
            proc.kill()

    if not reachable:
        output = ""
        if proc.stdout is not None:
            output = proc.stdout.read()[-1000:]
        pytest.skip(f"UI server not reachable (port/network?): {output}")


# ------------------------------------------------ per-unit segment player (UA-1)
def test_segment_map_maps_segments_to_unit_audio(tmp_path: Path) -> None:
    artifacts = tmp_path / "art"
    speech_dir = artifacts / "speech"
    speech_dir.mkdir(parents=True)
    (speech_dir / "unit_0000.wav").write_bytes(b"RIFF")
    (artifacts / "speech.json").write_text(
        json.dumps(
            [
                {"idx": 0, "start": 0.0, "end": 1.0, "speaker": "spk0", "vi": "a", "file": "speech/seg_0000.wav"},
                {"idx": 1, "start": 1.0, "end": 2.0, "speaker": "spk0", "vi": "b", "file": "speech/seg_0001.wav"},
            ]
        ),
        encoding="utf-8",
    )
    (artifacts / "timed.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "members": [0, 1],
                    "file": "speech/unit_0000.wav",
                    "start": 0.0,
                    "end": 2.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = JobManifest(
        cache_dir=str(tmp_path),
        video_hash="h",
        paths={"speech": "speech.json", "timed": "timed.json"},
    )

    files, choices = ui._segment_map(artifacts, manifest)

    # every member segment is listed, all pointing at its unit's audio file
    assert choices == ["0 · spk0 · 0.0s", "1 · spk0 · 1.0s"]
    assert set(files.values()) == {str(speech_dir / "unit_0000.wav")}


def test_segment_map_falls_back_without_members(tmp_path: Path) -> None:
    artifacts = tmp_path / "art"
    speech_dir = artifacts / "speech"
    speech_dir.mkdir(parents=True)
    (speech_dir / "seg_0000.wav").write_bytes(b"RIFF")
    (artifacts / "speech.json").write_text(
        json.dumps(
            [
                {"idx": 0, "start": 0.0, "end": 1.0, "speaker": "spk0", "vi": "a", "file": "speech/seg_0000.wav"},
            ]
        ),
        encoding="utf-8",
    )
    # opt-out timed.json: no ``members`` -> fall back to the speech.json files
    (artifacts / "timed.json").write_text(
        json.dumps(
            [{"idx": 0, "file": "speech/seg_0000.wav", "start": 0.0, "end": 1.0}]
        ),
        encoding="utf-8",
    )
    manifest = JobManifest(
        cache_dir=str(tmp_path),
        video_hash="h",
        paths={"speech": "speech.json", "timed": "timed.json"},
    )

    files, choices = ui._segment_map(artifacts, manifest)

    assert choices == ["0 · spk0 · 0.0s"]
    assert files["0 · spk0 · 0.0s"] == str(speech_dir / "seg_0000.wav")


# ------------------------------------------- SA-1-T4 aligned spans + slices
def _write_wav(path: Path, seconds: float, rate: int = 8000) -> None:
    """Write a mono 16-bit sine WAV of ``seconds`` seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for i in range(round(seconds * rate)):
        value = int(20000 * math.sin(2.0 * math.pi * 440.0 * i / rate))
        frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(bytes(frames))


def _span_fixture(root: Path) -> tuple[Path, JobManifest]:
    """A minimal SA-1 job: script.json + speech.json + timed.json (member_spans)."""
    artifacts = root / "art"
    artifacts.mkdir(parents=True)
    (artifacts / "script.json").write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 0.5, "speaker": "spk0", "en": "Hello", "vi": "Xin chào"},
                {"start": 0.5, "end": 1.0, "speaker": "spk0", "en": "world.", "vi": "thế giới."},
                {"start": 1.0, "end": 2.0, "speaker": "spk0", "en": "Next", "vi": "Tiếp"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifacts / "speech.json").write_text(
        json.dumps(
            [
                {
                    "idx": idx, "start": float(idx), "end": float(idx) + 1.0,
                    "speaker": "spk0", "vi": f"v{idx}",
                    "file": f"speech/seg_{idx:04d}.wav",
                }
                for idx in range(3)
            ]
        ),
        encoding="utf-8",
    )
    _write_wav(artifacts / "speech" / "unit_0000.wav", 2.0)
    (artifacts / "timed.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0, "members": [0, 1], "file": "speech/unit_0000.wav",
                    "start": 0.0, "end": 2.0, "align": "split",
                    "member_spans": [
                        [0, 0.2, 0.8, "Xin chào"],
                        [1, 0.8, 1.6, "thế giới."],
                    ],
                },
                {
                    "idx": 1, "members": [2], "file": "speech/unit_0000.wav",
                    "start": 0.0, "end": 2.0, "align": "split",
                    "member_spans": [[2, 1.6, 2.0, "Tiếp"]],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = JobManifest(
        cache_dir=str(root),
        video_hash="h",
        paths={"script": "script.json", "speech": "speech.json", "timed": "timed.json"},
    )
    return artifacts, manifest


def test_aligned_spans_and_script_columns(tmp_path: Path) -> None:
    artifacts, manifest = _span_fixture(tmp_path)

    spans = ui._aligned_spans(artifacts, manifest)
    assert set(spans) == {0, 1, 2}
    assert spans[0]["phrase"] == "Xin chào"
    assert (spans[0]["start"], spans[0]["end"]) == (0.2, 0.8)

    rows = ui._read_script_rows(artifacts, spans)
    assert len(rows) == 3
    assert all(len(row) == 8 for row in rows)  # 5 original + 3 new columns
    # start, end, speaker, en, vi, unit, aligned, vi_aligned
    assert rows[0] == [
        0.0, 0.5, "spk0", "Hello", "Xin chào", "unit_0000.wav", "0.2–0.8s", "Xin chào",
    ]
    assert rows[2][5:] == ["unit_0000.wav", "1.6–2.0s", "Tiếp"]


def test_read_script_rows_legacy_columns(tmp_path: Path) -> None:
    artifacts, _manifest = _span_fixture(tmp_path)

    rows = ui._read_script_rows(artifacts)  # no spans -> legacy values

    assert all(len(row) == 8 for row in rows)
    assert rows[0] == [0.0, 0.5, "spk0", "Hello", "Xin chào", "", "", "Xin chào"]


def test_read_text_selection_rows_uses_sentence_units(tmp_path: Path) -> None:
    artifacts = tmp_path / "art"
    artifacts.mkdir()
    (artifacts / "units.json").write_text(
        json.dumps(
            {
                "version": 1,
                "units": [
                    {
                        "index": 0,
                        "indices": [0, 1],
                        "start": 0.0,
                        "end": 2.0,
                        "source": "Hello there.",
                        "vi": "Xin chào.",
                        "spoken_vi": "Chào bạn.",
                        "subtitle_vi": "Xin chào.",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert ui._read_text_selection_rows(artifacts) == [
        [0, 0.0, 2.0, "Hello there.", "Xin chào.", "Chào bạn.", "Xin chào."]
    ]


def test_legacy_text_selection_rows_fall_back_to_vi_and_save_to_script(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "art"
    artifacts.mkdir()
    path = artifacts / "script.json"
    path.write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "start": 1.0,
                    "end": 2.0,
                    "en": "Hello.",
                    "vi": "Xin chào.",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rows = ui._read_text_selection_rows(artifacts)
    assert rows == [[0, 1.0, 2.0, "Hello.", "Xin chào.", "Xin chào.", "Xin chào."]]
    rows[0][5] = "Chào bạn."

    ui._write_text_selection_rows(artifacts, rows)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[0]["spoken_vi"] == "Chào bạn."
    assert saved[0]["subtitle_vi"] == "Xin chào."


def test_save_text_selection_rows_updates_only_selected_fields(tmp_path: Path) -> None:
    artifacts = tmp_path / "art"
    artifacts.mkdir()
    units = {
        "version": 1,
        "units": [
            {
                "index": 0,
                "indices": [0],
                "start": 0.0,
                "end": 1.0,
                "source": "Hello.",
                "vi": "Xin chào.",
                "spoken_vi": "Xin chào.",
                "subtitle_vi": "Xin chào.",
                "battery": "ok",
            }
        ],
    }
    path = artifacts / "units.json"
    path.write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")
    rows = ui._read_text_selection_rows(artifacts)
    rows[0][5] = "Chào bạn."
    rows[0][6] = "Xin chào!"

    ui._write_text_selection_rows(artifacts, rows)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["units"][0]["spoken_vi"] == "Chào bạn."
    assert saved["units"][0]["subtitle_vi"] == "Xin chào!"
    assert saved["units"][0]["vi"] == "Xin chào."
    assert saved["units"][0]["battery"] == "ok"


def test_save_text_selection_rows_rejects_reordered_units(tmp_path: Path) -> None:
    artifacts = tmp_path / "art"
    artifacts.mkdir()
    units = {
        "units": [
            {"index": 0, "start": 0.0, "end": 1.0, "source": "A", "vi": "A"},
            {"index": 1, "start": 1.0, "end": 2.0, "source": "B", "vi": "B"},
        ]
    }
    path = artifacts / "units.json"
    path.write_text(json.dumps(units), encoding="utf-8")
    rows = ui._read_text_selection_rows(artifacts)

    with pytest.raises(ValueError, match="order"):
        ui._write_text_selection_rows(artifacts, rows[::-1])

    assert json.loads(path.read_text(encoding="utf-8")) == units


def test_save_text_selection_ui_requires_a_finished_run(tmp_path: Path) -> None:
    assert "Run a video first" in ui._save_text_selection_rows({}, [])
    assert "Wait for the current run" in ui._save_text_selection_rows(
        {"status": "running", "artifacts_dir": str(tmp_path)}, []
    )


def test_save_text_selection_ui_prompts_to_rerun(tmp_path: Path) -> None:
    artifacts = tmp_path / "art"
    artifacts.mkdir()
    path = artifacts / "script.json"
    path.write_text(
        json.dumps([{"idx": 0, "vi": "Xin chào."}], ensure_ascii=False),
        encoding="utf-8",
    )
    state = {"status": "done", "artifacts_dir": str(artifacts)}
    rows = ui._read_text_selection_rows(artifacts)
    rows[0][5] = "Chào bạn."

    message = ui._save_text_selection_rows(state, rows)

    assert "Click **Run**" in message
    assert json.loads(path.read_text(encoding="utf-8"))[0]["spoken_vi"] == "Chào bạn."


def test_collect_exposes_segment_spans(tmp_path: Path) -> None:
    artifacts, manifest = _span_fixture(tmp_path)
    mpath = manifest.save(artifacts / "job.json")
    state = ui.new_state()
    state["manifest_path"] = str(mpath)

    ui._collect(state)

    assert set(state["segment_spans"]) == {0, 1, 2}
    assert all(len(row) == 8 for row in state["script_rows"])


def test_segment_map_labels_show_the_aligned_range(tmp_path: Path) -> None:
    artifacts, manifest = _span_fixture(tmp_path)
    spans = ui._aligned_spans(artifacts, manifest)

    _files, choices = ui._segment_map(artifacts, manifest, spans)

    assert choices == [
        "0 · spk0 · 0.0s · 0.2–0.8s",
        "1 · spk0 · 1.0s · 0.8–1.6s",
        "2 · spk0 · 2.0s · 1.6–2.0s",
    ]


def _label_for(state: dict, index: int) -> str:
    return next(
        name for name in state["segment_files"] if name.split(" · ")[0] == str(index)
    )


def test_segment_audio_slices_the_aligned_span(tmp_path: Path) -> None:
    artifacts, manifest = _span_fixture(tmp_path)
    spans = ui._aligned_spans(artifacts, manifest)
    state = ui.new_state()
    state["segment_spans"] = spans
    state["segment_files"], _choices = ui._segment_map(artifacts, manifest, spans)
    label = _label_for(state, 0)

    out = ui.segment_audio(state, label)

    assert out is not None
    assert Path(out).parent == Path(tempfile.gettempdir())  # never the artifacts dir
    with wave.open(out, "rb") as sliced:
        assert sliced.getnchannels() == 1
        assert sliced.getframerate() == 8000
        duration = sliced.getnframes() / sliced.getframerate()
    assert abs(duration - 0.6) <= 1.0 / 8000 + 1e-9  # span 0.2..0.8, within a frame

    source = artifacts / "speech" / "unit_0000.wav"
    with wave.open(str(source), "rb") as src, wave.open(out, "rb") as sliced:
        src.setpos(round(0.2 * 8000))
        assert sliced.readframes(sliced.getnframes()) == src.readframes(
            sliced.getnframes()
        )

    # repeated clicks reuse the cached slice (no new file)
    assert ui.segment_audio(state, label) == out


def test_segment_audio_falls_back_when_audio_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"
    state = ui.new_state()
    state["segment_files"] = {"0 · spk0 · 0.0s": str(missing)}
    state["segment_spans"] = {
        0: {"file": str(missing), "start": 0.2, "end": 0.8, "phrase": "x", "align": "split"}
    }

    # never raises; returns the original full path
    assert ui.segment_audio(state, "0 · spk0 · 0.0s") == str(missing)


def test_segment_audio_out_of_range_span_returns_full_file(tmp_path: Path) -> None:
    artifacts, _manifest = _span_fixture(tmp_path)
    full = str(artifacts / "speech" / "unit_0000.wav")
    state = ui.new_state()
    state["segment_files"] = {"0 · spk0 · 0.0s": full}
    state["segment_spans"] = {
        0: {"file": full, "start": 5.0, "end": 6.0, "phrase": "x", "align": "split"}
    }

    assert ui.segment_audio(state, "0 · spk0 · 0.0s") == full


def test_segment_audio_legacy_returns_the_full_file(tmp_path: Path) -> None:
    artifacts, _manifest = _span_fixture(tmp_path)
    full = str(artifacts / "speech" / "unit_0000.wav")
    state = ui.new_state()
    state["segment_files"] = {"2 · spk0 · 2.0s": full}
    state["segment_spans"] = {}  # legacy: no member_spans

    assert ui.segment_audio(state, "2 · spk0 · 2.0s") == full
    assert ui.segment_audio(state, "") is None
    assert ui.segment_audio(state, None) is None


def test_qc_summary_includes_sentence_units_block(tmp_path: Path) -> None:
    (tmp_path / "translation_qc.json").write_text(
        json.dumps(
            {
                "verdict": "review",
                "counts": {"ok": 3},
                "units": {
                    "flags": {
                        "guard_flagged": 2,
                        "omission": 1,
                        "sentence_loss": 1,
                        "addition": 0,
                        "oversized": 4,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "timing.json").write_text(
        json.dumps({"stretched": 0, "align": {"asr": 3, "split": 1, "single": 0, "unavailable": 2}}),
        encoding="utf-8",
    )

    text = ui._qc_summary(tmp_path, None)

    assert "**Sentence units**" in text
    assert "guard: guard_flagged=2, omission=1, sentence_loss=1, oversized=4" in text
    assert "align: asr=3, split=1, single=0, unavailable=2" in text


def test_qc_summary_omits_sentence_units_block_for_old_artifacts(tmp_path: Path) -> None:
    (tmp_path / "translation_qc.json").write_text(
        json.dumps({"verdict": "pass", "counts": {"ok": 3}}), encoding="utf-8"
    )
    (tmp_path / "timing.json").write_text(
        json.dumps({"stretched": 1}), encoding="utf-8"
    )

    text = ui._qc_summary(tmp_path, None)

    assert "**Sentence units**" not in text

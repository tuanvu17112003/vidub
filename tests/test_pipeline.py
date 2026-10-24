"""Tests for the full pipeline orchestration + resume (M1-T6).

The integration test is marked ``slow``: it runs the real tiny-whisper / envit5 /
piper / demucs models over ``samples/demo_en.mp4``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import vidub.pipeline as pipeline
from vidub.cli import _build_cfg, build_parser
from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.stages.export import build_report
from vidub.stages.ingest import video_hash
from vidub.utils.proc import find_tool, run

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "samples" / "demo_en.mp4"


def _by_type(path: Path) -> dict[str, dict]:
    ffprobe = find_tool("ffprobe")
    assert ffprobe is not None
    proc = run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name:stream_tags=language",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return {stream["codec_type"]: stream for stream in json.loads(proc.stdout)["streams"]}


def test_required_stages_by_mode() -> None:
    assert pipeline.required_stages("dub") == [
        "ingest", "transcribe", "translate", "speak", "timing", "mix", "export",
    ]
    assert pipeline.required_stages("voice-over") == pipeline.required_stages("dub")
    assert pipeline.required_stages("subtitles-only") == [
        "ingest", "transcribe", "translate", "export",
    ]


def test_report_marks_mode_skipped_stages(tmp_path: Path) -> None:
    manifest = JobManifest(
        input_path="x/demo_en.mp4",
        video_hash="h",
        cache_dir=str(tmp_path),
        mode="subtitles-only",
        stages={"ingest": "done", "transcribe": "done", "translate": "done", "export": "done"},
    )

    report = build_report(manifest)

    assert report["mode"] == "subtitles-only"
    assert report["stages"]["export"]["status"] == "done"
    for stage in ("speak", "timing", "mix"):
        assert report["stages"][stage]["status"] == "skipped"
    # No synthesis runs, so the report must say so instead of naming an engine.
    assert report["engines"]["tts"] == "skipped (subtitles-only)"


def test_subtitles_alias_same_required_stages_and_report(tmp_path: Path) -> None:
    """``subtitles`` and ``subtitles-only`` are one canonical mode (SU-1-T1)."""
    alias_cfg = JobConfig(mode="subtitles")
    canonical_cfg = JobConfig(mode="subtitles-only")
    assert alias_cfg.mode == canonical_cfg.mode == "subtitles-only"
    expected = ["ingest", "transcribe", "translate", "export"]
    assert pipeline.required_stages(alias_cfg.mode) == expected
    assert pipeline.required_stages("subtitles") == pipeline.required_stages(
        "subtitles-only"
    ) == expected

    stages = {
        "ingest": "done",
        "transcribe": "done",
        "translate": "done",
        "export": "done",
    }
    common = dict(
        input_path="x/demo_en.mp4",
        video_hash="h",
        cache_dir=str(tmp_path),
        stages=stages,
    )
    alias_report = build_report(JobManifest(mode=alias_cfg.mode, **common))
    canonical_report = build_report(JobManifest(mode=canonical_cfg.mode, **common))
    assert alias_report == canonical_report


def test_cli_subtitles_alias_is_canonical_mode() -> None:
    parser = build_parser()
    cfg = _build_cfg(parser.parse_args(["run", "x.mp4", "--mode", "subtitles"]))
    assert cfg.mode == "subtitles-only"
    assert pipeline.required_stages(cfg.mode) == [
        "ingest", "transcribe", "translate", "export",
    ]


def test_subtitles_only_skips_tts_validation(monkeypatch, tmp_path) -> None:
    """An unusable TTS config must not block a subtitles-only run (SU-1-T1)."""
    called: list[str] = []
    cache_dir = _force_test_manifest(tmp_path)
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)
    cfg = JobConfig(
        cache_dir=str(cache_dir),
        mode="subtitles-only",
        tts_engine="vixtts",  # NC weights, no --allow-nc-models: invalid for dub
        allow_nc_models=False,
    )

    pipeline.run_pipeline("video.mp4", cfg, resume=True)

    assert called == ["export"]  # speak/timing/mix never run
    saved = JobManifest.load(cache_dir / "hash1" / "job.json")
    assert saved.mode == "subtitles-only"  # canonical on disk


def test_dub_still_rejects_unusable_tts_config(tmp_path) -> None:
    """The dub/voice-over TTS guard is unchanged (SU-1-T1)."""
    cfg = JobConfig(
        cache_dir=str(tmp_path / "cache"),
        mode="dub",
        tts_engine="vixtts",
        allow_nc_models=False,
    )
    with pytest.raises(ValueError):
        pipeline.run_pipeline("video.mp4", cfg, resume=False)


def test_subtitles_only_still_validates_mt_engine(monkeypatch, tmp_path) -> None:
    """MT validation still applies in subtitles-only (translation still runs)."""
    called: list[str] = []
    cache_dir = _force_test_manifest(tmp_path, mt_signature="stale-signature")
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)
    cfg = JobConfig(
        cache_dir=str(cache_dir), mode="subtitles-only", mt_engine="bogus"
    )

    with pytest.raises(ValueError) as excinfo:
        pipeline.run_pipeline("video.mp4", cfg, resume=True)

    assert "bogus" in str(excinfo.value)
    assert called == []


def test_subtitles_only_clone_speakers_is_explicit_error(tmp_path) -> None:
    """--clone-speakers cannot apply here; it must fail loudly (SU-1-T1)."""
    cfg = JobConfig(
        cache_dir=str(tmp_path / "cache"),
        mode="subtitles-only",
        clone_speakers=True,
    )

    with pytest.raises(ValueError) as excinfo:
        pipeline.run_pipeline("video.mp4", cfg, resume=False)

    message = str(excinfo.value)
    assert "--clone-speakers" in message
    assert "subtitles-only" in message


@pytest.mark.slow
def test_full_pipeline_and_resume(tmp_path: Path) -> None:
    """End-to-end ``vidub run`` (real models) + a fast ``--resume`` second run.

    The pipeline loads CTranslate2 *and* torch in one process, so it runs as a
    subprocess: if the native stack ever hangs, the 900 s timeout fails just
    this test instead of wedging the whole pytest session.
    """
    if not DEMO.exists():
        pytest.skip("samples/demo_en.mp4 is not available")
    if not find_tool("ffmpeg"):
        pytest.skip("ffmpeg not available")

    cache = tmp_path / ".vidub-cache"
    out_dir = tmp_path / "output" / DEMO.stem

    def run_cli(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vidub.cli", *args],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )

    first = run_cli(
        "run", str(DEMO), "--asr-model", "tiny", "--cache-dir", str(cache)
    )
    assert first.returncode == 0, first.stderr

    dubbed = out_dir / "dubbed.vi.mp4"
    assert dubbed.exists()
    assert (out_dir / "report.json").exists()
    assert (out_dir / "subtitles.vi.srt").exists()
    assert len((out_dir / "script.tsv").read_text(encoding="utf-8").splitlines()) == 5

    streams = _by_type(dubbed)
    assert streams["audio"]["codec_name"] == "aac"
    assert streams["subtitle"]["codec_name"] == "mov_text"
    assert streams["subtitle"].get("tags", {}).get("language") in {"vi", "vie"}

    report1 = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    for stage in pipeline.STAGES:
        assert report1["stages"][stage]["status"] == "done"

    # Fingerprint the artifacts each finished stage produced.
    artifacts = cache / video_hash(str(DEMO))
    tracked = ["segments.json", "script.json", "timed.json", "dub_track.wav"]
    mtimes = {name: (artifacts / name).stat().st_mtime_ns for name in tracked}

    # A second run with --resume must not re-run any finished stage.
    second = run_cli(
        "run",
        str(DEMO),
        "--resume",
        "--asr-model",
        "tiny",
        "--cache-dir",
        str(cache),
    )
    assert second.returncode == 0, second.stderr

    report2 = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report2["stages"] == report1["stages"]
    for name, mtime in mtimes.items():
        assert (artifacts / name).stat().st_mtime_ns == mtime, (
            f"{name} was re-created on --resume"
        )


# --------------------------------------------------------------------- --force

def _force_test_manifest(
    tmp_path: Path,
    mt_signature: str = "",
    tts_signature: str = "",
    tts_engine: str = "piper",
) -> Path:
    """Write an all-done manifest and return its cache dir.

    ``mt_signature`` defaults to ``""`` (the pre-change ``job.json`` shape:
    unknown signature, so resume is never invalidated).
    """
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "hash1"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    JobManifest(
        input_path="demo.mp4",
        video_hash="hash1",
        cache_dir=str(cache_dir),
        mode="dub",
        tts_engine=tts_engine,
        stages={name: "done" for name in pipeline.STAGES},
        mt_signature=mt_signature,
        tts_signature=tts_signature,
    ).save(artifacts_dir / "job.json")
    return cache_dir


def _stub_stages(monkeypatch, cache_dir: Path, tmp_path: Path, called: list[str]) -> None:
    """Replace every stage with a recorder so run_pipeline can be driven cheaply."""
    artifacts_dir = cache_dir / "hash1"

    monkeypatch.setattr(
        pipeline, "stage_artifacts_present", lambda stage, manifest: True
    )
    monkeypatch.setattr(pipeline.ingest_mod, "video_hash", lambda video: "hash1")

    def make_stage(name: str):
        def _stage(manifest_path, cfg, **kwargs):
            called.append(name)
            loaded = JobManifest.load(manifest_path)
            loaded.stages[name] = "done"
            loaded.save(manifest_path)
            return loaded

        return _stage

    def fake_ingest(video, cfg):
        called.append("ingest")
        return JobManifest.load(artifacts_dir / "job.json")

    monkeypatch.setattr(pipeline.ingest_mod, "ingest", fake_ingest)
    monkeypatch.setattr(pipeline.transcribe_mod, "transcribe", make_stage("transcribe"))
    monkeypatch.setattr(pipeline.translate_mod, "translate", make_stage("translate"))
    monkeypatch.setattr(pipeline.speak_mod, "speak", make_stage("speak"))
    monkeypatch.setattr(pipeline.timing_mod, "fit", make_stage("timing"))
    monkeypatch.setattr(pipeline.mix_mod, "mix", make_stage("mix"))
    monkeypatch.setattr(pipeline.export_mod, "export", make_stage("export"))
    monkeypatch.setattr(
        pipeline.export_mod, "output_dir_for", lambda manifest: tmp_path / "out"
    )


def test_first_forced_index() -> None:
    stages = ["ingest", "transcribe", "translate", "speak", "timing", "mix", "export"]
    assert pipeline.first_forced_index(stages, set()) is None
    assert pipeline.first_forced_index(stages, {"export"}) == 6
    assert pipeline.first_forced_index(stages, {"translate"}) == 2
    # the earliest forced stage wins, even when several are named
    assert pipeline.first_forced_index(stages, {"mix", "transcribe"}) == 1


# --------------------------------------------------------- final QC gate (M4-T4)
def _cli_quality_report(tmp_path: Path, quality: dict) -> dict:
    """A report shaped like ``run_pipeline`` output, with a dubbed file present."""
    outputs = tmp_path / "output" / "demo"
    outputs.mkdir(parents=True, exist_ok=True)
    dubbed = outputs / "dubbed.vi.mp4"
    dubbed.write_bytes(b"x")
    return {
        "stages": {name: {"status": "done"} for name in pipeline.STAGES},
        "outputs": {"dubbed": str(dubbed)},
        "quality": quality,
    }


def test_cmd_run_exits_nonzero_on_quality_fail(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    import vidub.cli as cli

    report = _cli_quality_report(
        tmp_path, {"verdict": "fail", "reasons": ["transcript QC verdict is 'fail'"]}
    )
    monkeypatch.setattr("vidub.pipeline.run_pipeline", lambda *a, **k: report)

    video = tmp_path / "demo.mp4"
    video.write_bytes(b"\x00")
    rc = cli.main(["run", str(video), "--cache-dir", str(tmp_path / "cache")])

    assert rc == 1
    assert "quality check failed" in capsys.readouterr().err


def test_cmd_run_exits_zero_on_quality_review(
    monkeypatch, tmp_path: Path
) -> None:
    import vidub.cli as cli

    report = _cli_quality_report(tmp_path, {"verdict": "review", "reasons": ["x"]})
    monkeypatch.setattr("vidub.pipeline.run_pipeline", lambda *a, **k: report)

    video = tmp_path / "demo.mp4"
    video.write_bytes(b"\x00")
    rc = cli.main(["run", str(video), "--cache-dir", str(tmp_path / "cache")])

    assert rc == 0


def test_force_transcribe_reruns_it_and_all_downstream(monkeypatch, tmp_path) -> None:
    called: list[str] = []
    cache_dir = _force_test_manifest(tmp_path)
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline(
        "video.mp4",
        JobConfig(cache_dir=str(cache_dir)),
        resume=True,
        force=["transcribe"],
    )

    assert called == ["transcribe", "translate", "speak", "timing", "mix", "export"]


def test_force_export_reruns_only_export(monkeypatch, tmp_path) -> None:
    called: list[str] = []
    cache_dir = _force_test_manifest(tmp_path)
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline(
        "video.mp4",
        JobConfig(cache_dir=str(cache_dir)),
        resume=True,
        force=["export"],
    )

    assert called == ["export"]


def test_resume_without_force_still_skips_everything(monkeypatch, tmp_path) -> None:
    called: list[str] = []
    cache_dir = _force_test_manifest(tmp_path)
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline(
        "video.mp4", JobConfig(cache_dir=str(cache_dir)), resume=True
    )

    assert called == []


# ------------------------------------------- MT-configuration invalidation (UI-1-T1)

def test_mt_signature_covers_only_in_scope_fields() -> None:
    """The signature is deterministic and covers exactly the MT settings."""
    base = JobConfig()
    # Two equivalent configs are always equal (deterministic serialisation).
    assert pipeline.mt_signature(base) == pipeline.mt_signature(JobConfig())
    # Normalisation: engine case/whitespace does not matter.
    assert pipeline.mt_signature(
        JobConfig(mt_engine="  HUNYUAN ")
    ) == pipeline.mt_signature(JobConfig(mt_engine="hunyuan"))
    # Every in-scope field changes the signature.
    assert pipeline.mt_signature(JobConfig(mt_engine="hunyuan")) != pipeline.mt_signature(
        base
    )
    assert pipeline.mt_signature(
        JobConfig(hunyuan_refine=True)
    ) != pipeline.mt_signature(base)
    assert pipeline.mt_signature(
        JobConfig(hunyuan_samples=7)
    ) != pipeline.mt_signature(base)
    assert pipeline.mt_signature(
        JobConfig(fusion_model="org/model:Q4")
    ) != pipeline.mt_signature(base)
    # The sentence-unit translation switch participates (TU-1-T5): toggling it
    # changes what the engine receives and what script.json holds.
    assert pipeline.mt_signature(
        JobConfig(sentence_units=False)
    ) != pipeline.mt_signature(base)
    assert pipeline.mt_signature(
        JobConfig(sentence_units=True)
    ) == pipeline.mt_signature(JobConfig())
    # The active engine's own model/host participate.
    hunyuan = JobConfig(mt_engine="hunyuan")
    assert pipeline.mt_signature(
        JobConfig(mt_engine="hunyuan", hunyuan_model="other")
    ) != pipeline.mt_signature(hunyuan)
    assert pipeline.mt_signature(
        JobConfig(mt_engine="hunyuan", hunyuan_host="http://other:11434")
    ) != pipeline.mt_signature(hunyuan)
    # Same for the ollama active-engine branch.
    ollama = JobConfig(mt_engine="ollama")
    assert pipeline.mt_signature(
        JobConfig(mt_engine="ollama", ollama_model="other")
    ) != pipeline.mt_signature(ollama)
    assert pipeline.mt_signature(
        JobConfig(mt_engine="ollama", ollama_host="http://other:11434")
    ) != pipeline.mt_signature(ollama)
    # An engine that is NOT in use never participates.
    envit5 = JobConfig(mt_engine="envit5")
    assert pipeline.mt_signature(
        JobConfig(mt_engine="envit5", ollama_model="other")
    ) == pipeline.mt_signature(envit5)
    assert pipeline.mt_signature(
        JobConfig(mt_engine="envit5", ollama_host="http://other:11434")
    ) == pipeline.mt_signature(envit5)


def test_matching_mt_signature_still_skips_everything(monkeypatch, tmp_path) -> None:
    called: list[str] = []
    cfg = JobConfig(cache_dir=str(tmp_path / "cache"))
    expected = pipeline.mt_signature(cfg)
    cache_dir = _force_test_manifest(tmp_path, mt_signature=expected)
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline("video.mp4", cfg, resume=True)

    assert called == []
    # A skipped translate must leave the stored signature exactly as recorded.
    assert JobManifest.load(cache_dir / "hash1" / "job.json").mt_signature == expected


def test_mt_engine_change_reruns_translate_and_downstream(
    monkeypatch, tmp_path
) -> None:
    called: list[str] = []
    cache_dir = _force_test_manifest(
        tmp_path, mt_signature=pipeline.mt_signature(JobConfig())
    )
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline(
        "video.mp4",
        JobConfig(cache_dir=str(cache_dir), mt_engine="hunyuan"),
        resume=True,
    )

    assert called == ["translate", "speak", "timing", "mix", "export"]


def test_mt_refine_change_reruns_translate_and_downstream(
    monkeypatch, tmp_path
) -> None:
    called: list[str] = []
    cache_dir = _force_test_manifest(
        tmp_path, mt_signature=pipeline.mt_signature(JobConfig())
    )
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline(
        "video.mp4",
        JobConfig(cache_dir=str(cache_dir), hunyuan_refine=True),
        resume=True,
    )

    assert called == ["translate", "speak", "timing", "mix", "export"]


def test_sentence_units_off_change_reruns_translate_and_downstream(
    monkeypatch, tmp_path
) -> None:
    """Disabling sentence units invalidates a script cached with them on."""
    called: list[str] = []
    cache_dir = _force_test_manifest(
        tmp_path, mt_signature=pipeline.mt_signature(JobConfig(sentence_units=True))
    )
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline(
        "video.mp4",
        JobConfig(cache_dir=str(cache_dir), sentence_units=False),
        resume=True,
    )

    assert called == ["translate", "speak", "timing", "mix", "export"]


def test_sentence_units_on_change_reruns_translate_and_downstream(
    monkeypatch, tmp_path
) -> None:
    """Re-enabling sentence units invalidates a script cached with them off."""
    called: list[str] = []
    cache_dir = _force_test_manifest(
        tmp_path, mt_signature=pipeline.mt_signature(JobConfig(sentence_units=False))
    )
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline(
        "video.mp4",
        JobConfig(cache_dir=str(cache_dir), sentence_units=True),
        resume=True,
    )

    assert called == ["translate", "speak", "timing", "mix", "export"]


def test_voxcpm2_voice_signature_tracks_reference_audio(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    same = tmp_path / "same.wav"
    other = tmp_path / "other.wav"
    first.write_bytes(b"voice one")
    same.write_bytes(b"voice one")
    other.write_bytes(b"voice two")

    first_signature = pipeline.tts_signature(
        JobConfig(tts_engine="voxcpm2", ref_voice=str(first))
    )

    assert first_signature == pipeline.tts_signature(
        JobConfig(tts_engine="voxcpm2", ref_voice=str(same))
    )
    assert first_signature != pipeline.tts_signature(
        JobConfig(tts_engine="voxcpm2", ref_voice=str(other))
    )
    assert pipeline.tts_signature(JobConfig(tts_engine="voxcpm2"))


def test_voxcpm2_voice_signature_tracks_voice_profile() -> None:
    female = pipeline.tts_signature(
        JobConfig(tts_engine="voxcpm2", voxcpm2_voice="vi_female_warm")
    )
    male = pipeline.tts_signature(
        JobConfig(tts_engine="voxcpm2", voxcpm2_voice="vi_male_calm")
    )

    assert female != male


def test_voxcpm2_voice_change_reruns_speak_and_downstream(
    monkeypatch, tmp_path: Path
) -> None:
    called: list[str] = []
    old_voice = tmp_path / "old.wav"
    new_voice = tmp_path / "new.wav"
    old_voice.write_bytes(b"old voice")
    new_voice.write_bytes(b"new voice")
    old_cfg = JobConfig(tts_engine="voxcpm2", ref_voice=str(old_voice))
    cfg = JobConfig(
        cache_dir=str(tmp_path / "cache"),
        tts_engine="voxcpm2",
        ref_voice=str(new_voice),
    )
    cache_dir = _force_test_manifest(
        tmp_path,
        tts_signature=pipeline.tts_signature(old_cfg),
        tts_engine="voxcpm2",
    )
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline("video.mp4", cfg, resume=True)

    assert called == ["speak", "timing", "mix", "export"]
    saved = JobManifest.load(cache_dir / "hash1" / "job.json")
    assert saved.tts_signature == pipeline.tts_signature(cfg)


def test_legacy_voxcpm2_audio_is_rebuilt_for_stable_default_voice(
    monkeypatch, tmp_path: Path
) -> None:
    called: list[str] = []
    cache_dir = _force_test_manifest(tmp_path, tts_engine="voxcpm2")
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)
    cfg = JobConfig(cache_dir=str(cache_dir), tts_engine="voxcpm2")

    pipeline.run_pipeline("video.mp4", cfg, resume=True)

    assert called == ["speak", "timing", "mix", "export"]
    saved = JobManifest.load(cache_dir / "hash1" / "job.json")
    assert saved.tts_signature == pipeline.tts_signature(cfg)


def test_empty_mt_signature_never_invalidates_resume(monkeypatch, tmp_path) -> None:
    """A pre-change ``job.json`` (no/empty signature) keeps resuming as today."""
    called: list[str] = []
    cache_dir = _force_test_manifest(tmp_path)  # mt_signature="" (pre-change shape)
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline(
        "video.mp4",
        JobConfig(cache_dir=str(cache_dir), mt_engine="hunyuan"),
        resume=True,
    )

    assert called == []
    # A skipped translate must NOT rewrite the stored signature: the pre-change
    # (empty) signature is left untouched on disk.
    assert JobManifest.load(cache_dir / "hash1" / "job.json").mt_signature == ""


def test_changed_export_profile_only_reexports(monkeypatch, tmp_path) -> None:
    called: list[str] = []
    cache_dir = _force_test_manifest(tmp_path)
    job = cache_dir / "hash1" / "job.json"
    manifest = JobManifest.load(job)
    manifest.export_signature = pipeline.export_mod.export_signature(JobConfig())
    manifest.save(job)
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)
    pipeline.run_pipeline("video.mp4", JobConfig(cache_dir=str(cache_dir), sub_font_size=30), resume=True)
    assert called == ["export"]


def _text_selection_job(tmp_path: Path) -> tuple[Path, Path]:
    cache_dir = _force_test_manifest(tmp_path)
    artifacts_dir = cache_dir / "hash1"
    script = [
        {"idx": 0, "vi": "legacy", "spoken_vi": "spoken one", "subtitle_vi": "caption one"}
    ]
    units = {
        "units": [
            {
                "index": 0,
                "indices": [0],
                "vi": "legacy",
                "spoken_vi": "spoken one",
                "subtitle_vi": "caption one",
            }
        ]
    }
    (artifacts_dir / "script.json").write_text(
        json.dumps(script, ensure_ascii=False), encoding="utf-8"
    )
    (artifacts_dir / "units.json").write_text(
        json.dumps(units, ensure_ascii=False), encoding="utf-8"
    )
    manifest_path = artifacts_dir / "job.json"
    manifest = JobManifest.load(manifest_path)
    manifest.spoken_text_signature, manifest.subtitle_text_signature = (
        pipeline.text_signatures(manifest)
    )
    manifest.paths.update({"script": "script.json", "units": "units.json"})
    manifest.save(manifest_path)
    return cache_dir, artifacts_dir


def test_subtitle_text_edit_only_reexports(monkeypatch, tmp_path) -> None:
    called: list[str] = []
    cache_dir, artifacts_dir = _text_selection_job(tmp_path)
    units_path = artifacts_dir / "units.json"
    units = json.loads(units_path.read_text(encoding="utf-8"))
    units["units"][0]["subtitle_vi"] = "caption two"
    units_path.write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline("video.mp4", JobConfig(cache_dir=str(cache_dir)), resume=True)

    assert called == ["export"]
    saved = JobManifest.load(cache_dir / "hash1" / "job.json")
    assert (saved.spoken_text_signature, saved.subtitle_text_signature) == (
        pipeline.text_signatures(saved)
    )


def test_spoken_text_edit_reruns_speech_and_downstream(monkeypatch, tmp_path) -> None:
    called: list[str] = []
    cache_dir, artifacts_dir = _text_selection_job(tmp_path)
    script_path = artifacts_dir / "script.json"
    script = json.loads(script_path.read_text(encoding="utf-8"))
    script[0]["spoken_vi"] = "spoken two"
    script_path.write_text(json.dumps(script, ensure_ascii=False), encoding="utf-8")
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline("video.mp4", JobConfig(cache_dir=str(cache_dir)), resume=True)

    assert called == ["speak", "timing", "mix", "export"]
    saved = JobManifest.load(cache_dir / "hash1" / "job.json")
    assert (saved.spoken_text_signature, saved.subtitle_text_signature) == (
        pipeline.text_signatures(saved)
    )


def test_empty_text_signatures_keep_legacy_resume_behavior(monkeypatch, tmp_path) -> None:
    called: list[str] = []
    cache_dir, artifacts_dir = _text_selection_job(tmp_path)
    manifest_path = artifacts_dir / "job.json"
    manifest = JobManifest.load(manifest_path)
    manifest.spoken_text_signature = ""
    manifest.subtitle_text_signature = ""
    manifest.save(manifest_path)
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline("video.mp4", JobConfig(cache_dir=str(cache_dir)), resume=True)

    assert called == []
    saved = JobManifest.load(manifest_path)
    assert saved.spoken_text_signature == saved.subtitle_text_signature == ""


def test_subtitles_only_does_not_acknowledge_unspoken_edit(monkeypatch, tmp_path) -> None:
    called: list[str] = []
    cache_dir, artifacts_dir = _text_selection_job(tmp_path)
    manifest_path = artifacts_dir / "job.json"
    manifest = JobManifest.load(manifest_path)
    previous_spoken_signature = manifest.spoken_text_signature
    manifest.mode = "subtitles-only"
    manifest.save(manifest_path)

    script_path = artifacts_dir / "script.json"
    script = json.loads(script_path.read_text(encoding="utf-8"))
    script[0]["spoken_vi"] = "spoken after audio"
    script_path.write_text(json.dumps(script, ensure_ascii=False), encoding="utf-8")
    units_path = artifacts_dir / "units.json"
    units = json.loads(units_path.read_text(encoding="utf-8"))
    units["units"][0]["subtitle_vi"] = "caption after edit"
    units_path.write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline(
        "video.mp4",
        JobConfig(cache_dir=str(cache_dir), mode="subtitles-only"),
        resume=True,
    )

    saved = JobManifest.load(manifest_path)
    assert called == ["export"]
    assert saved.spoken_text_signature == previous_spoken_signature
    assert saved.subtitle_text_signature == pipeline.text_signatures(saved)[1]

    called.clear()
    pipeline.run_pipeline(
        "video.mp4", JobConfig(cache_dir=str(cache_dir), mode="dub"), resume=True
    )

    assert called == ["speak", "timing", "mix", "export"]


def test_translate_run_records_mt_signature(monkeypatch, tmp_path) -> None:
    called: list[str] = []
    cache_dir = _force_test_manifest(tmp_path)
    _stub_stages(monkeypatch, cache_dir, tmp_path, called)
    cfg = JobConfig(cache_dir=str(cache_dir))

    pipeline.run_pipeline("video.mp4", cfg, resume=True, force=["translate"])

    saved = JobManifest.load(cache_dir / "hash1" / "job.json")
    assert saved.mt_signature == pipeline.mt_signature(cfg)

"""Tests for M2-T5 folder batching + loudness config (M2-T6: silent clip fixed).

Fast in-process tests monkeypatch ``cli_batch.run_pipeline`` to exercise the
batch bookkeeping (skip/failure/resume/interrupt) without models; one
subprocess test drives the real command on short clips, including a
**speech-less** clip that must now succeed with no subtitle stream (the M2-T5
export bug fixed in M2-T6).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vidub import cli_batch
from vidub.config import JobConfig
from vidub.stages.mix import loudnorm_filter
from vidub.utils.proc import find_tool, run

ROOT = Path(__file__).resolve().parents[1]
DEMO_MP4 = ROOT / "samples" / "demo_en.mp4"


# ------------------------------------------------------------------ loudness
def test_loudnorm_filter_default_and_override() -> None:
    # default matches the old hard-coded string
    assert loudnorm_filter(JobConfig()) == "loudnorm=I=-16:TP=-1.5:LRA=11"

    custom = JobConfig(loudness_target_i=-20.0, loudness_tp=-2.0, loudness_lra=8.0)
    assert loudnorm_filter(custom) == "loudnorm=I=-20:TP=-2:LRA=8"


def test_cli_loudness_flags_reach_the_filter() -> None:
    from vidub.cli import _build_cfg, build_parser

    args = build_parser().parse_args(
        [
            "run", "whatever.mp4",
            "--loudness-target", "-14",
            "--loudness-tp", "-1.0",
            "--loudness-lra", "7",
        ]
    )
    cfg = _build_cfg(args)
    assert (cfg.loudness_target_i, cfg.loudness_tp, cfg.loudness_lra) == (
        -14.0, -1.0, 7.0,
    )
    assert loudnorm_filter(cfg) == "loudnorm=I=-14:TP=-1:LRA=7"


# --------------------------------------------------------- batch bookkeeping
def _fake_report(video: str) -> dict:
    stem = Path(video).stem
    return {
        "duration": 4.0,
        "stages": {"ingest": {"status": "done"}},
        "outputs": {"dubbed": f"output/{stem}/dubbed.vi.mp4"},
    }


def _make_folder(tmp_path: Path, names: list[str]) -> Path:
    folder = tmp_path / "vids"
    folder.mkdir()
    for name in names:
        (folder / name).write_bytes(b"x")
    return folder


def test_run_batch_reports_done_skipped_and_resumes(monkeypatch, tmp_path) -> None:
    folder = _make_folder(tmp_path, ["a.mp4", "b.mp4", "note.txt"])
    calls: list[str] = []

    def fake_run_pipeline(video, cfg, resume=True, force=None, burn=False):
        calls.append(Path(video).name)
        return _fake_report(video)

    monkeypatch.setattr(cli_batch, "run_pipeline", fake_run_pipeline)
    cfg = JobConfig(cache_dir=str(tmp_path / "cache"))

    summary, code = cli_batch.run_batch(folder, cfg)

    assert code == 0
    assert calls == ["a.mp4", "b.mp4"]
    assert summary["totals"] == {
        "total": 3, "done": 2, "failed": 0, "skipped": 1, "interrupted": 0,
    }
    by_input = {entry["input"]: entry for entry in summary["videos"]}
    assert by_input["a.mp4"]["status"] == "done"
    assert by_input["note.txt"]["status"] == "skipped"
    assert (folder / "batch-summary.json").exists()

    # --resume-batch must not re-run any already-done video
    calls.clear()
    summary2, code2 = cli_batch.run_batch(folder, cfg, resume_batch=True)
    assert code2 == 0
    assert calls == []
    done = [e for e in summary2["videos"] if e["status"] == "done"]
    assert len(done) == 2 and all(e.get("resumed") for e in done)


def test_run_batch_continues_after_a_failure(monkeypatch, tmp_path) -> None:
    folder = _make_folder(tmp_path, ["good.mp4", "bad.mp4"])

    def fake_run_pipeline(video, cfg, resume=True, force=None, burn=False):
        if Path(video).name == "bad.mp4":
            raise RuntimeError("corrupt video")
        return _fake_report(video)

    monkeypatch.setattr(cli_batch, "run_pipeline", fake_run_pipeline)
    cfg = JobConfig(cache_dir=str(tmp_path / "cache"))

    summary, code = cli_batch.run_batch(folder, cfg)

    assert code == 1
    by_input = {entry["input"]: entry for entry in summary["videos"]}
    assert by_input["good.mp4"]["status"] == "done"
    assert by_input["bad.mp4"]["status"] == "failed"
    assert "corrupt video" in by_input["bad.mp4"]["error"]
    assert summary["totals"]["failed"] == 1
    assert (folder / "batch-summary.json").exists()


def test_run_batch_writes_partial_summary_on_interrupt(monkeypatch, tmp_path) -> None:
    folder = _make_folder(tmp_path, ["a.mp4"])

    def boom(video, cfg, resume=True, force=None, burn=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_batch, "run_pipeline", boom)
    cfg = JobConfig(cache_dir=str(tmp_path / "cache"))

    summary, code = cli_batch.run_batch(folder, cfg)

    assert code == 130
    assert summary["totals"]["interrupted"] == 1
    assert (folder / "batch-summary.json").exists()


# ------------------------------------------------------------- real subprocess
def _demo_video() -> Path:
    """The speech sample, building it via ``samples/make_sample.py`` if needed."""
    if DEMO_MP4.exists():
        return DEMO_MP4
    spec = importlib.util.spec_from_file_location(
        "vidub_make_sample", ROOT / "samples" / "make_sample.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return Path(module.generate())
    except Exception as exc:  # noqa: BLE001 - environment problem -> skip
        pytest.skip(f"could not build samples/demo_en.mp4: {exc}")


def _trim_speech_clip(src: Path, dst: Path, start: float, duration: float = 4.0) -> None:
    """Cut a short speech-bearing clip from ``src`` with ffmpeg."""
    ffmpeg = find_tool("ffmpeg")
    assert ffmpeg is not None
    subprocess.run(
        [
            ffmpeg, "-y", "-ss", str(start), "-i", str(src), "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(dst),
        ],
        capture_output=True,
        check=True,
    )


def _silent_clip(dst: Path, duration: float = 3.0) -> None:
    """Write a speech-less clip (silent audio + test pattern video)."""
    ffmpeg = find_tool("ffmpeg")
    assert ffmpeg is not None
    subprocess.run(
        [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25",
            "-shortest", "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(dst),
        ],
        capture_output=True,
        check=True,
    )


def _stream_types(path: Path) -> set[str]:
    """ffprobe codec types present in ``path`` (video/audio/subtitle)."""
    ffprobe = find_tool("ffprobe")
    assert ffprobe is not None
    proc = run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "stream=codec_type",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
    )
    return {stream["codec_type"] for stream in json.loads(proc.stdout)["streams"]}


def test_batch_subprocess_clips_resume_and_failure(
    tmp_path: Path, ffmpeg_available: bool
) -> None:
    demo = _demo_video()
    folder = tmp_path / "vids"
    folder.mkdir()
    _trim_speech_clip(demo, folder / "a.mp4", start=0.0)
    _trim_speech_clip(demo, folder / "b.mp4", start=8.0)
    _silent_clip(folder / "silent.mp4")
    (folder / "note.txt").write_text("not a video", encoding="utf-8")
    cache = tmp_path / "cache"

    def run_batch(*extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable, "-m", "vidub.cli", "batch", str(folder),
                "--asr-model", "tiny", "--cache-dir", str(cache), *extra,
            ],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )

    first = run_batch()
    assert first.returncode == 0, first.stderr[-1500:]

    summary = json.loads((folder / "batch-summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["done"] == 3
    assert summary["totals"]["skipped"] == 1
    by_input = {entry["input"]: entry for entry in summary["videos"]}
    assert by_input["silent.mp4"]["status"] == "done"
    assert (tmp_path / "output" / "a" / "dubbed.vi.mp4").exists()
    assert (tmp_path / "output" / "b" / "dubbed.vi.mp4").exists()
    silent_out = tmp_path / "output" / "silent" / "dubbed.vi.mp4"
    assert silent_out.exists()
    # the speech-less clip must mux cleanly with no subtitle stream
    assert "subtitle" not in _stream_types(silent_out)
    assert {"video", "audio"} <= _stream_types(silent_out)

    tracked = [
        tmp_path / "output" / "a" / "dubbed.vi.mp4",
        tmp_path / "output" / "b" / "dubbed.vi.mp4",
        silent_out,
    ]
    mtimes = {str(path): path.stat().st_mtime_ns for path in tracked}

    second = run_batch("--resume-batch")
    assert second.returncode == 0, second.stderr[-1500:]
    for path in tracked:
        assert path.stat().st_mtime_ns == mtimes[str(path)], (
            f"{path} was re-created on --resume-batch"
        )

    # a corrupt video makes the batch fail, but the good ones stay done
    (folder / "corrupt.mp4").write_bytes(b"definitely not an mp4")
    third = run_batch()
    assert third.returncode == 1

    summary3 = json.loads((folder / "batch-summary.json").read_text(encoding="utf-8"))
    by_input = {entry["input"]: entry for entry in summary3["videos"]}
    assert by_input["corrupt.mp4"]["status"] == "failed"
    assert by_input["corrupt.mp4"].get("error")
    assert by_input["a.mp4"]["status"] == "done"
    assert by_input["b.mp4"]["status"] == "done"
    assert summary3["totals"]["failed"] == 1

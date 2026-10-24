"""Tests for the S0 ingest stage (M1-T1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.stages.ingest import ingest, video_hash
from vidub.utils.proc import find_tool, run


def _audio_info(path: Path) -> tuple[int, int]:
    """Return (sample_rate, channels) of the first audio stream in ``path``."""
    ffprobe = find_tool("ffprobe")
    assert ffprobe is not None
    proc = run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    stream = json.loads(proc.stdout)["streams"][0]
    return int(stream["sample_rate"]), int(stream["channels"])


def test_ingest_creates_cache_audio_and_manifest(
    tiny_clip: Path, tmp_path: Path, ffmpeg_available: bool
) -> None:
    cfg = JobConfig(cache_dir=str(tmp_path / "cache"))

    manifest = ingest(str(tiny_clip), cfg)

    # cache dir keyed by the video hash
    artifacts_dir = Path(cfg.cache_dir) / manifest.video_hash
    assert artifacts_dir.is_dir()
    assert manifest.artifacts_dir == artifacts_dir

    # audio extracted at 16 kHz mono
    audio_path = artifacts_dir / "audio.wav"
    assert audio_path.exists()
    sample_rate, channels = _audio_info(audio_path)
    assert sample_rate == 16000
    assert channels == 1

    # manifest round-trips from disk, with duration and stage status
    loaded = JobManifest.load(artifacts_dir / "job.json")
    assert loaded.video_hash == manifest.video_hash
    assert abs(loaded.duration - 4.0) <= 0.5
    assert loaded.stages["ingest"] == "done"
    assert loaded.paths["audio"] == "audio.wav"


def test_ingest_is_idempotent(tiny_clip: Path, tmp_path: Path, ffmpeg_available: bool) -> None:
    cfg = JobConfig(cache_dir=str(tmp_path / "cache"))

    first = ingest(str(tiny_clip), cfg)
    second = ingest(str(tiny_clip), cfg)

    assert first.video_hash == second.video_hash
    assert second.video_hash == video_hash(tiny_clip)

    subdirs = [d for d in Path(cfg.cache_dir).iterdir() if d.is_dir()]
    assert len(subdirs) == 1
    assert (subdirs[0] / "job.json").exists()


def test_ingest_missing_file_raises(tmp_path: Path) -> None:
    cfg = JobConfig(cache_dir=str(tmp_path / "cache"))
    with pytest.raises(FileNotFoundError):
        ingest(str(tmp_path / "does_not_exist.mp4"), cfg)

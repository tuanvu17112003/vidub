"""Shared pytest fixtures for the ViDub test-suite."""

from __future__ import annotations

import os

# CTranslate2 + torch bundle clashing OpenMP runtimes; set this before pytest
# imports any test module so every in-process model load (and every subprocess
# we spawn) is covered. Mirrors the guard in ``vidub/__init__.py``.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from vidub.utils.proc import find_tool, run


@pytest.fixture(scope="session")
def ffmpeg_available() -> bool:
    """Skip the test if ffmpeg/ffprobe are not on PATH; otherwise return True."""
    if not (find_tool("ffmpeg") and find_tool("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not available on PATH")
    return True


@pytest.fixture
def tiny_clip(tmp_path: Path, ffmpeg_available: bool) -> Path:
    """Generate a ~4 s silent-speech clip (testsrc2 video + sine audio)."""
    out = tmp_path / "tiny_clip.mp4"
    ffmpeg = find_tool("ffmpeg")
    assert ffmpeg is not None  # guaranteed by ffmpeg_available
    run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-t",
            "4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        capture_output=True,
    )
    return out

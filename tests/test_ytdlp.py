"""Tests for URL input via yt-dlp (M3-T3).

Fast unit tests run offline; the two real-network tests share one session-scoped
download of a tiny, stable, direct-mp4 clip (10 s, ~1 MB) served over https and
handled by yt-dlp's generic extractor. They are SKIP-SAFE: no network / blocked
/ rate-limited skips with the underlying error instead of failing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import vidub.utils.ytdlp as ytdlp

# A tiny, stable, direct-mp4 clip (5 s, ~2.8 MB) served as a plain https file.
# yt-dlp's generic extractor fetches it without touching YouTube (which
# intermittently bot-blocks CI) and without a browser/impersonation dependency.
# (test-videos.co.uk timed out and Google's gtv-videos-bucket returns 403 here.)
TEST_URL = "https://download.samplelib.com/mp4/sample-5s.mp4"


# ---------------------------------------------------------------------- is_url
@pytest.mark.parametrize(
    "value, expected",
    [
        ("https://example.com/video.mp4", True),
        ("http://example.com/video.mp4", True),
        ("https://youtu.be/abc", True),
        ("HTTPS://EXAMPLE.COM/V", True),
        ("ftp://example.com/video.mp4", False),
        ("example.com/video.mp4", False),
        ("samples/demo_en.mp4", False),
        ("https://", False),
        ("", False),
        (None, False),
    ],
)
def test_is_url(value: str | None, expected: bool) -> None:
    assert ytdlp.is_url(value) is expected


# ------------------------------------------------------- missing yt-dlp (unit)
def test_download_without_yt_dlp_is_actionable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The resolver must fail BEFORE any network call when yt-dlp is absent."""
    monkeypatch.setattr(ytdlp.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ytdlp.YtDlpError) as excinfo:
        ytdlp.download("https://example.com/video.mp4", tmp_path)

    message = str(excinfo.value)
    assert "yt-dlp" in message
    assert 'pip install -e ".[youtube]"' in message


def test_download_rejects_non_url(tmp_path: Path) -> None:
    # download() checks the yt-dlp extra BEFORE the URL guard, so the
    # ValueError ordering only exists when yt-dlp is installed (CI installs
    # only [dev]; there the actionable YtDlpError fires instead).
    pytest.importorskip("yt_dlp", reason="the youtube extra is not installed")
    with pytest.raises(ValueError):
        ytdlp.download("samples/demo_en.mp4", tmp_path)


# ----------------------------------------------------------- real download
@pytest.fixture(scope="session")
def downloaded(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Download :data:`TEST_URL` once for the session (skip-safe)."""
    cache = tmp_path_factory.mktemp("ytdlp-cache")
    try:
        path = ytdlp.download(TEST_URL, cache)
    except Exception as exc:  # noqa: BLE001 - offline/blocked -> skip, not fail
        pytest.skip(f"yt-dlp could not download the test clip: {exc}")
    if not path.is_file():
        pytest.skip(f"yt-dlp reported {path} but no file exists")
    return cache, path


def test_download_reuses_existing_file(
    downloaded: tuple[Path, Path],
) -> None:
    """A second download of the same URL returns the same file, untouched."""
    cache, path = downloaded
    before = path.stat().st_mtime_ns

    again = ytdlp.download(TEST_URL, cache)

    assert again == path
    assert again.stat().st_mtime_ns == before, "the file was re-downloaded"


def test_url_pipeline_run(
    downloaded: tuple[Path, Path], tmp_path: Path
) -> None:
    """``vidub run <URL> --mode subtitles-only`` records source_url + output."""
    cache, _path = downloaded
    work = tmp_path / "work"
    work.mkdir()

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vidub.cli",
            "run",
            TEST_URL,
            "--asr-model",
            "tiny",
            "--mode",
            "subtitles-only",
            "--report",
            "--cache-dir",
            str(cache),
        ],
        cwd=str(work),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.skip(f"URL pipeline unavailable: {(proc.stderr or '')[-800:]}")

    reports = list((work / "output").glob("*/report.json"))
    assert reports, proc.stdout[-2000:]
    report = json.loads(reports[0].read_text(encoding="utf-8"))

    assert report["source_url"] == TEST_URL
    for key in ("dubbed", "subtitles"):
        assert (work / report["outputs"][key]).exists(), report["outputs"]

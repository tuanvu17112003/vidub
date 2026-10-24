"""S0 Ingest: turn a video file into a cache directory with a 16 kHz mono WAV.

The cache directory is keyed by a stable hash of the absolute path, file size
and mtime, so re-running ingest on an unchanged file reuses the same directory
(idempotent) and a modified file gets a fresh one.

Standalone use::

    python -m vidub.stages.ingest samples/demo_en.mp4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.utils.proc import find_tool, run

__all__ = ["video_hash", "probe", "ingest", "main"]

log = logging.getLogger(__name__)

AUDIO_NAME = "audio.wav"
MANIFEST_NAME = "job.json"


def video_hash(video_path: str | Path) -> str:
    """Stable sha1 over (absolute path, file size, mtime) of ``video_path``."""
    path = Path(video_path).resolve()
    stat = path.stat()
    key = f"{path}\n{stat.st_size}\n{stat.st_mtime}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _require(tool: str) -> str:
    found = find_tool(tool)
    if not found:
        raise RuntimeError(
            f"{tool} is required for ingest but was not found on PATH. "
            "Install it with: winget install --id Gyan.FFmpeg -e "
            "--accept-source-agreements --accept-package-agreements"
        )
    return found


def _parse_fps(value: str | None) -> float:
    """Parse an ffprobe frame-rate like ``"25/1"`` into a float."""
    if not value:
        return 0.0
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den) if float(den) else 0.0
        return float(value)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(video_path: str | Path) -> dict[str, float | int]:
    """Probe ``video_path`` with ffprobe for duration, size and frame rate."""
    ffprobe = _require("ffprobe")
    proc = run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(proc.stdout or "{}")

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )
    fmt = data.get("format", {})

    duration = 0.0
    for candidate in (fmt.get("duration"), video_stream.get("duration")):
        try:
            duration = float(candidate)
            break
        except (TypeError, ValueError):
            continue

    return {
        "duration": duration,
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": _parse_fps(video_stream.get("r_frame_rate")),
    }


def ingest(video_path: str, cfg: JobConfig) -> JobManifest:
    """Ingest ``video_path`` into ``cfg.cache_dir`` and return the manifest."""
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"input video not found: {src}")

    ffmpeg = _require("ffmpeg")

    job_hash = video_hash(src)
    artifacts_dir = Path(cfg.cache_dir) / job_hash
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    audio_path = artifacts_dir / AUDIO_NAME
    if not audio_path.exists():
        log.info("extracting 16 kHz mono audio -> %s", audio_path)
        run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(src),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(audio_path),
            ],
            capture_output=True,
        )
    else:
        log.info("audio already present, reusing %s", audio_path)

    meta = probe(src)

    manifest = JobManifest(
        input_path=str(src.resolve()),
        video_hash=job_hash,
        duration=float(meta["duration"]),
        width=int(meta["width"]),
        height=int(meta["height"]),
        fps=float(meta["fps"]),
        src_lang=cfg.src_lang,
        tgt_lang=cfg.tgt_lang,
        mode=cfg.mode,
        asr_model=cfg.asr_model,
        mt_engine=cfg.mt_engine,
        tts_engine=cfg.tts_engine,
        cache_dir=cfg.cache_dir,
        paths={"audio": AUDIO_NAME, "job": MANIFEST_NAME},
        stages={"ingest": "done"},
    )
    manifest.save(artifacts_dir / MANIFEST_NAME)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m vidub.stages.ingest <video>``."""
    parser = argparse.ArgumentParser(
        prog="vidub.stages.ingest",
        description="S0 ingest: extract audio + probe a video into a cache dir.",
    )
    parser.add_argument("video", help="input video file")
    parser.add_argument(
        "--cache-dir", default=None, help="cache root (default: .vidub-cache)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = JobConfig()
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir

    manifest = ingest(args.video, cfg)
    manifest_path = manifest.artifacts_dir / MANIFEST_NAME
    print(str(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

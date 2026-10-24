"""S5 Mix: Demucs music/effects bed + placed segments + loudness normalization.

Runs Demucs once on the ingested audio (``--two-stems=vocals -n htdemucs``) and
uses ``no_vocals.wav`` as the bed. Separation runs on the auto-detected device
(``cuda`` when the torch build sees a GPU, else ``cpu``); the resolved device is
recorded on the manifest as ``sep_device`` for ``report.json``:

* mode ``dub``        -> bed only;
* mode ``voice-over`` -> bed + original ``vocals.wav`` at the configured
  ``cfg.voice_over_gain`` (linear, default ``0.15``).

Every fitted segment is overlaid at its effective start on a track as long as
the video, then normalized with ffmpeg ``loudnorm`` and written as 16-bit PCM to
``dub_track.wav``. The separation output is reused when already present.

Standalone use::

    python -m vidub.stages.mix <cache>/<hash>/job.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from pydub import AudioSegment

from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.utils.audio import run_ffmpeg, wav_duration
from vidub.utils.proc import run

__all__ = [
    "mix",
    "run_separation",
    "select_sep_device",
    "sep_track_dir",
    "loudnorm_filter",
    "main",
]

log = logging.getLogger(__name__)

AUDIO_NAME = "audio.wav"
TIMED_NAME = "timed.json"
DUB_TRACK_NAME = "dub_track.wav"
SEP_NAME = "sep"
DEMUCS_MODEL = "htdemucs"
VOCALS_NAME = "vocals.wav"
BED_NAME = "no_vocals.wav"
MIX_RATE = 44100
MIX_CHANNELS = 2
SUBTITLES_ONLY = "subtitles-only"


def _check_mode(mode: str) -> None:
    if mode == SUBTITLES_ONLY:
        raise RuntimeError(
            f"skipped: mode {mode!r} does not run the mix stage"
        )


def sep_track_dir(artifacts_dir: Path, audio_path: Path) -> Path:
    """Directory holding the separated stems for ``audio_path``."""
    return artifacts_dir / SEP_NAME / DEMUCS_MODEL / audio_path.stem


def select_sep_device() -> str:
    """Which device Demucs should run on: ``cuda`` when torch sees a GPU, else ``cpu``.

    torch is imported lazily here (the mix stage does not otherwise need it) and
    any import/probe failure is treated as CPU. The result is logged and recorded
    on the manifest (``sep_device``) so ``report.json`` shows where separation ran.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch missing -> CPU is a valid fallback
        log.debug("torch unavailable; demucs will run on cpu", exc_info=True)
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - probe must never take the mix stage down
        log.debug("torch CUDA probe failed; demucs will run on cpu", exc_info=True)
        return "cpu"


def run_separation(audio_path: Path, sep_root: Path, device: str | None = None) -> Path:
    """Run Demucs (once) and return the directory with the separated stems.

    ``device`` defaults to :func:`select_sep_device` (``cuda`` when torch sees a
    GPU, else ``cpu``). Reuses an existing
    ``<sep_root>/htdemucs/<track>/{vocals,no_vocals}.wav`` instead of re-running
    separation.
    """
    track_dir = sep_root / DEMUCS_MODEL / audio_path.stem
    expected = (track_dir / VOCALS_NAME, track_dir / BED_NAME)
    if all(path.exists() for path in expected):
        log.info("demucs output already present, reusing %s", track_dir)
        return track_dir

    device = device or select_sep_device()
    log.info(
        "running demucs (%s) on %s (device=%s)", DEMUCS_MODEL, audio_path, device
    )
    run(
        [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-n",
            DEMUCS_MODEL,
            "-d",
            device,
            "-o",
            str(sep_root),
            str(audio_path),
        ],
        capture_output=True,
    )

    missing = [path.name for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(
            f"demucs did not produce {', '.join(missing)} in {track_dir}"
        )
    return track_dir


def _load_clip(path: Path) -> AudioSegment:
    return (
        AudioSegment.from_wav(str(path))
        .set_frame_rate(MIX_RATE)
        .set_channels(MIX_CHANNELS)
    )


def _linear_gain_db(gain: float) -> float:
    return 20.0 * math.log10(gain) if gain > 0 else -120.0


def loudnorm_filter(cfg: JobConfig) -> str:
    """ffmpeg ``loudnorm`` filter built from the job's loudness settings.

    ``:g`` formatting keeps the default string byte-identical to the old
    hard-coded ``loudnorm=I=-16:TP=-1.5:LRA=11``.
    """
    return (
        f"loudnorm=I={cfg.loudness_target_i:g}"
        f":TP={cfg.loudness_tp:g}:LRA={cfg.loudness_lra:g}"
    )


def mix(manifest_path: str, cfg: JobConfig) -> JobManifest:
    """Build ``dub_track.wav`` from the Demucs bed and the fitted segments."""
    manifest = JobManifest.load(manifest_path)
    _check_mode(manifest.mode)

    artifacts_dir = manifest.artifacts_dir

    audio_path = artifacts_dir / manifest.paths.get("audio", AUDIO_NAME)
    if not audio_path.exists():
        raise FileNotFoundError(f"ingested audio not found: {audio_path}")

    timed_path = artifacts_dir / manifest.paths.get("timed", TIMED_NAME)
    if not timed_path.exists():
        raise FileNotFoundError(
            f"timed segments not found: {timed_path}. Run the timing stage first, "
            f"e.g. 'python -m vidub.stages.timing {manifest_path}'."
        )
    timed = json.loads(timed_path.read_text(encoding="utf-8"))

    sep_device = select_sep_device()
    track_dir = run_separation(audio_path, artifacts_dir / SEP_NAME, device=sep_device)
    bed_path = track_dir / BED_NAME
    base_seconds = manifest.duration if manifest.duration > 0 else wav_duration(bed_path)
    track = AudioSegment.silent(
        duration=max(int(round(base_seconds * 1000)), 0), frame_rate=MIX_RATE
    ).set_channels(MIX_CHANNELS)

    # Music/effects bed.
    track = track.overlay(_load_clip(bed_path), position=0)

    # voice-over keeps the original vocals, heavily ducked under the dub.
    if manifest.mode == "voice-over":
        vocals_path = track_dir / VOCALS_NAME
        if vocals_path.exists():
            vocals = _load_clip(vocals_path).apply_gain(
                _linear_gain_db(cfg.voice_over_gain)
            )
            track = track.overlay(vocals, position=0)

    # Placed, time-fitted speech segments.
    for record in timed:
        clip_path = artifacts_dir / str(record.get("file", ""))
        if not clip_path.exists():
            raise FileNotFoundError(f"fitted segment not found: {clip_path}")
        position = max(int(round(float(record.get("start", 0.0)) * 1000)), 0)
        track = track.overlay(_load_clip(clip_path), position=position)

    prelim_path = artifacts_dir / "_mix_prelim.wav"
    track.export(str(prelim_path), format="wav")

    dub_track_path = artifacts_dir / DUB_TRACK_NAME
    try:
        # loudnorm upsamples internally and leaks 192 kHz unless the output rate
        # is pinned explicitly.
        run_ffmpeg(
            [
                "-y",
                "-i",
                str(prelim_path),
                "-af",
                loudnorm_filter(cfg),
                "-ar",
                str(MIX_RATE),
                "-c:a",
                "pcm_s16le",
                str(dub_track_path),
            ]
        )
    finally:
        prelim_path.unlink(missing_ok=True)

    manifest.paths["dub_track"] = DUB_TRACK_NAME
    manifest.stages["mix"] = "done"
    # Record where separation ran so report.json can show it (like asr_device).
    manifest.sep_device = sep_device
    manifest.save(artifacts_dir / "job.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m vidub.stages.mix <job.json>``."""
    parser = argparse.ArgumentParser(
        prog="vidub.stages.mix",
        description="S5 mix: Demucs bed + placed segments -> dub_track.wav.",
    )
    parser.add_argument("job", help="path to the job.json manifest")
    parser.add_argument(
        "--cache-dir", default=None, help="cache root (default: manifest's)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = JobConfig()
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir

    try:
        manifest = mix(args.job, cfg)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(str(manifest.artifacts_dir / DUB_TRACK_NAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

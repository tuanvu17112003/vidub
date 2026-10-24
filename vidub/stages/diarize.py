"""S1.5 Diarize (optional): label segments with speakers via pyannote 3.1.

Runs between transcribe and translate, only when ``--diarize`` is set. Reads
``segments.json`` + ``audio.wav``, diarizes the audio, assigns each segment the
speaker with the largest time overlap, relabels speakers ``spk0``, ``spk1``, …
by descending total speech time, and writes ``segments.json`` back in place.

No partial writes: the diarizer runs to completion (or raises
:class:`~vidub.engines.diarize.pyannote_diarizer.DiarizeAuthError`) before the
file is touched.

Standalone use::

    python -m vidub.stages.diarize <cache>/<hash>/job.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vidub.config import JobConfig
from vidub.engines.diarize.pyannote_diarizer import (
    MIN_TURN_SECONDS,
    DiarizeAuthError,
    PyannoteDiarizer,
)
from vidub.manifest import JobManifest

__all__ = ["diarize", "assign_speakers", "preflight", "main"]

log = logging.getLogger(__name__)

SEGMENTS_NAME = "segments.json"
AUDIO_NAME = "audio.wav"
DEFAULT_SPEAKER = "spk0"


def _overlap(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(
    segments: list[dict],
    turns: list[dict],
    min_turn_seconds: float = MIN_TURN_SECONDS,
) -> list[dict]:
    """Assign each segment its dominant speaker and relabel speakers by talk time.

    Raw diarization labels (``SPEAKER_00`` …) are remapped to ``spk0``,
    ``spk1``, … ordered by descending total speech time. A segment with no
    overlap keeps its existing ``spk*`` label (or falls back to ``spk0``).
    """
    kept = [
        turn
        for turn in turns
        if float(turn.get("end", 0.0)) - float(turn.get("start", 0.0))
        >= min_turn_seconds
    ]

    totals: dict[str, float] = {}
    for turn in kept:
        speaker = str(turn["speaker"])
        totals[speaker] = totals.get(speaker, 0.0) + (
            float(turn["end"]) - float(turn["start"])
        )
    order = sorted(totals, key=lambda name: totals[name], reverse=True)
    remap = {raw: f"spk{index}" for index, raw in enumerate(order)}

    result: list[dict] = []
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        best: str | None = None
        best_overlap = 0.0
        for turn in kept:
            overlap = _overlap(
                start, end, float(turn["start"]), float(turn["end"])
            )
            if overlap > best_overlap:
                best_overlap, best = overlap, str(turn["speaker"])

        updated = dict(segment)
        if best is not None:
            updated["speaker"] = remap.get(best, DEFAULT_SPEAKER)
        else:
            existing = str(segment.get("speaker") or DEFAULT_SPEAKER)
            updated["speaker"] = (
                existing if existing.startswith("spk") else DEFAULT_SPEAKER
            )
        result.append(updated)
    return result


def preflight(cfg: JobConfig) -> None:
    """Fail fast (before any pipeline work) if diarization cannot run."""
    PyannoteDiarizer(model=cfg.diarize_model).check_ready()


def diarize(manifest_path: str, cfg: JobConfig) -> JobManifest:
    """Diarize the job's audio and write speaker labels back to ``segments.json``."""
    manifest = JobManifest.load(manifest_path)
    artifacts_dir = manifest.artifacts_dir

    segments_path = artifacts_dir / manifest.paths.get("segments", SEGMENTS_NAME)
    if not segments_path.exists():
        raise FileNotFoundError(
            f"segments not found: {segments_path}. Run the transcribe stage first, "
            f"e.g. 'python -m vidub.stages.transcribe {manifest_path}'."
        )

    audio_path = artifacts_dir / manifest.paths.get("audio", AUDIO_NAME)
    if not audio_path.exists():
        raise FileNotFoundError(f"ingested audio not found: {audio_path}")

    # This can raise DiarizeAuthError -> nothing is written (no partial output).
    turns = PyannoteDiarizer(model=cfg.diarize_model).diarize(str(audio_path))

    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    updated = assign_speakers(segments, turns)
    segments_path.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest.stages["diarize"] = "done"
    manifest.save(artifacts_dir / "job.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m vidub.stages.diarize <job.json>``."""
    parser = argparse.ArgumentParser(
        prog="vidub.stages.diarize",
        description="S1.5 diarize: label segments.json speakers with pyannote 3.1.",
    )
    parser.add_argument("job", help="path to the job.json manifest")
    parser.add_argument(
        "--cache-dir", default=None, help="cache root (default: manifest's)"
    )
    parser.add_argument(
        "--diarize-model", default=None, help="pyannote model id (default from config)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = JobConfig()
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir
    if args.diarize_model:
        cfg.diarize_model = args.diarize_model

    try:
        preflight(cfg)
        manifest = diarize(args.job, cfg)
    except (FileNotFoundError, DiarizeAuthError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(str(manifest.artifacts_dir / SEGMENTS_NAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

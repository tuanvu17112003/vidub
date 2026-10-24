"""S1.6 References (optional): per-speaker voice-cloning references.

Runs between diarize and translate, only when ``--clone-speakers`` is set. It
builds one clean vocal reference per diarized speaker so ViXTTS can dub each
speaker in a voice resembling their own (zero-shot cloning).

For every speaker found in ``segments.json`` the stage:

* collects that speaker's diarized turns (segment ``start``/``end`` windows);
* skips speakers with less than :data:`MIN_SPEAKER_SECONDS` total speech and
  turns shorter than :data:`MIN_TURN_SECONDS`, taking up to :data:`MAX_TURNS`
  longest turns, concatenated, capped at :data:`MAX_REF_SECONDS`;
* trims those windows out of the Demucs ``vocals.wav`` stem (running separation
  first if it is not cached yet) into ``<artifacts>/references/spkN.wav`` as
  16-bit PCM at the native sample rate.

The result is deterministic for a given job (same turns -> same cuts) and cached:
a re-run reuses an existing reference when its window signature is unchanged,
and a changed audio (new job hash) gets a fresh cache directory. Speakers that
fall back to the engine's default reference are recorded with ``fallback: true``
in the ``references.json`` sidecar (read by the speak stage and the report).

Standalone use::

    python -m vidub.stages.references <cache>/<hash>/job.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.stages.mix import (
    SEP_NAME,
    VOCALS_NAME,
    run_separation,
)
from vidub.utils.audio import run_ffmpeg

__all__ = [
    "extract_references",
    "select_windows",
    "REFERENCES_DIR",
    "REFERENCES_NAME",
    "MIN_TURN_SECONDS",
    "MIN_SPEAKER_SECONDS",
    "MAX_REF_SECONDS",
    "main",
]

log = logging.getLogger(__name__)

SEGMENTS_NAME = "segments.json"
AUDIO_NAME = "audio.wav"
REFERENCES_DIR = "references"
REFERENCES_NAME = "references.json"

# Turns shorter than this are too noisy to clone from.
MIN_TURN_SECONDS = 1.5
# A speaker needs at least this much speech, otherwise we keep the default ref.
MIN_SPEAKER_SECONDS = 3.0
# Reference length budget (XTTS wants >= 6 s and works best well under 30 s).
MAX_REF_SECONDS = 20.0
MAX_TURNS = 3


def _speaker_turns(segments: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """Group segment windows by speaker (skipping empty/invalid windows)."""
    turns: dict[str, list[tuple[float, float]]] = {}
    for segment in segments:
        speaker = str(segment.get("speaker") or "spk0")
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        if end <= start:
            continue
        turns.setdefault(speaker, []).append((start, end))
    return turns


def select_windows(turns: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Pick up to :data:`MAX_TURNS` longest turns for one speaker.

    Returns ``[]`` when the speaker has less than :data:`MIN_SPEAKER_SECONDS`
    total speech (or no turn reaches :data:`MIN_TURN_SECONDS`), so callers fall
    back to the engine's default reference. The total is capped at
    :data:`MAX_REF_SECONDS` and the choice is deterministic (longest first,
    earliest start as the tie-breaker).
    """
    if sum(end - start for start, end in turns) < MIN_SPEAKER_SECONDS:
        return []

    qualifying = [
        (start, end) for start, end in turns if end - start >= MIN_TURN_SECONDS
    ]
    qualifying.sort(key=lambda turn: (-(turn[1] - turn[0]), turn[0]))

    windows: list[tuple[float, float]] = []
    used = 0.0
    for start, end in qualifying:
        if len(windows) >= MAX_TURNS:
            break
        duration = end - start
        remaining = MAX_REF_SECONDS - used
        if remaining <= 0:
            break
        if duration > remaining:
            if remaining < MIN_TURN_SECONDS:
                break
            windows.append((start, start + remaining))
            used += remaining
            break
        windows.append((start, end))
        used += duration
    return windows


def _signature(windows: list[tuple[float, float]]) -> str:
    """Stable string for the chosen windows (reuse check across runs)."""
    return "|".join(f"{start:.3f}-{end:.3f}" for start, end in windows)


def _load_existing(artifacts_dir: Path) -> dict[str, dict]:
    """Previous ``references.json`` rows keyed by speaker (``{}`` when absent)."""
    try:
        data = json.loads(
            (artifacts_dir / REFERENCES_NAME).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, list):
        return {}
    return {
        str(row["speaker"]): row
        for row in data
        if isinstance(row, dict) and row.get("speaker")
    }


def _write_reference(
    vocals: Path, windows: list[tuple[float, float]], out: Path
) -> None:
    """Trim + concatenate ``windows`` from ``vocals`` into a 16-bit PCM WAV."""
    if len(windows) == 1:
        start, end = windows[0]
        filter_complex = (
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
            "asetpts=PTS-STARTPTS[out]"
        )
    else:
        parts = [
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
            f"asetpts=PTS-STARTPTS[a{index}]"
            for index, (start, end) in enumerate(windows)
        ]
        labels = "".join(f"[a{index}]" for index in range(len(windows)))
        filter_complex = (
            ";".join(parts)
            + f";{labels}concat=n={len(windows)}:v=0:a=1[out]"
        )

    run_ffmpeg(
        [
            "-y",
            "-i",
            str(vocals),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
    )


def _speaker_order(speaker: str) -> tuple[int, str]:
    """Sort ``spk0``, ``spk1``, … numerically (stable for other labels)."""
    digits = "".join(char for char in speaker if char.isdigit())
    return (int(digits) if digits else 0, speaker)


def extract_references(manifest_path: str, cfg: JobConfig) -> JobManifest:
    """Extract a per-speaker reference WAV into ``<artifacts>/references``."""
    manifest = JobManifest.load(manifest_path)
    artifacts_dir = manifest.artifacts_dir

    segments_path = artifacts_dir / manifest.paths.get("segments", SEGMENTS_NAME)
    if not segments_path.exists():
        raise FileNotFoundError(
            f"segments not found: {segments_path}. Run the transcribe (and "
            f"diarize) stages first, e.g. "
            f"'python -m vidub.stages.diarize {manifest_path}'."
        )

    audio_path = artifacts_dir / manifest.paths.get("audio", AUDIO_NAME)
    if not audio_path.exists():
        raise FileNotFoundError(f"ingested audio not found: {audio_path}")

    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    speakers = _speaker_turns(segments)

    refs_dir = artifacts_dir / REFERENCES_DIR
    refs_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(artifacts_dir)

    rows: list[dict[str, object]] = []
    vocals: Path | None = None
    for speaker in sorted(speakers, key=_speaker_order):
        windows = select_windows(speakers[speaker])
        if not windows:
            log.info(
                "speaker %s: too little usable speech; keeping the engine "
                "default reference",
                speaker,
            )
            rows.append(
                {
                    "speaker": speaker,
                    "file": "",
                    "seconds": 0.0,
                    "fallback": True,
                    "signature": "",
                }
            )
            continue

        signature = _signature(windows)
        relative = f"{REFERENCES_DIR}/{speaker}.wav"
        out_path = artifacts_dir / relative
        previous = existing.get(speaker)
        if (
            previous
            and previous.get("signature") == signature
            and out_path.is_file()
        ):
            seconds = float(previous.get("seconds") or 0.0)
            log.info("speaker %s: reusing reference %s", speaker, out_path)
        else:
            if vocals is None:
                # Reuses cached stems; runs Demucs only when they are missing.
                track_dir = run_separation(audio_path, artifacts_dir / SEP_NAME)
                vocals = track_dir / VOCALS_NAME
            _write_reference(vocals, windows, out_path)
            seconds = sum(end - start for start, end in windows)
            log.info(
                "speaker %s: extracted %.1fs reference -> %s",
                speaker,
                seconds,
                out_path,
            )
        rows.append(
            {
                "speaker": speaker,
                "file": relative,
                "seconds": seconds,
                "fallback": False,
                "signature": signature,
            }
        )

    (artifacts_dir / REFERENCES_NAME).write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest.paths["references"] = REFERENCES_DIR
    manifest.paths["references_json"] = REFERENCES_NAME
    manifest.stages["references"] = "done"
    manifest.save(artifacts_dir / "job.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m vidub.stages.references <job.json>``."""
    parser = argparse.ArgumentParser(
        prog="vidub.stages.references",
        description="S1.6 references: per-speaker cloning WAVs from the vocals stem.",
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
        manifest = extract_references(args.job, cfg)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(str(manifest.artifacts_dir / REFERENCES_NAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

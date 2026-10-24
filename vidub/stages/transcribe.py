"""S1 Transcribe: ingested ``audio.wav`` -> timed segments + a draft source SRT.

Reads ``job.json``, runs the ASR engine over the ingested audio, writes
``segments.json`` (the machine-readable segment list) and ``draft.srt`` (source
language subtitles, loadable with pysubs2), records both under
``manifest.paths`` and marks the ``transcribe`` stage done.

Standalone use::

    python -m vidub.stages.transcribe <cache>/<hash>/job.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pysubs2

from vidub.config import JobConfig
from vidub.engines.asr import build_asr
from vidub.manifest import JobManifest, Segment
from vidub.utils.langcheck import detect_mismatch, vn_diacritic_ratio
from vidub.utils.punct import restore_segments
from vidub.utils.qc import qc_transcript

__all__ = ["transcribe", "main"]

log = logging.getLogger(__name__)

SEGMENTS_NAME = "segments.json"
SRT_NAME = "draft.srt"
TRANSCRIPT_QC_NAME = "transcript_qc.json"
# How far after the last spoken word a subtitle cue may linger.
CUE_TAIL_SECONDS = 0.3


def write_segments(path: str | Path, segments: list[Segment]) -> Path:
    """Write ``segments`` as a JSON list of dicts, returning the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(seg) for seg in segments]
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target


def cue_end(seg: Segment) -> float:
    """End time for a subtitle cue.

    With word timing, the cue is tightened to ``last_word_end + 0.3`` (never
    later than the segment end, never before the cue start); without words the
    segment end is used unchanged.
    """
    end = float(seg.end)
    if seg.words:
        last_word_end = max(
            (float(word.get("end", 0.0)) for word in seg.words), default=0.0
        )
        if last_word_end > 0:
            end = min(end, last_word_end + CUE_TAIL_SECONDS)
    return max(end, float(seg.start))


def write_srt(path: str | Path, segments: list[Segment]) -> Path:
    """Write ``segments`` as an SRT subtitle file via pysubs2."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    subs = pysubs2.SSAFile()
    for seg in segments:
        subs.append(
            pysubs2.SSAEvent(
                start=pysubs2.make_time(s=seg.start),
                end=pysubs2.make_time(s=cue_end(seg)),
                text=seg.text,
            )
        )
    subs.save(str(target), format="srt")
    return target


def transcribe(manifest_path: str, cfg: JobConfig) -> JobManifest:
    """Run ASR for the job described by ``manifest_path``.

    The engine is selected by :func:`vidub.engines.asr.build_asr`, looked up as
    :data:`build_asr` on this module so tests can monkeypatch it out.
    """
    manifest = JobManifest.load(manifest_path)
    artifacts_dir = manifest.artifacts_dir

    audio_rel = manifest.paths.get("audio", "audio.wav")
    audio_path = artifacts_dir / audio_rel
    if not audio_path.exists():
        raise FileNotFoundError(
            f"ingested audio not found: {audio_path} (run the ingest stage first)"
        )

    engine = build_asr(manifest.asr_model)
    segments = engine.transcribe(
        str(audio_path),
        manifest.src_lang or None,
        word_timestamps=cfg.asr_words,
    )
    # Record the *resolved* device after inference: a CUDA engine can fall back
    # to CPU mid-run, and the report should reflect what actually ran.
    manifest.asr_device = str(getattr(engine, "device", "") or "")

    # Language guard: a Vietnamese-specialist ASR model on non-Vietnamese audio
    # hallucinates Vietnamese. Catch it here, *before* writing any artifact, so
    # no partial segments.json/draft.srt survives to poison later stages.
    full_text = " ".join(seg.text for seg in segments)
    mismatch = detect_mismatch(full_text, manifest.src_lang)
    if mismatch and not cfg.allow_lang_mismatch:
        raise RuntimeError(
            f"language mismatch after ASR with model {manifest.asr_model!r}: "
            f"{mismatch} (measured ratio "
            f"{vn_diacritic_ratio(full_text):.2f}). "
            "Fix: use the default auto/large-v3 model for English audio, or set "
            "--src-lang to match the audio, or pass --allow-lang-mismatch to "
            "force the run."
        )

    # Level-1 transcript QC (QUALITY-ROADMAP.md §3.2): classify every segment,
    # retry the flagged windows once with changed settings, and record a
    # transcript verdict. Runs after the language guard and before any artifact
    # is written, so a corrected hypothesis lands in segments.json/draft.srt.
    if cfg.qc_transcript:
        qc_result = qc_transcript(
            segments,
            manifest.src_lang,
            engine=engine,
            audio_path=audio_path,
            out_dir=artifacts_dir,
        )
        (artifacts_dir / TRANSCRIPT_QC_NAME).write_text(
            json.dumps(qc_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        manifest.paths["transcript_qc"] = TRANSCRIPT_QC_NAME
        manifest.transcript_verdict = str(qc_result.get("verdict", ""))

    # Opt-in punctuation/casing restoration (QUALITY-ROADMAP.md P1.6): a small
    # token-classification model inserts periods/commas/question marks and basic
    # casing into ASR text that lacks punctuation, improving the MT input. Runs
    # after QC and before the artifacts are written, so both segments.json and
    # draft.srt carry the restored text. Disabled (the default) = byte-identical.
    if cfg.restore_punct:
        inserted = restore_segments(segments, cfg.restore_punct)
        log.info(
            "punctuation restoration: %d mark(s) inserted across %d segment(s)",
            inserted,
            len(segments),
        )

    write_segments(artifacts_dir / SEGMENTS_NAME, segments)
    write_srt(artifacts_dir / SRT_NAME, segments)

    manifest.segments = segments
    manifest.paths["segments"] = SEGMENTS_NAME
    manifest.paths["srt"] = SRT_NAME
    manifest.stages["transcribe"] = "done"
    manifest.save(artifacts_dir / "job.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m vidub.stages.transcribe <job.json>``."""
    parser = argparse.ArgumentParser(
        prog="vidub.stages.transcribe",
        description="S1 transcribe: ASR over ingested audio -> segments + draft SRT.",
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

    manifest = transcribe(args.job, cfg)
    print(str(manifest.artifacts_dir / SEGMENTS_NAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""S6 Export: mux the dubbed audio + Vietnamese subtitles into the final video.

Outputs (all under ``output/<video-basename>/``):

* ``dubbed.vi.mp4``   — video (copied) + dubbed audio (AAC) + soft ``mov_text``
  Vietnamese subtitles, or burned-in subtitles with ``burn=True``;
* ``subtitles.vi.srt`` — the translated Vietnamese subtitle track. On the
  sentence-unit path it carries **one cue per sentence unit** (times from the
  unit's placed ``timed.json`` row, text = the unit's ``vi``), so a cue boundary
  never cuts a sentence; on legacy caches it is one cue per ``script.json`` row;
* ``subtitles.vi.ass`` — the same track with the ViDub style block (position,
  font size, vertical margin); used by players/editors and by the burn path's
  ``force_style`` (SU-1-T2);
* ``script.tsv``      — ``start \\t end \\t speaker \\t en \\t vi`` per segment;
* ``report.json``     — input/duration/mode/engines, the input ``source_url``
  (empty for a local file), a ``meta`` block
  (tool version, resolved ASR model + device, separation device, engines, mode,
  source url, cloned-speaker count, total seconds),
  the subtitle track description, a ``translation`` counter block
  (``passthrough`` / ``low_confidence``), a ``glossary`` block
  (``terms`` / ``replacements`` / ``spliced`` / ``enforced``), a
  ``pronunciation`` block (``replacements``), a top-level
  ``cleanup`` block (``hesitations`` / ``filler_removals`` / ``style_fixes``), a
  ``pacing`` block (``stretched`` / ``retranslated`` / ``clipped`` /
  ``smart_trims`` / ``clips`` / ``max_stretch`` / ``max_neighbor_delta``), a
  ``references`` block (per-speaker cloning reference lengths), ``speech_coverage``
  (the timing stage's coverage fraction), per-stage status + timings, and output
  paths.

A job with no speech (empty translated script) gets **no subtitle stream** in
the output video; the report records ``"subtitles": "none (no speech detected)"``.

Standalone use::

    python -m vidub.stages.export <cache>/<hash>/job.json [--burn]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pysubs2

from vidub import __version__
from vidub.config import JobConfig, normalize_sub_position
from vidub.manifest import JobManifest
from vidub.utils.audio import run_ffmpeg
from vidub.utils.finalqc import run_gate
from vidub.utils.subqc import qc_rows
from vidub.utils.text_selection import selected_text

__all__ = [
    "export",
    "build_report",
    "output_dir_for",
    "build_mux_command",
    "write_subtitles_srt",
    "write_subtitles_ass",
    "PIPELINE_STAGES",
    "REPORT_STAGES",
    "MODE_SKIPPED_STAGES",
    "TTS_SKIPPED_NOTE",
    "SUBTITLES_ASS_NAME",
    "ASS_STYLE_NAME",
    "SUB_POSITION_ALIGNMENT",
    "SUB_POSITION_LEGACY_ALIGNMENT",
    "main",
]

log = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output")
DUBBED_NAME = "dubbed.vi.mp4"
SUBTITLES_NAME = "subtitles.vi.srt"
# SU-1-T2: styled ASS twin of the SRT. Carries the ViDub style block so players
# and editors get the requested position/size/margin; written for every mode.
SUBTITLES_ASS_NAME = "subtitles.vi.ass"
# Name of the single ASS style the writer emits (events reference it by name).
ASS_STYLE_NAME = "ViDub"
SCRIPT_TSV_NAME = "script.tsv"
REPORT_NAME = "report.json"
SCRIPT_NAME = "script.json"
GLOSSARY_NAME = "glossary.json"
PRONUNCIATION_NAME = "pronunciation.json"
TIMING_NAME = "timing.json"
TIMED_NAME = "timed.json"
UNITS_NAME = "units.json"
REFERENCES_NAME = "references.json"
DUB_TRACK_NAME = "dub_track.wav"
SUBTITLES_QC_NAME = "subtitles_qc.json"
SUBTITLES_ONLY = "subtitles-only"
# Report value for ``engines.tts`` in a mode that never synthesises speech.
TTS_SKIPPED_NOTE = "skipped (subtitles-only)"
# Report note used when a job produced no speech (empty script / SRT).
NO_SPEECH_SUBTITLES = "none (no speech detected)"

# SU-1-T2 subtitle position -> ASS alignment for the `.ass` style block, using
# the standard ASS (``\an``) numbering: 2 = bottom-center, 8 = top-center,
# 5 = middle-center.
SUB_POSITION_ALIGNMENT: dict[str, int] = {"bottom": 2, "top": 8, "center": 5}
# The ffmpeg ``subtitles`` filter builds a v4.00 (legacy) default style from the
# SRT input, and libass applies ``force_style=Alignment=`` with the LEGACY SSA
# (``\a``) numbering, NOT the ASS ``\an`` one: 2 = bottom-center, 6 = top-center,
# 10 = middle-center. This was proven with real burns: the ASS numbers put "top"
# in the middle band and "center" at the top. The `.ass` artifact uses
# ``SUB_POSITION_ALIGNMENT`` and the burn filter uses this map, so both land in
# the same band.
SUB_POSITION_LEGACY_ALIGNMENT: dict[str, int] = {
    "bottom": 2,
    "top": 6,
    "center": 10,
}
# Generic font family used when the caller does not name one. If the family is
# not installed, the ASS renderer (libass) falls back to its default sans-serif.
DEFAULT_ASS_FONT = "Arial"

# Canonical stage list + which stages a mode deliberately skips.
PIPELINE_STAGES = (
    "ingest",
    "transcribe",
    "translate",
    "speak",
    "timing",
    "mix",
    "export",
)
MODE_SKIPPED_STAGES: dict[str, frozenset[str]] = {
    "subtitles-only": frozenset({"speak", "timing", "mix"}),
}

# Report ordering. ``diarize`` and ``references`` are optional (only run with
# --diarize / --clone-speakers) so they are reported as "skipped: not requested"
# when they never ran.
REPORT_STAGES = (
    "ingest",
    "transcribe",
    "diarize",
    "references",
    "translate",
    "speak",
    "timing",
    "mix",
    "export",
)

# Optional stages that may be absent entirely (reported as not requested).
OPTIONAL_REPORT_STAGES = frozenset({"diarize", "references"})


def output_dir_for(manifest: JobManifest) -> Path:
    """Recorded output destination, or ``output/<video-basename>/`` initially."""
    destination = manifest.paths.get("output_dir")
    if destination:
        return Path(destination)
    stem = Path(manifest.input_path).stem or "output"
    return OUTPUT_ROOT / stem


def _subtitles_note(manifest: JobManifest) -> str:
    """Describe the subtitle track for a report rebuilt without an export run."""
    script = _load_script(manifest)
    if script is None:
        return "unknown"
    return NO_SPEECH_SUBTITLES if not script else "soft (mov_text)"


def _load_script(manifest: JobManifest) -> list[dict] | None:
    """Load the job's ``script.json`` (``None`` when absent/unreadable)."""
    script_path = manifest.artifacts_dir / manifest.paths.get("script", SCRIPT_NAME)
    try:
        loaded = json.loads(script_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, list) else None


def _translation_counts(script: list[dict] | None) -> dict[str, int]:
    """Count flagged translation rows (``passthrough`` / ``low_confidence``)."""
    rows = script or []
    return {
        "passthrough": sum(1 for row in rows if row.get("passthrough")),
        "low_confidence": sum(1 for row in rows if row.get("low_confidence")),
    }


def _speech_coverage(manifest: JobManifest) -> float | None:
    """The timing stage's coverage fraction, rounded to 2 dp (``None`` if unset)."""
    if manifest.speech_coverage is None:
        return None
    return round(float(manifest.speech_coverage), 2)


def _load_sidecar(manifest: JobManifest, key: str, default_name: str) -> dict:
    """Load a JSON-object sidecar for ``manifest`` (``{}`` when absent/invalid)."""
    path = manifest.artifacts_dir / manifest.paths.get(key, default_name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _int_field(data: dict, key: str, default: int = 0) -> int:
    try:
        return int(data.get(key, default) or 0)
    except (TypeError, ValueError):
        return default


def _float_field(data: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default) or 0.0)
    except (TypeError, ValueError):
        return default


def _glossary_counts(manifest: JobManifest) -> dict[str, int]:
    """Glossary counters written by the translate stage's ``glossary.json``.

    * ``terms``        — rows loaded from the TSV;
    * ``replacements`` — output-side replacements (a term present in MT output);
    * ``spliced``      — source-side rebuilds (the MT output lacked the target);
    * ``enforced``     — source-side enforcements found, rebuilt *or* already
      present (M3-C1 MA ruling).

    Absent/unreadable sidecar -> zeros, so older jobs report ``0``s.
    """
    data = _load_sidecar(manifest, "glossary", GLOSSARY_NAME)
    return {
        "terms": _int_field(data, "terms"),
        "replacements": _int_field(data, "replacements"),
        "spliced": _int_field(data, "spliced"),
        "enforced": _int_field(data, "enforced"),
    }


def _pronunciation_counts(manifest: JobManifest) -> dict[str, int]:
    """TTS pronunciation counters written by the speak stage's sidecar.

    ``replacements`` counts written forms swapped for their phonetic form in the
    string handed to the TTS (the subtitles keep the written form). An
    absent/unreadable sidecar reports ``0``, so lexicon-free jobs look the same.
    """
    data = _load_sidecar(manifest, "pronunciation", PRONUNCIATION_NAME)
    return {"replacements": _int_field(data, "replacements")}


def _subtitles_qc_report(manifest: JobManifest) -> dict:
    """Subtitle-standards QC counters written by the export stage's sidecar.

    Shape mirrors ``vidub.utils.subqc.qc_rows``'s stats. An absent/unreadable
    sidecar (or ``--no-subtitle-qc``) reports zeros with ``enabled: false``, so
    older jobs and disabled runs look the same.
    """
    data = _load_sidecar(manifest, "subtitles_qc", SUBTITLES_QC_NAME)
    violations = data.get("violations")
    if not isinstance(violations, dict):
        violations = {}
    result = {
        "enabled": bool(data.get("enabled", False)),
        "cues_in": _int_field(data, "cues_in"),
        "cues_out": _int_field(data, "cues_out"),
        "split": _int_field(data, "split"),
        "violations": {
            "cps": _int_field(violations, "cps"),
            "duration": _int_field(violations, "duration"),
            "line_length": _int_field(violations, "line_length"),
            "lines": _int_field(violations, "lines"),
        },
        "cps_max_observed": _float_field(data, "cps_max_observed"),
    }
    if isinstance(data.get("scheduling"), dict):
        result["scheduling"] = data["scheduling"]
    return result


def _cleanup_counts(manifest: JobManifest) -> dict[str, int]:
    """Text-cleanup counters: hesitations + style fixes (translate) + filler removals.

    ``style_fixes`` counts the deterministic Vietnamese word-order refinements
    (:mod:`vidub.utils.vnstyle`) applied to the MT/spliced text.
    """
    glossary = _load_sidecar(manifest, "glossary", GLOSSARY_NAME)
    timing = _load_sidecar(manifest, "timing", TIMING_NAME)
    return {
        "hesitations": _int_field(glossary, "hesitations"),
        "filler_removals": _int_field(timing, "filler_removals"),
        "style_fixes": _int_field(glossary, "style_fixes"),
    }


def _pacing_report(manifest: JobManifest) -> dict:
    """The timing stage's ``pacing`` block, read from its ``timing.json`` sidecar.

    A job whose timing stage never ran (or an older cache) reports zeros with no
    clips, so the block is always shaped the same.
    """
    data = _load_sidecar(manifest, "timing", TIMING_NAME)
    from vidub.utils.pace_metrics import pace_metrics

    timed_path = manifest.artifacts_dir / manifest.paths.get("timed", TIMED_NAME)
    try:
        timed_rows = json.loads(timed_path.read_text(encoding="utf-8"))
        if isinstance(timed_rows, list):
            data["combined"] = pace_metrics([row for row in timed_rows if isinstance(row, dict)])
    except (OSError, ValueError):
        pass
    clips = data.get("clips")
    if not isinstance(clips, list):
        clips = []
    try:
        max_stretch = float(data.get("max_stretch", 1.0) or 1.0)
    except (TypeError, ValueError):
        max_stretch = 1.0
    try:
        max_delta = float(data.get("max_neighbor_delta", 0.0) or 0.0)
    except (TypeError, ValueError):
        max_delta = 0.0
    result = {
        "stretched": _int_field(data, "stretched", manifest.stretched_segments),
        "retranslated": _int_field(data, "retranslated"),
        "clipped": _int_field(data, "clipped", len(clips)),
        "smart_trims": _int_field(data, "smart_trims"),
        "clips": clips,
        "max_stretch": round(max_stretch, 3),
        "max_neighbor_delta": round(max_delta, 3),
    }
    # FIT-1: mirrored only when the timing stage recorded it, so a job that ran
    # with the rate/recenter flags off keeps the exact original shape.
    rate_fit = data.get("rate_fit")
    if isinstance(rate_fit, dict):
        result["rate_fit"] = rate_fit
    if isinstance(data.get("combined"), dict):
        result["combined"] = data["combined"]
    return result


def _references_report(manifest: JobManifest) -> dict[str, float]:
    """Per-speaker cloning reference lengths (seconds) from ``references.json``.

    Empty when the references stage never ran. A speaker that fell back to the
    engine default is recorded with ``0.0`` seconds.
    """
    path = manifest.artifacts_dir / manifest.paths.get(
        "references_json", REFERENCES_NAME
    )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, list):
        return {}

    refs: dict[str, float] = {}
    for row in data:
        if not isinstance(row, dict) or not row.get("speaker"):
            continue
        try:
            seconds = float(row.get("seconds", 0.0))
        except (TypeError, ValueError):
            seconds = 0.0
        refs[str(row["speaker"])] = round(seconds, 2)
    return refs


def _cloned_speakers(manifest: JobManifest) -> int:
    """Speakers with a real cloned reference in ``references.json``.

    Counts the sidecar rows whose ``fallback`` flag is false (a speaker that fell
    back to the engine default is not cloned). ``0`` when the references stage
    never ran or the sidecar is absent/unreadable.
    """
    path = manifest.artifacts_dir / manifest.paths.get(
        "references_json", REFERENCES_NAME
    )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return 0
    if not isinstance(data, list):
        return 0
    return sum(
        1
        for row in data
        if isinstance(row, dict) and row.get("speaker") and not row.get("fallback")
    )


def build_report(
    manifest: JobManifest,
    timings: dict[str, float] | None = None,
    skipped: frozenset[str] | None = None,
    subtitles: str | None = None,
) -> dict:
    """Assemble the ``report.json`` structure for ``manifest``.

    ``subtitles`` describes the exported subtitle track; when omitted it is
    derived from the job's script (so a report rebuilt from a skipped export
    still states whether there was speech).

    In the ``subtitles-only`` mode ``speak``/``timing``/``mix`` never run (they
    are reported ``skipped``) and no TTS engine is used, so ``engines.tts``
    reads ``"skipped (subtitles-only)"``; the ``outputs`` block keeps naming the
    artifacts that mode actually produces (the vietsub MP4 + the ``.srt``).
    """
    timings = timings or {}
    if skipped is None:
        skipped = MODE_SKIPPED_STAGES.get(manifest.mode, frozenset())
    if subtitles is None:
        subtitles = _subtitles_note(manifest)

    stages: dict[str, dict[str, object]] = {}
    for name in REPORT_STAGES:
        if name in OPTIONAL_REPORT_STAGES and name not in manifest.stages:
            # Optional stage: only present when its flags were requested.
            stages[name] = {"status": "skipped", "reason": "not requested"}
            continue
        if name in skipped:
            # A stage this mode deliberately skips is always reported as
            # skipped, even when an earlier run (same cache, other mode) did it.
            status: str = "skipped"
        elif name in manifest.stages:
            status = manifest.stages[name]
        else:
            status = "missing"
        entry: dict[str, object] = {"status": status}
        if name in timings:
            entry["seconds"] = round(float(timings[name]), 3)
        stages[name] = entry

    output_dir = output_dir_for(manifest)
    # In subtitles-only no TTS runs, so ``engines.tts`` must state that honestly
    # rather than name an engine that was never used. Every other mode keeps the
    # exact previous value (the report stays byte-identical for dub/voice-over).
    tts_engine = (
        TTS_SKIPPED_NOTE if manifest.mode == SUBTITLES_ONLY else manifest.tts_engine
    )
    return {
        "input": manifest.input_path,
        "source_url": manifest.source_url,
        "duration": manifest.duration,
        "mode": manifest.mode,
        "meta": {
            "tool_version": __version__,
            "source_url": manifest.source_url,
            "asr_model": manifest.asr_model,
            "asr_device": manifest.asr_device,
            "sep_device": manifest.sep_device,
            "mt_engine": manifest.mt_engine,
            "tts_engine": manifest.tts_engine,
            "mode": manifest.mode,
            "cloned_speakers": _cloned_speakers(manifest),
            "total_seconds": round(sum(float(value) for value in timings.values()), 3),
        },
        "engines": {
            "asr": manifest.asr_model,
            "mt": manifest.mt_engine,
            "tts": tts_engine,
        },
        "subtitles": subtitles,
        "translation": _translation_counts(_load_script(manifest)),
        "glossary": _glossary_counts(manifest),
        "pronunciation": _pronunciation_counts(manifest),
        "subtitles_qc": _subtitles_qc_report(manifest),
        "cleanup": _cleanup_counts(manifest),
        "pacing": _pacing_report(manifest),
        "references": _references_report(manifest),
        "speech_coverage": _speech_coverage(manifest),
        "stages": stages,
        "outputs": {
            "dubbed": (output_dir / DUBBED_NAME).as_posix(),
            "subtitles": (output_dir / SUBTITLES_NAME).as_posix(),
            "subtitles_ass": (output_dir / SUBTITLES_ASS_NAME).as_posix(),
            "script_tsv": (output_dir / SCRIPT_TSV_NAME).as_posix(),
            "report": (output_dir / REPORT_NAME).as_posix(),
        },
    }


def _placed_unit_times(manifest: JobManifest) -> dict[int, tuple[float, float]]:
    """``{unit idx: (placed_start, placed_end)}`` from ``timed.json``.

    ``{}`` when the timing stage has not run or ``timed.json`` is absent/broken,
    so the subtitle builder falls back to ``units.json``'s locked span.
    """
    path = manifest.artifacts_dir / manifest.paths.get("timed", TIMED_NAME)
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(rows, list):
        return {}
    placed: dict[int, tuple[float, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("idx"))
        except (TypeError, ValueError):
            continue
        placed[idx] = (_float_field(row, "start"), _float_field(row, "end"))
    return placed


def _placed_align_words(manifest: JobManifest) -> dict[int, list]:
    """``{unit idx: align_words}`` from ``timed.json`` (Q-1-T4).

    The timing stage persists each multi-member unit's ASR **word timeline** on
    its ``timed.json`` row when the ``asr`` alignment path ran. Only rows that
    actually carry a usable list are returned, so a unit without one keeps the
    historical proportional cue split. ``{}`` when ``timed.json`` is absent or
    broken.
    """
    path = manifest.artifacts_dir / manifest.paths.get("timed", TIMED_NAME)
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(rows, list):
        return {}
    words: dict[int, list] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("idx"))
        except (TypeError, ValueError):
            continue
        value = row.get("align_words")
        if isinstance(value, list) and value:
            words[idx] = value
    return words


def _unit_subtitle_rows(manifest: JobManifest) -> list[dict] | None:
    """One subtitle row per sentence unit; ``None`` on the legacy path.

    Reads ``units.json`` (written by the translate stage on the sentence-unit
    path) and prefers each unit's **placed** span from ``timed.json`` — the
    cut/fit result the viewer actually hears — falling back to the unit's locked
    ``start``/``end``. When the placed row also carries an ASR word timeline the
    row gets ``align_words`` (Q-1-T4) so :func:`~vidub.utils.subqc.qc_rows` can
    split the cue on real word edges. Returns ``None`` when ``units.json`` is
    absent or empty (legacy ``--no-sentence-units`` runs and every pre-SA-1
    cache), so the caller keeps today's script-row path byte-identically. Rows
    are returned in timeline order.
    """
    doc = _load_sidecar(manifest, "units", UNITS_NAME)
    units = doc.get("units")
    if not isinstance(units, list) or not units:
        return None
    placed = _placed_unit_times(manifest)
    align_words = _placed_align_words(manifest)
    timed_texts: dict[int, str] = {}
    timed_path = manifest.artifacts_dir / manifest.paths.get("timed", "timed.json")
    if timed_path.exists():
        try:
            timed_rows = json.loads(timed_path.read_text(encoding="utf-8"))
            for record in timed_rows:
                if isinstance(record, dict) and record.get("retranslated") and record.get("vi"):
                    timed_texts[int(record["idx"])] = str(record["vi"])
        except (OSError, ValueError, TypeError, KeyError):
            timed_texts = {}
    rows: list[dict] = []
    for position, unit in enumerate(units):
        if not isinstance(unit, dict):
            return None  # malformed units.json -> keep today's script-row path
        try:
            idx = int(unit.get("index", position))
        except (TypeError, ValueError):
            idx = position
        start, end = _float_field(unit, "start"), _float_field(unit, "end")
        if idx in placed:
            start, end = placed[idx]
        subtitle_text = str(unit.get("subtitle_vi", "")).strip()
        if subtitle_text:
            vi_text = selected_text(unit, "subtitle_vi")
        else:
            vi_text = timed_texts.get(idx, selected_text(unit, "subtitle_vi"))
        row: dict = {
            "start": float(start),
            "end": float(end),
            "vi": vi_text,
        }
        if idx in align_words:
            row["align_words"] = align_words[idx]
        rows.append(row)
    rows.sort(key=lambda row: row["start"])
    return rows


def write_subtitles_srt(script: list[dict], path: str | Path) -> Path:
    """Write the Vietnamese subtitle track (SRT) from the translated script."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    subs = pysubs2.SSAFile()
    for record in script:
        subs.append(
            pysubs2.SSAEvent(
                start=pysubs2.make_time(s=float(record.get("start", 0.0))),
                end=pysubs2.make_time(s=float(record.get("end", 0.0))),
                text=str(record.get("vi", "")),
            )
        )
    subs.save(str(target), format="srt")
    return target


def write_subtitles_ass(
    rows: list[dict],
    path: str | Path,
    *,
    position: str = "bottom",
    font_size: int = 24,
    margin_v: int = 24,
    font_name: str = DEFAULT_ASS_FONT,
) -> Path:
    """Write the Vietnamese subtitle track as ASS with one explicit style (SU-1-T2).

    Emits a single ``ViDub`` style the events reference by name, carrying:

    * ``alignment`` from ``position`` (:data:`SUB_POSITION_ALIGNMENT` —
      2 bottom-center, 8 top-center, 5 middle-center);
    * ``fontsize`` and ``marginv`` (the requested size and vertical margin);
    * ``bold=False`` (regular weight reads best over video and matches the SRT
      look), ``outline=2`` + ``shadow=1`` (the ASS defaults) so the white text
      keeps a black border and a soft drop shadow on any background;
    * ``fontname`` = ``font_name`` (generic family by default; if that family is
      not installed, the ASS renderer falls back to its default sans-serif).

    The rows are ``{start, end, vi}`` like :func:`write_subtitles_srt`; the SRT
    writer is untouched, so the default options keep today's SRT byte-identically.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    alignment = SUB_POSITION_ALIGNMENT[normalize_sub_position(position)]
    subs = pysubs2.SSAFile()
    subs.styles[ASS_STYLE_NAME] = pysubs2.SSAStyle(
        fontname=font_name,
        fontsize=int(font_size),
        bold=False,
        outline=2,
        shadow=1,
        # pysubs2 1.8 wants the Alignment enum (an IntEnum), not a bare int.
        alignment=pysubs2.Alignment(alignment),
        marginv=int(margin_v),
    )
    for record in rows:
        subs.append(
            pysubs2.SSAEvent(
                start=pysubs2.make_time(s=float(record.get("start", 0.0))),
                end=pysubs2.make_time(s=float(record.get("end", 0.0))),
                text=str(record.get("vi", "")),
                style=ASS_STYLE_NAME,
            )
        )
    subs.save(str(target), format="ass")
    return target


def write_script_tsv(script: list[dict], path: str | Path) -> Path:
    """Write ``start \\t end \\t speaker \\t en \\t vi`` lines (no header)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{record.get('start', 0.0)}\t{record.get('end', 0.0)}\t"
        f"{record.get('speaker', 'spk0')}\t"
        f"{record.get('en', '')}\t{record.get('vi', '')}"
        for record in script
    ]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target


def _filter_path(path: str | Path) -> str:
    """Escape a path for use inside an ffmpeg filter argument."""
    return (
        Path(path)
        .as_posix()
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
    )


def force_style_string(position: str, font_size: int, margin_v: int) -> str:
    """The ASS style-override list handed to ffmpeg's ``force_style`` (SU-1-T2).

    ``Alignment`` uses :data:`SUB_POSITION_LEGACY_ALIGNMENT` (2 bottom, 6 top,
    10 center) because the SRT-fed ``subtitles`` filter applies ``force_style``
    with the legacy SSA numbering; see that map. The values are safe (validated
    position + ints), so no escaping is needed here.
    """
    alignment = SUB_POSITION_LEGACY_ALIGNMENT[normalize_sub_position(position)]
    return f"Alignment={alignment},FontSize={int(font_size)},MarginV={int(margin_v)}"


def burn_subtitles_filter(
    srt_path: str | Path,
    position: str = "bottom",
    font_size: int = 24,
    margin_v: int = 24,
) -> str:
    """Value of the ffmpeg ``-vf`` ``subtitles`` filter for a styled burn (SU-1-T2).

    The style list is single-quoted **inside the filtergraph** so its commas are
    not parsed as filter separators (the quotes reach ffmpeg's parser verbatim
    because the command is an argv list, never a shell); ``_filter_path`` escapes
    the ``:`` of a Windows path. Result:
    ``subtitles=<escaped path>:force_style='Alignment=<n>,FontSize=<n>,MarginV=<n>'``.
    """
    style = force_style_string(position, font_size, margin_v)
    return f"subtitles={_filter_path(srt_path)}:force_style='{style}'"


def build_mux_command(
    video: Path,
    dub_track: Path | None,
    srt_path: Path | None,
    out_path: Path,
    mode: str,
    burn: bool = False,
    sub_position: str = "bottom",
    sub_font_size: int = 24,
    sub_margin_v: int = 24,
) -> list[str]:
    """Build the ffmpeg (argument) command that produces the export video.

    ``burn=False`` (the default) is untouched: a soft ``mov_text`` subtitle
    stream is muxed for non-subtitles modes. With ``burn=True`` the subtitles are
    rendered into the video and the requested style is applied through
    ``force_style`` (see :func:`burn_subtitles_filter`).
    """
    use_dub = mode != SUBTITLES_ONLY and dub_track is not None
    has_srt = srt_path is not None and Path(srt_path).exists()

    args: list[str] = ["-y", "-i", str(video)]
    maps = ["-map", "0:v:0"]
    if use_dub:
        args += ["-i", str(dub_track)]
        maps += ["-map", "1:a:0"]
    else:
        maps += ["-map", "0:a:0?"]

    soft_subs = has_srt and not burn
    if soft_subs:
        args += ["-i", str(srt_path)]
        maps += ["-map", f"{2 if use_dub else 1}:s:0"]

    args += maps

    if burn:
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        if has_srt:
            args += [
                "-vf",
                burn_subtitles_filter(
                    srt_path, sub_position, sub_font_size, sub_margin_v
                ),
            ]
    else:
        args += ["-c:v", "copy"]

    if use_dub:
        args += ["-c:a", "aac", "-b:a", "192k"]
    else:
        args += ["-c:a", "copy"]

    if soft_subs:
        # MP4 stores subtitle language as an ISO 639-2 code; this ffmpeg build
        # drops a 2-letter "vi" and emits no tag at all, so use "vie".
        args += ["-c:s", "mov_text", "-metadata:s:s:0", "language=vie"]

    args.append(str(out_path))
    return args


def export_signature(cfg: JobConfig, burn: bool = False) -> str:
    """Changing export policy/profile only invalidates the export stage."""
    return "|".join(("sub=bounded-reading-v1", "audio=placed-v1", "pace=combined-v1",
                     f"subqc={int(cfg.qc_subtitles)}", f"finalqc={int(cfg.qc_final)}",
                     f"position={cfg.sub_position}", f"font={cfg.sub_font_size}",
                     f"margin={cfg.sub_margin_v}", f"burn={int(burn)}"))


def export(
    manifest_path: str,
    cfg: JobConfig,
    burn: bool = False,
    timings: dict[str, float] | None = None,
) -> JobManifest:
    """Produce the export outputs for the job described by ``manifest_path``."""
    started = time.perf_counter()
    manifest = JobManifest.load(manifest_path)
    artifacts_dir = manifest.artifacts_dir
    output_dir = output_dir_for(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = artifacts_dir / manifest.paths.get("script", SCRIPT_NAME)
    if not script_path.exists():
        raise FileNotFoundError(
            f"script not found: {script_path}. Run the translate stage first, e.g. "
            f"'python -m vidub.stages.translate {manifest_path}'."
        )
    script = json.loads(script_path.read_text(encoding="utf-8"))

    video = Path(manifest.input_path)
    if not video.exists():
        raise FileNotFoundError(f"input video not found: {video}")

    # Subtitle standards QC applies ONLY to the subtitle track: the SRT rows may
    # be wrapped/re-split for readability, while script.json, script.tsv, the
    # speech audio and the timing are left exactly as produced upstream.
    # SA-1-T4: on the sentence-unit path the cues are ONE PER SENTENCE UNIT (times
    # from timed.json's placed unit row, text = the unit's ``vi``); when
    # ``units.json`` is absent (legacy --no-sentence-units runs and pre-SA-1
    # caches) the rows are today's script rows, byte-identically.
    unit_rows = _unit_subtitle_rows(manifest)
    subtitle_source = (
        unit_rows
        if unit_rows is not None
        else [
            {**row, "vi": selected_text(row, "subtitle_vi")}
            for row in script
        ]
    )
    subtitle_rows, subtitles_qc = qc_rows(
        subtitle_source, cfg.qc_subtitles, video_duration=float(manifest.duration or 0.0) or None
    )
    (artifacts_dir / SUBTITLES_QC_NAME).write_text(
        json.dumps(subtitles_qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest.paths["subtitles_qc"] = SUBTITLES_QC_NAME

    # A speech-less job (0 segments) produces an empty SRT. Handing that to
    # ffmpeg as a subtitle input makes it reject the whole mux, so drop the
    # subtitle stream entirely and say so in the report instead.
    has_speech = bool(script)
    subtitles_path = write_subtitles_srt(subtitle_rows, output_dir / SUBTITLES_NAME)
    # SU-1-T2: also write the styled ASS twin (same rows, ViDub style). The SRT
    # writer is untouched, so default options keep the SRT byte-identical.
    write_subtitles_ass(
        subtitle_rows,
        output_dir / SUBTITLES_ASS_NAME,
        position=cfg.sub_position,
        font_size=cfg.sub_font_size,
        margin_v=cfg.sub_margin_v or 0,
    )
    write_script_tsv(script, output_dir / SCRIPT_TSV_NAME)
    if has_speech:
        subtitles_note = "burned-in" if burn else "soft (mov_text)"
        srt_for_mux: Path | None = subtitles_path
    else:
        subtitles_note = NO_SPEECH_SUBTITLES
        srt_for_mux = None
        log.info("no speech segments: exporting without a subtitle stream")

    dub_track: Path | None = None
    if manifest.mode != SUBTITLES_ONLY:
        dub_track = artifacts_dir / manifest.paths.get("dub_track", DUB_TRACK_NAME)
        if not dub_track.exists() or dub_track.stat().st_size == 0:
            raise FileNotFoundError(
                f"dubbed track not found or empty: {dub_track}. Run the mix stage "
                f"first, e.g. 'python -m vidub.stages.mix {manifest_path}'."
            )

    dubbed_path = output_dir / DUBBED_NAME
    command = build_mux_command(
        video,
        dub_track,
        srt_for_mux,
        dubbed_path,
        manifest.mode,
        burn,
        sub_position=cfg.sub_position,
        sub_font_size=cfg.sub_font_size,
        sub_margin_v=cfg.sub_margin_v or 0,
    )
    log.info("ffmpeg %s", " ".join(command))
    run_ffmpeg(command)

    manifest.paths["dubbed"] = DUBBED_NAME
    manifest.paths["subtitles_srt"] = SUBTITLES_NAME
    manifest.paths["subtitles_ass"] = SUBTITLES_ASS_NAME
    manifest.paths["script_tsv"] = SCRIPT_TSV_NAME
    manifest.paths["output_dir"] = output_dir.as_posix()
    manifest.stages["export"] = "done"
    manifest.export_signature = export_signature(cfg, burn)
    manifest.save(artifacts_dir / "job.json")

    merged_timings = dict(timings or {})
    merged_timings["export"] = time.perf_counter() - started
    report = build_report(manifest, merged_timings, subtitles=subtitles_note)
    # Final QC gate (QUALITY-ROADMAP.md §3.4): aggregate the per-level verdicts
    # and fresh output checks into one ``quality`` block. Disabled -> a minimal
    # block so report.json keeps a stable shape.
    report["quality"] = run_gate(manifest, cfg) if cfg.qc_final else {"enabled": False}
    (output_dir / REPORT_NAME).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m vidub.stages.export <job.json>``."""
    parser = argparse.ArgumentParser(
        prog="vidub.stages.export",
        description="S6 export: mux dubbed audio + subtitles into the final video.",
    )
    parser.add_argument("job", help="path to the job.json manifest")
    parser.add_argument("--burn", action="store_true", help="burn subtitles in")
    parser.add_argument(
        "--cache-dir", default=None, help="cache root (default: manifest's)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = JobConfig()
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir

    try:
        manifest = export(args.job, cfg, burn=args.burn)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(str(output_dir_for(manifest) / DUBBED_NAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

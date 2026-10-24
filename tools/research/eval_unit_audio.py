"""Unit-level audio (UA-1) vs per-segment timing: reference A/B evaluation (UA-1-T2).

Re-runs the **complaint video** the user's silent-gap report was measured on —
``.vidub-cache/downloads/youtube-5sLYAQS9sWQ.mkv`` (job hash ``1334fb56…``) —
twice into separate scratch caches (never the User's ``.vidub-cache/`` jobs):

* **BEFORE** = ``sentence_units=False`` — v0.7.0 per-segment timing (one placed
  row per ASR segment, so an intra-sentence pause can fall between two rows);
* **AFTER**  = defaults — UA-1 unit audio (one synthesis + one placed row per
  sentence unit, so no placed-row boundary can fall inside a sentence).

Both runs use **envit5 + Piper**, QC on, refine off, forcing
``translate,speak,timing,mix,export`` so the seeded cache only supplies
ingest/transcribe (and the reused Demucs stems). Everything is written under the
scratch roots ``.vidub-cache-ua1/`` (caches + per-run work dirs) and
``eval-cache/ua1/`` (scores + run registry); the User's caches are only read.

Pre-registered metrics (declared before reading the numbers, see
``docs/unit-audio.md``):

1. placed-row boundaries **inside a sentence unit** — AFTER must be **0**
   (structural);
2. **mid-word split** count (a placed boundary that cuts a word in half, e.g.
   ``…đào || tạo…``) — target **0**;
3. **clips** — target **0**;
4. placed-boundary **silent gaps** > 0.8 s / > 0.25 s (the user's "silent gaps"
   metric) — reduced vs BEFORE;
5. per-unit per-synthesis **rate spread** — reported honestly.

Usage::

    ./.venv/Scripts/python.exe tools/research/eval_unit_audio.py run
    ./.venv/Scripts/python.exe tools/research/eval_unit_audio.py score
    ./.venv/Scripts/python.exe tools/research/eval_unit_audio.py rows
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The complaint video (job 1334fb56…): the User's job with 63 rows -> 49 units
# whose per-segment timing carried the "…đào || tạo…" mid-word split and the
# worst 6.72 s silent gap.
INPUT = ROOT / ".vidub-cache" / "downloads" / "youtube-5sLYAQS9sWQ.mkv"
REFERENCE_CACHE = ROOT / ".vidub-cache"
SCRATCH = ROOT / ".vidub-cache-ua1"
SCORES_DIR = ROOT / "eval-cache" / "ua1"
SCORES_PATH = SCORES_DIR / "scores.json"
RUNS_PATH = SCORES_DIR / "runs.json"

PHASES = ("before", "after")
SENTENCE_UNITS = {"before": False, "after": True}
FORCE_STAGES = ["translate", "speak", "timing", "mix", "export"]

# The two documented live mid-word examples from the User's jobs:
#   (job hash, left segment index, right segment index, label)
MID_WORD_EXAMPLES = (
    ("1334fb560d2783dee7cf99b287726cea4ac1c8d8", 10, 11, "1334fb56… rows 10/11"),
    ("14415df27f201efcef03d79260aa6d6d6d8c9652", 50, 51, "14415df2… rows 50/51"),
)

# Boundary tolerance (s) when matching a placed-row boundary to a member split.
BOUNDARY_TOL = 0.02


# ------------------------------------------------------------------ file helpers
def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _reference_hash() -> str:
    from vidub.stages.ingest import video_hash

    return video_hash(INPUT)


def _artifacts_dir(cache_dir: Path, video_hash: str) -> Path:
    return cache_dir / video_hash


def _work_dir(key: str) -> Path:
    return SCRATCH / f"work-{key}"


def _report_path(key: str) -> Path:
    return _work_dir(key) / "output" / INPUT.stem / "report.json"


# ------------------------------------------------------------- run subcommand
def _seed_cache(cache_dir: Path, video_hash: str) -> None:
    """Copy the reference job into a fresh scratch cache (read-only on source)."""
    source = _artifacts_dir(REFERENCE_CACHE, video_hash)
    if not source.is_dir():
        raise FileNotFoundError(f"reference job not found: {source}")
    target = _artifacts_dir(cache_dir, video_hash)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def _run_phase(phase: str, video_hash: str) -> dict[str, Any]:
    from vidub.config import JobConfig
    from vidub.pipeline import run_pipeline

    key = phase
    cache_dir = SCRATCH / f"cache-{key}"
    work = _work_dir(key)
    work.mkdir(parents=True, exist_ok=True)

    _seed_cache(cache_dir, video_hash)
    # envit5 + Piper, QC on, refine off; only sentence_units differs per phase.
    cfg = JobConfig(
        cache_dir=str(cache_dir),
        mt_engine="envit5",
        tts_engine="piper",
        sentence_units=SENTENCE_UNITS[phase],
        hunyuan_refine=False,
    )

    cwd = Path.cwd()
    os.chdir(work)
    try:
        run_pipeline(str(INPUT), cfg, resume=True, force=list(FORCE_STAGES))
    finally:
        os.chdir(cwd)

    return {
        "key": key,
        "phase": phase,
        "sentence_units": SENTENCE_UNITS[phase],
        "mt_engine": "envit5",
        "tts_engine": "piper",
        "cache_dir": str(cache_dir),
        "work_dir": str(work),
        "status": "ok",
    }


def cmd_run(args: argparse.Namespace) -> int:
    if not INPUT.exists():
        print(f"reference video not found: {INPUT}", file=sys.stderr)
        return 1
    video_hash = _reference_hash()
    runs = _load_runs()
    for phase in PHASES:
        print(f"=== running {phase} (sentence_units={SENTENCE_UNITS[phase]}) ===")
        try:
            entry = _run_phase(phase, video_hash)
        except Exception as exc:  # noqa: BLE001 - record and continue honestly
            entry = {
                "key": phase,
                "phase": phase,
                "sentence_units": SENTENCE_UNITS[phase],
                "status": f"error ({exc})",
            }
            print(f"{phase}: ERROR {exc}")
        else:
            print(f"{phase}: ok -> {entry['cache_dir']}")
        runs[phase] = entry
        _dump_json(RUNS_PATH, runs)
    print(f"runs written to {RUNS_PATH}")
    return 0


def _load_runs() -> dict[str, Any]:
    if RUNS_PATH.exists():
        return _load_json(RUNS_PATH)
    return {}


# ----------------------------------------------------------- scoring helpers
def _units_from_artifacts(artifacts: Path) -> tuple[list[Any], list[dict]]:
    """The sentence units the timing stage grouped (mirrors its ``group_units``)."""
    from vidub.utils.sentences import group_units

    segments = _load_json(artifacts / "segments.json")
    script = _load_json(artifacts / "script.json")
    passthrough = {
        index
        for index, row in enumerate(script)
        if isinstance(row, dict) and row.get("passthrough")
    }
    rows = []
    for index, seg in enumerate(segments):
        speaker = seg.get("speaker", "spk0")
        if index in passthrough:
            # Mirror translate/timing: a unique speaker isolates a passthrough row.
            speaker = f"__passthrough_{index}__"
        rows.append(
            {
                "text": str(seg.get("text", "")),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "speaker": speaker,
            }
        )
    return group_units(rows), script


def _ordered_timed(artifacts: Path) -> list[dict]:
    timed = _load_json(artifacts / "timed.json")
    return sorted(timed, key=lambda row: float(row.get("start", 0.0)))


def _boundary_times(ordered: list[dict]) -> list[float]:
    return [float(ordered[i + 1].get("start", 0.0)) for i in range(len(ordered) - 1)]


def _row_words(row: dict, script: list[dict]) -> int:
    """Vietnamese word count of a placed row (its members on the unit path)."""
    members = row.get("members")
    if isinstance(members, list) and members:
        total = 0
        for member in members:
            try:
                index = int(member)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(script):
                total += len(str(script[index].get("vi", "")).split())
        return total
    try:
        index = int(row.get("idx", 0))
    except (TypeError, ValueError):
        return 0
    if 0 <= index < len(script):
        return len(str(script[index].get("vi", "")).split())
    return 0


def _rate_stats(ordered: list[dict], script: list[dict]) -> dict[str, Any]:
    rates = []
    for row in ordered:
        placed = float(row.get("end", 0.0)) - float(row.get("start", 0.0))
        words = _row_words(row, script)
        if placed > 0:
            rates.append(words / placed)
    if not rates:
        return {"min": None, "median": None, "max": None, "p10": None, "p90": None}
    rates_sorted = sorted(rates)
    n = len(rates_sorted)
    return {
        "min": round(rates_sorted[0], 3),
        "median": round(statistics.median(rates_sorted), 3),
        "max": round(rates_sorted[-1], 3),
        "p10": round(rates_sorted[int(0.1 * (n - 1))], 3),
        "p90": round(rates_sorted[int(0.9 * (n - 1))], 3),
    }


def _mid_word_splits(
    ordered: list[dict], units: list[Any], script: list[dict]
) -> tuple[int, list[dict]]:
    """Placed boundaries that cut a word in half (e.g. ``…đào || tạo…``).

    A member boundary inside a unit is mid-word when the left ``vi`` ends on a
    letter and the right ``vi`` starts on a letter; it counts only when a placed
    row boundary actually sits at that member boundary. AFTER has no such
    boundary (one row per unit), so it is 0 by construction.
    """
    boundaries = _boundary_times(ordered)
    count = 0
    examples: list[dict] = []
    for unit in units:
        indices = list(unit.indices)
        for left_index, right_index in zip(indices, indices[1:]):
            left = str(script[left_index].get("vi", ""))
            right = str(script[right_index].get("vi", ""))
            if not (left and right):
                continue
            if not (left[-1].isalpha() and right[0].isalpha()):
                continue
            at = float(script[right_index].get("start", 0.0))
            if any(abs(boundary - at) <= BOUNDARY_TOL for boundary in boundaries):
                count += 1
                examples.append(
                    {
                        "at": round(at, 3),
                        "members": [left_index, right_index],
                        "left_tail": left[-24:],
                        "right_head": right[:24],
                    }
                )
    return count, examples


def _intra_unit_boundaries(ordered: list[dict], units: list[Any]) -> int:
    """Placed-row boundaries that fall *inside* one sentence unit (AFTER target 0).

    A boundary is intra-unit when two consecutive placed rows belong to the same
    sentence unit. AFTER places one row per unit, so this is 0 by construction
    (and a real check: two rows for one unit would make it non-zero). BEFORE
    places one row per segment, so every unit with >1 member contributes its
    internal boundaries. Recentering does not affect this: membership — not
    timing — decides.
    """
    segment_to_unit: dict[int, int] = {}
    for unit_index, unit in enumerate(units):
        for member in unit.indices:
            segment_to_unit[int(member)] = unit_index

    def row_unit(row: dict) -> int | None:
        members = row.get("members")
        if isinstance(members, list) and members:
            for member in members:
                try:
                    found = segment_to_unit.get(int(member))
                except (TypeError, ValueError):
                    continue
                if found is not None:
                    return found
            return None
        try:
            return segment_to_unit.get(int(row.get("idx", -1)))
        except (TypeError, ValueError):
            return None

    count = 0
    for index in range(len(ordered) - 1):
        current = row_unit(ordered[index])
        following = row_unit(ordered[index + 1])
        if current is not None and current == following:
            count += 1
    return count


def _phase_metrics(key: str, run: dict[str, Any], video_hash: str) -> dict[str, Any]:
    artifacts = _artifacts_dir(Path(run["cache_dir"]), video_hash)
    sentence_units = bool(run["sentence_units"])

    units, script = _units_from_artifacts(artifacts)
    ordered = _ordered_timed(artifacts)

    metrics: dict[str, Any] = {
        "key": key,
        "sentence_units": sentence_units,
        "rows": len(ordered),
        "units": len(units),
        "placed_boundaries": max(len(ordered) - 1, 0),
        "rows_with_members": sum(
            1 for row in ordered if isinstance(row.get("members"), list)
        ),
    }

    # 1. structural: no placed boundary may fall inside a sentence unit
    metrics["intra_unit_boundaries"] = _intra_unit_boundaries(ordered, units)

    # 2. silent gaps between consecutive placed rows (the user's metric)
    gaps = []
    for index in range(len(ordered) - 1):
        gap = float(ordered[index + 1].get("start", 0.0)) - float(
            ordered[index].get("end", 0.0)
        )
        gaps.append(
            {
                "at": round(float(ordered[index + 1].get("start", 0.0)), 3),
                "gap": round(gap, 3),
                "from_idx": ordered[index].get("idx"),
                "to_idx": ordered[index + 1].get("idx"),
            }
        )
    positive = [item["gap"] for item in gaps if item["gap"] > 0]
    metrics["gaps_gt_0_25s"] = sum(1 for gap in positive if gap > 0.25)
    metrics["gaps_gt_0_8s"] = sum(1 for gap in positive if gap > 0.8)
    metrics["max_gap"] = round(max(positive), 3) if positive else 0.0
    metrics["worst_gaps"] = sorted(gaps, key=lambda item: -item["gap"])[:5]

    # 3. mid-word splits
    mid_word, examples = _mid_word_splits(ordered, units, script)
    metrics["mid_word_splits"] = mid_word
    metrics["mid_word_examples"] = examples

    # 4. clips + stretch distribution
    actions = Counter(str(row.get("action", "")) for row in ordered)
    metrics["clips"] = sum(1 for row in ordered if row.get("clipped"))
    metrics["actions"] = {
        "asis": actions.get("asis", 0),
        "stretch": actions.get("stretch", 0),
        "clip": actions.get("clip", 0),
    }
    stretches = [
        float(row.get("stretch", 1.0)) for row in ordered if row.get("stretch")
    ]
    metrics["max_stretch"] = round(max(stretches), 3) if stretches else 1.0

    # 5. per-row rate spread (words per placed second)
    metrics["rate_words_per_second"] = _rate_stats(ordered, script)

    # 6. AFTER unit rate fit + recenter summary
    timing = (
        _load_json(artifacts / "timing.json")
        if (artifacts / "timing.json").exists()
        else {}
    )
    metrics["rate_fit"] = timing.get("rate_fit")
    rate_actions = Counter(
        str(row.get("rate_action")) for row in ordered if "rate_action" in row
    )
    metrics["rate_actions"] = dict(rate_actions)
    metrics["recentered_rows"] = sum(1 for row in ordered if row.get("recentered"))

    # 7. TTS QC + aggregate quality verdict
    tts_qc = (
        _load_json(artifacts / "tts_qc.json")
        if (artifacts / "tts_qc.json").exists()
        else {}
    )
    metrics["tts_qc"] = {
        "scored": tts_qc.get("scored"),
        "low_mos": tts_qc.get("low_mos"),
        "retried": tts_qc.get("retried"),
        "resolved": tts_qc.get("resolved"),
        "skipped": tts_qc.get("skipped"),
        "mos_mean": tts_qc.get("mos_mean"),
    }
    report_path = _report_path(key)
    report = _load_json(report_path) if report_path.exists() else {}
    metrics["quality_verdict"] = (report.get("quality") or {}).get("verdict")

    return metrics


def _fmt(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _print_table(scores: dict[str, Any]) -> None:
    before = scores.get("before", {})
    after = scores.get("after", {})
    print(f"\n{'metric':<34}{'BEFORE':<24}{'AFTER':<24}")
    rows = [
        ("rows", before.get("rows"), after.get("rows")),
        ("sentence units", before.get("units"), after.get("units")),
        ("placed boundaries", before.get("placed_boundaries"), after.get("placed_boundaries")),
        (
            "intra-unit boundaries (target 0)",
            before.get("intra_unit_boundaries"),
            after.get("intra_unit_boundaries"),
        ),
        ("mid-word splits (target 0)", before.get("mid_word_splits"), after.get("mid_word_splits")),
        ("clips (target 0)", before.get("clips"), after.get("clips")),
        ("gaps > 0.25 s", before.get("gaps_gt_0_25s"), after.get("gaps_gt_0_25s")),
        ("gaps > 0.8 s", before.get("gaps_gt_0_8s"), after.get("gaps_gt_0_8s")),
        ("worst gap (s)", before.get("max_gap"), after.get("max_gap")),
        (
            "rate w/s median",
            (before.get("rate_words_per_second") or {}).get("median"),
            (after.get("rate_words_per_second") or {}).get("median"),
        ),
        (
            "rate w/s min..max",
            (
                f"{(before.get('rate_words_per_second') or {}).get('min')}"
                f"..{(before.get('rate_words_per_second') or {}).get('max')}"
            ),
            (
                f"{(after.get('rate_words_per_second') or {}).get('min')}"
                f"..{(after.get('rate_words_per_second') or {}).get('max')}"
            ),
        ),
        ("max stretch", before.get("max_stretch"), after.get("max_stretch")),
        ("rows with members", before.get("rows_with_members"), after.get("rows_with_members")),
        ("tts_qc mos_mean", (before.get("tts_qc") or {}).get("mos_mean"), (after.get("tts_qc") or {}).get("mos_mean")),
        ("quality verdict", before.get("quality_verdict"), after.get("quality_verdict")),
    ]
    for name, left, right in rows:
        print(f"{name:<34}{str(left):<24}{str(right):<24}")
    print("\nAFTER worst gaps:", _fmt(after.get("worst_gaps")))
    print("BEFORE worst gaps:", _fmt(before.get("worst_gaps")))


def cmd_score(args: argparse.Namespace) -> int:
    runs = _load_runs()
    if not runs:
        print(f"no runs recorded ({RUNS_PATH}); run 'run' first", file=sys.stderr)
        return 1

    video_hash = _reference_hash()
    payload: dict[str, Any] = {
        "input": str(INPUT),
        "video_hash": video_hash,
        "phases": {},
    }
    for phase in PHASES:
        run = runs.get(phase)
        if run is None:
            payload[phase] = {"status": "missing"}
        elif run.get("status") != "ok":
            payload[phase] = {"status": run.get("status")}
        else:
            payload[phase] = _phase_metrics(phase, run, video_hash)

    before = payload.get("before", {})
    after = payload.get("after", {})
    payload["targets"] = {
        "intra_unit_boundaries_zero": after.get("intra_unit_boundaries") == 0,
        "mid_word_splits_zero": after.get("mid_word_splits") == 0,
        "clips_zero": after.get("clips") == 0,
        "gaps_gt_0_8s_reduced": (
            before.get("gaps_gt_0_8s") is not None
            and after.get("gaps_gt_0_8s") is not None
            and after["gaps_gt_0_8s"] < before["gaps_gt_0_8s"]
        ),
        "gap_reduction_gt_0_8s": (
            (before.get("gaps_gt_0_8s") or 0) - (after.get("gaps_gt_0_8s") or 0)
        ),
        "gap_reduction_gt_0_25s": (
            (before.get("gaps_gt_0_25s") or 0) - (after.get("gaps_gt_0_25s") or 0)
        ),
    }
    _dump_json(SCORES_PATH, payload)

    if before and after:
        _print_table(payload)
    print(f"\ntargets: {_fmt(payload['targets'])}")
    print(f"scores written to {SCORES_PATH}")
    return 0


# --------------------------------------------------------------- rows command
def _unit_rows_for_job(job_hash: str) -> tuple[list[Any], list[dict], list[dict]]:
    """``(units, script, timed)`` for a User job cache (read-only)."""
    from vidub.utils.sentences import group_units

    artifacts = REFERENCE_CACHE / job_hash
    script = _load_json(artifacts / "script.json")
    timed = _load_json(artifacts / "timed.json")
    rows = [
        {
            "text": str(row.get("en", "")),
            "start": float(row.get("start", 0.0)),
            "end": float(row.get("end", 0.0)),
            "speaker": row.get("speaker", "spk0"),
        }
        for row in script
    ]
    return group_units(rows), script, timed


def cmd_rows(args: argparse.Namespace) -> int:
    for job_hash, left_idx, right_idx, label in MID_WORD_EXAMPLES:
        if not (REFERENCE_CACHE / job_hash).is_dir():
            print(f"{label}: cache not found, skipped\n")
            continue
        units, script, timed = _unit_rows_for_job(job_hash)
        left = script[left_idx]
        right = script[right_idx]

        ordered = sorted(timed, key=lambda row: float(row.get("start", 0.0)))
        by_idx = {int(row.get("idx", -1)): row for row in ordered}
        timed_left = by_idx.get(left_idx)
        timed_right = by_idx.get(right_idx)
        if timed_left and timed_right:
            placed_gap = float(timed_right.get("start", 0.0)) - float(
                timed_left.get("end", 0.0)
            )
        else:
            placed_gap = None

        unit = next(
            (
                candidate
                for candidate in units
                if left_idx in candidate.indices and right_idx in candidate.indices
            ),
            None,
        )
        joined = ""
        members: list[int] = []
        if unit is not None:
            members = list(unit.indices)
            joined = " ".join(
                piece
                for piece in (
                    " ".join(str(script[index].get("vi", "")).split())
                    for index in members
                )
                if piece
            )

        print(f"=== {label} ===")
        print(f"EN     : {str(left.get('en', '')).strip()} || "
              f"{str(right.get('en', '')).strip()}")
        print(f"BEFORE : vi[{left_idx}]={left.get('vi')!r}")
        print(f"         vi[{right_idx}]={right.get('vi')!r}")
        print(f"         placed gap at the split: "
              f"{placed_gap:.3f} s" if placed_gap is not None else "         placed gap: n/a")
        if unit is None:
            print("AFTER  : the two rows are NOT in one unit (unexpected)\n")
        else:
            contiguous = "đào tạo" in joined
            print(f"AFTER  : one unit {members} -> synthesis text:")
            print(f"         {joined!r}")
            print(f"         contains 'đào tạo' contiguously: {contiguous}")
            print()
    return 0


# ---------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_unit_audio",
        description="A/B evaluate unit-level audio (UA-1) vs per-segment timing.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run BEFORE/AFTER into scratch caches")
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="score both caches; write scores.json")
    score.set_defaults(func=cmd_score)

    rows = sub.add_parser("rows", help="the two documented mid-word examples")
    rows.set_defaults(func=cmd_rows)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

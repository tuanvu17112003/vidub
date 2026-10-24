"""Two-pass TTS rate fitting vs the previous behaviour: reference A/B evaluation (FIT-1-T3).

Runs the reference video (``.vidub-cache/downloads/youtube-a7TOameRqoY.mkv``)
twice — BEFORE = ``tts_rate_fit=False, recenter_audio=False`` (the previous
behaviour) and AFTER = the FIT-1 defaults — into **separate scratch cache dirs**
(never the User's ``.vidub-cache/`` jobs) and scores the pre-registered metrics
from ``timed.json`` + ``script.json`` + ``timing.json``:

1. clips (``action == "clip"``) — target 0;
2. atempo-stretched count (``action == "stretch"``) + stretch distribution;
3. per-segment perceived rate = vi words / placed seconds (min/max/mean/std/CV);
4. unused window seconds (``window - placed``): total / rows > 1 s / max — target
   a >= 50 % reduction;
5. the AFTER ``timing.json`` ``rate_fit`` block + the ``length_scale``
   distribution.

Usage::

    ./.venv/Scripts/python.exe scripts/eval_timing_fit.py run
    ./.venv/Scripts/python.exe scripts/eval_timing_fit.py score
    ./.venv/Scripts/python.exe scripts/eval_timing_fit.py rows

Everything is written under the scratch roots ``.vidub-cache-fit1/`` (caches + the
per-run output dirs) and ``eval-cache/fit1/`` (scores + run registry); the User's
``.vidub-cache/`` and ``output/`` are only read (the reference cache is *copied*
in, and each run chdirs into its own scratch work dir so ``output/`` lands there).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUT = ROOT / ".vidub-cache" / "downloads" / "youtube-a7TOameRqoY.mkv"
REFERENCE_CACHE = ROOT / ".vidub-cache"
SCRATCH = ROOT / ".vidub-cache-fit1"
SCORES_DIR = ROOT / "eval-cache" / "fit1"
SCORES_PATH = SCORES_DIR / "scores.json"
RUNS_PATH = SCORES_DIR / "runs.json"

PHASES = ("before", "after")
# BEFORE is the previous behaviour; AFTER is the FIT-1 default (both flags on).
RATE_FIT = {"before": False, "after": True}
RECENTER = {"before": False, "after": True}
MT_ENGINE = "envit5"
FORCE_STAGES = ["translate", "speak", "timing", "mix", "export"]

# The clipped segment indices from the User's job (`53a7f0d6…`, 75 rows), kept as
# the pre-registered "worst rows" set for the side-by-side table.
USER_CLIPPED = (19, 24, 35, 44, 49, 65)


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


def _work_dir(phase: str) -> Path:
    return SCRATCH / f"work-{phase}"


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

    cache_dir = SCRATCH / f"cache-{phase}"
    work = _work_dir(phase)
    work.mkdir(parents=True, exist_ok=True)

    _seed_cache(cache_dir, video_hash)
    cfg = JobConfig(
        cache_dir=str(cache_dir),
        mt_engine=MT_ENGINE,
        tts_rate_fit=RATE_FIT[phase],
        recenter_audio=RECENTER[phase],
        hunyuan_refine=False,
    )

    cwd = Path.cwd()
    os.chdir(work)
    try:
        run_pipeline(str(INPUT), cfg, resume=True, force=list(FORCE_STAGES))
    finally:
        os.chdir(cwd)

    return {
        "key": phase,
        "phase": phase,
        "mt_engine": MT_ENGINE,
        "tts_rate_fit": RATE_FIT[phase],
        "recenter_audio": RECENTER[phase],
        "cache_dir": str(cache_dir),
        "work_dir": str(work),
        "status": "ok",
    }


def _load_runs() -> dict[str, Any]:
    if RUNS_PATH.exists():
        return _load_json(RUNS_PATH)
    return {}


def cmd_run(args: argparse.Namespace) -> int:
    video_hash = _reference_hash()
    runs = _load_runs()
    for phase in PHASES:
        print(
            f"=== running {phase} "
            f"(tts_rate_fit={RATE_FIT[phase]}, recenter_audio={RECENTER[phase]}) ==="
        )
        try:
            entry = _run_phase(phase, video_hash)
        except Exception as exc:  # noqa: BLE001 - record and continue honestly
            entry = {
                "key": phase,
                "phase": phase,
                "mt_engine": MT_ENGINE,
                "tts_rate_fit": RATE_FIT[phase],
                "recenter_audio": RECENTER[phase],
                "status": f"error ({exc})",
            }
            print(f"{phase}: ERROR {exc}")
        else:
            print(f"{phase}: ok -> {entry['cache_dir']}")
        runs[phase] = entry
        _dump_json(RUNS_PATH, runs)

    print(f"runs written to {RUNS_PATH}")
    return 0


# ----------------------------------------------------------- scoring helpers
def _stats(values: list[float]) -> dict[str, Any]:
    """min/max/mean/std/CV for a non-empty list (all zeros when empty)."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "cv": None,
        }
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "cv": round(std / mean, 4) if mean else None,
    }


def _phase_metrics(phase: str, run: dict[str, Any], video_hash: str) -> dict[str, Any]:
    artifacts = _artifacts_dir(Path(run["cache_dir"]), video_hash)
    script = _load_json(artifacts / "script.json")
    timed = _load_json(artifacts / "timed.json")
    timing_path = artifacts / "timing.json"
    timing = _load_json(timing_path) if timing_path.exists() else {}

    vi_by_idx = {
        index: str(row.get("vi", "")) for index, row in enumerate(script)
    }

    clips = 0
    stretches: list[float] = []
    rates: list[float] = []
    unused: list[float] = []
    length_scales: list[float] = []
    for record in timed:
        idx = int(record.get("idx", 0))
        action = str(record.get("action", ""))
        placed = max(
            float(record.get("end", 0.0)) - float(record.get("start", 0.0)), 0.0
        )
        window = float(record.get("window", 0.0))
        words = len(vi_by_idx.get(idx, "").split())
        if action == "clip":
            clips += 1
        if action == "stretch":
            stretches.append(float(record.get("stretch", 1.0)))
        if placed > 0.0 and words > 0:
            rates.append(words / placed)
        unused.append(window - placed)
        if record.get("rate_action") == "resynth":
            length_scales.append(float(record.get("length_scale", 1.0)))

    return {
        "key": phase,
        "status": "ok",
        "tts_rate_fit": RATE_FIT[phase],
        "recenter_audio": RECENTER[phase],
        "rows": len(timed),
        "clips": clips,
        "stretched": len(stretches),
        "stretch": _stats(stretches),
        "rate": _stats(rates),
        "unused_total": round(sum(unused), 3),
        "unused_gt_1s": sum(1 for value in unused if value > 1.0),
        "unused_max": round(max(unused), 3) if unused else 0.0,
        "length_scale": _stats(length_scales),
        "rate_fit_unit": timing.get("rate_fit"),
    }


def _phase_or_error(phase: str, runs: dict[str, Any], video_hash: str) -> dict[str, Any]:
    run = runs.get(phase)
    if run is None:
        return {"key": phase, "status": "missing"}
    if run.get("status") != "ok":
        return {"key": phase, "status": run.get("status")}
    return _phase_metrics(phase, run, video_hash)


def _acceptance(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered targets: clips -> 0; unused seconds down >= 50 %."""
    before_clips = before.get("clips")
    after_clips = after.get("clips")
    before_unused = float(before.get("unused_total", 0.0) or 0.0)
    after_unused = float(after.get("unused_total", 0.0) or 0.0)
    reduction = (
        (before_unused - after_unused) / before_unused * 100.0
        if before_unused > 0.0
        else 0.0
    )
    return {
        "clips_target_0": bool(after_clips == 0),
        "clips_before": before_clips,
        "clips_after": after_clips,
        "unused_reduction_pct": round(reduction, 2),
        "unused_reduction_target_50pct": bool(reduction >= 50.0),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _print_table(before: dict[str, Any], after: dict[str, Any]) -> None:
    br = before.get("rate") or {}
    ar = after.get("rate") or {}
    bs = before.get("stretch") or {}
    ass = after.get("stretch") or {}
    bl = before.get("length_scale") or {}
    al = after.get("length_scale") or {}
    rows = [
        ("status", before.get("status", "ok"), after.get("status", "ok")),
        ("rows", before.get("rows"), after.get("rows")),
        ("clips", before.get("clips"), after.get("clips")),
        ("atempo-stretched", before.get("stretched"), after.get("stretched")),
        (
            "stretch min/max/mean",
            f"{_fmt(bs.get('min'))}/{_fmt(bs.get('max'))}/{_fmt(bs.get('mean'))}",
            f"{_fmt(ass.get('min'))}/{_fmt(ass.get('max'))}/{_fmt(ass.get('mean'))}",
        ),
        (
            "rate min/max (words/s)",
            f"{_fmt(br.get('min'))}/{_fmt(br.get('max'))}",
            f"{_fmt(ar.get('min'))}/{_fmt(ar.get('max'))}",
        ),
        (
            "rate mean/std/CV",
            f"{_fmt(br.get('mean'))}/{_fmt(br.get('std'))}/{_fmt(br.get('cv'))}",
            f"{_fmt(ar.get('mean'))}/{_fmt(ar.get('std'))}/{_fmt(ar.get('cv'))}",
        ),
        ("unused total (s)", before.get("unused_total"), after.get("unused_total")),
        ("unused rows > 1 s", before.get("unused_gt_1s"), after.get("unused_gt_1s")),
        ("unused max (s)", before.get("unused_max"), after.get("unused_max")),
        (
            "length_scale n/min/max",
            f"{_fmt(bl.get('count'))}/{_fmt(bl.get('min'))}/{_fmt(bl.get('max'))}",
            f"{_fmt(al.get('count'))}/{_fmt(al.get('min'))}/{_fmt(al.get('max'))}",
        ),
        ("rate_fit unit", _fmt(before.get("rate_fit_unit")), _fmt(after.get("rate_fit_unit"))),
    ]
    print(f"\n{'metric':<26}{'BEFORE':<34}{'AFTER':<34}")
    for name, left, right in rows:
        print(f"{name:<26}{_fmt(left):<34}{_fmt(right):<34}")


def cmd_score(args: argparse.Namespace) -> int:
    runs = _load_runs()
    if not runs:
        print(f"no runs recorded ({RUNS_PATH}); run 'run' first", file=sys.stderr)
        return 1

    video_hash = _reference_hash()
    before = _phase_or_error("before", runs, video_hash)
    after = _phase_or_error("after", runs, video_hash)

    payload: dict[str, Any] = {
        "input": str(INPUT),
        "video_hash": video_hash,
        "mt_engine": MT_ENGINE,
        "before": before,
        "after": after,
        "acceptance": _acceptance(before, after),
    }
    _dump_json(SCORES_PATH, payload)
    _print_table(before, after)
    print(f"\nscores written to {SCORES_PATH}")
    print(f"acceptance: {_fmt(payload['acceptance'])}")
    return 0


# --------------------------------------------------------------- rows command
def _placed_rates(
    script: list[dict[str, Any]], timed: list[dict[str, Any]]
) -> dict[int, float]:
    vi_by_idx = {index: str(row.get("vi", "")) for index, row in enumerate(script)}
    rates: dict[int, float] = {}
    for record in timed:
        idx = int(record.get("idx", 0))
        placed = max(
            float(record.get("end", 0.0)) - float(record.get("start", 0.0)), 0.0
        )
        words = len(vi_by_idx.get(idx, "").split())
        if placed > 0.0 and words > 0:
            rates[idx] = words / placed
    return rates


def _row_indices(
    before: list[dict[str, Any]], rates: dict[int, float]
) -> list[int]:
    """Pre-registered worst rows + the actual BEFORE clips + extremes by rate."""
    indices: set[int] = set(USER_CLIPPED)
    for record in before:
        if record.get("action") == "clip":
            indices.add(int(record.get("idx", 0)))
    ordered = sorted(rates, key=lambda idx: rates[idx])
    indices.update(ordered[:5])
    indices.update(ordered[-5:])
    return sorted(indices)


def cmd_rows(args: argparse.Namespace) -> int:
    runs = _load_runs()
    before_run = runs.get("before")
    after_run = runs.get("after")
    if (
        not before_run
        or not after_run
        or before_run.get("status") != "ok"
        or after_run.get("status") != "ok"
    ):
        print("before/after runs are not both available", file=sys.stderr)
        return 1

    video_hash = _reference_hash()
    before_art = _artifacts_dir(Path(before_run["cache_dir"]), video_hash)
    after_art = _artifacts_dir(Path(after_run["cache_dir"]), video_hash)
    before_script = _load_json(before_art / "script.json")
    after_script = _load_json(after_art / "script.json")
    before_timed = _load_json(before_art / "timed.json")
    after_timed = _load_json(after_art / "timed.json")

    before_by_idx = {int(r.get("idx", 0)): r for r in before_timed}
    after_by_idx = {int(r.get("idx", 0)): r for r in after_timed}
    rates = _placed_rates(before_script, before_timed)
    indices = _row_indices(before_timed, rates)

    print(f"reference rows: {len(before_script)} (BEFORE) / {len(after_script)} (AFTER)")
    print(f"row indices shown: {indices}\n")
    header = (
        f"{'idx':>4}  {'BEFORE action/trim':<28} "
        f"{'AFTER action/ls/placed':<30} EN"
    )
    print(header)
    print("-" * len(header))
    for index in indices:
        b = before_by_idx.get(index, {})
        a = after_by_idx.get(index, {})
        before_desc = str(b.get("action"))
        if b.get("clipped"):
            before_desc += f" trim={_fmt(b.get('seconds_trimmed'))}"
        ls = a.get("length_scale")
        after_desc = str(a.get("action"))
        if a.get("rate_action") == "resynth":
            after_desc += f" ls={_fmt(ls)}"
        placed = max(
            float(a.get("end", 0.0)) - float(a.get("start", 0.0)), 0.0
        )
        after_desc += f" placed={placed:.2f}s"
        en = str(after_script[index].get("en") or before_script[index].get("en") or "")
        print(f"{index:>4}  {before_desc:<28} {after_desc:<30} {en}")
    return 0


# ---------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_timing_fit",
        description="A/B evaluate two-pass TTS rate fitting on the reference video.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run", help="run the BEFORE/AFTER pipeline pair into scratch caches"
    )
    run.set_defaults(func=cmd_run)

    score = sub.add_parser(
        "score", help="score the scratch caches and write scores.json"
    )
    score.set_defaults(func=cmd_score)

    rows = sub.add_parser(
        "rows", help="side-by-side worst-row table (EN / BEFORE / AFTER)"
    )
    rows.set_defaults(func=cmd_rows)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

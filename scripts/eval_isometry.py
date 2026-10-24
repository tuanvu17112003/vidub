"""ISO-1 isometric budgeting: measurement harness (stdlib-only, READ-ONLY scoring).

Two subcommands:

* ``run`` — shells out to the real pipeline (``python -m vidub run``) with a
  **scratch** cache, times the wall clock, prints the per-stage timings from the
  produced ``report.json`` and then scores the artifacts. It refuses any
  ``--cache-dir`` inside the User's ``.vidub-cache/`` (the first path component
  must start with ``.vidub-cache-``) and any ``--output-dir`` under the repo's
  ``output/`` (unless ``--allow-output-dir``). ``--seed-from`` copies an existing
  job's ``job.json`` + ``segments.json`` into the scratch cache so the expensive
  ASR does not need to re-run (the translate stage still runs for real).
* ``score`` — reads a cache/artifacts dir and prints, per unit: the window, the
  character budget, the actual characters, ``over_budget``, the isometric
  outcome, and the projected stretch before/after; plus the aggregate counts and
  the ``translation_qc.units.flags`` counters. A job without isometry records
  (one produced before ISO-1) prints ``null`` + the reason instead.

Usage::

    ./.venv/Scripts/python.exe scripts/eval_isometry.py run \
        --input .vidub-cache/downloads/generic-youtube-a7TOameRqoY.mkv \
        --cache-dir .vidub-cache-iso/run-on \
        --output-dir .vidub-cache-iso/work-on \
        --mt-engine hunyuan --mode subtitles-only --force translate \
        --seed-from .vidub-cache/fbae346ba03869da6f62f85cd97eeb82c9b5d964

    ./.venv/Scripts/python.exe scripts/eval_isometry.py score \
        --run .vidub-cache-iso/run-on/<video_hash> --json eval-cache/iso1/on.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

__all__ = ["main", "build_parser", "cmd_run", "cmd_score", "score_report"]

CACHE_PREFIX = ".vidub-cache-"
OUTPUT_ROOT = ROOT / "output"
USER_CACHE = ROOT / ".vidub-cache"

# The isometry record keys a post-ISO-1 ``units.json`` entry carries.
ISO_KEYS = (
    "budget_chars",
    "chars",
    "over_budget",
    "isometric",
    "chars_after",
    "projected_stretch_before",
    "projected_stretch_after",
)
# The aggregate counters added to ``translation_qc.json`` ``units.flags``.
ISO_FLAGS = ("over_budget_after", "isometric_shrunk", "isometric_kept")


# --------------------------------------------------------------------- helpers
def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifacts(path: str | Path) -> Path:
    """Accept an artifacts dir, or a cache root holding exactly one job dir."""
    root = Path(path)
    if (root / "script.json").exists() or (root / "units.json").exists():
        return root
    candidates = (
        [
            child
            for child in root.iterdir()
            if (child / "script.json").exists() or (child / "units.json").exists()
        ]
        if root.is_dir()
        else []
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"{root} is not an artifacts dir (no script.json/units.json); pass the "
        "job's artifact directory, e.g. <cache>/{<video_hash>}"
    )


def _units(artifacts: Path) -> list[dict]:
    path = artifacts / "units.json"
    if not path.exists():
        return []
    data = _load_json(path)
    entries = data.get("units") if isinstance(data, dict) else None
    return [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _isometry_fields(entry: dict) -> dict | None:
    """The isometry record of one unit, or ``None`` for a pre-feature job."""
    if not any(key in entry for key in ISO_KEYS):
        return None
    return {key: entry.get(key) for key in ISO_KEYS}


# ---------------------------------------------------------------------- score
def score_report(cache_or_artifacts: str | Path) -> dict:
    """Read one job and compute the isometry measurement (read-only)."""
    artifacts = _resolve_artifacts(cache_or_artifacts)
    units = _units(artifacts)
    report: dict[str, Any] = {
        "artifacts_dir": str(artifacts),
        "units": len(units),
    }

    qc_path = artifacts / "translation_qc.json"
    qc = _load_json(qc_path) if qc_path.exists() else None
    flags = None
    if isinstance(qc, dict) and isinstance(qc.get("units"), dict):
        candidate = qc["units"].get("flags")
        flags = candidate if isinstance(candidate, dict) else None
    report["translation_qc"] = {
        "verdict": qc.get("verdict") if isinstance(qc, dict) else None,
        "counts": qc.get("counts") if isinstance(qc, dict) else None,
        "units_flags": flags,
    }

    records: list[dict] = []
    for entry in units:
        iso = _isometry_fields(entry)
        records.append(
            {
                "index": entry.get("index"),
                "window": (
                    round(float(entry["end"]) - float(entry["start"]), 3)
                    if isinstance(entry.get("end"), (int, float))
                    and isinstance(entry.get("start"), (int, float))
                    else None
                ),
                "budget_chars": (iso or {}).get("budget_chars"),
                "chars": (iso or {}).get("chars"),
                "over_budget": (iso or {}).get("over_budget"),
                "isometric": (iso or {}).get("isometric"),
                "chars_after": (iso or {}).get("chars_after"),
                "projected_stretch_before": (iso or {}).get("projected_stretch_before"),
                "projected_stretch_after": (iso or {}).get("projected_stretch_after"),
                "has_isometry": iso is not None,
                "vi": entry.get("vi"),
            }
        )
    report["per_unit"] = records

    known = [record for record in records if record["has_isometry"]]
    reason = None if known else (
        "units.json has no isometry fields (the job predates ISO-1's "
        "isometric budgeting)"
    )
    report["reason"] = reason

    if not known:
        report["aggregate"] = {
            key: None
            for key in ("over_budget_before", "over_budget_after", "shrunk",
                        "kept", "unavailable")
        }
    else:
        report["aggregate"] = {
            "over_budget_before": sum(1 for r in known if r["over_budget"]),
            "over_budget_after": (
                flags.get("over_budget_after") if flags else
                sum(
                    1
                    for r in known
                    if isinstance(r["chars_after"], int)
                    and isinstance(r["budget_chars"], int)
                    and r["chars_after"] > r["budget_chars"]
                )
            ),
            "shrunk": (
                flags.get("isometric_shrunk") if flags else
                sum(1 for r in known if r["isometric"] == "shrunk")
            ),
            "kept": (
                flags.get("isometric_kept") if flags else
                sum(1 for r in known if r["isometric"] == "kept")
            ),
            "unavailable": sum(1 for r in known if r["isometric"] == "unavailable"),
        }
    # A unit whose final projected stretch rose above its pre-repair stretch
    # would violate the predicate's contract; report it loudly.
    report["worse"] = [
        {
            "index": r["index"],
            "stretch_before": r["projected_stretch_before"],
            "stretch_after": r["projected_stretch_after"],
        }
        for r in known
        if isinstance(r["projected_stretch_before"], (int, float))
        and isinstance(r["projected_stretch_after"], (int, float))
        and r["projected_stretch_after"] > r["projected_stretch_before"]
    ]
    report["kept_units"] = [
        {"index": r["index"], "chars": r["chars"], "budget_chars": r["budget_chars"],
         "vi_tail": str(r["vi"] or "")[-48:]}
        for r in known
        if r["isometric"] == "kept"
    ]
    report["shrunk_units"] = [
        {"index": r["index"], "chars": r["chars"], "chars_after": r["chars_after"],
         "budget_chars": r["budget_chars"]}
        for r in known
        if r["isometric"] == "shrunk"
    ]
    return report


def _print_report(report: dict) -> None:
    print(f"artifacts: {report['artifacts_dir']}")
    print(f"units: {report['units']}")
    if report["reason"]:
        print(f"isometry: null ({report['reason']})")
    print(
        f"{'idx':>3} {'window_s':>8} {'budget':>7} {'chars':>6} {'over':>5} "
        f"{'isometric':>11} {'chars_after':>11} {'stretch_before':>14} {'stretch_after':>13}"
    )
    for record in report["per_unit"]:
        print(
            f"{_fmt(record['index']):>3} {_fmt(record['window']):>8} "
            f"{_fmt(record['budget_chars']):>7} {_fmt(record['chars']):>6} "
            f"{_fmt(record['over_budget']):>5} {_fmt(record['isometric']):>11} "
            f"{_fmt(record['chars_after']):>11} {_fmt(record['projected_stretch_before']):>14} "
            f"{_fmt(record['projected_stretch_after']):>13}"
        )
    aggregate = report["aggregate"]
    print("\naggregate:")
    for key in ("over_budget_before", "over_budget_after", "shrunk", "kept", "unavailable"):
        value = aggregate.get(key)
        suffix = "" if value is not None else "  (null: no isometry records)"
        print(f"  {key}: {_fmt(value)}{suffix}")
    qc = report["translation_qc"]
    print("\ntranslation_qc:")
    print(f"  verdict: {_fmt(qc['verdict'])}")
    print(f"  counts: {_fmt(qc['counts'])}")
    print(f"  units.flags: {_fmt(qc['units_flags'])}")
    print(f"\nunits that got worse (stretch_after > stretch_before): {len(report['worse'])}")
    for item in report["worse"]:
        print(
            f"  unit {item['index']}: {item['stretch_before']} -> {item['stretch_after']}"
        )


# ------------------------------------------------------------------ safety
def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _check_cache_dir(cache_dir: str) -> None:
    resolved = Path(cache_dir)
    if _is_under(resolved, USER_CACHE) or _is_under(resolved, OUTPUT_ROOT):
        raise SystemExit(
            f"refusing --cache-dir {cache_dir!r}: it is inside the User's "
            f"{USER_CACHE.name}/ or output/ tree. Use a scratch cache such as "
            f"{CACHE_PREFIX}iso/run-on."
        )
    first = resolved.parts[0] if resolved.parts else ""
    if not first.startswith(CACHE_PREFIX):
        raise SystemExit(
            f"refusing --cache-dir {cache_dir!r}: eval runs must write to a "
            f"scratch cache whose first path component starts with "
            f"{CACHE_PREFIX!r} (e.g. {CACHE_PREFIX}iso/run-on). The User's "
            f"{USER_CACHE.name}/ and output/ are read-only."
        )


def _check_output_dir(output_dir: str, allow: bool) -> None:
    if allow:
        return
    if _is_under(Path(output_dir), OUTPUT_ROOT):
        raise SystemExit(
            f"refusing --output-dir {output_dir!r}: it is under the repo's "
            f"{OUTPUT_ROOT.name}/ deliverable tree. Pass --allow-output-dir to "
            "override, or use a scratch dir."
        )


def _seed_cache(seed_from: str, cache_dir: str) -> None:
    """Copy an existing job's ``job.json`` + ``segments.json`` into the cache."""
    source = _resolve_artifacts(seed_from)
    job_src = source / "job.json"
    seg_src = source / "segments.json"
    if not job_src.exists() or not seg_src.exists():
        raise SystemExit(
            f"--seed-from {seed_from!r} must hold job.json + segments.json "
            f"(found in {source})"
        )
    data = _load_json(job_src)
    video_hash = str(data.get("video_hash") or "")
    if not video_hash:
        raise SystemExit(f"--seed-from {seed_from!r}: job.json has no video_hash")
    dest = Path(cache_dir) / video_hash
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("job.json", "segments.json"):
        shutil.copyfile(source / name, dest / name)
    print(f"seeded {dest} (job.json + segments.json) from {source}")


def _report_timings(output_dir: Path, stem: str) -> dict | None:
    """Per-stage timings from the produced report.json (relative to CWD)."""
    report_path = output_dir / "output" / stem / "report.json"
    if not report_path.exists():
        candidates = list((output_dir / "output").glob("*/report.json"))
        report_path = candidates[0] if len(candidates) == 1 else report_path
    if not report_path.exists():
        return None
    data = _load_json(report_path)
    return {
        "report": str(report_path),
        "total_seconds": data.get("total_seconds"),
        "stages": {
            name: entry.get("seconds")
            for name, entry in (data.get("stages") or {}).items()
            if isinstance(entry, dict)
        },
    }


# ---------------------------------------------------------------- subcommands
def cmd_run(args: argparse.Namespace) -> int:
    _check_cache_dir(args.cache_dir)
    _check_output_dir(args.output_dir, args.allow_output_dir)

    # The subprocess runs with cwd=output_dir, so every path handed to it must be
    # absolute (the harness itself is invoked from the repo root).
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    input_arg = args.input
    if "://" not in input_arg:
        input_arg = str(Path(input_arg).resolve())
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.seed_from:
        _seed_cache(args.seed_from, str(cache_dir))

    command = [
        sys.executable,
        "-m",
        "vidub.cli",
        "run",
        input_arg,
        "--cache-dir",
        str(cache_dir),
    ]
    if args.mt_engine:
        command += ["--mt-engine", args.mt_engine]
    if args.mode:
        command += ["--mode", args.mode]
    if args.force:
        command += ["--force", args.force]
    command += list(args.extra or [])

    print("running:", " ".join(command))
    print("cwd:", output_dir)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(output_dir), check=False, text=True)
    wall = time.perf_counter() - started
    print(f"wall_clock_seconds: {wall:.1f}")

    if completed.returncode != 0:
        print(f"run failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode

    timings = _report_timings(output_dir, Path(input_arg).stem)
    if timings:
        print(f"report: {timings['report']}")
        print(f"total_seconds: {timings['total_seconds']}")
        for name, seconds in timings["stages"].items():
            print(f"  {name}: {seconds}")

    try:
        artifacts = _resolve_artifacts(cache_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report = score_report(artifacts)
    _print_report(report)
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nscores written to {json_path}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    report = score_report(args.run)
    _print_report(report)
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nscores written to {json_path}")
    return 0


# ----------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_isometry",
        description="ISO-1 isometric budgeting: run + score harness (stdlib-only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline into a scratch cache, then score")
    run.add_argument("--input", required=True, help="input video path or URL")
    run.add_argument("--cache-dir", required=True, help="scratch cache (first component .vidub-cache-)")
    run.add_argument("--output-dir", required=True, help="scratch CWD where the pipeline writes output/")
    run.add_argument("--mt-engine", default="hunyuan", choices=["envit5", "ollama", "hunyuan"])
    run.add_argument("--mode", default=None, help="pipeline mode (e.g. subtitles-only)")
    run.add_argument("--force", default=None, metavar="STAGE[,STAGE]", help="stages to re-run")
    run.add_argument(
        "--seed-from",
        default=None,
        help="existing job dir to copy job.json + segments.json from (skips ASR)",
    )
    run.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="FLAG",
        help="extra single token passed through to 'vidub run' (repeatable)",
    )
    run.add_argument("--json", default=None, help="where to write the score JSON")
    run.add_argument(
        "--allow-output-dir",
        action="store_true",
        help="allow an --output-dir under the repo's output/ tree",
    )
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="print the isometry numbers from artifacts (read-only)")
    score.add_argument("--run", required=True, help="cache dir or artifacts dir")
    score.add_argument("--json", default=None, help="where to write the score JSON")
    score.set_defaults(func=cmd_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""SA-1 sentence-locked units: measurement harness (pre-registered metrics M1-M5).

A stdlib-only A/B harness for the SA-1 rework that makes the **sentence** the
``script.json`` row unit and the **unit** the translation + QC unit. It reads
**pipeline artifacts only** (never a model, never the network) and computes five
pre-registered metrics:

* **M1 ``mid_sentence_unit_boundaries``** — placed-audio unit closures that cut a
  source sentence (``timed.json`` ``members`` + ``script.json`` ``en``),
  excluding speaker changes, pause cuts (a gap >= ``PAUSE_CUT_GAP_S`` between the
  previous member's end and the next member's start) and the transcript end.
* **M2 ``omission_units``** — units whose spoken text dropped content: coverage
  (spoken/source words) < 0.75, or a sentence-loss (>= 2 source sentences and at
  least one lost while coverage < 1.10). Uses ``units.json`` coverage/``vi`` when
  present, else the joined member ``vi``.
* **M3 ``mid_sentence_cue_texts``** — cue texts that do not end with
  sentence-final punctuation (the ``... là ví dụ của`` class). SA-1: one cue per
  sentence unit (``units.json``); legacy: one cue per ``script.json`` row.
* **M4 ``translation_qc_unresolved``** — ``translation_qc.json`` ``segments``
  entries with ``resolved == false``; the granularity (``unit`` vs ``row``) is
  reported and the two are never compared as if identical.
* **M5 ``pacing``** — ``timing.json`` clips / ``stretched``.

Safety: ``score`` is strictly read-only. ``run`` refuses any ``--cache-dir`` that
does not start with ``.vidub-cache-`` and any ``--output-dir`` under the repo's
``output/`` (unless ``--allow-output-dir``), so a run can never touch the User's
caches or deliverables.

Usage::

    ./.venv/Scripts/python.exe scripts/eval_sentence_lock.py \
        score --run .vidub-cache-sa1/run/<hash> \
              --baseline .vidub-cache/<hash> \
              --json eval-cache/sa1/scores.json

    ./.venv/Scripts/python.exe scripts/eval_sentence_lock.py \
        run --url <url> --cache-dir .vidub-cache-sa1 \
            --output-dir .vidub-cache-sa1/work --mt-engine hunyuan
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vidub.utils.sentences import PAUSE_CUT_GAP_S  # noqa: E402  (project constant)

__all__ = ["main", "build_parser", "cmd_score", "cmd_run", "metric_report"]

# Names of the seven artifacts the metrics read (printed by ``score``).
ARTIFACTS = (
    "script.json",
    "timed.json",
    "units.json",
    "translation_qc.json",
    "timing.json",
)

DEFAULT_SCORES = ROOT / "eval-cache" / "sa1" / "scores.json"
OUTPUT_ROOT = ROOT / "output"
CACHE_PREFIX = ".vidub-cache-"

# Sentence-final punctuation (matches vidub.utils.sentences.TERMINAL_CHARS) and
# the closers a sentence may carry after it.
TERMINAL_CHARS = ".!?…"
CLOSERS = "\"'”’)"


# --------------------------------------------------------------------- helpers
def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ends_sentence(text: str) -> bool:
    """True when ``text`` ends with terminal punctuation (closers skipped)."""
    stripped = str(text or "").strip()
    index = len(stripped) - 1
    while index >= 0 and stripped[index] in CLOSERS:
        index -= 1
    return index >= 0 and stripped[index] in TERMINAL_CHARS


def _sentence_count(text: str) -> int:
    """Number of terminal characters in ``text``."""
    return sum(1 for char in str(text or "") if char in TERMINAL_CHARS)


def _members(row: dict) -> list[int]:
    """The script-row indices a timed row covers (``[idx]`` on the legacy shape)."""
    members = row.get("members")
    if isinstance(members, list) and members:
        out: list[int] = []
        for member in members:
            try:
                out.append(int(member))
            except (TypeError, ValueError):
                continue
        if out:
            return out
    try:
        return [int(row.get("idx", 0))]
    except (TypeError, ValueError):
        return []


def _resolve_artifacts(path: str | Path) -> Path:
    """Accept an artifacts dir, or a cache root holding exactly one job dir."""
    root = Path(path)
    if (root / "script.json").exists():
        return root
    candidates = (
        [child for child in root.iterdir() if (child / "script.json").exists()]
        if root.is_dir()
        else []
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"{root} is not an artifacts dir (no script.json); pass the job's "
        f"artifact directory, e.g. <cache>/{'<video_hash>'}"
    )


def _ordered(timed: list[dict]) -> list[dict]:
    return sorted(timed, key=lambda row: int(row.get("idx", 0)))


def _fmt(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return "null" if value is None else str(value)


# ------------------------------------------------------------------- metrics
def _metric_m1(artifacts: Path, script: list[dict], timed: list[dict]) -> dict:
    """Mid-sentence unit boundaries in the placed audio (target 0)."""
    rows = _ordered(timed)
    examples: list[dict] = []
    count = 0
    for previous, following in zip(rows, rows[1:]):
        prev_members = _members(previous)
        next_members = _members(following)
        if not prev_members or not next_members:
            continue
        prev_member, next_member = prev_members[-1], next_members[0]
        if not (0 <= prev_member < len(script) and 0 <= next_member < len(script)):
            continue
        prev_en = str(script[prev_member].get("en", ""))
        if _ends_sentence(prev_en):
            continue  # a completed sentence is a legitimate unit end
        if script[prev_member].get("speaker") != script[next_member].get("speaker"):
            continue  # speaker changes are legitimate unit ends
        gap = float(script[next_member].get("start", 0.0)) - float(
            script[prev_member].get("end", 0.0)
        )
        if gap >= PAUSE_CUT_GAP_S:
            continue  # a counted pause cut is a documented last resort
        count += 1
        examples.append(
            {
                "previous_row": int(previous.get("idx", 0)),
                "next_row": int(following.get("idx", 0)),
                "previous_member": prev_member,
                "next_member": next_member,
                "gap": round(gap, 3),
                "previous_en_tail": prev_en[-48:],
            }
        )
    return {"value": count, "target": 0, "examples": examples}


def _authoritative_units(artifacts: Path) -> dict[int, dict] | None:
    """``units.json`` entries by their first member index (when present)."""
    path = artifacts / "units.json"
    if not path.exists():
        return None
    data = _load_json(path)
    entries = data.get("units") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return None
    indexed: dict[int, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        indices = entry.get("indices")
        if isinstance(indices, list) and indices:
            try:
                indexed[int(indices[0])] = entry
            except (TypeError, ValueError):
                continue
    return indexed


def _metric_m2(
    artifacts: Path, script: list[dict], timed: list[dict]
) -> dict:
    """Units whose spoken text dropped content (target 0)."""
    authoritative = _authoritative_units(artifacts)
    flagged: list[dict] = []
    for row in _ordered(timed):
        members = _members(row)
        valid = [m for m in members if 0 <= m < len(script)]
        if not valid:
            continue
        source = " ".join(str(script[m].get("en", "")) for m in valid)
        entry = authoritative.get(valid[0]) if authoritative is not None else None
        if entry is not None and [int(i) for i in entry.get("indices", [])] == valid:
            spoken = str(entry.get("vi", ""))
            coverage = entry.get("coverage")
        else:
            entry = None
            spoken = " ".join(
                piece
                for piece in (str(script[m].get("vi", "")).strip() for m in valid)
                if piece
            )
            coverage = None
        words_in = len(source.split())
        words_out = len(spoken.split())
        if not isinstance(coverage, (int, float)):
            coverage = words_out / max(words_in, 1)
        sentences_in = _sentence_count(source)
        sentences_out = _sentence_count(spoken)
        flags: list[str] = []
        if coverage < 0.75:
            flags.append("omission-risk")
        if (
            sentences_in >= 2
            and sentences_out <= sentences_in - 1
            and coverage < 1.10
        ):
            flags.append("sentence-loss")
        if flags:
            flagged.append(
                {
                    "unit": int(row.get("idx", 0)),
                    "members": valid,
                    "words_in": words_in,
                    "words_out": words_out,
                    "coverage": round(float(coverage), 3),
                    "sentences_in": sentences_in,
                    "sentences_out": sentences_out,
                    "flags": flags,
                    "text_source": "units.json" if entry is not None else "member vi",
                }
            )
    return {"value": len(flagged), "target": 0, "units": flagged}


def _metric_m3(artifacts: Path, script: list[dict]) -> dict:
    """Cue texts that end mid-sentence (target 0)."""
    authoritative = _authoritative_units(artifacts)
    if authoritative is not None:
        # SA-1: one cue per sentence unit, in pipeline (units.json) order.
        entries = _load_json(artifacts / "units.json").get("units", [])
        cues = [
            (int(entry.get("index", position)), str(entry.get("vi", "")))
            for position, entry in enumerate(entries)
        ]
        source = "units.json"
    else:
        cues = [
            (int(row.get("idx", position)), str(row.get("vi", "")))
            for position, row in enumerate(script)
        ]
        source = "script.json rows"
    incomplete = [
        {"cue": index, "vi_tail": text[-48:]}
        for index, text in cues
        if not _ends_sentence(text)
    ]
    return {
        "value": len(incomplete),
        "target": 0,
        "cues": len(cues),
        "source": source,
        "incomplete": incomplete,
    }


def _metric_m4(artifacts: Path, script: list[dict]) -> dict:
    """Unresolved translation-QC flags (target <= 5)."""
    path = artifacts / "translation_qc.json"
    data = _load_json(path)
    segments = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(segments, list):
        return {"value": None, "unavailable": "translation_qc.json has no segments"}
    unresolved = [entry for entry in segments if not entry.get("resolved")]

    unit_count: int | None = None
    units_path = artifacts / "units.json"
    if units_path.exists():
        entries = _load_json(units_path).get("units")
        if isinstance(entries, list):
            unit_count = len(entries)
    if unit_count is None and isinstance(data.get("units"), dict):
        recorded = data["units"].get("units")
        if isinstance(recorded, int):
            unit_count = recorded
    if unit_count is not None and len(segments) == unit_count:
        granularity = "unit"
    else:
        granularity = "row"
    denominator = len(segments)
    return {
        "value": len(unresolved),
        "target": 5,
        "granularity": granularity,
        "scored_entries": denominator,
        "unresolved_per_100": (
            round(100.0 * len(unresolved) / denominator, 1) if denominator else None
        ),
        "unresolved_indexes": [entry.get("index") for entry in unresolved],
    }


def _metric_m5(artifacts: Path) -> dict:
    """Clips / stretched from timing.json (targets: clips 0, stretched <= baseline)."""
    data = _load_json(artifacts / "timing.json")
    if not isinstance(data, dict):
        return {"value": None, "unavailable": "timing.json is not an object"}
    clips = data.get("clips")
    clip_count = len(clips) if isinstance(clips, list) else None
    clipped = data.get("clipped")
    if not isinstance(clipped, int):
        clipped = clip_count
    stretched = data.get("stretched")
    return {
        "clipped": clipped,
        "clips": clip_count,
        "stretched": stretched if isinstance(stretched, int) else None,
        "max_stretch": data.get("max_stretch"),
        "target": {"clipped": 0, "stretched": "<= baseline"},
    }


def metric_report(cache_dir: str | Path) -> dict:
    """Compute M1-M5 for one artifacts directory (read-only)."""
    artifacts = _resolve_artifacts(cache_dir)
    files = {name: (artifacts / name) for name in ARTIFACTS}
    report: dict[str, Any] = {
        "artifacts_dir": str(artifacts),
        "artifacts_read": [
            str(path) for path in files.values() if path.exists()
        ],
        "artifacts_missing": [
            name for name, path in files.items() if not path.exists()
        ],
    }

    script = _load_json(artifacts / "script.json")
    timed = _load_json(artifacts / "timed.json")
    report["row_count"] = len(script)
    report["timed_rows"] = len(timed)
    report["units_json"] = (artifacts / "units.json").exists()

    report["M1_mid_sentence_unit_boundaries"] = _metric_m1(artifacts, script, timed)
    report["M2_omission_units"] = _metric_m2(artifacts, script, timed)
    report["M3_mid_sentence_cue_texts"] = _metric_m3(artifacts, script)
    report["M4_translation_qc_unresolved"] = _metric_m4(artifacts, script)
    report["M5_pacing"] = _metric_m5(artifacts)
    return report


# ------------------------------------------------------------------- verdict
def _verdicts(run: dict, baseline: dict | None) -> dict[str, dict]:
    """Per-metric verdict against the pre-registered targets."""
    out: dict[str, dict] = {}

    def add(name: str, value, target, *, compare: str = "le") -> None:
        if value is None:
            out[name] = {"target": target, "verdict": None, "reason": "unavailable"}
            return
        if compare == "eq":
            met = value == target
        else:
            met = value <= target
        out[name] = {
            "target": target,
            "value": value,
            "verdict": "MET" if met else "NOT MET",
        }

    baseline_m5 = (baseline or {}).get("M5_pacing") if baseline else None
    run_m5 = run.get("M5_pacing") or {}
    add(
        "M1_mid_sentence_unit_boundaries",
        (run.get("M1_mid_sentence_unit_boundaries") or {}).get("value"),
        0,
        compare="eq",
    )
    add(
        "M2_omission_units",
        (run.get("M2_omission_units") or {}).get("value"),
        0,
        compare="eq",
    )
    add(
        "M3_mid_sentence_cue_texts",
        (run.get("M3_mid_sentence_cue_texts") or {}).get("value"),
        0,
        compare="eq",
    )
    add(
        "M4_translation_qc_unresolved",
        (run.get("M4_translation_qc_unresolved") or {}).get("value"),
        5,
        compare="le",
    )
    add("M5_clipped", run_m5.get("clipped"), 0, compare="eq")
    if baseline_m5 is not None and isinstance(baseline_m5.get("stretched"), int):
        add(
            "M5_stretched",
            run_m5.get("stretched"),
            baseline_m5["stretched"],
            compare="le",
        )
    else:
        out["M5_stretched"] = {
            "target": None,
            "verdict": None,
            "reason": "no baseline cache given",
        }
    return out


# --------------------------------------------------------------------- table
_ROWS = (
    ("M1 mid-sentence unit boundaries", "M1_mid_sentence_unit_boundaries"),
    ("M2 omission units", "M2_omission_units"),
    ("M3 mid-sentence cue texts", "M3_mid_sentence_cue_texts"),
    ("M4 translation_qc unresolved", "M4_translation_qc_unresolved"),
)


def _print_table(result: dict) -> None:
    run = result.get("run")
    baseline = result.get("baseline")
    verdicts = result["verdicts"]
    print(
        f"\n{'metric':<34}{'baseline':<14}{'run':<14}{'target':<14}{'verdict':<10}"
    )
    for label, key in _ROWS:
        run_value = (run.get(key) or {}).get("value") if run else None
        base_value = (baseline or {}).get(key, {}).get("value") if baseline else None
        verdict = verdicts.get(key, {})
        target = verdict.get("target")
        print(
            f"{label:<34}{_fmt(base_value):<14}{_fmt(run_value):<14}"
            f"{_fmt(target):<14}{str(verdict.get('verdict')):<10}"
        )
    run_m5 = (run or {}).get("M5_pacing") or {}
    base_m5 = (baseline or {}).get("M5_pacing") or {} if baseline else {}
    for label, key, vkey in (
        ("M5 clipped", "M5_clipped", "clipped"),
        ("M5 stretched", "M5_stretched", "stretched"),
    ):
        verdict = verdicts.get(key, {})
        print(
            f"{label:<34}{_fmt(base_m5.get(vkey)):<14}{_fmt(run_m5.get(vkey)):<14}"
            f"{_fmt(verdict.get('target')):<14}{str(verdict.get('verdict')):<10}"
        )
    run_gran = ((run or {}).get("M4_translation_qc_unresolved") or {}).get(
        "granularity"
    )
    base_gran = ((baseline or {}).get("M4_translation_qc_unresolved") or {}).get(
        "granularity"
    )
    print(
        f"\nM4 granularity: run={run_gran} (baseline={base_gran}) — "
        "unit vs row counts are not directly comparable"
    )


# ---------------------------------------------------------------- subcommands
def _refuse_cache_dir(cache_dir: str) -> None:
    if Path(cache_dir).name.startswith(CACHE_PREFIX) or cache_dir.startswith(
        CACHE_PREFIX
    ):
        return
    raise SystemExit(
        f"refusing --cache-dir {cache_dir!r}: eval runs must write to a scratch "
        f"cache whose name starts with {CACHE_PREFIX!r} (e.g. "
        f"{CACHE_PREFIX}sa1). The User's .vidub-cache/ and output/ are read-only."
    )


def _refuse_output_dir(output_dir: str, allow: bool) -> None:
    if allow:
        return
    resolved = Path(output_dir).resolve()
    try:
        resolved.relative_to(OUTPUT_ROOT.resolve())
    except ValueError:
        return
    raise SystemExit(
        f"refusing --output-dir {output_dir!r}: it is under the repo's "
        f"{OUTPUT_ROOT.name}/ deliverable tree. Pass --allow-output-dir to "
        "override, or use a scratch dir."
    )


def cmd_run(args: argparse.Namespace) -> int:
    _refuse_cache_dir(args.cache_dir)
    _refuse_output_dir(args.output_dir, args.allow_output_dir)

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "vidub",
        "run",
        args.url,
        "--cache-dir",
        str(cache_dir),
    ]
    if args.mt_engine:
        command += ["--mt-engine", args.mt_engine]
    command += list(args.extra or [])

    print("running:", " ".join(command))
    print("cwd:", output_dir)
    completed = subprocess.run(
        command, cwd=str(output_dir), check=False, text=True
    )
    if completed.returncode != 0:
        print(f"run failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode

    jobs = sorted(
        child
        for child in cache_dir.iterdir()
        if child.is_dir() and (child / "script.json").exists()
    )
    print(f"cache root: {cache_dir}")
    if jobs:
        print(f"resolved artifacts dir: {jobs[-1]}")
        manifest = jobs[-1] / "job.json"
        if manifest.exists():
            data = _load_json(manifest)
            stages = data.get("stages", {})
            for stage in ("ingest", "transcribe", "translate", "speak", "timing", "mix", "export"):
                if stage in stages:
                    print(f"  {stage}: {stages[stage]}")
    print(f"output: {output_dir}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    if not args.run and not args.baseline:
        print(
            "provide --run and/or --baseline (the artifacts dir of a job)",
            file=sys.stderr,
        )
        return 2
    run_report = metric_report(args.run) if args.run else None
    baseline_report = metric_report(args.baseline) if args.baseline else None
    result = {
        "run": run_report,
        "baseline": baseline_report,
        "baseline_reason": None if args.baseline else "no baseline cache given",
    }
    result["verdicts"] = (
        _verdicts(run_report, baseline_report) if run_report is not None else {}
    )

    if run_report is not None:
        print(f"run artifacts: {run_report['artifacts_dir']}")
        print("artifacts read:")
        for path in run_report["artifacts_read"]:
            print(f"  {path}")
        if run_report["artifacts_missing"]:
            print(f"artifacts missing: {run_report['artifacts_missing']}")
    else:
        print("run: null (no --run cache given)")
    if baseline_report is not None:
        print(f"baseline artifacts: {baseline_report['artifacts_dir']}")
        print("baseline artifacts read:")
        for path in baseline_report["artifacts_read"]:
            print(f"  {path}")
        if baseline_report["artifacts_missing"]:
            print(f"baseline artifacts missing: {baseline_report['artifacts_missing']}")
    else:
        print("baseline: null (no baseline cache given)")

    _print_table(result)

    if args.json:
        scores_path = Path(args.json)
    elif run_report is not None:
        scores_path = DEFAULT_SCORES
    else:
        # A baseline-only report must not clobber the A/B scores.json.
        scores_path = DEFAULT_SCORES.with_name("baseline.json")
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nscores written to {scores_path}")
    return 0


# ----------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_sentence_lock",
        description="SA-1 sentence-locked units: M1-M5 measurement harness.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline into a scratch cache")
    run.add_argument("--url", required=True, help="input URL or local media path")
    run.add_argument("--cache-dir", required=True, help="scratch cache (must start with .vidub-cache-)")
    run.add_argument("--output-dir", required=True, help="scratch output dir (used as CWD)")
    run.add_argument("--mt-engine", default=None, choices=["hunyuan", "envit5"])
    run.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="FLAG",
        help="extra flag passed through to 'vidub run' (repeatable)",
    )
    run.add_argument(
        "--allow-output-dir",
        action="store_true",
        help="allow an --output-dir under the repo's output/ tree",
    )
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="compute M1-M5 from artifacts (read-only)")
    score.add_argument("--run", default=None, help="run artifacts dir")
    score.add_argument("--baseline", default=None, help="optional baseline artifacts dir")
    score.add_argument("--json", default=None, help="where to write scores.json")
    score.set_defaults(func=cmd_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

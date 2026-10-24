"""Sentence-level translation units vs per-segment: reference A/B evaluation (TU-1-T4).

Runs the reference video (``.vidub-cache/downloads/youtube-a7TOameRqoY.mkv``)
twice — BEFORE = ``sentence_units=False`` (per-segment, today's behaviour) and
AFTER = ``sentence_units=True`` (default) — into **separate scratch cache dirs**
(never the User's ``.vidub-cache/`` jobs) and scores four pre-registered metrics:

1. unit summary (AFTER: ``translation_qc.json``'s ``units`` block);
2. mid-sentence boundaries at the engine input (primary target: 0);
3. translation QC (verdict / counts / unresolved / flags per 100 rows);
4. subtitles QC + the report's aggregate ``quality`` verdict.

Usage::

    ./.venv/Scripts/python.exe tools/research/eval_sentence_units.py run
    ./.venv/Scripts/python.exe tools/research/eval_sentence_units.py run --mt-engine hunyuan
    ./.venv/Scripts/python.exe tools/research/eval_sentence_units.py score
    ./.venv/Scripts/python.exe tools/research/eval_sentence_units.py rows

Everything is written under the scratch roots ``.vidub-cache-tu1/`` (caches + the
per-run output dirs) and ``eval-cache/tu1/`` (scores + run registry); the User's
``.vidub-cache/`` and ``output/`` are only read (the reference cache is *copied*
in, and each run chdirs into its own scratch work dir so ``output/`` lands there).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUT = ROOT / ".vidub-cache" / "downloads" / "youtube-a7TOameRqoY.mkv"
REFERENCE_CACHE = ROOT / ".vidub-cache"
SCRATCH = ROOT / ".vidub-cache-tu1"
SCORES_DIR = ROOT / "eval-cache" / "tu1"
SCORES_PATH = SCORES_DIR / "scores.json"
RUNS_PATH = SCORES_DIR / "runs.json"

PHASES = ("before", "after")
SENTENCE_UNITS = {"before": False, "after": True}
FORCE_STAGES = ["translate", "speak", "timing", "mix", "export"]

# The three screenshot fragments used for the side-by-side row table (matched by
# a distinctive substring of the source row, so it works on the canonical a7
# segmentation as well as the exact screenshot wording).
SCREENSHOT_FRAGMENTS = (
    "Meanwhile, these open-source models are sitting",
    "actually running",
    "small house",
)


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
def _ollama_has_model(host: str, model: str) -> tuple[bool, str]:
    """Best-effort probe of the local Ollama server for ``model`` (never raises)."""
    import urllib.request

    base = (host or "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - unavailable is a valid outcome
        return False, f"Ollama unreachable ({exc})"
    names = [
        str(entry.get("name", ""))
        for entry in (data.get("models") or [])
        if isinstance(entry, dict)
    ]
    wanted_base = model.split(":", 1)[0]
    if model in names or any(name.split(":", 1)[0] == wanted_base for name in names):
        return True, f"model {model} available"
    return False, f"model {model} not pulled"


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


def _run_phase(
    mt_engine: str,
    refine: bool,
    phase: str,
    video_hash: str,
) -> dict[str, Any]:
    from vidub.config import JobConfig
    from vidub.pipeline import run_pipeline

    key = f"{mt_engine}-{phase}"
    cache_dir = SCRATCH / f"cache-{key}"
    work = _work_dir(key)
    work.mkdir(parents=True, exist_ok=True)

    _seed_cache(cache_dir, video_hash)
    cfg = JobConfig(
        cache_dir=str(cache_dir),
        mt_engine=mt_engine,
        sentence_units=SENTENCE_UNITS[phase],
        hunyuan_refine=refine,
    )

    cwd = Path.cwd()
    os.chdir(work)
    try:
        run_pipeline(str(INPUT), cfg, resume=True, force=list(FORCE_STAGES))
    finally:
        os.chdir(cwd)

    return {
        "key": key,
        "mt_engine": mt_engine,
        "phase": phase,
        "sentence_units": SENTENCE_UNITS[phase],
        "refine": refine,
        "cache_dir": str(cache_dir),
        "work_dir": str(work),
        "status": "ok",
    }


def cmd_run(args: argparse.Namespace) -> int:
    video_hash = _reference_hash()
    engine = args.mt_engine

    if engine == "hunyuan":
        from vidub.config import DEFAULT_HUNYUAN_HOST, DEFAULT_HUNYUAN_MODEL

        ready, detail = _ollama_has_model(DEFAULT_HUNYUAN_HOST, DEFAULT_HUNYUAN_MODEL)
        print(f"hunyuan pre-flight: {detail}")
        if not ready:
            runs = _load_runs()
            for phase in PHASES:
                key = f"{engine}-{phase}"
                runs[key] = {
                    "key": key,
                    "mt_engine": engine,
                    "phase": phase,
                    "status": "unavailable (Ollama down)",
                    "detail": detail,
                }
            _dump_json(RUNS_PATH, runs)
            print("recorded unavailable (Ollama down); nothing to run")
            return 0

    runs = _load_runs()
    for phase in PHASES:
        key = f"{engine}-{phase}"
        print(f"=== running {key} (sentence_units={SENTENCE_UNITS[phase]}) ===")
        try:
            entry = _run_phase(engine, args.refine, phase, video_hash)
        except Exception as exc:  # noqa: BLE001 - record and continue honestly
            entry = {
                "key": key,
                "mt_engine": engine,
                "phase": phase,
                "sentence_units": SENTENCE_UNITS[phase],
                "refine": args.refine,
                "status": f"error ({exc})",
            }
            print(f"{key}: ERROR {exc}")
        else:
            print(f"{key}: ok -> {entry['cache_dir']}")
        runs[key] = entry
        _dump_json(RUNS_PATH, runs)

    print(f"runs written to {RUNS_PATH}")
    return 0


def _load_runs() -> dict[str, Any]:
    if RUNS_PATH.exists():
        return _load_json(RUNS_PATH)
    return {}


# ----------------------------------------------------------- scoring helpers
def _engine_inputs(
    artifacts: Path, sentence_units: bool
) -> tuple[list[str], list[list[int]], list[Any]]:
    """The texts the engine received, their segment spans, and (AFTER) the units."""
    from vidub.utils.sentences import group_units

    segments = _load_json(artifacts / "segments.json")
    script = _load_json(artifacts / "script.json")
    passthrough = {
        index
        for index, row in enumerate(script)
        if isinstance(row, dict) and row.get("passthrough")
    }

    if not sentence_units:
        indices = [i for i in range(len(segments)) if i not in passthrough]
        inputs = [str(segments[i].get("text", "")) for i in indices]
        return inputs, [[i] for i in indices], []

    rows = []
    for index, seg in enumerate(segments):
        speaker = seg.get("speaker", "spk0")
        if index in passthrough:
            # Mirror the stage: a unique speaker isolates a passthrough row.
            speaker = f"__passthrough_{index}__"
        rows.append(
            {
                "text": str(seg.get("text", "")),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "speaker": speaker,
            }
        )
    units = group_units(rows)
    active = [unit for unit in units if unit.indices[0] not in passthrough]
    inputs = [unit.source for unit in active]
    return inputs, [list(unit.indices) for unit in active], units


def _mid_sentence_count(inputs: list[str]) -> int:
    """Boundaries where the left does not end a sentence and the right starts
    lowercase / a digit (the pre-registered primary metric)."""
    from vidub.utils.sentences import ends_sentence

    count = 0
    for left, right in zip(inputs, inputs[1:]):
        stripped = right.lstrip()
        if not stripped:
            continue
        first = stripped[0]
        if (not ends_sentence(left)) and (first.isdigit() or first.islower()):
            count += 1
    return count


def _mid_sentence_breakdown(
    inputs: list[str], spans: list[list[int]], segments: list[dict]
) -> dict[str, int]:
    """Classify each mid-sentence boundary: speaker change / long pause / cap."""
    from vidub.utils.sentences import SENTENCE_GAP_S, ends_sentence

    breakdown = {"speaker": 0, "gap": 0, "cap": 0}
    for position in range(len(inputs) - 1):
        left, right = inputs[position], inputs[position + 1]
        stripped = right.lstrip()
        if not stripped:
            continue
        first = stripped[0]
        if ends_sentence(left) or not (first.isdigit() or first.islower()):
            continue
        last_index = spans[position][-1]
        next_index = spans[position + 1][0]
        if segments[last_index].get("speaker") != segments[next_index].get("speaker"):
            breakdown["speaker"] += 1
        elif (
            float(segments[next_index].get("start", 0.0))
            - float(segments[last_index].get("end", 0.0))
            >= SENTENCE_GAP_S
        ):
            breakdown["gap"] += 1
        else:
            breakdown["cap"] += 1
    return breakdown


def _phase_metrics(key: str, run: dict[str, Any], video_hash: str) -> dict[str, Any]:
    artifacts = _artifacts_dir(Path(run["cache_dir"]), video_hash)
    sentence_units = bool(run["sentence_units"])

    metrics: dict[str, Any] = {"key": key, "sentence_units": sentence_units}

    # 1. unit summary
    translation_qc = _load_json(artifacts / "translation_qc.json")
    metrics["unit_summary"] = translation_qc.get("units") or {"mode": "segment"}

    # 2. engine inputs + mid-sentence boundaries
    inputs, spans, units = _engine_inputs(artifacts, sentence_units)
    segments = _load_json(artifacts / "segments.json")
    metrics["engine_inputs"] = len(inputs)
    metrics["mid_sentence_inputs"] = _mid_sentence_count(inputs)
    metrics["mid_sentence_breakdown"] = _mid_sentence_breakdown(
        inputs, spans, segments
    )
    if sentence_units and units:
        metrics["reconstructed_units"] = len(units)
        metrics["reconstructed_segments"] = sum(len(u.indices) for u in units)
        block = translation_qc.get("units") or {}
        metrics["units_consistent"] = (
            block.get("units") == len(units)
            and block.get("segments") == sum(len(u.indices) for u in units)
        )

    # 3. translation QC
    counts = translation_qc.get("counts", {})
    ok = int(counts.get("ok", 0))
    total = sum(int(v) for v in counts.values())
    script = _load_json(artifacts / "script.json")
    rows = len(script)
    metrics["translation_qc"] = {
        "verdict": translation_qc.get("verdict"),
        "counts": counts,
        "unresolved": total - ok,
        "rows": rows,
        "flags_per_100_rows": round((total - ok) / max(rows, 1) * 100, 2),
    }

    # 4. subtitles QC + aggregate quality verdict
    subtitles_path = artifacts / "subtitles_qc.json"
    subtitles = _load_json(subtitles_path) if subtitles_path.exists() else {}
    metrics["subtitles_qc"] = {
        "cues_in": subtitles.get("cues_in"),
        "cues_out": subtitles.get("cues_out"),
        "split": subtitles.get("split"),
        "violations": subtitles.get("violations"),
        "cps_max_observed": subtitles.get("cps_max_observed"),
    }
    report_path = _report_path(key)
    report = _load_json(report_path) if report_path.exists() else {}
    metrics["quality_verdict"] = (report.get("quality") or {}).get("verdict")

    return metrics


def _score_engine(engine: str, runs: dict[str, Any], video_hash: str) -> dict[str, Any]:
    engine_runs: dict[str, Any] = {}
    for phase in PHASES:
        key = f"{engine}-{phase}"
        run = runs.get(key)
        if run is None:
            engine_runs[phase] = {"status": "missing"}
        elif run.get("status") != "ok":
            engine_runs[phase] = {"status": run.get("status")}
        else:
            engine_runs[phase] = _phase_metrics(key, run, video_hash)
    return engine_runs


def _fmt(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)


def _print_engine_table(engine: str, scores: dict[str, Any]) -> None:
    before = scores.get("before", {})
    after = scores.get("after", {})
    print(f"\n=== {engine} ===")
    print(f"{'metric':<28}{'BEFORE':<26}{'AFTER':<26}")
    rows = [
        ("status", before.get("status", "ok"), after.get("status", "ok")),
        ("engine inputs", before.get("engine_inputs"), after.get("engine_inputs")),
        (
            "mid-sentence inputs",
            before.get("mid_sentence_inputs"),
            after.get("mid_sentence_inputs"),
        ),
        (
            "mid-sentence causes",
            _fmt(before.get("mid_sentence_breakdown")),
            _fmt(after.get("mid_sentence_breakdown")),
        ),
        ("unit summary", _fmt(before.get("unit_summary")), _fmt(after.get("unit_summary"))),
        (
            "translation verdict",
            (before.get("translation_qc") or {}).get("verdict"),
            (after.get("translation_qc") or {}).get("verdict"),
        ),
        (
            "translation unresolved",
            (before.get("translation_qc") or {}).get("unresolved"),
            (after.get("translation_qc") or {}).get("unresolved"),
        ),
        (
            "flags / 100 rows",
            (before.get("translation_qc") or {}).get("flags_per_100_rows"),
            (after.get("translation_qc") or {}).get("flags_per_100_rows"),
        ),
        (
            "subtitle cues out",
            (before.get("subtitles_qc") or {}).get("cues_out"),
            (after.get("subtitles_qc") or {}).get("cues_out"),
        ),
        (
            "subtitle cps max",
            (before.get("subtitles_qc") or {}).get("cps_max_observed"),
            (after.get("subtitles_qc") or {}).get("cps_max_observed"),
        ),
        ("quality verdict", before.get("quality_verdict"), after.get("quality_verdict")),
    ]
    for name, left, right in rows:
        print(f"{name:<28}{str(left):<26}{str(right):<26}")


def cmd_score(args: argparse.Namespace) -> int:
    runs = _load_runs()
    if not runs:
        print(f"no runs recorded ({RUNS_PATH}); run 'run' first", file=sys.stderr)
        return 1

    video_hash = _reference_hash()
    engines = sorted({entry.get("mt_engine") for entry in runs.values() if entry.get("mt_engine")})
    payload: dict[str, Any] = {
        "input": str(INPUT),
        "video_hash": video_hash,
        "engines": {},
    }
    for engine in engines:
        scores = _score_engine(engine, runs, video_hash)
        payload["engines"][engine] = scores
        if "before" in scores and "after" in scores:
            _print_engine_table(engine, scores)

    payload["primary_target_mid_sentence_inputs"] = 0
    _dump_json(SCORES_PATH, payload)
    print(f"\nscores written to {SCORES_PATH}")
    return 0


# --------------------------------------------------------------- rows command
def _match_fragments(script: list[dict[str, Any]]) -> list[int]:
    indices: list[int] = []
    for fragment in SCREENSHOT_FRAGMENTS:
        for index, row in enumerate(script):
            if index in indices:
                continue
            if fragment.lower() in str(row.get("en", "")).lower():
                indices.append(index)
                break
    if len(indices) != len(SCREENSHOT_FRAGMENTS):
        return [1, 2, 3]  # fall back to the screenshot's segment indices
    return indices


def cmd_rows(args: argparse.Namespace) -> int:
    engine = args.mt_engine
    runs = _load_runs()
    before = runs.get(f"{engine}-before")
    after = runs.get(f"{engine}-after")
    if not before or not after or before.get("status") != "ok" or after.get("status") != "ok":
        print(f"{engine} before/after runs are not both available", file=sys.stderr)
        return 1

    video_hash = _reference_hash()
    before_script = _load_json(_artifacts_dir(Path(before["cache_dir"]), video_hash) / "script.json")
    after_script = _load_json(_artifacts_dir(Path(after["cache_dir"]), video_hash) / "script.json")

    indices = _match_fragments(after_script)
    print(f"{engine}: matched screenshot segment indices: {indices}\n")
    for index in indices:
        left = before_script[index] if index < len(before_script) else {}
        right = after_script[index] if index < len(after_script) else {}
        print(f"--- segment {index} ---")
        print(f"EN     : {right.get('en') or left.get('en')}")
        print(f"BEFORE : {left.get('vi')}")
        print(f"AFTER  : {right.get('vi')}")
        print()
    return 0


# ---------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_sentence_units",
        description="A/B evaluate sentence-level translation units on the reference video.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the BEFORE/AFTER pipeline pair into scratch caches")
    run.add_argument("--mt-engine", choices=["envit5", "hunyuan"], default="envit5")
    run.add_argument(
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the Hunyuan refine pass (default: off, to isolate the unit change)",
    )
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="score the scratch caches and write scores.json")
    score.set_defaults(func=cmd_score)

    rows = sub.add_parser("rows", help="side-by-side screenshot rows (EN / BEFORE vi / AFTER vi)")
    rows.add_argument("--mt-engine", choices=["envit5", "hunyuan"], default="envit5")
    rows.set_defaults(func=cmd_rows)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

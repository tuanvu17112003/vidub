"""Pre-registered En->Vi benchmark harness for QUALITY-ROADMAP.md §7.6 (M5-T3).

Two subcommands, each run under its own interpreter:

    # generation (MAIN venv; engine deps live here)
    ./.venv/Scripts/python.exe tools/research/eval_hunyuan.py generate [--limit N] [--systems a,b]

    # scoring (SCRATCH venv; transquest pins an old transformers)
    ./.eval-venv/Scripts/python.exe tools/research/eval_hunyuan.py score

``generate`` translates one shared corpus with every system
(``envit5``, ``qwen``, ``hunyuan-q6``, ``hunyuan-q4``, ``refine-envit5``,
``refine-hunyuan`` and, when a clean Chimera GGUF verifies, ``refine-fusion-hunyuan``)
and writes ``eval-cache/hunyuan/<system>.json`` (``[{"src", "hyp", "seconds"}]``).
Backends that are down are skipped with an explicit log line and recorded in
``meta.json`` -- never retried silently.

``score`` reads those JSONs and writes ``eval-cache/hunyuan/scores.json`` with the
four pre-registered metrics: the deterministic tqc battery (flags per 100 rows),
TransQuest DA mean/median + paired win-rate vs envit5, the dubbing length-ratio
distribution, and throughput. It imports no engine module (stdlib + tqc only), so
it runs in the scratch venv.

Numbers are never fabricated: unavailable systems are reported as such.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval-cache" / "hunyuan"
CACHE_GLOB = str(ROOT / ".vidub-cache" / "*" / "segments.json")

HOST = "http://127.0.0.1:11434"
TAG_QWEN = "qwen2.5:7b"
TAG_HUNYUAN_Q6 = "hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K"
TAG_HUNYUAN_Q4 = "hf.co/tencent/HY-MT1.5-7B-GGUF:Q4_K_M"
TAG_CHIMERA = "hf.co/mradermacher/Hunyuan-MT-Chimera-7B-GGUF:Q4_K_M"

REFINE_SAMPLES = 3
SRC_LANG = "en"
TGT_LANG = "vi"
LENGTH_RATIO = 2.2
LENGTH_RATIO_MIN = 0.4
# Corpus cap (M5-T3): at most 250 rows total, but every curated case is kept;
# the cached sources are taken first in deterministic (path-then-index) order.
CORPUS_CAP = 250

# 12 curated hard cases (numbers, entities, tech terms) from the task spec.
CURATED: tuple[str, ...] = (
    "The RTX 3080 has 10 GB of VRAM.",
    "NVIDIA released the driver in September 2025.",
    "I tested 3 GPUs and the 4070 beat the 3060 by 25 percent.",
    "Subscribe to the channel for more tutorials.",
    "Dubbing took 2 hours and 30 minutes for a 12 minute video.",
    "The peasants with low-end PCs will feel this upgrade.",
    "Riddle me this: which GPU gives 60 fps at 1080p?",
    "Version 2.5.1 fixed the bug from March.",
    "He said: 'Vidub runs on Windows and Linux'.",
    "Buy the $499 model, not the $599 one.",
    "Upload the 4K video to YouTube at 8 pm.",
    "The encoder settings: x265, CRF 18, preset slow.",
)


# ------------------------------------------------------------------ helpers
def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def log(message: str) -> None:
    print(f"[eval_hunyuan] {message}", flush=True)


def load_corpus() -> list[str]:
    """Deterministic dedup corpus: every cached segment (>= 3 words) + curated.

    Files are read in sorted-path order; within a file, segment order is kept.
    The result is capped at :data:`CORPUS_CAP` rows, but all 12 curated cases
    are always included (they are appended after the cached sources).
    """
    cached: list[str] = []
    seen: set[str] = set()
    files = sorted(glob.glob(CACHE_GLOB))
    for path in files:
        try:
            segments = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log(f"WARN unreadable {path}: {exc}")
            continue
        if not isinstance(segments, list):
            continue
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text", "")).strip()
            if len(text.split()) < 3 or text in seen:
                continue
            seen.add(text)
            cached.append(text)
    curated = [text.strip() for text in CURATED if text.strip() not in seen]
    room = max(0, CORPUS_CAP - len(curated))
    corpus = cached[:room] + curated
    log(
        f"corpus: {len(files)} segments.json file(s), {len(cached)} cached "
        f"(>=3 words), {len(curated)} curated -> {len(corpus)} row(s)"
    )
    return corpus


# ------------------------------------------------------------- ollama probes
def _http_get_json(url: str, timeout: float = 4.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001 - probe only
        return None


def ollama_version() -> str | None:
    data = _http_get_json(f"{HOST}/api/version")
    return str(data.get("version")) if data else None


def ollama_cli_models() -> set[str]:
    """Model names from ``ollama list`` (falls back to ``/api/tags``)."""
    try:
        proc = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=30
        )
        if proc.returncode == 0:
            names = set()
            for line in proc.stdout.splitlines()[1:]:
                parts = line.split()
                if parts:
                    names.add(parts[0])
            return names
    except (OSError, subprocess.SubprocessError):
        pass
    data = _http_get_json(f"{HOST}/api/tags") or {}
    return {
        str(item.get("name", ""))
        for item in (data.get("models") or [])
        if isinstance(item, dict)
    }


def ollama_ps() -> str:
    """Raw ``ollama ps`` output (processor split / size), or an error string."""
    try:
        proc = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=30
        )
        return (proc.stdout or proc.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable ({exc})"


# ------------------------------------------------------------- generation
def _rows_from_batch(corpus: list[str], hyps: list[str], elapsed: float) -> list[dict]:
    per = elapsed / max(len(corpus), 1)
    rows: list[dict] = []
    for index, src in enumerate(corpus):
        hyp = str(hyps[index]) if index < len(hyps) else ""
        rows.append({"src": src, "hyp": hyp, "seconds": round(per, 4)})
    return rows


def _empty_rows(corpus: list[str], reason: str) -> list[dict]:
    log(f"WARN generation failed ({reason}); recording empty outputs")
    return [{"src": src, "hyp": "", "seconds": 0.0} for src in corpus]


def _save_system(rows: list[dict], meta: dict, name: str, extra: dict | None = None) -> None:
    _write_json(EVAL_DIR / f"{name}.json", rows)
    entry = {"rows": len(rows), "status": "ok"}
    if extra:
        entry.update(extra)
    meta["systems"][name] = entry
    log(f"saved {name}: {len(rows)} row(s)")


def _skip(meta: dict, name: str, reason: str) -> None:
    meta["systems"][name] = {"rows": 0, "status": "skipped", "reason": reason}
    log(f"SKIP {name}: {reason}")


def _new_hunyuan(model: str):  # noqa: ANN202 - engine imported lazily
    from vidub.engines.mt.hunyuan_mt import HunyuanMTTranslator

    return HunyuanMTTranslator(model=model, host=HOST, timeout=900.0)


def _new_ollama(model: str):  # noqa: ANN202
    from vidub.engines.mt.ollama_mt import OllamaTranslator

    return OllamaTranslator(model=model, host=HOST, timeout=900.0)


def _new_envit5():  # noqa: ANN202
    from vidub.engines.mt.envit5_mt import Envit5Translator

    return Envit5Translator()


def _translate_batch(engine, corpus: list[str], contextual: bool) -> tuple[list[str], float]:
    start = time.perf_counter()
    if contextual:
        hyps = engine.translate_contextual(corpus, SRC_LANG, TGT_LANG)
    else:
        hyps = engine.translate(corpus, SRC_LANG, TGT_LANG)
    elapsed = time.perf_counter() - start
    return [str(h) for h in hyps], elapsed


def _run_refine(
    corpus: list[str], base_hyps: list[str], base_engine, generator, fusion
) -> tuple[list[dict], float, dict]:
    """Apply the tqc refiner contract to base rows; return rows/elapsed/meta."""
    from vidub.utils.refine import make_refiner
    from vidub.utils.tqc import run_battery

    envit5 = _new_envit5()
    refiner = make_refiner(
        generator,
        fusion=fusion,
        envit5=envit5,
        samples=REFINE_SAMPLES,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        glossary={},
    )
    rows = [
        {
            "start": 0.0,
            "end": 0.0,
            "speaker": "spk0",
            "en": src,
            "vi": hyp,
            "truncated": False,
        }
        for src, hyp in zip(corpus, base_hyps)
    ]
    start = time.perf_counter()
    rows, artifact = run_battery(
        rows, {}, engine=base_engine, src_lang=SRC_LANG, tgt_lang=TGT_LANG,
        refiner=refiner,
    )
    elapsed = time.perf_counter() - start
    outcomes = artifact.get("refine", {}).get("outcomes", [])
    action_counts: dict[str, int] = {}
    for outcome in outcomes:
        action_counts[outcome["action"]] = action_counts.get(outcome["action"], 0) + 1
    meta = {
        "counts_after": artifact["counts"],
        "actions": action_counts,
        "review_after": sum(
            value for label, value in artifact["counts"].items() if label != "ok"
        ),
    }
    hyps = [str(row["vi"]) for row in rows]
    return _rows_from_batch(corpus, hyps, elapsed), elapsed, meta


def cmd_generate(args: argparse.Namespace) -> int:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    if args.limit:
        corpus = corpus[: args.limit]
    log(f"corpus size: {len(corpus)} unique En source segments")
    _write_json(EVAL_DIR / "corpus.json", corpus)

    version = ollama_version()
    models = ollama_cli_models()
    meta: dict = {
        "corpus_size": len(corpus),
        "host": HOST,
        "ollama_version": version,
        "ollama_models": sorted(models),
        "systems": {},
        "vram": {},
    }
    # Accumulate across chunked per-system runs (``--systems``) so the final
    # meta.json describes every system and its VRAM probe.
    prior_meta_path = EVAL_DIR / "meta.json"
    if prior_meta_path.exists():
        try:
            prior_meta = _read_json(prior_meta_path)
            if isinstance(prior_meta, dict):
                meta["systems"] = dict(prior_meta.get("systems", {}))
                meta["vram"] = dict(prior_meta.get("vram", {}))
        except (OSError, json.JSONDecodeError):
            pass
    if version is None:
        log("WARN ollama server not reachable; LLM systems will be skipped")
    else:
        log(f"ollama {version}; models: {sorted(models)}")

    selected = {name.strip() for name in args.systems.split(",")} if args.systems else None

    def want(name: str) -> bool:
        return selected is None or name in selected

    def reuse(name: str) -> list[str] | None:
        """Base hypotheses from a complete existing ``<name>.json`` (or None).

        A partial re-run (``--systems``) must not recompute a finished system,
        but the refine systems still need its *outputs* as their baseline, so a
        system that is not selected is reused from disk when its row count
        matches the current corpus. Anything missing/short is ignored.
        """
        path = EVAL_DIR / f"{name}.json"
        if not path.exists():
            return None
        try:
            rows = _read_json(path)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(rows, list) or len(rows) != len(corpus):
            return None
        if not all(isinstance(row, dict) for row in rows):
            return None
        return [str(row.get("hyp", "")) for row in rows]

    def mark_reused(name: str, hyps: list[str], extra: dict | None = None) -> None:
        entry = {"rows": len(hyps), "status": "ok", "reused": True}
        if extra:
            entry.update(extra)
        meta["systems"][name] = entry
        log(f"reuse {name}: {len(hyps)} row(s) from existing output")

    base_hyps: dict[str, list[str]] = {}
    envit5_engine = None
    hunyuan_q6 = None
    hunyuan_q4 = None

    # --- envit5 (default baseline) -------------------------------------------
    if want("envit5"):
        try:
            envit5_engine = _new_envit5()
            hyps, elapsed = _translate_batch(envit5_engine, corpus, contextual=False)
            _save_system(_rows_from_batch(corpus, hyps, elapsed), meta, "envit5",
                         {"seconds_total": round(elapsed, 2)})
            base_hyps["envit5"] = hyps
        except Exception as exc:  # noqa: BLE001 - optional heavy engine
            _skip(meta, "envit5", f"envit5 unavailable: {exc}")
    else:
        prior = reuse("envit5")
        if prior is not None:
            base_hyps["envit5"] = prior
            try:
                envit5_engine = _new_envit5()
            except Exception as exc:  # noqa: BLE001 - fallback for round-trip
                log(f"WARN reuse envit5: engine unavailable ({exc})")
            mark_reused("envit5", prior)

    # --- qwen2.5:7b contextual ----------------------------------------------
    if want("qwen"):
        if version is None:
            _skip(meta, "qwen", "ollama server not reachable")
        elif TAG_QWEN not in models:
            _skip(meta, "qwen", f"model {TAG_QWEN} missing (ollama list)")
        else:
            try:
                hyps, elapsed = _translate_batch(_new_ollama(TAG_QWEN), corpus, True)
                _save_system(_rows_from_batch(corpus, hyps, elapsed), meta, "qwen",
                             {"seconds_total": round(elapsed, 2)})
            except Exception as exc:  # noqa: BLE001
                _save_system(_empty_rows(corpus, exc), meta, "qwen",
                             {"status": "error", "reason": str(exc)})
            meta["vram"]["qwen"] = ollama_ps()

    # --- Hunyuan Q6 ----------------------------------------------------------
    for name, tag, store in (
        ("hunyuan-q6", TAG_HUNYUAN_Q6, "q6"),
        ("hunyuan-q4", TAG_HUNYUAN_Q4, "q4"),
    ):
        if not want(name):
            prior = reuse(name)
            if prior is not None:
                if store == "q6":
                    base_hyps["hunyuan-q6"] = prior
                    try:
                        hunyuan_q6 = _new_hunyuan(tag)
                    except Exception as exc:  # noqa: BLE001 - refine generator
                        log(f"WARN reuse {name}: engine unavailable ({exc})")
                else:
                    try:
                        hunyuan_q4 = _new_hunyuan(tag)
                    except Exception as exc:  # noqa: BLE001
                        log(f"WARN reuse {name}: engine unavailable ({exc})")
                mark_reused(name, prior, {"model": tag})
            continue
        if version is None:
            _skip(meta, name, "ollama server not reachable")
            continue
        if tag not in models:
            _skip(meta, name, f"model {tag} missing (ollama list)")
            continue
        try:
            engine = _new_hunyuan(tag)
            hyps, elapsed = _translate_batch(engine, corpus, contextual=True)
            _save_system(_rows_from_batch(corpus, hyps, elapsed), meta, name,
                         {"seconds_total": round(elapsed, 2), "model": tag})
            if store == "q6":
                hunyuan_q6 = engine
                base_hyps["hunyuan-q6"] = hyps
            else:
                hunyuan_q4 = engine
        except Exception as exc:  # noqa: BLE001
            _save_system(_empty_rows(corpus, exc), meta, name,
                         {"status": "error", "reason": str(exc), "model": tag})
        meta["vram"][name] = ollama_ps()

    # --- refine systems ------------------------------------------------------
    if want("refine-envit5"):
        if hunyuan_q6 is None:
            _skip(meta, "refine-envit5", "hunyuan-q6 generator unavailable")
        elif "envit5" not in base_hyps:
            _skip(meta, "refine-envit5", "envit5 base unavailable")
        else:
            try:
                rows, elapsed, extra = _run_refine(
                    corpus, base_hyps["envit5"], envit5_engine, hunyuan_q6, None
                )
                _save_system(rows, meta, "refine-envit5",
                             {"seconds_total": round(elapsed, 2), **extra})
            except Exception as exc:  # noqa: BLE001
                _skip(meta, "refine-envit5", f"refine failed: {exc}")

    if want("refine-hunyuan"):
        if hunyuan_q6 is None:
            _skip(meta, "refine-hunyuan", "hunyuan-q6 generator unavailable")
        elif "hunyuan-q6" not in base_hyps:
            _skip(meta, "refine-hunyuan", "hunyuan-q6 base unavailable")
        else:
            try:
                rows, elapsed, extra = _run_refine(
                    corpus, base_hyps["hunyuan-q6"], hunyuan_q6, hunyuan_q6, None
                )
                _save_system(rows, meta, "refine-hunyuan",
                             {"seconds_total": round(elapsed, 2), **extra})
            except Exception as exc:  # noqa: BLE001
                _skip(meta, "refine-hunyuan", f"refine failed: {exc}")

    # --- optional Chimera fusion --------------------------------------------
    if want("refine-fusion-hunyuan"):
        if hunyuan_q6 is None:
            _skip(meta, "refine-fusion-hunyuan", "hunyuan-q6 generator unavailable")
        elif version is None or TAG_CHIMERA not in models:
            _skip(
                meta,
                "refine-fusion-hunyuan",
                f"Chimera fusion: unavailable (model {TAG_CHIMERA} missing)",
            )
        else:
            try:
                chimera = _new_hunyuan(TAG_CHIMERA)
                smoke = [
                    chimera.fuse(src, ["Xin chào", "Chào bạn"], SRC_LANG, TGT_LANG)
                    for src in (
                        "Hello world.",
                        "The GPU costs 300 dollars.",
                        "NVIDIA released a driver.",
                    )
                ]
                if not all(text.strip() for text in smoke):
                    raise RuntimeError("fusion smoke test returned empty output")
                smoke_seconds = None
                rows, elapsed, extra = _run_refine(
                    corpus, base_hyps["hunyuan-q6"], hunyuan_q6, hunyuan_q6, chimera
                )
                _save_system(rows, meta, "refine-fusion-hunyuan",
                             {"seconds_total": round(elapsed, 2),
                              "fusion_model": TAG_CHIMERA, "smoke": smoke, **extra})
                meta["vram"]["refine-fusion-hunyuan"] = ollama_ps()
            except Exception as exc:  # noqa: BLE001
                _skip(meta, "refine-fusion-hunyuan",
                      f"Chimera fusion: unavailable ({exc})")

    meta["vram"]["final"] = ollama_ps()
    _write_json(EVAL_DIR / "meta.json", meta)
    log("generate complete")
    return 0

# ------------------------------------------------------------------ scoring
def _battery_rows(rows: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    from vidub.utils.tqc import score_rows

    trows = [
        {"en": str(r.get("src", "")), "vi": str(r.get("hyp", "")), "truncated": False}
        for r in rows
    ]
    results, summary = score_rows(trows, {}, None, SRC_LANG, TGT_LANG)
    return trows, summary, results


def _transquest_scores(pairs: list[list[str]]) -> list[float] | None:
    try:
        from transquest.algo.sentence_level.monotransquest.run_model import (
            MonoTransQuestModel,
        )
    except Exception as exc:  # noqa: BLE001 - scratch venv only
        log(f"WARN TransQuest unavailable: {exc}")
        return None
    model = MonoTransQuestModel(
        "xlmroberta",
        "TransQuest/monotransquest-da-multilingual",
        num_labels=1,
        use_cuda=False,
    )
    predictions, _raw = model.predict(pairs)
    return [float(value) for value in predictions]


def cmd_score(args: argparse.Namespace) -> int:
    if not (EVAL_DIR / "corpus.json").exists():
        log("error: no corpus.json; run generate first")
        return 1
    corpus: list[str] = _read_json(EVAL_DIR / "corpus.json")
    meta = _read_json(EVAL_DIR / "meta.json") if (EVAL_DIR / "meta.json").exists() else {}

    system_files = sorted(
        path
        for path in EVAL_DIR.glob("*.json")
        if path.stem not in {"corpus", "meta", "scores"}
    )
    data: dict[str, list[dict]] = {
        path.stem: _read_json(path) for path in system_files
    }
    log(f"scoring {len(data)} system(s): {sorted(data)}")

    report: dict = {"n_corpus": len(corpus), "systems": {}, "transquest": {}}
    labels_per_system: dict[str, dict[str, int]] = {}

    for name, rows in data.items():
        trows, summary, results = _battery_rows(rows)
        counts = summary["counts"]
        n = len(trows)
        flags = sum(value for label, value in counts.items() if label != "ok")
        empty = sum(1 for row in rows if not str(row.get("hyp", "")).strip())
        ratios = []
        for row in rows:
            src_words = max(len(str(row.get("src", "")).split()), 1)
            hyp_words = len(str(row.get("hyp", "")).split())
            ratios.append(hyp_words / src_words)
        mean_seconds = (
            statistics.fmean(float(r.get("seconds", 0.0)) for r in rows)
            if rows
            else 0.0
        )
        report["systems"][name] = {
            "n": n,
            "battery": {
                "counts": counts,
                "flags": flags,
                "flags_per_100": round(100.0 * flags / n, 2) if n else None,
                "omission": counts.get("omission", 0),
                "addition": counts.get("addition", 0),
                "empty_output": empty,
            },
            "length": {
                "mean_ratio": round(statistics.fmean(ratios), 4) if ratios else None,
                "share_gt_2_2": round(
                    sum(1 for r in ratios if r > LENGTH_RATIO) / n, 4
                ) if n else None,
                "share_lt_0_4": round(
                    sum(1 for r in ratios if r < LENGTH_RATIO_MIN) / n, 4
                ) if n else None,
            },
            "throughput": {"mean_seconds_per_row": round(mean_seconds, 4)},
        }
        labels_per_system[name] = counts

    # TransQuest: score every (src, hyp) pair once per system, on CPU.
    order = list(data)
    all_pairs: list[list[str]] = []
    spans: dict[str, tuple[int, int]] = {}
    for name in order:
        start = len(all_pairs)
        all_pairs.extend([[r["src"], str(r.get("hyp", ""))] for r in data[name]])
        spans[name] = (start, len(all_pairs))
    scores = _transquest_scores(all_pairs) if all_pairs else None

    if scores is not None:
        for name in order:
            start, end = spans[name]
            values = scores[start:end]
            report["transquest"][name] = {
                "mean": round(statistics.fmean(values), 4) if values else None,
                "median": round(statistics.median(values), 4) if values else None,
            }
        baseline = "envit5"
        if baseline in spans:
            b_start, b_end = spans[baseline]
            baseline_scores = scores[b_start:b_end]
            win_rate = {}
            for name in order:
                if name == baseline:
                    continue
                start, end = spans[name]
                values = scores[start:end]
                if len(values) != len(baseline_scores):
                    win_rate[name] = None
                    continue
                wins = sum(1 for a, b in zip(values, baseline_scores) if a > b)
                win_rate[name] = round(wins / len(values), 4)
            report["win_rate_vs_envit5"] = win_rate

    report["meta"] = meta.get("systems", {})
    report["vram"] = meta.get("vram", {})
    _write_json(EVAL_DIR / "scores.json", report)
    log("score complete -> eval-cache/hunyuan/scores.json")
    return 0


# --------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_hunyuan",
        description="Pre-registered En->Vi benchmark harness (QUALITY-ROADMAP §7.6).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="translate the corpus with every system")
    p_gen.add_argument("--limit", type=int, default=0, help="use only the first N rows")
    p_gen.add_argument(
        "--systems",
        default="",
        help="comma-separated subset to run (default: all)",
    )
    p_gen.set_defaults(func=cmd_generate)

    p_score = sub.add_parser("score", help="score generated systems (scratch venv)")
    p_score.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Make the repo importable regardless of the interpreter/cwd used.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

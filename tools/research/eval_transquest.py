"""Reference-free QE validation harness for QUALITY-ROADMAP.md decision D6.

Scores a controlled set of En->Vi pairs with TransQuest's MonotransQuest
reference-free quality-estimation model and prints a JSON summary (mean score per
class, separation, AUC, best-separating threshold).

RUNS UNDER THE SCRATCH INTERPRETER ONLY (transquest pins an old transformers):

    ./.eval-venv/Scripts/python.exe tools/research/eval_transquest.py

The main ``.venv`` never sees this script's dependency: ``transquest`` is
deliberately absent from ``pyproject.toml``.

Test set construction:
  * "good" pairs are real (en, vi) rows pulled from the committed demo/sample
    ``output/*/script.tsv`` files (complete, diacritic-bearing translations);
  * "bad" pairs are derived from those same pairs by one deterministic
    corruption each: number swap, clause drop, glossary-term swap, entity drop.

The corruption set is synthetic and small, so the numbers bound TransQuest's
*relative* behaviour on Vi, not its absolute calibration (see
``docs/transquest-validation.md`` for the caveats and verdict).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODEL_TYPE = "xlmroberta"
MODEL_NAME = "TransQuest/monotransquest-da-multilingual"

# Real source/vi pairs (the script.tsv files named in the task spec).
PAIR_FILES = (
    ROOT / "output" / "youtube-a7TOameRqoY" / "script.tsv",
    ROOT / "output" / "demo_en2" / "script.tsv",
)
GLOSSARY_FILE = ROOT / "samples" / "glossary.example.tsv"

MAX_GOOD = 30
MIN_GOOD = 20
MIN_BAD = 20

# Vietnamese-specific letters + tone-marked vowels (mirrors vidub.utils.langcheck;
# duplicated so this evaluation script stays independent of the production pkg).
VN_CHARS = frozenset(
    "ăâêôơưđ"
    "áàảãạắằẳẵặấầẩẫậ"
    "éèẻẽẹếềểễệ"
    "íìỉĩị"
    "óòỏõọốồổỗộớờởỡợ"
    "úùủũụứừửữự"
    "ýỳỷỹỵ"
    "ĂÂÊÔƠƯĐÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬ"
    "ÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢ"
    "ÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)

_DIGIT_RE = re.compile(r"\d+")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9.+#-]{1,}\b")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


# --------------------------------------------------------------- test set
def load_rows(paths: tuple[Path, ...]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in paths:
        if not path.exists():
            print(f"warning: missing pair file {path}", file=sys.stderr)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            en, vi = parts[3].strip(), parts[4].strip()
            if en and vi:
                rows.append((en, vi))
    return rows


def has_diacritics(text: str) -> bool:
    return any(char in VN_CHARS for char in text)


def build_good(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    good: list[tuple[str, str]] = []
    for en, vi in rows:
        if len(en.split()) < 4 or len(vi.split()) < 3:
            continue
        if en == vi or not has_diacritics(vi):
            continue  # passthrough / untranslated
        # Prefer complete sentences (skip mid-sentence ASR fragments).
        if not en.rstrip().endswith((".", "!", "?")):
            continue
        good.append((en, vi))
        if len(good) >= MAX_GOOD:
            break
    return good


def load_glossary() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not GLOSSARY_FILE.exists():
        return mapping
    for line in GLOSSARY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "\t" not in line:
            continue
        term, translation = line.split("\t", 1)
        if term.strip() and translation.strip():
            mapping[term.strip()] = translation.strip()
    return mapping


# --------------------------------------------------------------- corruptions
def _corrupt_number(_en: str, vi: str) -> tuple[str, str] | None:
    match = _DIGIT_RE.search(vi)
    if not match:
        return None
    old = match.group(0)
    new = str(int(old) + 3) if old.isdigit() else "42"
    if new == old:
        new = "42"
    corrupted = vi[: match.start()] + new + vi[match.end() :]
    return corrupted, "number-swap"


def _corrupt_clause(_en: str, vi: str) -> tuple[str, str] | None:
    parts = [p for p in re.split(r"(?<=[,;.!?])\s+", vi) if p.strip()]
    if len(parts) >= 2:
        return " ".join(parts[:-1]).strip(), "clause-drop"
    words = vi.split()
    if len(words) >= 4:
        keep = max(2, math.ceil(len(words) * 0.6))
        return " ".join(words[:keep]), "clause-drop"
    return None


def _corrupt_term(_en: str, vi: str, glossary: dict[str, str]) -> tuple[str, str] | None:
    translations = [t for t in glossary.values()]
    for term, translation in glossary.items():
        if translation and translation.lower() in vi.lower():
            wrong = next(
                (t for t in translations if t.lower() != translation.lower()),
                "sai thuật ngữ",
            )
            idx = vi.lower().find(translation.lower())
            return vi[:idx] + wrong + vi[idx + len(translation) :], "term-swap"
    return None


def _corrupt_entity(en: str, vi: str) -> tuple[str, str] | None:
    words = en.split()
    candidates = [
        tok
        for tok in _ENTITY_RE.findall(en)
        if not (words and tok == words[0].strip(".,!?"))
    ]
    for token in candidates:
        idx = vi.lower().find(token.lower())
        if idx >= 0:
            return (vi[:idx] + vi[idx + len(token) :]).strip(), "entity-drop"
    return None


_KINDS = ("number", "clause", "term", "entity")


def corrupt(
    en: str, vi: str, index: int, glossary: dict[str, str]
) -> tuple[str, str, str]:
    """Return ``(bad_vi, corruption_kind)`` deterministically for pair ``index``."""
    order = [_KINDS[index % len(_KINDS)]] + [
        k for k in _KINDS if k != _KINDS[index % len(_KINDS)]
    ]
    for kind in order:
        if kind == "number":
            result = _corrupt_number(en, vi)
        elif kind == "clause":
            result = _corrupt_clause(en, vi)
        elif kind == "term":
            result = _corrupt_term(en, vi, glossary)
        else:
            result = _corrupt_entity(en, vi)
        if result is not None:
            bad_vi, label = result
            if bad_vi and bad_vi != vi:
                return bad_vi, label
    return _corrupt_clause(en, vi) or (vi + " và", "clause-drop")


# --------------------------------------------------------------- metrics
def auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC (labels: 1 = good, 0 = bad); ties get average ranks."""
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    n_pos, n_neg = len(positives), len(negatives)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ordered = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and scores[ordered[j + 1]] == scores[ordered[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[ordered[k]] = average
        i = j + 1
    rank_sum = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def best_threshold(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """Threshold (score >= t => good) maximizing accuracy; return (t, accuracy)."""
    best_t, best_acc = 0.0, -1.0
    for t in sorted(set(scores)):
        predicted = [1 if s >= t else 0 for s in scores]
        acc = sum(p == y for p, y in zip(predicted, labels)) / len(labels)
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc


def main() -> int:
    rows = load_rows(PAIR_FILES)
    good = build_good(rows)
    if len(good) < MIN_GOOD:
        print(
            f"error: only {len(good)} good pair(s) found (< {MIN_GOOD}); "
            f"checked {[str(p) for p in PAIR_FILES]}",
            file=sys.stderr,
        )
        return 1
    glossary = load_glossary()

    pairs: list[dict] = []
    for i, (en, vi) in enumerate(good):
        pairs.append({"kind": "good", "corruption": None, "src": en, "mt": vi})
        bad_vi, label = corrupt(en, vi, i, glossary)
        pairs.append({"kind": "bad", "corruption": label, "src": en, "mt": bad_vi})

    from transquest.algo.sentence_level.monotransquest.run_model import (
        MonoTransQuestModel,
    )

    use_cuda = False
    try:
        import torch

        use_cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - CPU is fine for validation
        use_cuda = False

    model = MonoTransQuestModel(
        MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=use_cuda
    )
    predictions, _raw = model.predict([[p["src"], p["mt"]] for p in pairs])
    scores = [float(s) for s in predictions]
    labels = [1 if p["kind"] == "good" else 0 for p in pairs]

    good_scores = [s for s, p in zip(scores, pairs) if p["kind"] == "good"]
    bad_scores = [s for s, p in zip(scores, pairs) if p["kind"] == "bad"]
    mean_good = sum(good_scores) / len(good_scores)
    mean_bad = sum(bad_scores) / len(bad_scores)
    threshold, accuracy = best_threshold(scores, labels)

    per_corruption: dict[str, dict[str, float]] = {}
    for score, pair in zip(scores, pairs):
        if pair["kind"] != "bad":
            continue
        key = str(pair["corruption"])
        bucket = per_corruption.setdefault(key, {"count": 0, "sum": 0.0})
        bucket["count"] += 1
        bucket["sum"] += score
    per_corruption = {
        key: {"count": int(v["count"]), "mean": round(v["sum"] / v["count"], 4)}
        for key, v in sorted(per_corruption.items())
    }

    summary = {
        "engine": "TransQuest MonotransQuest (MonoTransQuestModel)",
        "model_type": MODEL_TYPE,
        "model": MODEL_NAME,
        "device": "cuda" if use_cuda else "cpu",
        "pair_files": [str(p.relative_to(ROOT)) for p in PAIR_FILES],
        "num_pairs": len(pairs),
        "num_good": len(good_scores),
        "num_bad": len(bad_scores),
        "mean_good": round(mean_good, 4),
        "mean_bad": round(mean_bad, 4),
        "separation": round(mean_good - mean_bad, 4),
        "auc": round(auc(scores, labels), 4),
        "best_threshold": round(threshold, 4),
        "accuracy_at_threshold": round(accuracy, 4),
        "misclassified_good": sum(1 for s in good_scores if s < threshold),
        "misclassified_bad": sum(1 for s in bad_scores if s >= threshold),
        "per_corruption": per_corruption,
        "scores": [
            {
                "kind": p["kind"],
                "corruption": p["corruption"],
                "score": round(s, 4),
                "src": p["src"],
                "mt": p["mt"],
            }
            for p, s in zip(pairs, scores)
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out_path = os.environ.get("EVAL_OUT")
    if out_path:
        Path(out_path).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

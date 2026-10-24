"""Translation refine pass (QUALITY-ROADMAP.md §7.3 step 2, M5-T2).

After the deterministic tqc battery (:mod:`vidub.utils.tqc`) has scored and
corrected every row, rows still flagged ``mistranslation-risk`` /
``inconsistent`` / ``unnatural`` -- plus unresolved ``entity-error`` /
``terminology-error`` rows -- get ONE refine cycle. The cycle proposes fresh
candidates (K samples from a Hunyuan engine, plus the current vi, plus an
envit5 candidate), optionally fuses them with a Chimera model, and accepts the
winning candidate ONLY when it is STRICTLY better under the tqc labels (the
existing severity ordering in :func:`vidub.utils.tqc._is_better`). There are no
refine loops: exactly one cycle per run.

Stdlib-only orchestration: this module imports no engine class. The Hunyuan
generator, the optional fusion model and the envit5 fallback are dependency
injected by :mod:`vidub.stages.translate`, so ``tqc`` stays engine-free and the
whole pass is an opt-in no-op otherwise. Every winner is gated by the shared
deterministic validator
(:func:`vidub.engines.mt.validators._validate`, language/numbers/length): the
LLM only proposes, deterministic code decides.

Contract: :func:`make_refiner` returns ``refine(rows, final_entries) ->
(rows, final_entries, summary)``. ``run_battery`` calls it after its own
corrections and before computing the final counts, verdict and review packets;
the refiner mutates ``rows`` in place, updates the matching ``final_entries``
entries and returns a summary dict. Any exception the refiner raises propagates
to the caller (the stage wraps the pass so refine can never fail a run).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from vidub.engines.mt.validators import _validate
from vidub.utils.textclean import clean_hesitation
from vidub.utils.tqc import _is_better, _score_row, digits_of, extract_numbers
from vidub.utils.vnstyle import apply_vnstyle

__all__ = ["REFINE_LABELS", "make_refiner"]

log = logging.getLogger(__name__)

# Labels that trigger a refine cycle. ``ok`` and the deterministic-only labels
# (``number-error`` / ``omission`` / ``addition``) stay with the tqc corrections.
REFINE_LABELS: tuple[str, ...] = (
    "mistranslation-risk",
    "inconsistent",
    "unnatural",
    "entity-error",
    "terminology-error",
)


def _best_candidate(source: str, candidates: list[str]) -> str:
    """Deterministic best-of-K pick (needs no engine).

    Ranking, in order: (1) every source number's digits must be present;
    (2) the word-count ratio closest to 1.0 (dubbing fit); (3) the earliest
    candidate (stable). Fully deterministic.
    """
    source_numbers = [
        digits_of(token) for token in extract_numbers(source) if digits_of(token)
    ]
    src_words = max(len((source or "").split()), 1)

    def _key(item: tuple[int, str]) -> tuple[bool, float, int]:
        position, candidate = item
        digits = digits_of(candidate)
        numbers_ok = all(number in digits for number in source_numbers)
        ratio = len(candidate.split()) / src_words
        return (numbers_ok, -abs(ratio - 1.0), -position)

    return max(enumerate(candidates), key=_key)[1]


def make_refiner(
    generator: Any,
    fusion: Any = None,
    envit5: Any = None,
    samples: int = 3,
    src_lang: str = "en",
    tgt_lang: str = "vi",
    glossary: dict[str, str] | None = None,
    max_units: int | None = None,
) -> Callable[[list[dict], list[dict]], tuple[list[dict], list[dict], dict]]:
    """Build the refine callback injected into :func:`vidub.utils.tqc.run_battery`.

    ``generator`` is any object with ``sample_candidates(text, src, tgt, k)`` (a
    :class:`vidub.engines.mt.hunyuan_mt.HunyuanMTTranslator`); ``fusion`` an
    object with ``fuse(source, candidates, src, tgt)`` (Chimera; optional);
    ``envit5`` any MT engine with ``translate`` (optional). ``samples`` is K.
    ``src_lang`` / ``tgt_lang`` / ``glossary`` are captured for the winners'
    rescore and mirror how ``run_battery`` was called.

    ``max_units`` is the **cost budget**: at most that many still-flagged units
    are refined per run; further flagged units are recorded as
    ``refine-budget`` outcomes (text untouched) and counted in the summary's
    ``skipped_budget``. ``None`` means unbounded (the historical M5 behaviour)
    and keeps the summary dict byte-identical to before — the budget keys
    (``max_units`` / ``skipped_budget``) appear only when a budget is set.
    """
    captured_glossary = glossary or {}

    def refine(
        rows: list[dict], final_entries: list[dict]
    ) -> tuple[list[dict], list[dict], dict]:
        outcomes: list[dict] = []
        attempted = 0
        improved = 0
        skipped_budget = 0
        fusion_used = False

        for entry in final_entries:
            before = str(entry.get("label", "ok"))
            if before not in REFINE_LABELS:
                continue
            index = int(entry.get("index", -1))
            if not 0 <= index < len(rows):
                continue
            row = rows[index]
            if row.get("passthrough"):
                continue  # already user-language: never touched

            if max_units is not None and attempted >= max_units:
                skipped_budget += 1
                outcomes.append(
                    {
                        "index": index,
                        "action": "refine-budget",
                        "before": before,
                        "after": before,
                    }
                )
                continue

            attempted += 1
            source = str(row.get("en", ""))
            current = str(row.get("vi", ""))

            candidates = list(
                generator.sample_candidates(source, src_lang, tgt_lang, samples)
            )
            candidates.append(current)
            if envit5 is not None:
                try:
                    extra = envit5.translate([source], src_lang, tgt_lang)
                except Exception as exc:  # noqa: BLE001 - a candidate, never fatal
                    log.warning("refine: envit5 candidate failed (%s)", exc)
                    extra = None
                if extra:
                    candidates.append(str(extra[0]))

            if not candidates:
                outcomes.append(
                    {
                        "index": index,
                        "action": "refine-skipped",
                        "before": before,
                        "after": before,
                    }
                )
                continue

            used_fusion = bool(fusion is not None and len(candidates) >= 2)
            if used_fusion:
                winner = fusion.fuse(source, candidates, src_lang, tgt_lang)
                fusion_used = True
            else:
                winner = _best_candidate(source, candidates)

            winner = (winner or "").strip()
            if _validate(winner, source, src_lang, tgt_lang) is not None:
                outcomes.append(
                    {
                        "index": index,
                        "action": "refine-invalid",
                        "before": before,
                        "after": before,
                    }
                )
                continue

            # Mirror the stage's final passes (clean_hesitation -> vnstyle) so a
            # refine candidate cannot reintroduce hesitation or unsorted order.
            cleaned, _fixed = clean_hesitation(winner)
            cleaned, _styled = apply_vnstyle(cleaned)
            candidate_row = dict(row)
            candidate_row["vi"] = cleaned
            # low_confidence described the replaced text, so drop it from the
            # scoring copy before the rescore; otherwise an "unnatural" row
            # could never improve. The real row's flag is cleared as well when
            # the candidate is accepted below.
            candidate_row.pop("low_confidence", None)
            after = _score_row(index, candidate_row, captured_glossary, src_lang, tgt_lang)

            if _is_better(after["label"], before):
                rows[index]["vi"] = cleaned
                # Clear the real row's stale flag too: it described the replaced
                # text, and leaving it makes script.json carry low_confidence on
                # a now-good row (and a later rescore would re-flag it unnatural).
                row.pop("low_confidence", None)
                entry["label"] = after["label"]
                entry["signals"] = after["signals"]
                improved += 1
                outcomes.append(
                    {
                        "index": index,
                        "action": "refine-fusion" if used_fusion else "refine-best",
                        "before": before,
                        "after": after["label"],
                    }
                )
            else:
                # A valid candidate that is not STRICTLY better is never kept.
                outcomes.append(
                    {
                        "index": index,
                        "action": "refine-skipped",
                        "before": before,
                        "after": before,
                    }
                )

        summary = {
            "attempted": attempted,
            "improved": improved,
            "fusion_used": fusion_used,
            "outcomes": outcomes,
        }
        if max_units is not None:
            # Budget bookkeeping, recorded in translation_qc.json's ``refine``
            # block. Absent when unbounded so the M5 summary stays byte-identical.
            summary["max_units"] = max_units
            summary["skipped_budget"] = skipped_budget
        return rows, final_entries, summary

    return refine

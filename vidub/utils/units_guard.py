"""Unit completeness guard for the sentence-unit translate path (SA-1-T2 / Q-1-T1).

Pure, stdlib-only logic plus a thin, **never-raising** orchestrator that decides
whether one translation *unit*'s output dropped content or reproduced a
neighbour's content and, when it did, tries to repair it before the deterministic
passes run.

Signals (all derived from word counts, terminal punctuation and content-token
containment, see :func:`guard_signals` / :func:`contamination_signals`):

* ``omission-risk`` — the vi text is far shorter than the source (coverage below
  :data:`UNIT_COVERAGE_MIN`); an empty vi on a non-empty source always gets it.
* ``sentence-loss`` — a multi-word, multi-sentence source rendered with at least
  one fewer sentence and not longer than the source (the signature of a
  translation that dropped a whole clause/sentence).
* ``addition-risk`` — the vi text is far longer than the source (coverage above
  :data:`UNIT_COVERAGE_MAX`).
* ``context-bleed`` — the vi text reproduces a **neighbouring** unit's content
  (the contextual-template failure where the model translated the previous or
  following segment instead of its own).  Measured as neighbour *containment*
  (:func:`contamination_signals`), not Jaccard.
* ``sentence-addition`` — a verbose output (coverage at/above
  :data:`ADDITION_SENTENCE_COVERAGE`) that adds a whole sentence
  (``sentences_out > sentences_in``).

Calibration evidence (the User's broken unit ``[10,11,12]``): coverage ``0.725``
with ``3 -> 1`` sentences, so the :data:`UNIT_COVERAGE_MIN` floor *and* the
``sentence-loss`` signal both catch exactly that defect.

Q-1-T1 evidence (the contaminated hunyuan job
``326ea9c45ee94afbfe3186be9b798ee8a05fbf63``): unit 24's vi swallowed the whole
previous unit *and* most of the next one, yet the old guard reported no flags —
contamination *adds* content (coverage ``1.65`` stays below
:data:`UNIT_COVERAGE_MAX` ``2.6``) and the only sentence signal was *loss*.  The
containment signals above catch it, and the **direct template** (an engine
``translate`` call without neighbour context) is a repair lever for
:meth:`~vidub.engines.mt.hunyuan_mt.HunyuanMTTranslator.translate_contextual`
bleeding the neighbours into the prompt.

:func:`guard_units` never raises: every candidate attempt is wrapped, so a
raising engine only drops that candidate (with a ``log.warning``) and the
original text is kept. A guard failure degrades to "keep the original text and
record the flags". Candidates never *worsen* a unit: the original wins ties
(``guard_rank`` is only used to choose a strictly better candidate).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

__all__ = [
    "UNIT_COVERAGE_MIN",
    "UNIT_COVERAGE_MAX",
    "MIN_WORDS_FOR_SENTENCE_LOSS",
    "RESAMPLE_K",
    "TERMINAL_CHARS",
    "CONTAMINATION_OVERLAP",
    "ADDITION_SENTENCE_COVERAGE",
    "count_sentences",
    "contamination_signals",
    "guard_signals",
    "guard_signals_with_context",
    "guard_rank",
    "guard_units",
]

log = logging.getLogger(__name__)

# A vi text below this fraction of the source word count is at risk of having
# dropped content (calibration: the broken unit measured 0.725).
UNIT_COVERAGE_MIN = 0.75
# A vi text above this fraction of the source word count is a verbose addition.
UNIT_COVERAGE_MAX = 2.6
# Below this many source words a missing sentence is not a meaningful signal.
MIN_WORDS_FOR_SENTENCE_LOSS = 8
# How many extra candidates the engine is asked for when a unit is flagged.
RESAMPLE_K = 3

# A neighbour whose content tokens are contained at/above this share flags the
# unit as a context bleed (Q-1-T1: the User's unit 24 measured 1.00 against its
# previous unit and 0.70 against its next one).
CONTAMINATION_OVERLAP = 0.5
# A vi text this much longer than its source *with an extra sentence* is a
# sentence addition (below the UNIT_COVERAGE_MAX "far longer" threshold).
ADDITION_SENTENCE_COVERAGE = 1.4
# Content tokens shorter than this are dropped: "a"/"to"/"of"-length noise.
MIN_CONTENT_TOKEN_LEN = 3

# Terminal (sentence-ending) characters, matching vidub.utils.sentences.
TERMINAL_CHARS = ".!?…"

# A *content token* is a run of alphanumeric characters (``[^\W_]+``, so Unicode
# letters/digits, underscore excluded).  Matching runs, not characters, keeps a
# Vietnamese syllable like "không" one token.
_CONTENT_TOKEN_RE = re.compile(r"[^\W_]+")

# Flag groups used by :func:`guard_rank` to order defects.  Both content-critical
# flags are *drops*; the contamination flags are adds/bleeds.
_CONTENT_CRITICAL_FLAGS = ("omission-risk", "sentence-loss")
_CONTAMINATION_FLAGS = ("context-bleed", "sentence-addition")


def count_sentences(text: str) -> int:
    """Number of terminal characters (``.!?…``) in ``text``; ``0`` when empty.

    A sentence is counted by its terminal punctuation, not by a grammar model:
    this is deliberately simple, deterministic and stdlib-only.
    """
    return sum(1 for char in str(text or "") if char in TERMINAL_CHARS)


def _content_tokens(text: str) -> set[str]:
    """The *content tokens* of ``text`` (see :func:`contamination_signals`).

    Lowercase alphanumeric runs of length >= :data:`MIN_CONTENT_TOKEN_LEN`,
    deduplicated; the empty set for empty/whitespace text.
    """
    return {
        token
        for token in _CONTENT_TOKEN_RE.findall(str(text or "").lower())
        if len(token) >= MIN_CONTENT_TOKEN_LEN
    }


def _containment(neighbour: str, vi: str) -> float:
    """Share of ``neighbour``'s content tokens that also appear in ``vi``.

    Containment (intersection over the *neighbour's* token count), not Jaccard:
    a unit that swallowed a whole neighbour scores ``1.0`` even though it also
    carries its own extra content.  ``0.0`` when the neighbour has no content
    tokens (an empty neighbour can never flag a bleed).
    """
    neighbour_tokens = _content_tokens(neighbour)
    if not neighbour_tokens:
        return 0.0
    return len(neighbour_tokens & _content_tokens(vi)) / len(neighbour_tokens)


def contamination_signals(
    vi: str, previous_vi: str = "", next_vi: str = "", source: str = ""
) -> dict:
    """Neighbour-contamination signals for one unit translation.

    ``content tokens`` are the lowercase alphanumeric tokens of length >= 3
    (deduplicated) of a text.  ``overlap_prev`` / ``overlap_next`` are the
    **containment** of the previous / next unit's content tokens in ``vi`` —
    intersection over the neighbour's token count — so a unit that swallowed a
    neighbour scores near ``1.0`` (containment, **not** Jaccard, is used for
    exactly that reason: a bleed is small relative to the whole unit).

    Returns ``{"overlap_prev", "overlap_next", "flags"}`` where ``flags`` is an
    ordered subset of:

    * ``context-bleed`` — ``overlap_prev >= CONTAMINATION_OVERLAP`` or
      ``overlap_next >= CONTAMINATION_OVERLAP``;
    * ``sentence-addition`` — ``source`` was given and
      :func:`guard_signals` reports ``sentences_out > sentences_in`` with
      ``coverage >= ADDITION_SENTENCE_COVERAGE``.

    Pure and never-raising: empty/blank inputs yield ``0.0`` overlaps and no
    flags.
    """
    vi = str(vi or "")
    overlap_prev = _containment(previous_vi, vi)
    overlap_next = _containment(next_vi, vi)

    flags: list[str] = []
    if overlap_prev >= CONTAMINATION_OVERLAP or overlap_next >= CONTAMINATION_OVERLAP:
        flags.append("context-bleed")
    if source:
        base = guard_signals(source, vi)
        if (
            base["sentences_out"] > base["sentences_in"]
            and base["coverage"] >= ADDITION_SENTENCE_COVERAGE
        ):
            flags.append("sentence-addition")

    return {
        "overlap_prev": overlap_prev,
        "overlap_next": overlap_next,
        "flags": flags,
    }


def guard_signals(
    source: str, vi: str, *, min_coverage: float = UNIT_COVERAGE_MIN
) -> dict:
    """Completeness signals for one unit translation.

    Returns ``{"words_in", "words_out", "coverage", "sentences_in",
    "sentences_out", "flags"}`` where ``coverage`` is
    ``words_out / max(words_in, 1)`` and ``flags`` is an ordered subset of
    ``omission-risk`` / ``sentence-loss`` / ``addition-risk``:

    * ``omission-risk`` — ``coverage < min_coverage``, or an empty ``vi`` on a
      non-empty source;
    * ``sentence-loss`` — ``words_in >= MIN_WORDS_FOR_SENTENCE_LOSS`` and
      ``sentences_in >= 2`` and ``sentences_out <= sentences_in - 1`` and
      ``coverage < 1.10``;
    * ``addition-risk`` — ``coverage > UNIT_COVERAGE_MAX``.

    ``min_coverage`` is additive and **defaults to :data:`UNIT_COVERAGE_MIN`**
    (0.75), so every existing caller is unchanged. ISO-1-T3-C1 passes the tiered
    :func:`vidub.utils.isometry.coverage_floor` (0.65 for windows below 1.80 s)
    on the sentence-unit path, and the caller records which floor applied.

    This is the *source-completeness* half only; the neighbour-contamination
    flags live in :func:`contamination_signals` and are merged by
    :func:`guard_signals_with_context`.
    """
    source = str(source or "")
    vi = str(vi or "")
    words_in = len(source.split())
    words_out = len(vi.split())
    coverage = words_out / max(words_in, 1)
    sentences_in = count_sentences(source)
    sentences_out = count_sentences(vi)

    flags: list[str] = []
    if coverage < float(min_coverage) or (not vi.strip() and bool(source.strip())):
        flags.append("omission-risk")
    if (
        words_in >= MIN_WORDS_FOR_SENTENCE_LOSS
        and sentences_in >= 2
        and sentences_out <= sentences_in - 1
        and coverage < 1.10
    ):
        flags.append("sentence-loss")
    if coverage > UNIT_COVERAGE_MAX:
        flags.append("addition-risk")

    return {
        "words_in": words_in,
        "words_out": words_out,
        "coverage": coverage,
        "sentences_in": sentences_in,
        "sentences_out": sentences_out,
        "flags": flags,
    }


def guard_signals_with_context(
    source: str,
    vi: str,
    previous_vi: str = "",
    next_vi: str = "",
    *,
    min_coverage: float = UNIT_COVERAGE_MIN,
) -> dict:
    """``guard_signals`` plus the :func:`contamination_signals` flags.

    Returns the same dict as :func:`guard_signals` with ``overlap_prev`` /
    ``overlap_next`` added and any ``context-bleed`` / ``sentence-addition`` flag
    appended to ``flags`` (deduplicated, base flags first).  This is the wrapper
    ``guard_units`` scores units with; plain :func:`guard_signals` keeps its
    source-completeness-only contract for existing callers. ``min_coverage`` is
    forwarded to :func:`guard_signals` (additive, defaults to
    :data:`UNIT_COVERAGE_MIN`).
    """
    signals = guard_signals(source, vi, min_coverage=min_coverage)
    contamination = contamination_signals(vi, previous_vi, next_vi, source=source)
    flags = list(signals["flags"])
    for flag in contamination["flags"]:
        if flag not in flags:
            flags.append(flag)
    return {
        **signals,
        "flags": flags,
        "overlap_prev": contamination["overlap_prev"],
        "overlap_next": contamination["overlap_next"],
    }


def guard_rank(signals: dict) -> tuple:
    """Sortable "quality" of a candidate; **lower is better**.

    Ordering, worst-first for the second/third element (all ascending, so
    smaller means better): no flag beats any flag; among flagged candidates a
    *content-critical* drop (``omission-risk`` / ``sentence-loss``) ranks **worse**
    than a contamination flag (``context-bleed`` / ``sentence-addition``), which
    in turn ranks worse than a pure length flag (``addition-risk``); then fewer
    flags, then the coverage closest to ``1.0``.  So a candidate that fixes a
    bleed is preferred over one that leaves an omission, and a clean candidate is
    always preferred.  Every value is a plain ``int``/``bool``/``float`` so
    tuples compare deterministically; ties are broken by the candidate's
    position, where index ``0`` is the original (so the original wins ties).
    """
    flags = list(signals.get("flags", []))
    coverage = float(signals.get("coverage", 1.0))
    critical = sum(1 for flag in flags if flag in _CONTENT_CRITICAL_FLAGS)
    contamination = sum(1 for flag in flags if flag in _CONTAMINATION_FLAGS)
    return (len(flags) > 0, critical, contamination, len(flags), abs(coverage - 1.0))


def _record(
    position: int,
    signals: dict,
    action: str,
    candidates: int,
    *,
    after_signals: dict | None = None,
    include_overlap: bool = True,
) -> dict:
    """One guard record: the **detected** signals, the chosen action.

    ``indices`` is the unit's position in the batch (``[position]``); the caller
    maps it onto the unit's member row indices. ``candidates`` is the number of
    candidate translations compared (the original plus any direct/resample/
    fallback).  ``include_overlap`` is set when neighbour context was supplied:
    the record then also carries the before (``overlap_prev`` / ``overlap_next``)
    and after (``overlap_prev_after`` / ``overlap_next_after``) containment.
    ``flags`` are the **detected** (before) flags, so the record documents what
    the guard saw, not just what survived.
    """
    record: dict = {
        "indices": [position],
        "words_in": signals["words_in"],
        "words_out": signals["words_out"],
        "coverage": signals["coverage"],
        "sentences_in": signals["sentences_in"],
        "sentences_out": signals["sentences_out"],
        "flags": list(signals["flags"]),
        "action": action,
        "candidates": candidates,
    }
    if include_overlap:
        after = after_signals if after_signals is not None else signals
        record["overlap_prev"] = float(signals.get("overlap_prev", 0.0))
        record["overlap_next"] = float(signals.get("overlap_next", 0.0))
        record["overlap_prev_after"] = float(after.get("overlap_prev", 0.0))
        record["overlap_next_after"] = float(after.get("overlap_next", 0.0))
    return record


def _safe_sample(
    engine: object, source: str, src_lang: str, tgt_lang: str, k: int
) -> list[str]:
    """Up to ``k`` candidates from ``engine.sample_candidates`` (never raises)."""
    try:
        sampled = engine.sample_candidates(source, src_lang, tgt_lang, k)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - a broken engine drops its candidate
        log.warning("unit guard: sample_candidates raised (%s); candidate dropped", exc)
        return []
    return [str(text) for text in (sampled or []) if str(text or "").strip()]


def _safe_direct(
    engine: object, source: str, src_lang: str, tgt_lang: str
) -> str | None:
    """One direct-template translation from ``engine`` (never raises).

    ``engine.translate([source], ...)`` is the *non-contextual* path: for the
    Hunyuan engine it uses :func:`~vidub.engines.mt.hunyuan_mt._template_direct`
    (no neighbours), which is exactly the repair lever for a context bleed.
    Returns ``None`` on error or an empty/blank result, so the caller only drops
    the direct candidate.
    """
    try:
        out = engine.translate([source], src_lang, tgt_lang)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - a broken engine drops its candidate
        log.warning("unit guard: direct template raised (%s); candidate dropped", exc)
        return None
    if not out:
        return None
    text = str(out[0])
    return text if text.strip() else None


def _safe_fallback(
    fallback_engine: object, source: str, src_lang: str, tgt_lang: str
) -> str | None:
    """One translation from ``fallback_engine`` (never raises; ``None`` on error)."""
    try:
        out = fallback_engine.translate([source], src_lang, tgt_lang)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - a broken engine drops its candidate
        log.warning("unit guard: fallback engine raised (%s); candidate dropped", exc)
        return None
    return str(out[0]) if out else None


def _still_bad(signals: dict) -> bool:
    """True when the chosen candidate still carries a content-critical or bleed flag.

    ``addition-risk`` alone is not "still bad": a verbose-but-faithful unit is
    left to the deterministic length guard.  A drop
    (``omission-risk`` / ``sentence-loss``) **or** a contamination
    (``context-bleed`` / ``sentence-addition``) that survived every candidate is.
    """
    flags = signals.get("flags", [])
    return any(
        flag in _CONTENT_CRITICAL_FLAGS or flag in _CONTAMINATION_FLAGS
        for flag in flags
    )


def guard_units(
    sources: list[str],
    translations: list[str],
    engine: object,
    src_lang: str,
    tgt_lang: str,
    *,
    fallback_engine: object | None = None,
    k: int = RESAMPLE_K,
    neighbours: Sequence[tuple[str, str]] | None = None,
    min_coverage: Sequence[float] | float | None = None,
) -> tuple[list[str], list[dict]]:
    """Guard every unit, repairing what can be repaired; **never raises**.

    ``neighbours`` optionally supplies ``(previous_vi, next_vi)`` **raw engine
    translations** per source (empty strings at the ends).  When given, each
    unit is scored by :func:`guard_signals_with_context` so a translation that
    reproduced a neighbour gets the ``context-bleed`` / ``sentence-addition``
    flags and records carry the before/after overlaps.

    ``min_coverage`` is the additive per-unit omission floor (ISO-1-T3-C1): a
    single float for every unit, or a sequence parallel to ``sources`` (the
    tiered :func:`vidub.utils.isometry.coverage_floor`). ``None`` keeps every
    unit at :data:`UNIT_COVERAGE_MIN`, so existing callers are unchanged. The
    floor is applied to the original *and* every candidate, and the caller records
    which floor applied.

    Per unit: a clean score keeps the text (``action: "none"``). Otherwise
    candidates are built — the original, the **direct template**
    (``engine.translate([source], src_lang, tgt_lang)``, skipped when it is
    byte-identical to the original), up to ``k`` from
    ``engine.sample_candidates(source, src_lang, tgt_lang, k)`` when the engine
    exposes that method, and one
    ``fallback_engine.translate([source], src_lang, tgt_lang)`` when a fallback
    engine is given — and the best is picked by :func:`guard_rank` (the original
    wins ties). Every candidate attempt is wrapped, so a raising engine only
    drops that candidate.

    ``action`` is one of:

    * ``"none"`` — the original was clean (or only carried a lower-severity
      flag and no candidate improved on it);
    * ``"decontext"`` — the direct-template candidate was chosen (the
      contamination repair);
    * ``"resampled"`` — a ``sample_candidates`` candidate was chosen;
    * ``"fallback"`` — the fallback engine's candidate was chosen;
    * ``"flagged"`` — the best candidate still carries ``omission-risk`` /
      ``sentence-loss`` / ``context-bleed`` / ``sentence-addition`` (nothing
      fixed it; the caller forces the unit low-confidence).

    Returns ``(chosen_texts, records)`` with one chosen text and one record per
    source (in order).  When ``neighbours`` is falsy the records keep the
    original (overlap-free) shape, so existing callers are unchanged.
    """
    chosen: list[str] = []
    records: list[dict] = []
    neighbour_list = list(neighbours) if neighbours else []
    include_overlap = bool(neighbour_list)

    def _floor(position: int) -> float:
        if min_coverage is None:
            return UNIT_COVERAGE_MIN
        if isinstance(min_coverage, Sequence) and not isinstance(
            min_coverage, (str, bytes)
        ):
            if position < len(min_coverage):
                return float(min_coverage[position])
            return UNIT_COVERAGE_MIN
        return float(min_coverage)

    for position, raw_source in enumerate(sources):
        source = str(raw_source or "")
        floor = _floor(position)
        previous_vi = ""
        next_vi = ""
        if position < len(neighbour_list):
            pair = neighbour_list[position]
            previous_vi = str(pair[0] or "")
            next_vi = str(pair[1] or "")
        original = str(translations[position]) if position < len(translations) else ""
        signals = guard_signals_with_context(
            source, original, previous_vi, next_vi, min_coverage=floor
        )
        if not signals["flags"]:
            chosen.append(original)
            records.append(
                _record(position, signals, "none", 1, include_overlap=include_overlap)
            )
            continue

        candidates: list[tuple[str, dict, str]] = [(original, signals, "original")]
        # Direct-template candidate: no neighbour context, the repair lever for a
        # context bleed.  Skipped when byte-identical to the original — a tie
        # always keeps the original, so it could never be chosen and would only
        # inflate the candidate count.
        direct = _safe_direct(engine, source, src_lang, tgt_lang)
        if direct is not None and direct != original:
            candidates.append(
                (
                    direct,
                    guard_signals_with_context(
                        source, direct, previous_vi, next_vi, min_coverage=floor
                    ),
                    "direct",
                )
            )
        if hasattr(engine, "sample_candidates"):
            for text in _safe_sample(engine, source, src_lang, tgt_lang, k):
                candidates.append(
                    (
                        text,
                        guard_signals_with_context(
                            source, text, previous_vi, next_vi, min_coverage=floor
                        ),
                        "resampled",
                    )
                )
        if fallback_engine is not None:
            text = _safe_fallback(fallback_engine, source, src_lang, tgt_lang)
            if text is not None:
                candidates.append(
                    (
                        text,
                        guard_signals_with_context(
                            source, text, previous_vi, next_vi, min_coverage=floor
                        ),
                        "fallback",
                    )
                )

        best = min(
            range(len(candidates)),
            key=lambda index: (guard_rank(candidates[index][1]), index),
        )
        best_text, best_signals, best_origin = candidates[best]

        if _still_bad(best_signals):
            action = "flagged"
        elif best == 0:
            action = "none"
        elif best_origin == "fallback":
            action = "fallback"
        elif best_origin == "resampled":
            action = "resampled"
        else:
            action = "decontext"

        chosen.append(best_text)
        records.append(
            _record(
                position,
                signals,
                action,
                len(candidates),
                after_signals=best_signals,
                include_overlap=include_overlap,
            )
        )

    return chosen, records

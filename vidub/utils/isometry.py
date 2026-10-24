"""Isometric length policy (ISO-1-T1): per-unit character budgets from time.

Pure, stdlib-only policy for the isometric length-budgeting workstream (ISO-1
plan, ``WorkFlowMultiAgent/tasks/ISO-1-plan.md``). It answers four questions
about one sentence unit **before** any audio exists:

* how many Vietnamese characters does the unit's time window allow at a natural
  speaking pace (:func:`unit_budget`)?
* is a given translation over that budget (:func:`over_budget`)?
* how much faster than natural would it have to be spoken to fit
  (:func:`projected_stretch`)?
* is a *shorter* candidate translation acceptable — does it pass the engine
  validators and the completeness guard floors without reproducing a neighbour
  (:func:`accept_shrunk`)?

Nothing here talks to an engine, reads a file or writes a log line: identical
inputs always produce identical outputs, so the policy is cheap to unit-test and
safe to reuse from the translate stage.

**Guaranteed:** the budget is deterministic; its floor (:data:`MIN_BUDGET_CHARS`)
and ceiling (:data:`MAX_BUDGET_CHARS`) are always enforced, so a tiny or
mis-measured window can never demand a zero-length or absurd length; the
acceptance predicate can only ever reject a candidate that is empty, fails an
engine validator, breaks an omission floor, reproduces a neighbour or does not
actually shrink the projected stretch.

**NOT guaranteed:** :func:`projected_stretch` is an *estimate* derived from the
same :data:`CONCISE_CHARS_PER_SECOND` calibration as the budget — it is not a
synthesis measurement. Whether a translation really fits its window is decided
later by the measured TTS duration (:mod:`vidub.utils.fit`); this module only
plans and gates the shortening attempt.

The calibration constants are **imported, never re-declared**: the pace and the
minimum come from :mod:`vidub.utils.fit`, the omission floor from
:mod:`vidub.utils.units_guard`, and the acceptance validator from
:mod:`vidub.engines.mt.validators`, so this policy cannot drift from the numbers
the rest of the pipeline uses.
"""

from __future__ import annotations

from collections.abc import Sequence

from vidub.engines.mt.validators import _validate
from vidub.utils.fit import (
    CONCISE_CHARS_PER_SECOND,
    MIN_CONCISE_CHARS,
    SHORT_WINDOW_MAX,
)
from vidub.utils.units_guard import (
    UNIT_COVERAGE_MIN,
    contamination_signals,
    guard_signals,
)

__all__ = [
    "BUDGET_HEADROOM",
    "MIN_BUDGET_CHARS",
    "MAX_BUDGET_CHARS",
    "NATURAL_CHARS_PER_SECOND",
    "COVERAGE_FLOOR",
    "SHORT_COVERAGE_MIN",
    "coverage_floor",
    "unit_budget",
    "char_count",
    "over_budget",
    "projected_stretch",
    "budget_report",
    "accept_shrunk",
    "select_over_budget",
]

# Fraction of the window's natural-pace capacity a translation may use. The 10 %
# headroom lets FIT-1's "as-is" band (``SPEED_AS_IS = 1.10``) absorb normal pace
# variation instead of triggering a stretch (ISO-1 plan D1).
BUDGET_HEADROOM = 0.9

# Re-exports of the shared calibration — never re-declared literals:
# * NATURAL_CHARS_PER_SECOND == vidub.utils.fit.CONCISE_CHARS_PER_SECOND (20.0),
#   the pace calibrated on the User's tech video (range ~18-25 cps);
# * MIN_BUDGET_CHARS == vidub.utils.fit.MIN_CONCISE_CHARS (8), the smallest
#   budget a very short window may fall back to;
# * COVERAGE_FLOOR == vidub.utils.units_guard.UNIT_COVERAGE_MIN (0.75), the
#   omission floor a shorter candidate must stay above.
NATURAL_CHARS_PER_SECOND = CONCISE_CHARS_PER_SECOND
MIN_BUDGET_CHARS = MIN_CONCISE_CHARS
COVERAGE_FLOOR = UNIT_COVERAGE_MIN

# ISO-1-T3-C1: the User-approved *adaptive* coverage floor for tiny windows. A
# window below :data:`vidub.utils.fit.SHORT_WINDOW_MAX` (1.80 s) cannot hold a
# 0.75-coverage text at the natural pace at all, so the omission floor is lowered
# to :data:`SHORT_COVERAGE_MIN` **only there** — a bounded meaning/pacing trade,
# always recorded (`coverage_floor` + `short_floor_applied`). Normal windows keep
# :data:`COVERAGE_FLOOR` unchanged everywhere else.
SHORT_COVERAGE_MIN = 0.65

# A sanity ceiling on the budget (~30 s at the natural pace). A unit window this
# long is almost certainly a mis-measured boundary — the User's tech video's
# longest sentence unit is ~10 s — and the budget must never demand an absurd
# re-translation just because two timestamps are far apart.
MAX_BUDGET_CHARS = 600


def coverage_floor(
    window: float,
    *,
    short_window: float = SHORT_WINDOW_MAX,
    normal: float = UNIT_COVERAGE_MIN,
    short: float = SHORT_COVERAGE_MIN,
) -> float:
    """The omission floor for a unit of ``window`` seconds.

    A window **strictly below** ``short_window`` (default
    :data:`vidub.utils.fit.SHORT_WINDOW_MAX` = 1.80 s) returns the lowered
    :data:`SHORT_COVERAGE_MIN` (0.65); every other window returns the normal
    :data:`COVERAGE_FLOOR` (0.75). The exception is deliberate and bounded: a
    sub-1.8 s window cannot physically hold a 0.75-coverage text at the natural
    pace, so the alternative would be an unsatisfiable demand. Pure and
    deterministic. Callers that use it must record which floor applied.
    """
    return float(short) if float(window) < float(short_window) else float(normal)


def unit_budget(
    start: float,
    end: float,
    next_start: float | None = None,
    *,
    headroom: float = BUDGET_HEADROOM,
    chars_per_second: float = NATURAL_CHARS_PER_SECOND,
    include_following_silence: float = 0.0,
) -> int:
    """Character budget of a unit's time window at the natural speaking pace.

    The window is the unit's own span plus a bounded share of the following
    silence::

        window    = max(end - start, 0.0)
        silence   = max(next_start - end, 0.0)      # 0.0 when next_start is None
        share     = clamp(include_following_silence, 0.0, 1.0)
        effective = window + silence * share
        budget    = round(effective * chars_per_second * headroom)

    and the result is then clamped, in this order, to
    ``[MIN_BUDGET_CHARS, MAX_BUDGET_CHARS]``:

    * the **floor** (:data:`MIN_BUDGET_CHARS`, 8) keeps a very short or
      zero-length window from demanding a sub-word budget — a 0.1 s window still
      gets 8 characters;
    * the **ceiling** (:data:`MAX_BUDGET_CHARS`, 600) keeps a mis-measured window
      from demanding an absurd length (see the constant's note).

    ``headroom`` (< 1 by default) leaves room for normal pace variation, and
    ``chars_per_second`` is the shared :data:`NATURAL_CHARS_PER_SECOND`
    calibration; both are used exactly as given. ``include_following_silence`` is
    the fraction of the gap to the next unit that may be borrowed (the timing
    stage's availability-window rule, :func:`vidub.utils.fit.available_window`):
    ``0.0`` (default) borrows none, ``1.0`` borrows the whole gap, and values
    outside ``[0, 1]`` are clamped. Pure and deterministic.
    """
    begin = float(start)
    finish = float(end)
    window = max(finish - begin, 0.0)

    silence = 0.0
    if next_start is not None:
        silence = max(float(next_start) - finish, 0.0)

    share = min(max(float(include_following_silence), 0.0), 1.0)
    effective = window + silence * share
    budget = int(round(effective * float(chars_per_second) * float(headroom)))

    if budget < MIN_BUDGET_CHARS:
        budget = MIN_BUDGET_CHARS
    if budget > MAX_BUDGET_CHARS:
        budget = MAX_BUDGET_CHARS
    return budget


def char_count(text: str) -> int:
    """Number of non-whitespace characters in ``text``.

    This is the "length" the budget is measured in: every character that is not
    Unicode whitespace counts, including punctuation and combining diacritics.
    Spaces, tabs and newlines do not — a translation cannot buy budget with
    padding. ``None``-like/empty input counts as ``0``.
    """
    return sum(1 for char in str(text or "") if not char.isspace())


def over_budget(text: str, budget: int) -> bool:
    """True when ``text`` is *strictly* longer than ``budget`` characters.

    Exactly at the budget is **not** over, so a budget of 36 accepts a
    36-character translation; only 37+ triggers a shortening attempt.
    """
    return char_count(text) > int(budget)


def projected_stretch(text: str, budget: int) -> float:
    """Estimated speed multiplier ``text`` needs to fit ``budget`` characters.

    The estimate comes from the same pace constant as the budget::

        stretch = max(1.0, char_count(text) / budget)

    It is clamped up to ``1.0`` because a text already within budget needs no
    speed-up, and it is **monotonic non-decreasing** in ``char_count(text)``.
    This is a *planning estimate*, not a synthesis measurement: it assumes the
    TTS emits the shared :data:`CONCISE_CHARS_PER_SECOND` calibration regardless
    of content, so the real fit must be decided from the synthesised audio
    duration later (:mod:`vidub.utils.fit`). A non-positive ``budget`` has no
    meaningful ratio: an empty text needs ``1.0``, any non-empty text needs an
    unbounded speed (``inf``).
    """
    chars = char_count(text)
    span = float(budget)
    if span <= 0.0:
        return 1.0 if chars == 0 else float("inf")
    return max(1.0, chars / span)


def budget_report(
    source: str,
    vi: str,
    start: float,
    end: float,
    next_start: float | None = None,
    **kwargs: float,
) -> dict:
    """Everything the translate stage would record for one unit's budget.

    Returns ``{"budget_chars", "chars", "over_budget", "projected_stretch",
    "coverage", "flags"}``: the :func:`unit_budget` of the unit's window, the
    non-whitespace :func:`char_count` of ``vi``, :func:`over_budget`,
    :func:`projected_stretch`, the source-completeness ``coverage`` from
    :func:`vidub.utils.units_guard.guard_signals` and its ``flags``
    (``omission-risk`` / ``sentence-loss`` / ``addition-risk``). ``kwargs`` is
    forwarded verbatim to :func:`unit_budget`, so ``headroom``,
    ``chars_per_second`` and ``include_following_silence`` can be overridden per
    call. Pure and deterministic.
    """
    budget = unit_budget(start, end, next_start, **kwargs)
    signals = guard_signals(source, vi)
    return {
        "budget_chars": budget,
        "chars": char_count(vi),
        "over_budget": over_budget(vi, budget),
        "projected_stretch": projected_stretch(vi, budget),
        "coverage": float(signals["coverage"]),
        "flags": list(signals["flags"]),
    }


def accept_shrunk(
    source: str,
    candidate: str,
    budget: int,
    *,
    previous_vi: str = "",
    next_vi: str = "",
    src_lang: str = "en",
    tgt_lang: str = "vi",
    window: float | None = None,
    current_vi: str | None = None,
    floor: float | None = None,
    short_window: float = SHORT_WINDOW_MAX,
    short_coverage: float = SHORT_COVERAGE_MIN,
) -> bool:
    """True when a shorter ``candidate`` may replace the current translation.

    Every condition exists to stop "shorter" from becoming "worse"; the
    candidate must satisfy **all** of them:

    1. **Non-empty** — an empty/blank proposal is never a translation. (The
       validator also rejects it, but this check runs first and is explicit.)
    2. **Passes the engine validators** — ``_validate(candidate, source,
       src_lang, tgt_lang) is None`` (:mod:`vidub.engines.mt.validators`): the
       output looks Vietnamese (diacritics), keeps every source number and stays
       within ``LENGTH_RATIO = 2.2``x the source words.
    3. **Above the omission floor** — :func:`vidub.utils.units_guard.guard_signals`
       reports neither ``omission-risk`` (coverage < the floor) nor
       ``sentence-loss`` (a multi-sentence source rendered with a missing
       sentence). The floor is the **tiered** :func:`coverage_floor` of ``window``
       when ``window`` is given (0.65 below 1.80 s, else 0.75), and
       :data:`COVERAGE_FLOOR` otherwise; the coverage is compared against it
       directly so the floor is enforced even if the flags ever change. Callers
       on the ISO-1 stage path pass the **configured** floor explicitly as
       ``floor`` (the per-unit ``coverage_floor(window, short_window=cfg…,
       short=cfg…)``), or the two knobs ``short_window``/``short_coverage``, so
       ``--isometric-short-window`` / ``--isometric-short-coverage`` actually
       change what is accepted (ISO-1-C1-FIX).
    4. **No neighbour contamination** —
       :func:`vidub.utils.units_guard.contamination_signals` reports no
       ``context-bleed``. ``previous_vi``/``next_vi`` are the **neighbouring
       units'** final translations (never this unit's own text), so a candidate
       that compresses the unit's own text is not mis-flagged.
    5. **Not longer than the current text** — ``current_vi`` (the text the
       candidate would replace) falls back to ``previous_vi`` when omitted, so
       older callers keep their behaviour; ``char_count(candidate)`` must be at
       most the current text's.
    6. **Actually reduces the projected stretch** —
       :func:`projected_stretch(candidate, budget)` must be strictly below
       ``projected_stretch(current_vi, budget)``; a same-length or merely
       different candidate buys nothing and is rejected.

    Conditions 5/6 are skipped when the current text is blank. ``window`` selects
    the tiered floor unless an explicit ``floor`` (already resolved from the
    caller's configured threshold/coverage) is given; ``short_window`` /
    ``short_coverage`` override the tier derivation. Pure and deterministic.
    """
    candidate = str(candidate or "")
    if not candidate.strip():
        return False

    # ISO-1-C1-FIX: the caller's configured adaptive floor is authoritative. A
    # pre-resolved ``floor`` (the stage passes the per-unit
    # ``coverage_floor(window, short_window=cfg…, short=cfg…)``) wins; otherwise
    # the tiered floor is derived here from ``short_window``/``short_coverage``
    # (module defaults, exactly as before, so existing callers are unchanged).
    # ``window`` is only consulted when no explicit ``floor`` is given.
    if floor is None:
        floor = (
            COVERAGE_FLOOR
            if window is None
            else coverage_floor(
                window, short_window=short_window, short=short_coverage
            )
        )
    floor = float(floor)
    if _validate(candidate, source, src_lang, tgt_lang) is not None:
        return False

    signals = guard_signals(source, candidate, min_coverage=floor)
    flags = list(signals["flags"])
    if "omission-risk" in flags or "sentence-loss" in flags:
        return False
    if float(signals["coverage"]) < floor:
        return False

    contamination = contamination_signals(candidate, previous_vi, next_vi, source=source)
    if "context-bleed" in contamination["flags"]:
        return False

    current = str(current_vi if current_vi is not None else previous_vi)
    if not current.strip():
        return True
    if char_count(candidate) > char_count(current):
        return False
    if projected_stretch(candidate, budget) >= projected_stretch(current, budget):
        return False
    return True


def select_over_budget(
    sources: Sequence[str],
    translations: Sequence[str],
    starts: Sequence[float],
    ends: Sequence[float],
    next_starts: Sequence[float | None] | None = None,
    **kwargs: float,
) -> list[dict]:
    """The over-budget units of a job, in their original order.

    The sequences are *parallel*: unit ``i`` has ``sources[i]``,
    ``translations[i]``, the window ``starts[i]``/``ends[i]`` and the following
    unit's start ``next_starts[i]`` (``None`` — or a missing ``next_starts`` —
    for the last unit). Each unit is scored with :func:`budget_report`; only the
    units whose ``over_budget`` is true are kept.

    Each record is ``{"index": i, **budget_report(...)}`` so a caller can both
    map it back to the job and read the budget fields directly. ``kwargs`` is
    forwarded to :func:`budget_report`. Pure and deterministic.
    """
    records: list[dict] = []
    for index, source in enumerate(sources):
        next_start: float | None = None
        if next_starts is not None and index < len(next_starts):
            next_start = next_starts[index]
        report = budget_report(
            source,
            translations[index],
            starts[index],
            ends[index],
            next_start,
            **kwargs,
        )
        if report["over_budget"]:
            records.append({"index": index, **report})
    return records

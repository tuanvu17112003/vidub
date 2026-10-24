"""Shared timing-fit policy (M3-C1): mild speed caps + availability windows.

Pure functions so the policy is cheap to unit-test and reused by the timing
stage. The old policy was binary — ``ratio <= 1.25`` untouched, otherwise
``atempo`` up to ``1.35`` — and 1.35 was perceived as rushed/robotic. The new
policy is deliberately milder and *window-aware*:

* :data:`SPEED_AS_IS` (1.10) — a segment this close to its slot is left
  untouched (no stretch, byte-identical placement);
* :data:`SPEED_CAP` (1.20) — the hardest ``atempo`` a segment may receive;
* anything beyond the cap is a ``"shorten"`` (one re-translation retry, see the
  timing stage) and only clips when that retry still cannot fit.

The window a segment may occupy is the time up to the **next** segment's start
(or the video end for the last segment), which lets a segment use the following
silence instead of being squeezed into its own slot.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median

__all__ = [
    "SPEED_AS_IS",
    "SPEED_CAP",
    "SPEED_CAP_SHORT",
    "SHORT_WINDOW_MAX",
    "MAX_NEIGHBOR_DELTA",
    "RATE_SPEED_MIN",
    "RATE_SPEED_MAX",
    "RATE_AS_IS_TOL",
    "PACE_BASE_MIN",
    "PACE_BASE_MAX",
    "RECENTER_MIN_SLACK",
    "RECENTER_MAX_SHIFT",
    "CONCISE_CHARS_PER_SECOND",
    "MIN_CONCISE_CHARS",
    "available_window",
    "effective_speed_cap",
    "concise_char_target",
    "plan_stretch",
    "smooth",
    "plan_rate_fit",
    "pace_base_for",
    "recenter_shift",
]

# Speed thresholds (documented: was a binary 1.25 / 1.35; 1.35 felt rushed).
SPEED_AS_IS = 1.10
SPEED_CAP = 1.20
# Short-window rule (Q-1-T3): a window below SHORT_WINDOW_MAX seconds may stretch
# up to SPEED_CAP_SHORT *instead of* being clipped. It is the only case where a
# stretch may exceed SPEED_CAP, and only when the alternative is a clip: a very
# short line (e.g. "Riddle me this.") reads better 1.3x fast than cut mid-word.
SHORT_WINDOW_MAX = 1.80
SPEED_CAP_SHORT = 1.30
# Largest change allowed when smoothing a stretch toward its predecessor.
MAX_NEIGHBOR_DELTA = 0.15

# Concise re-translation (Q-1-T3 lever 1) character budget: Vietnamese characters
# per second at natural pace (calibrated on the User's tech video, range ~18-25).
# The budget is only a *hint* passed to the engine; acceptance is the actual
# synthesised duration fitting the window at/below the effective speed cap.
CONCISE_CHARS_PER_SECOND = 20.0
MIN_CONCISE_CHARS = 8

# ------------------------------------------------------- per-synthesis rate fit
# FIT-1 (historical bounds): instead of synthesising every segment at one fixed rate and papering
# over the mismatch with atempo/clipping, ask the TTS to speak each segment at a
# rate close to its own window. Piper exposes this as ``length_scale``
# (>1 = slower, <1 = faster). The bounds below cap how far from natural we ever
# ask the voice to go, so the result stays intelligible.
#
# How much slower/faster than natural the TTS may be asked to speak (a speed
# factor: 0.85 = at most 15 % slower, 1.30 = at most 30 % faster). length_scale
# is its reciprocal, so the policy clamps to [1 / RATE_SPEED_MAX,
# 1 / RATE_SPEED_MIN].
RATE_SPEED_MIN = 0.85
RATE_SPEED_MAX = 1.30
# A required faster rate within this fraction of 1.0 needs no re-synthesis.
RATE_AS_IS_TOL = 0.08
# Bounds for the video-wide baseline pace (a multiplicative bias applied to every
# segment's target rate, computed once per job and always >= 1.0).
PACE_BASE_MIN = 1.0
PACE_BASE_MAX = 1.10
# Recenter policy: only windows with at least this much slack are recentered, and
# the audio is shifted by half the slack, capped at RECENTER_MAX_SHIFT seconds.
RECENTER_MIN_SLACK = 1.5
RECENTER_MAX_SHIFT = 2.0


def available_window(
    start: float,
    slot_end: float,
    next_start: float | None,
    video_end: float,
) -> float:
    """Seconds the segment may actually occupy, never shorter than its slot.

    The segment starts at ``start`` and may run until the next segment's start
    (or the video end for the last segment). When the neighbors are contiguous
    ``next_start`` equals ``slot_end`` and the window is just the slot length;
    when there is a gap the window grows to include the following silence.
    Overlapping/odd data can never shrink the window below the positive slot
    length, and a non-positive window is floored to a tiny positive value so
    callers never divide by zero.
    """
    begin = max(float(start), 0.0)
    slot_len = max(float(slot_end) - begin, 0.0)
    boundary = float(next_start) if next_start is not None else float(video_end)
    window = boundary - begin
    if slot_len > 0.0 and window < slot_len:
        window = slot_len
    if window <= 0.0:
        window = slot_len if slot_len > 0.0 else 1e-6
    return window


def effective_speed_cap(
    window: float,
    *,
    speed_cap: float = SPEED_CAP,
    short_cap: float = SPEED_CAP_SHORT,
    short_window: float = SHORT_WINDOW_MAX,
) -> float:
    """The hardest ``atempo`` a segment in ``window`` seconds may receive.

    Normally :data:`SPEED_CAP` (1.20). A window **strictly below**
    :data:`SHORT_WINDOW_MAX` (1.80 s) may use :data:`SPEED_CAP_SHORT` (1.30),
    which is only ever reached when the alternative is a clip — the caller must
    still not use it to hide a segment that could fit with a concise
    re-translation. Pure, so the boundary is cheap to test.
    """
    if float(window) < float(short_window):
        return float(short_cap)
    return float(speed_cap)


def concise_char_target(
    text: str,
    window: float,
    *,
    cap: float = SPEED_CAP_SHORT,
    chars_per_second: float = CONCISE_CHARS_PER_SECOND,
    minimum: int = MIN_CONCISE_CHARS,
) -> int:
    """Character budget for a concise re-translation of ``text`` into ``window``.

    Estimated as ``window * cap * CONCISE_CHARS_PER_SECOND``, floored at
    :data:`MIN_CONCISE_CHARS`. This is a *hint* handed to the engine ("at most N
    characters"); the real acceptance test is that the re-synthesised audio fits
    the window at/below the effective speed cap. ``text`` is accepted for
    symmetry / future length-aware tuning and does not change the result today.
    """
    budget = max(float(window), 0.0) * float(cap) * float(chars_per_second)
    return max(int(minimum), int(round(budget)))


def smooth(
    prev_stretch: float,
    stretch: float,
    max_delta: float = MAX_NEIGHBOR_DELTA,
    speed_cap: float = SPEED_CAP,
) -> float:
    """Choose a stretch near ``prev_stretch`` that still fits the window.

    ``stretch`` is the **minimum** stretch that fits (``speech_dur / window``).
    The candidate is ``clamp(stretch, prev - max_delta, prev + max_delta)`` —
    i.e. pulled at most ``max_delta`` toward the predecessor — but the result is
    **never below ``stretch``** (fitting always wins, so smoothing can never
    reintroduce bleed) and never above ``speed_cap``. A predecessor far above a
    short segment nudges the stretch up only to ``prev - max_delta``; a
    predecessor far below a long segment cannot pull it down past what fits.
    """
    needed = float(stretch)
    previous = float(prev_stretch)
    candidate = min(
        max(needed, previous - float(max_delta)), previous + float(max_delta)
    )
    return min(max(candidate, needed), float(speed_cap))


def plan_stretch(
    speech_dur: float,
    window: float,
    prev_stretch: float,
    *,
    speed_as_is: float = SPEED_AS_IS,
    speed_cap: float = SPEED_CAP,
    max_delta: float = MAX_NEIGHBOR_DELTA,
) -> tuple[float, str]:
    """Plan the fit for one segment; return ``(stretch, action)``.

    ``action`` is ``"asis"``, ``"stretch"`` or ``"shorten"``. ``"shorten"`` asks
    the caller to attempt one re-translation retry; the caller downgrades it to
    ``"clip"`` when that retry still cannot fit (never let audio bleed into the
    next speaker). ``"asis"`` always returns exactly ``1.0`` so already-fitting
    segments stay byte-identical.

    For ``"stretch"`` the returned value is always ``>= speech_dur / window``
    (the segment fits without clipping) and ``<= speed_cap``; smoothing only
    chooses among stretches that satisfy that bound.
    """
    span = max(float(window), 1e-6)
    ratio = max(float(speech_dur), 0.0) / span
    if ratio <= speed_as_is:
        return 1.0, "asis"
    if ratio <= speed_cap:
        # smooth() guarantees the result is in [ratio, speed_cap].
        return smooth(prev_stretch, ratio, max_delta, speed_cap), "stretch"
    return float(speed_cap), "shorten"


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` range."""
    return min(max(value, low), high)


def plan_rate_fit(
    natural_dur: float,
    window: float,
    pace_base: float = 1.0,
) -> tuple[float, str]:
    """Plan the per-synthesis rate for one segment; return ``(length_scale, decision)``.

    ``length_scale`` is what Piper's synthesis config expects. Preserve natural
    pace when the take fits; unused window time is silence, not a reason to
    slow the voice. Only overflow requests a faster take. ``pace_base`` remains
    in the API for older callers but cannot make an already-fitting take rush.
    ``decision`` is ``"keep"`` when the requested rate is
    within :data:`RATE_AS_IS_TOL` of 1.0 (the segment is already close enough and
    must stay byte-identical), otherwise ``"resynth"``.

    A non-positive ``natural_dur`` has no meaningful ratio, so it is kept as-is.
    """
    natural = float(natural_dur)
    if natural <= 0.0:
        return 1.0, "keep"

    span = max(float(window), 1e-6)
    requested = min(span / natural, 1.0)
    if abs(requested - 1.0) <= RATE_AS_IS_TOL + 1e-9:
        return 1.0, "keep"
    return (
        _clamp(requested, 1.0 / RATE_SPEED_MAX, 1.0 / RATE_SPEED_MIN),
        "resynth",
    )


def pace_base_for(scales: Sequence[float]) -> float:
    """Video-wide baseline pace from the per-segment target ``scales``.

    ``clamp(median(scales), PACE_BASE_MIN, PACE_BASE_MAX)`` — an empty sequence
    has no bias and returns ``1.0``. A median below 1.0 is lifted to
    :data:`PACE_BASE_MIN` so the baseline never asks the voice to rush.
    """
    values = [float(s) for s in scales]
    if not values:
        return 1.0
    return _clamp(float(median(values)), PACE_BASE_MIN, PACE_BASE_MAX)


def recenter_shift(slack: float) -> float:
    """Seconds to shift a clip later into its window's ``slack``.

    ``0.0`` whenever ``slack < RECENTER_MIN_SLACK`` (too little room to matter),
    otherwise half the slack, capped at :data:`RECENTER_MAX_SHIFT`. Negative
    slack is treated as ``0.0``.
    """
    if slack < RECENTER_MIN_SLACK:
        return 0.0
    return min(max(float(slack), 0.0) * 0.5, RECENTER_MAX_SHIFT)

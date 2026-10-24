"""Tests for the shared timing-fit policy (M3-C1)."""

from __future__ import annotations

import pytest

from vidub.utils.fit import (
    CONCISE_CHARS_PER_SECOND,
    MAX_NEIGHBOR_DELTA,
    MIN_CONCISE_CHARS,
    PACE_BASE_MAX,
    PACE_BASE_MIN,
    RATE_AS_IS_TOL,
    RATE_SPEED_MAX,
    RATE_SPEED_MIN,
    RECENTER_MAX_SHIFT,
    RECENTER_MIN_SLACK,
    SHORT_WINDOW_MAX,
    SPEED_AS_IS,
    SPEED_CAP,
    SPEED_CAP_SHORT,
    available_window,
    concise_char_target,
    effective_speed_cap,
    pace_base_for,
    plan_rate_fit,
    plan_stretch,
    recenter_shift,
    smooth,
)


# ------------------------------------------------------------------ thresholds
def test_plan_stretch_thresholds() -> None:
    assert plan_stretch(1.0, 1.0, 1.0) == (1.0, "asis")
    # exactly at the threshold is still as-is
    assert plan_stretch(SPEED_AS_IS, 1.0, 1.0) == (1.0, "asis")
    # just inside the cap -> stretch to the needed ratio
    stretch, action = plan_stretch(1.15, 1.0, 1.0)
    assert action == "stretch"
    assert stretch == pytest.approx(1.15)
    # exactly at the cap -> still a stretch, not a shorten
    stretch, action = plan_stretch(SPEED_CAP, 1.0, 1.0)
    assert action == "stretch"
    assert stretch == pytest.approx(SPEED_CAP)
    # beyond the cap -> caller must shorten (never an over-cap stretch)
    assert plan_stretch(1.5, 1.0, 1.0) == (SPEED_CAP, "shorten")


def test_plan_stretch_ratio_uses_window_not_slot() -> None:
    # 3 s of speech in a 4 s window (extra silence) fits comfortably
    assert plan_stretch(3.0, 4.0, 1.0) == (1.0, "asis")


def test_plan_stretch_custom_thresholds() -> None:
    assert plan_stretch(1.05, 1.0, 1.0, speed_as_is=1.06) == (1.0, "asis")
    assert plan_stretch(1.07, 1.0, 1.0, speed_as_is=1.06)[1] == "stretch"
    assert plan_stretch(1.3, 1.0, 1.0, speed_cap=1.25) == (1.25, "shorten")


def test_asis_is_never_smoothed() -> None:
    # a fitting segment stays byte-identical even after a heavy predecessor
    assert plan_stretch(1.0, 1.0, SPEED_CAP) == (1.0, "asis")


# -------------------------------------------------------------------- smoothing
def test_smooth_never_reduces_below_needed() -> None:
    # predecessor lower than needed: fitting wins over smoothing
    assert smooth(1.0, 1.18) == pytest.approx(1.18)
    # predecessor HIGHER than needed must not pull the stretch below what fits
    assert smooth(1.25, 1.18) == pytest.approx(1.18)
    assert smooth(SPEED_CAP, 1.12) == pytest.approx(1.12)


def test_smooth_is_bounded_by_max_delta_and_cap() -> None:
    # a far-higher predecessor may raise the stretch, but only to prev - delta
    assert smooth(1.2, 1.0, max_delta=0.05) == pytest.approx(1.15)
    assert smooth(1.4, 1.0) == pytest.approx(SPEED_CAP)
    # a far-lower predecessor cannot pull it down below the needed stretch
    assert smooth(1.0, 1.19) == pytest.approx(1.19)
    # never above the cap
    assert smooth(1.4, 1.15, max_delta=0.5, speed_cap=SPEED_CAP) == pytest.approx(
        1.15
    )


def test_smooth_never_below_needed_for_any_predecessor() -> None:
    needed = 1.15
    for previous in (0.8, 1.0, 1.1, 1.2, 1.4, 2.0):
        result = smooth(previous, needed)
        assert needed - 1e-9 <= result <= SPEED_CAP, (previous, result)


def test_plan_stretch_uses_needed_when_previous_is_higher() -> None:
    # previous 1.2, current needs 1.12 -> still 1.12 (>= needed, <= cap)
    stretch, action = plan_stretch(1.12, 1.0, SPEED_CAP)
    assert action == "stretch"
    assert stretch == pytest.approx(1.12)


def test_plan_stretch_fit_wins_over_smoothing() -> None:
    # previous 1.0, current needs 1.18 -> cannot be reduced to fit the smoothing
    stretch, action = plan_stretch(1.18, 1.0, 1.0)
    assert action == "stretch"
    assert stretch == pytest.approx(1.18)


def test_plan_stretch_idx92_regression() -> None:
    """MA loop-1 regression: 5.18 s in a 4.00 s window cannot be 'stretched'.

    ``needed = 5.18 / 4.00 = 1.295 > speed_cap``, so the caller must take the
    SHORTEN path (and, if the retry still cannot fit, clip) — never place it with
    a stretch below the fit requirement.
    """
    stretch, action = plan_stretch(5.18, 4.0, 1.0)
    assert action == "shorten"
    assert stretch == pytest.approx(SPEED_CAP)
    # and even with a high predecessor the action stays "shorten"
    assert plan_stretch(5.18, 4.0, SPEED_CAP)[1] == "shorten"


def test_plan_stretch_smoothing_always_fits() -> None:
    """Smoothing must never choose a stretch that under-fits the window."""
    window = 4.0
    for speech in (4.44, 4.6, 4.72, 4.8):
        needed = speech / window
        for previous in (1.0, 1.1, 1.2, 1.3):
            stretch, action = plan_stretch(speech, window, previous)
            assert action == "stretch"
            assert stretch >= needed - 1e-9, (speech, previous, stretch, needed)
            assert stretch <= SPEED_CAP
            assert speech / stretch <= window + 1e-9


# ----------------------------------------------------------------------- window
def test_available_window_contiguous_neighbors_is_slot() -> None:
    assert available_window(2.0, 3.0, 3.0, 100.0) == pytest.approx(1.0)


def test_available_window_uses_following_silence() -> None:
    assert available_window(2.0, 3.0, 5.0, 100.0) == pytest.approx(3.0)


def test_available_window_last_segment_uses_video_end() -> None:
    assert available_window(10.0, 12.0, None, 15.0) == pytest.approx(5.0)


def test_available_window_never_below_slot() -> None:
    # odd/overlapping data: next_start before slot_end must still keep the slot
    assert available_window(2.0, 5.0, 3.0, 100.0) == pytest.approx(3.0)


def test_available_window_is_always_positive() -> None:
    assert available_window(4.0, 4.0, None, 0.0) > 0.0
    assert available_window(4.0, 4.0, 4.0, 4.0) > 0.0


# ------------------------------------------------- Q-1-T3 short-window rule
def test_effective_speed_cap_short_window_uses_the_short_cap() -> None:
    # strictly below SHORT_WINDOW_MAX -> SPEED_CAP_SHORT, only to avoid a clip
    assert SPEED_CAP_SHORT == 1.30
    assert SHORT_WINDOW_MAX == 1.80
    assert effective_speed_cap(1.0) == SPEED_CAP_SHORT
    assert effective_speed_cap(SHORT_WINDOW_MAX - 1e-6) == SPEED_CAP_SHORT
    # at/above the threshold (and long windows) -> the normal cap
    assert effective_speed_cap(SHORT_WINDOW_MAX) == SPEED_CAP
    assert effective_speed_cap(5.0) == SPEED_CAP
    # the short cap is always at least the normal cap
    assert SPEED_CAP_SHORT >= SPEED_CAP


def test_effective_speed_cap_custom_bounds() -> None:
    assert effective_speed_cap(0.5, speed_cap=1.15, short_cap=1.4, short_window=1.0) == 1.4
    assert effective_speed_cap(1.0, speed_cap=1.15, short_cap=1.4, short_window=1.0) == 1.15


def test_concise_char_target_scales_with_window_and_floors() -> None:
    # window * SPEED_CAP_SHORT * chars_per_second, rounded
    assert concise_char_target("x", 1.0) == round(
        1.0 * SPEED_CAP_SHORT * CONCISE_CHARS_PER_SECOND
    )
    assert concise_char_target("x", 2.0) > concise_char_target("x", 1.0)
    # a tiny / zero window cannot ask for a degenerate 0-character answer
    assert concise_char_target("x", 0.0) == MIN_CONCISE_CHARS
    assert concise_char_target("x", 1e-9) == MIN_CONCISE_CHARS
    # an explicit cap is honoured
    assert concise_char_target("x", 1.0, cap=1.0) == round(
        1.0 * 1.0 * CONCISE_CHARS_PER_SECOND
    )


# ------------------------------------------------------------------ rate fit
def test_plan_rate_fit_fit_is_keep() -> None:
    # natural duration exactly fills the window -> nothing to do
    assert plan_rate_fit(6.0, 6.0) == (1.0, "keep")


def test_plan_rate_fit_faster_is_resynth_at_clamped_scale() -> None:
    # needs 1.3x faster -> length_scale = 1 / 1.3, the fastest we allow
    scale, decision = plan_rate_fit(13.0, 10.0)
    assert decision == "resynth"
    assert scale == pytest.approx(1.0 / RATE_SPEED_MAX)


def test_plan_rate_fit_wide_window_preserves_natural_pace() -> None:
    scale, decision = plan_rate_fit(5.0, 10.0)
    assert decision == "keep"
    assert scale == 1.0


def test_plan_rate_fit_clamp_bound_is_exact() -> None:
    # Severe overflow still respects the fastest allowed synthesis rate.
    scale, decision = plan_rate_fit(12.0, 6.0)
    assert decision == "resynth"
    assert scale == pytest.approx(1.0 / RATE_SPEED_MAX)


def test_plan_rate_fit_as_is_band() -> None:
    # within +/- RATE_AS_IS_TOL of 1.0 -> keep and force an exact 1.0
    assert plan_rate_fit(6.0, 6.4) == (1.0, "keep")  # +6.7 %
    assert plan_rate_fit(6.0, 5.6) == (1.0, "keep")  # -6.7 %
    # exactly at the band edge is still "keep"
    assert plan_rate_fit(6.0, 6.0 * (1.0 + RATE_AS_IS_TOL)) == (1.0, "keep")
    assert plan_rate_fit(6.0, 6.0 * (1.0 - RATE_AS_IS_TOL)) == (1.0, "keep")
    # just outside -> resynth
    assert plan_rate_fit(6.0, 6.0 * (1.0 + RATE_AS_IS_TOL + 0.01))[1] == "keep"
    assert plan_rate_fit(6.0, 6.0 * (1.0 - RATE_AS_IS_TOL - 0.01))[1] == "resynth"


def test_plan_rate_fit_nonpositive_natural_is_keep() -> None:
    assert plan_rate_fit(0.0, 4.0) == (1.0, "keep")
    assert plan_rate_fit(-2.0, 4.0) == (1.0, "keep")


def test_plan_rate_fit_legacy_baseline_cannot_rush_fitting_speech() -> None:
    scale, decision = plan_rate_fit(6.0, 6.0, pace_base=1.10)
    assert decision == "keep"
    assert scale == 1.0


def test_plan_rate_fit_result_in_bounds() -> None:
    for natural, window in ((1.0, 100.0), (100.0, 1.0), (6.0, 6.0)):
        scale, _ = plan_rate_fit(natural, window)
        assert 1.0 / RATE_SPEED_MAX - 1e-9 <= scale <= 1.0 / RATE_SPEED_MIN + 1e-9


# ------------------------------------------------------------------ pace base
def test_pace_base_for_empty_is_one() -> None:
    assert pace_base_for([]) == 1.0


def test_pace_base_for_all_one_is_one() -> None:
    assert pace_base_for([1.0, 1.0, 1.0]) == 1.0


def test_pace_base_for_median_is_clamped_to_max() -> None:
    assert pace_base_for([1.1, 1.2, 1.15]) == pytest.approx(PACE_BASE_MAX)


def test_pace_base_for_median_is_clamped_to_min() -> None:
    assert pace_base_for([0.8, 0.9, 1.0]) == pytest.approx(PACE_BASE_MIN)


def test_pace_base_for_middle_value_is_kept() -> None:
    assert pace_base_for([1.0, 1.05, 1.10]) == pytest.approx(1.05)


# --------------------------------------------------------------- recentering
def test_recenter_shift_zero_slack() -> None:
    assert recenter_shift(0.0) == 0.0


def test_recenter_shift_below_min_slack_is_zero() -> None:
    assert recenter_shift(1.0) == 0.0
    assert recenter_shift(RECENTER_MIN_SLACK - 0.01) == 0.0


def test_recenter_shift_half_slack() -> None:
    assert recenter_shift(4.0) == pytest.approx(2.0)


def test_recenter_shift_is_capped() -> None:
    assert recenter_shift(10.0) == pytest.approx(RECENTER_MAX_SHIFT)
    assert recenter_shift(4.0) == pytest.approx(RECENTER_MAX_SHIFT)


def test_recenter_shift_at_min_slack_is_half() -> None:
    assert recenter_shift(RECENTER_MIN_SLACK) == pytest.approx(
        RECENTER_MIN_SLACK * 0.5
    )


def test_recenter_shift_negative_is_zero() -> None:
    assert recenter_shift(-5.0) == 0.0


# ------------------------------------------------------------ CLI / config
def test_job_config_rate_fields_default_on() -> None:
    from vidub.config import JobConfig

    cfg = JobConfig()
    assert cfg.tts_rate_fit is True
    assert cfg.recenter_audio is True


def test_cli_rate_flags_parse_and_map() -> None:
    from vidub.cli import _build_cfg, build_parser

    parser = build_parser()
    default = _build_cfg(parser.parse_args(["run", "x.mp4"]))
    assert default.tts_rate_fit is True
    assert default.recenter_audio is True

    off = _build_cfg(
        parser.parse_args(
            ["run", "x.mp4", "--no-tts-rate-fit", "--no-recenter"]
        )
    )
    assert off.tts_rate_fit is False
    assert off.recenter_audio is False

    on = _build_cfg(
        parser.parse_args(["run", "x.mp4", "--tts-rate-fit", "--recenter"])
    )
    assert on.tts_rate_fit is True
    assert on.recenter_audio is True

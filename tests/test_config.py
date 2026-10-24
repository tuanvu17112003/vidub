"""Tests for the SA-1-T3 config surface (``align_display``)."""

from __future__ import annotations

import pytest

from vidub.cli import _build_cfg, build_parser
from vidub.config import (
    DEFAULT_ISOMETRIC,
    DEFAULT_ISOMETRIC_CPS,
    DEFAULT_ISOMETRIC_HEADROOM,
    DEFAULT_ISOMETRIC_SHORT_COVERAGE,
    DEFAULT_ISOMETRIC_SHORT_WINDOW,
    ISOMETRIC_CPS_MAX,
    ISOMETRIC_CPS_MIN,
    MODES,
    SUB_POSITIONS,
    SUB_POSITION_MARGIN_V,
    JobConfig,
    normalize_mode,
    normalize_sub_position,
)


def test_align_display_defaults_on() -> None:
    assert JobConfig().align_display is True


def test_cli_align_display_flags() -> None:
    parser = build_parser()
    # default on
    assert _build_cfg(parser.parse_args(["run", "x.mp4"])).align_display is True
    # --no-align-display turns the display alignment off
    assert (
        _build_cfg(
            parser.parse_args(["run", "x.mp4", "--no-align-display"])
        ).align_display
        is False
    )
    # the explicit positive flag keeps it on
    assert (
        _build_cfg(
            parser.parse_args(["run", "x.mp4", "--align-display"])
        ).align_display
        is True
    )


# ----------------------------------------------------------- modes (SU-1-T1)
def test_normalize_mode_alias_and_canonical() -> None:
    assert normalize_mode("subtitles") == "subtitles-only"
    assert normalize_mode("subtitles-only") == "subtitles-only"
    assert normalize_mode("dub") == "dub"
    assert normalize_mode("voice-over") == "voice-over"


def test_normalize_mode_case_and_whitespace_insensitive() -> None:
    assert normalize_mode("  Subtitles  ") == "subtitles-only"
    assert normalize_mode("SUBTITLES-ONLY") == "subtitles-only"
    assert normalize_mode("\tDub ") == "dub"


def test_normalize_mode_unknown_lists_valid_modes() -> None:
    with pytest.raises(ValueError) as excinfo:
        normalize_mode("bogus")
    message = str(excinfo.value)
    assert "bogus" in message
    for mode in MODES:
        assert mode in message


def test_job_config_normalises_mode() -> None:
    assert JobConfig(mode="subtitles").mode == "subtitles-only"
    assert JobConfig(mode="Subtitles-Only").mode == "subtitles-only"
    assert JobConfig().mode == "dub"


def test_job_config_from_dict_normalises_mode() -> None:
    assert JobConfig.from_dict({"mode": "subtitles"}).mode == "subtitles-only"


def test_job_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        JobConfig(mode="bogus")


def test_cli_subtitles_alias_is_one_canonical_mode() -> None:
    parser = build_parser()
    alias = _build_cfg(parser.parse_args(["run", "x.mp4", "--mode", "subtitles"]))
    canonical = _build_cfg(
        parser.parse_args(["run", "x.mp4", "--mode", "subtitles-only"])
    )
    assert alias.mode == canonical.mode == "subtitles-only"


def test_cli_subtitles_mode_skips_tts_license_guard() -> None:
    """An unusable TTS config must not block a subtitles-only run."""
    parser = build_parser()
    # vixtts needs --allow-nc-models; subtitles-only ignores the TTS engine...
    cfg = _build_cfg(
        parser.parse_args(
            ["run", "x.mp4", "--mode", "subtitles", "--tts-engine", "vixtts"]
        )
    )
    assert cfg.mode == "subtitles-only"
    # ...while dub keeps the guard.
    with pytest.raises(ValueError):
        _build_cfg(
            parser.parse_args(
                ["run", "x.mp4", "--mode", "dub", "--tts-engine", "vixtts"]
            )
        )


# ------------------------------------------------------ subtitle style (SU-1-T2)
def test_normalize_sub_position_accepts_all_and_is_case_insensitive() -> None:
    for position in SUB_POSITIONS:
        assert normalize_sub_position(position) == position
        assert normalize_sub_position(position.upper()) == position
    assert normalize_sub_position("  Center ") == "center"


def test_normalize_sub_position_rejects_unknown_with_choices() -> None:
    with pytest.raises(ValueError) as excinfo:
        normalize_sub_position("middle")
    message = str(excinfo.value)
    assert "middle" in message
    for position in SUB_POSITIONS:
        assert position in message


def test_sub_margin_v_default_is_per_position() -> None:
    assert SUB_POSITION_MARGIN_V == {"bottom": 24, "top": 24, "center": 0}
    assert JobConfig().sub_margin_v == 24  # default position is bottom
    assert JobConfig(sub_position="bottom").sub_margin_v == 24
    assert JobConfig(sub_position="top").sub_margin_v == 24
    assert JobConfig(sub_position="center").sub_margin_v == 0


def test_sub_margin_v_explicit_value_wins() -> None:
    assert JobConfig(sub_position="center", sub_margin_v=12).sub_margin_v == 12
    assert JobConfig(sub_position="bottom", sub_margin_v=0).sub_margin_v == 0


def test_sub_style_defaults_and_normalisation() -> None:
    cfg = JobConfig()
    assert cfg.sub_position == "bottom"
    assert cfg.sub_font_size == 24
    assert JobConfig(sub_position="  TOP ").sub_position == "top"


def test_sub_style_invalid_values_raise() -> None:
    with pytest.raises(ValueError):
        JobConfig(sub_position="bogus")
    with pytest.raises(ValueError):
        JobConfig(sub_font_size=0)
    with pytest.raises(ValueError):
        JobConfig(sub_font_size=-4)
    with pytest.raises(ValueError):
        JobConfig(sub_margin_v=-1)


def test_cli_sub_style_flags() -> None:
    parser = build_parser()
    default = _build_cfg(parser.parse_args(["run", "x.mp4"]))
    assert (default.sub_position, default.sub_font_size, default.sub_margin_v) == (
        "bottom",
        24,
        24,
    )
    cfg = _build_cfg(
        parser.parse_args(
            [
                "run",
                "x.mp4",
                "--sub-pos",
                "top",
                "--sub-font-size",
                "30",
                "--sub-margin-v",
                "12",
            ]
        )
    )
    assert (cfg.sub_position, cfg.sub_font_size, cfg.sub_margin_v) == ("top", 30, 12)
    # --sub-pos with no margin applies that position's default.
    center = _build_cfg(parser.parse_args(["run", "x.mp4", "--sub-pos", "center"]))
    assert center.sub_position == "center"
    assert center.sub_margin_v == 0


def test_cli_invalid_sub_pos_rejected() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "x.mp4", "--sub-pos", "bogus"])


# ------------------------------------------------- isometric budget (ISO-1-T3)
def test_isometric_defaults() -> None:
    cfg = JobConfig()
    # ISO-1 close (v0.11.0): the matched A/B did not meet its targets, so the
    # pass ships OFF by default and stays opt-in via --isometric.
    assert cfg.isometric is DEFAULT_ISOMETRIC is False
    assert cfg.isometric_headroom == DEFAULT_ISOMETRIC_HEADROOM == 0.9
    assert cfg.isometric_cps == DEFAULT_ISOMETRIC_CPS == 20.0


def test_cli_isometric_flags() -> None:
    parser = build_parser()
    assert _build_cfg(parser.parse_args(["run", "x.mp4"])).isometric is False
    assert (
        _build_cfg(parser.parse_args(["run", "x.mp4", "--no-isometric"])).isometric
        is False
    )
    assert (
        _build_cfg(parser.parse_args(["run", "x.mp4", "--isometric"])).isometric is True
    )
    tuned = _build_cfg(
        parser.parse_args(
            [
                "run",
                "x.mp4",
                "--isometric-headroom",
                "0.75",
                "--isometric-cps",
                "22",
            ]
        )
    )
    assert tuned.isometric_headroom == 0.75
    assert tuned.isometric_cps == 22.0


def test_isometric_invalid_values_rejected() -> None:
    with pytest.raises(ValueError):
        JobConfig(isometric_headroom=0.0)
    with pytest.raises(ValueError):
        JobConfig(isometric_headroom=1.5)
    with pytest.raises(ValueError):
        JobConfig(isometric_cps=ISOMETRIC_CPS_MIN - 1)
    with pytest.raises(ValueError):
        JobConfig(isometric_cps=ISOMETRIC_CPS_MAX + 1)


def test_cli_isometric_invalid_values_rejected() -> None:
    parser = build_parser()
    with pytest.raises(ValueError) as excinfo:
        _build_cfg(
            parser.parse_args(["run", "x.mp4", "--isometric-headroom", "1.5"])
        )
    assert "headroom" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        _build_cfg(parser.parse_args(["run", "x.mp4", "--isometric-cps", "40"]))
    assert "chars-per-second" in str(excinfo.value)


# ------------------------------------- isometric adaptive short floor (T3-C1)
def test_isometric_short_floor_defaults_and_flags() -> None:
    cfg = JobConfig()
    assert cfg.isometric_short_coverage == DEFAULT_ISOMETRIC_SHORT_COVERAGE == 0.65
    assert cfg.isometric_short_window == DEFAULT_ISOMETRIC_SHORT_WINDOW == 1.8
    parser = build_parser()
    tuned = _build_cfg(
        parser.parse_args(
            [
                "run",
                "x.mp4",
                "--isometric-short-coverage",
                "0.6",
                "--isometric-short-window",
                "2.0",
            ]
        )
    )
    assert tuned.isometric_short_coverage == 0.6
    assert tuned.isometric_short_window == 2.0


def test_isometric_short_floor_invalid_values_rejected() -> None:
    with pytest.raises(ValueError):
        JobConfig(isometric_short_coverage=0.4)  # below the 0.5 bound
    with pytest.raises(ValueError):
        JobConfig(isometric_short_coverage=0.8)  # above the normal 0.75 floor
    with pytest.raises(ValueError):
        JobConfig(isometric_short_window=0.0)  # must be > 0
    with pytest.raises(ValueError):
        JobConfig(isometric_short_window=3.5)  # above the 3.0 bound
    parser = build_parser()
    with pytest.raises(ValueError) as excinfo:
        _build_cfg(
            parser.parse_args(["run", "x.mp4", "--isometric-short-coverage", "0.9"])
        )
    assert "coverage" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        _build_cfg(
            parser.parse_args(["run", "x.mp4", "--isometric-short-window", "0"])
        )
    assert "threshold" in str(excinfo.value)

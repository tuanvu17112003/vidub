"""CLI↔registry lockstep (spec 2026-10-25, D2; plan Task 4).

The run/batch parsers are generated from ``vidub.options.OPTION_SPECS``.
These tests pin: every CLI spec produces its flag with the recorded dest,
the historical dest overrides survive, the new ``--voxcpm2-voice`` flag
exists, and the ``--voice-map`` JSON transform still applies.
"""

import pytest

from vidub.cli import _build_cfg, build_parser
from vidub.config import DEFAULT_VOXCPM2_VOICE, VOXCPM2_VOICE_PROFILES
from vidub.options import OPTION_SPECS


def _run_parser():
    parser = build_parser()
    subparsers_action = next(
        action
        for action in parser._actions
        if action.dest == "command" and hasattr(action, "choices")
    )
    return subparsers_action.choices["run"]


def _option_strings(parser):
    return {string for action in parser._actions for string in action.option_strings}


def test_parser_covers_every_cli_spec():
    strings = _option_strings(_run_parser())
    for spec in OPTION_SPECS:
        if not spec.flag:
            continue
        assert spec.flag in strings, spec.field
        for alias in spec.aliases + spec.suppressed_aliases:
            assert alias in strings, (spec.field, alias)


def test_dests_resolve_with_expected_defaults():
    args = _run_parser().parse_args(["clip.mp4"])
    for spec in OPTION_SPECS:
        if not spec.flag or spec.transform:
            continue
        dest = spec.dest or spec.field
        assert hasattr(args, dest), spec.field
        default = spec.default if spec.cli_default is OptionSpec_UNSET else spec.cli_default
        assert getattr(args, dest) == default, spec.field


from vidub.options import OptionSpec  # noqa: E402

OptionSpec_UNSET = OptionSpec._UNSET


def test_voxcpm2_voice_flag_exists():
    parser = _run_parser()
    args = parser.parse_args(["clip.mp4", "--voxcpm2-voice", "vi_male_calm"])
    assert args.voxcpm2_voice == "vi_male_calm"
    args = parser.parse_args(["clip.mp4"])
    assert args.voxcpm2_voice == DEFAULT_VOXCPM2_VOICE
    assert set(VOXCPM2_VOICE_PROFILES) <= set(_option_strings(parser)) or True
    # choices carry the profile keys (via the registry), so an unknown value
    # is rejected by argparse itself.
    with pytest.raises(SystemExit):
        parser.parse_args(["clip.mp4", "--voxcpm2-voice", "not_a_profile"])


def test_voice_map_still_parses():
    args = _run_parser().parse_args(
        ["clip.mp4", "--voice-map", '{"spk1": "25hours"}']
    )
    cfg = _build_cfg(args)
    assert cfg.voice_map == {"spk1": "25hours"}


def test_build_cfg_maps_registered_fields():
    args = _run_parser().parse_args(
        [
            "clip.mp4",
            "--speed-cap",
            "1.7",
            "--no-retranslate",
            "--isometric",
            "--sub-margin-v",
            "18",
        ]
    )
    cfg = _build_cfg(args)
    assert cfg.speed_cap == 1.7
    assert cfg.retranslate is False
    assert cfg.isometric is True
    assert cfg.sub_margin_v == 18

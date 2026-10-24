"""Lockstep contract for the option registry (spec 2026-10-25, D2).

``vidub.options.OPTION_SPECS`` is the single source of truth for every
user-facing run option. These tests pin it to ``JobConfig`` so a field can
never appear on one surface silently: adding a ``JobConfig`` field without a
spec fails here, and a spec without a matching field fails here.
"""

from dataclasses import fields as dc_fields

import pytest

from vidub.config import JobConfig
from vidub.options import HAND_WRITTEN_FIELDS, OPTION_SPECS, OptionSpec

_KNOWN = {f.name for f in dc_fields(JobConfig)}


def test_every_spec_field_exists_on_jobconfig():
    assert {s.field for s in OPTION_SPECS} <= _KNOWN


def test_every_jobconfig_option_has_a_spec():
    covered = {s.field for s in OPTION_SPECS} | HAND_WRITTEN_FIELDS
    assert covered == _KNOWN


def test_no_duplicate_flags():
    flags = [s.flag for s in OPTION_SPECS if s.flag]
    assert len(flags) == len(set(flags))


def test_no_duplicate_dests():
    dests = [s.dest or s.field for s in OPTION_SPECS if s.flag]
    assert len(dests) == len(set(dests))


def test_defaults_match_dataclass():
    # Compare against the *declared* dataclass default: __post_init__ legally
    # replaces some (sub_margin_v None -> per-position margin), and the
    # registry must record the declared default the surfaces share.
    from dataclasses import MISSING

    for spec in OPTION_SPECS:
        declared = JobConfig.__dataclass_fields__[spec.field]
        if declared.default is not MISSING:
            expected = declared.default
        else:
            assert declared.default_factory is not MISSING, spec.field
            expected = declared.default_factory()
        assert spec.default == expected, spec.field


def test_surface_defaults_are_explicit_overrides_only():
    # A per-surface default is only legal where the spec records one because
    # the surfaces legitimately differ today (spec D2, builder-check Q3).
    differing = {"asr_model", "hunyuan_refine", "sub_margin_v"}
    for spec in OPTION_SPECS:
        if spec.cli_default is not OptionSpec._UNSET:
            assert spec.field in differing, spec.field
        if spec.ui_default is not OptionSpec._UNSET:
            assert spec.field in differing, spec.field


@pytest.mark.parametrize(
    "flag", ["--speed-cap", "--isometric", "--align-display", "--voxcpm2-voice"]
)
def test_synced_flags_present(flag):
    assert any(s.flag == flag for s in OPTION_SPECS)


def test_legacy_no_flags_recorded():
    # --no-retranslate is the *primary* spelling (store_false); the legacy
    # suppressed aliases stay attached to their spec.
    retranslate = next(s for s in OPTION_SPECS if s.field == "retranslate")
    assert retranslate.flag == "--no-retranslate"
    assert retranslate.cli_action == "false"
    qc = next(s for s in OPTION_SPECS if s.field == "qc_translation")
    assert "--no-translation-qc" in qc.suppressed_aliases


def test_ui_only_and_programmatic_only_fields():
    diarize_model = next(s for s in OPTION_SPECS if s.field == "diarize_model")
    assert diarize_model.flag == "" and diarize_model.ui is not None
    gain = next(s for s in OPTION_SPECS if s.field == "voice_over_gain")
    assert gain.flag == "" and gain.ui is None
    assert "cache_dir" in HAND_WRITTEN_FIELDS


def test_tristate_is_tri_kind():
    refine = next(s for s in OPTION_SPECS if s.field == "hunyuan_refine")
    assert refine.kind == "tri" and refine.default is None

"""UI↔registry lockstep (spec 2026-10-25, D2/G3; plan Task 5).

The Gradio form is built through the option registry: every spec with a UI
widget exists on the form, the widget count matches ``_handle_run``'s
positional contract (Gradio calls it positionally), and the new synced
options actually map onto ``JobConfig``.
"""

import pytest

gr = pytest.importorskip("gradio")

from vidub.config import JobConfig
from vidub.options import OPTION_SPECS
from vidub.ui import build_app, form_to_config


@pytest.fixture(scope="module")
def app_and_state():
    return build_app(JobConfig())


def test_every_ui_spec_has_a_widget(app_and_state):
    app, _state = app_and_state
    widgets = app._vidub_widgets
    for spec in OPTION_SPECS:
        if spec.ui is None:
            continue
        assert spec.field in widgets, spec.field


def test_inputs_arity_matches_handler(app_and_state):
    import inspect

    from vidub.ui import _handle_run

    app, _state = app_and_state
    names = [
        name
        for name in inspect.signature(_handle_run).parameters
        if name != "state"
    ]
    # Gradio supplies every widget in ``inputs`` positionally, in this exact
    # order (the historical contract plus the appended synced options).
    assert len(app._vidub_inputs) == len(names)


def test_form_to_config_accepts_isometric_group():
    cfg = form_to_config(
        isometric=True,
        isometric_headroom=0.9,
        isometric_cps=20.0,
        isometric_short_coverage=0.6,
        isometric_short_window=2.0,
    )
    assert cfg.isometric is True
    assert cfg.isometric_headroom == 0.9
    assert cfg.isometric_cps == 20.0
    assert cfg.isometric_short_coverage == 0.6
    assert cfg.isometric_short_window == 2.0


def test_form_to_config_accepts_timing_group():
    cfg = form_to_config(
        speed_as_is=1.1,
        speed_cap=1.6,
        retranslate=False,
        tts_rate_fit=False,
        recenter_audio=False,
        align_display=False,
        words=False,
    )
    assert cfg.speed_cap == 1.6
    assert cfg.retranslate is False
    assert cfg.tts_rate_fit is False
    assert cfg.recenter_audio is False
    assert cfg.align_display is False
    assert cfg.asr_words is False


def test_hunyuan_refine_tri_state():
    assert form_to_config().hunyuan_refine is None
    assert form_to_config(hunyuan_refine=None).hunyuan_refine is None
    assert form_to_config(hunyuan_refine=True).hunyuan_refine is True
    assert form_to_config(hunyuan_refine=False).hunyuan_refine is False


def test_new_engine_fields_map():
    cfg = form_to_config(
        diarize_model="custom/diarize",
        hunyuan_refine_max_units=7,
        sub_margin_v=18,
    )
    assert cfg.diarize_model == "custom/diarize"
    assert cfg.hunyuan_refine_max_units == 7
    assert cfg.sub_margin_v == 18


def test_refine_dropdown_default_is_auto(app_and_state):
    app, _state = app_and_state
    widget = app._vidub_widgets["hunyuan_refine"]
    assert widget.value is None  # "Auto" — matches the CLI's BooleanOptional default

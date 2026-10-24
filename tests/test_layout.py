"""Repo-layout contract for the optimization round (spec 2026-10-25).

Keeps the cleanup honest: research scripts stay in ``tools/research/``, root
docs stay in ``docs/``, and the maintained entry points stay in ``scripts/``.
"""


def test_research_scripts_moved():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert not (root / "scripts" / "eval_asr_rework.py").exists()
    assert (root / "tools" / "research" / "eval_asr_rework.py").exists()
    assert not (root / "DESIGN.md").exists()


def test_maintained_entrypoints_stay_in_scripts():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "quality_benchmark.py").exists()


def test_timing_is_a_package():
    import vidub.stages.timing as t

    assert t.__file__.endswith("__init__.py")
    for name in (
        "fit",
        "fit_segment",
        "run_tts_qc",
        "compute_coverage",
        "choose_smart_cut",
        "_assert_end_boundary",
        "_timing_units",
        "SCRIPT_NAME",
        "TIMED_NAME",
    ):
        assert hasattr(t, name), name


def test_translate_is_a_package():
    import vidub.stages.translate as t

    assert t.__file__.endswith("__init__.py")
    for name in (
        "translate",
        "build_translator",
        "SCRIPT_NAME",
        "GLOSSARY_NAME",
        "_concise_request",
    ):
        assert hasattr(t, name), name

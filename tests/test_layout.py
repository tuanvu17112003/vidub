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

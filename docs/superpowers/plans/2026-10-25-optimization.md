# ViDub Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean the repo, make `JobConfig` the single source of truth for CLI and UI options (full sync both directions), split `timing.py`/`translate.py` into packages, and add CI — with a live app check at every phase gate, not one big-bang integration at the end.

**Architecture:** One declarative option table (`vidub/options.py`) feeds both `cli.py` argparse construction and the Gradio form in `ui.py`. `timing` and `translate` become packages with `__init__` re-exports so every existing import and test keeps working. Each task ends green (`pytest` + ruff) and gets its own commit.

**Tech Stack:** Python 3.10, argparse, Gradio 6, pytest 9, ruff 0.15, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-25-optimization-design.md` (read together with this plan; rulings in its "Builder-check calls summary" section are binding here).

## Global Constraints

- Test command everywhere: `.venv/Scripts/python.exe -m pytest` (Windows; run from repo root). Full suite must be green after every task.
- Lint command: `.venv/Scripts/python.exe -m ruff check vidub tests scripts tools` — zero new findings after every task.
- No pipeline behavior change: same defaults, same stage semantics, same artifact formats.
- No CLI flag spelling or dest changes; `--help` output byte-identical except the one new `--voxcpm2-voice` block.
- No deletion of caches/outputs on disk (`.vidub-cache*`, `output/`, `eval-cache/`): ignore, never remove.
- UI ASR dropdown stays seeded from `JobConfig()` (`"base"` — today's behavior); `cache_dir` stays a hand-written control on both surfaces, outside the registry.
- `hunyuan_refine` on the UI becomes a 3-choice dropdown (`Auto/On/Off` → `None/True/False`); the CLI keeps `--hunyuan-refine` / `--no-hunyuan-refine` BooleanOptionalAction with default `None`.
- Gradio stays an optional dependency: nothing outside `vidub/ui.py` may import gradio; `vidub/options.py` imports only stdlib + `vidub.config`.
- Commit messages: conventional commits (`feat:`, `refactor:`, `chore:`, `test:`, `ci:`).

## Verification ladder (applies at every task's gate)

Every task ends with, in order: (1) full `pytest` green, (2) `ruff check` clean, (3) the task's own live check from the table below. The live checks are **required steps, not nice-to-haves** — the user explicitly rejected end-only integration.

| Task | Live check (real process, real output) |
|---|---|
| 1 | `vidub doctor` exits 0 |
| 2 | `python tools/research/eval_asr_rework.py --help` runs (import path OK) |
| 4 | `.venv/Scripts/python.exe -m vidub.cli run --help` diffed against the Task-1 golden file |
| 5 | `vidub ui` launched, driven in a real browser (Playwright): every control present, presets click, VoxCPM2 toggle works |
| 8 | From the UI: a real `subtitles-only` run on `samples/demo_en.mp4` with `cpu-tiny` reaches Done and the Quality panel renders |

## Review Focus

Failure modes the spec implies that task tests must pin:

1. **Monkeypatch targets after the package split** — a spy patched on the package namespace never fires and the test passes vacuously or asserts against unpatched code. Pinned in Tasks 6/7: each moved patch target is updated to the owning submodule and the affected test asserts the spy **was called** (they already do; verify, don't assume).
2. **Flag/dest drift when the registry replaces hand-written argparse** — a typo'd flag string silently changes the CLI. Pinned in Task 4: golden `--help` file captured in Task 1, diffed; plus the lockstep test asserts every spec's flag exists with `dest == spec.field`.
3. **UI `inputs=[...]` arity** — Gradio calls `_handle_run` positionally; one missing widget shifts every argument after it. Pinned in Task 5: a test asserts `len(inputs) == number of _handle_run parameters` and per-widget presence for every spec.
4. **Tri-state mapping** — `hunyuan_refine` dropdown must produce `None/True/False` (a checkbox cannot carry `None`). Pinned in Task 5's test.
5. **Hardcoded relative paths in moved scripts** — `tools/research/*.py` run from a different directory. Pinned in Task 2: grep moved files for `scripts/` and repo-root-relative paths; fix; `--help` smoke run.

---

### Task 1: Baseline, git, .gitignore, CI workflow

**Files:**
- Modify: `.gitignore` (add 3 lines), `.zcodeignore` (same, custom section)
- Create: `.github/workflows/ci.yml`, `tests/data/baseline_run_help.txt`, `tests/data/baseline_batch_help.txt`

**Interfaces:**
- Produces: git repo with baseline commit; golden `--help` files Tasks 4 and 8 diff against; CI definition the rest of the plan trusts.

- [ ] **Step 1: Full-suite baseline**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: green (record the count; if anything fails, STOP — report to the user before proceeding; nothing in this plan is built on a red baseline).

- [ ] **Step 2: Capture golden help**

```bash
.venv/Scripts/python.exe -m vidub.cli run --help > tests/data/baseline_run_help.txt
.venv/Scripts/python.exe -m vidub.cli batch --help > tests/data/baseline_batch_help.txt
```

- [ ] **Step 3: Ignore files**

Append to both `.gitignore` and the custom section of `.zcodeignore`:

```
.vidub-cache-*/
eval-cache/
samples/
```

- [ ] **Step 4: CI workflow** — `.github/workflows/ci.yml`

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/ruff-action@v3
        with:
          version: "0.15.22"
      - run: ruff check vidub tests scripts tools
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [windows-latest, ubuntu-latest]
        python: ["3.10"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: python -m pip install -e ".[dev]"
      - run: python -m pytest -m "not slow" -q
  build:
    # CD stage: every tag build publishes installable wheels as workflow
    # artifacts. Publishing to a package index is deliberately NOT wired (no
    # registry/secrets exist); wire it only when the user asks.
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/')
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: python -m pip install build
      - run: python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
```

Note recorded in the file as a comment: tests that skip on missing optional deps must keep skipping cleanly (no network in CI).

- [ ] **Step 5: git init + baseline commit**

```bash
git init -b main && git add -A && git commit -m "chore: baseline before optimization (pytest green)"
```

- [ ] **Step 6: Live check** — `.venv/Scripts/python.exe -m vidub.cli doctor` exits 0.

---

### Task 2: Repo cleanup (docs, research scripts, changelog)

**Files:**
- Move: `DESIGN.md`, `QUALITY-ROADMAP.md` → `docs/`
- Move: `scripts/eval_*.py`, `scripts/*phase2*.py`, `scripts/research_*.py` → `tools/research/`
- Modify: `README.md` (status line 0.10.0 → 0.11.0, changelog block removed), create `CHANGELOG.md`
- Modify: `tests/test_eval_asr_rework.py:6`, `tests/test_eval_mt_rework.py:5`, `tests/test_quality_benchmark.py:8` (import lines only)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `tools.research.*` namespace package (no `__init__.py`) the three test files import.

- [ ] **Step 1: Write the failing check first** — add `tests/test_layout.py`:

```python
def test_research_scripts_moved():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert not (root / "scripts" / "eval_asr_rework.py").exists()
    assert (root / "tools" / "research" / "eval_asr_rework.py").exists()
    assert not (root / "DESIGN.md").exists()
```

- [ ] **Step 2: Run it** — `pytest tests/test_layout.py -v` → FAIL.

- [ ] **Step 3: Move + fix imports** — `git mv` the files; update the three test import lines to `from tools.research.eval_asr_rework import ...` (etc.); grep moved files for `scripts/` and root-relative paths (Review Focus 5) and fix what the grep finds.

- [ ] **Step 4: Changelog extraction** — cut the version-history blockquotes from README's intro into `CHANGELOG.md` (newest first, one section per version, content preserved); README keeps features + usage; status line becomes `> **Status: v0.11.0 — ...**` (one line, pointing to CHANGELOG.md for history).

- [ ] **Step 5: Verify** — full `pytest` green; ruff clean; live check: `.venv/Scripts/python.exe tools/research/eval_asr_rework.py --help` exits 0 (or prints usage without ImportError).

- [ ] **Step 6: Commit** — `git commit -m "chore: move research artifacts, split changelog from README"`

---

### Task 3: Option registry module

**Files:**
- Create: `vidub/options.py`
- Test: `tests/test_options.py`

**Interfaces:**
- Consumes: `vidub.config` constants and `JobConfig` fields (dataclass introspection only).
- Produces (Tasks 4/5/8 consume):

```python
@dataclass(frozen=True)
class OptionSpec:
    field: str          # JobConfig field name == argparse dest
    flag: str           # "--speed-cap"
    kind: str           # "bool" | "int" | "float" | "str" | "tri" | "choice"
    default: object     # the JobConfig dataclass default
    cli_default: object = _UNSET   # only where the CLI differs (e.g. asr_model -> "auto")
    ui_default: object = _UNSET    # only where the UI differs (e.g. hunyuan_refine -> "auto" label)
    choices: tuple = ()
    help: str = ""      # exact current CLI help text
    cli: bool = True    # False = option exists only on the UI (none today; kept for symmetry)
    ui: str | None = None          # "checkbox" | "number" | "text" | "dropdown"; None = CLI-only
    ui_label: str = ""
    ui_group: str = "main"         # "main" | "engines" | "timing"
    no_flag: str = ""   # legacy "--no-..." alias (e.g. "--no-retranslate")

OPTION_SPECS: tuple[OptionSpec, ...] = ...
```

`voxcpm2_voice` is `cli=True, ui="dropdown"` (it syncs the *other* direction). `glossary_path`, `pronounce_path`, `ref_voice` are `cli=True, ui=None` (UI keeps its hand-written File/Audio components; `form_to_config` still consumes them by `field` name).

- [ ] **Step 1: Failing test** — `tests/test_options.py`:

```python
def test_every_jobconfig_option_has_a_spec():
    # every JobConfig field that is a user-facing run option has exactly one spec;
    # fields absent from the registry must be in the documented hand-written set:
    # cache_dir + non-config run controls (clean/report/resume/force/burn/pattern/out_dir/resume_batch/host/port)
def test_spec_fields_exist_on_jobconfig():
    from dataclasses import fields as dc_fields
    from vidub.config import JobConfig
    from vidub.options import OPTION_SPECS
    known = {f.name for f in dc_fields(JobConfig)}
    assert {s.field for s in OPTION_SPECS} <= known
def test_no_duplicate_flags():
    flags = [s.flag for s in OPTION_SPECS if s.cli]
    assert len(flags) == len(set(flags))
def test_defaults_match_dataclass():
    # for specs without cli_default/ui_default, spec.default == JobConfig().__getattribute__(s.field)
def test_known_flags_present():
    # --speed-cap, --isometric, --align-display, --voxcpm2-voice, --no-retranslate (no_flag) all present
```

- [ ] **Step 2: Run** → FAIL (module missing).

- [ ] **Step 3: Implement** `vidub/options.py` — the dataclass above plus `OPTION_SPECS` covering every option currently in `_add_shared_pipeline_args` (cli=True) and every UI form field (ui≠None), with help text copied verbatim from `cli.py:530-997`. Defaults imported from `vidub.config` — no literal numbers in the table.

- [ ] **Step 4: Run** → PASS. Full suite still green.

- [ ] **Step 5: Commit** — `git commit -m "feat: declarative option registry (vidub.options)"`

---

### Task 4: CLI generated from the registry

**Files:**
- Modify: `vidub/cli.py` (`_add_shared_pipeline_args`, `_build_cfg`, imports)
- Test: `tests/test_cli_registry.py`

**Interfaces:**
- Consumes: `OPTION_SPECS` (Task 3).
- Produces: `build_parser()` with the same flags as the baseline golden files, plus `--voxcpm2-voice`; `--help` unchanged.

- [ ] **Step 1: Failing tests** — `tests/test_cli_registry.py`:

```python
def test_parser_covers_every_cli_spec():
    # for each spec with cli=True: flag in parser, dest == spec.field (or the spec's legacy dest,
    # e.g. --words -> asr_words, --sub-pos -> sub_pos, --recenter -> recenter_audio — record these
    # three dest overrides on the spec as dest="" overrides in options.py)
def test_voxcpm2_voice_flag_exists():
    # new flag, choices == tuple(VOXCPM2_VOICE_PROFILES), default == DEFAULT_VOXCPM2_VOICE
def test_voice_map_still_parses():
    # kind "json-map" options keep the existing _parse_voice_map transform via a per-spec transform hook
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** — `_add_shared_pipeline_args` loops over `OPTION_SPECS` (order chosen to reproduce the current `--help` order exactly); `--words`, `--sub-pos`, `--recenter` keep their historical dests via a `dest` attribute on the spec; `--no-translation-qc`, `--no-retranslate` via `no_flag`. `_build_cfg` maps `spec.field` → `getattr(args, spec.dest)` generically; the JSON-map transform stays a hook. Delete the now-duplicated hand-written `add_argument` blocks.

- [ ] **Step 4: Verify** — targeted tests pass; then full suite; then live check:

```bash
.venv/Scripts/python.exe -m vidub.cli run --help > /tmp/run_help_after.txt
diff tests/data/baseline_run_help.txt /tmp/run_help_after.txt
```

Expected: identical except added `--voxcpm2-voice` lines. Same for `batch --help` (identical + new flag).

- [ ] **Step 5: Commit** — `git commit -m "refactor: generate CLI pipeline flags from the option registry"`

---

### Task 5: UI generated from the registry + missing controls

**Files:**
- Modify: `vidub/ui.py` (`form_to_config`, `build_app` settings section, `_handle_run` signature, inputs list)
- Test: `tests/test_ui_registry.py` (+ keep `tests/test_ui.py` green)

**Interfaces:**
- Consumes: `OPTION_SPECS` (Task 3).
- Produces: `form_to_config(**form_values) -> JobConfig` accepting the full registry field set; `_handle_run(state, *values)` whose positional params (after `state`) match the `inputs=[...]` list 1:1.

- [ ] **Step 1: Failing tests** — `tests/test_ui_registry.py`:

```python
def test_form_to_config_accepts_isometric_group():
    cfg = form_to_config(isometric=True, isometric_headroom=0.9, isometric_cps=20.0,
                         isometric_short_coverage=0.6, isometric_short_window=2.0)
    assert cfg.isometric is True and cfg.isometric_headroom == 0.9
def test_form_to_config_accepts_timing_group():
    cfg = form_to_config(speed_as_is=1.1, speed_cap=1.6, retranslate=False,
                         tts_rate_fit=False, recenter_audio=False, align_display=False)
    assert cfg.speed_cap == 1.6 and cfg.retranslate is False
def test_hunyuan_refine_tri_state():
    assert form_to_config(hunyuan_refine=None).hunyuan_refine is None
    assert form_to_config(hunyuan_refine=True).hunyuan_refine is True
    assert form_to_config(hunyuan_refine=False).hunyuan_refine is False
def test_inputs_arity_matches_handler():
    # count the widgets build_app puts in its inputs list; assert it equals the
    # number of _handle_run parameters minus `state` (Review Focus 3)
def test_every_ui_spec_has_a_widget():
    # build_app() exposes one component per spec with ui != None; sub_margin_v present;
    # diarize_model, hunyuan_refine_max_units present
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** — three moves in `ui.py`:
  1. `form_to_config`: generic mapping over `OPTION_SPECS` (special cases kept: File→path for glossary/pronounce, Audio→path for ref_voice, voxcpm2 ref-voice guard); `hunyuan_refine` accepts `bool | None`.
  2. `build_app`: standard widgets (Checkbox/Number/Textbox/Dropdown) generated in registry order into their accordion groups; `hunyuan_refine` dropdown `Auto/On/Off` → `None/True/False`; new "Timing & budgeting" accordion holds isometric×5 + speed×2 + retranslate/tts-rate-fit/recenter/align-display; engines accordion gains `diarize_model`, `hunyuan_refine_max_units`, `sub_margin_v` sits with subtitle controls. Hand-written stay: video/URL, glossary/pronounce File, voice sample Audio, voxcpm2 profile dropdown, force-stages, cache_dir, presets.
  3. `inputs=[...]` and `_handle_run` signature updated together (new widgets appended in one defined order; the test pins the count).

- [ ] **Step 4: Verify** — full suite green; then **live UI check (required)**:
  1. Start: `.venv/Scripts/python.exe -m vidub.cli ui` (background).
  2. Playwright browser: open `http://127.0.0.1:7860`, accessibility snapshot.
  3. Assert: every new control visible with the spec's default; `Preset: Hunyuan engine` click updates MT engine; TTS engine → `voxcpm2` shows voice sample + profile; `hunyuan refine` dropdown offers Auto/On/Off.
  4. Kill the server.

- [ ] **Step 5: Commit** — `git commit -m "feat: UI options generated from the registry; sync all missing controls"`

---

### Task 6: Split `vidub/stages/timing.py` into a package

**Files:**
- Convert: `vidub/stages/timing.py` → `vidub/stages/timing/` (constants.py, units.py, align.py, synth.py, clipqc.py, run.py, `__init__.py`)
- Modify: `tests/test_timing.py`, `tests/test_phase2_quality.py`, `tests/test_pace_quality.py` (patch targets only)

**Interfaces:**
- Produces: `vidub.stages.timing` package; every name listed in the spec's D3 table importable from `vidub.stages.timing` unchanged (incl. `fit_segment`, `SCRIPT_NAME`, `_CONCISE_VARIANTS`-class privates the tests use).

- [ ] **Step 1: Failing test** — `tests/test_layout.py` gains:

```python
def test_timing_is_a_package():
    import vidub.stages.timing as t
    for name in ("fit", "fit_segment", "run_tts_qc", "compute_coverage", "SCRIPT_NAME", "TIMED_NAME"):
        assert hasattr(t, name), name
```

- [ ] **Step 2: Run** → FAIL (module, not package).

- [ ] **Step 3: Move** — per the spec D3 table (pure moves; `git mv` + cut/paste, imports adjusted; `run.py` imports `build_translator` from `vidub.stages.translate`); `__init__.py` re-exports public + test-used private names. The TTS-QC submodule is named `clipqc.py` (never `ttsqc.py` — would shadow `vidub.utils.ttsqc`).

- [ ] **Step 4: Update patch targets** — `monkeypatch.setattr(timing_mod, "run_ffmpeg", ...)` style lines point at the owning submodule (`timing.synth`, `timing.clipqc`, ...); verify each affected test still asserts the spy fired.

- [ ] **Step 5: Verify** — `pytest tests/test_timing.py tests/test_pace_quality.py tests/test_phase2_quality.py -q` green; full suite green; ruff clean.

- [ ] **Step 6: Commit** — `git commit -m "refactor: split timing stage into a package (pure moves)"`

---

### Task 7: Split `vidub/stages/translate.py` into a package

**Files:**
- Convert: `vidub/stages/translate.py` → `vidub/stages/translate/` (constants.py, engines.py, units.py, qc.py, run.py, `__init__.py`)
- Modify: `tests/test_translate.py`, `tests/test_ollama_mt.py`, `tests/test_hunyuan_mt.py`, `tests/test_glossary.py` (patch targets only)

**Interfaces:**
- Produces: `vidub.stages.translate` package with `build_translator`, `translate`, `SCRIPT_NAME`, `GLOSSARY_NAME`, `_concise_request`, `_CONCISE_VARIANTS` importable unchanged; `timing` package's cross-import keeps working.

- [ ] **Step 1: Failing test** — `tests/test_layout.py` gains `test_translate_is_a_package` (same shape as Task 6).

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Move** — spec D3 table; `__init__.py` re-exports; `_translate_sentence_units` + helpers into `units.py`; `translate` + `main` into `run.py`.

- [ ] **Step 4: Update patch targets** — `_build_refiner` → `translate.engines`, `unit_cut_stats` → owning submodule; spy assertions verified.

- [ ] **Step 5: Verify** — translate-adjacent tests green, then full suite, ruff; also `python -c "import vidub.stages.timing, vidub.pipeline"` (cross-import sane).

- [ ] **Step 6: Commit** — `git commit -m "refactor: split translate stage into a package (pure moves)"`

---

### Task 8: Final gate — CI green locally, live end-to-end from the UI

**Files:**
- Modify: only what earlier tasks' gaps require; `docs/superpowers/specs/2026-10-25-optimization-design.md` status line → approved/implemented.

- [ ] **Step 1: Full verification battery**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check vidub tests scripts tools
.venv/Scripts/python.exe -m vidub.cli run --help | diff tests/data/baseline_run_help.txt -   # only +voxcpm2-voice
.venv/Scripts/python.exe -m vidub.cli doctor
```

- [ ] **Step 2: Live end-to-end from the UI (required, the "no big-bang" proof)** — with the UI running:
  1. Upload `samples/demo_en.mp4` (or paste URL — file upload preferred, no network).
  2. Mode `Subtitles only`, ASR model `cpu-tiny`, `--qc-tts` irrelevant, Run.
  3. Watch the Stages table tick through `ingest → transcribe → translate`; expected stages only (no speak/timing/mix — the panel says Skipped).
  4. Done state: Quality panel shows verdict + levels; Script table has rows; report.json widget populated.
  5. Cancel path: start a second run, click Cancel mid-stage, expect Cancelled state (no thread leak — Run works again).

  Abort criteria: if model download or runtime exceeds ~15 minutes, stop, record where it stopped, and report — do not silently skip.

- [ ] **Step 3: Docs sync** — README feature section mentions the UI option parity; CHANGELOG gains an "Unreleased" section listing: repo cleanup, registry, synced UI/CLI, package splits, CI.

- [ ] **Step 4: Commit** — `git commit -m "docs: changelog + status for the optimization round"`

- [ ] **Step 5: Final report to user** — baseline vs final: test counts, diff of `--help`, live-check evidence (screenshots/observations), CI file ready for push.

# ViDub repo cleanup + config/UI sync + core-file split — design

Date: 2026-10-25
Status: approved (2026-10-25) and implemented; see docs/superpowers/plans/2026-10-25-optimization.md
Chosen approach: **B** (user picked "A + split large core files" from the
2026-10-25 analysis conversation).

## Intent and why

The project is functionally sound (clean stage/engine/utils layering, 47 test
files) but the repository surface has drifted:

- Research leftovers (8 `.vidub-cache-*` dirs, `eval-cache/` containing a whole
  venv, 16 run outputs in `output/`) make the repo root unreadable and pollute
  the CodeGraph index with `site-packages` hits.
- One run option is declared in three hand-maintained places — `config.py`
  (dataclass field + default), `cli.py` (argparse), `ui.py` (Gradio component
  + `form_to_config`). They have already drifted: the UI is missing
  `--isometric*` (4 options), `--speed-as-is`, `--speed-cap`,
  `--retranslate`, `--tts-rate-fit`, `--recenter`, `--align-display`,
  `--hunyuan-refine-max-units`, `--diarize-model`; conversely the CLI has no
  `--voxcpm2-voice` even though the UI exposes the VoxCPM2 voice profile. The
  UI's `hunyuan_refine` form default is `False` while the CLI default is
  `None` (auto).
- `vidub/stages/timing.py` (2,030 lines) and `vidub/stages/translate.py`
  (1,494 lines) each carry 6–8 unrelated concerns in one module; `fit_segment`
  alone is ~360 lines and `fit` ~650, `_translate_sentence_units` ~540.
- `README.md` is 95 KB with a ten-version changelog nested in its intro and
  says "v0.10.0" while `pyproject.toml` is 0.11.0.

Goal: one source of truth per fact, a repo root that reads like a product, and
no core file over ~1,000 lines. Success = `pytest` green before and after,
`vidub --help` byte-compatible flags, UI exposing the same option set as the
CLI (except deliberate UI-only/CLI-only controls, listed below).

## Goals

1. **G1 — Repo hygiene.** Ignore/move research artifacts; scripts to
   `tools/research/`; root docs into `docs/`; changelog out of README.
2. **G2 — Single option registry.** One declarative table describes every
   run option (flag name, dest, type, default, choices, help, UI widget,
   UI group). `cli.py` and `ui.py` build from it. Adding an option = edit
   the table (plus stage code), never three places.
3. **G3 — Sync the option set.** UI gains the 9 missing options; CLI gains
   `--voxcpm2-voice`; `hunyuan_refine` becomes a tri-state (auto/on/off) in
   both surfaces.
4. **G4 — Split `timing` and `translate` into packages.** Pure code moves,
   no logic changes; `vidub.stages.timing` / `vidub.stages.translate` stay
   importable with the same public names (plus the private names tests use).

## Non-goals

- No pipeline behavior change: same defaults, same stage semantics, same
  artifact formats (`job.json`, `report.json`, `timed.json`, ...).
- No split of `export.py` (939), `pipeline.py` (594), `cli.py` (1,072) —
  they stay whole; `cli.py` shrinks naturally via the registry.
- No new features, no UI redesign beyond adding the missing controls in the
  existing accordion structure.
- No deletion of caches/outputs on disk — they are ignored, not removed
  (deleting run artifacts is the user's call).
- No migration of the `eval-cache/quality-rework/voxcpm2-env` venv out of the
  repo — moving a venv breaks it (absolute paths). It gets ignored instead.

## Anti-goals (failure even if it "works")

- A registry so generic it needs decoding (the shortest working diff wins —
  if the table + two loops is harder to read than today's argparse, it is a
  failure).
- Breaking any import path used by tests or by `vidub.pipeline`.
- Silent flag renames. Every `--flag` keeps its exact spelling and dest.

## Constraints

- No git repo exists yet. **Step 0 is `git init` + a baseline commit** after
  a green `pytest` run, so every later step is individually revertable.
  (`.gitignore` needs `.vidub-cache-*/`, `eval-cache/`, `samples/` first so
  the baseline stays small.)
- Tests are the contract: 47 files, ~18.4k lines, including private-name
  imports (`_CONCISE_VARIANTS`, `_concise_request`, `fit_segment`,
  `SCRIPT_NAME`, ...). Package splits must re-export these.
- Windows-first; commands must run under Git Bash + PowerShell.
- Gradio is optional (`[ui]` extra) — `ui.py` import failure must stay
  graceful, and the registry must live in a module importable without
  Gradio (it goes in `vidub/options.py`, importing nothing from `ui.py`).

## Approaches considered

### Approach 1 — Detect-only drift guard
Keep the three hand-written declarations; add a test that fails when the
CLI flag set, `JobConfig` fields and UI form parameters disagree.
Smallest diff, but it only *reports* the drift the user already complained
about; the UI would still be missing 9 options unless they are added by
hand anyway. **Rejected** as the primary approach (kept as a side effect:
the registry makes such a test trivial).

### Approach 2 — Full codegen (chosen)
`vidub/options.py` declares an `OptionSpec` per run option. CLI
`_add_shared_pipeline_args` and the UI settings section are generated by
looping over the table. Hand-written parts stay hand-written: command
structure (`doctor/run/batch/ui`), input controls (upload/URL), run
controls (`--clean`, `--report`, `--resume`, `--force`, `--burn`,
`--cache-dir`, batch-only flags), Gradio wiring (presets, visibility
toggles, handlers, state). Special value transforms (JSON `--voice-map`,
File/Audio components) are per-option `Note` fields with explicit
`ui_only`/`cli_only` opt-outs. **Chosen**: it fixes the root cause, and the
diff, while real, is mostly mechanical.

### Approach 3 — Full declarative app (Typer/Click + Pydantic + auto UI)
Rewrite the CLI on Typer and generate both surfaces from a Pydantic model.
**Rejected**: largest diff, rewrites working code, new dependencies —
against the house rule of reusing what exists.

## Decided (how it gets built)

### D1 — Cleanup map (G1)

| Item | Action |
|---|---|
| `.gitignore` | add `.vidub-cache-*/`, `eval-cache/`, `samples/` |
| `.zcodeignore` | same three lines (custom section below the sync marker) |
| `scripts/eval_*.py`, `scripts/*phase2*.py`, `research_*.py` | move to `tools/research/` (namespace package, no `__init__.py` needed); `quality_benchmark.py` stays in `scripts/` (it is a maintained entry point) |

**Test-import ruling (builder check Q2, settled).** Three test files import
moved modules (`tests/test_eval_asr_rework.py`, `tests/test_eval_mt_rework.py`
import `scripts.eval_*`; `tests/test_quality_benchmark.py` imports
`scripts.quality_benchmark`, which imports `eval_sentence_lock`). Their
import lines are updated to `tools.research.*` — three one-line edits, no
test logic changes.
| `DESIGN.md`, `QUALITY-ROADMAP.md` | move to `docs/` |
| README changelog block | extract to `CHANGELOG.md`; README keeps features + usage; README status line updated to 0.11.0 |
| `vidub.egg-info`, `__pycache__`, caches | left on disk, ignored |
| CodeGraph | after the venv dir is ignored, request the user re-index if the index still surfaces `eval-cache` hits |

### D2 — Option registry (G2, G3)

New `vidub/options.py` (stdlib only, no gradio import):

```python
@dataclass(frozen=True)
class OptionSpec:
    field: str          # JobConfig field name (source of truth for dest)
    flag: str           # "--speed-cap"
    kind: str           # "bool" | "int" | "float" | "str" | "tri" | "path" | "json-map"
    default: object     # same constant config.py uses
    choices: tuple | None
    help: str           # the exact current help text
    ui: str | None      # "checkbox" | "number" | "text" | "dropdown" | None (cli_only)
    ui_label: str
    ui_group: str       # "main" | "engines" | "timing" (accordion placement)
    cli: bool = True    # False for UI-only options
```

- `cli.py::_add_shared_pipeline_args` becomes a loop; `--no-*` aliases that
  exist today (`--no-translation-qc`, `--no-retranslate`) are declared as
  alias fields on the spec so spellings stay byte-identical.
- `ui.py::build_app` builds standard widgets from the table in the same
  order as today's `inputs=[...]`; special components (video upload, URL,
  glossary/pronounce files, VoxCPM2 voice sample, force-stages checkboxes,
  presets, cache dir) stay hand-written and are marked `ui_only` in the
  table (glossary/pronounce/ref-voice/voxcpm2-voice) or remain outside it
  (run controls).
- `form_to_config` maps form kwargs by `spec.field`; only the File→path and
  Audio→path coercions stay as explicit special cases.
- `hunyuan_refine` becomes `"tri"` kind: CLI keeps
  `--hunyuan-refine {auto,on,off}` **no** — keep the exact current
  `--hunyuan-refine` / `--no-hunyuan-refine` BooleanOptionalAction with
  default `None` on the CLI; the UI replaces the checkbox with a dropdown
  `Auto (recommended) / On / Off` mapping to `None/True/False`. No CLI
  spelling changes.
- New synced options (builder check Q4, settled — rule: **every registry
  option appears on both surfaces unless marked `cli_only`/`ui_only`**):
  CLI gains `--voxcpm2-voice` (choices from `VOXCPM2_VOICE_PROFILES`,
  default `DEFAULT_VOXCPM2_VOICE`); UI gains the missing run options —
  `isometric`, `isometric_headroom`, `isometric_cps`,
  `isometric_short_coverage`, `isometric_short_window` (5 flags, a new
  "Timing & budgeting" accordion), `speed_as_is`, `speed_cap`,
  `retranslate`, `tts_rate_fit`, `recenter_audio`, `align_display`,
  `hunyuan_refine_max_units`, `diarize_model` (engines accordion), and
  `sub_margin_v` (a Number next to the subtitle controls). `tail_pad` is
  already on the UI and stays.
- A new test asserts registry ↔ `JobConfig` fields ↔ CLI dests ↔ UI form
  coverage stay in lockstep (the Approach-1 guard, now nearly free).
- **Per-surface defaults (builder check Q3, settled).** The registry cannot
  carry a single default for every option: `asr_model` is `"auto"` on the
  CLI but `"base"` on the dataclass, `cache_dir` is `None` on the CLI and
  `".vidub-cache"` on the dataclass. So: `OptionSpec.default` = the
  dataclass default; optional `cli_default` / `ui_default` overrides where
  a surface legitimately differs today. The lockstep test compares flag/
  field/widget *existence*, never default equality except where the spec
  carries one default. Deliberate behavior keep: the UI's ASR dropdown
  stays seeded from `JobConfig()` (`"base"`, today's behavior), and
  `cache_dir` stays a hand-written run control on both surfaces, outside
  the registry.

### D3 — Package splits (G4)

`vidub/stages/timing.py` → `vidub/stages/timing/`:

| New module | Moves (current symbols) |
|---|---|
| `constants.py` | artifact names, suffixes, thresholds (`SPEECH_NAME`... `_RATE_SUFFIX`) |
| `units.py` | `_group_script_units`, `_unit_text`, `_TimingUnit`, `_source_shares`, `_units_from_file`, `_timing_units`, `compute_coverage`, `_missing_indices` |
| `align.py` | `_word_times`, `_check_align_words`, `_align_unit` |
| `synth.py` | `_synth_retranslation`, `_clamp_rate`, `_fit_to_window`, `detect_silences`, `choose_smart_cut`, `_fit_clip`, `fit_segment`, `_assert_no_bleed`, `_assert_end_boundary`, `_speech_rel_dir` |
| `clipqc.py` | `_normalized_engine`, `_piper_voice_files`, `_synth_variant`, `_write_tts_review`, `run_tts_qc` |
| `run.py` | `fit`, `main`, `_check_mode` |
| `__init__.py` | re-exports every public name + the private names `tests/` uses (`fit_segment`, `SCRIPT_NAME`, ...) |

`vidub/stages/translate.py` → `vidub/stages/translate/`:

| New module | Moves |
|---|---|
| `constants.py` | `SCRIPT_NAME`, `GLOSSARY_NAME`, ... |
| `engines.py` | `build_translator`, `_build_refiner`, `_refine_engine`, `_is_low_confidence`, `truncate_translation` |
| `units.py` | `_UnitPathResult`, `_accepts_budgets`, `_engine_accepts_budgets`, `_budgeted_repair`, `_concise_request`, `_chat_candidate`, `_compression_candidates`, `_rejection_reason`, `_unit_rows`, `_passthrough_guard_record`, `_translate_sentence_units`, `_CONCISE_VARIANTS` |
| `qc.py` | `_run_qc_battery`, `_script_row` |
| `run.py` | `translate`, `_build_per_segment_script`, `main` |
| `__init__.py` | re-exports (incl. test-used privates `_concise_request`, `_CONCISE_VARIANTS`) |

Rules: pure moves, imports adjusted, no signature or logic edits; the one
cross-import (`timing` needs `build_translator`) goes through the package
`__init__`. Module-level docstrings move with their sections.

**Monkeypatch ruling (builder check Q1, settled).** Tests patch
module-level names (`tests/test_timing.py:812` patches `run_ffmpeg` on the
timing module, `tests/test_translate.py:1569` patches `_build_refiner` on
the translate module, ...). After the split those patch targets move to the
owning submodule, so the affected test lines are updated to patch the new
owner (`vidub.stages.timing.synth.run_ffmpeg`,
`vidub.stages.translate.engines._build_refiner`, ...). Assertion semantics
stay identical; only the patch target strings change. The `ttsqc`
submodule of the timing package is named `clipqc.py` (not `ttsqc.py`) so it
cannot shadow the existing `vidub.utils.ttsqc` binding the tests patch
through `timing_mod.ttsqc`.

### D4 — Verification

1. Baseline: `pytest` green → `git init` + commit.
2. After each of D1/D2/D3: full `pytest`; for D2 also
   `python -m vidub.cli run --help` diffed against the baseline (must be
   identical plus the one new `--voxcpm2-voice`), and a scripted
   `form_to_config` smoke check.
3. `vidub doctor` still exits 0.

## Left to the builder

- Exact line-range boundaries of each moved function (doc says which
  symbols, not which lines).
- Whether `scripts/quality_benchmark.py` imports any other moved `eval_*`
  module beyond `eval_sentence_lock` (fix the import or also move it).
- Any test helper in `tests/_engine_runner.py` that hardcodes `scripts/`
  paths.
- The CodeGraph re-index request wording to the user.

## Builder-check calls summary (settled by the author, 2026-10-25)

1. Tests patch moved module-level names → those patch targets update to the
   owning submodule; assertion semantics unchanged (D3 ruling).
2. Moved eval scripts → the three affected test import lines update to
   `tools.research.*` (D1 ruling).
3. Registry carries per-surface defaults where they legitimately differ;
   lockstep test compares existence, not default equality (D2 ruling).
4. Sync rule is "every registry option on both surfaces unless marked
   cli_only/ui_only"; UI gains `sub_margin_v` and the 5-flag isometric
   group; `tail_pad` already synced (D2/G3 ruling).

# Vietnamese Quality Benchmark Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, reviewable En-to-Vi benchmark that compares the current pipeline with candidate ASR and MT engines before any production default changes.

**Architecture:** Read existing job artifacts into a versioned, immutable case list. Run each candidate against identical audio or verified source text, save outputs separately, then create blinded review sheets and machine summaries. Keep model dependencies in evaluation environments; production stages and defaults remain unchanged in this plan.

**Tech Stack:** Python 3.10+, stdlib JSON/CSV/subprocess/hashlib, existing `faster-whisper` and `envit5`, isolated `qwen-asr`, local Ollama serving official Hy-MT2 GGUF, `jiwer` in the evaluation environment, pytest, ffmpeg.

**Spec:** `docs/superpowers/specs/2026-10-24-vietnamese-quality-rework-design.md`

## Global Constraints

- Local operation; retain the commercially usable default path and all existing CLI modes.
- RTX 3080 with 10 GB VRAM: run candidate models sequentially and record elapsed time and observed peak GPU memory.
- First promotion comparison is English-to-Vietnamese only; other source-language defaults remain unchanged.
- Candidate outputs never overwrite `.vidub-cache/`, `output/`, `segments.json`, `units.json`, `script.json`, or existing human edits.
- Use a new `.eval-venv-qwen3/` for `qwen-asr`. Do not reuse the existing `.eval-venv/`, which the TransQuest experiment pins to an older Transformers stack.
- Human listening is unavailable to Codex in this host. Machine scores and bilingual text review cannot promote a TTS or mix default.
- This workspace has no `.git` directory. Do not initialize Git solely for the plan; save reproducible outputs and verification logs under a new `eval-cache/quality-rework/` directory. Commit each task only if the owner later provides a Git checkout.

## Review Focus

1. A missing or unreadable job audio file must fail `prepare` with its case/job path, rather than create unreviewable cases (Task 1 test).
2. Two jobs with the same unit index must produce distinct case IDs and preserve input ordering (Task 1 test).
3. An ASR/MT candidate that returns fewer outputs than inputs must fail with the missing IDs; it must never shift texts onto neighboring cases (Tasks 2 and 3 tests).
4. A human review sheet may contain blank judgments but must never treat them as approval or a scored win (Task 4 test).
5. A held-out case must remain outside development aggregates and prompt/threshold tuning; its score appears separately (Task 4 test).

---

## File map

- `scripts/quality_benchmark.py`: corpus preparation, schema checks, candidate joins, blinded review-sheet export, and score summaries. No model inference.
- `scripts/eval_asr_rework.py`: run existing faster-whisper and isolated Qwen3-ASR against the same clipped audio windows.
- `scripts/eval_mt_rework.py`: run envit5 and official-template Hy-MT2 through Ollama against the same verified source text.
- `tests/test_quality_benchmark.py`, `tests/test_eval_asr_rework.py`, `tests/test_eval_mt_rework.py`: small synthetic artifact and fake-engine checks; no model downloads.
- `docs/quality-rework-benchmark.md`: corpus provenance, commands, frozen review rubric, limits, actual results and promotion decision after runs.

## Task 1: Immutable case corpus

**Files:** Create `scripts/quality_benchmark.py`; create `tests/test_quality_benchmark.py`.

**Interfaces:** `collect_cases(job_paths: list[Path], holdout_paths: set[Path]) -> list[dict]` reads each job's `job.json`, `units.json`, and `audio.wav`. Each JSONL row has `id = f"{video_hash}:unit:{index}"`, `split` (`development` or `holdout`), absolute `audio`, `audio_sha256`, `start`, `end`, `speaker`, `source_asr`, and `baseline_vi`. `write_cases(cases: list[dict], output: Path) -> None` writes UTF-8 JSONL to a new path and refuses overwrite. `apply_verified_sources(cases: list[dict], corrections: dict[str, str]) -> list[dict]` copies cases and adds `verified_source` by ID without changing `source_asr`. CLI: `prepare --job PATH` (repeatable), `--holdout-job PATH` (repeatable), `--out PATH`; `apply-source --cases PATH --corrections PATH --out PATH`, where corrections is UTF-8 TSV with `id` and `verified_source` columns.

- [x] **Step 1: Write failing tests** for stable distinct IDs across two job fixtures, retained unit order and Unicode, a missing audio error naming the job, refusal to overwrite an output file, and a correction that preserves original `source_asr` while adding `verified_source` to only its matching ID.
- [x] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/test_quality_benchmark.py -q`; expect failures for the missing functions.
- [x] **Step 3: Implement** the three signatures and `prepare`/`apply-source` CLI in `scripts/quality_benchmark.py`. Use the existing `units.json` `units` list and `job.json` `video_hash`/`paths.audio`; hash each audio file once per job. Reject malformed unit spans, duplicate IDs, absent files, unknown correction IDs, and non-En-to-Vi jobs explicitly.
- [x] **Step 4: Run** the focused test; expect all Task 1 tests to pass.
- [x] **Step 5: Prepare** a scratch corpus from the two `eval-cache/phase2-rerender/*/job.json` jobs. Verify case count, unique IDs, audio hashes, and that both source job directories are byte-identical before/after. Holdout selection remains an explicit input, never inferred from the same two jobs.

## Task 2: Matched ASR candidate outputs

**Files:** Create `scripts/eval_asr_rework.py`; create `tests/test_eval_asr_rework.py`.

**Interfaces:** `run_asr_cases(cases: list[dict], transcribe: Callable[[Path, float, float], str], engine_id: str) -> list[dict]` returns exactly one `{id, engine, text, elapsed_seconds}` row per case. CLI: `--cases PATH --engine {faster-whisper,qwen3} --model NAME --out PATH`. The Qwen3 path uses the official model in an isolated evaluation environment; it does not import `qwen-asr` into the production environment. Each runner records model/checkpoint revision, command and observed peak GPU memory in a sidecar `run.json`; poll `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` during generation and mark the measurement unavailable if the command is absent. The ASR CLI requires a valid 64-hex `audio_sha256` for every case and verifies it before inference.

- [x] **Step 1: Write failing tests** with a fake transcriber for time-window and ID preservation, Unicode text, and a missing/empty output that names the offending ID. Test `--out` overwrite refusal.
- [x] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/test_eval_asr_rework.py -q`; expect failures.
- [x] **Step 3: Implement** `run_asr_cases` and the existing faster-whisper runner first. Clip each source window with ffmpeg into a per-run scratch directory; validate `start < end` and that the clip has audio. Keep beam/word-timestamp settings in `run.json`.
- [x] **Step 4: Add** the isolated Qwen3-ASR runner with the same output contract. Pin its model revision and package versions in `run.json`; do not use Qwen3-ForcedAligner for Vietnamese timing, which it does not support.
- [x] **Step 5: Run** the focused tests, then both engines on identical prepared cases. Verify ID sets are equal and each model's text is stored separately. Record runtime and actual peak VRAM; do not change the pipeline default. (Inference was safely skipped: neither faster-whisper checkpoint nor the isolated Qwen environment/checkpoint is cached; runners refuse downloads and leave no output files.)

## Task 3: Matched MT candidate outputs

**Files:** Create `scripts/eval_mt_rework.py`; create `tests/test_eval_mt_rework.py`.

**Interfaces:** `run_mt_cases(cases: list[dict], translate: Callable[[str], str], engine_id: str) -> list[dict]` returns `{id, engine, text, elapsed_seconds, input_status, source_sha256}` per case. It consumes a reviewed `verified_source` field when present; otherwise `input_status` is `source_asr_unverified` and excludes that row from promotion scoring. CLI: `--cases PATH --engine {envit5,hy-mt2} --model NAME --out PATH`. The Hy-MT2 adapter sends the official target-language/terminology template to local Ollama and extracts translation text only. Save the SHA-256 of exact prompt templates, checkpoint ID, decoding settings and GPU observation in `run.json`, using the same GPU poll as Task 2. Review-sheet generation and scoring reject same-ID translation rows with different source-text hashes.

- [x] **Step 1: Write failing tests** for identical source input across engines, exact ID cardinality, a blank translation error naming its ID, and the `source_asr_unverified` label.
- [x] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/test_eval_mt_rework.py -q`; expect failures.
- [x] **Step 3: Implement** `run_mt_cases` with the existing `Envit5Translator` and a stdlib `urllib` Ollama caller using Hy-MT2's documented prompt. Load only source-relevant glossary terms. Keep this evaluation adapter outside `vidub/stages/translate.py`.
- [x] **Step 4: Run** the focused tests, then compare envit5, Hy-MT2-1.8B and Hy-MT2-7B sequentially on the same verified development cases. Fail a run if any ID is missing, duplicated, or shifted. Keep raw outputs and reasons for failures. (Envit5 ran on all 112 cases. Hy-MT2-1.8B/7B were skipped because neither the local Ollama server nor either model is available; the runner refused a pull and left no candidate output.)

## Task 4: Blinded review and measured comparison

**Files:** Modify `scripts/quality_benchmark.py`; modify `tests/test_quality_benchmark.py`; create `docs/quality-rework-benchmark.md`.

**Interfaces:** `make_review_sheet(cases: list[dict], candidates: list[list[dict]], seed: int, output: Path) -> None` writes a UTF-8 TSV with stable case IDs, source and anonymized candidate columns plus a separate private candidate key. `summarize(cases: list[dict], candidates: list[list[dict]], judgments: list[dict]) -> dict` produces separate development and holdout counts, ASR WER/CER where corrected source exists, per-engine/per-split translation judgment and severity counts, unresolved rows, runtime and VRAM. Completed judgments and difficult-case tags require reviewer attribution; blank judgments count as `unreviewed`, never `pass`. `score --cases PATH --candidate PATH... --judgments PATH --out PATH` writes `comparison.json` to a new path and verifies that the candidate files match the review key.

- [x] **Step 1: Write failing tests** for deterministic blinding with a seed, candidate ID mismatch, blank judgment treatment, source corrections, and holdout isolation. A missing holdout or fewer than 30 reviewer-tagged difficult cases must mark the report `provisional`, never promotion-eligible.
- [x] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/test_quality_benchmark.py -q`; expect failures for review/score functions.
- [x] **Step 3: Implement** the two signatures and `score` CLI. Load `jiwer` lazily, with a named standard-library fallback when it is unavailable. Preserve exact text and keep machine metrics separate from reviewer severity labels. The review TSV has a `difficulty_reason` column; at least 30 populated difficult cases and one held-out job are required for a promotion-eligible report. Add the reviewer rubric, scope rule and candidate provenance to `docs/quality-rework-benchmark.md` before looking at candidate scores. The blinded candidate mapping is kept in a separate `.key.json` file.
- [x] **Step 4: Run** all three focused files, then `.\.venv\Scripts\python.exe -m pytest -q`; expect no new failures. Produce `cases.jsonl`, model output JSONLs, review TSV and `comparison.json` under a new `eval-cache/quality-rework/` run directory. (Final focused suite: 36 passed; Ruff and py_compile passed. Fresh verified artifacts are review-verified.tsv, its private key, and comparison-verified.json. The full suite remains unverified: the run stalled in `tests/test_ui.py::test_build_app_returns_app_and_state_without_launching`; three known model/network tests and long optional model smoke tests were also excluded. See the SDD ledger.)
- [x] **Step 5: Review** corrected source, translation and technical audio signals with Codex; leave perceptual TTS/mix ratings explicitly unreviewed. Fill final benchmark results only after the owner or designated reviewers return ratings. No engine is promoted by this task. (No corrected transcript or human ratings were available. Codex reviewed representative bilingual text differences and objective audio stream/loudness signals only; the final report remains provisional.)

## Self-review and handoff

- Spec coverage in this milestone: immutable baseline, matched ASR and MT candidates, provenance, independent evaluation, holdout separation, human review, and no default promotion. The separate subtitle text contract, TTS integration and audio-mix changes depend on these results and require their own plans.
- Every task has a focused failing test, implementation action, and verification. The five Review Focus cases map to Tasks 1–4.
- The shared case/output `id` remains the join key throughout. `verified_source` is optional but required for promotion scoring.
- This plan produces working, testable benchmark software without assuming that a new model wins. Write the follow-on plans only after the benchmark identifies useful candidates and the human review gate is staffed.


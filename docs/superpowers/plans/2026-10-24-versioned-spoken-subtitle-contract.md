# Versioned Spoken and Subtitle Text Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (Native execution was selected). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each translated sentence keep separate spoken and subtitle Vietnamese while old jobs continue to use `vi` unchanged.

**Architecture:** Add optional `spoken_vi` and `subtitle_vi` selections to sentence units and script rows. A shared resolver falls back to `vi` for old artifacts. Speech and timing read `spoken_vi`; subtitle export reads `subtitle_vi`. Deterministic signatures make a spoken edit invalidate speak through export, while a subtitle-only edit invalidates export alone.

**Tech Stack:** Python 3.10+, existing JSON artifacts, hashlib/json stdlib, pytest, ffmpeg.

**Spec:** `docs/superpowers/specs/2026-10-24-vietnamese-quality-rework-design.md`

## Global Constraints

- Keep the local commercially usable default path, CLI modes, stage resume, artifact formats, and old jobs usable.
- Preserve `vi` as the fallback for old caches and existing consumers.
- Subtitle edits may shorten text only while retaining source meaning; unresolved or infeasible cues remain reviewable.
- Candidate outputs stay isolated; do not overwrite `.vidub-cache/`, `output/`, source jobs, or human edits.
- No new model, dependency, network service, or default change is part of this contract step.

## Review Focus

1. Old rows with only `vi` must speak and export the same text as before.
2. A unit with explicit `spoken_vi` and `subtitle_vi` must synthesize the spoken selection and export the subtitle selection.
3. A subtitle-only edit must leave speech, timing, and mix stages cached, then rerun export.
4. A spoken-text edit must rerun speak, timing, mix, and export.
5. Sentence-unit and legacy per-segment jobs must both retain their current fallback behavior.

## File map

- `vidub/utils/text_selection.py`: resolve selected text and compute stable signatures.
- `vidub/manifest.py`: persist text-contract version and last successful spoken/subtitle signatures.
- `vidub/stages/translate.py`: initialize both selections from the accepted translation.
- `vidub/stages/speak.py`, `vidub/stages/timing.py`: consume spoken text.
- `vidub/stages/export.py`: consume subtitle text, preserving legacy timed-retranslation behavior when no explicit subtitle selection exists.
- `vidub/pipeline.py`: invalidate only stages that consume a changed selection.
- `tests/test_text_selection.py`, `tests/test_speak.py`, `tests/test_timing.py`, `tests/test_export.py`, `tests/test_pipeline.py`: regression coverage at the shared selection and stage boundaries.
- `docs/quality-rework-benchmark.md`: describe how to review separate text selections and the current preview's limits.

## Task 1: Add the versioned text selections

**Interfaces:** `selected_text(row: dict, field: str) -> str` returns a nonblank requested selection or legacy `vi`. `selection_signature(rows: list[dict], field: str) -> str` hashes ordered IDs/indices and effective text. `units.json` and `script.json` carry the two selections; new translations initialize both to accepted `vi`.

- [x] Write failing tests for fallback, explicit selection, blank optional-field fallback, and stable/different signatures.
- [x] Run `pytest tests/test_text_selection.py -q`; confirm the new module/API is missing.
- [x] Implement the small shared resolver and signature helper; add a manifest contract version field.
- [x] Update translation output to populate equal initial selections on sentence units and member script rows.
- [x] Add stage tests proving `speak`/`timing` use spoken text and subtitle export uses subtitle text, including old-cache fallback and explicit text despite a timing retry.
- [x] Run focused tests; verify old `vi`-only fixtures remain byte-equivalent in visible text.

## Task 2: Invalidate cached stages by selected text

- [x] Add a pipeline test where only `subtitle_vi` changes and only export reruns.
- [x] Add a pipeline test where `spoken_vi` changes and speak, timing, mix, and export rerun.
- [x] Implement per-selection signatures on the manifest. Empty signatures mean legacy/unknown and do not invalidate; record new signatures only after their consuming stages finish successfully.
- [x] Run focused tests; 211 passed, 3 model-backed tests deselected.
- [x] Run the broad suite and `ruff check`; document model/network-dependent exclusions.

## Task 3: Rebuild the isolated preview and document it

- [x] Write an isolated preview artifact with `spoken_vi` equal to the existing voiced text and `subtitle_vi` equal to the edited candidate text.
- [x] Export into a new `output/quality-rework-preview/` path; preserve all earlier outputs.
- [x] Decode the MP4 and check the generated SRT contains the selected subtitle text. Keep QC `review` when constraints remain unresolved.
- [x] Update the benchmark doc with the contract, separate audio/subtitle provenance, and exact preview links.

## Self-review

- The selected design requires separate text choices, backward compatibility, distinct invalidation, and reviewable infeasible captions; Tasks 1–3 cover these requirements.
- Test scope includes legacy `vi`-only artifacts, explicit selections, blank values, sentence-unit/legacy paths, and both cache-invalidation paths.
- This milestone does not select a TTS winner or claim perceptual audio improvement; the current Piper checkpoint is not cached locally.

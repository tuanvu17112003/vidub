# Sentence-locked units + aligned display (SA-1): reference A/B

**Status:** complete — the A/B is measured on the User's own video and every
target is reported as MET / NOT MET below, misses included.
**Milestone:** SA-1 · **Version:** v0.9.0 · **Date:** 2026-09-30

This is the measurement record for **SA-1**. It changes the translation +
timing pipeline so that a **sentence** is the `script.json` row unit and a
sentence-scoped **unit** is the translation + QC + synthesis unit, then
force-aligns each unit's synthesized audio to give every member row a real
display span. The harness that produced these numbers is
[`scripts/eval_sentence_lock.py`](../scripts/eval_sentence_lock.py).

## The defect, traced in the User's artifacts (read-only)

Job studied: `.vidub-cache/109d180f5a6102dfc1719e0f96713441b7796f36/` — the
User's own run (`youtube-5sLYAQS9sWQ`, `--mt-engine hunyuan`, refine off).

The User reported that a sentence came out broken: `… là ví dụ của` ⏸ `… được
đào` / `tạo …`. Reading the artifacts shows six independent links in one chain,
each one evidence rather than inference:

1. **The unit boundary itself was mid-sentence.** The ASR rows
   `… generalizable and adaptable output. And large language models are
   instances of` / `foundation models applied specifically to text and text like
   things. I'm talking …` were split across two units. The cut fell *inside one
   sentence* because the old `MAX_UNIT_SECONDS = 18.0` cap fired and there was
   no "safe" cut inside the run — no terminal punctuation at a row end and no
   ≥ 0.45 s pause (`end(9) = split(10) = 63.66`, a zero-length gap).
2. **The unit translation dropped content.** The concatenation of the three
   `vi` pieces of unit `[10,11,12]` is only the translation of its *last*
   sentence; the first sentence's rendering appears nowhere in the job
   directory. The old validators only checked emptiness / diacritics / digits /
   an *upper* length bound, so an omission was never flagged.
3. **`split_translation` then cut that short translation inside a Vietnamese
   clause and inside a compound** — the `… được đào` / `tạo trên …` break the
   User heard, plus a dangling `… là ví dụ của`. Proportional share-splitting
   can only snap to whitespace, so a compound word is always splittable.
4. **The dub spoke the broken text.** Timing synthesized one WAV per unit from
   the joined member `vi`, so the audio was continuous — but a continuous
   reading of an incomplete text.
5. **The QC battery scored the fragments, not the unit.** `translation_qc.json`
   flagged rows 10/11/12 `mistranslation-risk` with a per-fragment retry that
   could never see (and so never recover) the missing sentence.
6. **The display was per fragment.** `script.json` rows, the UI table and
   `subtitles.vi.srt` all showed the split pieces, so the User *saw* the
   truncation.

Root causes: (a) units could end mid-sentence; (b) a translation unit could
silently drop content with no completeness check; (c) the text that was spoken
was derived by proportional splitting of another string; (d) display text had no
relation to the audio actually synthesized.

## Architecture (SA-1)

The pipeline now runs, in order:

1. **Sentence rows.** The translate stage splits every ASR row at internal
   sentence boundaries (`vidub.utils.sentences.split_into_sentences`). Each
   sentence's `start`/`end` are **locked to the ASR word edges** when the row's
   `words` line up with its tokens, and interpolated by character fraction
   (clamped inside the row) only when they do not. `manifest.segments` and
   `segments.json` (the ASR rows) stay untouched; `script.json` rows are the
   **sentence rows**.
2. **Sentence-scoped units + `units.json`.** `group_units` merges a fragment
   run with the complete sentence it finishes; a unit ends **only** at a
   completed sentence or a speaker change (see the guarantees below). The
   authoritative `units.json` records, per unit, the member row `indices`, the
   locked `start`/`end`, the source, the final `vi`, and the guard/battery
   status.
3. **Unit-level guard + QC.** A completeness guard
   (`vidub.utils.units_guard`) checks each unit's translation against its source
   (word coverage and sentence loss) and re-samples (best-of-K) or falls back
   before the deterministic passes run **on the unit text**. The translation-QC
   battery then scores **units**, and the per-row display text is re-derived
   from the corrected unit text.
4. **Per-unit audio.** The timing stage synthesizes each multi-member unit
   **once** from its `units.json` `vi` (verbatim) into `speech/unit_XXXX.wav`
   and places a single `timed.json` row across the unit's window; single-member
   units reuse the speak-stage WAV.
5. **Aligned display spans.** After the final audio exists, each multi-member
   unit's WAV is transcribed with the project ASR and the words are allocated to
   members (`vidub.utils.align.allocate_spans`), producing
   `timed.json`'s `member_spans` (`align: "asr"`); without usable ASR words the
   fallback is the proportional `split_translation` split over the members'
   locked script spans (`align: "split"`).

## Guarantees — exactly as the code implements them

**Sentence rows and units**

* `split_into_sentences` returns one row per sentence, with `start`/`end`
  clamped to and monotone inside the ASR row's own `[start, end]`. A row with no
  internal sentence boundary (or empty text) is returned unchanged.
* `group_units` ends a unit **only** when the last member's text ends a sentence
  (terminal `.` `!` `?` `…`, trailing closers `"'”’)` skipped) or when the next
  row has a **different speaker**. A silence gap never ends a unit. Consecutive
  complete sentences are therefore **not** merged.
* The hard ceiling `HARD_UNIT_WORDS = 150` / `HARD_UNIT_SECONDS = 45.0` is the
  **sole** size guard. When a single oversized sentence exceeds it and holds no
  internal sentence boundary, the unit closes at the largest internal gap
  ≥ `PAUSE_CUT_GAP_S = 0.6` s (a counted *pause cut*, the documented last
  resort); with no such gap the unit **stays whole** and is counted `oversized`.
  No code path takes a mid-sentence cut, so
  `unit_cut_stats(...)["mid_sentence_cuts"]` is always `0`.
* With `units.json` present and consistent (its `indices` partition every script
  row exactly once), the timing stage uses the recorded members, locked span and
  `vi` **verbatim** and does not re-group. A missing or inconsistent file logs a
  warning and falls back to re-grouping for the whole stage.

**Unit completeness guard** (`vidub/utils/units_guard.py`)

* Signals: `omission-risk` (coverage < `UNIT_COVERAGE_MIN = 0.75`, or empty `vi`
  on a non-empty source), `sentence-loss` (`words_in >= 8`, `sentences_in >= 2`,
  `sentences_out <= sentences_in - 1`, coverage < `1.10`) and `addition-risk`
  (coverage > `UNIT_COVERAGE_MAX = 2.6`).
* On a flagged unit it compares the original, up to `RESAMPLE_K = 3`
  `sample_candidates` and one optional fallback-engine candidate, picks the best
  by rank (no flags beats any flag, then fewer flags, then coverage closest to
  `1.0`; the original wins ties) and records the action
  (`none` / `resampled` / `fallback` / `flagged`).
* The guard **never raises**: a raising engine only drops that candidate.

**Audio placement (unchanged invariants)**

* A placed row never extends past its boundary (the next unit's first member
  start, or the padded video end): the no-bleed assertion holds.
* A hard mid-word cut is only allowed as a documented last resort — a smart trim
  at a detected silence (`smart_trim`), or a blind `clip` flagged with a reason.
* Coverage math (`compute_coverage`) aggregates over the unit's member spans.

**Display alignment**

* `member_spans` is monotone, starts at/after the row's placed `start`, ends
  at/at-before the placed `end` plus a small tolerance, and every `vi` piece is
  non-empty.
* On the `asr` path every span boundary is an ASR **word edge**.

## Non-guarantees (stated so they are not over-claimed)

* **Not a phoneme-level forced aligner.** The display spans come from ASR word
  timestamps on the *synthesized Vietnamese* audio. If the ASR mis-recognizes or
  returns no words, the spans fall back to a **proportional split** over the
  members' locked script spans (`align: "split"`), which is **not** word-snapped.
  The alignment pass is engine-independent in shape but its quality is
  ASR-dependent.
* **The guard checks completeness, not correctness.** It can catch a dropped
  clause or a lost sentence; it cannot tell a fluent-but-wrong translation from
  a good one.
* **No cross-unit pacing guarantee.** Each unit is fitted independently inside
  its own window; neighbouring stretches are smoothed but the stage makes no
  global pacing/timeline promise, and a unit that cannot fit is still stretched
  to the speed cap and, as a last resort, clipped with a reason.
* **The hard ceiling can leave a unit `oversized`.** When an over-ceiling
  sentence has no internal gap ≥ 0.6 s it is kept whole and counted `oversized`
  rather than cut mid-sentence.
* **A unit can still be longer than one "natural" breath.** Sentence-scoped
  units trade finer boundary control for never cutting a sentence.
* The measured engine/ASR numbers below are **one video, one speaker** and are
  not generalizable.

## Flags

| Flag | Effect |
|---|---|
| `--sentence-units` / `--no-sentence-units` | Sentence rows + sentence-scoped units with unit-level audio/QC (default **on**). `--no-sentence-units` restores the per-segment path byte-for-byte (no `units.json`, no `align`/`member_spans`). |
| `--align-display` / `--no-align-display` | ASR forced alignment of each multi-member unit's final audio for `member_spans` (default **on**). `--no-align-display` skips the alignment ASR entirely and records the `split` fallback. |
| `--mt-engine {envit5,hunyuan,ollama}` | Machine-translation engine. |
| `--asr-model MODEL` | ASR model, reused (no new download) for the alignment pass. |
| `--no-tts-rate-fit` / `--no-recenter` | Restore the single fixed rate / no recentring (FIT-1, unchanged). |

## Pre-registered targets (declared before reading the numbers)

These targets were fixed before the A/B was measured. The measured column is
filled by the Results section below.

| # | Metric | Target | Measured |
|---|---|---|---|
| M1 | placed-audio boundaries that cut a source sentence (excluding speaker changes, pause cuts ≥ 0.6 s, transcript end) | 0 | |
| M2 | units with dropped content (coverage < 0.75 or sentence-loss) | 0 | |
| M3 | translation units whose text ends mid-sentence | 0 | |
| M4 | unresolved translation-QC flags (`translation_qc.json` `segments`, `resolved == false`) | ≤ 5 | |
| M5 | clips / `stretched` | 0 / ≤ baseline | |

## Results — measured A/B

Baseline cache (the User's own pre-SA-1 hunyuan run):
`.vidub-cache/109d180f5a6102dfc1719e0f96713441b7796f36`.
Fixed run (SA-1 build, same input file byte-for-byte, `--asr-model base
--mt-engine hunyuan`):
`.vidub-cache-sa1/run/ff3f5fa05378cfa6da6a2fb07d86b7a02d659344`.

| # | Metric | Baseline | SA-1 run | Target | Verdict |
|---|---|---|---|---|---|
| M1 | placed-audio boundaries that cut a source sentence (excluding speaker changes, pause cuts ≥ 0.6 s, transcript end) | 3 | **0** | 0 | MET |
| M2 | units with dropped content (coverage < 0.75 or sentence-loss) | 2 | **0** | 0 | MET |
| M3 | translation units whose text ends mid-sentence (the `... là ví dụ của` / `... được đào` class; subtitle wrapping splits are excluded) | 15 | **0** | 0 | MET |
| M4 | unresolved translation-QC flags (`translation_qc.json` `segments` with `resolved == false`) | 13 (of 63 rows) | 9 (of 54 units) | ≤ 5 | **NOT MET** (rate improved 20.6 % → 16.7 %; granularity changed row→unit, so the counts are not directly comparable) |
| M5 | clips / stretched | 0 / 5 | 0 / 8 | 0 / ≤ 5 | clips MET; **stretched NOT MET** (8 > 5, max_stretch 1.109 vs 1.106) |

### Supporting facts (from the artifacts)

* **The fixed run's units are sentence-scoped:** 72 sentence rows → 54 units,
  `mid_sentence_cuts 0`, `oversized 0`, and every guard flag `0`
  (`guard_flagged` / `omission` / `sentence_loss` / `addition` / `oversized`).
* **The alignment ran for real:** `timing.json` `align: {asr: 15, split: 0,
  single: 39, unavailable: 0}`.
* **The guard's calibration was verified on the User's broken unit:** coverage
  `29/40 = 0.725` with 3 source sentences → 1 spoken sentence, so both
  `omission-risk` and `sentence-loss` fire, and a faithful candidate ranks
  strictly better.
* **The User's reported sentence is now ONE unit:** source `And large language
  models are instances of foundation models applied specifically to text and
  text like things.`; `vi` is its full Vietnamese translation; locked span
  `61.56 → 69.04` s; placed `61.56 → 67.92` s with `align: asr`.
* **The dubbed output** of the fixed run is
  `output/sa1-youtube-5sLYAQS9sWQ/dubbed.vi.mp4`.

### M4 and M5 — the misses, stated plainly

* **M4 is NOT MET** and the counts are not directly comparable: the baseline
  score is **per ASR/display row** (13 of 63 rows), the fixed run is **per unit**
  (9 of 54 units). As a *rate* it improved from **20.6 % to 16.7 %** of scored
  entries, but the absolute target (≤ 5) is measured on a different granularity,
  so it is not met. The harness records granularity in `scores.json` and never
  compares the two as if identical.
* **M5 `stretched` is NOT MET** and is arguably *harder* to meet by design: a
  sentence unit can be longer than a fragment, so more units need a stretch
  (8 vs 5) and the max stretch rose slightly (1.106 → 1.109). Clips stayed at
  `0`. There is **no cross-unit pacing guarantee** (see above); this is the
  honest cost of never cutting a sentence.

## Reproduce

Score the recorded A/B (read-only; both caches already exist):

```bash
./.venv/Scripts/python.exe scripts/eval_sentence_lock.py \
  score \
  --run .vidub-cache-sa1/run/ff3f5fa05378cfa6da6a2fb07d86b7a02d659344 \
  --baseline .vidub-cache/109d180f5a6102dfc1719e0f96713441b7796f36 \
  --json eval-cache/sa1/scores.json
```

Re-run the fixed arm into a scratch cache (the harness refuses any `--cache-dir`
that does not start with `.vidub-cache-`, and any `--output-dir` under
`output/`, so it cannot touch the User's data):

```bash
./.venv/Scripts/python.exe scripts/eval_sentence_lock.py \
  run --url <youtube-url> \
      --cache-dir .vidub-cache-sa1/run \
      --output-dir .vidub-cache-sa1/work \
      --mt-engine hunyuan --extra=--asr-model --extra=base
```

`score` prints the exact artifact paths it read (`script.json`, `timed.json`,
`units.json`, `translation_qc.json`, `timing.json`); a metric that cannot be
computed from the artifacts is reported `null` with a reason, never invented.

## Threats to validity

* **One video, one speaker, one domain.** The structural results (M1/M2/M3) are
  code-level; the QC/pacing results (M4/M5) are single-sample.
* **Granularity change on M4.** Row counts and unit counts are different
  denominators; only the per-100 rate is comparable, and even that is a coarse
  proxy.
* **The tqc battery is a weak proxy** for sentence-level fluency (numbers /
  entities / terminology / length / consistency, not coherence).
* **ASR-dependent display spans.** `align: asr` quality follows the ASR's
  recognition of the synthesized Vietnamese; mis-recognition degrades spans.
* **No listening study** was run.

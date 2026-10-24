# Q-1 + SU-1 close — measured quality round (v0.10.0)

**Status:** closed at **v0.10.0** — every pre-registered target is reported
MET / NOT MET below, the **three misses included**.
**Milestones:** Q-1 (quality completion) + SU-1 (subtitles-only / vietsub)
· **Version:** v0.10.0 · **Date:** 2026-09-30

This is the measurement record for the round that closes **Q-1** and **SU-1**.
Q-1 attacked the defects the User saw on the tech video
(`.vidub-cache/326ea9c45ee94afbfe3186be9b798ee8a05fbf63`, `--mt-engine hunyuan`):
hunyuan's neighbour-content contamination, the translation-QC flags that survived
it, the audible clips and stretches on tight speech, and the subtitle cues that
drifted from the placed audio. SU-1 added the **vietsub** output mode. All
numbers below are the measured values; nothing is recomputed or smoothed over.

## What changed

- **Q-1-T1 — unit contamination guard.** `vidub/utils/units_guard.py` gains a
  pure `contamination_signals` that flags a unit which **reproduces a neighbour's
  content** (token containment ≥ `CONTAMINATION_OVERLAP = 0.5`) or **adds a
  sentence** (`coverage ≥ ADDITION_SENTENCE_COVERAGE = 1.4`), and a
  **direct, context-free template** candidate
  (`engine.translate([source], …)`, the repair lever). A unit repaired that way
  records `action: "decontext"`; a unit that stays contaminated records
  `"flagged"` and can **never** be labelled `ok`. The translate stage feeds the
  neighbouring units' raw translations in and counts the new flags in
  `translation_qc.json` `units.flags` (`contamination`); `units.detail` carries
  the before/after overlaps.
- **Q-1-T2 — refine pass ON by default for hunyuan.** `cfg.hunyuan_refine` is
  now **tri-state** (`None` = auto → **on for `--mt-engine hunyuan`**, `True` =
  force, `False` = `--no-hunyuan-refine`). The pass re-samples still-flagged units
  and keeps a candidate only when it is **strictly better** under the same
  validators; it is bounded by `hunyuan_refine_max_units = 40` (units beyond the
  budget are recorded as skips) and its outcomes land in `translation_qc.json`.
  `envit5`/`ollama` runs never refine and stay byte-identical.
- **Q-1-T3 — pacing levers.** `vidub/utils/fit.py` + `vidub/stages/timing.py`
  add, in order: a **concise re-translation** dedicated to a clip-bound unit
  (`concise_char_target`), a **bounded short-window stretch**
  (`SPEED_CAP_SHORT = 1.30` for windows below `SHORT_WINDOW_MAX = 1.80` s, only
  when the alternative is a clip), then the **minimal** trim. Every clip records
  `{idx, seconds, reason}`; the zero-bleed invariant is unchanged.
- **Q-1-T4 — word-edge subtitle cues.** The timing stage persists the placed
  unit audio's ASR word timings as `align_words` on **sentence-unit rows** (the
  `asr` alignment path only), and `vidub/utils/subqc.py` splits an over-long cue
  **on real word edges** instead of the proportional character split. Legacy rows
  (no `units.json`, `--no-subtitles-qc`) stay byte-identical.
- **SU-1 — vietsub mode.** `--mode subtitles` is an alias of
  `--mode subtitles-only`: it runs `ingest → transcribe → translate → export`,
  skips `speak` / `timing` / `mix` and **keeps the original audio**; `speak()`
  refuses the mode, the TTS license/language validation is skipped, and
  `report.json` records `engines.tts = "skipped (subtitles-only)"`. Subtitle
  styling is `--sub-pos {bottom,top,center}`, `--sub-font-size` and
  `--sub-margin-v`; the export writes `subtitles.vi.srt` **and**
  `subtitles.vi.ass` and `--burn` applies a styled filter. Full description:
  [`docs/subtitles-only.md`](subtitles-only.md).

## Measured results

Measured on the User's tech video (`326ea9c4…`, `--mt-engine hunyuan`) unless
stated; the pacing row was measured twice (LLM variance) and both arms are shown.

| # | Change | Pre-registered target | Measured | Verdict |
|---|---|---|---|---|
| Q-1-T1 | hunyuan **context contamination** guard | **0 contaminated units** | 11 of 81 units flagged; **4 repaired below the threshold** — including the reported case **unit 24: overlap 1.00/0.70 → 0.17/0.13** — and **6 survivors stay `flagged`** (never silently accepted) | **NOT MET** |
| Q-1-T2 | **refine default-on** for hunyuan | unresolved translation-QC units **≤ 8** | **21 → 0** (direct refine: 21 attempted / 21 improved / 0 worse, 59 s; end-to-end stage run: `verdict=pass`, 81/81 `ok`, 30/30 improved, 373 s) | **MET** |
| Q-1-T3 / T3-C1 | pacing levers — **clips** | **0 clips** | **7 → 4 / 5** | **NOT MET** |
| Q-1-T3 / T3-C1 | pacing levers — **`stretched`** | **≤ 12** | **18 → 22 / 19** (`max_stretch` 1.2 → 1.275 / 1.241, `retranslated` 6 → 8) | **NOT MET** |
| Q-1-T4 | subtitle cues on **word edges** | violations ≤ baseline, boundaries track the audio | controlled A/B on identical audio: cps violations **160 → 145**, duration **72 → 68**, cue boundaries within 50 ms of a word edge **0.217 → 0.321**, `cps_max` unchanged | **MET** (both violation counts fell) |
| SU-1 | **vietsub mode** (`--mode subtitles`) | V1–V4: alias normalises, real burns land in the right band, `.srt` **and** `.ass`, original audio kept | verified by **three real burns** (text in the correct band) and a **real subtitles-only run** (original aac audio + `mov_text` track) | **MET** |

**Suite at the close: 773 passed / 4 skipped.** SA-1 closed at 700; this round
added **73 tests**.

### The three misses, stated plainly

* **Q-1-T1 — NOT MET.** The guard flagged 11 of 81 units on the real replay and
  repaired **4** below the 0.5 containment threshold; the other **6 stayed
  `flagged`** (forced `low_confidence`, never `ok`). The pre-registered target was
  "0 contaminated units"; as measured it is not met. The repair is a
  deterministic win (unit 24 went `1.00/0.70 → 0.17/0.13`), but it is partial.
* **Q-1-T3 clips — NOT MET.** Clips fell **7 → 4 / 5** across the two runs, not
  to 0. The survivors need **1.30–1.54×** while the caps are **1.20× / 1.30×**:
  the text is simply longer than the window can speak at the allowed speed, so
  the levers cannot close the gap without a shorter translation.
* **Q-1-T3 `stretched` — NOT MET.** `stretched` moved **18 → 22 / 19** (up, not
  down to ≤ 12): trading a clip for a bounded short-window stretch makes more
  rows count as stretched even as audible cuts disappear. Both misses point at
  the same missing lever — **translation-side length budgeting**.

## Reproduce

**Contamination guard (Q-1-T1).** Thresholds and action in
`vidub/utils/units_guard.py`: `CONTAMINATION_OVERLAP = 0.5`,
`ADDITION_SENTENCE_COVERAGE = 1.4`; a repaired unit records
`action: "decontext"`. Re-read the fixtures in `tests/test_units_guard.py`, or
replay the real job read-only — load
`.vidub-cache/326ea9c45ee94afbfe3186be9b798ee8a05fbf63/{units.json,script.json}`
and run the guard path with the real hunyuan engine, printing per unit the
`overlap_prev` / `overlap_next` before and after, the chosen action and the
before/after text. A clean neighbour pair must stay silent; a survivor must be
`flagged`, never `ok`.

**Refine default (Q-1-T2).** Defaults and flags: `--mt-engine hunyuan` turns the
refine pass **on** (`cfg.hunyuan_refine` is `None` = auto); `--no-hunyuan-refine`
opts out; `--hunyuan-refine-max-units` (default **40**) bounds the spend;
`--hunyuan-samples` (default 3) sets the per-unit candidates. Outcomes are in
`translation_qc.json`. Re-run `vidub run <tech video> --mt-engine hunyuan` into a
**scratch** cache (never the User's `.vidub-cache/` or `output/`) and compare the
unresolved count (`translation_qc.json` `segments` with `resolved == false`).

**Pacing constants (Q-1-T3).** In `vidub/utils/fit.py`:
`SPEED_CAP = 1.20`, `SPEED_CAP_SHORT = 1.30`, `SHORT_WINDOW_MAX = 1.80`,
`CONCISE_CHARS_PER_SECOND = 20.0`, `MIN_CONCISE_CHARS = 8`; the pure policy is
`effective_speed_cap` + `concise_char_target`. The per-clip `{idx, seconds,
reason}` records live in `timing.json`'s `pacing.clips`.

**Subtitle cues (Q-1-T4).** `align_words` is the placed unit's ASR word timeline,
`[[start_abs, end_abs, "word"], …]` in absolute seconds, monotone and inside the
placed range; it is written only on the **sentence-unit** `asr` path (never on
legacy rows) and consumed by `vidub/utils/subqc.py`'s word-edge split. The
controlled A/B fixes the audio and changes only the split policy, then reads
`subtitles_qc.json` (`cps` / `duration` violations, `cps_max_observed`) and the
**word-edge accuracy** metric: the fraction of cue boundaries that land within
**50 ms** of an ASR word edge (baseline `0.217` → word-edge split `0.321`). The
residual is honest — a boundary can still be up to one word away.

## What is still open

The three misses share one cause: **the translation is longer than the window
can speak at the allowed speed**, so the timing stage can only clip, stretch or
trim. The next workstream is therefore **isometric length-budgeting** in
translation (the QUALITY-ROADMAP's "isometric MT" line): give the MT/LLM a
per-unit length budget derived from the window and the speed caps, so the text
arrives close to the right length instead of being fixed downstream. That is the
principled successor to Q-1-T3's concise re-translation and the lever the
surviving 1.30–1.54× clips need.

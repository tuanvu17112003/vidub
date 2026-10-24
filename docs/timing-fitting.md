# Two-pass TTS rate fitting — reference A/B evaluation (FIT-1-T3)

**Status:** complete — the reference A/B ran on this machine and every
pre-registered metric is reported below, including the one target that was NOT
met.
**Task:** FIT-1-T3 · **Date:** 2026-09-30

## What is being compared

FIT-1 adds a **rate pass before** the stretch / re-translate / clip logic: a
segment whose natural Piper take does not match its availability window is
re-synthesized once at a fitted `length_scale` (0.85×–1.30× around natural, plus
a video-wide pace offset), and leftover window slack is used by bounded
re-centering. This document measures that change against the previous
single-rate behaviour on the reference video.

| Run | `tts_rate_fit` | `recenter_audio` | Meaning |
|---|---|---|---|
| **BEFORE** | `False` (`--no-tts-rate-fit`) | `False` (`--no-recenter`) | previous behaviour |
| **AFTER** | `True` (default) | `True` (default) | the FIT-1 change |

## Setup

| Item | Value |
|---|---|
| Input | `.vidub-cache/downloads/youtube-a7TOameRqoY.mkv` (`video_hash = ca63d19c…`) |
| Video | single speaker (`spk0`), 74 ASR segments |
| Engine | `envit5` (`VietAI/envit5-translation`, the default) + Piper `vi_VN-vais1000-medium` |
| QC | default on (transcript / subtitle / translation / TTS / final gate) |
| Refine | **off**, so the comparison isolates the rate change |
| The knob under test | `PiperTTS.synth(..., length_scale=…)` per-synthesis rate |
| Rate caps | `RATE_SPEED_MIN = 0.85`, `RATE_SPEED_MAX = 1.30`, `RATE_AS_IS_TOL = 0.08`; baseline `PACE_BASE_MIN = 1.0`, `PACE_BASE_MAX = 1.10`; re-center `RECENTER_MIN_SLACK = 1.5`, `RECENTER_MAX_SHIFT = 2.0` |
| Hardware | Windows 10 (26200), Python 3.10.11, AMD64 12 CPU, NVIDIA RTX 3080 (10240 MiB, driver 591.86) |
| Stages run | `translate,speak,timing,mix,export` (ingest/transcribe reused from a scratch copy of the reference job) |
| Scratch caches | `.vidub-cache-fit1/cache-{before,after}/` (+ `work-{before,after}/` output dirs) |
| Scores | `eval-cache/fit1/scores.json` |

The User's `.vidub-cache/` job (`ca63d19c…`) and `output/` are **read only**;
each arm copied the reference job into its own scratch cache and ran with the
CWD set to a scratch work dir, so its `output/` and every artifact land under
`.vidub-cache-fit1/`.

## Pre-registered acceptance (stated before looking at the results)

1. **Clips → 0.** The AFTER run must produce **zero** `action == "clip"` rows.
2. **Total unused-window seconds ↓ ≥ 50 %.** Sum over rows of
   `window − placed` must fall by at least half versus BEFORE.
3. **Per-segment rate spread** (max/min of vi-words-per-placed-second) reported
   **honestly**, with **no hard target**.
4. **atempo-stretched count** reported (no target).
5. **No new clipping** — the AFTER run must not introduce a clip where BEFORE had
   none.

## Metric 1 — clips (target: 0)

| | BEFORE | AFTER |
|---|---|---|
| `action == "clip"` rows | 1 | **0** |

**Result: MET.** The AFTER run produced **no clips**. BEFORE's single clip was
segment **56** (2.14 s window, needed 1.24×); after the rate pass it re-synthesized
at `length_scale 0.769` and fitted as a **stretch** — no trim, no clip. (The
User's own `53a7f0d6…` job had 6 clips; the fresh reference run used a scratch
re-translation/re-synthesis and happened to produce only 1, so the absolute
BEFORE number is lower than the User's job but the direction is the same.)

## Metric 2 — atempo-stretched (reported, no target)

| | BEFORE | AFTER |
|---|---|---|
| `action == "stretch"` rows | 7 | 7 |
| stretch min / max / mean | 1.0623 / 1.1713 / 1.1063 | 1.0006 / 1.1013 / 1.0502 |
| stretch std / CV | 0.0382 / 0.0345 | 0.0327 / 0.0311 |

**Result: neutral count, milder stretches.** The count is unchanged (7 → 7) but
the stretch distribution moved toward 1.0 (max 1.1713 → 1.1013, mean 1.1063 →
1.0502): the rate pass absorbs most of the rate mismatch, so the surviving
`atempo` work is lighter.

## Metric 3 — per-segment perceived rate (reported, no target)

vi words ÷ placed seconds (`timed` `end − start`) per segment, over the 74 rows:

| | BEFORE | AFTER |
|---|---|---|
| min (words/s) | 2.4493 | 2.4493 |
| max (words/s) | 6.3218 | 6.3638 |
| mean | 4.6571 | 4.6191 |
| std | 0.7101 | 0.7040 |
| **CV (spread proxy)** | **0.1525** | **0.1524** |

**Result: essentially flat / neutral.** The rate spread is unchanged (CV
0.1525 → 0.1524); the mean drifts down very slightly (4.6571 → 4.6191) and the
maximum rises very slightly (6.3218 → 6.3638). This metric is a coarse proxy (see
threats) and shows **no meaningful improvement**, which is reported honestly.

## Metric 4 — unused window (target: ↓ ≥ 50 %)

Sum over rows of `window − placed`:

| | BEFORE | AFTER |
|---|---|---|
| total unused (s) | 93.844 | 92.100 |
| rows with unused > 1 s | 34 | 34 |
| max unused (s) | 5.470 | 5.458 |

**Result: NOT MET — only a 1.86 % reduction** (target ≥ 50 %). The dominant
cause is that the leftover time is **source silence between segments**, not a
mismatch the rate pass can fix: the dead air is silence the *video* has, and the
slow-fill can only lengthen a line to **`RATE_SPEED_MIN` = 0.85×** its natural
pace — a bounded amount that cannot stretch a 3–9 s gap unless the line was
already short. Re-centering (21 rows) **redistributes** that leftover silence
(it shifts a clip later within its window) instead of removing it. The
`rate_fit` block adds a second, visible factor: the video-wide baseline landed
at its **1.10 cap** (`global_length_scale = 1.1`), so after dividing by it the 57
re-synthesized rows cluster near natural (`length_scale` mean **1.023**, min
0.769, max 1.070) — little net slow-fill. Net: unused seconds fell 1.7 s, far
short of the pre-registered half.

## Metric 5 — unit summary + `length_scale` distribution (AFTER)

`timing.json`'s `rate_fit` block (no block on BEFORE):

| Field | Value |
|---|---|
| `resynthesized` | 57 |
| `kept` | 17 |
| `failed` | 0 |
| `global_length_scale` | 1.1 |
| `recentered` | 21 |

`length_scale` over the 57 re-synthesized rows: min **0.7692**, max **1.0695**,
mean **1.0232**, std 0.0972, CV 0.0950. `rate_action` counts: 57 `resynth`,
17 `keep`, 0 `failed`, 0 `off`. No re-synthesis failed.

## Worst-row comparison

Indices = the 6 clipped indices from the User's job (`19, 24, 35, 44, 49, 65`) ∪
the actual BEFORE clips (56) ∪ the 5 fastest / 5 slowest BEFORE rows by
perceived rate. `placed` is the AFTER `end − start`.

| idx | BEFORE action/trim | AFTER action / length_scale / placed | EN |
|---|---|---|---|
| 18 | stretch | asis · ls=0.856 · 3.98 s | some fancy offloading where you constantly swap weights in and out of |
| 19 | asis | asis · — · 4.92 s | memory, but that's like trying to cook dinner by running to the grocery store for each ingredient. |
| 23 | asis | asis · — · 4.90 s | differences between numbers, like 3.141 vs. 3.142. |
| 24 | asis | asis · ls=1.070 · 3.59 s | With FP8, you're looking at chunks of 0.125. |
| 35 | asis | asis · ls=1.070 · 1.14 s | when the internet was dial-up. |
| 44 | asis | asis · ls=1.070 · 1.37 s | and merging the two text files. |
| 47 | asis | asis · ls=0.769 · 1.27 s | and model from Hugging Face, |
| 49 | asis | asis · — · 10.61 s | We imported our custom dataset from merge.txt and created a tokenization function that converts text to numerical tokens with max length 128. |
| 54 | asis | asis · ls=1.070 · 2.39 s | Um, I think we made him too stupid? That's just random slop. |
| **56** | **clip** trim=n/a | **stretch** · ls=0.769 · 2.13 s | So let's try it again without the zeroing. |
| 61 | asis | asis · — · 5.93 s | As you can see guys, the file size is only 63.66 megabytes with 78 tokens a second. |
| 63 | stretch | stretch · ls=0.816 · 1.73 s | Does he know what's 1 plus 1? |
| 65 | asis | asis · ls=1.070 · 0.67 s | One plus one equals… |

The headline row is **56**: the only BEFORE clip becomes a light stretch under
the rate pass. Rows 18 and 63 (BEFORE stretches) keep or gain a fitted rate; the
pre-registered User-clipped indices were already `asis` in the fresh reference
run.

## Verdict (pre-registered targets, real outcome)

| # | Pre-registered target | Outcome | Status |
|---|---|---|---|
| 1 | clips → 0 | 1 → **0** | **MET** |
| 2 | unused-window seconds ↓ ≥ 50 % | 93.844 → 92.100 (−1.86 %) | **NOT MET** |
| 3 | per-segment rate spread (no target) | CV 0.1525 → 0.1524 (flat) | neutral |
| 4 | atempo-stretched count | 7 → 7 (milder: max 1.1713 → 1.1013) | neutral / improved |
| 5 | no new clipping | BEFORE 1 clip, AFTER 0 | **MET** |

**Honest summary.** The win that landed is real but narrow: the rate pass
**eliminated the reference run's only clip** (target 1 MET) and **softened the
surviving atempo stretches** (max 1.1713 → 1.1013), with **57 of 74 rows carried
by naturally-paced re-synthesis** (`rate_fit.resynthesized = 57`) instead of a
fixed rate — so the clipped-segment class is gone on this video. But it did
**not** deliver the pre-registered **≥ 50 % unused-window reduction**: only
**1.86 %** (93.8 → 92.1 s), because the dead air is mostly source silence a
bounded 0.85× slow-fill cannot occupy and re-centering only moves audio. And the
**perceived-rate spread did not improve on this pair** (CV 0.1525 → 0.1524; the
max rate even rose marginally, 6.32 → 6.36 words/s) — reported honestly rather
than dressed up.

## Levers if the User wants to close the unused-window gap

The pre-registered target was not met, so here are the honest options, in order of
expected effect:

1. **Raise the slow side of the cap** (`RATE_SPEED_MIN`, today 0.85×): allowing
   the voice to speak more slowly would let more lines fill wider windows, at the
   cost of pacing naturalness — needs its own listening check.
2. **Pull-forward placement**: start the *next* line a little early inside its
   own window instead of always at `slot_start`, so a short line can occupy the
   tail of the previous gap (a placement change, not a synthesis change).
3. **Isometric translation** (length-aware MT): translate with a target length so
   the Vietnamese matches the slot by construction. Larger effort; deferred.

## Threats to validity

- **No listening study.** All metrics are computed from `timed.json` /
  `script.json` / `timing.json`. Perceived-rate CV is a **proxy** for pacing, not
  a listener test; "sounds better" is not measured.
- **One video, one speaker** (`spk0`, Vietnamese Piper `vais1000`). The result may
  not generalise to other content, languages or voices.
- **envit5 + Piper only.** Rate re-synthesis uses Piper's `length_scale`; ViXTTS
  keeps the atempo/clip path, so these numbers say nothing about ViXTTS. The
  translation is `envit5` on both arms. **The User's original complaint
  measurements (6 clips, perceived rate 2.42–7.75 words/s) came from their
  `hunyuan` job (`53a7f0d6…`); the `hunyuan` path was NOT re-run here** — this
  A/B is the default `envit5` path, and the hunyuan engine's segmentation/output
  differs, so the absolute BEFORE numbers are not the User's numbers.
- **The 9.5 s-scale dead air in the User's job comes from source pauses** between
  segments. No per-synthesis rate change can fill silence the video itself
  contains; only placement (pull-forward) or length-aware translation can.
- **Both arms re-ran `translate,speak,timing`** from a scratch copy, so the
  BEFORE arm is a fresh previous-behaviour run, not the User's cached job; its
  clip/stretch counts (1 / 7) are lower than the User's job (6 / 21).
- **TTS re-synthesis is not bit-deterministic** (Piper + float ops), so a single
  A/B pair has run-to-run noise; the small deltas (rate CV, max) are within that
  noise.
- **Baseline semantics.** `global_length_scale` is the clamped median of
  `window / natural`; because this video's median sits at the 1.10 cap, the
  pace offset pushed some exactly-fitting rows to speak slightly *faster*
  (`length_scale < 1`) rather than filling wide windows.

## Reproduce

```bash
# 1. two full scratch A/B runs (BEFORE = flags off, AFTER = defaults)
./.venv/Scripts/python.exe scripts/eval_timing_fit.py run

# 2. score + write eval-cache/fit1/scores.json + console table
./.venv/Scripts/python.exe scripts/eval_timing_fit.py score

# 3. worst-row side-by-side (EN / BEFORE / AFTER)
./.venv/Scripts/python.exe scripts/eval_timing_fit.py rows
```

Both runs used a scratch copy of the reference job (`ca63d19c…`) under
`.vidub-cache-fit1/`; the User's `.vidub-cache/` and `output/` were never written.

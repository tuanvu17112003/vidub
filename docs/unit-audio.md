# Unit-level audio (UA-1): reference A/B

This is the measurement record for **UA-1**: the timing stage now speaks whole
**sentence units** instead of ASR fragments. When `sentence_units` is on (the
default), the timing stage groups the script rows with the *same*
`group_units` call the translator used, synthesizes **one WAV per
multi-segment unit** from the joined member text, fits that single audio across
the unit's span, and emits **one `timed.json` row per unit** (carrying
`members`). An intra-sentence pause — e.g. the User's `…đào` ⏸ `tạo…` — can no
longer exist, because no placed-row boundary is allowed to fall inside a unit.

## Setup

* **Input video:** `.vidub-cache/downloads/youtube-5sLYAQS9sWQ.mkv` — the User's
  complaint job **`1334fb56…`** (63 ASR rows → 49 sentence units). This is the
  job the silent-gap report was measured on: *50 of 62* placed-row boundaries
  carried > 0.25 s of silence, *40* carried > 0.8 s (worst **6.72 s**), with the
  live mid-word split `…đào || tạo…` at rows 10/11.
* **BEFORE** = `sentence_units=False` (v0.7.0 per-segment timing: one placed row
  per ASR segment).
* **AFTER** = defaults (`sentence_units=True`, unit audio).
* **Engines:** `envit5` + Piper, all QC on, `--no-hunyuan-refine`. The two runs
  hold the translation engine fixed so the A/B isolates the **timing** change;
  the User's original complaint job ran `hunyuan`, so its absolute BEFORE
  numbers (quoted above) are context, not this run's baseline.
* **Isolation:** both runs seed from the reference job then force
  `translate,speak,timing,mix,export`, writing only into the scratch roots
  `.vidub-cache-ua1/` and `eval-cache/ua1/`. The User's `.vidub-cache/` jobs and
  `output/` are read-only.

## Pre-registered acceptance (declared before reading the numbers)

1. **Placed-row boundaries inside a sentence unit = 0** (AFTER) — structural.
2. **Mid-word split count = 0** (a placed boundary that cuts a word in half,
   e.g. `…đào || tạo…`).
3. **Clips → 0.**
4. **Total placed-boundary silent gaps > 0.8 s reduced vs BEFORE** (the User's
   "silent gaps" metric); the > 0.25 s count reported alongside.
5. **Per-unit rate spread reported honestly** (words per placed second, plus the
   unit `length_scale` / recentring summary).

## Results

From `eval-cache/ua1/scores.json` (produced by
`scripts/eval_unit_audio.py score`):

| metric | BEFORE | AFTER |
|---|---|---|
| placed rows | 63 | 49 |
| sentence units | 49 | 49 |
| placed boundaries | 62 | 48 |
| **intra-unit boundaries** (target 0) | **14** | **0** |
| **mid-word splits** (target 0) | **6** | **0** |
| **clips** (target 0) | **0** | **0** |
| gaps > 0.25 s | 57 | 47 |
| **gaps > 0.8 s** (target: reduced) | **45** | **39** |
| worst gap (s) | 6.606 | 10.459 |
| rate (vi words / placed s) median | 4.611 | 4.486 |
| rate min … max | 2.899 … 6.343 | 2.871 … 6.242 |
| max atempo stretch | 1.133 | 1.077 |
| rows with `members` | 0 | 49 |
| TTS-QC MOS mean (NISQA) | 3.88 | 3.99 |
| `quality` verdict | review | review |

Worst placed-boundary gaps, BEFORE:
`6.606 s @ 49.32` (rows 6→7), `6.344 @ 161.77`, `5.612 @ 149.76`,
`4.248 @ 184.467`, `3.902 @ 114.18`.

Worst placed-boundary gaps, AFTER:
`10.459 s @ 82.005` (units 8→9), `7.495 @ 50.209` (units 6→7),
`6.310 @ 161.77`, `5.566 @ 149.76`, `5.281 @ 104.34`.

Unit rate fit (AFTER, `timing.json` `rate_fit`): `resynthesized 40`, `kept 9`,
`failed 0`, `global_length_scale 1.1`, `recentered 23`; per-row `rate_action`
`resynth 40 / keep 9`, and 23 rows were recentred. BEFORE (per segment):
`resynthesized 48 / kept 15`, `recentered 26`.

## Verdict — per pre-registered target

1. **Intra-unit boundaries = 0 → MET.** AFTER is `0` (structural), BEFORE `14`.
   Two consecutive placed rows can never belong to the same sentence unit on the
   AFTER path; the checker verifies this by membership, not by construction.
2. **Mid-word splits = 0 → MET.** AFTER `0`, BEFORE `6` (letter-adjacent
   intra-unit boundaries, the class that contains `…đào || tạo…`).
3. **Clips → 0 → MET.** Both phases `0` on this video (the reference run was
   already clip-free; the target is retained as a guard).
4. **Silent gaps > 0.8 s reduced → MET** (`45 → 39`, **−6**); the > 0.25 s count
   also fell (`57 → 47`, **−10**) and the count of placed boundaries dropped
   `62 → 48`. **Unfavourable, reported verbatim:** the **worst** gap rose
   `6.606 s → 10.459 s` — a genuine inter-sentence pause is now measured between
   two whole-unit rows instead of being partly masked by fragment boundaries, and
   bounded recentring can push the next unit's row later. Overall the "silent
   gap" distribution improves, but the single worst gap is larger.
5. **Per-unit rate spread reported → DONE.** Median `4.611 → 4.486` vi words per
   placed second, range `2.899…6.343 → 2.871…6.242`, max stretch
   `1.133 → 1.077`. The spread is essentially unchanged; joining fragments into
   one utterance did not make the rate distribution worse.

## Mid-word examples

`scripts/eval_unit_audio.py rows` prints both documented examples straight from
the User's own caches (read-only); the AFTER column is the unit text the stage
now synthesizes (deterministic `group_units` + member join).

### `1334fb56…` rows 10/11

| | |
|---|---|
| EN | `…foundation models applied specifically to text and text like things. I'm talking` ‖ `about things like code. Now large language models are trained on large data sets` |
| BEFORE | `vi[10] = "Bây giờ các mô hình ngôn ngữ lớn được đào"` ‖ `vi[11] = "tạo trên các bộ dữ liệu lớn của văn bản như sách"` |
| BEFORE placed gap at the split | **3.595 s** |
| AFTER | one unit `[10, 11, 12]` → `"Bây giờ các mô hình ngôn ngữ lớn được đào tạo trên các bộ dữ liệu lớn của văn bản như sách bài báo và các cuộc hội thoại."` |
| AFTER | contains `đào tạo` contiguously: **True** |

### `14415df2…` rows 50/51

| | |
|---|---|
| EN | `To address this, chatbots undergo another type of training` ‖ `just as important, called reinforcement learning` |
| BEFORE | `vi[50] = "Để giải quyết vấn đề này, chatbot phải trải qua một loại đào"` ‖ `vi[51] = "tạo quan trọng khác, được gọi là học tăng cường với"` |
| BEFORE placed gap at the split | **0.181 s** |
| AFTER | one unit `[50, 51, 52]` → `"Để giải quyết vấn đề này, chatbot phải trải qua một loại đào tạo quan trọng khác, được gọi là học tăng cường với phản hồi của con người."` |
| AFTER | contains `đào tạo` contiguously: **True** |

## Threats to validity

* **One video, one speaker.** Both runs are a single `spk0` clip; the
  structural targets are engine-/video-independent, but the gap and rate numbers
  are not generalizable.
* **Engine fixed at envit5 + Piper.** The User's complaint job used `hunyuan`;
  only the *structural* result (0 intra-unit boundaries / 0 mid-word splits) is
  engine-independent. ViXTTS keeps the atempo path (the per-synthesis
  `length_scale` knob is Piper-specific).
* **Continuous dub speech over micro-pauses is a deliberate trade-off.** A unit
  is one utterance placed across the unit's span, so a real pause that falls
  *inside* a long unit is no longer audible as a break. This is the intended fix
  for intra-sentence pauses; it is a cost for naturally-paced pauses within a
  unit.
* **The worst gap can grow** (see target 4): silence between sentences is still
  silence, and recentring leaves the pre-pause tail empty.
* **`mid_word_splits` is a proxy.** It flags placed boundaries inside a unit
  where the left `vi` ends on a letter and the right begins on a letter; that
  catches `…đào || tạo…` but also flags adjacent unrelated words. It is not a
  lexicon check.
* **No listening study.** `mos_mean` is NISQA-TTS on the fitted clips
  (`3.88 → 3.99`) over different row cardinalities, so it is not a like-for-like
  perceptual comparison.

## Reproduce

```bash
./.venv/Scripts/python.exe scripts/eval_unit_audio.py run
./.venv/Scripts/python.exe scripts/eval_unit_audio.py score
./.venv/Scripts/python.exe scripts/eval_unit_audio.py rows
```

`run` writes `.vidub-cache-ua1/cache-{before,after}/…` and
`.vidub-cache-ua1/work-{before,after}/`; `score` writes
`eval-cache/ua1/scores.json` (and `runs.json`). Neither writes into
`.vidub-cache/` or `output/`.

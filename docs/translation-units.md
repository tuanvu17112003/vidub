# Sentence-level translation units — reference A/B evaluation (TU-1-T4)

> **Superseded in part by SA-1 (v0.9.0).** This page is the **TU-1 historical
> A/B** (v0.6.0) and its measured numbers are kept as-is. The unit **contract it
> documents is no longer current**: TU-1 ended a unit at a silence gap of at
> least `SENTENCE_GAP_S = 0.45` s and let the `MAX_UNIT_WORDS = 60` /
> `MAX_UNIT_SECONDS = 18.0` caps cut at the last "safe" boundary (which could
> still be mid-sentence). The current **SA-1 contract** is described in the
> section [The SA-1 contract (current)](#the-sa-1-contract-current) below and
> measured in [`docs/sentence-locked-units.md`](sentence-locked-units.md).

**Status:** complete — the primary (`envit5`) and optional secondary
(`--mt-engine hunyuan`) pairs are both run, scored and reported below.
**Task:** TU-1-T4 · **Date:** 2026-09-30

## What is being compared

TU-1 makes the translate stage merge ASR fragment runs into **sentence-level
translation units** before the MT call and split each unit translation back onto
its member segments (`vidub/utils/sentences.py`). This document measures that
change against the previous **per-segment** behaviour on the reference video,
applying the pre-registered acceptance criteria verbatim.

| Run | `sentence_units` | Engine input | Meaning |
|---|---|---|---|
| **BEFORE** | `False` (`--no-sentence-units`) | one text per ASR segment | previous behaviour |
| **AFTER** | `True` (default) | one text per sentence unit | the TU-1 change |

## Setup

| Item | Value |
|---|---|
| Input | `.vidub-cache/downloads/youtube-a7TOameRqoY.mkv` (`video_hash = ca63d19c…`) |
| Video | 351.36 s, 1280×720 @ 30 fps, single speaker (`spk0`) |
| Engines | `envit5` (`VietAI/envit5-translation`, the default) — **primary**; `hunyuan` (`hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K` via the local Ollama server) — **optional secondary** |
| QC | default on (transcript / subtitle / translation / TTS / final gate) |
| Refine | **off** (`--no-refine`), so the comparison isolates the unit change |
| Sentence-unit constants | *(superseded — TU-1)* `SENTENCE_GAP_S = 0.45`, `MAX_UNIT_WORDS = 60`, `MAX_UNIT_SECONDS = 18.0` |
| Hardware | NVIDIA RTX 3080 (10240 MiB, driver 591.86); CUDA torch 2.14.0+cu126 |
| Stages run | `translate,speak,timing,mix,export` (ingest/transcribe reused from a scratch copy of the reference job) |
| Scratch caches | `.vidub-cache-tu1/cache-{envit5,hunyuan}-{before,after}/` (+ `work-*` output dirs) |
| Scores | `eval-cache/tu1/scores.json` |

The User's `.vidub-cache/` job (`ca63d19c…`) and `output/youtube-a7TOameRqoY/`
are **read only**; every run copied the reference job into its own scratch cache
dir and ran with the CWD set to a scratch work dir, so its `output/` (and every
other artifact) lands under `.vidub-cache-tu1/`.

## The SA-1 contract (current)

SA-1 (v0.9.0) replaces the TU-1 gap/cap rules above. The current behaviour of
`vidub/utils/sentences.py` and its consumers is:

* **Sentences are the row unit.** `split_into_sentences` splits every ASR row at
  internal sentence boundaries; each sentence's `start`/`end` are **locked to
  the ASR word edges** when the row's `words` line up with its tokens, and
  interpolated by character fraction (clamped inside the row) otherwise.
  `script.json` rows are therefore sentence rows; `segments.json` keeps the ASR
  fragments untouched.
* **Units are SENTENCE-SCOPED.** `group_units` ends a unit **only** at a
  completed sentence (terminal `.` `!` `?` `…`, trailing closers skipped) or a
  **speaker change**. A silence gap never ends a unit, and consecutive complete
  sentences are **not** merged.
* **The hard ceiling is the only size guard.** `HARD_UNIT_WORDS = 150` /
  `HARD_UNIT_SECONDS = 45.0`. When an over-ceiling sentence has no internal
  sentence boundary, the unit closes at the largest internal gap
  ≥ `PAUSE_CUT_GAP_S = 0.6` s (a counted *pause cut*, the last resort); with no
  such gap the unit stays **whole** and is counted `oversized`. No path takes a
  mid-sentence cut, so `unit_cut_stats(...)["mid_sentence_cuts"] == 0`.
* **Completeness guard** (`vidub/utils/units_guard.py`): each unit is checked for
  word-coverage (`UNIT_COVERAGE_MIN = 0.75`) and sentence loss
  (`MIN_WORDS_FOR_SENTENCE_LOSS = 8`, coverage < `1.10`); a flagged unit is
  re-sampled (best-of-`RESAMPLE_K = 3`) and/or given a fallback-engine candidate,
  records the action (`none`/`resampled`/`fallback`/`flagged`) and never raises.
* **Unit-level QC.** The translation-QC battery scores **units** (one
  `translation_qc.json` `segments` entry per unit, with `members`), and the
  per-row display text is re-derived from the corrected unit text.
* **`units.json`** is the authoritative unit map (members, locked span, final
  `vi`, guard/battery status); the timing stage consumes it verbatim.
* **Aligned display.** The timing stage force-aligns each multi-member unit's
  final audio and records `member_spans` + `align`
  (`asr`/`split`/`single`/`unavailable`) in `timed.json`; `--no-align-display`
  skips the alignment ASR and uses the `split` fallback.

The TU-1 numbers below were measured **before** this change and are retained for
history; they describe the superseded rules.

## Metric 1 — unit summary (AFTER)

From `translation_qc.json`'s `units` block (`none` on BEFORE, which has no block):

| Field | BEFORE | AFTER |
|---|---|---|
| `mode` | `segment` (per-segment) | `sentence` |
| `units` | 74 (= segments) | **53** |
| `segments` | 74 | 74 |
| `merged_segments` | 0 | **21** |
| `max_unit_words` | — | 51 |
| `max_unit_seconds` | — | 17.94 |

The reconstructed unit list from `group_units` over the AFTER job's
`segments.json` matches the recorded block exactly (`units_consistent = true`):
53 units, 74 segments. Both caps hold (the largest unit is 51 words / 17.94 s,
just under the 60-word / 18 s limits). Unit grouping happens **before** the
engine call, so the hunyuan pair records the identical block (53 / 74 / 21
merged).

## Metric 2 — mid-sentence input boundaries (primary, pre-registered target = 0)

> **TU-1-era measurement.** The counts below are the v0.6.0 behaviour (gap ends
> and cap cuts could leave mid-sentence boundaries). Under the SA-1 contract
> above there are no gap ends and the hard ceiling is the only size guard; the
> SA-1 A/B measures **0** mid-sentence unit boundaries on the reference video
> (see [`docs/sentence-locked-units.md`](sentence-locked-units.md)).

Count of consecutive engine-input pairs where the **left** text does not end a
sentence (no terminal punctuation) **and** the **right** text starts lowercase or
with a digit.

| | BEFORE | AFTER |
|---|---|---|
| Engine inputs | 74 | 53 |
| **Mid-sentence boundaries** | **29** | **9** |
| — cause: cap (`MAX_UNIT_*` forced) | 28 | 8 |
| — cause: long pause (≥ 0.45 s) | 1 | 1 |
| — cause: speaker change | 0 | 0 |

**Result: the pre-registered target of 0 is NOT MET** (AFTER = 9). The change
removes 20 of the 29 mid-sentence inputs (−69 %), and the 9 that remain are not
ASR-fragment boundaries any more: 8 are **size-cap cuts** (a unit that would
exceed 60 words / 18 s is closed at the previous segment) and 1 is a **silence
gap ≥ `SENTENCE_GAP_S` treated as a sentence boundary** even though the text
does not end a sentence. These are exactly the two documented mid-sentence cut
cases; the primary target can only reach 0 by raising the caps.

The hunyuan pair is **identical** for this metric (the unit list is built from
`segments.json` before the engine runs): 74 → 53 inputs, 29 → 9 boundaries
(28 → 8 cap, 1 → 1 gap).

## Metric 3 — translation QC (`translation_qc.json`)

| | BEFORE | AFTER |
|---|---|---|
| Verdict | `review` | `review` |
| Counts | ok 66, number-error 1, mistranslation-risk 7 | ok 68, mistranslation-risk 5, entity-error 1 |
| Unresolved flags | 8 | **6** |
| Flags / 100 rows | 10.81 | **8.11** |

Rows = 74 in both runs. The verdict is unchanged (`review`), and the unresolved
flag count and flags-per-100-rows both improve (8 → 6, 10.81 → 8.11). The tqc
battery is only a coarse proxy (see threats), so this is supporting evidence,
not proof of better fluency.

**Secondary (`hunyuan`) pair:**

| | BEFORE | AFTER |
|---|---|---|
| Verdict | `review` | `review` |
| Counts | ok 53, mistranslation-risk 20, number-error 1 | ok 52, mistranslation-risk 20, entity-error 2 |
| Unresolved flags | 21 | 22 |
| Flags / 100 rows | 28.38 | 29.73 |

On hunyuan the change **slightly regresses** this metric (21 → 22 unresolved,
28.38 → 29.73 per 100 rows); the verdict stays `review`. The hunyuan engine
already re-translates with neighbouring-segment context, so merging fragments
buys it less — and one extra entity row is flagged on the unit path.

## Metric 4 — subtitles QC (`subtitles_qc.json`) + aggregate quality

| | BEFORE | AFTER |
|---|---|---|
| `cues_in` | 74 | 74 |
| `cues_out` | 136 | **132** |
| `split` | 33 | 32 |
| `violations.cps` | 66 | 70 |
| `violations.duration` | 53 | **40** |
| `violations.line_length` | 11 | 15 |
| `violations.lines` | 0 | 0 |
| `cps_max_observed` | 32.14 | 32.14 |
| `quality.verdict` (report) | `review` | `review` |

Mixed: fewer cues are needed (136 → 132) and the duration violations drop
sharply (53 → 40), while CPS violations and line-length violations rise slightly
(66 → 70 and 11 → 15); the worst-case CPS is identical. The aggregate quality
verdict is unchanged (`review`). Subtitle metrics are mostly a downstream
artefact of the (unchanged) split-back timing, not of unit quality.

**Secondary (`hunyuan`) pair:**

| | BEFORE | AFTER |
|---|---|---|
| `cues_in` | 74 | 74 |
| `cues_out` | 216 | **204** |
| `split` | 50 | 50 |
| `violations.cps` | 161 | **151** |
| `violations.duration` | 104 | **74** |
| `violations.line_length` | 9 | **8** |
| `violations.lines` | 0 | 0 |
| `cps_max_observed` | 35.67 | **34.05** |
| `quality.verdict` (report) | `review` | `review` |

On hunyuan every subtitle metric improves (cues 216 → 204, cps 161 → 151,
duration 104 → 74, line_length 9 → 8, worst-case cps 35.67 → 34.05); the
aggregate verdict is unchanged (`review`). Hunyuan produces longer Vietnamese
(hence more cues overall than envit5), so the absolute numbers are not
comparable across engines — only the BEFORE/AFTER direction is.

## Screenshot fragment run — EN / BEFORE vi / AFTER vi

The three documented screenshot segments (indices 1–3), matched from the two
`script.json` files:

**Segment 1 — EN:** `You can actually afford to look at the Ferrari. Meanwhile, these open-source models are sitting`

- BEFORE: `Bạn có thể đủ khả năng để nhìn vào chiếc Ferrari. Trong khi đó, những mẫu mã nguồn mở này`
- AFTER : `Bạn thực sự có thể đủ khả năng để nhìn vào chiếc Ferrari. Trong khi đó, những mẫu mã nguồn mở này đang`

**Segment 2 — EN:** `there like some cruel joke. Sure, you can download them, you can even name them, but actually running`

- BEFORE: `Chắc chắn, bạn có thể tải chúng, bạn thậm chí có thể đặt tên cho chúng, nhưng thực sự chạy`
- AFTER : `ngồi đó như một trò đùa tàn nhẫn. Chắc chắn, bạn có thể tải chúng, thậm chí có thể đặt tên cho chúng`

**Segment 3 — EN:** `them? That's going to cost you more than a small house. We're talking half a million in hardware`

- BEFORE: `Anh sẽ phải trả giá hơn cả một ngôi nhà nhỏ. Chúng ta đang nói về nửa triệu phần cứng.`
- AFTER : `nhưng thực sự vận hành chúng? Nó sẽ tốn của bạn nhiều hơn một ngôi nhà nhỏ. Chúng ta đang nói về nửa triệu phần cứng`

The AFTER outputs are visibly more complete sentences (e.g. `ngồi đó như một
trò đùa tàn nhẫn.` — "sitting there like some cruel joke" — appears in AFTER and
is absent from the BEFORE fragment), because `envit5` is a sentence-level NMT
model that had been fed bare fragments. The seam is still visible: AFTER segment
1 ends on the fragment `…mở này đang` and segment 3 begins with the tail
`…vận hành chúng?` split off from segment 2 — the split-back seam can land inside
a Vietnamese clause even when the English unit was whole.

**Secondary (`hunyuan`) pair — same segments:**

**Segment 1 — EN:** `You can actually afford to look at the Ferrari. Meanwhile, these open-source models are sitting`

- BEFORE: `Bạn có thể đủ khả năng để nhìn vào chiếc Ferrari. Trong khi đó, những mẫu mã nguồn mở này`
- AFTER : `Bạn hoàn toàn có đủ khả năng chi trả để sở hữu một chiếc Ferrari. Trong khi đó, những mô hình AI mở nguồn này lại`

**Segment 2 — EN:** `there like some cruel joke. Sure, you can download them, you can even name them, but actually running`

- BEFORE: `Chắc chắn, bạn có thể tải chúng, bạn thậm chí có thể đặt tên cho chúng, nhưng thực sự chạy`
- AFTER : `chỉ tồn tại như những trò đùa đáng thương mà thôi. Đúng là bạn có thể tải chúng về, thậm chí đặt tên cho chúng`

**Segment 3 — EN:** `them? That's going to cost you more than a small house. We're talking half a million in hardware`

- BEFORE: `Anh sẽ phải trả giá hơn cả một ngôi nhà nhỏ. Chúng ta đang nói về nửa triệu phần cứng.`
- AFTER : `nhưng việc vận hành chúng thực sự lại tiêu tốn rất nhiều chi phí – có thể lên đến nửa triệu đô la chỉ cho phần cứng.`

Hunyuan shows the same pattern (AFTER recovers `trò đùa đáng thương` — "cruel
joke" — and the `vận hành chúng` tail), with the same seam caveat: AFTER segment
1 ends mid-clause on `…này lại`.

## Pre-registered verdict

Applying the acceptance criteria verbatim (`eval-cache/tu1/scores.json`):

| # | Metric | Target | envit5 (BEFORE → AFTER) | hunyuan (BEFORE → AFTER) | Verdict |
|---|---|---|---|---|---|
| 2 | **Mid-sentence engine inputs** (primary) | **0** | 29 → **9** | 29 → **9** | **TARGET NOT MET** on both — improved 29→9 (−69 %); residual = 8 cap cuts + 1 pause |
| 1 | Unit summary | AFTER records a `units` block | `segment` → 53 units / 74 seg / 21 merged | `segment` → 53 / 74 / 21 | **MET** (informational; engine-independent) |
| 3 | Translation QC unresolved / flags per 100 | lower is better | 8 → **6**; 10.81 → **8.11** | 21 → **22**; 28.38 → **29.73** | **MIXED** — improved on envit5, slight **regression** on hunyuan (verdict `review` both) |
| 4 | Subtitle QC | lower is better | cues 136→132, cps 66→70, duration 53→40, line_len 11→15, cps_max 32.14→32.14 | cues 216→204, cps 161→151, duration 104→74, line_len 9→8, cps_max 35.67→34.05 | **MIXED** — envit5 mixed/neutral, hunyuan improved (quality verdict `review` both) |

Overall: the **primary pre-registered objective (0 mid-sentence boundaries at the
engine input) is NOT met on either engine** — envit5 and hunyuan both leave 9.
The residual is structural, not engine-specific: 8 of the 9 are **cap-forced**
(`MAX_UNIT_WORDS = 60` / `MAX_UNIT_SECONDS = 18.0`) and 1 is a ≥ 0.45 s pause
treated as a sentence boundary. Raising `MAX_UNIT_WORDS` / `MAX_UNIT_SECONDS`
(both in `vidub/utils/sentences.py`) is the lever if the User wants fewer
mid-sentence inputs; it trades against dubbing fit for very long units.
Translation QC improves on envit5 (the sentence-level NMT model the change
targets) but slightly **regresses** on hunyuan (which already sees neighbour
context); subtitle QC is mixed on envit5 and improved on hunyuan. Adoption rests
on the 69 % reduction in fragment inputs and the envit5 QC improvement, with the
cap interaction and the hunyuan non-gain stated plainly.

## Threats to validity

- **The tqc battery is a weak proxy** for sentence-level fluency: it scores
  numbers / entities / terminology / length / consistency, not coherence, so the
  8→6 change is a weak signal and is not a substitute for reading the output.
- **No human evaluation.** The qualitative reading of the three screenshot rows
  is one sample; no listener/reader study was run.
- **One video, one speaker, one domain** (an English tech explainer). The result
  may not generalise to multi-speaker, noisy, or non-tech content; the speaker
  change/gap cut paths were not exercised here (0 speaker-change boundaries).
- **`envit5` is a sentence-level NMT model**, so feeding it whole sentences is
  exactly the situation it is designed for; the observed benefit is plausibly an
  upper bound. An LLM/contextual engine (ollama/hunyuan) may respond differently.
- **The secondary hunyuan pair does not improve the translation battery** (21 →
  22 unresolved; 28.38 → 29.73 per 100 rows). Hunyuan already receives
  neighbour-segment context, so sentence units are closer to redundant for it,
  and it scores worse overall than envit5 on this battery — the hunyuan numbers
  are a different operating point, not a like-for-like baseline.
- **The split-back seam can still land inside a Vietnamese clause** — the AFTER
  segment 1/2/3 example shows a clause starting mid-sentence and finishing in the
  next segment. Sentence units reduce fragment inputs; they do not guarantee the
  Vietnamese output is split at Vietnamese clause boundaries.
- **Cap-forced cuts remain mid-sentence.** The primary target of 0 cannot be met
  while `MAX_UNIT_*` caps exist and a single safe-cut-free run exceeds them; the
  8 remaining cap cuts are the inevitable consequence of the documented fallback.
- **Single run per configuration** (deterministic engines, but the refine pass
  was off and ASR output was reused, so run-to-run variance is not characterised).

## Reproduce

```bash
./.venv/Scripts/python.exe scripts/eval_sentence_units.py run            # envit5 pair
./.venv/Scripts/python.exe scripts/eval_sentence_units.py run --mt-engine hunyuan
./.venv/Scripts/python.exe scripts/eval_sentence_units.py score          # eval-cache/tu1/scores.json
./.venv/Scripts/python.exe scripts/eval_sentence_units.py rows --mt-engine envit5
./.venv/Scripts/python.exe scripts/eval_sentence_units.py rows --mt-engine hunyuan
```

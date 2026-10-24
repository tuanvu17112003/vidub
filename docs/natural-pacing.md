# Natural pacing and translation safety — 2026-10-01

This correction addresses reproducible defects in the existing pipeline. It
does not establish that Vietnamese semantic accuracy or listener preference
has improved across videos.

## Research and implications

* [Dubbing in Practice](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00551/115968/Dubbing-in-Practice-A-Large-Scale-Study-of-Human)
  finds that character isometry is not a reliable proxy for speech isochrony,
  and human dubbing does not simply vary speaking rate to meet timing demands.
* [VideoDubber](https://arxiv.org/abs/2211.16934) accounts for speech duration
  rather than controlling only word or character counts.
* [IsoChronoMeter](https://aclanthology.org/2024.wmt-1.29/) measures isochrony
  through TTS duration predictors. For this local project, measured Piper
  audio is available and avoids adding another model.
* [Tencent HY-MT's official templates](https://github.com/Tencent-Hunyuan/HY-MT)
  provide distinct direct, terminology and contextual translation prompts.
  Those capabilities do not establish that an arbitrary length instruction
  preserves meaning; candidates still need quality checks.

The engineering inference is to preserve a fitting sentence's natural pace,
measure whole-sentence audio, and accept a shorter translation only after
checking content and its actual spoken duration. ISO-1 remains off by default.

## Implemented corrections

1. `plan_rate_fit` requests a faster synthesis only for overflow. A wide
   availability window cannot slow the voice. The stage no longer computes
   a global pace offset from fragmented display-row lengths.
2. Single-member units use their unit window. Multi-member units are
   synthesized at natural pace, measured, and fitted with the same policy.
   They may require a second synthesis when they overflow.
3. Concise timing retries pass both the existing engine validators and the
   completeness guard. They are accepted only if measured audio is shorter
   than the current take and fits within the window's stretch cap. Legacy
   filler retries must improve duration too. Rejection preserves the old take.
4. Accepted timing translations persist as `timed.json` `vi`; alignment,
   sentence-unit subtitles and subsequent TTS-QC retries use that wording.
5. Numeric validation now compares complete tokens and their multiplicity.
   `16` cannot pass as `160`, nor separate `1` and `6` as `16`. Dot/comma
   substitutions are allowed for localized numeric punctuation. The check
   cannot detect swapped numbers associated with different entities, changed
   units, signs, or every numeric-format ambiguity.
6. MT signatures version the LLM validator and timing signatures version the
   new rate policy so existing non-empty signatures invalidate cached stages.

## Matched real-Piper experiment

`scripts/eval_natural_pace.py` synthesizes the same **83 cached Vietnamese
sentence translations**, using the same Piper voice and availability windows
in both arms. The legacy arm applies the prior ratio/clamped-median formula
at sentence granularity; the new arm preserves fitting natural takes. This is
an isolated rate-policy experiment, **not a replay of the full historical
pipeline** (which planned from display fragments and included MT retries).

No translation, ASR, separation, mixing, or new model download occurs. Source
cache files are read only. Audio and raw scores are in
`eval-cache/natural-pace-20261001/`.

| Measured criterion | Legacy policy | Natural-pace policy | Verdict |
|---|---:|---:|---|
| Slowed sentence takes | 45 | 0 | MET: eliminate slowing to fill silence |
| Synthesis speed coefficient of variation | 0.1211 | 0.0655 | MET: lower variation, down 45.9% |
| Resynthesized takes | 66 | 12 | MET: fewer redundant syntheses |
| Audio exceeding the window (>20 ms) | 19 | 20 | NOT MET: no reduction in overflow |
| Audio exceeding the window's stretch cap (>20 ms) | 1 | 1 | NOT MET: remove every clip risk |

These speed factors concern synthesis only; downstream `atempo` can further
accelerate audio. They are not measured articulation rates or listener scores.
The remaining overflow is handled by the existing bounded stretch/trim policy.

## VERIFY

```bash
./.venv/Scripts/python.exe -m pytest -q --tb=short
HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe scripts/eval_natural_pace.py \
  .vidub-cache-iso/run-off-timing/fbae346ba03869da6f62f85cd97eeb82c9b5d964/job.json \
  --output eval-cache/natural-pace-reproduction
```

The output directory must be new; the experiment refuses an existing one.
The HF offline setting requires the project's Piper voice to be cached.
Regression tests cover omitted content, worsening audio duration, unit rate
fitting, subtitle wording and numeric-token errors. Full-suite verification:
**MET — 862 passed, 4 skipped in 135.74 s**. The full test output is in
`eval-cache/pace-full-tests.txt`; no new dependencies were installed.

## Remaining investigation

Translation quality still needs bilingual review against the source transcript.
The current word-count completeness heuristic cannot establish semantic
equivalence, and it can reject valid concise Vietnamese. Prioritize ASR errors,
terminology, negation, entities, and number-to-entity associations in the review.
Test a bounded set of validated translation candidates ranked by measured
natural-pace duration on several videos; evaluate fidelity and listener pacing
alongside clips, total trimmed seconds and the combined TTS/atempo speed.
Keep the reported ISO-1 negative result and evaluate these changes separately.

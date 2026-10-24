# Subtitle scheduling and placed audio QC — 2026-10-01

## Changes

The active subtitle export policy now partitions text for two-line layout and
maximum duration, then uses dynamic programming to schedule chunk boundaries.
It borrows adjacent silence within **0.25 s early / 1.5 s late** of each speech
reference. Chunks stay chronological, non-overlapping and within those bounds;
all words and punctuation survive. ASR word edges supply references when the
timeline has the same word count; otherwise character-proportional references
are estimates. Bounded movement is recorded, rather than claiming every final
boundary is an exact word edge.

The profile remains **17 CPS, 5/6–7 seconds, two lines, 42 characters per line**.
High CPS alone no longer causes recursive splitting. Scheduling uses SRT
millisecond precision; QC ignores only floating-point noise at thresholds.
Impossible reading windows remain flagged. No subtitle paraphrasing or speech
speed change is used to satisfy the caption metric.

Final audio QC selects `timed.json` placed units rather than original script
fragments. It checks both the speech WAV and the mixed-track placement, including
missing, unreadable, empty and short speech files, silence and peak saturation.
Missing timed units remain review issues. Legacy caches lacking timed data use
an explicitly labeled script fallback. Pending TTS MOS warnings now participate
in the aggregate verdict.

`pacing.combined` records `stretch / length_scale`, speed variation and onset
shift, including older cached timing records. Export signatures invalidate only
export when its profile/policy changes. Legacy empty signatures retain their
existing behavior: use `--force export` once to upgrade. The export helper now
honors the destination recorded in `manifest.paths.output_dir`.

## Translation follow-up

The opt-in AI glossary supports an explicit third column such as
`aliases=đào tạo trước|tiền đào tạo`. These replacements apply only when the
corresponding English term occurs in the source. They preserve the surrounding
Vietnamese sentence, and survive entity/other corrective retries. Plural
acronyms such as `GPUs` canonicalize to `GPU` during entity checks. Resolved
machine-generated translation review packets are cleared; unrelated notes stay.
TTS QC also replaces its generated packets on each run. In the completed
rerenders, superseded TTS packets were cleared only after SHA-256 comparison
proved their copied WAVs differed from the currently fitted speech files.

The previously rejected terminology case now passes. Two exact-source reference
sentences repair the awkward closing sentence and the isolated `song song`
phrase. These are transparent editorial examples in the chosen profile, not a
claim that envit5 has learned to translate unseen sentences better.

## Matched fixed-text/audio experiment

The comparison uses each export's original unit text and original placed audio.
It reconstructs the previous caption policy, including its strict floating-point
comparisons. Original job caches are not modified by this audit.

| Criterion | LPZh9BOjkQs | 5sLYAQS9sWQ | Status |
|---|---:|---:|---|
| Total subtitle violations | 203 → 35 | 186 → 9 | MET — at least 50% reduction |
| CPS violations | 160 → 35 | 117 → 9 | Improved; remaining windows infeasible |
| Duration violations | 35 → 0 | 68 → 0 | MET |
| Line-length violations | 8 → 0 | 1 → 0 | MET |
| Original words/punctuation preserved | Yes | Yes | MET |
| Caption overlaps | 0 | 0 | MET |
| Lead/lag bounds honored | Yes | Yes | MET |
| False script-window silence warnings | 3 → 0 | 2 → 0 | MET |
| Missing/empty speech still detected | Synthetic regression | Synthetic regression | MET |

The newer baseline's combined closing-line speed is **1.199×**, despite
`atempo` alone being about 1.073×. It is now reported directly. Scheduling
displacement bounds reference speech/chunk estimates; they do not establish
lip synchronization or prove every estimated word timestamp correct.

Artifacts: `eval-cache/subtitle-schedule.json`, plus SRT/ASS and `quality.json`
under `output/phase2-review/<video>/`.

## Full rerenders

Both videos are regenerated through translate → speak → timing → mix → export
using the installed envit5/Piper models and the explicit AI glossary. Original
ASR and separated stems are reused. All QC layers remain enabled. Results are
stored under `output/phase2-improved/<video>/`; these are a changed-text/audio
experiment, separate from the matched subtitle comparison above.

The first attempt exposed a destination bug: export ignored the explicit
destination and used the default folders. The helper is corrected and covered
by a regression test. Baseline videos/audio/captions were rebuilt from the
unchanged original caches and passed full decoding. Baseline reports were
reconstructed with current QC and are labeled accordingly; they are not exact
byte copies of the previous reports. The final rerender checks its destination
before writing.

Machine-readable final metrics are in `eval-cache/phase2-rerender/results.json`.
The reports retain remaining round-trip, subtitle and TTS MOS warnings. Clearing
the known terminology/wording defects does not establish universal translation
accuracy or bilingual/listener approval.

| Final rerender criterion | LPZh9BOjkQs | 5sLYAQS9sWQ | Status |
|---|---:|---:|---|
| Retained grammar leaks | 0 | 0 | MET |
| Missing configured terminology | 0 | 0 | MET |
| Missing numeric tokens (strict check) | 0 | 0 | MET |
| Two configured reference sentences | Both exact | Not present in this video | MET for the affected video |
| Subtitle CPS violations | 35 | 13 | NOT MET for a zero-violation target; still reported |
| Duration/layout violations | 0 | 0 | MET |
| Missing/empty/short/silent/clipped speech | 0 | 0 | MET |
| Timing clips | 0 | 0 | MET |
| Full MP4 decoding | Exit 0 | Exit 0 | MET |
| Unresolved TTS MOS flags | 1 | 3 | NOT MET for a zero-flag target; included in final review |

Translation QC still records two lexical round-trip risks on each video, plus
five low-confidence/unnatural flags on the newer video. Those are retained
rather than claiming all translation quality is solved. The reference closing
sentence can still trigger lexical round-trip risk despite its corrected
wording. TTS MOS is a model proxy, not a listener rating.

## VERIFY

Final full suite: **889 passed, 4 skipped** in 159.34 seconds. Five known grammar
cases again produced zero leaks; all eight sampled terminology repairs passed.
Both original baselines reconstructed during recovery and both final isolated
videos passed complete ffmpeg decoding. The final rerender had all QC enabled.
After the final stale-TTS-packet cleanup change, focused timing/TTS/final-QC
verification passed **111 tests, 1 skipped** (33.56 s); compileall also passed.

```bash
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe scripts/eval_translation_grammar_safety.py
./.venv/Scripts/python.exe scripts/eval_subtitle_schedule.py
./.venv/Scripts/python.exe scripts/rerender_phase2.py
./.venv/Scripts/python.exe scripts/summarize_phase2.py
```

The rerender copies job data into `eval-cache/phase2-rerender`, uses the explicit
destination above and fully decodes each final MP4 with ffmpeg. No dependencies
or new models are installed. The separate recovery script documents the
one-time restoration; it is not part of normal VERIFY or rerendering.

# Vietnamese quality rework benchmark

**Protocol status:** frozen for the first En-to-Vi text benchmark; no candidate is promoted by this report. The current run is provisional because it has no held-out video, no verified source transcript, and no completed human judgments.

## Scope and frozen review rules

This benchmark compares sentence-level English-to-Vietnamese ASR and translation on fixed, hashed source audio. Keep ASR, translation, subtitle shaping, speech synthesis, and audio mixing as separate experiments. Candidate outputs never overwrite existing jobs or production defaults.

Before judging translation, verify the English transcript against the audio and attach corrections with `quality_benchmark.py apply-source`. When an output row has `input_status=source_asr_unverified`, it is exploratory and excluded from promotion scoring. Reviewers must not infer missing or uncertain source words from a translation candidate.

For translation review, inspect each anonymized candidate against the verified source. Use `pass`, `fail`, or `needs-review` for the decision; `needs-review` is unresolved. Record severity as `none`, `minor`, `major`, or `critical` for every `pass` or `fail`; missing or invalid severity leaves the judgment unresolved. Choose an error type such as `accuracy`, `omission`, `addition`, `terminology`, `fluency`, or `register`. A critical error includes a changed negation, number, named entity, or other fact that changes the scene's meaning. Leave uncertain cases unresolved and add a note.

Fill `difficulty_reason` only when the source has a concrete challenge: a proper name, number, negation, technical term, idiom, disfluency/overlap, very short window, or multiple speakers. At least 30 distinct cases need a reviewer-supplied reason. Keep one complete video as holdout; select it before candidate scores are inspected. Do not tune prompts, glossary, or thresholds on holdout outputs.

The private candidate key is saved beside the review TSV as `<stem>.key.json`. Keep it away from reviewers until annotations are complete. Reviewer identity is recorded in the TSV. Blank decisions count as unreviewed, never as a pass.

## Frozen corpus

Prepared corpus: `eval-cache/quality-rework/baseline-20261024/cases.jsonl`.

| Video hash | Cases | Split | Audio SHA-256 |
| --- | ---: | --- | --- |
| `1a372a6d345535b258048bedfe488989ac45fed8` | 58 | development | `fbedabbb8edd1afeb6d3e55a565c7a667b3b7579096d21a653da0555733dc1d0` |
| `e53bb23d1989ba4d70a3871d412f01327bfaa74e` | 54 | development | `9dd598cfbe0859b5b88b4cf48070386b31a1f35df5822bf4cc9cd46574eeae66` |
| **Total** | **112** | development only | |

Audio files are referenced by absolute path in each case. Each carries `audio_sha256`; the ASR runner verifies each unique audio file once before inference. Case IDs are `{video_hash}:unit:{index}` and are the only join key. Source job trees were hash-checked before and after corpus preparation; no source artifact changed.

This corpus has no holdout video and no `verified_source` corrections. The two existing videos remain development data; do not retroactively relabel either as holdout after examining candidate outputs.

## Candidate provenance and current run

| Candidate | State | Provenance |
| --- | --- | --- |
| Existing job translation | Available | `baseline_vi` from the frozen jobs; generated from the existing source transcript and glossary handling. |
| Envit5 | Ran on 112 cases | `mt-envit5-sourcehash.jsonl`, checkpoint `VietAI/envit5-translation` revision `840bc88104d5a4277af740eaedb024df8c3093e7`; 68.59 s total, observed GPU peak 2,858 MiB. Every row includes the source-text SHA-256 and is marked `source_asr_unverified`. |
| Blinded review packet | Created | `review-verified.tsv` compares `baseline_vi` with Envit5; `review-verified.key.json` privately binds each randomized label to the exact engine, source hash, and translation text hash. All judgment fields are blank. |
| Comparison report | Created | `comparison-verified.json` is provisional: 112 development cases, no holdout, no verified source text, 224 unreviewed candidate judgments, and zero difficult cases tagged. It reports judgments by engine and split. |
| Faster-Whisper | Not run | Neither base nor tiny checkpoint is cached locally. The runner refused a network download and wrote no candidate output. |
| Qwen3-ASR | Not run | `.eval-venv-qwen3` and the checkpoint are absent. The runner isolates this model and refuses uncached downloads. |
| Hy-MT2 | Not run | Local Ollama server and Hy-MT2 GGUF are unavailable. The runner refused a model pull and wrote no candidate output. |

The current source-hashed Envit5 output and run metadata are beside the case corpus. Earlier `mt-current-baseline.jsonl`, `mt-envit5.jsonl`, `review.tsv`, `review.key.json`, `comparison.json`, `review-sourcehash.tsv`, `review-sourcehash.key.json`, `comparison-sourcehash.json`, `review-final.tsv`, `review-final.key.json`, and `comparison-final.json` are retained as pre-fix artifacts; do not use them for scoring because they predate exact candidate-text/source binding, complete run-provenance gating, and severity-completeness checks. Use `mt-current-baseline-sourcehash.jsonl` and `mt-envit5-sourcehash.jsonl` with `review-verified.tsv` and its matching key. The baseline checkpoint revision was not preserved in the original job, so its provenance remains incomplete and the comparison stays provisional. Run metadata records model revision, package versions, command, runtime, observed GPU peak, and exact prompt-template hash for Hy-MT2. The numeric runtime/GPU readings are operational measurements, not translation-quality scores.

Hy-MT2 uses the official English-language prompt and terminology form from [Tencent's Hy-MT2 README](https://github.com/Tencent-Hunyuan/Hy-MT2/blob/main/README.md). Terminology is filtered to source terms found in each sentence. The run sidecar stores the SHA-256 of the exact prompt templates, so the prompt provenance does not depend on the mutable README branch. Every MT result stores a SHA-256 of the exact source text it received; review-sheet generation and scoring reject stale or mismatched candidate inputs. The ASR candidate shortlist and licensing research are summarized in [dubbing improvement research](dubbing-improvement-research.md). No Qwen3 forced aligner is used for Vietnamese timing.

## Versioned text selections and preview

New translations write `spoken_vi` and `subtitle_vi` on sentence units and script rows, initially equal to the accepted `vi`. Old artifacts and blank optional fields continue to resolve through `vi`. Sentence-unit speech uses the unit's spoken selection and splits it back over its member audio rows; subtitles use the unit subtitle selection. Resume signatures make spoken edits rerun speak through export and subtitle-only edits rerun export. Signature fields stay empty on legacy jobs until all consuming stages have completed successfully.

The contract-v1 reference preview is isolated at `output/quality-rework-preview/LPZh9BOjkQs-20261024-contract-v1/dubbed.vi.mp4`; its SRT is `subtitles.vi.srt`, with provenance in `preview-provenance.json`. `spoken_vi` and the copied dub track come from the existing phase2 rerender, while `subtitle_vi` comes from the manually edited `preview_vi` candidates. No new TTS synthesis ran. The audio SHA-256 is `77e2cbb8509ffbb1860dbfacfbffd979f703aaa426222d7efc6454779522c9d3`; the subtitle SHA-256 is `f9a0e339f86678abf783928d8f09c732d0138acdc3405d53503181b5927b04d1`.

The new local voice preview is `output/quality-rework-preview/LPZh9BOjkQs-20261024-voxcpm2-v4/dubbed.vi.mp4`, with model and render details in `voxcpm2-provenance.json`. The short voice sample is `output/quality-rework-preview/voxcpm2-sample-20261024.wav`. VoxCPM2 was installed in the isolated `eval-cache/quality-rework/voxcpm2-env`; weights are in `eval-cache/quality-rework/voxcpm2-models`. The tested run used `voxcpm==2.0.3`, the workspace CUDA PyTorch, `torchaudio==2.11.0`, 48 kHz output, `cfg_value=2.0`, and 10 inference steps. The current OpenBMB deployment guide lists Vietnamese support, Apache-2.0 licensing, 2B parameters, and about 8 GB inference VRAM; see [VoxCPM2 deployment guide](https://openbmb.cn/docs/en/guides/voxcpm/voxcpm-2). The installed PyPI API does not accept the `seed` argument shown in the current guide, so this preview seeds PyTorch before each unit instead.

A separate cold-start measurement on the same RTX 3080 produced a 5.44-second sample in 4.21 seconds after a 22.47-second model load (RTF 0.77); process peak was 5.28 GiB allocated and 5.46 GiB reserved, including model load and warm-up. This is one sample, not a sustained-throughput or cross-device benchmark.

For timing fit, 8 `spoken_vi` unit texts were shortened in the isolated preview only. Three `subtitle_vi` unit texts were shortened for the v4 preview; two more were shortened in v5 after its new placements exposed one remaining CPS violation. The v5 subtitle QC reports 100 cues and zero CPS, duration, line-length, or line-count violations. Timing reports zero clipped segments; all 58 placed audio rows are present, readable, non-empty, non-silent, and placed. Loudness is -16.5 LUFS. The 477.481-second MP4 has AV1 video, AAC audio, and a soft `mov_text` subtitle track; a full FFmpeg decode completed successfully.

The silence investigation compared the source audio/vocal stem with placed TTS intervals. The source vocal stem contains silence from 419.22–425.70 s and 456.90 s to the end; these correspond to v4 output gaps at 421.00–425.24 s and 459.56–477.48 s. They are present in the source track. Across all units, source spans total 417.62 s while v4 placed TTS totals 402.83 s, leaving about 14.79 s of additional pauses spread across the clip. The source-paced comparison at `output/quality-rework-preview/LPZh9BOjkQs-20261024-voxcpm2-v5-source-paced/dubbed.vi.mp4` disables recentering and gently extends 26 short takes by at most 10%; it adds 13.06 s of voice without overlap or clipping, bringing total placed voice to 415.89 s. This preserves the long source silences while reducing the extra inter-sentence gaps.

The user reviewed the v5 source-paced output and said the voice quality and spacing are good. VoxCPM2 is now an opt-in application engine; Piper remains the default. This listening approval is not a blinded A/B benchmark and does not promote a new default. The v5 artifact gate remains `review` because the existing translation QC verdict is `review`; transcript QC and subtitle QC pass. The wider text benchmark remains provisional because it has no verified source transcript, held-out video, or completed bilingual judgments.

## UI text and TTS options

The local UI TTS dropdown includes `voxcpm2`. Its `voxcpm==2.0.3` model runs in a separate Python worker so its dependencies and model stay out of the default ViDub process. The worker reuses one model for a speech stage, then closes it. `VIDUB_VOXCPM2_PYTHON` and `VIDUB_VOXCPM2_MODEL_CACHE` override the interpreter and local checkpoint cache.

After a run, use **Review/edit spoken and subtitle text** to load sentence-unit text, edit only `spoken_vi` and `subtitle_vi`, save, then click **Run**. A spoken edit invalidates speech and downstream stages; a caption-only edit invalidates export. Legacy jobs without `units.json` use script rows. These UI edits are preserved in the job cache and do not rewrite an earlier preview.

## Commands

```powershell
.\.venv\Scripts\python.exe scripts/quality_benchmark.py prepare `
  --job eval-cache/phase2-rerender/1a372a6d345535b258048bedfe488989ac45fed8/job.json `
  --job eval-cache/phase2-rerender/e53bb23d1989ba4d70a3871d412f01327bfaa74e/job.json `
  --out eval-cache/quality-rework/<run>/cases.jsonl

.\.venv\Scripts\python.exe scripts/quality_benchmark.py apply-source `
  --cases eval-cache/quality-rework/<run>/cases.jsonl `
  --corrections <reviewed-source.tsv> `
  --out eval-cache/quality-rework/<run>/cases-verified.jsonl

.\.venv\Scripts\python.exe scripts/eval_asr_rework.py `
  --cases eval-cache/quality-rework/<run>/cases-verified.jsonl `
  --engine faster-whisper --model base `
  --out eval-cache/quality-rework/<run>/asr-faster-whisper.jsonl

.\.venv\Scripts\python.exe scripts/eval_mt_rework.py `
  --cases eval-cache/quality-rework/<run>/cases-verified.jsonl `
  --engine envit5 --model VietAI/envit5-translation `
  --out eval-cache/quality-rework/<run>/mt-envit5.jsonl

@'
import json
from pathlib import Path
from scripts.quality_benchmark import make_review_sheet

root = Path('eval-cache/quality-rework/<run>')
cases = [json.loads(line) for line in (root / 'cases-verified.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
paths = [root / 'mt-envit5.jsonl', root / 'mt-hy-mt2.jsonl']
candidates = [[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] for path in paths]
make_review_sheet(cases, candidates, seed=20261024, output=root / 'review.tsv')
'@ | .\.venv\Scripts\python.exe -

.\.venv\Scripts\python.exe scripts/quality_benchmark.py score `
  --cases eval-cache/quality-rework/<run>/cases-verified.jsonl `
  --candidate eval-cache/quality-rework/<run>/mt-envit5.jsonl `
  --candidate eval-cache/quality-rework/<run>/mt-hy-mt2.jsonl `
  --judgments eval-cache/quality-rework/<run>/review.tsv `
  --out eval-cache/quality-rework/<run>/comparison.json
```

Generate a new review sheet and private key for the exact candidate files passed to `score`; never reuse a key for a different candidate set. Candidate and report paths are create-only. `score` reads run metadata from each candidate's `<stem>.run.json` sidecar and unblinds review labels with the matching private key. Per-engine results are reported separately for development and holdout.

For the recorded run, score `mt-current-baseline-sourcehash.jsonl` with `mt-envit5-sourcehash.jsonl`, using `review-verified.tsv` and writing `comparison-verified.json`. The generic command above shows a future Envit5/Hy-MT2 experiment; generate its new review sheet after both candidate files exist.

## Metrics and promotion gate

For ASR, report WER/CER separately by development and holdout using only `verified_source`. `jiwer` is loaded lazily when available; otherwise the report uses a normalized standard-library edit-distance implementation and names that implementation. Vietnamese WER is whitespace-token based, so it is a syllable proxy rather than word segmentation. Report sample counts beside every rate.

For translation, report human decision, severity, error-type, difficult-case, and unresolved counts separately per engine and split. Every candidate row must carry the SHA-256 of the exact source text it received; scoring rejects mismatched inputs. ASR outputs must carry the exact frozen audio SHA-256 for each case. Candidate JSONL files require a `<stem>.run.json` sidecar whose engine matches the rows; the report preserves model/checkpoint provenance and promotion eligibility requires both fields. Do not score unverified source rows as translation wins. Completed judgments and difficult-case tags count only when attached to a reviewer, a completed pass/fail decision, and a valid severity. Runtime and peak GPU memory are supporting operational facts only. Do not combine these with a single weighted score.

The report stays `provisional` unless it has a held-out video, at least 30 reviewer-tagged difficult cases, verified source text for the holdout, completed judgments with reviewer identity and valid severity, complete candidate model/checkpoint provenance, and no unresolved critical errors. This benchmark does not encode a candidate win threshold: freeze that threshold after the human-reviewed baseline and before looking at candidate results. `promotion_eligible` only means the data is complete enough for a promotion review; `promoted` remains false and this run does not alter defaults.

Codex can review bilingual text and inspect objective audio signals/transcription read-back, but this host cannot directly hear local audio. Naturalness, emotion, speaker resemblance, residual dialogue, and subjective mix preference remain unreviewed until a listener assesses the preview. Never claim perceptual improvement from this benchmark alone.

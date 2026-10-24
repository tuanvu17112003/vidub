# Vietnamese Subtitle and Dubbing Quality Rework

**Status:** approved by user  
**Date:** 2026-10-24  
**Scope:** ViDub's local video-to-Vietnamese pipeline

## Intent

Rework ViDub for the best demonstrable Vietnamese subtitle and dubbing quality on the user's hardware. Quality means accurate source transcription, faithful and natural Vietnamese, readable captions, intelligible and natural speech, sound timing, and a clean final mix. A machine-readable `pass` is useful evidence but is not a substitute for bilingual and listening review.

The first benchmark targets the English-language technology videos already used by the project. A separate held-out video must test whether a gain generalizes. The architecture remains language-aware so this first benchmark does not silently degrade other source languages.

## Goals and acceptance

1. Establish a reproducible, human-reviewed baseline before promoting any model. Keep the existing two videos, at least 30 difficult source units, and at least one held-out video. Include clear and difficult speech, proper names, numbers, negation, technical terms, short windows, and multiple speakers where available. Record source hashes, model/checkpoint versions, prompts, runtime, peak VRAM, and selected/rejected candidates.
2. Measure ASR against corrected source transcripts: normalized WER/CER, named-entity and numeric errors, and timestamp error on a manually checked sample. Measure translation using blinded bilingual error annotations (accuracy, omission, addition, terminology, fluency, and severity). Use deterministic checks and optional reference-free scores for triage, never as sole proof of fidelity.
3. Produce a validated Vietnamese translation before optimizing either its caption length or spoken duration. Preserve every source clause and critical fact. Make unresolved cases visible for review.
4. Derive separate `spoken_vi` and `subtitle_vi` selections from that validated translation. Caption edits may shorten phrasing only after meaning checks. Keep the 17 CPS, 42-character, two-line, 5/6-to-7-second project profile and bounded timing policy as measured constraints. Flag infeasible cues instead of silently dropping content.
5. Select TTS and mixing policies by matched listening tests on the same fixed Vietnamese text. Measure intelligibility, pronunciation, naturalness, speaker similarity where cloning applies, output failures, final clip duration, effective playback speed, onset displacement, residual source voice, and background damage.
6. On the held-out set, improve human judgment versus the current default without increasing critical factual errors. Pre-register the exact promotion thresholds after the baseline annotation and before comparing candidate outputs. Preserve no-bleed, no-overlap, and no-missing-audio invariants.

## Constraints

- Local operation and the current commercially usable default path remain. Non-commercial or proprietary checkpoints may be evaluated separately but cannot become the default without an explicit change in product requirements.
- The RTX 3080 has 10 GB VRAM. Run heavy models sequentially. Record actual peak VRAM and elapsed time rather than relying on model-card estimates.
- Keep the CLI, subtitles-only mode, dub mode, stage resume, existing output formats, and existing human review access usable during the transition.
- Preserve today's review behavior: unresolved translation or infeasible-caption issues may still export an inspectable artifact with a `review` verdict and review packet. Structural failures and missing required media still fail. A `review` output is not counted as quality-approved or eligible for automatic default promotion.
- Limit the first candidate-promotion decision to English-to-Vietnamese. Keep other source-language paths and defaults unchanged until each has a separately reviewed corpus. Run smoke regressions for supported modes and languages to detect accidental contract breakage.
- Existing cache artifacts and manually reviewed edits must not be silently overwritten. Version the new artifact contract; invalidate only outputs whose inputs changed. Old jobs remain readable or receive an explicit migration path.
- Keep new model dependencies out of the default environment until a matched benchmark justifies adoption. Candidate runs can use isolated environments.

## Non-goals and anti-goals

No real-time streaming, lip-sync generation, cloud dependency, or new target language is required by this rework. Avoid a single opaque speech-to-speech model that prevents inspecting and correcting transcript, translation, captions, and speech independently. Avoid accepting a shorter translation solely because it fits the audio. Avoid treating TTS MOS, back-translation, word-count ratio, or an ASR self-check as human approval. Do not raise subtitle reading-speed limits to hide infeasible captions.

## Approaches considered

| Approach | Benefit | Cost and failure mode | Decision |
| --- | --- | --- | --- |
| Swap ASR, MT, and TTS models in the current pipeline | Fastest to wire | Leaves one-text-for-both-outputs, weak semantic proof, and caption capacity problems | Reject as the main rework |
| Replace the pipeline with end-to-end speech translation | Potentially simpler flow | Less local control, harder review and caching, uncertain Vietnamese quality and licensing | Reject for this project |
| Rework selection and evidence around the current stages | Reuses stage artifacts, engine interfaces, and QC while fixing the actual quality bottlenecks | Requires a versioned text contract and matched evaluation | **Selected by the user** |

## Proposed flow

1. **Source evidence:** Retain the source audio and `segments.json`. Compare the current faster-whisper output against Qwen3-ASR-1.7B on the same clips; consider PhoWhisper only for Vietnamese-source clips. Select or manually correct a transcript before translation. Preserve ASR alternatives and uncertainty, but do not merge model outputs by vote without hearing the audio. The Qwen3 forced aligner does not support Vietnamese, so it is not the target-speech alignment solution.
2. **Canonical translation:** Keep sentence-scoped units and the current `BaseTranslator` contract. Run envit5 as baseline and Hy-MT2 1.8B/7B as isolated challengers. Give each candidate the same verified source, limited context, and relevant terminology. Reuse deterministic guards for omission, contamination, numbers, and glossary terms. Human review decides semantic failures and validates any automatic candidate-ranking method.
3. **Two output texts:** Add explicit `spoken_vi` and `subtitle_vi` selections with provenance and review state. Preserve the existing `vi` field during migration for old consumers. Speak/timing use `spoken_vi`; subtitle scheduling/export use `subtitle_vi`. A spoken-text change invalidates synthesis and downstream artifacts. A subtitle-only edit invalidates subtitle QC/export, not synthesized audio. The subtitles-only mode works from the same canonical translation without requiring TTS.
4. **Subtitle shaping:** Reuse `pysubs2` and the current bounded scheduler. For infeasible cues, propose concise caption candidates while retaining all factual content. Validate against the source and canonical Vietnamese translation; retain the original and mark review when no faithful candidate fits. Continue to measure duration, CPS, line length, overlap, and timing displacement.
5. **Speech shaping:** Keep natural-pace whole-sentence synthesis. Compare Piper and opt-in ViXTTS with VieNeu-TTS v3 Turbo, VoxCPM2, and a lightweight Vietnamese voice such as KorvaTTS. Apply pronunciation normalization only to the TTS input, not the displayed text. Validate numbers, acronyms, and English technical terms before using a normalizer. Candidate selection accounts for human preference and measured spoken duration; it never clips or deletes content merely to meet a window.
6. **Audio finish:** Compare Demucs against a compatible `audio-separator` model on actual dialogue clips, measuring residual source voice and damage to music/effects. Try FFmpeg sidechain compression for scene-aware bed ducking after the voice is selected. Keep the current loudness and final-file checks.

## Evaluation and promotion rules

Use fixed input hashes and matched text/audio wherever only one component is being compared. Run ASR, translation, captions, TTS, and separation as distinct experiments. At least two bilingual reviewers should annotate the translation comparison where practical; at least two Vietnamese listeners should hear randomized, loudness-matched TTS and mix samples. Record disagreements and resolve critical errors by listening to the source.

Codex performs bilingual text review and technical audio triage from the available files, transcripts, objective audio measurements, and ASR read-back. This Codex session cannot receive audio for direct perceptual listening, so its assessment cannot establish naturalness, emotion, or speaker resemblance. The project owner or a designated human listener supplies the final perceptual ratings and resolves disputed audio cases. Until that happens, the team may build corpus tools and run opt-in experiments, but must not claim listener-quality improvement or promote a TTS/mix default. Review assignments and decisions are recorded with the benchmark artifacts.

The first release gate is a reviewed baseline and a frozen evaluation protocol. Later gates require a candidate to beat that baseline on the held-out material for the quality dimension it changes, with no critical-content regression or broken structural invariant. Report confidence and sample size; a small positive proxy-score change alone cannot promote a model. Keep the prior negative ISO-1 and HY-MT1.5 results as controls, not assumptions about newer systems.

Useful evaluation tools include `jiwer` for ASR WER/CER and an Apache-2.0 COMET-QE checkpoint for optional offline MT ranking. NISQA-TTS remains a clip triage signal. None is a standalone release gate.

## Delivery order

1. Baseline corpus, annotation guide, blinded comparison harness, and recorded promotion thresholds.
2. ASR comparison and transcript correction path; translation comparison on the same corrected transcript.
3. Versioned `spoken_vi` / `subtitle_vi` contract and caption-specific improvement on the existing 48 over-CPS cues.
4. Matched Vietnamese TTS and pronunciation benchmark; integrate only its winner as an opt-in engine first.
5. Separation and ducking A/B; final held-out video review; promote proven defaults and document failures.

The builder may choose file layout, test harness details, and exact checkpoint quantizations. Model downloads, dependency pins, thresholds, and defaults follow measured feasibility and the frozen evaluation protocol rather than this document's untested predictions.

## Evidence and source research

- Current pipeline and evaluation: `vidub/pipeline.py`, `vidub/stages/translate.py`, `vidub/stages/timing.py`, `vidub/stages/export.py`, `eval-cache/phase2-rerender/results.json`, `docs/subtitle-scheduling-and-placed-qc.md`, `docs/hunyuan-evaluation.md`, and `docs/isometric-budgeting.md`.
- Candidate projects: https://github.com/QwenLM/Qwen3-ASR ; https://github.com/Tencent-Hunyuan/Hy-MT2 ; https://github.com/pnnbao97/VieNeu-TTS ; https://github.com/OpenBMB/VoxCPM ; https://huggingface.co/danhcao/KorvaTTS ; https://github.com/nomadkaraoke/python-audio-separator .
- Quality and license checks: https://partnerhelp.netflixstudios.com/hc/en-us/articles/220447048-Vietnamese-Timed-Text-Style-Guide ; https://aclanthology.org/2021.tacl-1.87/ ; https://aclanthology.org/2024.wmt-1.29/ ; https://github.com/Unbabel/COMET/blob/master/LICENSE.models.md .

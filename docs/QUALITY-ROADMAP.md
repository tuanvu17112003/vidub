# QUALITY-ROADMAP — ViDub end-to-end output quality: audit, QC architecture, and improvement plan (v2)

**Status:** MA design review v3 — v2 approved and **implemented as M4 (v0.4.0, 2026-09-29)**. §7 added 2026-09-29 and **implemented as M5 (v0.5.0, 2026-09-29)**: Hunyuan-MT translation integration (opt-in HY-MT1.5-7B engine + refine pass + pre-registered benchmark) — Gate G1 passed, all M5 tasks (T1, T2 + C1, T3, T4) passed Gate G2, and the §7.6 promotion rule was applied verbatim (HY-MT1.5-7B NOT promoted; the refine pass is the measured win).
**Author:** Management Agent, per `WorkFlowMultiAgent/Rule.md` (research → analyze → consult → only then build).
**Date:** 2026-09-29

**What changed from v1:** Source Transcript QC is now an explicit quality-control layer with correction/re-ASR/review paths (not just ASR confidence checks). The QC design is organized around an actionable **Detect → Classify → Correct/Retry/Review → Verify → Accept** lifecycle instead of measurement-only reporting. Translation QC is a **multi-signal battery aligned with the MQM error typology** — semantic preservation, omissions/additions, terminology, entities, numbers, consistency, naturalness — with round-trip translation demoted to an auxiliary signal and generalized to **Source → VI → Source** (no EN→VI→EN hard-coding). Context-aware translation is elevated from an optional extra to the default quality-critical design point. An explicit **engine-selection principle** governs where deterministic rules, small task models, existing MT/ASR models, and LLM reasoning are each appropriate. P0/P1/P2 priorities are preserved.

---

## 1. Audit — what quality control exists today

Verdict up front: ViDub's quality control is **strong on structural correctness** (no audio bleed, no overlap, guards, resume safety, per-stage artifacts) and **near-zero on content quality**. There is no stage whose job is to ask "is this transcript/translation/clip *good*?", no error classification, and no corrective path beyond one re-translation retry. The workflow's G2 gates verify *code*; content quality is checked only by the User listening after the fact.

### Level 1 — Source transcript / subtitles

| Exists | Missing |
|---|---|
| Whisper large-v3 (GPU) / PhoWhisper, Silero VAD, word timestamps | Whisper's **own confidence signals are discarded** — `avg_logprob`, `compression_ratio`, `no_speech_prob` are returned per segment and never inspected |
| Language-mismatch guard (M2-C1) | No hallucination/repetition detector, no cross-check, no re-ASR path when a segment looks unreliable |
| Cue-end tightening from word timing | Subtitles follow raw ASR chunk boundaries — no sentence-aware splitting, no reading-speed/line-length standards |
| — | No punctuation/casing restoration where Whisper under-punctuates |
| — | No transcript **semantic** checks: garbled sentences, missing context, entity/number sanity |

### Level 2 — Vietnamese translation

| Exists | Missing |
|---|---|
| envit5 (beam + repetition guard) | **Per-segment isolation** — every segment is translated blind; cross-video context, terminology continuity, and discourse flow are invisible |
| Two-sided glossary + plural tolerance (M3-C3) | **No quality estimation or error classification** — `low_confidence` is one weak heuristic |
| vnstyle word-order pass + hesitation cleanup (M3-C2) | No checks for omissions/additions, number/entity preservation, or cross-segment consistency |
| Length-aware re-translation retry (M3-C1) | Truncation (2.2× rule) is a **silent content loss** |

### Level 3 — Dubbed output (TTS, timing, audio, subtitles, video)

| Exists | Missing |
|---|---|
| Timing-fit v2: availability windows, 1.20 cap, smoothing, re-translation retry, smart trim at silences | TTS **prosody ceiling** (Piper flat; User rated "Average"); no expressive control |
| Pronunciation lexicon, TTS-only (M3-C3) | **No per-clip TTS QC** — a garbled/silent/clipped synthesis flows into the mix unnoticed |
| Demucs bed + loudnorm + zero-bleed + tail pad | No ducking/level matching per scene; no subtitle styling |
| report.json counters per stage | **No final-video QC gate** (loudness/sync/silence verification of the finished file) |

## 2. Research findings — reuse before reinvent

### Translation quality
- **VideoLingo** translates in a **three-step LLM loop (initial → reflection → improvement)** over the whole transcript and segments subtitles with NLP/LLM "Netflix-style" cutting ([Huanshere/VideoLingo](https://github.com/Huanshere/VideoLingo)) — the same **translate → reflect → improve** pattern as Andrew Ng's agentic translation workflow.
- **MQM (Multidimensional Quality Metrics)** is the industry error taxonomy for translation QC: a hierarchical typology under **Accuracy (Addition, Omission, Mistranslation), Fluency, Terminology, Style, Locale Convention, Verity**, with severity weights ([MQM Council](https://www.themqm.org), [error typology](https://www.themqm.org/mqm-pillars/typology)). ViDub's QC battery should classify errors with an MQM-aligned subset instead of inventing one.
- **Reference-free QE models**: [Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) (source + translation → 0–1) via [`unbabel-comet`](https://github.com/Unbabel/COMET) — **HF-gated, non-commercial** (opt-in only). [TransQuest](https://github.com/TharinduDR/TransQuest) (Apache-2.0, reference-free, XLM-R based) is a fully open alternative — **verify accuracy on En→Vi at build time before adoption**.
- **Isometric MT** is the research line for our length-fitting problem: [Isometric MT for Automatic Dubbing (Amazon)](https://arxiv.org/abs/2112.08682) + [code](https://github.com/amazon-science/isometric-slt), an RL variant ([arXiv 2403.15469](https://arxiv.org/html/2403.15469v1)), and **LLM length-control prompting** ([arXiv 2506.04855](https://arxiv.org/pdf/2506.04855)) — the principled version of our filler-stripping retry.

### Transcript quality
- faster-whisper exposes three per-segment confidence signals; the consensus hallucination filter is **`compression_ratio > 2.4`**, **`avg_logprob < -1.0`**, **`no_speech_prob > 0.6`** ([Deepgram's Whisper explainer](https://deepgram.com), [OpenAI community thread](https://community.openai.com/t/whisper-hallucination-how-to-recognize-and-solve/168410)).
- **Post-ASR correction is an active research area** — NVIDIA's **GenSEC-LLM challenge** ([GERBar baselines](https://github.com/hyprbound/GERBar)) defines LLM post-ASR correction from **N-best hypotheses**; [Whispering-LLaMA](https://github.com/Srijith-rkr/Whispering-LLaMA) (EMNLP 2023) and zero-shot GER with N-best prompting ([Ma et al.](https://arxiv.org/abs/2307.04172)) show the pattern. Key design point for us: **faster-whisper can emit N-best hypotheses** (`beam_size > 1` + `return_hypotheses`), so flagged segments can be corrected *with the alternatives the ASR already considered*, not just re-translated blind.
- **Punctuation/casing restoration**: [`oliverguhr/fullstop-punctuation-multilang-large`](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large); published work on **Vietnamese joint capitalization+punctuation restoration** for ASR output exists.
- [**stable-ts**](https://github.com/jianfch/stable-ts) improves Whisper timestamp/segmentation stability — candidate to reuse instead of hand-rolling better segmentation.

### Subtitles
- Netflix's Timed Text Style Guide gives **measurable** QC: reading speed ≤ **20 CPS** adult English (≈17 CPS many non-English), duration **5/6 s min / 7 s max**, ~**42 chars/line**, 2 lines ([English guide](https://partnerhelp.netflixstudios.com/hc/en-us/articles/217350977-English-USA-Timed-Text-Style-Guide), [general requirements](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617-Timed-Text-Style-Guide-General-Requirements)).

### TTS / prosody / per-clip QC
- **Non-intrusive speech-quality scoring can be applied to every synthesized clip**: [NISQA](https://github.com/gabrielmittag/NISQA) ships **NISQA-TTS weights specifically for TTS-naturalness** (MOS + noisiness/coloration/discontinuity), with a [TorchMetrics wrapper](https://lightning.ai/docs/torchmetrics/stable/audio/non_intrusive_speech_quality_assessment); Microsoft's **DNSMOS** (SIG/BAK/OVRL) is the other standard. This is the missing per-clip TTS QC signal — no reference audio needed.
- Expressive/prosody control in open TTS is moving to **style/instruction prompting**: [Parler-TTS](https://github.com/huggingface/parler-tts) (style via text prompt), **VoxCPM-0.5B** (tokenizer-free, SOTA-claiming zero-shot). Piper = fast but flat; ViXTTS = cloning without fine control. Vietnamese support + license audit required before adopting any new engine.

### Multi-language / source-agnostic
- envit5 is **En↔Vi only**. For Source → VI → Source checks on non-English sources, **MADLAD-400** (Apache-2.0, 400+ languages, 3B/7B/10B) is the open generalist MT candidate; **NLLB-200** is capable but CC-BY-NC (opt-in only). ASR side: whisper large-v3 covers ~100 languages; per-language specialist profiles (like PhoWhisper for vi) should be a data table, not code.

## 3. QC architecture — Detect → Classify → Correct/Retry/Review → Verify → Accept

Every QC signal feeds the same five-step lifecycle. The pipeline is already **stage-resumable** (`--force <stage>` re-runs exactly the affected stages), so a corrective action is *actionable by construction*: each detection maps to (a) a classified cause, (b) a corrective action that re-runs specific stages/segments with a *changed input or setting*, (c) a verification re-check of only the affected segments, and (d) an accept verdict. Signals that cannot be auto-corrected land in a **human review packet** (flagged segment + audio snippet + both texts) rather than producing a warning nobody reads.

### 3.1 Engine-selection principle (not every QC problem is an LLM problem)

| QC task | Deterministic rules | Small task model | Existing MT/ASR model | LLM reasoning |
|---|---|---|---|---|
| Hallucination/garble detection | ✅ confidence thresholds, repetition stats | — | — | — |
| Language identification / mismatch | ✅ script/lexicon stats | ✅ fastText-style LID if needed | ✅ Whisper language probs | — |
| Punctuation/casing restoration | — | ✅ fullstop multilingual / CPR models | — | fallback only |
| Sentence segmentation for subtitles | ✅ rules on word timestamps | ✅ spaCy / stable-ts | — | ✅ only for stubborn cases |
| Omission/addition/number/entity checks | ✅ regex + token coverage | — | — | — |
| Terminology consistency | ✅ glossary cross-segment match | — | — | ✅ conflict resolution |
| Mistranslation / naturalness / register | — | ✅ QE model (TransQuest/cometkiwi) | ✅ round-trip aux signal | ✅ reflect pass (primary) |
| Length fitting | ✅ atempo/window policy | — | ✅ isometric prompting in the MT/LLM | ✅ rewrite pass |
| Transcript error correction (GER) | — | — | ✅ N-best from the ASR itself | ✅ LLM correction over N-best (GenSEC pattern) |

Principle: **deterministic first, task models second, existing models third, LLM last — and the LLM only ever *proposes*; deterministic validators always check its output** (numbers, glossary, length, language).

### 3.2 Level 1 QC — Source transcript (explicit layer)

**Detect** (per segment, computed in the transcribe stage and stored in the manifest):
- ASR confidence: `compression_ratio`, `avg_logprob`, `no_speech_prob` (faster-whisper); PhoWhisper path: output-length/entropy proxies + the existing language guard.
- Cross-hypothesis disagreement: keep N-best for flagged regions; disagreement between hypotheses ⇒ uncertainty.
- Structural: empty/overlong segments, gap overlaps, word-timestamp monotonicity.
- Semantic coherence (cheap, language-agnostic first): immediate repetition rate, punctuation density, digit/entity density anomalies, language-ID consistency with `src_lang` (existing guard generalized per language profile).
- Cross-ASR agreement (opt-in, targeted): re-run only flagged segments with a second model (e.g., PhoWhisper for vi, whisper medium for others) and measure similarity — disagreement ⇒ low trust.

**Classify** (MQM-style for the transcript): `hallucination` (repetition/loop) · `garbled` (low confidence/low coherence) · `language-mismatch` · `segmentation-issue` (runs-on, overlong) · `ok`.

**Correct / Retry / Review:**
- `hallucination`/`garbled` → re-ASR that segment window only, with a changed setting (alternate model size/specialist per language profile, different beam/temperature, LLM correction over the kept N-best per the GenSEC pattern); re-classify the new hypothesis.
- `segmentation-issue` → sentence-aware re-split using word timestamps (spaCy/stable-ts), never text edits.
- Still failing after one retry → **review packet** (segment + audio snippet + hypotheses) and the job's final verdict becomes `review`.

**Verify:** re-run the same detectors on the corrected segment; the flag must clear or escalate.
**Accept:** per-job transcript verdict `pass` / `review` / `fail` (fail = flagged ratio over threshold ⇒ pipeline exits non-zero with machine-readable reasons).

### 3.3 Level 2 QC — Vietnamese translation (multi-signal battery, MQM-aligned)

**Detect** — each segment is scored on seven independent signals (no single score decides):

| Signal (MQM dimension) | Detector |
|---|---|
| Accuracy — omission/addition | content-word coverage + length ratio vs source (per-language rate table) |
| Accuracy — numbers | deterministic: every number in the source must appear in the translation (regex, locale-aware) |
| Accuracy — entities | deterministic: glossed/capitalized entities preserved or mapped via glossary |
| Terminology | glossary enforcement status (already two-sided, plural-tolerant) |
| Consistency | same source term/phrase → same Vietnamese rendering across segments (cross-segment check) |
| Fluency/naturalness | vnstyle + hesitation stats; optional QE model score (TransQuest open / cometkiwi opt-in) |
| Fidelity aux | **Source → VI → Source** round-trip with a *source-language-capable* MT (envit5 for En; MADLAD-400 for others) — auxiliary only, never the decision signal |

**Classify** into MQM-aligned buckets: `omission` / `addition` / `mistranslation-risk` / `number-error` / `entity-error` / `terminology-error` / `inconsistent` / `unnatural` / `ok`.

**Correct / Retry / Review:**
- `number-error`/`entity-error` → deterministic splice (the term/number is inserted; surroundings re-translated) — no LLM needed.
- `inconsistent` → normalize to the majority rendering for that term.
- `unnatural`/`mistranslation-risk` → re-translation retry with a changed strategy (existing length-aware path, then LLM reflect pass when configured); still failing → review packet.
- Every corrective action re-runs only the affected segments and downstream stages via the existing resumable machinery.

**Verify:** re-score only corrected segments; counters in the report.
**Accept:** translation verdict `pass` / `review` / `fail` (e.g., fail if any `number-error` or `omission` remains uncorrected).

**Context handling (elevated):** isolated-segment translation is treated as a systemic quality risk, not a footnote. The quality-critical path is **contextual translation** — a sliding/document window (neighbor segments + running terminology + speaker/turn info) so pronouns, tense, and terminology stay coherent; per-segment MT remains the fallback when no contextual engine is configured. The QC battery's *consistency* signal is what proves context worked.

### 3.4 Level 3 QC — Dubbed output

**Detect:** per-clip TTS quality via **NISQA-TTS** (naturalness MOS, no reference needed) + structural checks (near-silence, clipping, duration sanity vs the fit plan); final-file checks: loudness vs target, A/V duration match, silence-gap map in speech regions, subtitle CPS/duration/line-length per the Netflix guide (Vietnamese profile), zero-bleed re-verified on the real files.

**Classify:** `tts-low-mos` / `tts-silent` / `tts-clipped` / `loudness-off` / `subtitle-cps` / `sync` / `ok`.

**Correct / Retry / Review:** `tts-low-mos`/`tts-silent` → re-synthesize that clip with changed generation settings (ViXTTS temperature/top-k; Piper fallback voice) — one retry, then review packet; `loudness-off` → re-run mix; `subtitle-cps` → re-split/re-translate shorter (existing path).

**Verify:** re-score only the affected clips.
**Accept:** final verdict `pass` / `review` / `fail` in report.json; `fail` exits non-zero with reasons; `review` ships the human-review packet.

## 4. Gap analysis (v1 table preserved, updated references)

| # | Gap | Levels hit | Improvement that closes it |
|---|---|---|---|
| G1 | ASR confidence signals unused; no re-ASR path | 1 | §3.2 detect→classify→re-ASR→review |
| G2 | Translation has no QE, no context, no error classification | 2 | §3.3 battery + contextual translation + optional QE models |
| G3 | Subtitles ignore measurable standards | 1 | sentence-aware splitting + CPS/duration checks (Netflix thresholds) |
| G4 | TTS prosody ceiling + no per-clip QC | 3 | NISQA-TTS per-clip gate + expressive-engine evaluation (Parler-TTS/VoxCPM) |
| G5 | Truncation = silent content loss | 2→3 | isometric-style length budgeting in translation (§3.3) + smart trim as last resort (already) |
| G6 | No automated content QC gate in the workflow | all | §3 lifecycle + final verdict in report.json; corrective actions re-run stages |

## 5. Improvement plan (P0/P1/P2 preserved and enriched)

### P0 — QC framework + measurement layer (no new heavy deps; the M4 baseline)
1. **Level 1 QC (§3.2):** confidence-signal capture + classification + segment-scoped re-ASR (N-best kept for flagged regions) + review packet + transcript verdict.
2. **Subtitle standards QC (G3):** sentence-aware re-splitting on word timestamps (spaCy/stable-ts candidate) + CPS/duration/line-length checks with a Vietnamese profile (≈17 CPS, ≤42 chars/line) + violations flagged/re-split.
3. **Level 2 QC battery (§3.3):** numbers/entities/omission-addition/consistency detectors (deterministic) + round-trip aux signal for En; corrective splices; translation verdict.
4. **Final QC gate (G6):** automated export checks + `quality` verdict (`pass`/`review`/`fail`) in report.json; `fail` exits non-zero; review packet generation.

### P1 — Contextual and expressive layer (needs the M4 LLM engine / new models)
5. **Contextual translate→reflect→improve (G2, §3.3):** sliding/document window + running terminology + per-segment length budgets (isometric prompting); LLM only *proposes* — deterministic validators (numbers/entities/glossary/length/language) always check the output; envit5 per-segment stays the fallback. Requires Ollama + a local model on this machine.
6. **Punctuation/casing restoration** (opt-in small model; Vietnamese CPR work exists) before segmentation/translation.
7. **Per-clip TTS QC (NISQA-TTS/DNSMOS)** + retry with changed generation settings; **expressive engine evaluation** (Parler-TTS / VoxCPM-0.5B) with a Vietnamese-support and license audit before any adoption.
8. **TransQuest QE** (Apache-2.0) as an additional fluency signal — validate En→Vi accuracy at build time.

### P2 — Opt-in polish
9. **cometkiwi QE** (HF-gated, non-commercial) behind `--allow-nc-models`.
10. **Per-scene ducking** of the bed; subtitle styling beyond soft `mov_text`.
11. **GER fine-tune exploration** (GenSEC/GERBar pattern) only if the N-best + LLM correction proves insufficient.

## 6. Decisions requested from the User (Gate G1 for M4)

| # | Decision | MA recommendation |
|---|---|---|
| D1 | Adopt the QC framework (§3) + P0 items 1–4 | **Yes** — measurement + actionable recovery, no new heavy deps, covers all three levels |
| D2 | Install **Ollama + a local model** and build the contextual translate→reflect→improve pass (P1.5) | **Yes** — the single biggest translation-quality lever; deterministic validators gate its output |
| D3 | Punctuation/casing restoration opt-in (P1.6) | **Yes** — small model, clear win for MT input + subtitles |
| D4 | Evaluate an expressive TTS engine (Parler-TTS / VoxCPM-0.5B) as a third engine (P1.7) | **Evaluate first** — Vietnamese support + license audit before adoption; Piper/ViXTTS stay default |
| D5 | NISQA-TTS per-clip QC (P1.7, part of) | **Yes** — small model, closes the "bad clip reaches the mix" gap |
| D6 | TransQuest (open) / cometkiwi (NC) QE models (P1.8 / P2.9) | **TransQuest yes (validate first); cometkiwi defer** unless the User accepts NC tooling |

**Gate G1:** the roadmap proceeds to TASK_SPECs only on the User's explicit approval of the M4 scope (which of D1–D6 to greenlight).

---

## 7. M5 — Hunyuan-MT integration (design 2026-09-29; **built as M5 / v0.5.0, 2026-09-29**)

Directive from the User: evaluate and integrate **Tencent Hunyuan-MT-Chimera-7B** into the runtime pipeline, keeping the quality flow `Source Transcript → Transcript QC → Verified Transcript → Translation → Translation QC/Refinement → TTS → Final Dub QC → Final Dub`, the **Detect → Correct/Refine → Verify → Accept** lifecycle, deterministic checks wherever they suffice, and sensible fallbacks. **No code is written before Gate G1 approval of this section.** The pipeline shape does not change — only the capability of the Translation and Translation-QC/Refinement box increases.

**Outcome (2026-09-29):** Gate G1 passed; D1–D5 built as M5-T1..T4 (+T2-C1) with Gate G2 PASS on every task. Measured result in `docs/hunyuan-evaluation.md` (raw numbers `eval-cache/hunyuan/scores.json`): HY-MT1.5-7B **NOT promoted** (TransQuest +0.0059 < +0.02; battery 41.2 > 37.6 flags/100; 3.57 > 3.0 s/row), **refine pass adopted as an opt-in win** (−16 % flags: 37.6 → 31.6 per 100 rows), **Chimera fusion NOT adopted** (19 vs 2 unresolved packets). All D1–D5 deliverables shipped in v0.5.0.

### 7.1 Research findings (all facts verified against primary sources 2026-09-29)

**The model family** (Hugging Face `tencent/Hunyuan-MT-7B`, `tencent/Hunyuan-MT-Chimera-7B`, `tencent/HY-MT1.5-7B`; tech reports [arXiv 2509.05209](https://arxiv.org/abs/2509.05209) and [arXiv 2512.24092](https://arxiv.org/abs/2512.24092); [GitHub Tencent-Hunyuan/Hy-MT](https://github.com/Tencent-Hunyuan/Hy-MT)):

| Model | Released | Role | Key facts |
|---|---|---|---|
| Hunyuan-MT-7B | 2025-09-01 | direct En↔Vi↔… translation specialist | WMT25 champion; 33 languages incl. **Vietnamese**; no system prompt |
| Hunyuan-MT-Chimera-7B | 2025-09-01 | **ensemble-fusion refiner** | input = source + up to **6 candidate translations** → one refined translation; documented fusion prompt; **not positioned as a direct translator** |
| **HY-MT1.5-7B / 1.8B** | 2025-12-30 | upgraded direct translators | aggregates **beat Hunyuan-MT-7B** (XCOMET-XXL EN⇔XX 0.9093 vs 0.9065; XX⇔XX 0.8098 vs 0.7828; WMT25 0.6159 vs 0.6023); ≈95% of Gemini-3.0-Pro on Flores-200; natively trained for **terminology intervention, context-aware translation, format preservation** |

- **Vietnamese** is in the 33-language list of every family member. **No per-pair En→Vi scores are published** (papers report aggregate groups: ZH⇔XX / EN⇔XX / XX⇔XX) — a local En→Vi benchmark is therefore mandatory before any adoption claim (§7.6).
- **Architecture:** custom `hunyuan_v1_dense` (`HunYuanDenseV1ForCausalLM`, own tokenizer, 128 256 vocab, 32 768 context) — **not** Qwen2-compatible. Needs transformers ≥ 4.56.0; the `translation` pipeline class was removed in transformers v5 (our `.venv` runs v5) → the transformers-direct path is avoided by design.
- **Official usage:** plain-text prompt, **no system prompt**, recommended params `{"top_k": 20, "top_p": 0.6, "repetition_penalty": 1.05, "temperature": 0.7}`, `max_tokens` 2048. Non-Chinese template: `Translate the following segment into {target_language}, without additional explanation.\n\n{source_text}`. Documented contextual and terminology-intervention templates exist (GitHub Hy-MT README; languages named in full, e.g. "Vietnamese"). Chimera fusion prompt: `Analyze the following multiple {target_language} translations of the {source_language} segment surrounded in triple backticks and generate a single refined {target_language} translation. Only output the refined translation, do not explain.` + source in backticks + numbered candidate list.
- **Deployment (this machine):** official GGUF repo `tencent/HY-MT1.5-7B-GGUF` — Q4_K_M 4.62 GB / **Q6_K 6.16 GB** / Q8_0 7.98 GB, runnable as `ollama run hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K` or via llama.cpp (`llama serve -hf …`). Q6_K fits fully in the 10 GB RTX 3080; 74 GB free on H:. Chimera-7B has **community GGUFs only** (mradermacher; no official one). No official `ollama.com/library` entry exists for any family member — the `hf.co/` pull path uses official weights. Tech-report quantization data: FP8 ≈ lossless; Int4 degrades noticeably (XCOMET ZH⇔XX 0.8361 → 0.8060) → benchmark Q6_K (primary) vs Q4_K_M (secondary); Q4 is the VRAM-contingency when two models are loaded.
- **vLLM ≥ 0.10.0** + a specific transformers branch also works (OpenAI-compatible server) but is a heavy install — not the primary path.
- **Existing integrations:** nothing ViDub-reusable on GitHub beyond the runtimes above; integration is via our own engine class following the proven `ollama_mt.py` stdlib-urllib pattern.
- **License:** Tencent HY Community License (2025) — commercial use allowed **< 100 M MAU**, royalty-free; **territory: EU, UK, South Korea excluded**; Acceptable Use Policy; **outputs may not be used to improve other AI models**; Notice-file + license flow-down obligations for distribution. Consequence: **not** an `--allow-nc-models` model (it is commercially usable), but it is a community license with conditions — it ships as an **opt-in engine, never the default**, and `vidub doctor` prints the license name. Users download the weights themselves (ViDub bundles no weights), so the tool's Apache-2.0/MIT default path is unaffected.

### 7.2 Role assignment (the User's candidate roles, decided)

| Candidate role | Decision | Rationale |
|---|---|---|
| Primary translation generation | **New opt-in engine `--mt-engine hunyuan` = HY-MT1.5-7B** (Q6_K GGUF via Ollama) | specialist MT with WMT25-level aggregate quality; official GGUF; native contextual + terminology modes; envit5 stays default until the §7.6 benchmark passes |
| Contextual translation | **Official contextual template** (previous/next segment as context block), same contract as `translate_contextual` | trained capability of 1.5 (context-aware translation) |
| Alternative generation / comparison | **K samples** (temp 0.7) of the hunyuan engine, generated **only for rows the deterministic battery flags** | Detect→Correct, not generate-K-always (cost) |
| Translation refinement | **Chimera-7B fusion** over the candidate set — opt-in `--fusion-model`; without it, deterministic best-of-K selection | fusion is Chimera's designed purpose; nothing else in the OSS ecosystem fuses candidates; adopt as default only if the benchmark proves it |
| Semantic consistency checking | **NO Hunyuan.** Stays deterministic (`tqc.py` battery) + aux round-trip; optional per-row TransQuest score (validated M4-T8, AUC 0.772) may gate refinement | a generator is not a judge; engine-selection principle §3.1 |
| Terminology / entity consistency | **Deterministic two-sided glossary stays the gate**; Hunyuan additionally gets the terminology-intervention prompt as input conditioning | validators always gate LLM output |
| Detecting translation errors before TTS | **`tqc.py` unchanged** (deterministic detectors + verdict policy per decision 10); Hunyuan only ever *responds* to its flags | Detect stays deterministic and auditable |
| Improving the final translation used by TTS | refine-pass output must pass the deterministic validators **and strictly reduce** the row's tqc flags, else the original text is kept | Verify→Accept; no regression possible |
| Transcript QC (before translation) | **UNCHANGED.** Deterministic signals + window re-ASR + review packets are sufficient: the release E2E transcript verdict was `pass` with zero unresolved hallucinations, and Hunyuan is a translation-locked specialist (off-label for transcript repair). Revisit only if review packets show a high false-flag rate. | evidence + §3.1 principle |
| Final Dub QC (after TTS) | **UNCHANGED.** NISQA per-clip, ebur128 loudness, silence/clip checks, subtitle profile, zero-bleed — all deterministic | measurables are deterministic; §3.4 stands |

### 7.3 Architecture changes (proposed code shape)

**New module `vidub/engines/mt/hunyuan_mt.py`** — `HunyuanMTTranslator(BaseTranslator)`, sibling of `OllamaTranslator`:

- Talks to the **Ollama `/api/chat`** endpoint with stdlib `urllib` (same host as the existing engine by default); no system prompt; the official template is the user message; options `{"temperature": 0.7, "top_k": 20, "top_p": 0.6, "repeat_penalty": 1.05}`. Plain-text output parsing (no JSON mode — the specialist was not trained for JSON choreography).
- `translate()` = one official-template call per segment (the specialist prompt is per-segment; ~1–2 s/segment at Q6_K on the 3080). `translate_contextual()` = same call with the contextual block (PREVIOUS/NEXT source text) and the terminology-intervention block built from the loaded glossary (this also fixes the dormant-glossary debt for this engine).
- Deterministic validators after every output — `_lang_ok` / `_numbers_ok` / `_length_ok` are **extracted from `ollama_mt.py` into a shared `vidub/engines/mt/validators.py`** (both engines import them; no logic change). On failure: exactly one targeted re-ask at temperature 0.3, then the **envit5 per-segment fallback** (unchanged, preserved). Unreachable host → `TranslationError` (same contract as the ollama engine; `doctor` pre-checks).
- `stats` counters like the ollama engine (`llm_used` / `fallback_used` / `validator_retries`) land in `report.json`.

**Refinement pass (Translation QC/Refinement box)** — new stage logic wired after the tqc battery in `vidub/stages/translate.py`:

1. tqc battery runs first, exactly as today (deterministic Detect + deterministic splices + verdict).
2. Rows still flagged `mistranslation-risk` / `inconsistent` / `unnatural` (and entity/terminology rows the splice could not fix) enter **one** refine cycle when `cfg.hunyuan_refine` is on: candidates = K hunyuan samples (temp 0.7) + the current vi + the envit5 vi.
3. If `cfg.fusion_model` is set → **Chimera fusion call** (official fusion prompt, same Ollama host, second GGUF) producing one fused candidate; else → **deterministic best-of-K** (numbers survival → length ratio → round-trip token-F1).
4. The winning candidate passes the shared validators and is kept **only if it strictly reduces the row's tqc flags** (re-scored); otherwise the original text is kept. One cycle, no loops. Counters + per-row outcomes appended to `translation_qc.json`.

**Config/CLI surface:** `--mt-engine {envit5,ollama,hunyuan}`; `--hunyuan-model` (default `hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K`); `--hunyuan-host`; `--hunyuan-refine` (default off); `--fusion-model` (default empty); `--hunyuan-samples` (K, default 3); `--no-hunyuan-refine` alias. Doctor gains a `hunyuan:` line (endpoint reachable, model pulled, license note). Version 0.5.0. **Nothing changes for `--mt-engine envit5` (default) or `ollama` runs.**

### 7.4 What does not change

Transcript QC (`qc.py`), punctuation restoration, subtitle QC (`subqc.py`), TTS + per-clip NISQA QC (`ttsqc.py`), timing fit, final QC gate (`finalqc.py`), the verdict policy (decision 10), and the envit5/ollama engines are untouched. No stage becomes an LLM task; the LLM (two specialist models, both opt-in) only ever proposes inside the Translation box, gated by the same deterministic validators as today.

### 7.5 License and distribution posture

- envit5 (Apache-2.0) remains the **default** MT engine — the default path stays fully open-source and commercially usable (hard rule, DESIGN.md §4.2).
- The Hunyuan engines are opt-in; ViDub ships no weights. `doctor` and README document: Tencent HY Community License — commercial use < 100 M MAU; **not valid in the EU/UK/South Korea**; AUP; no use of outputs to train other models; users accept it on their own HF download.

### 7.6 Benchmark procedure (pre-registered — must pass before any promotion)

**Harness:** `scripts/eval_hunyuan.py` (scratch, runs in `.eval-venv` + HTTP calls to Ollama; mirrors `scripts/eval_transquest.py`), deliverable `docs/hunyuan-evaluation.md`. **No new `.venv` runtime dependency** (engines speak HTTP).

**Systems compared on identical inputs:** envit5 (default baseline) · qwen2.5:7b contextual (current alt) · HY-MT1.5-7B Q6_K · HY-MT1.5-7B Q4_K_M · best-of-K refine without fusion · refine with Chimera fusion (if a clean Chimera GGUF verifies). Hunyuan-MT-7B 1.0 is optional (community GGUF) — 1.5 supersedes it on aggregates.

**Corpus:** (a) all En→Vi rows from the existing real job caches (a7TOameRqoY 74 rows, LPZh9BOjkQs, demo_en2 — source texts re-translated per system, judged against each system's output); (b) 30 curated hard cases from the glossary/pronounce test data (numbers, entities, technical terms, long sentences); (c) the full reference video E2E re-run per system.

**Metrics (all four must be reported):**
1. **Deterministic battery flag count per 100 rows** (tqc: number/entity/terminology/omission/addition/inconsistent/unnatural) — the primary quality proxy.
2. **TransQuest DA score** (reference-free human proxy; already validated, AUC 0.772) mean per system.
3. **Dubbing fit:** length-ratio distribution, share of rows > 2.2×, resulting subtitle-CPS violations.
4. **Cost:** s/row and VRAM peak per system/config on the RTX 3080.

**Pre-registered decision rule:**
- Promote `hunyuan` to the **documented recommended engine** (docs + README; still not the hard default) iff: TransQuest mean ≥ envit5 + 0.02 **and** battery flag count ≤ envit5 **and** ≤ 3 s/row average at Q6_K.
- Adopt **Chimera fusion** into the refine default iff it reduces unresolved review packets ≥ 30% vs deterministic best-of-K without increasing validator failures; otherwise fusion stays behind `--fusion-model`.
- envit5 remains the default engine regardless; promotion changes recommendations, never the shipped default without a new User decision.

### 7.7 Proposed task decomposition (after Gate G1)

| Task | Content | New deps |
|---|---|---|
| M5-T1 | `validators.py` extraction (no behavior change) + `hunyuan_mt.py` engine + config/CLI/doctor + unit tests (live-guarded) | none |
| M5-T2 | refine pass in the translate stage (best-of-K + optional fusion) + `translation_qc.json` counters + tests | none |
| M5-T3 | benchmark harness + `docs/hunyuan-evaluation.md` + decision record (incl. Ollama/hf.co pull feasibility VERIFY: `ollama run hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K`) | none (scratch `.eval-venv`) |
| M5-T4 | README/docs + version 0.5.0 + release E2E on the reference video | none |

Expect ~one 40-step continuation per task (M4 pattern). Builder chain ≤ 8 dispatches, Reviewer fresh-session per task, corrections ≤ 2.

### 7.8 Decisions requested from the User (Gate G1 for M5)

| # | Decision | MA recommendation |
|---|---|---|
| D1 | Build the **HY-MT1.5-7B opt-in engine** (`--mt-engine hunyuan`, official GGUF via Ollama, envit5 fallback, envit5 stays default) | **Yes** — biggest measured translation-quality lever, zero default-path impact |
| D2 | Build the **refine pass** (deterministic best-of-K over samples for flagged rows, strict-better acceptance) | **Yes** — completes Detect→Correct→Verify on the translation level |
| D3 | Add **Chimera-7B fusion** behind `--fusion-model` (community GGUF, opt-in; promote only on benchmark evidence) | **Yes, evaluate** — the model's designed purpose maps 1:1 onto our Correct step; adopt as default only if §7.6 passes |
| D4 | Run the **pre-registered benchmark** (§7.6) before any promotion claim | **Yes** — required by the directive ("before declaring the new approach better") |
| D5 | Keep **Transcript QC and Final Dub QC deterministic** (no LLM in either) | **Yes** — evidence shows they suffice; §3.1 principle |

**Gate G1:** M5 TASK_SPECs are written and dispatched only after the User explicitly approves this scope (any subset of D1–D5, with D5 recommended as non-negotiable). — **PASSED 2026-09-29** (User: "approve the M5 scope, start do it"); all five decisions approved as recommended.

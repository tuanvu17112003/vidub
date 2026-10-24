# DESIGN — ViDub: Free & Open-Source Video Translation + Vietnamese Dubbing Tool

**2026-10-01 subtitle/QC correction:** readable subtitle chunks now use bounded
dynamic scheduling and adjacent silence (0.25 s lead / 1.5 s lag); high CPS alone
does not recursively split text. Matched fixed-text/audio diagnostics: violations
203 → 35 and 186 → 9, all words preserved, no overlaps. Final QC probes placed
units and checks speech WAV existence, empty/short audio, saturation and pending
TTS MOS flags. Reports include combined synthesis/stretch speed and onset shift.
The opt-in AI profile fixes source-conditioned terminology aliases and includes
two reference sentences; entity handling canonicalizes plural acronyms. See
`docs/subtitle-scheduling-and-placed-qc.md` for VERIFY and remaining limits.

**2026-10-01 algorithm correction:** natural-pace fitting now preserves speech
that fits, measures whole sentence-unit audio before rate adjustment, guards
concise timing retries for completeness and improved measured duration, and
carries accepted wording into subtitle/alignment/TTS-QC consumers. Numeric
validators compare tokens and multiplicity. Isolated matched Piper experiment
(83 fixed translations): speed CV 0.1211 → 0.0655, slowed takes 45 → 0;
overflow 19 → 20 and clip risk 1 → 1 remain unresolved. See
[`docs/natural-pacing.md`](docs/natural-pacing.md) for research, VERIFY and limits.

**Status:** **ISO-1 CLOSED — v0.11.0, isometric length budgeting: BUILT, reviewed PASS, shipped OFF by default with a documented negative result.** The pass computes each sentence unit's character budget from its locked window at translate time (window × 20 chars/s × 0.9 headroom), carries it in the FIRST engine call (envit5 decode steering; hunyuan/ollama prompt budget), repairs still-over-budget units with ≤ 3 concise candidates under a tiered acceptance predicate (engine validator + guard floors + neighbour-based contamination check), and persists per-unit `isometric_reason`/`candidates`/`coverage_floor`/`short_floor_applied`. Matched A/B (`docs/isometric-budgeting.md`; both arms same input, 74 ASR rows → 113 script rows → 83 units): translation side **29 over-budget units → 17 shrunk** (0 before the two fixes: the self-bleed contamination check compared a candidate against the unit's own text, and one prompt omitted the target language); adaptive floor (0.65 below 1.8 s) used by 21 units. **Pre-registered pacing targets NOT MET: clips 2 → 2 (target 0) and `stretched` 19 → 27 (target ≤ 12, worse than the matched control)** — hence `DEFAULT_ISOMETRIC = False`, the feature kept behind `--isometric`, and the open hypothesis (shortened text re-shapes the fitter's slack/pace decisions instead of removing the squeeze) recorded for the next investigation. Suite 850 passed / 4 skipped. **Q-1 + SU-1 BUILT and verified (v0.10.0, 2026-09-30)** — the translation/audio quality round and the vietsub mode. **Q-1:** the hunyuan contextual template's neighbour-contamination failure is now caught deterministically — a **contamination guard** (neighbour-token containment ≥ 0.5 or a sentence addition at coverage ≥ 1.4) plus a **direct-template `decontext` repair**; the User's reported duplication case was repaired (unit 24: overlap 1.00/0.70 → 0.17/0.13) while the 6 surviving units stay `flagged` (**C1 = 0 units NOT MET**). **Refinement is default-on for the hunyuan engine** (tri-state, budget 40, `--no-hunyuan-refine` to opt out), measured **21 → 0 unresolved QC units** on the tech video (target ≤ 8 MET). **Pacing levers** now run in the order concise re-translation → bounded short-window stretch (`SPEED_CAP_SHORT = 1.30`) → minimal trim, with per-clip `{idx, seconds, reason}`; measured clips **7 → 4/5**, `stretched` **18 → 19/22** (**C3/C4 NOT MET** — the survivors need 1.30–1.54× against the 1.20×/1.30× caps). **Subtitle cues** split on the placed audio's real ASR word edges (`align_words`), improving cps 160 → 145, duration 72 → 68 and word-edge accuracy 0.217 → 0.321 on a controlled A/B. Suite 773 passed / 4 skipped. **SU-1:** `--mode subtitles` (alias of `subtitles-only`) skips TTS/pacing/mixing and keeps the original audio, with subtitle **position** (`--sub-pos bottom|top|center`), font size, margin, a real **`.ass` export** and a **styled burn** (note the two alignment numberings: ASS `n` for the `.ass`, legacy SSA `` for ffmpeg `force_style=Alignment=`), UI controls and `docs/subtitles-only.md`. Next: **isometric length budgeting** (translation-level window fitting) — the open item behind the pacing misses. Earlier: **SA-1 BUILT and verified (v0.9.0, 2026-09-30)** — sentences are now the atomic unit of the whole chain (the User's "broken sentences / mid-word pause" report). A new pre-pass (`vidub/utils/sentences.py::split_into_sentences`) splits ASR rows at internal sentence boundaries using the **ASR word timestamps**, so every `script.json` row *is a sentence with word-edge-locked `start`/`end`*; `group_units` can end a unit **only** at a completed sentence or a speaker change (a silence gap is a cut point, never a unit end) and the sole size guard is a hard ceiling (45 s / 150 words) that falls back to the largest gap ≥ 0.6 s as a counted last resort. On the unit path the translate stage now writes **`units.json`** (authoritative unit text + locked span + member rows), runs a **unit completeness guard** (coverage < 0.75 or a lost sentence ⇒ re-sample → envit5 fallback → `flagged`, always recorded — this is what catches the HY-MT omission that produced the User's broken text), scores the translation-QC battery **on units** instead of display fragments, and re-derives each row's display text from the final unit text; `mt_signature` encodes the pipeline version (`units=2`), so every pre-SA-1 cache re-translates. The timing stage synthesizes each multi-member unit from `units.json`'s text verbatim and adds **ASR forced alignment** of the placed unit WAV (`--align-display`, default on) → `timed.json` gains `align` + `member_spans` (word-edge-snapped, monotone, inside the placed audio; `asr` 15 / `split` 0 / `single` 39 on the reference run), plus a new `timing_signature` that invalidates the timing stage and everything downstream when a timing flag flips. Subtitles are **one cue per sentence unit**; the UI table gains `unit` / `aligned` / `vi_aligned` and the segment player plays only the selected row's aligned slice. Reference A/B on the User's own video (identical input file, `--asr-model base --mt-engine hunyuan`; `docs/sentence-locked-units.md`, raw `eval-cache/sa1/scores.json`): placed boundaries that cut a source sentence **3 → 0 (MET)**, units with dropped content **2 → 0 (MET)**, sentence-unit texts ending mid-sentence (`… là ví dụ của` / `… được đào`) **15 → 0 (MET)**, clips **0 → 0 (MET)**; **NOT MET: unresolved translation-QC flags 13 rows → 9 units (target ≤ 5 — the rate improved 20.6 % → 16.7 %, but the row→unit granularity change makes the counts non-comparable) and `stretched` 5 → 8 (target ≤ 5; `max_stretch` 1.106 → 1.109)**. Suite **700 passed / 4 skipped**. The User's reported sentence is now ONE unit (`And large language models are instances of foundation models applied specifically to text and text like things.` → its full Vietnamese translation, locked 61.56 → 69.04 s, placed 61.56 → 67.92 s, `align: asr`). Earlier: **UA-1 BUILT and verified (v0.8.0, 2026-09-30)** — the audio chain speaks **whole sentence units**: with `sentence_units` on (default) the timing stage groups the script rows into units, synthesizes **one WAV per multi-segment unit** from the member texts joined (single-segment units reuse the speak-stage WAV), fits it to the unit window and emits **per-unit `timed.json` rows** (`members: [...]`) — so a pause *inside* a sentence (the User's `…đào` ⏸ `tạo…`) is structurally impossible. Mix placement, TTS-QC, coverage, the UI segment player and the report counts are unit-aware; subtitles stayed per segment; `speak`/`speech.json` are untouched; `--no-sentence-units` keeps the per-segment path byte-for-byte. A/B on the User's complaint video (`docs/unit-audio.md`): intra-unit placed boundaries **14 → 0**, mid-word splits **6 → 0**, clips **0**, gaps > 0.8 s **45 → 39**, atempo nearly eliminated (48/49 units `asis`, max stretch 1.077), per-unit rate p10–p90 **3.72–4.87** words/s. **FIT-1 BUILT and verified (v0.7.0, 2026-09-30)** — the timing stage now **re-synthesizes mismatched segments at a fitted rate instead of squeezing or clipping them**: Piper's `length_scale` (speech speed clamped 0.85×–1.30×, as-is band ±8 %), a video-wide pace baseline (clamped median 1.0–1.10) so the voice stays consistent, bounded slower speech to fill wide windows and re-centering of leftover slack (1.5 s / 2.0 s bounds); `--tts-rate-fit` / `--recenter` (defaults on). Reference A/B (`docs/timing-fitting.md`): **clips 1 → 0 (pre-registered target MET)**, atempo severity max 1.17 → 1.10, unused-window seconds only **−1.9 % against the ≥ 50 % target (NOT MET — dead air is dominated by source pauses)**, rate spread unchanged. **TU-1 BUILT and verified (v0.6.0, 2026-09-30)** — the translate stage now works on **sentence-level translation units**: consecutive ASR fragments are merged into whole sentences before the MT call and each unit translation is split back onto the original segment windows (`vidub/utils/sentences.py`; `--sentence-units` / `--no-sentence-units`, **default on**; UI checkbox in the Engines & quality panel). Units are closed at a terminal sentence end, a speaker change or a silence gap ≥ 0.45 s, capped at 60 source words / 18 s. A changed `sentence_units` value participates in the MT cache signature, so toggling it always re-translates. Measured on the reference video (`docs/translation-units.md`): mid-sentence engine inputs **29 → 9** (−69 %; the residual 8 are size-cap cuts and 1 is a pause boundary, so the pre-registered target of 0 was NOT met), translation-QC flags **10.81 → 8.11 per 100 rows** (unresolved 8 → 6), subtitle QC mixed (cue duration violations 53 → 40, CPS 66 → 70), aggregate verdict unchanged (`review`). **UI-1 BUILT and verified (v0.5.1, 2026-09-30)** — the web UI is a real test surface for the quality features: an "Engines & quality (M4/M5)" panel (MT engine + model/host, refine + samples, Chimera fusion, punctuation restore, the five QC toggles, TTS-QC MOS floor), a force/re-run stage control, a "Quality results" panel (report verdict + transcript/translation/TTS/subtitles QC incl. the refine summary), one-click test presets and an Ollama pre-flight that blocks a run needing Ollama with the exact fix commands — **and a changed MT configuration now invalidates the cached `translate` stage and everything downstream** (`mt_signature`), so a new engine can never silently reuse a cached script (CLI included). Suite 503 passed / 4 skipped; live UI run verified (defaults vs `hunyuan`+refine on a short sample). Milestone **M5 BUILT and verified (v0.5.0, 2026-09-29)** — the opt-in **Hunyuan MT engine** (`--mt-engine hunyuan`: Tencent HY-MT1.5-7B served by a local Ollama server from the official GGUF, official specialist prompts, deterministic validators, per-segment envit5 fallback) and the opt-in **translation refine pass** (`--hunyuan-refine`: one best-of-K cycle over the rows the QC battery still flags, strict-better acceptance; `--fusion-model` Chimera fusion stays evaluation-only). The pre-registered En→Vi benchmark (250 segments, `docs/hunyuan-evaluation.md`) measured HY-MT1.5-7B as **comparable to envit5 but NOT promoted** (misses all three §7.6 criteria) — **envit5 stays the default engine** — and the **refine pass is the measured win (−16 % battery flags: 37.6 → 31.6 per 100 rows)**; Chimera fusion is NOT adopted. Historical baseline: **M4 (v0.4.0)** BUILT and verified — the quality-control framework (transcript / subtitle / translation / final QC gates), the opt-in Ollama contextual MT engine (`--mt-engine ollama`), source-transcript punctuation restoration (`--restore-punct`) and NISQA per-clip TTS QC; envit5 + Piper remain the default engines, so the default path stays open-source and commercially usable. Earlier baseline: M1, M2 (+C1), M3 (+C1, C2, C3 corrections) BUILT and verified (**v0.3.0 + M3-C3**, 2026-09-29) — 251 tests green, every engine on GPU. **M3-C3** (user-reported tech/gaming dub issues on youtube-a7TOameRqoY) added: a **TTS pronunciation lexicon** (`--pronounce`, `written<TAB>spoken` applied to the TTS text only — subtitles keep the written form; both speak and timing re-synthesis honor it), **plural-tolerant source-side glossary matching** (`GPU` finds `GPUs`, `peasant` finds `peasants`; output-side stays exact), and an expanded tech/gaming example glossary (GPU/VRAM identity rows, graphics card→card màn hình, peasants→anh em cấu hình thấp, Riddle me this→Đố anh em biết). The Run.bat launcher was shelved by the user (a corrupt partial was removed). M4 and M5 were delivered later. **M3-C2** (user-reported on the full ~8-min dub): expanded technical glossary (7 terms, two-sided enforcement, `enforced: 20` on the production run), deterministic Vietnamese word-order refinement (`vidub/utils/vnstyle.py`: time-adverbial fronting), and tail-safety in timing (smart trim lands cuts in detected silences with a 60 ms fade instead of mid-word, `smart_trim` vs `clipped` distinction, `--tail-pad` gives the final segment 2 s past the video end). The Run.bat launcher was shelved by the user (a corrupt partial was removed). M4 and M5 were delivered later. **M3-C1** (user-reported pacing/terminology defects on a real 8-min YouTube dub) reworked the timing-fit policy: availability windows (a segment may use the following silence), stretch cap lowered 1.35→**1.20** with an as-is band of 1.10, neighbor-smoothing (max delta 0.15) that can never under-fit, **filler-stripping re-translation** as the alternative to hard stretching, and last-resort clips that are hard-trimmed at the window end and flagged with reasons in `report.json` `pacing` (zero-bleed invariant, audibly verified). Glossary enforcement became two-sided (output replace + **source-side splice**), and translations are cleaned of hesitation/repetition artifacts before TTS. M4 and M5 were delivered later. M3 additions: CUDA torch (Demucs/envit5/ViXTTS on GPU, `sep_device` in report), translation glossary (`--glossary`, post-MT enforcement), TTS language matrix (`validate_langs` fail-fast), URL input (`vidub run <URL>`/`batch <URL>` via yt-dlp, `source_url` in report), per-speaker voice cloning (`--clone-speakers`: Demucs vocals + diarization → deterministic ≤20 s references → ViXTTS zero-shot), Gradio web UI (`vidub ui`, localhost-only, cooperative cancel via `PipelineCancelled`/`cancel_cb`), doctor extras lines. **Substitution vs this design:** the M3 "RVC voice cloning" item was delivered as ViXTTS zero-shot per-speaker cloning (reuses proven stack; no new training subsystem) — true RVC timbre conversion is an M4 option. M2 deviations below still apply, with these updates: torch is now the CUDA wheel (cu126); PhoWhisper therefore also runs on GPU; the doc nit list is resolved. Remaining M4 backlog: RVC timbre conversion, LLM whole-script translation + length-aware re-translation (needs Ollama installed), lip-sync (LatentSync), playlist/.txt URL lists. **M5 built 2026-09-29 (v0.5.0, Gate G2 PASS on all four tasks + the T2 correction; QUALITY-ROADMAP.md §7):** Tencent Hunyuan-MT integration delivered as designed — HY-MT1.5-7B as an opt-in `--mt-engine hunyuan` (official Q6_K GGUF via Ollama, official specialist prompts, deterministic validators + envit5 fallback preserved) plus the translation refinement pass (best-of-K, strict-better; optional Chimera-7B fusion behind `--fusion-model`) and the pre-registered En→Vi benchmark (`docs/hunyuan-evaluation.md`: **not promoted**, the refine pass is the measured win); envit5 stays the default engine; lip-sync/RVC shift to M6.

**Date:** 2026-09-30

**Milestone status:**

| Milestone | Status | Highlights |
|---|---|---|
| M1 (0.1.0) | ✅ done | En→Vi pipeline end-to-end: faster-whisper ASR, envit5 MT, Piper TTS, Demucs mix, ffmpeg mux, `vidub doctor`. |
| M2 (0.2.0) | ✅ done | PhoWhisper (vi ASR), ViXTTS + diarization + voice map, word timestamps, batch mode, `report.json`. |
| M3 (0.3.0) | ✅ done | CUDA-torch GPU pipeline, translation glossary, TTS language matrix, YouTube input, speaker cloning, local web UI. |
| **M4 (0.4.0)** | ✅ **done** | Quality-control framework (transcript → subtitle → translation → per-clip TTS → final aggregate gate), opt-in Ollama contextual MT, source-transcript punctuation restoration, NISQA per-clip TTS QC. |
| **M5 (0.5.0 — translation quality)** | ✅ **done** | Opt-in Hunyuan MT engine (HY-MT1.5-7B via Ollama, official GGUF) + opt-in translation refine pass (best-of-K, strict-better) + pre-registered En→Vi benchmark (`docs/hunyuan-evaluation.md`: NOT promoted — envit5 stays default, refine is the measured win −16 % flags, Chimera fusion evaluation-only). |
| **UI-1 (0.5.1)** | ✅ **done** | Web-UI test surface (engines/QC panel, force re-run, Quality results panel, test presets, Ollama pre-flight) + MT-config cache invalidation. |
| **TU-1 (0.6.0 — translation units)** | ✅ **done** | Sentence-level translation units (fragment merge + deterministic split-back, default on) + `--sentence-units` flag + reference A/B (`docs/translation-units.md`: mid-sentence inputs 29 → 9, tqc flags 10.81 → 8.11 per 100 rows). |
| **FIT-1 (0.7.0 — timing/rate fitting)** | ✅ **done** | Two-pass TTS rate fitting (Piper `length_scale`, speed clamped 0.85×–1.30×) + video-wide pace + bounded re-centering, default on; reference A/B (`docs/timing-fitting.md`): clips 1 → 0 (target MET), atempo severity down, unused-window −1.9 % (≥ 50 % target NOT MET — source-silence dominated). |
| **UA-1 (0.8.0 — unit audio)** | ✅ **done** | One synthesis + one placed row per sentence unit (intra-sentence pauses structurally impossible); per-unit `timed.json` rows (`members`), mix/TTS-QC/coverage/UI player/report made unit-aware; `--no-sentence-units` unchanged. A/B on the User's complaint video (`docs/unit-audio.md`): intra-unit boundaries 14 → 0, mid-word splits 6 → 0, clips 0, gaps > 0.8 s 45 → 39, max stretch 1.077. |
| **SA-1 (0.9.0 — sentence-locked units + aligned display)** | ✅ **done** | Sentence rows with word-edge-locked timestamps (`split_into_sentences`); units end only at a completed sentence or a speaker change (hard ceiling the only size guard, counted pause cut as last resort); `units.json` (authoritative unit text + locked span + members); unit completeness guard (re-sample → envit5 fallback → flagged); QC battery on units; per-unit audio from the unit text verbatim; ASR forced alignment of the placed unit WAV → `timed.json` `align` + `member_spans` (`--align-display`); subtitles one cue per sentence unit; UI aligned columns + slice player; `timing_signature` invalidation; suite 700 passed / 4 skipped; A/B `docs/sentence-locked-units.md` (**M1 3 → 0, M2 2 → 0, M3 15 → 0, clips 0 → 0 MET; M4 13 → 9 and `stretched` 5 → 8 NOT MET**). |
| **Q-1 (0.10.0 — translation/audio quality round)** | ✅ **done** | Hunyuan contamination guard + `decontext` repair (unit 24 repaired; 6 survivors flagged); refine default-on for hunyuan (21 → 0 unresolved); pacing levers concise → stretch → trim with per-clip reasons (clips 7 → 4/5, stretched 18 → 19/22 — both NOT MET); subtitle cues on real word edges (`align_words`; cps 160 → 145). Page: `docs/quality-round-q1.md`. |
| **SU-1 (0.10.0 — vietsub mode)** | ✅ **done** | `--mode subtitles` alias (skip speak/timing/mix, keep the original audio), `speak` mode guard, TTS-validation skip, report wording; `--sub-pos/-sub-font-size/-sub-margin-v`, `.ass` export, styled burn (ASS vs legacy alignment maps); UI controls; `docs/subtitles-only.md`. |
| **ISO-1 (0.11.0 — isometric length budgeting)** | ✅ **closed, default off** | Per-unit character budgets in the first engine call + Best-of-K compression (+≤3 candidates, tiered acceptance, persisted reasons, adaptive 0.65 floor below 1.8 s); matched A/B: 17/29 units shrunk but clips 2 → 2 and `stretched` 19 → 27 (targets 0 / ≤ 12 **NOT MET**) ⇒ ships off by default, negative result documented in `docs/isometric-budgeting.md`; correction fixed the floor-propagation defect the review caught. |
| M6 (lip-sync / RVC) | ⏳ not started | See §5 "Advanced". |

**Correction & deviation log (historical, resolved where noted):**
- **M2-C1 (user-reported, fixed):** a vi-specialist ASR model run on English audio hallucinated Vietnamese → gibberish dub + silence after ~00:08. Fixed with the post-ASR language-mismatch guard (`vidub/utils/langcheck.py`, `--allow-lang-mismatch` override), translation hardening (beam search, repetition guard, already-Vietnamese passthrough, low-confidence flags), and the speech-coverage invariant (missing segments abort; warning < 0.6).
- **PhoWhisper license is BSD-3-Clause** (not MIT as §4.2 assumed) — still commercial-usable; listed correctly in doctor/README. It also cannot load via faster-whisper (transformers-format weights) → transformers-pipeline fallback, now GPU-capable with the CUDA torch.
- **ViXTTS is CPML (non-commercial)** — opt-in only behind `--allow-nc-models`, per the license guardrail (Piper stays the MIT default). Requires the torchaudio/transformers/tokenizer-vi compatibility shims in `vidub/engines/tts/vixtts.py`; pyannote is installed `--no-deps` for the same wheel-mismatch reason.
- **Length-aware re-translation loop deferred to M4:** Ollama is not installed on this machine and envit5 cannot be length-prompted; the timing policy (atempo ≤1.35, truncation guard, coverage invariant) covers it for now.
- **Resolved during M2/M3:** GPU torch (was CPU-first), `--force` downstream invalidation, voice-over gain configurable, speech-less clips export without a subtitle stream, `vidub clean` doc contradiction.
- **M4 (0.4.0) — QC framework delivered (deviation log):**
  - `manifest.Segment` gained a **`meta`** field (per-segment ASR confidence signals: `avg_logprob` / `compression_ratio` / `no_speech_prob` / `temperature`); `JobManifest` gained **`transcript_verdict`** and **`translation_verdict`**.
  - `report.json` gained a **`subtitles_qc`** block (M4-T2) and the aggregate **`quality`** block (M4-T4). Per-level artifacts live in the cache dir as `transcript_qc.json`, `translation_qc.json`, `subtitles_qc.json`, `tts_qc.json`, with human-review packets under `review/{transcript,translation,tts}/`.
  - **Dependency added:** `torchmetrics>=1.6.1` (M4-T7) for the NISQA non-intrusive MOS model. Its other requirements (`librosa`, `requests`) and `torch` were already installed; weights (~1 MB) download once on first use into `~/.torchmetrics/NISQA`. A missing scorer degrades to `skipped` counters — it never fails a run.
  - **Ollama LLM MT engine** behind `--mt-engine ollama` (`--ollama-model`, `--ollama-host`): the model only *proposes*, deterministic validators (language / numbers / length) gate every output, and a failed segment falls back to envit5. **envit5 remains the default engine**, so the default path stays Apache-2.0 and commercially usable; the LLM path is an explicit opt-in.
  - **Punctuation restoration** (`--restore-punct`, M4-T6) is opt-in (default off), using `oliverguhr/fullstop-punctuation-multilang-large` (~2.2 GB first-use download); a density guard never re-writes already-punctuated text.
  - **No behavior change when disabled:** every QC layer is individually disableable (`--no-qc-transcript`, `--no-subtitle-qc`, `--no-translation-qc`, `--no-qc-tts`, `--no-qc-final`); a disabled layer is byte-identical to v0.3.0. The QC layers default **on**.
  - **Length-aware re-translation (deferred in M2-C1)** is now covered by the M4-T5 Ollama engine plus the existing timing filler-strip re-translation retry.

---

## 1. Goal

A local Python tool that takes a video (any language, En→Vi first-class) and produces:

1. a **dubbed MP4** with a Vietnamese voice track (original music/effects preserved where possible),
2. an **Vietnamese SRT** (soft-embedded and/or burned in),
3. an intermediate **dubbing script** (timed segments: source text → Vietnamese text → audio file), inspectable and editable for manual corrections.

Everything in the **default path is free and open-source with commercial-usable licenses (MIT / Apache-2.0)**. Models with non-commercial licenses exist only behind an explicit opt-in flag.

**Working name:** `ViDub` (CLI package `vidub`). Open to renaming — §10 D6.

## 2. Verified environment (this machine, checked 2026-09-28)

| Fact | Value | Consequence for design |
|---|---|---|
| GPU | NVIDIA RTX 3080, 10 GB VRAM | GPU presets default; stages run **sequentially** so nothing stacks above 10 GB |
| RAM | 31.6 GB | fine |
| Python | 3.10.11 | matches all engine requirements |
| ffmpeg | **NOT installed** | M1 must install it (winget/choco) and the tool must verify it at startup |
| Disk | H: has **42 GB free** | total model download ≈ 10–14 GB — fits, but cache policy and `vidub clean` are Core features |
| OpenCode | v2.0.18, `opencode-go/deepseek-v4.1-flash` | implementation engine per `WorkFlowMultiAgent/04-opencode-execution.md` |
| Shell | Git Bash on Windows | Unix syntax in all Builder VERIFY commands |

## 3. Research: what exists, what we take from it

Comparable open-source projects studied (Phase 0):

| Project | License | Take / leave |
|---|---|---|
| [VideoLingo](https://github.com/Huanshere/VideoLingo) | permissive (verify LICENSE at build) | **Take the pipeline shape**: whisperX ASR → NLLB/LLM translation → dubbing TTS, Netflix-style single-line subtitles. Leave: its cloud-API defaults (we want local-first). |
| [pyvideotrans](https://github.com/jianchang512/pyvideotrans) | GPL-3.0 | **Inspiration only, no code reuse** (copyleft would contaminate). Confirms feature set: many TTS engines, subtitle modes, batch. |
| SoniTranslate | non-standard license (NOASSERTION) | **Inspiration only.** Confirms multi-speaker dubbing + voice mimicking are achievable with OSS components (whisper + RVC + XTTS). |
| Linly-Dubbing / OpenDub / PersoDub | various | Confirmation of the same architecture; nothing we must depend on. |

Vietnamese-specific components verified by search:

- **PhoWhisper** (VinAI, [GitHub](https://github.com/VinAIResearch/PhoWhisper), [HF](https://github.com/VinAIResearch/PhoWhisper)) — Whisper fine-tuned for Vietnamese on 844 h, 5 sizes, MIT. Vietnamese-specialist ASR.
- **envit5-translation** (VietAI) — En↔Vi translation, Apache-2.0.
- **ViXTTS** — Vietnamese-specialized VITS-family TTS with **zero-shot voice cloning** (~500★ GitHub; license re-verify on model card at build).
- **F5-TTS Vietnamese fine-tune** — community model trained on ~1000 h (Vi-Voice, VLSP 2021–2023); highest reported quality; base F5-TTS weights are CC-BY-NC → Advanced/opt-in only.
- **Piper `vi_VN` voices** (MIT, CPU, instant) — `vi_VN-vais1000-medium`, `vi_VN-25hours_single-low`.
- **Qwen3-TTS is NOT open-source** (API product) — excluded.

## 4. Architecture

### 4.1 Pipeline (stages, each independently runnable and resumable)

```text
 video ──► S0 Ingest ──► S1 Transcribe ──► S2 Diarize(opt) ──► S3 Translate ──► S4 TTS ──► S5 Timing fit ──► S6 Audio mix ──► S7 Mux/Export
              │  ffmpeg      faster-whisper      pyannote         envit5 / LLM      Piper /      atempo +        Demucs BGM      ffmpeg
              │  16k mono    / PhoWhisper        (speaker→        (dubbing-style,   ViXTTS       silence pad     + loudnorm      mux + SRT
              │  wav         + Silero VAD         voice map)       length-aware)                     + re-translate
              ▼                ▼                    ▼                  ▼                ▼            ▼               ▼              ▼
          job.json        segments.json      speakers.json      script.vi.json   seg_0001.wav  timed.json     bgm.wav, mix.wav  dubbed.mp4
```

Every stage reads/writes a **job manifest** (`job.json`) + artifacts under `.vidub-cache/<video-hash>/`. A stage that already produced its artifact (hash match) is skipped → **crash-resumable, replayable, per-stage testable**:

```bash
vidub run input.mp4 --lang en->vi --mode dub          # full pipeline
vidub run input.mp4 --stages s1,s3 --resume           # re-run only translation
vidub stage transcribe --job .vidub-cache/abc/job.json  # one stage standalone (Builder VERIFY unit)
```

### 4.2 Engine layer (pluggable, chosen per run)

| Role | Default (M1) | Quality upgrade (M2) | Advanced (M4) | License notes |
|---|---|---|---|---|
| ASR | `faster-whisper` large-v3 (int8 on GPU) — MIT | **PhoWhisper-large** (vi-specialist) — MIT | — | all MIT ✅ |
| Speaker diarization | off (single voice) | **pyannote 3.1** (HF-gated, free after accepting terms) | — | MIT code, gated weights ⚠️ |
| Translation En→Vi | **envit5-translation** — Apache-2.0 | envit5 + **length-aware re-translation loop** + **refine pass** (`--hunyuan-refine`, M5 — the measured win, default off) | local LLM via Ollama: **qwen2.5:7b** contextual translate→reflect→improve (M4, Apache-2.0) · **HY-MT1.5-7B** GGUF (M5, opt-in, benchmarked **NOT promoted** — envit5 stays default) · **Chimera-7B fusion** (`--fusion-model`, evaluation-only) — Tencent HY Community License ⚠️ opt-in, never default | ✅ (NLLB excluded from default: CC-BY-NC; Hunyuan documented in doctor, weights never bundled) |
| TTS Vi | **Piper** `vi_VN` (CPU, instant, robotic-but-clear) | **ViXTTS** (cloning from ~10 s reference; natural) | F5-TTS-vi finetune (best quality, **CC-BY-NC-suspect** → opt-in) / GPT-SoVITS vi finetune | Piper MIT ✅; ViXTTS verify card; F5 base NC ❌ |
| Voice conversion | — | — | **RVC** — MIT (map TTS timbre → original speaker) | ✅ |
| Source separation | **Demucs htdemucs** (vocals vs BGM) — MIT | same | same | ✅ |
| Subtitles | pysubs2 (MIT) + ffmpeg/libass burn | whisperX word-level timing — verify BSD | — | ✅ |
| Input sources | local file | + folder batch | + YouTube via yt-dlp (Unlicense) | ✅ |
| UI | CLI | — | Gradio web UI (Apache-2.0) | ✅ |

**License guardrail (hard rule):** `vidub doctor` prints every active engine + license; engines flagged non-commercial (F5-TTS-vi, NLLB, MMS-TTS, XTTS) require `--allow-nc-models` and are refused otherwise. The default path must stay commercially usable end-to-end.

**Documented fallback:** `--tts edge` (Microsoft Edge read-aloud voices, incl. vi-VN) is *free but not open-source and an unofficial API* — shipped off by default, clearly labeled.

### 4.3 Timing fit (the hard part of dubbing)

Per segment: measured TTS duration vs. original speech slot.

1. `atempo` time-stretch if 1.0 < ratio ≤ 1.25 (pitch-preserving, barely audible);
2. if TTS is too long by > 25 % → **re-translate shorter** (translation step asked for a ≤ N-word variant) — max 1 retry, then accept 1.35× stretch;
3. if too short → pad silence to slot start (keeps lip rhythm);
4. Vietnamese tends to run *longer* than English → the translation prompt is length-constrained from the start (`≤ 0.9 × source word count`), which removes most retries.

### 4.4 Audio mix

- Demucs splits `vocals` / `no_vocals`. Dub track = TTS segments at their slots + `no_vocals` bed.
- Mode `dub` (replace vocals) vs `voice-over` (original vocals ducked to −18 dB under the Vi voice) vs `subtitles-only` (no audio change).
- Loudness normalized to EBU R128 (ffmpeg `loudnorm`).

### 4.5 Project layout

```text
3/                          ← repo root (this directory)
├── DESIGN.md               ← this document (MA work product)
├── WorkFlowMultiAgent/     ← process docs, tasks/ specs, session register
├── pyproject.toml
├── vidub/
│   ├── cli.py
│   ├── manifest.py         ← job.json schema (single source of truth)
│   ├── stages/             ← s0_ingest … s7_mux, one file each
│   ├── engines/
│   │   ├── asr/  mt/  tts/  vc/  sep/
│   └── utils/    (ffmpeg wrapper, cache, license guard, logging)
├── tests/  +  samples/     ← bundled 30 s clip with known transcript
└── .vidub-cache/           ← per-job artifacts (gitignored, `vidub clean`)
```

## 5. Feature groups (Rule.md §Core/Important/Nice/Advanced)

**Core — M1 "end-to-end works"** (En→Vi, local files, single voice)
1. Repo scaffold, ffmpeg bootstrap + `vidub doctor` env check
2. S0 ingest + manifest
3. S1 faster-whisper ASR (CPU/GPU autodetect) + Silero VAD → segments + draft SRT
4. S3 envit5 translation (length-constrained)
5. S4 Piper vi TTS → S5 timing fit → S6 Demucs BGM mix + loudnorm → S7 mux
6. CLI `vidub run` + resume + embedded/burned SRT + README
   **Demo:** `vidub run samples/demo.mp4` → `demo.dub.vi.mp4`, duration drift < 1 s, subtitle stream present (ffprobe-verified).

**Important — M2 "sounds good, scales"**
PhoWhisper engine · ViXTTS engine + per-speaker diarization & voice map · length-aware re-translation loop · whisperX word timing · loudness profile config · batch folder mode · `report.json` per job.

**Nice-to-have — M3 "pleasant to use"**
Gradio web UI (upload → progress → preview audio per segment → download) · YouTube input (yt-dlp) · RVC voice cloning of original speakers · glossary/terms file for translation · multi-language target stub (vi first-class, structure ready).

**Advanced — M4 "studio grade"**
F5-TTS-vi engine (opt-in NC) · GPT-SoVITS vi finetune · whole-script LLM translation pass (Qwen3-4B local) · lip-sync (LatentSync — verify license, RTX-heavy) · speaker voice library (save/load cloned voices) · scene-aware BGM ducking.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| OSS Vietnamese TTS quality gap vs cloud voices | Tiered engines; ViXTTS promoted to M2; `edge` fallback documented but off; voice cloning via RVC masks base-voice timbre |
| 10 GB VRAM contention | Stages run sequentially, one model loaded at a time; large-v3-int8 (~5 GB), Demucs (~3 GB), ViXTTS (~3 GB) all fit individually |
| Disk pressure (42 GB free) | Model prefetch ≈ 12 GB total; `vidub clean` per-job; HF cache pinned on H: |
| Dubbing rhythm feels off (segments too long) | Length-constrained translation + atempo + silence padding (§4.3); segments capped ~8 s by VAD |
| pyannote models are HF-gated | Diarization optional; single-voice fallback; docs include 1-minute token setup |
| License contamination (GPL projects) | pyvideotrans/SoniTranslate: inspiration only, zero code reuse; license guard in `vidub doctor` |
| Whisper hallucinations on music/noise | VAD + `condition_on_previous_text=False` + no-speech filter; Demucs vocals feed ASR (M2) |

## 7. Execution plan after Gate G1 (per `WorkFlowMultiAgent/03-workflow-lifecycle.md`)

M1 becomes 7 sequential TASK_SPECs (`WorkFlowMultiAgent/tasks/M1-T<n>-spec.md`), dispatched to the **Builder** via `opencode run --agent builder --auto`, chained on one session (`--session`, register in `tasks/session-register.md`), each reviewed by the **Reviewer** + MA verification (Gate G2), correction loop ≤ 2. Estimated: T0 scaffold → T1 ingest → T2 ASR → T3 translate → T4 TTS → T5 mix → T6 CLI/docs.

Suggested milestone demo (acceptance evidence for G3): dubbed `samples/demo.mp4` + one real ~2 min video of the User's choosing.

## 8. Out of scope (v1)

Streaming/real-time dubbing, macOS/Linux packaging (Windows-first, keep code portable), commercial voice models, cloud APIs in the default path.

## 9. Open verification items (Builder checks at build time, before coding)

1. ViXTTS model-card license + VRAM footprint at inference.
2. Piper `vi_VN` voice-card data licenses (both voices).
3. VideoLingo LICENSE exact terms (reference only — no code reuse planned).
4. whisperX license + whether word timing is worth M2 vs plain VAD segments.
5. F5-TTS-vi finetune: hosting location, exact license chain (Vi-Voice/VLSP data) — decides M4 opt-in wording.

## 10. Decisions requested from the User (Gate G1 prep)

| # | Decision | MA recommendation |
|---|---|---|
| D1 | M1 scope = En→Vi dubbing of local files, single voice | Approve — smallest thing that proves the whole pipeline |
| D2 | TTS order: Piper (M1) → ViXTTS (M2) | Approve — Piper de-risks M1; ViXTTS is the quality jump |
| D3 | Translation: envit5 (M1) + LLM whole-script pass (M4) | Approve — dedicated MT model is fast, deterministic, Apache-2.0 |
| D4 | Modes needed in M1: `dub` + `voice-over` + `subtitles-only` | Approve — dub & voice-over share 95 % of code; subtitles-only is nearly free |
| D5 | UI: CLI in M1, Gradio in M3 | Approve — CLI is testable by the Reviewer; UI only after pipeline is stable |
| D6 | Name: `ViDub` (package `vidub`) | Approve or propose alternative |

**Gate G1 passes on explicit approval of this design (or of an amended one).** After that, the MA writes the M1 TASK_SPECs and starts dispatching to the OpenCode Builder — no production code before that point.

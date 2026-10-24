# ViDub

ViDub is a free and open-source CLI that turns a video in any language into a
**Vietnamese-dubbed video** with a Vietnamese subtitle track and an inspectable
dubbing script. Everything runs locally, the default path uses
commercially-usable (MIT / Apache-2.0) components, and the pipeline is
crash-resumable and run per stage.

> **Status: v0.11.0 (in development).** The full En→Vi pipeline works
> end-to-end (`vidub run clip.mp4`): ingest → transcribe → (optional
> diarize → references) → translate → speak → timing → mix → export.
> Version history: [CHANGELOG.md](CHANGELOG.md).

## Features (0.10.0)

- **GPU Whisper** — faster-whisper auto-detects CUDA (`float16`) and falls back
  to CPU (`int8`) when the CUDA runtime is missing; `vidub doctor` reports both.
- **GPU pipeline (CUDA torch)** — with the CUDA torch wheel, Demucs separation,
  envit5 translation and the opt-in ViXTTS TTS run on the GPU as well; stages run
  **sequentially** so only one model is resident at a time. See
  [GPU acceleration](#gpu-acceleration).
- **ASR presets + PhoWhisper** — `--asr-model` takes presets (`cpu-tiny` …,
  `gpu-large`, `vi-small`, `vi-medium`), a raw faster-whisper size, or any HF
  repo id. `vinai/PhoWhisper-*` runs through the transformers fallback.
  **`vi-*` presets are Vietnamese-source only** — on other languages they
  hallucinate (guarded; see below).
- **Wrong-language ASR guard** — after transcription, a transcript that looks
  Vietnamese while `--src-lang` isn't aborts with an actionable error (naming the
  model and the measured diacritic ratio) instead of dubbing gibberish. Override
  with `--allow-lang-mismatch`.
- **TTS language matrix** — `--tgt-lang` is validated **fail-fast, before any
  work**, against the chosen engine's languages: Vietnamese is first-class for
  every engine and `vixtts` adds 17 more XTTS-v2 languages; an unsupported pair
  aborts with an actionable error. See [Language support](#language-support).
- **Translation glossary (opt-in)** — `--glossary terms.tsv` enforces fixed
  `term<TAB>translation` rows on the machine translation, both on the MT output
  and (new in M3-C1) from the **source side**, and records the applied counters
  in `report.json`. See [Translation glossary](#translation-glossary).
- **Pronunciation lexicon (opt-in)** — `--pronounce lexicon.tsv` maps
  `written<TAB>spoken` rows so an acronym like `VRAM` (`Vê-ram`) or `GPU`
  (`Gờ-Pê-U`) is spoken the way you want while the subtitles keep the written
  form. See [Pronunciation lexicon](#pronunciation-lexicon).
- **Sentence-locked translation units with unit-level audio (default on)** — ASR
  rows are split at internal sentence boundaries with **word-edge-locked
  timestamps**, so `script.json` rows are **sentence rows**. Consecutive rows are
  merged into a **sentence-scoped unit** that ends only at a completed sentence
  or a speaker change (no gap-based ends; consecutive complete sentences are not
  merged); the only size guard is the hard ceiling (150 source words / 45 s),
  which cuts at the largest gap ≥ 0.6 s as a counted last resort and otherwise
  leaves the unit whole (`oversized`). A **completeness guard** re-samples or
  falls back when a unit translation drops content, and the translation-QC
  battery scores **units** (the authoritative map is `units.json`). The timing
  stage synthesizes each multi-member unit as **one utterance** and places a
  **single timed row** across its span, and force-aligns that audio so every
  member row gets a real `member_spans` display span
  (`--align-display`, default on; `--no-align-display` uses the split fallback).
  `--no-sentence-units` restores the previous per-segment translation and timing
  byte-for-byte. See
  [Sentence-level translation units](#sentence-level-translation-units) and
  [Unit-level audio (UA-1)](#unit-level-audio-ua-1).
- **LLM translation (opt-in, local)** — `--mt-engine ollama` runs the contextual
  translate → reflect → improve loop on a local [Ollama](https://ollama.com)
  model. The LLM only *proposes*: deterministic validators (language, numbers,
  length) gate every output and each failed segment falls back to envit5. No new
  dependency (stdlib `urllib`). See [LLM translation (Ollama)](#llm-translation-ollama).
- **Hunyuan MT engine (opt-in, M5)** — `--mt-engine hunyuan` runs Tencent
  **HY-MT1.5-7B**, a *translation-specialist* LLM served by the same local
  [Ollama](https://ollama.com) server (official GGUF; you pull it yourself). The
  default engine stays `envit5`; the LLM only proposes and deterministic
  validators gate every output, with a per-segment envit5 fallback. Tencent HY
  Community License (not valid in the EU/UK/South Korea). See
  [Hunyuan MT engine (opt-in, M5)](#hunyuan-mt-engine-opt-in-m5).
- **Translation refine pass (default on for hunyuan, Q-1-T2)** —
  `--hunyuan-refine` re-samples the units the QC battery still flags
  (best-of-`--hunyuan-samples`, default 3, bounded by
  `--hunyuan-refine-max-units`, default 40) with the local Hunyuan GGUF plus an
  envit5 candidate, and keeps a unit only when a candidate is **strictly
  better** under the deterministic validators. `--mt-engine hunyuan` turns it
  **on** by default; `--no-hunyuan-refine` restores the previous behaviour and
  non-hunyuan engines never refine; `--fusion-model` (Chimera) stays
  evaluation-only.
- **Mild pacing fit (M3-C1)** — the timing policy caps atempo at 1.20, uses the
  following silence as an availability window, re-translates segments that do not
  fit (one retry) and only clips as a last resort; neighbour stretch jitter is
  smoothed. See [Dubbing pacing](#dubbing-pacing-timing-fit).
- **Two-pass TTS rate fitting (FIT-1)** — Piper segments that would be squeezed
  or clipped are first **re-synthesized at a fitted rate** (`length_scale`
  0.85×–1.30× around natural, with a video-wide pace offset), so the voice stays
  in its comfort band and the clip path is a rarer last resort; leftover window
  slack is used by bounded re-centering. Default on; `--no-tts-rate-fit` /
  `--no-recenter` restore the previous behaviour. See
  [Two-pass rate fitting](#two-pass-rate-fitting-fit-1).
- **Word-level timestamps** — `--words` (default on) tightens each subtitle cue
  to the last spoken word.
- **Punctuation restoration (opt-in)** — `--restore-punct` inserts
  periods/commas/question marks and basic casing into the **source transcript**
  before translation (XLM-R punctuation model, ~2.2 GB on first use). A density
  guard leaves already-punctuated text untouched; default off, so the pipeline
  is byte-identical without it. See [Punctuation restoration](#punctuation-restoration).
- **Per-clip TTS QC** — every final fitted clip is scored with **NISQA-TTS**
  (MOS 1–5); low clips get one changed-settings retry (vixtts temperature/top_k;
  piper alternate voice) and a review packet if still low. `--no-qc-tts`
  disables it; `--tts-qc-mos-min` sets the threshold (default 3.0). See
  [Per-clip TTS QC](#per-clip-tts-qc).
- **ViXTTS (opt-in)** — natural Vietnamese speech + zero-shot voice cloning via
  `--tts-engine vixtts --allow-nc-models` (CPML non-commercial weights).
- **Speaker cloning (opt-in)** — `--clone-speakers` (with `--diarize
  --tts-engine vixtts --allow-nc-models`) extracts a clean reference per diarized
  speaker from the separated vocals and dubs each person in a voice resembling
  their own. See [Clone the original speakers](#clone-the-original-speakers).
- **Diarization (opt-in)** — `--diarize` labels speakers with pyannote 3.1 and
  gives each speaker its own voice (`--voice-map`).
- **Batch mode** — `vidub batch <folder>` dubs every clip sequentially, writes
  `batch-summary.json`, and supports `--resume-batch`.
- **Subtitles-only mode (vietsub)** — `--mode subtitles-only` (alias
  `--mode subtitles`) runs only `ingest → transcribe → translate → export` and
  keeps the **original audio**, producing a Vietnamese vietsub video with no TTS
  (the `speak` / `timing` / `mix` stages are skipped); `--burn` renders the styled
  subtitles into the picture instead of a soft track. See
  [Subtitles only (vietsub)](#subtitles-only-vietsub).
- **URL input (opt-in)** — `vidub run <URL>` (and `vidub batch <URL>`) downloads
  the media with yt-dlp, then runs the normal pipeline; re-runs reuse the
  download. Needs `pip install -e ".[youtube]"`.
- **Local web UI (opt-in)** — `vidub ui` opens a Gradio page to upload/paste a
  video or URL, watch per-stage progress, inspect the script/segment audio, and
  download the dub. The form is generated from the same option registry as the
  CLI (`vidub.options`), so both surfaces expose the same run options. An
  **Engines & quality (M4/M5)** accordion exposes the MT
  engine + model/host, the refine pass and Chimera fusion, the sentence-level
  units switch, restore-punct and the five QC toggles; a **Timing & budgeting**
  accordion exposes the stretch/re-translation levers and the opt-in ISO-1
  isometric budget; one-click **test presets**
  configure the M5 features; a
  **Re-run these stages (force)** control re-processes a cached video; and a
  **Quality results** panel shows the QC verdicts/counters. A run that needs
  Ollama **refuses to start** with the fix commands when the server or model tag
  is missing. Needs `pip install -e ".[ui]"`.
- **Configurable loudness** — EBU R128 targets via `--loudness-target/-tp/-lra`.
- **Coverage + hardened translation** — the timing stage guarantees every script
  segment has speech/timed audio and reports `speech_coverage` (a WARNING below
  0.6, i.e. a half-silent dub); already-Vietnamese segments pass through the
  translator untouched and diacritic-free output is flagged `low_confidence`.
- **Per-job `report.json`** — a `meta` block (tool version, input `source_url`,
  resolved ASR model + device, separation device, engines, mode, cloned-speaker
  count, total seconds), a `translation` counter block
  (`passthrough` / `low_confidence`), a `glossary` block (`terms` /
  `replacements` / `spliced` / `enforced`), a `pronunciation` block
  (`replacements`), a `cleanup` block (`hesitations` /
  `filler_removals` / `style_fixes`), a `pacing` block (`stretched` /
  `retranslated` / `clipped` / `smart_trims` / `clips` / `max_stretch` /
  `max_neighbor_delta`), a `references` block
  (per-speaker reference lengths), and `speech_coverage`; M4 adds a
  `subtitles_qc` block, the per-level verdicts, and the aggregate `quality`
  block. Speech-less clips export with no subtitle stream and report
  `"subtitles": "none (no speech detected)"`.

## Quality control (M4)

ViDub 0.4.0 adds a five-layer, **default-on quality-control framework**
(QUALITY-ROADMAP.md P0/P1) around the pipeline. Each layer *detects* something,
records a machine-readable result, and — where a fix is safe — *corrects it
deterministically*. An LLM is never the final word: the Ollama engine only
proposes, and deterministic validators gate every output. Every layer can be
disabled with its flag, and a disabled layer is byte-identical to 0.3.0.

| Layer | Detects | Auto-fixes | Verdict / artifact | Disable with |
|---|---|---|---|---|
| **Transcript QC** (L1) | hallucination loops, low-confidence garble, wrong-language, run-on/non-monotonic timing | one re-ASR of the flagged window with changed settings | `transcript_verdict` + `transcript_qc.json` + `review/transcript/` packets | `--no-qc-transcript` |
| **Subtitle QC** | cues over 17 CPS, < 0.83 s / > 7 s, > 2 lines or > 42 chars/line | readable chunks with bounded scheduling and adjacent silence; infeasible cues stay flagged | `subtitles_qc` report block + `subtitles_qc.json` | `--no-subtitle-qc` |
| **Translation QC** (L2) | number / entity / terminology errors, omission / addition, cross-segment inconsistency, low naturalness / round-trip fidelity | glossary splice, terminology re-enforcement, one re-translation retry | `translation_verdict` + `translation_qc.json` + `review/translation/` packets | `--no-translation-qc` |
| **Per-clip TTS QC** | low naturalness (NISQA-TTS MOS) on every final fitted clip | one re-synth with changed settings (vixtts `temperature`/`top_k`; piper alternate voice), keeps the higher-MOS clip | `tts_qc.json` + `review/tts/` packets | `--no-qc-tts` |
| **Final QC gate** (L3) | aggregates the per-level verdicts and adds fresh output checks (loudness delta, A/V duration, silent/clipped speech rows, subtitle violations) | — (reports only) | `quality` block in `report.json`; a `fail` makes `vidub run` exit non-zero | `--no-qc-final` |

Two opt-in quality upgrades ride alongside: the **Ollama LLM MT engine**
(`--mt-engine ollama`) and **source-transcript punctuation restoration**
(`--restore-punct`).

The current subtitle scheduler keeps the same readability profile and limits
display boundaries to 0.25 s early / 1.5 s late relative to speech references.
Final audio QC checks placed units, missing/empty/short WAVs and unresolved TTS
MOS warnings. `pacing.combined` reports synthesis plus stretching speed and onset
displacement. Use `--force export` once for a legacy cached export; newer exports
record a policy/profile signature so changes only rerun export. See
[measured phase 2 results](docs/subtitle-scheduling-and-placed-qc.md).

For the audited AI tutorials, opt into `--glossary samples/glossary-ai-tutorial.tsv`.
An optional third TSV column `aliases=wrong rendering|another rendering` corrects
known MT wording only when the corresponding English term occurs in the source.
The profile also includes two explicit source-sentence reference translations.
It is intended for AI/programming contexts, where `Transformer` and `code` have
their technical meanings.

### M4 flags (complete)

| Flag | Default | Meaning |
|---|---|---|
| `--qc-transcript` / `--no-qc-transcript` | on | Level-1 transcript QC: classify ASR segments, retry flagged windows once, write a verdict. |
| `--qc-subtitles` / `--no-subtitle-qc` (`--subtitle-qc`) | on | Subtitle standards QC: wrap/re-split over-long or over-CPS Vietnamese cues and count violations. |
| `--qc-translation` / `--no-translation-qc` | on | Level-2 translation QC battery (numbers/entities/terminology/omission/consistency + round-trip) and verdict. |
| `--qc-final` / `--no-qc-final` | on | Final QC gate: aggregate verdicts + output checks into the `quality` block; `fail` exits non-zero. |
| `--mt-engine {envit5,ollama,hunyuan}` | `envit5` | MT engine. `ollama` = contextual translate → reflect → improve with deterministic validators and a per-segment envit5 fallback; `hunyuan` = the **HY-MT1.5-7B** translation specialist served by the same local Ollama server (opt-in, official GGUF). |
| `--ollama-model MODEL` | `qwen2.5:7b` | Ollama model for `--mt-engine ollama` (`ollama pull qwen2.5:7b`). |
| `--ollama-host URL` | `http://127.0.0.1:11434` | Ollama server base URL. |
| `--hunyuan-model MODEL` | `hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K` | Hunyuan GGUF for `--mt-engine hunyuan` (`ollama run hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K`). |
| `--hunyuan-host URL` | `http://127.0.0.1:11434` | Hunyuan server base URL. |
| `--hunyuan-refine` / `--no-hunyuan-refine` | auto (ON for hunyuan) | Translation refine pass (Q-1-T2): still-flagged units get one best-of-`--hunyuan-samples` cycle, kept only when strictly better; non-hunyuan engines never refine. |
| `--hunyuan-refine-max-units N` | `40` | Refine cost guard: at most N still-flagged units refined per run, the rest recorded as budget skips. |
| `--hunyuan-samples K` | `3` | Hunyuan candidates sampled per refined row. |
| `--fusion-model MODEL` | (empty) | Optional Hunyuan-MT-Chimera-7B GGUF tag used to fuse refine candidates (evaluation-only). |
| `--restore-punct` | off | Opt-in punctuation/casing restoration of the **source transcript** before translation. |
| `--qc-tts` / `--no-qc-tts` | on | Per-clip NISQA-TTS MOS QC on the final fitted clips. |
| `--tts-qc-mos-min MOS` | `3.0` | Retry/review a fitted clip below this NISQA MOS. |

**Dependency note.** Per-clip TTS QC needs **`torchmetrics>=1.6.1`** — now a core
dependency. NISQA's other requirements (`librosa`, `requests`) and `torch` are
already installed; the ~1 MB NISQA weights download once on first use into
`~/.torchmetrics/NISQA`. When the scorer is unavailable the layer degrades to
`skipped` counters and never fails a run. The Ollama engine needs **no new
dependency** (stdlib `urllib`); install/pull instructions are in
[LLM translation (Ollama)](#llm-translation-ollama) and
[Hunyuan MT engine (opt-in, M5)](#hunyuan-mt-engine-opt-in-m5). `vidub doctor`
reports `ollama:`, `hunyuan:` and `tts-qc:` lines.

## Install

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Windows (Git Bash)
# source .venv/bin/activate && pip install -e ".[dev]"  # Linux/macOS

./.venv/Scripts/vidub.exe doctor                        # environment check
```

**GPU machine?** The plain install pulls the CPU torch wheel from PyPI. To run
Demucs / envit5 / ViXTTS on an NVIDIA GPU, install the CUDA torch wheel first
(pip extras cannot carry an `--index-url`), then the GPU extra:

```bash
./.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --upgrade
./.venv/Scripts/python.exe -m pip install -e ".[gpu]"
```

See [GPU acceleration](#gpu-acceleration) for details and VRAM notes.

**ffmpeg is required** (ingest, timing, mix, export). Install it with:

```bash
powershell -ExecutionPolicy Bypass -File tools/install_ffmpeg.ps1   # winget
bash tools/install_ffmpeg.sh                                        # Git Bash
# fallback: choco install ffmpeg
```

**URL input?** Downloading YouTube (or any yt-dlp-supported) URLs is opt-in:

```bash
./.venv/Scripts/python.exe -m pip install -e ".[youtube]"
```

See [URL input](#url-input-youtube--friends) for details and caveats.

**Web UI?** The local Gradio UI is opt-in too:

```bash
./.venv/Scripts/python.exe -m pip install -e ".[ui]"
./.venv/Scripts/vidub.exe ui
```

See [`vidub ui`](#vidub-ui--local-web-ui).

### Optional extras

| Extra | Adds | Install |
|---|---|---|
| *(core)* | the CPU pipeline: faster-whisper ASR, envit5 MT, Piper TTS, Demucs mix | `pip install -e .` |
| `gpu` | CTranslate2 CUDA wheels for GPU Whisper (pair with the CUDA torch wheel above) | `pip install -e ".[gpu]"` |
| `youtube` | URL input via yt-dlp (`vidub run <URL>`) | `pip install -e ".[youtube]"` |
| `ui` | the local Gradio web UI (`vidub ui`) | `pip install -e ".[ui]"` |
| `nc-tts` | the opt-in ViXTTS engine (CPML non-commercial weights) | `pip install -e ".[nc-tts]"` |
| `diarize` | pyannote 3.1 speaker diarization (HF-gated models) | `pip install -e ".[diarize]"` |

`vidub doctor` reports whether the optional `youtube` and `ui` extras are
installed, so a missing extra is one command away from being explained.

## Quick start

```bash
./.venv/Scripts/vidub.exe run samples/demo_en.mp4
# -> output/demo_en/dubbed.vi.mp4  (video + Vietnamese audio + soft vi subtitles)

./.venv/Scripts/vidub.exe run samples/demo_en2.mp4   # the second sample clip
```

First run downloads models (~1.5 GB): faster-whisper, `VietAI/envit5-translation`,
the Piper `vi_VN-vais1000-medium` voice, and Demucs `htdemucs`.

## Commands

### `vidub doctor`

Prints python / ffmpeg / ffprobe / nvidia-smi availability, the CUDA-wiring and
GPU-whisper probes, the **ASR presets with their licenses**, the TTS engines with
their licenses, whether the optional **`youtube`** (yt-dlp) and **`ui`** (gradio)
extras are installed, an **`ollama`** line (server version + whether the default
model is pulled), a **`hunyuan`** line (the HY-MT1.5 model + license note), a
**`tts-qc`** line (whether the NISQA scorer is importable), and free disk space. Always exits 0 so it can be used as a
pre-flight check — if the GPU is not being used, this is the first place to look.

### `vidub run <video>`

Runs the whole pipeline and exits 0 only if every required stage finished and
the output video exists.

| Flag | Meaning |
|---|---|
| `--mode {dub,voice-over,subtitles-only,subtitles}` | Output mode (default `dub`). `subtitles` is an alias of `subtitles-only`: it runs ingest+transcribe+translate+export only, keeps the original audio and produces a vietsub video (see [Subtitles only (vietsub)](#subtitles-only-vietsub)). |
| `--src-lang LANG` / `--tgt-lang LANG` | Source / target language (default `en` / `vi`). |
| `--asr-model MODEL` | ASR model. A preset (`cpu-tiny`, `cpu-base`, `cpu-small`, `gpu-large`, `vi-small`, `vi-medium`), a faster-whisper size (`tiny` … `large-v3`), or a Hugging Face repo id. Default `auto` → `large-v3` on GPU, else `base`. |
| `--words` / `--no-words` | Word-level ASR timestamps (default **on**); tighten subtitle cue ends. |
| `--restore-punct` | Opt-in punctuation/casing restoration of the **source transcript** with `oliverguhr/fullstop-punctuation-multilang-large` (~2.2 GB on first use). Already-punctuated segments are left untouched. Default **off** (see [Punctuation restoration](#punctuation-restoration)). |
| `--sentence-units` / `--no-sentence-units` | Sentence-locked units **with unit-level audio and aligned display** (default **on**): ASR rows are split at internal sentence boundaries with **word-edge-locked timestamps**, so `script.json` rows are sentence rows; a unit ends only at a completed sentence or a speaker change (no gap-based ends), and the hard ceiling (150 source words / 45 s) is the only size guard (a cut at the largest gap ≥ 0.6 s is a counted last resort; a unit with no such gap stays whole as `oversized`). A completeness guard re-samples/falls back when a unit translation drops content and the translation-QC battery scores units (`units.json` is the authoritative map). The timing stage synthesizes each multi-member unit **once** from its `units.json` `vi` and places a **single timed row**, then force-aligns that audio into per-member `member_spans`. `--no-sentence-units` restores per-segment translation **and timing** byte-for-byte (see [Sentence-level translation units](#sentence-level-translation-units) and [Unit-level audio (UA-1)](#unit-level-audio-ua-1)). |
| `--isometric` / `--no-isometric` | Translation: compute each sentence unit's character budget from its locked time window, pass it to the MT engine and run **one** bounded budgeted re-translation for still-over-budget units, accepted only when the deterministic predicate says it is genuinely better (default **off** since v0.11.0 — the matched A/B kept clips unchanged 2 → 2 and made `stretched` worse 19 → 27 against a target of ≤ 12; opt in with `--isometric`; sentence-unit path only — a no-op with `--no-sentence-units`). See [Isometric length budgeting (ISO-1)](#isometric-length-budgeting-iso-1). |
| `--isometric-headroom FLOAT` | Translation: fraction of the window's natural-pace capacity a translation may use (default `0.9`; must be in `(0, 1]`). |
| `--isometric-cps FLOAT` | Translation: natural Vietnamese speaking pace in characters per second used by the budget (default `20.0`; must be in `[10, 30]`). |
| `--mt-engine {envit5,ollama,hunyuan}` | Machine-translation engine (default `envit5`). `ollama` runs the contextual `translate → reflect → improve` loop on a local Ollama model; `hunyuan` runs the **HY-MT1.5-7B** translation specialist on the same server — both opt-in, with deterministic validators and a per-segment envit5 fallback (see [LLM translation (Ollama)](#llm-translation-ollama) and [Hunyuan MT engine (opt-in, M5)](#hunyuan-mt-engine-opt-in-m5)). |
| `--ollama-model MODEL` | Ollama model for `--mt-engine ollama` (default `qwen2.5:7b`). |
| `--ollama-host URL` | Ollama server base URL (default `http://127.0.0.1:11434`). |
| `--hunyuan-model MODEL` | Hunyuan GGUF for `--mt-engine hunyuan` (default `hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K`; pull it with `ollama run hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K`). |
| `--hunyuan-host URL` | Hunyuan server base URL (default `http://127.0.0.1:11434`). |
| `--hunyuan-refine` / `--no-hunyuan-refine` | Translation refine pass (Q-1-T2), **default auto — ON for `--mt-engine hunyuan`**, `--no-hunyuan-refine` opts out: after the QC battery, still-flagged units get one best-of-`--hunyuan-samples` cycle, kept only when strictly better. Non-hunyuan engines never refine. |
| `--hunyuan-refine-max-units N` | Refine cost guard: at most N still-flagged units are refined per run, the rest recorded as budget skips (default `40`). |
| `--hunyuan-samples K` | Hunyuan candidates sampled per refined row (default `3`). |
| `--fusion-model MODEL` | Optional Hunyuan-MT-Chimera-7B GGUF tag on the same host, used to fuse `--hunyuan-refine` candidates (no-op when empty; evaluation-only). |
| `--tts-engine {piper,vixtts}` | TTS engine (default `piper`). `vixtts` is natural vi speech + voice cloning but needs `--allow-nc-models`. |
| `--allow-nc-models` | Allow non-commercial model weights (required for `--tts-engine vixtts`, CPML). |
| `--allow-lang-mismatch` | Skip the post-ASR language guard — allow a transcript that looks like the wrong language (e.g. a `vi-*` model on English audio). |
| `--ref-voice WAV` | Reference voice for zero-shot cloning (`vixtts`; default: the sample shipped with the model). |
| `--diarize` | Detect speakers with [pyannote.audio](https://github.com/pyannote/pyannote-audio) 3.1 and give each its own voice. Needs `HF_TOKEN` + accepted gated models. |
| `--clone-speakers` | Clone each diarized speaker's own voice from the separated vocals with ViXTTS. Requires `--diarize --tts-engine vixtts --allow-nc-models`. |
| `--voice-map JSON` | Per-speaker voice overrides, e.g. `{"spk1": "25hours"}`. |
| `--glossary PATH` | TSV of `term<TAB>translation` rows enforced on the MT output after translation (see [Translation glossary](#translation-glossary)). |
| `--pronounce PATH` | TSV of `written<TAB>spoken` rows applied to the **TTS text only**; subtitles/`script.tsv` keep the written form (see [Pronunciation lexicon](#pronunciation-lexicon)). |
| `--resume` / `--no-resume` | Skip stages already `done` whose artifacts exist (default **on**). |
| `--force STAGE[,STAGE]` | Re-run the named stages regardless of resume. |
| `--burn` | Burn subtitles into the video (re-encodes with x264) instead of a soft track. |
| `--sub-pos {bottom,top,center}` | Subtitle position for the `.ass` export and `--burn` (default `bottom`). |
| `--sub-font-size N` | Subtitle font size in points for the `.ass` export and `--burn` (default `24`). |
| `--sub-margin-v N` | Subtitle vertical margin in pixels for the `.ass` export and `--burn` (default per position: `bottom`/`top` `24`, `center` `0`). |
| `--clean` | Wipe the video's cache directory first. |
| `--report` | Print `report.json` to stdout when finished. |
| `--loudness-target LUFS` | EBU R128 integrated loudness target for the mix (default `-16`). |
| `--loudness-tp DBTP` | True-peak ceiling (default `-1.5`). |
| `--loudness-lra LU` | Loudness range (default `11`). |
| `--speed-as-is R` | Timing: leave a segment within this ratio of its window untouched (default `1.10`). |
| `--speed-cap R` | Timing: hardest atempo before re-translation (default `1.20`). |
| `--no-retranslate` | Timing: disable the shortening re-translation retry (over-long segments are clipped instead). |
| `--tail-pad S` | Timing: extra seconds added to the **final** segment's window past the video end so the last sentence is never cut mid-word (default `2.0`). |
| `--tts-rate-fit` / `--no-tts-rate-fit` | Timing: re-synthesize a Piper segment at a fitted `length_scale` (0.85×–1.30× around natural) before stretching/clipping it (default **on**); `--no-tts-rate-fit` restores the single fixed rate. |
| `--recenter` / `--no-recenter` | Timing: shift a fitted clip later into its window's leftover slack (bounded) instead of always starting at the segment start (default **on**). |
| `--align-display` / `--no-align-display` | Timing: force-align each multi-member unit's final audio with the project ASR so every member row gets a real, word-edge-snapped `member_spans` display span (default **on**); `--no-align-display` skips the alignment ASR and records the `split` fallback. |
| `--cache-dir DIR` | Cache root (default `.vidub-cache`). |

Examples:

```bash
vidub run clip.mp4 --mode voice-over --report
vidub run clip.mp4 --asr-model large-v3 --burn
vidub run clip.mp4 --asr-model vi-small          # Vietnamese-source audio only
vidub run clip.mp4 --allow-lang-mismatch         # force past the language guard
vidub run clip.mp4 --asr-model Systran/faster-whisper-large-v3
vidub run clip.mp4 --force translate,mix        # redo translation + mix only
vidub run clip.mp4 --mode subtitles-only        # subtitles/SRT only, no dubbing
```

Every stage also has a standalone entrypoint, e.g.
`python -m vidub.stages.transcribe .vidub-cache/<hash>/job.json`.

### Subtitles only (vietsub)

`--mode subtitles-only` (alias `--mode subtitles`) is the lightest output mode: it
runs only `ingest → transcribe → translate → export` and **keeps the original
audio**, producing a **Vietnamese vietsub video** instead of a dub. The `speak` /
`timing` / `mix` stages are skipped (no TTS, no speaker separation, no loudness
mix) and the TTS license/language guards no longer block the run. Add `--burn` to
render the styled subtitles into the picture rather than a soft track:

```bash
vidub run clip.mp4 --mode subtitles-only            # soft vietsub video
vidub run clip.mp4 --mode subtitles --burn          # hard-burned, styled
```

Placement and size are set with `--sub-pos {bottom,top,center}` (default
`bottom`), `--sub-font-size` (default `24`) and `--sub-margin-v` (default per
position). The full description — what it skips and why, the outputs, the
ASS-vs-legacy alignment numbering behind the position flag, the burn cost and a
measured dub-vs-subtitles-only comparison — is in
[`docs/subtitles-only.md`](docs/subtitles-only.md).

### URL input (YouTube & friends)

`vidub run <URL>` and `vidub batch <URL>` accept any URL yt-dlp supports
(YouTube, direct mp4 links, …) instead of a local file. The media is downloaded
once to `<cache-dir>/downloads/<extractor>-<id>.<ext>` (video height capped at
720p, audio+video merged to mp4 with an mkv fallback), then the normal pipeline
runs on it. The cache directory is keyed on the downloaded **local file**, so
re-running the same URL reuses both the download and every finished stage, and
`report.json` / the manifest gain a `source_url` field.

```bash
vidub run https://www.youtube.com/watch?v=... --report
vidub batch https://www.youtube.com/watch?v=...      # a batch of one
vidub run https://example.com/clip.mp4 --mode subtitles-only --asr-model tiny
```

Requirements and caveats:

- install the optional downloader first: `pip install -e ".[youtube]"` (yt-dlp,
  Unlicense). Without it a URL input fails immediately with that command —
  before any network call.
- videos are capped at 720p to keep runs (and tests) light.
- downloads need working network access; YouTube sometimes blocks automated
  clients (bot checks) — that is an environment limitation, not a ViDub bug.
  For batch, only a **direct URL** is supported (a `.txt` list file is not).

### `vidub batch <folder>`

Dubs every matching video in a folder **sequentially** (one at a time, for
VRAM/memory safety). Each video gets its own cache dir; outputs go to
`output/<video-basename>/` as usual. A single failure is recorded and the batch
continues; Ctrl-C writes a partial summary and returns exit code 130.

| Flag | Meaning |
|---|---|
| `--pattern GLOB` | Which files to process (default `*.mp4`). Non-matching files are listed as `skipped`. |
| `--out-dir DIR` | Where to write `batch-summary.json` (default: the input folder). |
| `--resume-batch` | Skip videos whose summary entry is already `done`. |

`batch` accepts the same pipeline flags as `run` (`--asr-model`, `--tts-engine`,
`--mode`, `--loudness-*`, …). `batch-summary.json` holds per-video
`{input, status, duration, stages, output?|error?}` plus `totals`; the exit code
is 0 when every processed video succeeded, 1 if any failed.

```bash
vidub batch ./clips/
vidub batch ./clips/ --pattern "*.mkv" --out-dir ./reports
vidub batch ./clips/ --resume-batch          # after an interruption
```

### `vidub ui` — local web UI

A **local** Gradio page over the same pipeline (optional extra):

```bash
./.venv/Scripts/python.exe -m pip install -e ".[ui]"
./.venv/Scripts/vidub.exe ui                  # http://127.0.0.1:7860
./.venv/Scripts/vidub.exe ui --port 7899
```

The TTS dropdown also offers **VoxCPM2** as an opt-in Vietnamese voice. Install
its package with `pip install -e ".[tts-voxcpm2]"` and cache the
`openbmb/VoxCPM2` checkpoint locally. The default checkpoint cache is
`eval-cache/quality-rework/voxcpm2-models`; set `VIDUB_VOXCPM2_MODEL_CACHE` or
`VIDUB_VOXCPM2_PYTHON` to use another cache or Python environment. ViDub runs
VoxCPM2 in a separate process and keeps it loaded for the full speech stage.

Upload a video **or** paste a URL, pick the mode/ASR model/TTS engine and any
opt-ins (diarize, clone-speakers, NC models), set the loudness targets and an
optional glossary, then click **Run**. The run happens in a background thread;
the page polls every second and shows:

- the current stage plus per-stage status/timings while it runs;
- the translated script table;
- a per-segment audio player (pick a segment from the dropdown);
- the dubbed video player and a download link;
- the full `report.json`;
- a **Quality results** panel — the aggregate quality verdict + per-level verdicts
  plus the transcript / translation / TTS / subtitles QC counters (including the
  refine summary), read tolerantly so a section shows `not run` until it exists.

**Engines & quality (M4/M5) accordion.** The MT engine dropdown (`envit5`
default, `ollama`, `hunyuan`), the Ollama/Hunyuan model + host, the refine pass
(with sample count) and Chimera fusion model, a **Sentence-level translation
units** checkbox (on by default; untick to translate ASR segments as-is),
source-transcript punctuation restoration, the five QC toggles (transcript /
subtitles / translation / TTS / final gate) and the TTS-QC MOS floor. Changing
the MT engine or the refine settings now re-runs `translate` **and everything
downstream automatically**, so a new engine never silently reuses a cached
script.

**Review/edit spoken and subtitle text.** After a run finishes, open this
accordion and click **Load text choices**. Edit only `spoken_vi` (voice) and
`subtitle_vi` (captions), then save and click **Run** to rebuild the stages
affected by those text changes. The accepted translation and source columns are
read-only; sentence-unit jobs show one row per spoken sentence.

**One-click test presets** sit above **Run**: *Hunyuan engine* selects the opt-in
engine, *refine pass* turns on re-sampling, *Chimera fusion* is **evaluation
only** (the M5 benchmark measured it as a regression), *defaults (envit5)*
restores the default configuration, and *Force full re-run* ticks every stage.
Presets only edit the form — the auto-invalidation above is what re-runs the
pipeline. A **Re-run these stages (force)** control forces selected stages (and
every later stage) for a cached video.

**Ollama pre-flight.** A run that needs Ollama (the `hunyuan` engine, the refine
pass, or the `ollama` engine) probes the selected server before starting. If the
server is unreachable or the model tag is missing, the run **does not start** and
the status shows the reason with the fix commands (`ollama serve`, `ollama run
<model>`); `envit5` with refine off never probes.

It binds to `127.0.0.1` only — `--share` is intentionally not exposed — and
writes to the normal `output/` + cache locations, so UI and CLI runs share state
(and `--resume` semantics). **Cancel** stops cooperatively at the next stage
boundary, keeping finished stages. Without the extra, `vidub ui` exits 1 with
`pip install -e ".[ui]"`.

### Loudness / quality

The mix stage normalizes to EBU R128 (`ffmpeg loudnorm`). Targets are
configurable per run (and per batch):

| Flag | Default | Meaning |
|---|---|---|
| `--loudness-target LUFS` | `-16` | integrated loudness target (`I`) |
| `--loudness-tp DBTP` | `-1.5` | true-peak ceiling (`TP`) |
| `--loudness-lra LU` | `11` | loudness range (`LRA`) |

```bash
vidub run clip.mp4 --loudness-target -14 --loudness-tp -1.0
```

## Output layout

```
output/<video-basename>/
├── dubbed.vi.mp4      # final video (video copied, AAC audio, soft vi subtitles)
│                      # speech-less jobs: no subtitle stream at all
├── subtitles.vi.srt   # Vietnamese subtitle track (one cue per sentence unit on the SA-1 path)
├── subtitles.vi.ass   # the same track with the ViDub style block (position/size/margin)
├── script.tsv         # start \t end \t speaker \t en \t vi, one row per sentence row
└── report.json        # input, source_url, duration, mode, engines, subtitles,
                       # `translation` + `glossary` + `pronunciation` counters, `cleanup`,
                       # `pacing`, `references` lengths, `speech_coverage`,
                       # per-stage status + timings, and a `meta` block
                       # (tool_version, source_url, asr_model/asr_device,
                       # sep_device, mt/tts engines, mode, cloned_speakers,
                       # total_seconds)

.vidub-cache/<video-hash>/     # per-job artifacts (gitignored, wiped by `run --clean`)
├── job.json                   # manifest (single source of truth)
├── audio.wav                  # 16 kHz mono
├── segments.json / draft.srt  # S1 transcribe (the ASR fragment rows)
├── script.json                # S2 translate (sentence rows on the SA-1 path)
├── units.json                 # S2 authoritative unit map (indices/span/vi/guard/battery)
├── glossary.json              # S2 glossary counters (terms/replacements/spliced/enforced/hesitations)
├── pronunciation.json         # S3 TTS lexicon counters (replacements)
├── speech/seg_XXXX[_fit|_re|_clip].wav  # S3 speak + S4 timing
├── speech/unit_XXXX[_fit|_clip].wav     # S4 one synthesis per multi-member unit
├── timed.json                 # S4 timing (window/action/stretch, `members`, `align`+`member_spans` on the SA-1 path)
├── timing.json                # S4 pacing counters (stretched/retranslated/clipped/clips, `align`)
├── references.json            # S1.6 cloning refs + fallback flags (--clone-speakers)
├── references/spkN.wav        # S1.6 per-speaker reference WAVs
├── sep/htdemucs/audio/*.wav   # S5 Demucs stems
└── dub_track.wav              # S5 mix
```

## Pipeline

```text
video ─► S0 ingest ─► S1 transcribe ─┬─► S2 translate ─► S3 speak ─► S4 timing ─► S5 mix ─► S6 export
         ffmpeg       faster-whisper │   envit5           Piper        atempo        Demucs     ffmpeg
         16k mono     (+VAD)        │   En→Vi + glossary vi_VN        + cap 1.20    bed +      mux + SRT
                                      │                               re-translate  loudnorm
                                      └─► S1.5 diarize (optional, --diarize)
                                          pyannote 3.1: speaker → voice map
                                          └─► S1.6 references (optional, --clone-speakers)
                                              Demucs vocals → per-speaker reference WAVs
```

Every stage reads/writes the per-job manifest (`job.json`); a stage whose
artifact already exists is skipped, so runs are crash-resumable.

## Engines (0.10.0 defaults)

| Role | Engine | License |
|---|---|---|
| ASR | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, CPU/GPU autodetect) | MIT |
| ASR (vi) | [PhoWhisper](https://github.com/VinAIResearch/PhoWhisper) `vinai/PhoWhisper-*` | BSD-3-Clause |
| Translation En→Vi | [VietAI/envit5-translation](https://huggingface.co/VietAI/envit5-translation) | Apache-2.0 |
| Translation En→Vi (opt-in) | [Hunyuan HY-MT1.5-7B](https://huggingface.co/tencent/HY-MT1.5-7B) via Ollama (official GGUF) — **opt-in** | Tencent HY Community License (not valid in EU/UK/South Korea) |
| TTS vi | [Piper](https://github.com/rhasspy/piper) `vi_VN-vais1000-medium` (default) | MIT |
| TTS vi (natural) | [ViXTTS](https://huggingface.co/capleaf/viXTTS) + voice cloning — **opt-in** | CPML (non-commercial) |
| Speaker diarization | [pyannote.audio](https://github.com/pyannote/pyannote-audio) `pyannote/speaker-diarization-3.1` — **opt-in** | MIT (code); gated models |
| Source separation | [Demucs](https://github.com/facebookresearch/demucs) `htdemucs` | MIT |
| Subtitles | pysubs2 | MIT |

### ASR engines

`--asr-model` accepts a preset name, a bare faster-whisper size, or any Hugging
Face repo id. Presets:

| Preset | Model | License | Use when |
|---|---|---|---|
| `cpu-tiny` | `tiny` (faster-whisper / CTranslate2) | MIT | fastest CPU smoke tests |
| `cpu-base` | `base` | MIT | CPU default / low memory |
| `cpu-small` | `small` | MIT | best CPU quality at modest memory |
| `gpu-large` | `large-v3` | MIT | best accuracy on a CUDA GPU |
| `vi-small` | `vinai/PhoWhisper-small` | BSD-3-Clause | **Vietnamese-source audio ONLY** — using it on other languages hallucinates (guarded), CPU |
| `vi-medium` | `vinai/PhoWhisper-medium` | BSD-3-Clause | **Vietnamese-source audio ONLY** — using it on other languages hallucinates (guarded), CPU |
| `auto` | `large-v3` if CUDA is usable, else `base` | MIT | default — no need to choose |

> **Warning — `vi-small` / `vi-medium` are Vietnamese-specialist models.** Run
> them **only on Vietnamese-source audio**. On English (or any other language)
> they hallucinate Vietnamese text, which then makes the translator echo the
> input and the dub collapse into long silences. ViDub guards against this: after
> ASR a Vietnamese-looking transcript for a non-`vi` `--src-lang` aborts with an
> actionable error. Use the default `auto`/`gpu-large` model for English audio,
> or pass `--allow-lang-mismatch` to override deliberately.

`--words` (default on) records per-word timestamps in `segments.json` and
tightens each subtitle cue end to the last spoken word.

**PhoWhisper note.** The `vinai/PhoWhisper-*` repos ship PyTorch weights only —
there is no CTranslate2 `model.bin`, so faster-whisper cannot load them. ViDub
runs them through the already-installed **transformers** ASR pipeline and groups
the returned word timestamps into segments. They run on the auto-detected torch
device (**GPU** with a CUDA torch build, else CPU). The upstream model card lists
**BSD-3-Clause** (not MIT).

#### Transcript QC

The transcribe stage runs a default-on **Level-1 transcript QC** pass over the
ASR output. Each segment is classified from faster-whisper's confidence signals
(`compression_ratio`, `avg_logprob`, `no_speech_prob`), cheap text statistics
(repetition loops) and the existing language guard into **hallucination**,
**garbled**, **language-mismatch**, **segmentation-issue** or **ok**;
`hallucination`/`garbled` segments are re-transcribed **once** with changed
settings (beam search + temperature fallback, clipped to the segment window)
and re-classified. Anything still flagged is written to a human-review packet
(`review/transcript/seg-<i>.txt` + a `.wav` snippet) and the job records a
`transcript_verdict` (`pass`/`review`/`fail`) in `job.json`, with the full
machine-readable report in `transcript_qc.json`. Missing signals (the PhoWhisper
path) fall back to text-only statistics. Disable the whole pass with
`--no-qc-transcript`.

#### Punctuation restoration

`--restore-punct` (opt-in, default **off**) inserts periods/commas/question
marks and basic casing into the **source transcript** before translation, using
[`oliverguhr/fullstop-punctuation-multilang-large`](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large)
— an XLM-R token-classification model (~2.2 GB download on first use; it runs on
CPU). Better-punctuated source text improves both machine translation (envit5 /
the [Ollama engine](#llm-translation-ollama)) and sentence-aware subtitle
splitting.

It runs **after** transcript QC and **before** `segments.json`/`draft.srt` are
written, so both the source SRT and the MT input are punctuated; it touches the
**source transcript only**. A **density guard** skips any segment whose
terminal-punctuation density (`. ! ?` per word) already reaches `0.1`, so text
that already looks punctuated is never re-written. `transformers`/`torch` are
already installed, so no extra dependency is needed; with the flag unset the
transcribe stage is byte-identical to before.

```bash
vidub run clip.mp4 --restore-punct --report
vidub run clip.mp4 --restore-punct --force transcribe,translate,export
```

### TTS engines

| Engine | Model | License | Notes |
|---|---|---|---|
| `piper` (**default**) | `vi_VN-vais1000-medium` | MIT | fast, CPU-only, robotic but clear |
| `vixtts` (**opt-in**) | `capleaf/viXTTS` | [CPML](https://coqui.ai/cpml) (non-commercial) | natural vi speech + zero-shot cloning from a 6–30 s sample |

ViXTTS weights are **non-commercial**, so the license guard refuses the engine
unless you opt in:

```bash
vidub run clip.mp4 --tts-engine vixtts                      # refused (non-zero exit)
vidub run clip.mp4 --tts-engine vixtts --allow-nc-models    # opt in
vidub run clip.mp4 --tts-engine vixtts --allow-nc-models --ref-voice me.wav
```

Install the optional engine with `pip install -e ".[nc-tts]"`. ViXTTS calls the
`coqui-tts` API, downloads `capleaf/viXTTS` + `speakers_xtts.pth` on first use,
and caches them in the Hugging Face cache. It runs on the auto-detected torch
device (**GPU** with a CUDA torch build, else CPU) and reports where it ran in
`report.json` / the `_engine_runner` JSON. Plumbing note: `coqui-tts`' XTTS
expects `torchaudio` and (on torch ≥ 2.9) `torchcodec`, neither of which has a
matching wheel for this environment's torch; `vidub/engines/tts/vixtts.py`
registers a small soundfile/librosa `torchaudio` shim instead of adding those
binary dependencies.

**Vendored code note.** `vidub/engines/tts/vixtts_norm.py` vendors the small
Vietnamese text-cleanup step from
[thinhlpg/vixtts-demo](https://github.com/thinhlpg/vixtts-demo) (file-level
attribution, **MPL-2.0**). Only that module is vendored; the rest of ViDub is
MIT, so the default path stays commercially usable.

### Subtitle QC

On the SA-1 path the subtitle builder writes **one cue per sentence unit**
(times from the unit's placed `timed.json` row, text = the unit's `vi`), so a
cue boundary never cuts a sentence; on legacy caches it stays **one cue per
`script.json` row**. The export stage then measures the Vietnamese subtitle track
against a Netflix-derived profile (**≤ 17 CPS**, cue duration **≥ 0.83 s /
≤ 7 s**, **≤ 2 lines**, **≤ 42 chars per line**). Over-long lines are
auto-wrapped at word boundaries and over-CPS cues are auto-split at
sentence/clause boundaries (never truncated); anything that still exceeds the
profile is counted as a violation. The pass touches **only** `subtitles.vi.srt`
— `script.json`, `script.tsv`, the speech audio and the timing are never
modified. Counters (`cues_in` / `cues_out` / `split` / per-violation counts /
`cps_max_observed`) land in a `subtitles_qc` block in `report.json` and in the
`subtitles_qc.json` sidecar. Disable the pass with `--no-subtitle-qc`.

### Language support

ViDub is En→Vi by default, but the binding constraint is the **TTS engine** —
not ASR or MT. ASR accepts any language faster-whisper supports (free-form
`--src-lang`), and envit5 MT is En↔Vi for now (M4 adds pluggable LLM engines).
The target language is validated **fail-fast, before any work**; Vietnamese
(`vi`) is first-class for every engine, so only a non-`vi` target can be refused.

| Stage | Languages | Notes |
|---|---|---|
| ASR (faster-whisper / PhoWhisper) | any whisper language + `vi` specialist | `vi-small`/`vi-medium` are Vietnamese-source only |
| MT (envit5) | En ↔ Vi | M4 adds LLM MT engines |
| TTS (`piper`) | `vi` | single Vietnamese voice set |
| TTS (`vixtts`) | `vi`, `en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, `tr`, `ru`, `nl`, `cs`, `ar`, `zh`, `hu`, `ko`, `ja`, `hi` | XTTS-v2 language list; opt-in (`--allow-nc-models`) |

Unsupported pairs abort with an actionable error naming the pair and the engine:

```bash
vidub run clip.mp4 --tts-engine piper --tgt-lang en
# error: unsupported language pair en->en for --tts-engine 'piper': ...
vidub run clip.mp4 --tts-engine vixtts --allow-nc-models --tgt-lang en   # passes validation
```

### Sentence-level translation units

Translation runs on **sentence-level units by default** (`--sentence-units`,
**on**). The translate stage first splits every ASR row at internal sentence
boundaries (`vidub.utils.sentences.split_into_sentences`), locking each
sentence's `start`/`end` to the **ASR word edges** (interpolating by character
fraction only when a row has no usable `words`). **`script.json` rows are
therefore sentence rows** — the same `{start, end, speaker, en, vi, truncated}`
keys, plus the existing optional flags — and `segments.json` keeps the ASR
fragments untouched.

Consecutive sentence rows are merged into a **sentence-scoped unit** that ends
**only** at a completed sentence (terminal `.` `!` `?` `…`) or a **speaker
change**. There are no gap-based ends and consecutive complete sentences are not
merged; the contextual LLM engines (`--mt-engine ollama` / `hunyuan`) still see
neighbouring units in their template. The single size guard is the hard ceiling
(`HARD_UNIT_WORDS = 150` / `HARD_UNIT_SECONDS = 45.0`): an over-ceiling sentence
with no internal sentence boundary is closed at the largest gap ≥
`PAUSE_CUT_GAP_S = 0.6` s (a counted *pause cut*, the last resort) and otherwise
kept whole and counted `oversized`. No path cuts mid-sentence.

A **completeness guard** (`vidub.utils.units_guard`) then checks each unit's
translation against its source (word coverage < `0.75`, or a lost sentence at
`>= 2` source sentences) and re-samples (best-of-3) or uses a fallback engine
before the deterministic passes (glossary, hesitation cleanup, `vnstyle`, length
guard) run **on the unit text**; the per-row display text is re-derived from the
corrected unit text with `split_translation`. The **translation-QC battery
scores units** (one `translation_qc.json` `segments` entry per unit, with
`members`), and the authoritative map — members, locked span, final `vi`,
guard/battery status — is written to **`units.json`**. The translate stage logs a
`sentence units: <n> units from <m> sentence rows (<k> merged)` summary.

Disable with `--no-sentence-units` (or untick **Sentence-level translation
units** in the `vidub ui` Engines & quality panel) to translate each ASR segment
as-is — the previous per-segment behavior, byte-for-byte.

### Isometric length budgeting (ISO-1)

With sentence units on, the translate stage can also compute each unit's
**character budget** from its locked time window (`window × chars-per-second ×
headroom`, floored at 8 and capped at 600 chars; defaults `--isometric-cps 20.0`,
`--isometric-headroom 0.9`) and pass it to the MT engine as an optional hint
(envit5 steers its decode; hunyuan/ollama add the proven *"at most N characters"*
clause). A unit whose translation still exceeds the budget gets **one** bounded
budgeted re-translation, which replaces the text **only** when the deterministic
`accept_shrunk` predicate says it is genuinely better (validators pass, no
omission/sentence-loss, no neighbour bleed, not longer, lower projected stretch);
otherwise the original is kept and recorded. `units.json` gains
`budget_chars` / `chars` / `over_budget` / `isometric` / `chars_after` /
`projected_stretch_before` / `projected_stretch_after`, and
`translation_qc.json` `units.flags` gains `over_budget_after` /
`isometric_shrunk` / `isometric_kept`. The pass is **off by default** since
v0.11.0 (the matched A/B kept clips unchanged 2 → 2 and made `stretched` worse
19 → 27 against a target of ≤ 12); enable it with `--isometric`. It is a no-op
with `--no-sentence-units`, and a change to any isometric knob re-runs translate
via `mt_signature`. The measured A/B (including the honest NOT-MET verdict) is in
[`docs/isometric-budgeting.md`](docs/isometric-budgeting.md).

### Unit-level audio (UA-1)

By default the **timing** stage fits **sentence units**, not fragments. It uses
the translate stage's authoritative **`units.json`** when it is present and
consistent (members, locked span and synthesis `vi` verbatim) and otherwise
re-groups the script rows with `group_units`. Each multi-segment unit is
synthesized **once** into `speech/unit_XXXX.wav` and fitted across the unit's
window (the first member's start up to the next unit's first member start, or the
padded video end for the last unit); the stage writes **one `timed.json` row per
unit** carrying `members`. An intra-sentence pause is therefore **structurally
impossible**: no placed-row boundary can fall inside a unit, so the
`…đào ⏸ tạo…` split cannot reappear. Single-segment units are the degenerate case
and reuse the member's pass-1 WAV (and its existing rate/re-translation plan).

**Aligned display spans.** After the final audio exists, the timing stage
transcribes each multi-member unit's WAV with the project ASR
(`--align-display`, default **on**), allocates the ASR word timeline to the
members (`vidub.utils.align.allocate_spans`) and writes `member_spans` +
`align` on the row: `asr` (word-edge-snapped spans), `split` (the proportional
`split_translation` fallback over the members' locked script spans),
`single` (one member) or `unavailable`. The ASR is built lazily and any
ASR/parse error degrades that unit to the fallback — never the run; the summary
counts are in `timing.json`'s `align` block. `--no-align-display` skips the
alignment ASR entirely.

Consumers handle the shape: the per-clip **TTS QC** scores each unit WAV and
retries it with the unit's text; `compute_coverage` aggregates over the unit's
**member spans**; the web UI's segment player maps **every** member segment to
its unit's audio. A multi-segment unit is never re-translated (it keeps the
`units.json` text); an over-long unit is stretched up to the cap and, as a last
resort, clipped with a reason. `--no-sentence-units` restores the per-segment
timing path byte-for-byte (no `units.json`, no `align`/`member_spans`).

The reference A/B, the pre-registered targets and the honest verdict — including
the M4/M5 misses — are in
[`docs/sentence-locked-units.md`](docs/sentence-locked-units.md). The earlier
UA-1 timing A/B is in [`docs/unit-audio.md`](docs/unit-audio.md).

### Translation QC

The translate stage runs a default-on **Level-2 translation QC battery**
(QUALITY-ROADMAP.md §3.3) over the fully built script rows. Every non-passthrough
segment is scored on independent deterministic signals: **numbers** (each source
number's digits must survive, separator-aware), **entities** (capitalized
mid-sentence tokens), **terminology** (glossed terms must render as their
enforced target), an **omission/addition** length-ratio proxy, cross-segment
**consistency** (a repeated source phrase rendered divergently), and an
auxiliary En→Vi→En **round-trip** fidelity check (never a hard failure). The
worst present class labels the row (`number-error` / `entity-error` /
`terminology-error` / `omission` / `addition` / `inconsistent` /
`mistranslation-risk` / `unnatural` / `ok`).

Corrections only ever go through existing helpers: dropped mid-sentence entities
and digit-free missing numbers get a deterministic glossary **splice**, missing
glossary terms are re-enforced, and any remaining flag gets **one**
re-translation retry (the better re-classification is kept). Still-flagged rows
are written to a `review/translation/seg-<i>.txt` packet and the job records a
`translation_verdict` (`pass` / `review` / `fail`) in `job.json`, with the full
machine-readable report in `translation_qc.json`. Passthrough rows (already
user-language) are never touched. Disable the battery with
`--no-translation-qc`.

On the SA-1 path the battery scores **units**: `translation_qc.json`'s
`segments` has **one entry per sentence unit** (each carrying `members`) and its
`units` block keeps `mode` / `units` / `segments` / `merged_segments` /
`max_unit_words` / `max_unit_seconds` and adds `detail` (the per-unit guard +
battery records) and `flags` (`guard_flagged` / `omission` / `sentence_loss` /
`addition` / `oversized`). The timing stage's `timing.json` adds an `align` block
(`asr` / `split` / `single` / `unavailable`) and each `timed.json` row carries
`align` + `member_spans`.

### Final QC gate

Export ends with a default-on **final QC gate** (QUALITY-ROADMAP.md §3.4) that
aggregates the per-level verdicts and adds fresh checks of the finished output
into one `quality` block in `report.json`: `verdict` (`pass` / `review` /
`fail` / `unknown`), `levels` (the `transcript` / `translation` / `subtitles`
verdicts read from their sidecars), `audio` (`duration_delta`, `loudness_i`,
`loudness_delta`, `silent_rows`, `clipped_rows`), and `review_packets` (an index
of the packet files under `<cache>/<hash>/review/`).

* **fail** — any level verdict `fail`, **or** more than 20 % of speech rows
  silent, **or** a measured loudness delta beyond ±5 LU. A `fail` makes
  `vidub run` exit non-zero with the reasons printed to stderr.
* **review** — any level verdict `review`, any silent or clipped speech row, a
  loudness delta beyond ±2 LU, an A/V duration delta over 2 s, or any remaining
  subtitle-profile violation.
* **unknown** — nothing measurable (old caches with no QC sidecars); treated as
  a pass for the exit code, so a missing artifact can never fail a run.

Disable the gate with `--no-qc-final` (the block then reads
`{"enabled": false}`).

### Translation glossary

`--glossary PATH` enforces fixed terminology on the machine translation. envit5
cannot take prompt/terminology hints, so enforcement has two mechanisms:

* **output-side** — every whole-word, case-insensitive occurrence of a *source*
  term in the MT output is replaced with its translation;
* **source-side (new in M3-C1)** — when a segment's **source** text contains a
  glossed term but the MT output does **not** contain the enforced phrase, the
  source is split at the term, the surrounding chunks are re-translated and the
  result is joined around the enforced phrase (up to 2 terms per segment). This
  is what makes `AI assistant -> trợ lý AI` stick even when the model would
  phrase it differently. A rebuilt row is flagged `"glossary_spliced": true`.

Both run **before** the 2.2x length guard and the low-confidence check.
Already-Vietnamese passthrough segments are never touched. `report.json` gains a
`glossary` block: `terms` (rows loaded), `replacements` (output-side),
`spliced` (source-side rebuilds) and `enforced` (source-side enforcements found,
rebuilt *or* already present).

The file is a UTF-8 TSV, one `term<TAB>translation` per row:

```text
# lines starting with '#' are comments; blank lines are ignored
AI assistant	trợ lý AI
large language model	mô hình ngôn ngữ lớn
neural network	mạng nơ-ron
parameters	tham số
training data	dữ liệu huấn luyện
supercomputer	siêu máy tính
probability	xác suất
GPU	GPU
VRAM	VRAM
graphics card	card màn hình
peasants	anh em cấu hình thấp
peasant	anh em cấu hình thấp
Riddle me this	Đố anh em biết
```

`Vigib -> ViDub`, `AI assistant -> trợ lý AI`, `large language model ->
mô hình ngôn ngữ lớn`, `neural network -> mạng nơ-ron`, `parameters -> tham số`,
`training data -> dữ liệu huấn luyện`, `supercomputer -> siêu máy tính`,
`probability -> xác suất`, `graphics card -> card màn hình`, `peasants` /
`peasant -> anh em cấu hình thấp` and `Riddle me this -> Đố anh em biết` are the
bundled example terms. The identity rows `GPU -> GPU` and `VRAM -> VRAM` keep
those acronyms **untranslated** in the text via source-side enforcement (pair
them with a [pronunciation lexicon](#pronunciation-lexicon) entry so the TTS
still says them out loud readably). **Extend the file per video** with the terms
*your* video uses that the model tends to phrase differently.

Longer terms win over shorter prefixes (`large language model` is matched before
`model`, `AI assistant` before `AI`). **Source-side matching is plural-tolerant**
(new in M3-C3): a term also matches its English plural, so `GPU` finds `GPUs` and
`peasant` finds `peasants`; the singular enforced phrase is what gets spliced in.
The output-side replace stays exact, so `GPUs` is never half-replaced. Invalid
rows (no tab, empty term, or empty translation) are skipped with a warning
instead of aborting the run.

```bash
vidub run clip.mp4 --glossary glossary.tsv --force translate,export
```

### LLM translation (Ollama)

`--mt-engine ollama` swaps the per-segment envit5 translator for a **local LLM**
running under [Ollama](https://ollama.com), using the contextual
**translate → reflect → improve** pattern (QUALITY-ROADMAP.md P1.5 / §3.3). The
model sees its neighbouring segments and the terminology table, drafts the whole
batch, critiques it, and rewrites only the segments the critique marks as major
issues — so pronouns, tense and terminology stay coherent across sentences.

ViDub talks to Ollama over its HTTP API with the standard library only
(`urllib.request`); there is **no extra dependency** and no `ollama`/`requests`
package. Install Ollama and pull the model once:

```bash
winget install --id Ollama.Ollama -e   # Windows; then restart the shell
ollama serve                           # usually already running as a service
ollama pull qwen2.5:7b                 # the default --ollama-model
```

Then run with the engine:

```bash
vidub run clip.mp4 --mt-engine ollama --report
vidub run clip.mp4 --mt-engine ollama --ollama-model qwen2.5:7b --ollama-host http://127.0.0.1:11434
```

| Flag | Default | Meaning |
|---|---|---|
| `--mt-engine {envit5,ollama,hunyuan}` | `envit5` | MT engine. `ollama` = contextual LLM; `hunyuan` = HY-MT1.5-7B specialist. |
| `--ollama-model MODEL` | `qwen2.5:7b` | Ollama model to use. |
| `--ollama-host URL` | `http://127.0.0.1:11434` | Ollama server base URL. |
| `--hunyuan-model MODEL` | `hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K` | Hunyuan GGUF for `--mt-engine hunyuan`. |
| `--hunyuan-host URL` | `http://127.0.0.1:11434` | Hunyuan server base URL. |
| `--hunyuan-refine` / `--no-hunyuan-refine` | auto (ON for hunyuan) | Refine pass: one best-of-`--hunyuan-samples` cycle over still-flagged units; `--no-hunyuan-refine` opts out. |
| `--hunyuan-refine-max-units N` | `40` | Refine cost guard: at most N still-flagged units refined per run. |
| `--hunyuan-samples K` | `3` | Hunyuan candidates sampled per refined row. |
| `--fusion-model MODEL` | (empty) | Optional Chimera tag to fuse refine candidates (evaluation-only). |

**The LLM only ever *proposes*.** Every candidate output — the initial draft, the
improved rewrite and the single targeted retry — is checked by deterministic
validators before it is accepted:

* **language** — a Vietnamese output must reach the 0.30 diacritic ratio
  (`vidub.utils.langcheck.vn_diacritic_ratio`); skipped when the source is `vi`;
* **numbers** — every digit-run in the source must survive in the translation;
* **length** — at most 2.2x the source word count.

A segment that fails validation gets **exactly one targeted re-ask** (the failure
reason is put back into the prompt); if it still fails it falls back to the
**envit5 result for that segment**, so a bad suggestion can never reach the TTS.
The counters `llm_used` / `fallback_used` / `validator_retries` are tracked on the
engine. If the Ollama server is unreachable the engine raises an actionable
`TranslationError` ("is Ollama running? start it with 'ollama serve'") naming the
host; the plain `envit5` default path is unchanged and byte-identical.

`vidub doctor` prints one **`ollama`** line and one **`hunyuan`** line, e.g.
`ollama: 0.34.4 at http://127.0.0.1:11434 (model qwen2.5:7b: present)` or
`ollama: not reachable (start it with 'ollama serve')`; the `hunyuan` line names
the HY-MT1.5 model + license note. Neither raises and the doctor always exits 0.

### Hunyuan MT engine (opt-in, M5)

`--mt-engine hunyuan` swaps the per-segment envit5 translator for Tencent
**HY-MT1.5-7B**, a *translation-specialist* LLM served by the same local
[Ollama](https://ollama.com) server over the standard-library HTTP path (no new
dependency). It runs the official GGUF `hf.co/tencent/HY-MT1.5-7B-GGUF` — `Q6_K`
by default, `Q4_K_M` as the VRAM contingency. ViDub ships no weights, so pull it
yourself once:

```bash
ollama run hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K
```

Then run with the engine (flags: `--hunyuan-model`, `--hunyuan-host`,
`--hunyuan-refine` / `--no-hunyuan-refine`, `--hunyuan-samples`, `--fusion-model`):

```bash
vidub run clip.mp4 --mt-engine hunyuan --report
vidub run clip.mp4 --mt-engine hunyuan --hunyuan-model hf.co/tencent/HY-MT1.5-7B-GGUF:Q4_K_M
```

**The LLM only ever *proposes*.** Exactly like the Ollama engine, every output —
draft, rewrite and the single targeted retry — is checked by the deterministic
validators (language, numbers, length); a segment that still fails falls back to
its **envit5 result**, so the shipped `envit5` default path is unchanged and
byte-identical without the flag. The `--hunyuan-refine` pass is **on by default
for this engine** (`cfg.hunyuan_refine` is `None` = auto): it re-samples only the
units the QC battery still flags (best-of-`--hunyuan-samples`, default 3, plus
the envit5 candidate, bounded by `--hunyuan-refine-max-units`, default 40) and
accepts a candidate only when it is **strictly better** under the same
validators; `--no-hunyuan-refine` opts out and `--fusion-model` (Chimera) stays
evaluation-only.

**License.** The weights are under the **Tencent HY Community License**: free
below 100 M MAU but **not valid in the EU/UK/South Korea**, with an AUP and a
no-distillation clause (do not use the outputs to train other models) — so
Hunyuan stays **opt-in** and `envit5` remains the default engine. `vidub doctor`
prints the license note on its `hunyuan:` line.

**Benchmark.** `docs/hunyuan-evaluation.md` (250 En→Vi segments): translation
quality comparable to envit5 (**NOT promoted** — it misses every §7.6 criterion);
the **refine pass is the measured win** (−16 % battery flags on the default path,
`--hunyuan-refine`); Chimera fusion is evaluated-only.

### Pronunciation lexicon

`--pronounce PATH` fixes how the TTS **speaks** a term without changing how it is
**written**. Acronyms and brand names often reach the engine as bare letters
(`VRAM` read as "vê-ram" or garbled, `ChatGPT` spelled out letter-by-letter); the
lexicon maps the written form to the phonetic form handed to the engine:

```text
# lines starting with '#' are comments; blank lines are ignored
VRAM	Vê-ram
GPU	Gờ-Pê-U
ChatGPT	Chét-Gê-Pê-Tê
CPU	Xê-Pê-U
LLM	Lờ-Lờ-M
```

These five rows are the bundled example lexicon (`samples/pronounce.example.tsv`)
— **extend it per video**. The switch is **TTS-only**: `script.json`,
`subtitles.vi.srt` and `script.tsv` always keep the written form, so subtitles
stay readable while the audio says the phonetic form. It is applied to the normal
speak pass *and* to the timing stage's shortening re-synthesis, so a retried
segment is not spoken with a different reading. `report.json` gains a
`pronunciation` block counting how many written forms were swapped (`{
"replacements": n }`; `0` when no lexicon is supplied). Matching is
whole-word/case-insensitive and longer written forms win (`ChatGPT` before `GPT`).

```bash
vidub run clip.mp4 --pronounce pronounce.tsv --force speak,timing,export
```

Pair an identity glossary row with a lexicon row when an acronym should stay
written but be spoken readably:

```text
# glossary.tsv          # pronounce.tsv
GPU<tab>GPU             GPU<tab>Gờ-Pê-U
```

### Dubbing pacing (timing fit)

S4 timing fits every spoken **unit** into its **availability window** — the time
from the unit's start to the next unit's start (or the video end for the last
one), so a unit can use the following silence instead of being squeezed into its
own slot. With `--sentence-units` on (default) a unit is a whole sentence, so a
placed row may span several ASR segments; with `--no-sentence-units` every row is
a single segment (see [Unit-level audio (UA-1)](#unit-level-audio-ua-1)). The
policy (thresholds in `vidub/utils/fit.py`):

| Ratio (`speech / window`) | Action |
|---|---|
| `<= --speed-as-is` (default **1.10**) | left untouched (byte-identical) |
| `<= --speed-cap` (default **1.20**) | `atempo` time-stretch to fit |
| `> --speed-cap` | **one re-translation retry**, then clip (last resort) |

The **re-translation retry** strips English fillers from the source, re-runs the
translator and re-synthesises only that row; if it now fits it is placed
normally. It applies to **single-segment** rows: a multi-segment sentence unit is
never re-translated (it keeps the joined member text) and goes straight to a
stretch (≤ cap) and, if needed, a clip. If a row still cannot fit, it is
stretched to the cap and **trimmed at the window end** (`atempo` + `atrim`) with a
WARNING — it never bleeds into the next unit.

**Tail safety (M3-C2).** A trim is not a blind cut. The stretched segment's tail
is analysed with ffmpeg `silencedetect`; when a pause starts at/after
`window - 0.6 s` (or straddles the window) the cut lands **in the pause**
(`"smart_trim": true`) so no sentence ends mid-word. When no pause is found the
cut is a **blind clip** flagged `"clipped"` with a `reason`. Every cut gets a
60 ms fade-out, so it never clicks. Each clip is listed in the report's
`pacing.clips` with its `idx`, `start`, `seconds_trimmed`, `reason` and a
`smart_trim` flag; the `pacing` block also counts `smart_trims`. The **final**
segment's window additionally extends `--tail-pad` seconds (default **2.0**)
past the video end — nothing follows it, so a mild overrun past the last frame is
acceptable mux behavior and the last sentence is never cut mid-word.
`--no-retranslate` turns over-long segments straight into clips.

Stretches are smoothed across neighbours (`max_neighbor_delta` 0.15) but
smoothing may only choose a stretch that still **fits** the window (it never
goes below the needed `speech / window`). Fitted audio is trimmed at the window
end and a fitting segment that would still overrun is upgraded to a minimal
stretch, so the invariant is strictly **zero bleed**: no unit's audio extends
past the next unit's start (or, for the last unit, past the padded video end).
The `pacing` report block shows `stretched` / `retranslated` / `clipped` /
`smart_trims` / `max_stretch` / `max_neighbor_delta`; `report.json`'s `cleanup`
block shows how many hesitation artifacts (`hesitations`), English fillers
(`filler_removals`) and Vietnamese word-order refinements (`style_fixes`) were
applied.

```bash
vidub run clip.mp4 --speed-cap 1.25 --report
vidub run clip.mp4 --no-retranslate            # clip instead of re-translate
vidub run clip.mp4 --tail-pad 3                # give the last sentence 3 s more
```

### Two-pass rate fitting (FIT-1)

**2026-10-01 correction:** the active policy preserves natural pace whenever
speech fits. It no longer slows short lines to fill silence or derives a global
rate offset from fragmented display rows. Overflow alone requests faster
synthesis (up to 1.30×); multi-row units are measured at natural pace before
the same fit policy is applied. The `global_length_scale` report field is now
neutral (1.0). Timing retries must improve measured duration; concise retries
also pass completeness checks, and accepted wording follows the audio into
subtitles and TTS-QC. See [research and matched measurements](docs/natural-pacing.md).

The following describes the historical FIT-1 experiment, before that correction:

The previous pacing policy synthesised **every** segment at one fixed rate and
then papered over the mismatch with `atempo` (≤ 1.20×) or a clip, which the User
heard as alternating fast/slow speech plus dead air. FIT-1 adds a **rate pass
before** the stretch/retranslate/clip logic (Piper only; **default on**):

1. **Planning pre-pass.** For every row the stage computes its availability
   `window` and the natural take's duration, derives the ratio `window / natural`
   and takes the video-wide baseline `pace_base` = clamped median of those ratios
   (1.0–1.10).
2. **Per-row re-synthesis.** When the pure policy
   ([`vidub/utils/fit.py`](vidub/utils/fit.py), `plan_rate_fit`) says a row is
   worth re-synthesizing, its Vietnamese text is re-synthesized once via Piper's
   per-synthesis `length_scale` — **this runs first**, before the re-translation
   retry (it is cheaper and preserves the translation). The rate is clamped to
   **0.85×–1.30×** around natural (`length_scale`, > 1 = slower, < 1 = faster),
   divided by the baseline pace. On success the new take flows through the
   unchanged `asis` / `stretch` / `shorten` logic, so a clip only survives when
   even the clamped rate cannot fit; on failure the natural take is kept.
3. **Bounded re-centering.** Leftover window slack is used by shifting a fitted
   clip later — half the slack, capped at 2 s, only when the slack exceeds 1.5 s —
   and never past the window end (the no-bleed invariant still holds).

With `--sentence-units` on (default) the plan runs per **sentence unit**: a
multi-segment unit has no natural take yet, so it is synthesized **once** at the
video-wide `pace_base` into `speech/unit_XXXX.wav` and the same atempo/re-center
machinery micro-fits it against the unit window (see
[Unit-level audio (UA-1)](#unit-level-audio-ua-1)). The unit's timed row records
its `length_scale` / `rate_action` (`resynth` when the baseline scale was
applied, else `off`) and its `members`.

The timed rows gain `length_scale` / `rate_action` (`keep` / `resynth` /
`failed` / `off`) and `recentered` / `recenter_seconds`; `timing.json` (and
`report.json`'s `pacing`) gain a `rate_fit` block
(`resynthesized` / `kept` / `failed` / `global_length_scale` / `recentered`).
Both the rate pass and re-centering are **off** with `--no-tts-rate-fit` /
`--no-recenter`, which reproduces the previous timing stage byte-for-byte. The
reference A/B, the pre-registered acceptance and the honest verdict (including
where targets were not met) are in
[`docs/timing-fitting.md`](docs/timing-fitting.md).

Two standing caps apply regardless: the atempo cap (`--speed-cap`, 1.20) and the
single re-translation retry above it. ViXTTS keeps the atempo path — the
per-synthesis rate knob is Piper-specific.

### Per-clip TTS QC

The timing stage ends with a default-on **per-clip TTS QC** pass
(QUALITY-ROADMAP.md §3.4 / P1.7): **every FINAL fitted clip** — the exact audio
that goes into the mix, after atempo/trim — is scored with **NISQA v2.0**, a
non-intrusive MOS model (1–5, higher is better) run through the TorchMetrics
wrapper. A garbled, silent or low-naturalness synthesis therefore cannot reach
the mix unnoticed.

* A clip scoring **< `--tts-qc-mos-min`** (default **3.0**) gets **one retry**
  with changed generation settings: `vixtts` re-samples with
  `temperature 0.65 / top_k 80`; `piper` re-synthesizes with the **other shipped
  voice** (`25hours` when the active voice is `vais1000`, else `vais1000`). The
  retry is re-fitted to the same window with the original stretch and re-scored;
  whichever of the two clips scores higher is kept (a tie keeps the original).
* A clip still below the threshold is written to a human-review packet
  (`review/tts/seg-<i>.txt` with both MOS values + `seg-<i>.wav`).

Counters land in the `tts_qc.json` sidecar (and `manifest.paths["tts_qc"]`):
`{enabled, mos_min, scored, low_mos, retried, resolved, skipped, mos_mean}` —
`scored` counts how many clips were scored, `resolved` how many low clips were
lifted back above the threshold. When the scorer (or its weights) is unavailable
the gate **degrades to `skipped` counters and never crashes the run**.

NISQA comes from the **`torchmetrics>=1.6.1`** core dependency (the same version
`pyannote-audio` requires); `librosa`/`requests` — its other requirements — are
installed with the other extras, so the only extra download is the **~1 MB NISQA
weights** fetched once on first use into `~/.torchmetrics/NISQA`. `vidub doctor`
prints a `tts-qc:` line showing whether NISQA is importable. Disable the whole
pass with `--no-qc-tts`.

```bash
vidub run clip.mp4 --tts-qc-mos-min 3.5 --report
vidub run clip.mp4 --no-qc-tts                 # skip per-clip TTS scoring
```

### Diarization (optional)

`--diarize` runs [pyannote.audio](https://github.com/pyannote/pyannote-audio)
3.1 between transcribe and translate, labels each segment with its dominant
speaker (`spk0`, `spk1`, … by descending talk time) and gives each speaker its
own voice.

The pyannote models are **gated**: you need a Hugging Face token *and* to have
accepted the model conditions.

```bash
pip install -e ".[diarize]"          # optional extra
export HF_TOKEN=hf_...               # or: huggingface-cli login
# accept: https://hf.co/pyannote/speaker-diarization-3.1
#         https://hf.co/pyannote/segmentation-3.0
vidub run clip.mp4 --diarize --report
```

Without a token/access the run fails **before** any synthesis with those setup
steps and exit code 1. Default voice assignment:

| Speaker | Piper voice |
|---|---|
| `spk0` | `vi_VN-vais1000-medium` (female) |
| `spk1` | `vi_VN-25hours_single-low` (male) |
| other | `spk0`'s voice |

Override speakers with a JSON map — for Piper the value is a voice name, for
ViXTTS a reference-WAV path:

```bash
vidub run clip.mp4 --diarize --voice-map '{"spk1": "25hours", "spk2": "vais1000"}'
```

Without `--diarize` the stage is skipped entirely (the report shows
`"diarize": {"status": "skipped", "reason": "not requested"}`) and the default
single-speaker `spk0` path is unchanged. Note: `pyannote.audio` depends on
`torchaudio`/`torchcodec`; install wheels that match your torch build.

M2 (v0.2.0) shipped PhoWhisper (vi ASR, `--asr-model vi-small`), word-level
timestamps, the opt-in ViXTTS engine (`--tts-engine vixtts`), optional pyannote
diarization (`--diarize`), GPU Whisper, batch mode, configurable loudness and the
`report.json` `meta` block. **M3 (v0.3.0)** adds the CUDA-torch GPU pipeline
(Demucs/envit5/ViXTTS on GPU), the translation glossary (`--glossary`), the TTS
language matrix, YouTube URL input (`vidub run <URL>`), per-speaker voice cloning
(`--clone-speakers`) and the local web UI (`vidub ui`). **M4 (v0.4.0)** added the
five-layer quality-control framework (transcript / subtitle / translation /
per-clip TTS QC + a final aggregate gate), the opt-in Ollama LLM MT engine and
source-transcript punctuation restoration. **M5 (v0.5.0)** adds the opt-in
Hunyuan HY-MT1.5-7B engine (`--mt-engine hunyuan`), the translation refine pass
(`--hunyuan-refine`, optional Chimera fusion) and the pre-registered En→Vi
benchmark (`docs/hunyuan-evaluation.md`) whose verdict keeps **`envit5` as the
default** and recommends the refine pass on the default path. **v0.5.1** finishes
the local web UI — the M4/M5 engine + QC controls, one-click test presets, a
force/re-run control, a **Quality results** panel and an Ollama pre-flight that
blocks a run with fix commands — and makes a changed MT configuration invalidate
the cached `translate` stage and everything downstream automatically. **v0.6.0**
adds **sentence-level translation units** (default on): ASR fragment runs are
merged into whole sentences before translation and the translation is split back
onto the original segment windows, keeping `script.json` 1:1 with the audio
segments (`--no-sentence-units` restores per-segment translation), with the unit
statistics surfaced in `translation_qc.json` and the web UI. **v0.7.0** adds
**two-pass TTS rate fitting** (default on, Piper): a segment that would be
squeezed or clipped is first re-synthesized at a fitted `length_scale`
(0.85×–1.30× around natural, plus a video-wide pace offset), leftover window
slack is used by bounded re-centering, and the previous single-rate behaviour is
restored with `--no-tts-rate-fit` / `--no-recenter` (reference A/B and the honest
verdict in `docs/timing-fitting.md`). **v0.8.0** adds **unit-level audio**
(default on): the timing stage synthesizes each multi-segment sentence unit
**once** from its joined member text and places a **single timed row** across
the unit's span, so an intra-sentence pause (`…đào ⏸ tạo…`) is structurally
impossible and no placed-row boundary can fall inside a sentence; single-segment
units reuse the pass-1 WAV, subtitles stay per segment, and the previous
per-segment timing is restored with `--no-sentence-units` (reference A/B and the
honest verdict in `docs/unit-audio.md`). **v0.9.0** turns the **sentence** into
the row unit: ASR rows are split at internal sentence boundaries with
**word-edge-locked timestamps**, units are **sentence-scoped** (a unit ends only
at a completed sentence or a speaker change; the hard ceiling is the only size
guard and an oversized unit stays whole), a new **completeness guard** re-samples
or falls back when a unit translation drops content, the **translation-QC
battery scores units** (authoritative map `units.json`), the timing stage
**force-aligns** each unit's audio into per-member `member_spans`
(`--align-display`, default on) and subtitles become **one cue per sentence
unit**; `--no-sentence-units` stays byte-identical (measured A/B and the honest
M4/M5 misses in
[`docs/sentence-locked-units.md`](docs/sentence-locked-units.md)). **v0.10.0**
closes the Q-1 **quality round** and adds the SU-1 **vietsub mode**: (1) a unit
**contamination guard** that flags hunyuan neighbour-context bleed or a sentence
addition and repairs the unit with a direct, context-free template
(`action: "decontext"`); (2) the translation **refine pass is now ON by default
for `--mt-engine hunyuan`** (tri-state `hunyuan_refine`, budget
`--hunyuan-refine-max-units` 40, `--no-hunyuan-refine` opts out); (3) **pacing
levers** — a concise re-translation, then a bounded short-window stretch
(`SPEED_CAP_SHORT` 1.30 up to a 1.8 s window), then the minimal trim, every clip
recording `{idx, seconds, reason}` with zero bleed; (4) subtitle cues that split
on the placed audio's **ASR word edges** (`align_words`); and (5) `--mode
subtitles`, an alias of `--mode subtitles-only`, with `--sub-pos` /
`--sub-font-size` / `--sub-margin-v`, a real `.ass` export and a styled burn
filter. The round's measured results — three pre-registered targets **NOT met** —
are in [`docs/quality-round-q1.md`](docs/quality-round-q1.md); the vietsub mode
is in [`docs/subtitles-only.md`](docs/subtitles-only.md).

## Clone the original speakers

`--clone-speakers` gives every diarized speaker a voice resembling their own by
extracting a clean reference from the job's own audio and feeding it to ViXTTS
zero-shot cloning. It is the M3 alternative to integrating RVC (see the note
below).

```bash
vidub run clip.mp4 --diarize --tts-engine vixtts --allow-nc-models --clone-speakers
```

Requirements (validated **fail-fast**, before any work): `--clone-speakers`
needs `--diarize` (speakers must exist) **and** the zero-shot cloning engine
`--tts-engine vixtts --allow-nc-models`. Any other combination aborts with a
message listing the missing flags.

How the references are built (stage `S1.6 references`, between diarize and
translate):

- for each speaker found in `segments.json`, the **Demucs `vocals.wav`** stem is
  cut at that speaker's diarized turns;
- up to **3 of the longest turns** (each ≥ 1.5 s) are concatenated into a
  reference of at most **20 s**, written to
  `<cache>/references/spkN.wav` (16-bit PCM, native rate);
- a speaker with **less than 3 s** of usable speech keeps the **engine default**
  reference and is recorded with `fallback: true` in `references.json` (it is not
  counted as cloned);
- extraction is **deterministic per job** (same turns → same cuts) and cached: a
  re-run reuses an unchanged reference, and a new job hash gets a fresh cache dir.

The `speak` stage then feeds each speaker's reference to ViXTTS;
`report.json` lists the per-speaker reference lengths under `references` and the
number of actually-cloned speakers as `meta.cloned_speakers`.

> **RVC note (M4).** This is **zero-shot** cloning — ViXTTS conditions on the
> extracted reference and synthesises in a similar voice. True **timbre
> conversion** (synthesise with a base voice, then convert the timbre with RVC)
> stays an M4 option; it needs extra model weights and is deliberately not part
> of M3.

## GPU acceleration

ViDub uses the GPU whenever the installed **torch** build can see one. Every
stage probes independently and falls back to CPU when CUDA is unavailable, so a
missing/broken CUDA install degrades to the CPU pipeline instead of failing.

| Stage | GPU path | Where the device is reported |
|---|---|---|
| ASR (`faster-whisper`) | CTranslate2 `cuda` / `float16` | `report.json` `meta.asr_device`; `vidub doctor` (`gpu whisper`) |
| ASR (`PhoWhisper`) | torch `cuda` / `float16` | `report.json` `meta.asr_device` |
| Source separation (Demucs) | torch `cuda` | `report.json` `meta.sep_device` + the mix log |
| Translation (envit5) | torch `cuda` | runner JSON `device` |
| TTS (ViXTTS, opt-in) | torch `cuda` (CPU fallback on failure) | runner JSON `device` |

Install the CUDA torch wheel **and** the CTranslate2 CUDA wheels:

```bash
./.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --upgrade
./.venv/Scripts/python.exe -m pip install -e ".[gpu]"
```

`pyproject.toml` keeps the CUDA torch wheel out of the `gpu` extra on purpose:
pip extras cannot carry an `--index-url`, and torch's CUDA wheel ships its own
CUDA DLLs in `site-packages/torch/lib` (Demucs/envit5/ViXTTS load them from
there), so it must be pulled from the PyTorch index in a separate command. A
plain `pip install -e .` therefore stays CPU-safe.

**VRAM.** Stages run **sequentially** and load one model at a time, so peak usage
is the largest single model (~5 GB for `large-v3`, ~3 GB for Demucs, ~3 GB for
ViXTTS) rather than their sum — fine on a 10 GB RTX 3080. NVIDIA's driver must be
recent enough for the wheel's CUDA version (cu126 here); `vidub doctor` shows what
is actually wired.

## Troubleshooting

- **`ffmpeg : missing`** — install it (see above), then open a new shell so PATH
  refreshes. `vidub doctor` reports it without crashing.
- **Model download fails** — the models come from Hugging Face on first use; check
  the network/proxy. Set `HF_TOKEN` to raise rate limits. Failures raise a clear
  error naming the model.
- **CPU vs GPU** — every stage tries CUDA first and transparently falls back to
  CPU when the CUDA runtime (e.g. `cublas64_12.dll`) is missing. ASR falls back to
  CPU int8, Demucs/envit5/ViXTTS run on CPU torch. Expect slower first runs while
  models download.
- **Re-running is cheap** — `--resume` (default) skips finished stages; use
  `--force <stage>` to redo just part of the pipeline, or `--clean` to start fresh.
- **GPU not used?** — run `vidub doctor`: its `cuda dlls`, `cuda available`,
  and `gpu whisper` lines say whether CUDA is wired and whether faster-whisper
  will use it. Missing cuBLAS/cuDNN DLLs fall back to CPU int8 automatically
  (install the `gpu` extra for the CUDA wheels). For Demucs/envit5/ViXTTS, check
  `python -c "import torch;print(torch.cuda.is_available())"` — False means the
  CPU torch wheel is installed (see [GPU acceleration](#gpu-acceleration)).
- **HF gated models (diarization)** — `--diarize` needs a Hugging Face token
  (`HF_TOKEN` or `huggingface-cli login`) *and* you must accept the conditions
  at https://hf.co/pyannote/speaker-diarization-3.1 and
  https://hf.co/pyannote/segmentation-3.0; otherwise the run fails before any
  synthesis with those setup steps.
- **Speech-less clips** — a video with no detected speech exports with **no
  subtitle stream at all** (not an empty/broken one) and its `report.json`
  records `"subtitles": "none (no speech detected)"`.
- **`language mismatch after ASR with model ...`** — the transcript looks
  Vietnamese (the error prints the measured diacritic ratio) but you did not
  declare `--src-lang vi`. This almost always means a `vi-small`/`vi-medium`
  PhoWhisper model ran on non-Vietnamese audio. Fix: use the default
  `auto`/`gpu-large` model for English audio, set `--src-lang` to match the
  audio, or pass `--allow-lang-mismatch` to force the run.
- **`speech coverage ... is below 0.6` / low `speech_coverage` in
  `report.json`** — the dub only fills part of the spoken time, so stretches of
  the output are silent. Likely causes: wrong-language ASR (short/hallucinated
  segments — `segments.json` text is gibberish, use the default model for English
  audio), unusually short TTS output, or a translation that was truncated/came
  back `low_confidence`. Inspect `output/<video>/script.tsv` and
  `.vidub-cache/<hash>/segments.json`, then re-run with `--force
  transcribe,translate,speak,timing,mix,export`.

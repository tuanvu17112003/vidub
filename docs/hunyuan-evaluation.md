# Hunyuan-MT evaluation — pre-registered En→Vi benchmark (M5-T3/D4)

**Status:** COMPLETE — decision section filled from real measurements. **Verdict: HY-MT1.5-7B is NOT promoted to recommended engine; the refine pass (best-of-K) is the win; Chimera fusion is NOT adopted as refine default.**
**Date:** 2026-09-29 · **Author:** Management Agent, applying QUALITY-ROADMAP.md §7.6 verbatim (thresholds were fixed in writing before any system ran).
**Reproduce:** `./.venv/Scripts/python.exe scripts/eval_hunyuan.py generate` → `./.eval-venv/Scripts/python.exe scripts/eval_hunyuan.py score` (raw numbers in `eval-cache/hunyuan/scores.json` + `meta.json`).

## Setup

- **Corpus:** 250 unique English source segments = 238 deduped real sources (>= 3 words) from all 12 `.vidub-cache/*/segments.json` job caches (incl. the 74-segment reference dub youtube-a7TOameRqoY) + 12 curated hard cases (numbers, entities, tech terms, prices, timestamps).
- **Systems (identical inputs, official parameters per model card — temp 0.7, top_k 20, top_p 0.6, repeat_penalty 1.05):**
  - `envit5` — VietAI envit5-translation, per-segment (the shipped default, Apache-2.0);
  - `qwen` — qwen2.5:7b via Ollama, contextual loop (the M4 opt-in engine, Apache-2.0);
  - `hunyuan-q6` / `hunyuan-q4` — Tencent HY-MT1.5-7B, official Q6_K / Q4_K_M GGUFs via Ollama, contextual mode;
  - `refine-envit5` / `refine-hunyuan` — primary system's rows + deterministic tqc corrections + ONE refine cycle (best-of-3 hunyuan samples, no fusion);
  - `refine-fusion-hunyuan` — hunyuan-q6 rows + refine with **Chimera-7B fusion** (community Q4_K_M GGUF; fusion smoke-verified on 3 sources before the run).
- **Hardware:** RTX 3080 10 GB (Ollama reported every model at 100% GPU: qwen 4.7 GB, Q6_K 6.8 GB, Q4_K_M 5.3 GB), 32 GB RAM, ffmpeg 9.0.2.
- **Metrics:** the four pre-registered in §7.6 — deterministic tqc battery flags/100 rows (primary), TransQuest DA (`monotransquest-da-multilingual`, reference-free human proxy, validated AUC 0.772 in M4-T8), dubbing length ratios, cost (s/row, VRAM).

## Results

### 1. Deterministic tqc battery — flags per 100 rows (lower is better)

| System | ok | number-error | entity-error | inconsistent | omission | addition | **flags/100** |
|---|---|---|---|---|---|---|---|
| envit5 (default) | 156 | 5 | 17 | 70 | 0 | 2 | **37.6** |
| qwen contextual | 148 | 0 | 22 | 75 | 4 | 1 | **40.8** |
| hunyuan-q6 | 147 | 4 | 19 | 78 | 0 | 2 | **41.2** |
| hunyuan-q4 | 149 | 4 | 19 | 76 | 0 | 2 | **40.4** |
| **refine-envit5** | 171 | 1 | 1 | 75 | 0 | 2 | **31.6** |
| refine-hunyuan | 166 | 1 | 0 | 82 | 0 | 1 | **33.6** |
| refine-fusion-hunyuan | 167 | 1 | 0 | 81 | 0 | 1 | **33.2** |

(No system produced a `terminology-error` or `addition`-beyond-table flag in this run; the ok/ok counts above are the battery's per-row labels from `score_rows` — raw systems scored without corrections, refine systems after corrections+refine.)

Unresolved review packets after corrections+refine (meta.json): raw envit5's flagging is what refine-envit5 starts from; **refine-envit5 → 4 packets, refine-hunyuan → 2 packets, refine-fusion-hunyuan → 19 packets** (17 of them because the Chimera-fused candidate failed the deterministic validators and was rejected).

### 2. TransQuest DA (reference-free human proxy; higher is better)

| System | mean | median | paired win-rate vs envit5 (strictly-greater) |
|---|---|---|---|
| envit5 (baseline) | 0.7404 | 0.7668 | — |
| qwen contextual | 0.7247 | 0.7540 | 0.384 |
| hunyuan-q6 | 0.7463 | 0.7680 | 0.276 |
| hunyuan-q4 | 0.7451 | 0.7685 | 0.292 |
| refine-envit5 | 0.7398 | 0.7653 | 0.032 |
| refine-hunyuan | 0.7430 | 0.7655 | 0.212 |
| refine-fusion-hunyuan | **0.7505** | **0.7708** | 0.352 |

The win-rate column counts a row as a win only when the system scores *strictly higher* than envit5 on that segment (near-ties count as losses) — with ~0 ties between different texts this behaves like a loss rate from 1.0. Read together with the means: qwen scores **lower** than envit5 overall (confirms the M4 note that qwen2.5:7b occasionally leaves English words in Vietnamese output); the hunyuan systems and fusion score slightly **higher** on average but lose on most individual segments, i.e. small margins.

### 3. Dubbing length fit (word-count ratio vs source)

| System | mean ratio | share > 2.2× |
|---|---|---|
| envit5 | 1.33 | 0.8% |
| qwen | 1.38 | 0.4% |
| hunyuan-q6 | 1.52 | 0.8% |
| hunyuan-q4 | 1.53 | 0.8% |
| refine-envit5 | 1.33 | 0.8% |
| refine-hunyuan | 1.47 | 0.4% |
| refine-fusion-hunyuan | 1.59 | 0.4% |

The HY-MT models run ~14 % longer than envit5 on average (hunyuan-q6 1.52 / hunyuan-q4 1.53 vs envit5 1.33) — a real dubbing-rhythm cost at the current prompt, though the truncation guard keeps >2.2× shares equal. Subtitle reading-speed (CPS) violations — part of §7.6 metric 3 — were **not** computed in this run; the length-ratio distribution is the dubbing-fit proxy that was reported.

### 4. Cost

| System | mean s/row (generation) | VRAM (ollama ps) |
|---|---|---|
| envit5 | 0.08 | (in-process CUDA) |
| qwen contextual | 1.01 | 4.7 GB, 100% GPU |
| hunyuan-q4 | 2.67 | 5.3 GB, 100% GPU |
| hunyuan-q6 | 3.57 | 6.8 GB, 100% GPU |
| refine-envit5 | 1.81 | — |
| refine-hunyuan | 2.37 | — |
| refine-fusion-hunyuan | 4.15 | Chimera Q4_K_M + Q6_K (sequential) |

## Decision (applying §7.6 verbatim — thresholds fixed before the run)

**Rule 1 — promote `hunyuan` to recommended engine iff ALL of:**
1. TransQuest mean ≥ envit5 + 0.02 → hunyuan-q6 0.7463 − 0.7404 = **+0.0059 < +0.02 → NOT MET**
2. battery flags/100 ≤ envit5 → 41.2 > 37.6 → **NOT MET**
3. ≤ 3 s/row at Q6_K → 3.57 > 3.0 → **NOT MET**

→ **HY-MT1.5-7B is NOT promoted.** It remains an opt-in engine (`--mt-engine hunyuan`), **envit5 remains the default**. Its measured translation quality is comparable to envit5 on this corpus (not measurably better by our proxies), it runs ~14 % longer translations, and it is ~45× slower per row.

**Rule 2 — adopt Chimera fusion into the refine default iff it cuts unresolved review packets ≥ 30 % vs best-of-K without increasing validator failures:**
refine-fusion-hunyuan left **19** unresolved packets vs refine-hunyuan's **2** — a 9.5× **increase**, driven by 17 `refine-invalid` rejections (fused candidates failing the deterministic validators). → **Chimera fusion is NOT adopted as default.** It stays behind `--fusion-model` (off by default), consistent with the T2 design.

**Shipped outcome (unchanged defaults, opt-in flags):**
- `envit5` remains the default MT engine.
- `--hunyuan-refine` (default off) is **recommended for quality-critical runs on the default path**: it cuts the battery flag count from 37.6 to **31.6 per 100 rows (−16 %)** at a throughput cost (0.08 → 1.8 s/row) and requires the HY-MT1.5 GGUF locally.
- `--fusion-model` (Chimera) remains an evaluation-only option — under the current community Q4_K_M quant it degrades the refine pass; revisit only with a better quant or a future official Chimera GGUF.

## Threats to validity (stated honestly)

1. **Corpus provenance:** sources are real ViDub ASR transcripts (YouTube tech/gaming content + short demos) — representative of ViDub's use case, but not a balanced MT benchmark; the curated set shipped as 12 hard cases (the §7.6 plan called for 30 — the original task spec listed 12; the shortfall is visible rather than hidden).
2. **TransQuest is a proxy,** validated AUC 0.772 (M4-T8) on 60 En→Vi pairs; it misses number swaps (which the battery catches) and its absolute deltas of ±0.006 are within plausible proxy noise. No human evaluation was performed in this milestone.
3. **Corrections confound:** `refine-*` systems include the deterministic tqc corrections as well as the refine pass; the flags/100 improvements therefore measure the shipped pipeline configuration, not the refine pass in isolation.
4. **Quantization:** Chimera fusion ran on a community Q4_K_M GGUF (no official one exists); its `refine-invalid` rate may partly be a quantization artifact.
5. **One crashed run was discarded** (Ollama timeout during the first Q6 pass) and re-run cleanly; all reported outputs are 0-empty on the final run.

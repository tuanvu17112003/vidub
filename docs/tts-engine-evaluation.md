# Expressive TTS engine evaluation — Parler-TTS vs VoxCPM (QUALITY-ROADMAP.md D4)

**Status:** audit complete · **Task:** M4-T8 · **Date:** 2026-09-29
**Scope:** decide whether either candidate deserves adoption for the *Vietnamese*
dubbing path. This is a desk audit (model cards + READMEs); no weights were
downloaded or run.

The binding criterion is **Vietnamese support**; a licensing-clean model that
cannot speak Vietnamese cannot replace Piper/ViXTTS. Each engine is scored on the
four criteria the task defines: (a) Vietnamese support, (b) license, (c)
VRAM/runtime feasibility on an RTX 3080 10 GB alongside the existing pipeline, and
(d) integration cost (shim required? cloning support?).

## Summary

| Engine | (a) Vietnamese | (b) License | (c) VRAM / RTX 3080 10 GB | (d) Integration | Verdict |
|---|---|---|---|---|---|
| **Parler-TTS large v1** | ❌ English-only | Apache-2.0 (permissive) | ~8.8 GB fp32 / ~4.5 GB bf16 — bf16 only | transformers-native, but **no reference cloning** | **REJECT** |
| **VoxCPM-0.5B** | ❌ zh + en only | Apache-2.0 (commercial OK) | ~5 GB; RTF ~0.17 (4090) | custom `voxcpm` package; continuation-only cloning | **REJECT** |
| *VoxCPM2 (2B)* — follow-up | ✅ **30 langs incl. Vietnamese** | Apache-2.0 (commercial OK) | ~8 GB | custom package, cloning + voice design | **DEFER** (recommended spike) |

## Parler-TTS (`parler-tts/parler-tts-large-v1`)

* **(a) Vietnamese — NO.** The model is a 2.2B-parameter TTS "trained on 45K
  hours of audio data"; the card's training datasets are all English —
  `parler-tts/libritts_r_filtered`, `parler-tts/mls_eng`,
  `parler-tts/mls-eng-speaker-descriptions` — and its sole language tag is
  **English**. There is no Vietnamese data, tokenizer or speaker. Source:
  <https://huggingface.co/parler-tts/parler-tts-large-v1>.
* **(b) License — Apache-2.0**, permissive, explicitly "free for commercial use"
  in spirit ("fully open-source ... under permissive license"). Sources:
  <https://huggingface.co/parler-tts/parler-tts-large-v1> (§License) and
  <https://github.com/huggingface/parler-tts> (Apache-2.0).
* **(c) VRAM / runtime.** 2.2B params stored **F32** → ~8.8 GB of weights, which
  does not co-exist comfortably with activations + the audio codec on a 10 GB
  card; a bf16/fp16 cast (~4.5 GB) fits, and ViDub runs one model at a time, so a
  bf16 load is feasible. No RTF is published for a 3080; the repo ships an
  `INFERENCE.md` optimisation guide (SDPA/torch.compile). Source:
  <https://huggingface.co/parler-tts/parler-tts-large-v1>.
* **(d) Integration.** Usable through **stock `transformers`**
  (`ParlerTTSForConditionalGeneration` / `pipeline("text-to-speech", ...)`) or the
  `parler-tts` git package — no torchaudio shim needed. Voice is steered by a
  **text description** and 34 named speakers; there is **no reference-audio
  zero-shot cloning** (unlike ViXTTS), so it cannot reproduce a video's own
  speakers. Sources: <https://huggingface.co/parler-tts/parler-tts-large-v1>,
  <https://github.com/huggingface/parler-tts>.

**Verdict: REJECT.** Clean license and a transformers-native integration, but it
is English-only and has no reference cloning — neither requirement of a
Vietnamese dubbing engine.

## VoxCPM-0.5B (`openbmb/VoxCPM-0.5B`)

* **(a) Vietnamese — NO.** The project's own model table lists VoxCPM-0.5B as
  supported on **2 languages (zh, en)**; the HF card's language tags are `en` and
  `zh`, and its "Risks and limitations" state verbatim: *"VoxCPM is trained
  primarily on Chinese and English data. Performance on other languages is not
  guaranteed and may result in unpredictable or low-quality audio."* Sources:
  <https://huggingface.co/openbmb/VoxCPM-0.5B>,
  <https://github.com/OpenBMB/VoxCPM> (Models & Versions table).
* **(b) License — Apache-2.0**, "free for commercial use"; the card adds a
  research/development caveat (no production recommendation without testing).
  Sources: <https://huggingface.co/openbmb/VoxCPM-0.5B> (§License),
  <https://github.com/OpenBMB/VoxCPM> (Apache-2.0).
* **(c) VRAM / runtime.** **~5 GB VRAM**, **RTF ~0.17 on an RTX 4090**, 16 kHz
  output; requirements Python ≥3.10, **PyTorch ≥2.5, CUDA ≥12** (the existing
  `torch 2.14.0+cu126` satisfies this). 5 GB fits a 10 GB 3080 with headroom and
  ViDub's one-model-at-a-time staging. Source:
  <https://github.com/OpenBMB/VoxCPM> (Models & Versions table).
* **(d) Integration.** `pip install voxcpm` (a **new third-party package** beyond
  transformers); Python API + CLI; **zero-shot cloning** support
  (`prompt_wav_path`/`prompt_text`, continuation mode). No torchaudio shim appears
  necessary (it writes via `soundfile`). But since it has no Vietnamese, the cost
  buys nothing. Sources: <https://huggingface.co/openbmb/VoxCPM-0.5B>,
  <https://github.com/OpenBMB/VoxCPM>.

**Verdict: REJECT.** Excellent size/latency and an Apache-2.0 license, but
zh/en-only — not usable for Vi.

## Follow-up worth a scoped spike: VoxCPM2 (2B)

The same project's newer release is the only audited engine that explicitly
supports Vietnamese:

* **(a) Vietnamese — YES.** VoxCPM2 supports **30 languages, including
  Vietnamese**. Published numbers: MiniMax-MLS-test **vi WER 3.307** / **SIM 80.6**
  (vs. FishAudio S2 7.410 / 74.0 and ElevenLabs 73.415 / 36.9), and an internal
  30-language ASR benchmark with **vi WER 1.56%**. Source:
  <https://github.com/OpenBMB/VoxCPM> (Models, Minimax-MLS-test, Internal ASR table).
* **(b) License — Apache-2.0**, explicitly "commercial-ready". Sources:
  <https://github.com/OpenBMB/VoxCPM>, <https://huggingface.co/openbmb/VoxCPM2>.
* **(c) VRAM / runtime — ~8 GB VRAM**, RTF ~0.30 on an RTX 4090 (~0.13 with
  Nano-vLLM). 8 GB fits a 10 GB 3080 only because ViDub runs stages sequentially;
  it is the heaviest model in the pipeline. Sources:
  <https://github.com/OpenBMB/VoxCPM>, <https://arxiv.org/abs/2606.06928>.
* **(d) Integration.** `pip install voxcpm`; zero-shot and "controllable" cloning
  from a short reference clip + optional transcript, plus text-prompt Voice Design;
  48 kHz output. Needs an evaluation of Vietnamese naturalness vs ViXTTS and a
  check for any runtime shim. Sources:
  <https://huggingface.co/openbmb/VoxCPM2>, <https://voxcpm.readthedocs.io>.

**Verdict: DEFER** (not in this task — no weights run, and the D4 scope was the
0.5B/Parler pair), but it is the strongest future candidate for the expressive-dub
slot.

## Recommendation

Keep **Piper** and **ViXTTS** as the shipped TTS engines: **reject** Parler-TTS
(no vi) and VoxCPM-0.5B (no vi), and **defer** a time-boxed VoxCPM2 (2B) spike as
the one vi-capable expressive-engine candidate — measure it against ViXTTS on
Vietnamese naturalness and confirm it fits the 10 GB VRAM budget before any
adoption.

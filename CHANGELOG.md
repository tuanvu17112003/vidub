# Changelog
Linux/macOS are untested (Windows-first).

## Unreleased

- Repo cleanup: research scripts moved to `tools/research/`, root design docs into `docs/`, changelog split out of README.
- Run options now have a single registry (`vidub.options`); the CLI and the web UI are generated from it and expose the same option set.
- `vidub/stages/timing` and `vidub/stages/translate` became packages (pure code moves).
- CI: GitHub Actions lint + test (Windows/Ubuntu), wheel build on tags.

## v0.11.0 (in development)

No pipeline behavior change in this round; see Unreleased.

## v0.10.0

**Status: v0.10.0 — quality round (Q-1) + vietsub mode (SU-1).** The full En→Vi
pipeline works end-to-end
(`vidub run clip.mp4`): ingest → transcribe → (optional diarize → references) →
translate → speak → timing → mix → export. **M5 adds opt-in quality upgrades** —
the Tencent **Hunyuan HY-MT1.5-7B** translation specialist (`--mt-engine
hunyuan`) and a **translation refine pass** (`--hunyuan-refine`) that re-samples
still-flagged rows with the local Hunyuan GGUF and keeps a candidate only when
it is strictly better — plus a pre-registered En→Vi benchmark whose verdict
keeps **`envit5` as the default engine** and *recommends the refine pass* for
quality-critical runs. **M4 added the quality-control framework** — transcript /
subtitle / translation / per-clip TTS QC with a final aggregate gate — plus the
opt-in Ollama LLM MT engine and source-transcript punctuation restoration. M3
added the CUDA-torch GPU pipeline, the translation glossary, the TTS language
matrix, YouTube URL input, per-speaker voice cloning and a local web UI.
## v0.10.0

closes the Q-1 **quality round** and adds the SU-1 **vietsub
mode**: a unit-contamination guard that catches and repairs hunyuan's
neighbour-context bleed, a translation **refine pass that is now ON by default
for `--mt-engine hunyuan`** (`--no-hunyuan-refine` opts out), pacing levers
that trade clips for a bounded short-window stretch and a concise
re-translation, and subtitle cues that split on the placed audio's ASR
**word edges**; `--mode subtitles` is an explicit alias of
`--mode subtitles-only`. Three pre-registered quality targets are **NOT met**
as measured — the honest record is in
[`docs/quality-round-q1.md`](docs/quality-round-q1.md); the vietsub mode is in
[`docs/subtitles-only.md`](docs/subtitles-only.md).
Linux/macOS are untested (Windows-first).
## v0.9.0

turns the **sentence** into the row unit: ASR rows are split at
sentence boundaries with word-edge-locked timestamps, units are
**sentence-scoped** (a unit ends only at a completed sentence or a speaker
change), a new completeness guard re-samples/drops a unit translation that lost
content, the translation-QC battery scores **units**, and the timing stage
force-aligns each unit's audio so every member row gets a real
`member_spans` display span (`--align-display`, default on). The authoritative
unit map is `units.json`; `--no-sentence-units` is still byte-identical. See
[Sentence-level translation units](#sentence-level-translation-units) and
[`docs/sentence-locked-units.md`](docs/sentence-locked-units.md).
## v0.8.0

makes the timing stage speak **whole sentence units** (default
on): each multi-segment sentence unit is synthesized **once** from its joined
member text and placed as a **single timed row** across the unit's span, so an
intra-sentence pause — e.g. the `…đào ⏸ tạo…` split — can no longer exist
(`--no-sentence-units` restores per-segment timing byte-for-byte). Subtitle
rows stay per segment. See [Unit-level audio (UA-1)](#unit-level-audio-ua-1)
and [`docs/unit-audio.md`](docs/unit-audio.md).
## v0.7.0

adds **two-pass TTS rate fitting** (default on): each Piper segment
is first re-synthesized at a fitted `length_scale` (0.85×–1.30× around natural,
plus a video-wide pace offset) so it matches its window without an aggressive
`atempo` or a clip, and leftover window slack is used by bounded re-centering
(`--no-tts-rate-fit` / `--no-recenter` restore the previous behaviour). See
[Two-pass rate fitting](#two-pass-rate-fitting-fit-1) and
[`docs/timing-fitting.md`](docs/timing-fitting.md).
## v0.6.0

adds **sentence-level translation units** (default on): ASR
fragment runs are merged into whole sentences before translation and the
translation is split back onto the original segment windows, so `script.json`
rows stay 1:1 with the audio segments (`--no-sentence-units` restores the
previous per-segment translation).
## v0.5.1

rounds out the local web UI — one-click M5 test presets, an Ollama
pre-flight that refuses to start a run with fix commands when the server/model is
missing, the remaining M3 controls and a **Quality results** panel — and makes a
changed MT configuration re-run `translate` and everything downstream
automatically (no more silently reused cached scripts).

# Subtitles only (vietsub) — `--mode subtitles-only`

`--mode subtitles-only` (alias `--mode subtitles`) turns a video into a
**Vietnamese vietsub video** — the original footage with a Vietnamese subtitle
track — **without any dubbing**. It is the lightest output mode: no TTS voice, no
speaker separation and no audio mix run, and the **original audio track is kept
untouched**.

## What it does (and what it skips)

The mode runs only the text half of the pipeline:

```text
ingest → transcribe → translate → export
```

and deliberately **skips `speak`, `timing` and `mix`** (see
`MODE_SKIPPED_STAGES` in `vidub/stages/export.py`). Concretely:

- **runs** `ingest` (audio extraction), `transcribe` (ASR), `translate` (En→Vi,
  including the default sentence-unit path) and `export` (subtitle build + mux);
- **skips** `speak` (no Piper/ViXTTS synthesis), `timing` (no rate fitting /
  alignment) and `mix` (no Demucs separation, no `loudnorm`) — there is no new
  audio to place;
- **keeps the ORIGINAL audio** verbatim: the export mux copies it (`-c:a copy`)
  instead of muxing a generated `dub_track.wav`;
- **skips the TTS license guard** (`check_tts_engine_allowed`) and the TTS
  language matrix (`validate_langs`): with no synthesis the TTS engine cannot
  block a subtitles-only run. The CLI, the web UI and `run_pipeline` all apply
  this skip, so the three front-ends agree. `--clone-speakers` still errors
  explicitly — it needs diarize + vixtts and cannot run in this mode;
- **still validates MT** (`check_mt_engine`) and the clone-speakers combination,
  because translation still runs.

`report.json` records the mode and states the skip: `engines.tts` is
`"skipped (subtitles-only)"`, and the `speak` / `timing` / `mix` entries are
`skipped`. The web UI's **Quality results** panel mirrors this with a *Mode* line
and a *Skipped: `speak`, `timing`, `mix`* note.

## When to use it

- You want a **vietsub** (Vietnamese subtitle) video and do not need a dub voice.
- You are **reviewing or fixing the translation** before paying for synthesis:
  the SRT/ASS/`script.tsv` are produced, and a later `--mode dub` run on the same
  `--cache-dir` **reuses the cached `transcribe`/`translate`** stages (the cache
  is keyed on the video, not the mode).
- You want **hard-burned, styled subtitles** in a single shareable file
  (`--burn`).
- You want the **fastest** En→Vi output: no synthesis, no Demucs separation and
  no loudness mix.

## CLI examples

Both spellings select the same mode — `subtitles` is an alias for
`subtitles-only`, and the manifest/report always store the canonical value:

```bash
vidub run clip.mp4 --mode subtitles                 # alias spelling
vidub run clip.mp4 --mode subtitles-only            # canonical spelling
vidub run clip.mp4 --mode subtitles --burn          # hard-burned vietsub
vidub batch ./clips/ --mode subtitles-only          # batch accepts it too
vidub run <URL> --mode subtitles-only --asr-model tiny
```

## Subtitle style flags

| Flag | Default | Meaning |
|---|---|---|
| `--sub-pos {bottom,top,center}` | `bottom` | On-screen placement; drives both the `.ass` export and `--burn`. |
| `--sub-font-size N` | `24` | Font size in points for the `.ass` export and `--burn`. |
| `--sub-margin-v N` | per position (`bottom`/`top` `24`, `center` `0`) | Vertical margin in pixels for the `.ass` export and `--burn`. |

The same three controls are exposed in the web UI as the *Mode* dropdown,
*Subtitle position*, *Subtitle font size* and the *Burn subtitles* checkbox.

**ASS vs legacy alignment numbering (a gotcha worth knowing).** The `.ass`
artifact is written with the standard **ASS `\an`** alignment: `2` =
bottom-centre, `8` = top-centre, `5` = middle-centre
(`SUB_POSITION_ALIGNMENT` in `vidub/stages/export.py`). The `--burn` path does
**not** use that file: ffmpeg's `subtitles` filter reads the **SRT**, builds a
v4.00/legacy style from it, and applies `force_style=Alignment=` with the
**legacy SSA `\a`** numbering — `2` = bottom, `6` = top, `10` = centre
(`SUB_POSITION_LEGACY_ALIGNMENT`). The two numbers differ but land in the **same
band**; this was verified with real burns (feeding the ASS numbers to the filter
put `top` in the middle band, `center` at the top). You normally only touch the
flag (`--sub-pos top`), not the numbers — but those two maps are the place to
look if you ever hand-edit a style.

## Outputs

`--mode subtitles-only` writes the normal `output/<video-basename>/` tree:

```text
output/<video-basename>/
├── dubbed.vi.mp4      # the vietsub video: original footage + original audio
│                      #   (soft vi subtitle stream, or burned-in text with --burn)
├── subtitles.vi.srt   # Vietnamese subtitle track (one cue per sentence unit)
├── subtitles.vi.ass   # the same track with the ViDub style block (position/size/margin)
├── script.tsv         # start / end / speaker / en / vi
└── report.json        # mode, engines (tts "skipped (subtitles-only)"), QC, timings
```

- `dubbed.vi.mp4` keeps its cross-mode name for consistency; in this mode it is
  the **vietsub video** — the **original audio is copied, not replaced**.
- Without `--burn` the Vietnamese subtitles are a **soft `mov_text` stream** in
  the mp4; with `--burn` the text is **rendered into the picture** and there is
  **no subtitle stream** at all.
- A speech-less clip (0 segments) exports with **no subtitle stream**, and
  `report.json` records `"subtitles": "none (no speech detected)"` (this holds in
  every mode).

## Burn cost

`--burn` is the expensive option: instead of `-c:v copy` the export re-encodes
the **whole** video with **`libx264` / `yuv420p`** and applies the `subtitles=`
filter (`vidub.stages.export.build_mux_command`). The soft-subtitle path copies
the video stream untouched, so it is near-instant; the burn path pays a full
`libx264` encode of the clip regardless of length. Burn only when the player
cannot render a soft subtitle track (or you want the text locked onto the
pixels).

## Measured: dub vs subtitles-only

> **Caveat: a short clip makes these numbers indicative, not a benchmark.** On a
> clip this short the fixed startup costs (model load, ffmpeg spawn, and the
> shared ingest → transcribe → translate passes) dominate, so the wall-clock gap
> below is a *lower bound* on the saving a longer video would show.

Same input and machine for both runs; the only differences are the mode and the
scratch cache root. Input:
`.vidub-cache/downloads/generic-demo_en.mp4` — **21.6 s**, 1 h264 video stream +
1 aac audio stream.

| Run | Mode | Wall clock | `dubbed.vi.mp4` streams |
|---|---|---|---|
| `vidub run <in> --mode dub` | `dub` | **35.33 s** | h264 video + **aac (synthesized dub)** + `mov_text` subtitle |
| `vidub run <in> --mode subtitles --burn` | `subtitles-only` | **16.75 s** | h264 (text burned in) + **aac (original audio copy)** + *no subtitle stream* |

`ffprobe -show_entries stream=codec_type,codec_name` on each output:

```text
# --mode dub            (35.33 s)
index=0  codec_name=h264      codec_type=video
index=1  codec_name=aac       codec_type=audio
index=2  codec_name=mov_text  codec_type=subtitle
duration=21.600000

# --mode subtitles --burn   (16.75 s)
index=0  codec_name=h264      codec_type=video
index=1  codec_name=aac       codec_type=audio
duration=21.600000
```

The `--burn` run confirms both properties: the subtitles are baked into the
picture (no `subtitle` stream) and the audio is the **original** stream —
the export command is
`ffmpeg -i <input> -map 0:v:0 -map 0:a:0? -c:v libx264 -pix_fmt yuv420p
-vf subtitles=subtitles.vi.srt:force_style='Alignment=2,FontSize=24,MarginV=24'
-c:a copy dubbed.vi.mp4`. On this clip the mode change removed the `speak` +
`timing` + `mix` work and roughly **halved** the wall clock (35.33 s → 16.75 s,
≈2.1× faster); a longer video — where synthesis and separation scale with runtime
— would widen the gap.

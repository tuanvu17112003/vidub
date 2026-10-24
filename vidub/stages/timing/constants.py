"""Shared artifact names, thresholds and suffixes for the timing stage."""

from __future__ import annotations

import re

SPEECH_NAME = "speech.json"
SCRIPT_NAME = "script.json"
TIMED_NAME = "timed.json"
TIMING_NAME = "timing.json"
# Authoritative sentence-unit map written by the translate stage (SA-1-T2). When
# present and consistent it supersedes ``group_units`` for unit membership/text.
UNITS_NAME = "units.json"
# Per-clip TTS QC sidecar + review packets (M4-T7, QUALITY-ROADMAP.md §3.4/P1.7).
TTS_QC_NAME = "tts_qc.json"
TTS_QC_REVIEW_SUBDIR = "review/tts"
# Changed generation settings for the one low-MOS retry (viXTTS sampling).
VXTTS_RETRY_TEMPERATURE = 0.65
VXTTS_RETRY_TOP_K = 80
SUBTITLES_ONLY = "subtitles-only"
# Below this playback coverage the dub is audibly incomplete (long silences);
# the M2 demo measured 0.17 when the ASR hallucination collapsed the slots.
SPEECH_COVERAGE_WARN = 0.6
# Tolerance (s) for the no-bleed invariant and float comparisons. Fitted audio
# is hard-trimmed at the window end, so real bleeds are sub-millisecond.
BLEED_TOLERANCE = 0.01
# Minimum speech a non-clipped row must keep before a hard cut (M3-C2).
MIN_RETAINED_SPEECH = 0.05
# Tail-safety (M3-C2). A trim must land in a pause when one exists near the end
# of the stretched audio: silences are detected and a pause that starts within
# _SMART_TRIM_LOOKBACK of the window (or straddles it) becomes the cut. Blind
# cuts stay flagged "clipped"; every cut gets a short fade so it cannot click.
_SMART_TRIM_LOOKBACK = 0.6
_FADE_OUT = 0.06
_SILENCE_NOISE = "-35dB"
_SILENCE_MIN_DUR = 0.15
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[0-9.]+)")
# Default padding added to a fitted clip so a frame-boundary round never bleeds.
_RETRANSLATE_SUFFIX = "_re"
_FIT_SUFFIX = "_fit"
_CLIP_SUFFIX = "_clip"
# FIT-1: a segment re-synthesised at a fitted rate lands in ``seg_XXXX_ls.wav``.
_RATE_SUFFIX = "_ls"
# UA-1: a multi-segment sentence unit is synthesised once into
# ``speech/unit_XXXX.wav`` (one WAV, one placed row, no intra-sentence pause).
_UNIT_SYNTH_DIR = "speech"
# Reason recorded when a unit cannot fit its window and the unit path does not
# run the per-segment re-translation retry (see ``fit_unit``).
UNIT_RETRANSLATE_OFF = "sentence unit: no re-translation retry"
# Target-language display names for the concise re-translation prompt (Q-1-T3).
_TGT_LANG_NAMES = {
    "vi": "Vietnamese",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}

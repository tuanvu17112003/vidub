"""Declarative registry of every ViDub run option (spec 2026-10-25, D2).

``OPTION_SPECS`` is the single source of truth shared by the CLI
(``vidub.cli``) and the web UI (``vidub.ui``): one row per user-facing
``JobConfig`` field, carrying the flag spelling, type, defaults, help text
and UI widget. Adding an option means editing the table here — never three
places.

Defaults follow the spec's D2 ruling: ``default`` is always the
``JobConfig`` dataclass default; ``cli_default`` / ``ui_default`` record
legitimate per-surface overrides where the surfaces differ today
(``asr_model``, ``hunyuan_refine``, ``sub_margin_v``). ``vidub.ui`` imports
nothing from this module's consumers, so this module stays importable
without Gradio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from vidub.config import (
    DEFAULT_ASR_MODEL,
    DEFAULT_DIARIZE_MODEL,
    DEFAULT_HUNYUAN_MODEL,
    DEFAULT_HUNYUAN_HOST,
    DEFAULT_ISOMETRIC,
    DEFAULT_ISOMETRIC_CPS,
    DEFAULT_ISOMETRIC_HEADROOM,
    DEFAULT_ISOMETRIC_SHORT_COVERAGE,
    DEFAULT_ISOMETRIC_SHORT_WINDOW,
    DEFAULT_LOUDNESS_I,
    DEFAULT_LOUDNESS_LRA,
    DEFAULT_LOUDNESS_TP,
    DEFAULT_MT_ENGINE,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SPEED_AS_IS,
    DEFAULT_SPEED_CAP,
    DEFAULT_SUB_FONT_SIZE,
    DEFAULT_SUB_POSITION,
    DEFAULT_TAIL_PAD,
    DEFAULT_VOXCPM2_VOICE,
    MT_ENGINES,
    SPEED_CAP as _SPEED_CAP_SENTINEL,  # noqa: F401  (re-export guard; unused)
    SUB_POSITIONS,
    SUBTITLES_ONLY,
    TTS_ENGINES,
    VOXCPM2_VOICE_PROFILES,
)

__all__ = ["OptionSpec", "OPTION_SPECS", "HAND_WRITTEN_FIELDS"]

#: Sentinel for "this surface uses ``default``" on a per-surface override.
_UNSET = object()


@dataclass(frozen=True)
class OptionSpec:
    """One user-facing run option, shared by the CLI and the web UI."""

    field: str
    """``JobConfig`` field name; also the argparse dest unless ``dest`` says otherwise."""

    flag: str
    """Primary CLI option string (``""`` when the option is UI-only or programmatic-only)."""

    kind: str
    """``bool`` | ``int`` | ``float`` | ``str`` | ``tri`` | ``choice`` | ``path`` | ``json-map``."""

    default: object
    """The ``JobConfig`` dataclass default (never a second literal)."""

    dest: str = ""
    """argparse dest override for historical spellings; ``""`` means ``field``."""

    cli_action: str = "value"
    """``value`` | ``opt`` (BooleanOptionalAction) | ``true`` (store_true) | ``false`` (store_false)."""

    cli_default: object = _UNSET
    """CLI-only default override where the CLI legitimately differs (spec D2)."""

    choices: tuple = ()
    """Valid values for ``choice`` specs; empty otherwise."""

    metavar: str = ""
    """argparse metavar for value options (``""`` uses the default spelling)."""

    aliases: tuple = ()
    """Extra CLI option strings attached to the same dest (same help text)."""

    suppressed_aliases: tuple = ()
    """Legacy CLI aliases whose help is ``argparse.SUPPRESS`` (store_false)."""

    help: str = ""
    """The exact CLI help text (verbatim from the pre-registry argparse)."""

    ui: str | None = None
    """``checkbox`` | ``number`` | ``text`` | ``dropdown`` | ``tri``; ``None`` = no standard UI widget."""

    ui_label: str = ""
    """The Gradio component label."""

    ui_group: str = "main"
    """UI placement group: ``main`` | ``engines`` | ``timing``."""

    ui_default: object = _UNSET
    """UI-only default override where the UI legitimately differs (spec D2)."""

    transform: str = ""
    """Post-parse transform hook name (``"voice_map"`` for ``json-map``)."""

    ui_choices: tuple = field(default_factory=tuple)
    """Dropdown choices as ``(label, value)`` pairs; empty uses ``choices``."""

    _UNSET: ClassVar[object] = _UNSET
    """Sentinel accessor (``OptionSpec._UNSET``) for "no per-surface override"."""


HAND_WRITTEN_FIELDS = frozenset(
    {
        # Cache root: a run control on both surfaces (CLI default None, UI
        # seeds the JobConfig default) — deliberately outside the registry.
        "cache_dir",
        # Programmatic-only: no CLI flag, no UI widget (mix-stage gain).
        "voice_over_gain",
    }
)

# Historical dest overrides: the CLI flag spelling predates the dataclass
# field name and both are load-bearing (docs, scripts, muscle memory).
_DEST_WORDS = "asr_words"
_DEST_SUB_POS = "sub_pos"
_DEST_RECENTER = "recenter_audio"
_DEST_LOUD_I = "loudness_target"

OPTION_SPECS: tuple[OptionSpec, ...] = (
    # ---------------------------------------------------------------- modes
    OptionSpec(
        field="mode",
        flag="--mode",
        kind="choice",
        default="dub",
        choices=("dub", "voice-over", SUBTITLES_ONLY, "subtitles"),
        help=(
            "output mode (default: dub). 'subtitles' is an alias for "
            "'subtitles-only' (the same mode)"
        ),
        ui="dropdown",
        ui_label="Mode (Subtitles only = vietsub, no TTS)",
        ui_choices=(
            ("Dub (replace the vocals)", "dub"),
            ("Voice-over (keep the original audio ducked)", "voice-over"),
            ("Subtitles only (no TTS)", "subtitles-only"),
        ),
    ),
    OptionSpec(
        field="src_lang",
        flag="--src-lang",
        kind="str",
        default="en",
        help="source language (default: en)",
        ui="text",
        ui_label="Source language",
    ),
    OptionSpec(
        field="tgt_lang",
        flag="--tgt-lang",
        kind="str",
        default="vi",
        help="target language (default: vi)",
        ui="text",
        ui_label="Target language",
    ),
    OptionSpec(
        field="asr_model",
        flag="--asr-model",
        kind="str",
        default="base",
        cli_default=DEFAULT_ASR_MODEL,
        metavar="MODEL",
        help=(
            "ASR model: a preset (cpu-tiny, cpu-base, cpu-small, gpu-large, "
            "vi-small, vi-medium), a faster-whisper size (tiny, base, small, "
            "medium, large-v3), or a Hugging Face repo id. "
            f"Default: {DEFAULT_ASR_MODEL} (large-v3 on GPU, else base)."
        ),
        ui="dropdown",
        ui_label="ASR model",
    ),
    OptionSpec(
        field="asr_words",
        flag="--words",
        dest=_DEST_WORDS,
        kind="bool",
        default=True,
        cli_action="opt",
        help="word-level ASR timestamps (default: on)",
        ui="checkbox",
        ui_label="Word-level ASR timestamps",
    ),
    OptionSpec(
        field="restore_punct",
        flag="--restore-punct",
        kind="bool",
        default=False,
        cli_action="true",
        help=(
            "opt-in punctuation/casing restoration of the source transcript "
            "(oliverguhr/fullstop-punctuation-multilang-large; ~2.2 GB download "
            "on first use). Already-punctuated segments are left untouched. "
            "Default: off"
        ),
        ui="checkbox",
        ui_label="Restore punctuation (source)",
        ui_group="engines",
    ),
    OptionSpec(
        field="qc_transcript",
        flag="--qc-transcript",
        kind="bool",
        default=True,
        cli_action="opt",
        help=(
            "Level-1 transcript QC: classify ASR segments, retry flagged "
            "windows once and record a transcript verdict (default: on)"
        ),
        ui="checkbox",
        ui_label="QC: transcript",
        ui_group="engines",
    ),
    OptionSpec(
        field="qc_subtitles",
        flag="--qc-subtitles",
        kind="bool",
        default=True,
        cli_action="opt",
        aliases=("--subtitle-qc",),
        help=(
            "subtitle standards QC: wrap/re-split over-long or over-CPS "
            "Vietnamese cues and count profile violations (default: on; "
            "disable with --no-qc-subtitles or --no-subtitle-qc)"
        ),
        ui="checkbox",
        ui_label="QC: subtitles",
        ui_group="engines",
    ),
    OptionSpec(
        field="qc_translation",
        flag="--qc-translation",
        kind="bool",
        default=True,
        cli_action="opt",
        suppressed_aliases=("--no-translation-qc",),
        help=(
            "Level-2 translation QC: deterministic number/entity/terminology "
            "and consistency signals, glossary-splice corrections and a "
            "translation verdict (default: on)"
        ),
        ui="checkbox",
        ui_label="QC: translation",
        ui_group="engines",
    ),
    OptionSpec(
        field="sentence_units",
        flag="--sentence-units",
        kind="bool",
        default=True,
        cli_action="opt",
        help=(
            "translate whole sentences instead of ASR fragments: merged "
            "fragment runs are translated together and split back onto the "
            "original segments (default: on; disable with --no-sentence-units)"
        ),
        ui="checkbox",
        ui_label="Sentence-level translation units",
        ui_group="engines",
    ),
    # ------------------------------------------------- isometric (ISO-1)
    OptionSpec(
        field="isometric",
        flag="--isometric",
        kind="bool",
        default=DEFAULT_ISOMETRIC,
        cli_action="opt",
        help=(
            "isometric length budgeting (ISO-1, sentence-unit path): compute each "
            "unit's character budget from its time window, pass it to the engine "
            "and run one bounded budgeted re-translation for still-over-budget "
            "units, accepted only when it is genuinely better (default: off since "
            "v0.11.0; enable with --isometric)"
        ),
        ui="checkbox",
        ui_label="Isometric length budgeting (ISO-1)",
        ui_group="timing",
    ),
    OptionSpec(
        field="isometric_headroom",
        flag="--isometric-headroom",
        kind="float",
        default=DEFAULT_ISOMETRIC_HEADROOM,
        metavar="FLOAT",
        help=(
            "fraction of the window's natural-pace capacity a translation may use "
            f"(0 < value <= 1; default {DEFAULT_ISOMETRIC_HEADROOM})"
        ),
        ui="number",
        ui_label="Isometric headroom (0-1]",
        ui_group="timing",
    ),
    OptionSpec(
        field="isometric_cps",
        flag="--isometric-cps",
        kind="float",
        default=DEFAULT_ISOMETRIC_CPS,
        metavar="FLOAT",
        help=(
            "natural Vietnamese speaking pace in characters per second used by "
            f"the budget (10-30; default {DEFAULT_ISOMETRIC_CPS:g})"
        ),
        ui="number",
        ui_label="Isometric chars/sec (10-30)",
        ui_group="timing",
    ),
    OptionSpec(
        field="isometric_short_coverage",
        flag="--isometric-short-coverage",
        kind="float",
        default=DEFAULT_ISOMETRIC_SHORT_COVERAGE,
        metavar="FLOAT",
        help=(
            "adaptive omission floor for tiny windows: a window below "
            "--isometric-short-window may accept a shorter text down to this word "
            f"coverage (0.5-0.75; default {DEFAULT_ISOMETRIC_SHORT_COVERAGE:g}); "
            "always recorded per unit"
        ),
        ui="number",
        ui_label="Isometric short-window coverage",
        ui_group="timing",
    ),
    OptionSpec(
        field="isometric_short_window",
        flag="--isometric-short-window",
        kind="float",
        default=DEFAULT_ISOMETRIC_SHORT_WINDOW,
        metavar="SECONDS",
        help=(
            "tiny-window threshold for the adaptive omission floor (strictly below "
            f"this many seconds; 0-3; default {DEFAULT_ISOMETRIC_SHORT_WINDOW:g})"
        ),
        ui="number",
        ui_label="Isometric short-window threshold (s)",
        ui_group="timing",
    ),
    # ---------------------------------------------------------------- QC
    OptionSpec(
        field="qc_final",
        flag="--qc-final",
        kind="bool",
        default=True,
        cli_action="opt",
        help=(
            "final QC gate: aggregate transcript/translation/subtitle verdicts "
            "and fresh output checks into a quality verdict (default: on; a "
            "fail makes the run exit non-zero)"
        ),
        ui="checkbox",
        ui_label="QC: final gate",
        ui_group="engines",
    ),
    OptionSpec(
        field="qc_tts",
        flag="--qc-tts",
        kind="bool",
        default=True,
        cli_action="opt",
        help=(
            "per-clip TTS QC: score every final fitted clip with NISQA-TTS, "
            "retry low-MOS clips once with changed settings and write review "
            "packets (default: on; needs the 'torchmetrics' dependency)"
        ),
        ui="checkbox",
        ui_label="QC: TTS",
        ui_group="engines",
    ),
    OptionSpec(
        field="tts_qc_mos_min",
        flag="--tts-qc-mos-min",
        kind="float",
        default=3.0,
        metavar="MOS",
        help=(
            "per-clip TTS QC: retry/review clips below this NISQA MOS (default 3.0)"
        ),
        ui="number",
        ui_label="TTS QC min MOS",
        ui_group="engines",
    ),
    # --------------------------------------------------- machine translation
    OptionSpec(
        field="mt_engine",
        flag="--mt-engine",
        kind="choice",
        default=DEFAULT_MT_ENGINE,
        choices=MT_ENGINES,
        help=(
            "machine-translation engine (default: envit5). 'ollama' runs the "
            "contextual translate->reflect->improve loop on a local Ollama "
            "model; every LLM output is gated by deterministic validators and "
            "falls back to envit5 per failed segment. 'hunyuan' runs the "
            "HY-MT1.5-7B translation specialist via a local Ollama server "
            "(official GGUF, opt-in); envit5 stays the default and per-segment "
            "fallback."
        ),
        ui="dropdown",
        ui_label="MT engine",
        ui_group="engines",
    ),
    OptionSpec(
        field="ollama_model",
        flag="--ollama-model",
        kind="str",
        default=DEFAULT_OLLAMA_MODEL,
        metavar="MODEL",
        help=(
            "Ollama model for --mt-engine ollama "
            f"(default: {DEFAULT_OLLAMA_MODEL}; pull it with 'ollama pull "
            f"{DEFAULT_OLLAMA_MODEL}')"
        ),
        ui="text",
        ui_label="Ollama model",
        ui_group="engines",
    ),
    OptionSpec(
        field="ollama_host",
        flag="--ollama-host",
        kind="str",
        default=DEFAULT_OLLAMA_HOST,
        metavar="URL",
        help=f"Ollama server base URL (default: {DEFAULT_OLLAMA_HOST})",
        ui="text",
        ui_label="Ollama host",
        ui_group="engines",
    ),
    OptionSpec(
        field="hunyuan_model",
        flag="--hunyuan-model",
        kind="str",
        default=DEFAULT_HUNYUAN_MODEL,
        metavar="MODEL",
        help=(
            "Hunyuan model for --mt-engine hunyuan "
            f"(default: {DEFAULT_HUNYUAN_MODEL}; pull it with 'ollama run "
            f"{DEFAULT_HUNYUAN_MODEL}')"
        ),
        ui="text",
        ui_label="Hunyuan model",
        ui_group="engines",
    ),
    OptionSpec(
        field="hunyuan_host",
        flag="--hunyuan-host",
        kind="str",
        default=DEFAULT_HUNYUAN_HOST,
        metavar="URL",
        help=f"Hunyuan server base URL (default: {DEFAULT_HUNYUAN_HOST})",
        ui="text",
        ui_label="Hunyuan host",
        ui_group="engines",
    ),
    OptionSpec(
        field="hunyuan_refine",
        flag="--hunyuan-refine",
        kind="tri",
        default=None,
        cli_action="opt",
        help=(
            "translation refine pass (M5-T2). ON by default for --mt-engine "
            "hunyuan: after the QC battery, still-flagged units get ONE cycle "
            "of Hunyuan candidate samples, accepted only when strictly better. "
            "--no-hunyuan-refine restores the previous (no-refine) behaviour; "
            "non-hunyuan engines never refine (default: auto)"
        ),
        ui="tri",
        ui_label="Hunyuan refine pass (auto/on/off)",
        ui_group="engines",
    ),
    OptionSpec(
        field="hunyuan_refine_max_units",
        flag="--hunyuan-refine-max-units",
        kind="int",
        default=40,
        metavar="N",
        help=(
            "cost guard for the refine pass: at most N still-flagged units are "
            "refined per run, the rest are recorded as budget skips "
            "(default: 40)"
        ),
        ui="number",
        ui_label="Refine max units",
        ui_group="engines",
    ),
    OptionSpec(
        field="fusion_model",
        flag="--fusion-model",
        kind="str",
        default="",
        metavar="MODEL",
        help=(
            "optional Hunyuan-MT-Chimera-7B GGUF model tag on the same Ollama "
            "host, used to fuse --hunyuan-refine candidates (no-op when empty; "
            "see the docs for the exact tag)"
        ),
        ui="text",
        ui_label="Fusion model",
        ui_group="engines",
    ),
    OptionSpec(
        field="hunyuan_samples",
        flag="--hunyuan-samples",
        kind="int",
        default=3,
        metavar="K",
        help="Hunyuan candidates sampled per refined row (default: 3)",
        ui="number",
        ui_label="Hunyuan refine samples",
        ui_group="engines",
    ),
    # ---------------------------------------------------------------- TTS
    OptionSpec(
        field="tts_engine",
        flag="--tts-engine",
        kind="choice",
        default="piper",
        choices=tuple(TTS_ENGINES),
        help=(
            "TTS engine (default: piper). 'vixtts' needs --allow-nc-models "
            "(CPML non-commercial weights)."
        ),
        ui="dropdown",
        ui_label="TTS engine",
    ),
    OptionSpec(
        field="voxcpm2_voice",
        flag="--voxcpm2-voice",
        kind="choice",
        default=DEFAULT_VOXCPM2_VOICE,
        choices=tuple(VOXCPM2_VOICE_PROFILES),
        help=(
            "VoxCPM2 voice-design prompt profile (default: "
            f"{DEFAULT_VOXCPM2_VOICE}). Only used with --tts-engine voxcpm2."
        ),
        ui="dropdown",
        ui_label="VoxCPM2 voice profile",
    ),
    OptionSpec(
        field="allow_nc_models",
        flag="--allow-nc-models",
        kind="bool",
        default=False,
        cli_action="true",
        help="allow non-commercial model weights (required for vixtts)",
        ui="checkbox",
        ui_label="Allow non-commercial models (vixtts)",
    ),
    OptionSpec(
        field="allow_lang_mismatch",
        flag="--allow-lang-mismatch",
        kind="bool",
        default=False,
        cli_action="true",
        help=(
            "skip the post-ASR language guard (allow a transcript that looks "
            "like the wrong language, e.g. a vi-specialist model on English)"
        ),
        ui="checkbox",
        ui_label="Allow language mismatch (override the ASR guard)",
    ),
    OptionSpec(
        field="ref_voice",
        flag="--ref-voice",
        kind="path",
        default=None,
        metavar="WAV",
        help="reference voice for cloning (ViXTTS; default: shipped sample)",
        ui=None,  # the UI keeps its hand-written Audio (upload/microphone) component
    ),
    OptionSpec(
        field="diarize",
        flag="--diarize",
        kind="bool",
        default=False,
        cli_action="true",
        help=(
            "detect speakers with pyannote 3.1 and give each a voice "
            "(needs HF_TOKEN + accepted gated models)"
        ),
        ui="checkbox",
        ui_label="Diarize speakers",
    ),
    OptionSpec(
        field="diarize_model",
        flag="",
        kind="str",
        default=DEFAULT_DIARIZE_MODEL,
        ui="text",
        ui_label="Diarize model (pyannote)",
        ui_group="engines",
    ),
    OptionSpec(
        # Programmatic-only (mix stage); no CLI flag, no UI widget — recorded
        # here so every JobConfig field is accounted for in one place.
        field="voice_over_gain",
        flag="",
        kind="float",
        default=0.15,
    ),
    OptionSpec(
        field="clone_speakers",
        flag="--clone-speakers",
        kind="bool",
        default=False,
        cli_action="true",
        help=(
            "clone each diarized speaker's own voice from the separated vocals "
            "(needs --diarize --tts-engine vixtts --allow-nc-models)"
        ),
        ui="checkbox",
        ui_label="Clone speakers (needs diarize + vixtts)",
    ),
    OptionSpec(
        field="voice_map",
        flag="--voice-map",
        kind="json-map",
        default={},
        metavar="JSON",
        help='per-speaker voice map, e.g. \'{"spk1": "25hours"}\'',
        transform="voice_map",
    ),
    OptionSpec(
        field="glossary_path",
        flag="--glossary",
        kind="path",
        default=None,
        metavar="PATH",
        help=(
            "TSV glossary of 'term<TAB>translation' rows applied to the MT "
            "output after translation (see README 'Translation glossary')"
        ),
        ui=None,  # the UI keeps its hand-written File component
    ),
    OptionSpec(
        field="pronounce_path",
        flag="--pronounce",
        kind="path",
        default=None,
        metavar="PATH",
        help=(
            "TSV pronunciation lexicon of 'written<TAB>spoken' rows applied to "
            "the TTS text only; subtitles keep the written form (see README "
            "'Pronunciation lexicon')"
        ),
        ui=None,  # the UI keeps its hand-written File component
    ),
    # ------------------------------------------------------------ subtitles
    OptionSpec(
        field="sub_position",
        flag="--sub-pos",
        dest="sub_pos",
        kind="choice",
        default=DEFAULT_SUB_POSITION,
        choices=SUB_POSITIONS,
        metavar="POS",
        help=(
            "subtitle position for the .ass export and --burn: "
            f"{'|'.join(SUB_POSITIONS)} (default {DEFAULT_SUB_POSITION})"
        ),
        ui="dropdown",
        ui_label="Subtitle position (subtitles-only / --burn)",
    ),
    OptionSpec(
        field="sub_font_size",
        flag="--sub-font-size",
        dest="sub_font_size",
        kind="int",
        default=DEFAULT_SUB_FONT_SIZE,
        metavar="N",
        help=(
            "subtitle font size in points for the .ass export and --burn "
            f"(default {DEFAULT_SUB_FONT_SIZE})"
        ),
        ui="number",
        ui_label="Subtitle font size (subtitles-only / --burn)",
    ),
    OptionSpec(
        field="sub_margin_v",
        flag="--sub-margin-v",
        dest="sub_margin_v",
        kind="int",
        default=None,
        metavar="N",
        help=(
            "subtitle vertical margin in pixels for the .ass export and --burn; "
            "default is per position (bottom/top 24, center 0)"
        ),
        ui="number",
        ui_label="Subtitle vertical margin (px, blank = per position)",
    ),
    # ----------------------------------------------------------------- mix
    OptionSpec(
        field="loudness_target_i",
        flag="--loudness-target",
        dest=_DEST_LOUD_I,
        kind="float",
        default=DEFAULT_LOUDNESS_I,
        metavar="LUFS",
        help=(
            f"loudnorm integrated loudness target (default {DEFAULT_LOUDNESS_I:g})"
        ),
        ui="number",
        ui_label="Loudness target (LUFS)",
    ),
    OptionSpec(
        field="loudness_tp",
        flag="--loudness-tp",
        dest="loudness_tp",
        kind="float",
        default=DEFAULT_LOUDNESS_TP,
        metavar="DBTP",
        help=f"loudnorm true-peak target (default {DEFAULT_LOUDNESS_TP:g})",
        ui="number",
        ui_label="True peak (dBTP)",
    ),
    OptionSpec(
        field="loudness_lra",
        flag="--loudness-lra",
        dest="loudness_lra",
        kind="float",
        default=DEFAULT_LOUDNESS_LRA,
        metavar="LU",
        help=f"loudnorm loudness range (default {DEFAULT_LOUDNESS_LRA:g})",
        ui="number",
        ui_label="Loudness range (LU)",
    ),
    # -------------------------------------------------------------- timing
    OptionSpec(
        field="speed_as_is",
        flag="--speed-as-is",
        kind="float",
        default=DEFAULT_SPEED_AS_IS,
        metavar="R",
        help=(
            "timing: do not stretch a segment within this ratio of its window "
            f"(default {DEFAULT_SPEED_AS_IS:g})"
        ),
        ui="number",
        ui_label="Speed as-is ratio",
        ui_group="timing",
    ),
    OptionSpec(
        field="speed_cap",
        flag="--speed-cap",
        kind="float",
        default=DEFAULT_SPEED_CAP,
        metavar="R",
        help=(
            "timing: hardest atempo a segment may receive before it is "
            f"re-translated (default {DEFAULT_SPEED_CAP:g})"
        ),
        ui="number",
        ui_label="Speed cap ratio",
        ui_group="timing",
    ),
    OptionSpec(
        field="retranslate",
        flag="--no-retranslate",
        kind="bool",
        default=True,
        cli_action="false",
        help=(
            "timing: disable the shortening re-translation retry (over-long "
            "segments are clipped instead)"
        ),
        ui="checkbox",
        ui_label="Shortening re-translation retry",
        ui_group="timing",
    ),
    OptionSpec(
        field="tail_pad",
        flag="--tail-pad",
        kind="float",
        default=DEFAULT_TAIL_PAD,
        metavar="S",
        help=(
            "timing: extend the FINAL segment's window past the video end by "
            "this many seconds so it is never cut mid-word "
            f"(default {DEFAULT_TAIL_PAD:g})"
        ),
        ui="number",
        ui_label="Tail pad (s)",
        ui_group="timing",
    ),
    OptionSpec(
        field="tts_rate_fit",
        flag="--tts-rate-fit",
        kind="bool",
        default=True,
        cli_action="opt",
        help=(
            "timing: preserve natural Piper pace for fitting speech and "
            "re-synthesize overflow faster (per-synthesis length_scale); "
            "--no-tts-rate-fit restores the single fixed rate (default on)"
        ),
        ui="checkbox",
        ui_label="Per-synthesis TTS rate fitting",
        ui_group="timing",
    ),
    OptionSpec(
        field="recenter_audio",
        flag="--recenter",
        dest=_DEST_RECENTER,
        kind="bool",
        default=True,
        cli_action="opt",
        help=(
            "timing: shift a fitted clip later into its window's slack instead "
            "of always starting at the segment start; --no-recenter disables it "
            "(default on)"
        ),
        ui="checkbox",
        ui_label="Re-center audio in window slack",
        ui_group="timing",
    ),
    OptionSpec(
        field="align_display",
        flag="--align-display",
        dest="align_display",
        kind="bool",
        default=True,
        cli_action="opt",
        help=(
            "timing: force-align each multi-member unit's synthesized audio so "
            "every member row gets a real, word-edge-snapped display span; "
            "--no-align-display skips the alignment ASR and uses the split "
            "fallback (default on)"
        ),
        ui="checkbox",
        ui_label="Aligned display spans",
        ui_group="timing",
    ),
)

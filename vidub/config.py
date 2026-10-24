"""Job configuration for a single ViDub run.

The fields mirror the CLI flags and the per-job manifest (``job.json``). Keep
this dataclass as the single source of truth for defaults so later stages can
assume sane values.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from vidub.utils.fit import SHORT_WINDOW_MAX, SPEED_AS_IS, SPEED_CAP
from vidub.utils.isometry import (
    BUDGET_HEADROOM,
    NATURAL_CHARS_PER_SECOND,
    SHORT_COVERAGE_MIN,
)
from vidub.utils.units_guard import UNIT_COVERAGE_MIN

__all__ = [
    "JobConfig",
    "ASR_PRESETS",
    "DEFAULT_ASR_MODEL",
    "WHISPER_SIZES",
    "MODES",
    "MODE_ALIASES",
    "SUBTITLES_ONLY",
    "normalize_mode",
    "SUB_POSITIONS",
    "DEFAULT_SUB_POSITION",
    "DEFAULT_SUB_FONT_SIZE",
    "SUB_POSITION_MARGIN_V",
    "normalize_sub_position",
    "MT_ENGINES",
    "DEFAULT_MT_ENGINE",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_HUNYUAN_MODEL",
    "DEFAULT_HUNYUAN_HOST",
    "TTS_ENGINES",
    "VOXCPM2_VOICE_PROFILES",
    "DEFAULT_VOXCPM2_VOICE",
    "TTS_LANGS",
    "DEFAULT_VOICE_MAP",
    "DEFAULT_DIARIZE_MODEL",
    "DEFAULT_LOUDNESS_I",
    "DEFAULT_LOUDNESS_TP",
    "DEFAULT_LOUDNESS_LRA",
    "DEFAULT_SPEED_AS_IS",
    "DEFAULT_SPEED_CAP",
    "DEFAULT_TAIL_PAD",
    "DEFAULT_ISOMETRIC",
    "DEFAULT_ISOMETRIC_HEADROOM",
    "DEFAULT_ISOMETRIC_CPS",
    "DEFAULT_ISOMETRIC_SHORT_COVERAGE",
    "DEFAULT_ISOMETRIC_SHORT_WINDOW",
    "ISOMETRIC_CPS_MIN",
    "ISOMETRIC_CPS_MAX",
    "asr_model_help",
    "resolve_asr_model",
    "check_tts_engine_allowed",
    "is_nc_tts_engine",
    "tts_engine_license",
    "validate_langs",
    "validate_clone_speakers",
    "check_mt_engine",
]

# ------------------------------------------------------------------- modes
# Canonical output modes. ``subtitles`` is an accepted alias for
# ``subtitles-only`` (the same mode); normalisation happens in ONE place,
# :func:`normalize_mode`, so the manifest/report always store the canonical
# ``subtitles-only`` string regardless of which spelling was requested.
SUBTITLES_ONLY = "subtitles-only"
MODES: tuple[str, ...] = ("dub", "voice-over", SUBTITLES_ONLY)
MODE_ALIASES: dict[str, str] = {"subtitles": SUBTITLES_ONLY}


def normalize_mode(value: str) -> str:
    """Normalise a mode spelling to its canonical name.

    ``"subtitles"`` and ``"subtitles-only"`` are the same mode; the canonical
    string (what the manifest and report always store) is ``"subtitles-only"``.
    Matching is case-insensitive and whitespace-trimmed. Raises
    :class:`ValueError` listing the valid modes for anything else, so the CLI,
    the UI and :class:`JobConfig` all fail fast with the same actionable message.
    """
    mode = (value or "").strip().lower()
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in MODES:
        valid = ", ".join(MODES)
        raise ValueError(f"unknown mode {value!r}: choose from {valid}")
    return mode


# --------------------------------------------------------------- subtitles
# SU-1-T2 subtitle styling. ``sub_position`` is the on-screen placement of the
# subtitle text; it maps to an ASS alignment (see ``vidub.stages.export``) and
# drives BOTH the ``.ass`` export and the burned-in video. The vertical margin
# defaults are per position: bottom/top keep the text clear of the frame edge,
# center needs no margin. ``--sub-font-size`` / ``--sub-margin-v`` override them.
SUB_POSITIONS: tuple[str, ...] = ("bottom", "top", "center")
DEFAULT_SUB_POSITION = "bottom"
DEFAULT_SUB_FONT_SIZE = 24
SUB_POSITION_MARGIN_V: dict[str, int] = {"bottom": 24, "top": 24, "center": 0}


def normalize_sub_position(value: str) -> str:
    """Normalise a ``--sub-pos`` value (case-insensitive) to its canonical name.

    Raises :class:`ValueError` listing the valid positions for anything else, so
    the CLI/UI fail fast with an actionable message.
    """
    position = (value or "").strip().lower()
    if position not in SUB_POSITIONS:
        valid = ", ".join(SUB_POSITIONS)
        raise ValueError(f"unknown subtitle position {value!r}: choose from {valid}")
    return position

# --------------------------------------------------------------------- ASR
# ASR preset name -> model id handed to the engine. Presets wrap faster-whisper
# (CTranslate2) sizes or Hugging Face repo ids — see README "ASR engines".
ASR_PRESETS: dict[str, str] = {
    "cpu-tiny": "tiny",
    "cpu-base": "base",
    "cpu-small": "small",
    "gpu-large": "large-v3",
    "vi-small": "vinai/PhoWhisper-small",
    "vi-medium": "vinai/PhoWhisper-medium",
}

# CLI default: pick the best model for this machine at run time.
DEFAULT_ASR_MODEL = "auto"

# Bare faster-whisper/CTranslate2 size names (mirrors faster_whisper.utils._MODELS).
WHISPER_SIZES: frozenset[str] = frozenset(
    {
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
        "distil-large-v2",
        "distil-large-v3",
        "distil-large-v3.5",
        "distil-medium.en",
        "distil-small.en",
        "large-v3-turbo",
        "turbo",
    }
)


def asr_model_help() -> str:
    """Human-readable list of accepted ``--asr-model`` values."""
    presets = ", ".join(ASR_PRESETS)
    sizes = "tiny, base, small, medium, large-v3"
    return (
        f"a preset ({presets}), a faster-whisper size ({sizes}), or a Hugging "
        "Face repo id ('org/model')"
    )


def _auto_asr_model() -> str:
    """``auto`` -> best model available: ``large-v3`` on GPU, else ``base``."""
    try:
        from vidub.utils.cuda import gpu_whisper_available

        gpu_ok, _reason = gpu_whisper_available()
    except Exception:  # noqa: BLE001 - a probe failure must not break resolution
        gpu_ok = False
    return "large-v3" if gpu_ok else "base"


def resolve_asr_model(value: str) -> str:
    """Resolve a preset name / bare size / repo id to a concrete model id.

    ``"auto"`` (and an empty value) resolves to ``large-v3`` when the GPU is
    usable, otherwise ``base``. Raises :class:`ValueError` listing the valid
    options when ``value`` is not recognised, so the CLI can fail fast.
    """
    model = (value or "").strip()
    if not model:
        model = DEFAULT_ASR_MODEL
    if model == "auto":
        return _auto_asr_model()
    if model in ASR_PRESETS:
        return ASR_PRESETS[model]
    if model in WHISPER_SIZES or "/" in model:
        return model
    raise ValueError(f"unknown ASR model {value!r}: use {asr_model_help()}.")


# ---------------------------------------------------------------------- MT
# Machine-translation engines selectable via ``--mt-engine``. ``envit5`` is the
# byte-identical default; ``ollama`` is the contextual LLM engine (M4-T5) that
# falls back to envit5 per failed segment; ``hunyuan`` (M5-T1) is the opt-in
# Tencent HY-MT1.5-7B translation specialist served the same way by Ollama.
# ``build_translator`` is the authority (it raises TranslationError for anything
# else); ``check_mt_engine`` lets the CLI fail fast with the same list.
MT_ENGINES: tuple[str, ...] = ("envit5", "ollama", "hunyuan")
DEFAULT_MT_ENGINE = "envit5"
# Default local Ollama model/server for ``--mt-engine ollama`` (Ollama 0.34+).
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
# Default local Ollama model/server for ``--mt-engine hunyuan`` (M5-T1): the
# official Tencent HY-MT1.5-7B GGUF, served by the same default host.
DEFAULT_HUNYUAN_MODEL = "hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K"
DEFAULT_HUNYUAN_HOST = DEFAULT_OLLAMA_HOST


def check_mt_engine(name: str) -> str:
    """Validate an ``--mt-engine`` value and return it lowercased.

    Raises :class:`ValueError` listing the valid choices for an unknown engine,
    so the CLI can fail fast before any stage runs.
    """
    engine = (name or DEFAULT_MT_ENGINE).strip().lower()
    if engine not in MT_ENGINES:
        valid = ", ".join(MT_ENGINES)
        raise ValueError(f"unknown mt engine {name!r}: choose from {valid}")
    return engine


# --------------------------------------------------------------------- TTS
# TTS engine registry. ``nc`` marks engines whose weights are non-commercial:
# they are refused unless ``--allow-nc-models`` / ``JobConfig.allow_nc_models``
# is set (the hard license guardrail from DESIGN.md §4.2).
TTS_ENGINES: dict[str, dict[str, object]] = {
    "piper": {"license": "MIT", "nc": False},
    "voxcpm2": {"license": "Apache-2.0", "nc": False},
    "vixtts": {"license": "CPML (non-commercial)", "nc": True},
}

# VoxCPM2 voice design uses a natural-language description, not model speaker IDs.
VOXCPM2_VOICE_PROFILES: dict[str, dict[str, str]] = {
    "vi_female_warm": {
        "label": "Nữ · ấm áp, tự nhiên",
        "style": "(Giọng nữ Việt Nam trưởng thành, ấm áp, rõ lời, nhịp kể chuyện tự nhiên, tốc độ vừa phải)",
    },
    "vi_male_calm": {
        "label": "Nam · trầm ấm, điềm tĩnh",
        "style": "(Giọng nam Việt Nam trưởng thành, trầm ấm, điềm tĩnh, rõ lời, tốc độ vừa phải)",
    },
    "vi_female_bright": {
        "label": "Nữ · trẻ trung, tươi sáng",
        "style": "(Giọng nữ Việt Nam trẻ trung, tươi sáng, thân thiện, rõ lời, tốc độ vừa phải)",
    },
    "vi_male_natural": {
        "label": "Nam · tự nhiên, rõ ràng",
        "style": "(Giọng nam Việt Nam tự nhiên, gần gũi, rõ lời, nhịp nói vừa phải)",
    },
}
DEFAULT_VOXCPM2_VOICE = "vi_female_warm"


def tts_engine_license(name: str) -> str:
    """Human-readable license for a TTS engine (``"unknown"`` if not registered)."""
    spec = TTS_ENGINES.get((name or "").strip().lower())
    return str(spec["license"]) if spec else "unknown"


def is_nc_tts_engine(name: str) -> bool:
    """True when ``name`` is a registered non-commercial TTS engine."""
    spec = TTS_ENGINES.get((name or "").strip().lower())
    return bool(spec and spec["nc"])


def check_tts_engine_allowed(name: str, allow_nc: bool) -> None:
    """Validate a TTS engine and enforce the non-commercial license guard.

    Raises :class:`ValueError` for an unknown engine, or for an NC engine when
    ``allow_nc`` is False — the message points at ``--allow-nc-models``.
    """
    engine = (name or "piper").strip().lower()
    if engine not in TTS_ENGINES:
        valid = ", ".join(TTS_ENGINES)
        raise ValueError(f"unknown tts engine {name!r}: choose from {valid}")
    if is_nc_tts_engine(engine) and not allow_nc:
        raise ValueError(
            f"tts engine {engine!r} uses non-commercial weights "
            f"({tts_engine_license(engine)}); pass --allow-nc-models to enable it"
        )


# Target languages each TTS engine can speak, by engine name (lowercase).
# Vietnamese is first-class everywhere; ``vixtts`` (XTTS-v2) additionally
# covers the 17 other languages listed on the upstream model card. ASR is
# multilingual (any whisper language) and envit5 MT is En↔Vi for now, so the
# TTS engine is the only hard constraint checked by :func:`validate_langs`.
TTS_LANGS: dict[str, frozenset[str]] = {
    "piper": frozenset({"vi"}),
    "voxcpm2": frozenset({"vi"}),
    "vixtts": frozenset(
        {
            "vi",
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "pl",
            "tr",
            "ru",
            "nl",
            "cs",
            "ar",
            "zh",
            "hu",
            "ko",
            "ja",
            "hi",
        }
    ),
}


def validate_langs(src_lang: str, tgt_lang: str, tts_engine: str) -> None:
    """Fail fast when ``tts_engine`` cannot speak ``tgt_lang``.

    ASR accepts any whisper language and envit5 is En↔Vi, so the TTS engine is
    the binding constraint. Vietnamese is first-class for every registered
    engine; any other target must be listed in :data:`TTS_LANGS`. Raises
    :class:`ValueError` naming the unsupported pair and pointing at the
    ``--tts-engine`` choices.
    """
    src = (src_lang or "").strip().lower() or "en"
    tgt = (tgt_lang or "").strip().lower() or "vi"
    engine = (tts_engine or "piper").strip().lower()

    if tgt == "vi":  # Vietnamese is first-class for every engine.
        return

    langs = TTS_LANGS.get(engine)
    if langs is None:
        valid = ", ".join(TTS_ENGINES)
        raise ValueError(f"unknown tts engine {tts_engine!r}: choose from {valid}")
    if tgt not in langs:
        supported = ", ".join(sorted(langs))
        choices = ", ".join(TTS_LANGS)
        raise ValueError(
            f"unsupported language pair {src}->{tgt} for --tts-engine "
            f"{engine!r}: '{engine}' can only dub target languages {supported}. "
            f"Vietnamese (vi) is first-class for every engine; for more target "
            f"languages choose --tts-engine vixtts (with --allow-nc-models). "
            f"TTS engines: {choices}."
        )


def validate_clone_speakers(cfg: "JobConfig") -> None:
    """Fail fast when per-speaker cloning is requested but cannot run.

    ``--clone-speakers`` needs speaker labels (``--diarize``) and the zero-shot
    cloning engine (``--tts-engine vixtts --allow-nc-models``). Raises
    :class:`ValueError` listing every missing flag so the message is actionable.

    In ``subtitles-only`` the speak stage never runs, so cloning cannot apply at
    all; that is reported as an explicit, actionable error (never a silent skip).
    """
    if not cfg.clone_speakers:
        return

    if normalize_mode(cfg.mode) == SUBTITLES_ONLY:
        raise ValueError(
            "--clone-speakers cannot run in mode 'subtitles-only': voice cloning "
            "happens in the speak stage, which this mode skips. Use --mode dub "
            "(or --mode voice-over), or drop --clone-speakers."
        )

    missing: list[str] = []
    if not cfg.diarize:
        missing.append("--diarize")
    if (cfg.tts_engine or "").strip().lower() != "vixtts":
        missing.append("--tts-engine vixtts")
    if not cfg.allow_nc_models:
        missing.append("--allow-nc-models")

    if missing:
        raise ValueError(
            "--clone-speakers needs a reference voice per speaker; also pass "
            + " ".join(missing)
            + ". Cloning requires speaker diarization (--diarize) and the "
            "zero-shot cloning engine (--tts-engine vixtts --allow-nc-models)."
        )


# --------------------------------------------------------------- diarization
# Speaker -> TTS voice name. spk0/spk1 map to the two shipped Piper vi voices;
# any extra speaker falls back to spk0's voice (see vidub.stages.speak).
DEFAULT_VOICE_MAP: dict[str, str] = {"spk0": "vais1000", "spk1": "25hours"}
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-3.1"

# --------------------------------------------------------------- loudness
# EBU R128 loudnorm targets (DESIGN.md §4.4): integrated loudness, true peak, LRA.
DEFAULT_LOUDNESS_I = -16.0
DEFAULT_LOUDNESS_TP = -1.5
DEFAULT_LOUDNESS_LRA = 11.0

# --------------------------------------------------------------- timing fit
# Timing-fit thresholds (see vidub.utils.fit); ``SPEED_CAP`` is the hardest
# atempo a segment may receive before it is re-translated and, as a last resort,
# clipped. ``retranslate`` disables the shortening retry (``--no-retranslate``).
DEFAULT_SPEED_AS_IS = SPEED_AS_IS
DEFAULT_SPEED_CAP = SPEED_CAP
# Tail-safety (M3-C2): the final segment's availability window is extended past
# the video end by this many seconds. Nothing follows the last segment, so a mild
# audio overrun past the last frame is acceptable mux behavior (see timing stage).
DEFAULT_TAIL_PAD = 2.0

# ------------------------------------------------------- isometric budgeting
# ISO-1 isometric length budgeting (translation-level window fitting). The
# default headroom and characters-per-second come from the shared isometry
# calibration (``vidub.utils.isometry``) rather than being re-declared here.
# ``--isometric-headroom`` is the fraction of the window's natural-pace capacity
# a translation may use (must be in (0, 1]); ``--isometric-cps`` is the
# characters-per-second pace (bounded to a sane [10, 30] range).
DEFAULT_ISOMETRIC_HEADROOM = BUDGET_HEADROOM
DEFAULT_ISOMETRIC_CPS = NATURAL_CHARS_PER_SECOND
ISOMETRIC_CPS_MIN = 10.0
ISOMETRIC_CPS_MAX = 30.0
# ISO-1 close (v0.11.0): the pass ships OFF by default. The matched A/B on the
# reference video (identical structure, 83 units each) showed the pre-registered
# targets NOT MET — clips unchanged (2 -> 2, target 0) and `stretched` worse
# (19 -> 27, target <= 12, i.e. worse than the matched control) — so the feature
# stays available via `--isometric` for further work but is never on unless the
# user asks for it. `JobConfig.isometric` and the CLI argparse default both read
# this single constant, so the two surfaces can never disagree.
DEFAULT_ISOMETRIC = False
# ISO-1-T3-C1 adaptive floor: below `DEFAULT_ISOMETRIC_SHORT_WINDOW` seconds the
# omission floor drops to `DEFAULT_ISOMETRIC_SHORT_COVERAGE` (a bounded,
# User-approved meaning/pacing trade). Bounds: coverage in [0.5, UNIT_COVERAGE_MIN]
# (never above the normal floor), window in (0, 3] seconds.
DEFAULT_ISOMETRIC_SHORT_COVERAGE = SHORT_COVERAGE_MIN
DEFAULT_ISOMETRIC_SHORT_WINDOW = SHORT_WINDOW_MAX
ISOMETRIC_SHORT_COVERAGE_MIN = 0.5
ISOMETRIC_SHORT_WINDOW_MAX = 3.0


@dataclass
class JobConfig:
    """Configuration for one ViDub job (En→Vi dubbing by default)."""

    src_lang: str = "en"
    tgt_lang: str = "vi"
    mode: str = "dub"
    # Programmatic fallback; the CLI defaults to DEFAULT_ASR_MODEL ("auto").
    asr_model: str = "base"
    asr_words: bool = True
    # Opt-in punctuation/casing restoration of the SOURCE transcript
    # (QUALITY-ROADMAP.md P1.6): a small token-classification model inserts
    # periods/commas/question marks and basic casing into ASR text that lacks
    # punctuation. Default off; enable with --restore-punct. When off the
    # transcribe stage is byte-identical to before.
    restore_punct: bool = False
    # Level-1 transcript QC (QUALITY-ROADMAP.md §3.2): classify ASR segments,
    # retry flagged windows once and record a transcript verdict. Default on;
    # disable with --no-qc-transcript.
    qc_transcript: bool = True
    # Subtitle standards QC (QUALITY-ROADMAP.md G3): measure the Vietnamese
    # subtitles against the Netflix-derived profile, auto-wrap/re-split the SRT
    # and count violations. Affects only the .srt deliverable. Default on;
    # disable with --no-subtitle-qc.
    qc_subtitles: bool = True
    # Level-2 translation QC battery (QUALITY-ROADMAP.md §3.3): deterministic
    # MQM-aligned signals, glossary-splice corrections and a single
    # re-translation retry, plus a translation verdict. Default on; disable with
    # --no-translation-qc.
    qc_translation: bool = True
    # TU-1: sentence-level translation units, default on; --no-sentence-units
    # restores per-segment translation.
    sentence_units: bool = True
    # ISO-1 isometric length budgeting (sentence-unit path only): compute each
    # unit's character budget from its locked window and (a) pass it to the
    # engine as a hint, then (b) run ONE bounded budgeted re-translation for the
    # units still over budget, accepted only by ``vidub.utils.isometry``'s
    # deterministic ``accept_shrunk`` predicate. Default OFF (DEFAULT_ISOMETRIC)
    # since v0.11.0: the matched A/B kept clips unchanged (2 -> 2) and made
    # `stretched` worse (19 -> 27, target <= 12), so the pass is opt-in via
    # --isometric. When off the path is byte-identical to before, and the legacy
    # (--no-sentence-units) path never uses it.
    isometric: bool = DEFAULT_ISOMETRIC
    # Fraction of the window's natural-pace capacity a translation may use
    # (0 < headroom <= 1).
    isometric_headroom: float = DEFAULT_ISOMETRIC_HEADROOM
    # Characters per second of natural Vietnamese speech (the isometry pace).
    isometric_cps: float = DEFAULT_ISOMETRIC_CPS
    # ISO-1-T3-C1 adaptive omission floor: a window below
    # `isometric_short_window` seconds may accept a shorter text down to this
    # coverage (a bounded, User-approved meaning/pacing trade, always recorded
    # per unit as `coverage_floor` / `short_floor_applied`). Bounded:
    # isometric_short_coverage in [0.5, UNIT_COVERAGE_MIN].
    isometric_short_coverage: float = DEFAULT_ISOMETRIC_SHORT_COVERAGE
    # The tiny-window threshold for the adaptive floor (strictly below this many
    # seconds); bounded in (0, 3].
    isometric_short_window: float = DEFAULT_ISOMETRIC_SHORT_WINDOW
    # Per-clip TTS QC (QUALITY-ROADMAP.md §3.4 / P1.7): score every final fitted
    # clip with NISQA-TTS, retry low-MOS clips once with changed settings and
    # record counters + review packets. Default on; disable with --no-qc-tts.
    qc_tts: bool = True
    # MOS (1-5) below which a fitted clip is retried/reviewed.
    tts_qc_mos_min: float = 3.0
    # Final QC gate (QUALITY-ROADMAP.md §3.4): aggregate the per-level verdicts
    # and fresh output checks into one ``quality`` verdict in report.json. A
    # "fail" makes ``vidub run`` exit non-zero. Default on; disable with
    # --no-qc-final.
    qc_final: bool = True
    mt_engine: str = DEFAULT_MT_ENGINE
    # Ollama LLM MT engine settings (used when mt_engine == "ollama").
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    # Hunyuan HY-MT1.5-7B settings (M5-T1, opt-in; envit5 stays default).
    hunyuan_model: str = DEFAULT_HUNYUAN_MODEL
    hunyuan_host: str = DEFAULT_HUNYUAN_HOST
    # M5-T2 refine pass: after the QC battery, still-flagged units get ONE
    # best-of-K refine cycle whose winner is accepted only when STRICTLY better.
    # Tri-state, default-on for the hunyuan engine:
    #   None  (default) -> auto: the pass runs when mt_engine == "hunyuan";
    #   True            -> force on  (explicit --hunyuan-refine);
    #   False           -> force off (--no-hunyuan-refine).
    # The stage only ever builds/uses it for the hunyuan engine, so envit5 and
    # ollama runs are byte-identical to before.
    hunyuan_refine: bool | None = None
    fusion_model: str = ""
    hunyuan_samples: int = 3
    # Cost guard: at most this many still-flagged units get a refine cycle per
    # run; the rest are recorded as ``refine-budget`` skips (see refine.py).
    hunyuan_refine_max_units: int = 40
    tts_engine: str = "piper"
    # ViXTTS et al. ship non-commercial weights -> opt in explicitly.
    allow_nc_models: bool = False
    # Allow an ASR transcript that looks like the wrong language (e.g. a
    # Vietnamese-specialist model hallucinating Vietnamese on English audio).
    allow_lang_mismatch: bool = False
    ref_voice: str | None = None
    # Optional TSV glossary (term<TAB>translation) applied to the MT output
    # after translation; None disables it. See vidub.utils.glossary.
    glossary_path: str | None = None
    # Optional TSV pronunciation lexicon (written<TAB>spoken) applied to the TTS
    # text only (subtitles keep the written form); None disables it. See
    # vidub.utils.pronounce.
    pronounce_path: str | None = None
    # Optional speaker diarization (pyannote 3.1, HF-gated).
    diarize: bool = False
    diarize_model: str = DEFAULT_DIARIZE_MODEL
    # Per-speaker voice cloning: extract a clean reference per diarized speaker
    # from the separated vocals and feed it to ViXTTS. Requires diarize + vixtts
    # (see validate_clone_speakers).
    clone_speakers: bool = False
    # Per-speaker voice overrides; merged over DEFAULT_VOICE_MAP at speak time.
    voice_map: dict[str, str] = field(default_factory=dict)
    voice_over_gain: float = 0.15
    # EBU R128 loudnorm targets for the mix stage.
    loudness_target_i: float = DEFAULT_LOUDNESS_I
    loudness_tp: float = DEFAULT_LOUDNESS_TP
    loudness_lra: float = DEFAULT_LOUDNESS_LRA
    # Timing-fit policy (see vidub.utils.fit): speed_as_is is the untouch
    # threshold, speed_cap the hardest atempo a segment may receive before it is
    # re-translated. retranslate disables the shortening retry (--no-retranslate).
    speed_as_is: float = DEFAULT_SPEED_AS_IS
    speed_cap: float = DEFAULT_SPEED_CAP
    retranslate: bool = True
    # Extra seconds added to the FINAL segment's window (past the video end) so
    # the last sentence is never cut mid-word (--tail-pad).
    tail_pad: float = DEFAULT_TAIL_PAD
    # FIT-1 per-synthesis rate fit: ask Piper to speak each segment closer to its
    # window (length_scale) instead of one fixed rate; default on, disable with
    # --no-tts-rate-fit.
    tts_rate_fit: bool = True
    # FIT-1 recenter: shift a fitted clip later into its window's slack instead of
    # always starting at the segment start; default on, disable with --no-recenter.
    recenter_audio: bool = True
    # SA-1-T3 display alignment: force-align each multi-member unit's synthesized
    # WAV and record a real, word-edge-snapped span per member row in
    # ``timed.json``'s ``member_spans``; default on, disable with
    # --no-align-display (which skips the ASR and uses the split fallback).
    align_display: bool = True
    # SU-1-T2 subtitle styling (see the "subtitles" section above): on-screen
    # position, font size and vertical margin. ``sub_margin_v`` defaults to None
    # so ``__post_init__`` can apply the per-position default (bottom/top 24,
    # center 0) unless the caller set one explicitly.
    sub_position: str = DEFAULT_SUB_POSITION
    sub_font_size: int = DEFAULT_SUB_FONT_SIZE
    sub_margin_v: int | None = None
    cache_dir: str = ".vidub-cache"
    # VoxCPM2 voice-design prompt profile; last to preserve positional callers.
    voxcpm2_voice: str = DEFAULT_VOXCPM2_VOICE

    def __post_init__(self) -> None:
        # Single canonical mode spelling for the whole system: ``subtitles`` is
        # normalised to ``subtitles-only`` here (and an unknown mode fails fast
        # with the valid choices) so every caller — CLI builder, UI form and
        # ``from_dict`` — gets the same canonical value without duplicating logic.
        self.mode = normalize_mode(self.mode)
        # SU-1-T2 subtitle style: canonical position, validated size/margin, and
        # the per-position default margin when the caller did not set one.
        self.sub_position = normalize_sub_position(self.sub_position)
        try:
            self.sub_font_size = int(self.sub_font_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"subtitle font size must be an integer, got "
                f"{self.sub_font_size!r}"
            ) from exc
        if self.sub_font_size <= 0:
            raise ValueError(
                f"subtitle font size must be positive, got {self.sub_font_size}"
            )
        if self.sub_margin_v is None:
            self.sub_margin_v = SUB_POSITION_MARGIN_V[self.sub_position]
        try:
            self.sub_margin_v = int(self.sub_margin_v)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"subtitle vertical margin must be an integer, got "
                f"{self.sub_margin_v!r}"
            ) from exc
        if self.sub_margin_v < 0:
            raise ValueError(
                f"subtitle vertical margin must be >= 0, got {self.sub_margin_v}"
            )
        # ISO-1 isometric budgeting bounds: a headroom outside (0, 1] would make
        # the character budget meaningless, and a pace outside [10, 30] cps is a
        # misconfiguration (natural Vietnamese is ~18-25 cps). Fail fast with an
        # actionable message so the CLI/UI report the bad value.
        try:
            self.isometric_headroom = float(self.isometric_headroom)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"isometric headroom must be a number, got {self.isometric_headroom!r}"
            ) from exc
        if not (0.0 < self.isometric_headroom <= 1.0):
            raise ValueError(
                "isometric headroom must be in (0, 1] (a fraction of the window's "
                f"natural-pace capacity), got {self.isometric_headroom}"
            )
        try:
            self.isometric_cps = float(self.isometric_cps)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "isometric chars-per-second must be a number, got "
                f"{self.isometric_cps!r}"
            ) from exc
        if not (ISOMETRIC_CPS_MIN <= self.isometric_cps <= ISOMETRIC_CPS_MAX):
            raise ValueError(
                "isometric chars-per-second must be in "
                f"[{ISOMETRIC_CPS_MIN:g}, {ISOMETRIC_CPS_MAX:g}] (natural Vietnamese "
                f"is ~18-25 cps), got {self.isometric_cps}"
            )
        # ISO-1-T3-C1 adaptive floor bounds. The short coverage may only lower the
        # normal floor (never raise it) and must stay meaningfully bounded; the
        # short window is a tiny-window threshold.
        try:
            self.isometric_short_coverage = float(self.isometric_short_coverage)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "isometric short-window coverage must be a number, got "
                f"{self.isometric_short_coverage!r}"
            ) from exc
        if not (ISOMETRIC_SHORT_COVERAGE_MIN <= self.isometric_short_coverage <= UNIT_COVERAGE_MIN):
            raise ValueError(
                "isometric short-window coverage must be in "
                f"[{ISOMETRIC_SHORT_COVERAGE_MIN:g}, {UNIT_COVERAGE_MIN:g}] (at most "
                "the normal omission floor), got "
                f"{self.isometric_short_coverage}"
            )
        try:
            self.isometric_short_window = float(self.isometric_short_window)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "isometric short-window threshold must be a number, got "
                f"{self.isometric_short_window!r}"
            ) from exc
        if not (0.0 < self.isometric_short_window <= ISOMETRIC_SHORT_WINDOW_MAX):
            raise ValueError(
                "isometric short-window threshold must be in "
                f"(0, {ISOMETRIC_SHORT_WINDOW_MAX:g}] seconds, got "
                f"{self.isometric_short_window}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobConfig":
        """Build a config from ``data``, ignoring unknown keys.

        This lets manifests gain fields over time without breaking older code
        that reads them.
        """
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

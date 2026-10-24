"""Local Gradio web UI for ViDub (``vidub ui``).

A **thin layer** over :func:`vidub.pipeline.run_pipeline`: the UI only maps form
values to a :class:`~vidub.config.JobConfig`, starts the pipeline in a background
thread, and polls a small state dict for progress. No stage logic is duplicated,
and outputs go to the normal ``output/`` + cache locations (shared with the CLI).
The "Quality results" panel is only a tolerant read of the QC sidecars + report.

Cancellation is cooperative: the run passes ``cancel_cb`` to ``run_pipeline``
(checked between stages) and reports ``cancelled`` when it raises
:class:`~vidub.pipeline.PipelineCancelled`.

Gradio is an optional dependency (``pip install -e ".[ui]"``); importing this
module without it raises :class:`ImportError`, which :mod:`vidub.cli` turns into
an actionable message.
"""

from __future__ import annotations

import functools
import json
import logging
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from pathlib import Path

import gradio as gr

from vidub.config import (
    ASR_PRESETS,
    DEFAULT_ASR_MODEL,
    DEFAULT_HUNYUAN_HOST,
    DEFAULT_HUNYUAN_MODEL,
    DEFAULT_LOUDNESS_I,
    DEFAULT_LOUDNESS_LRA,
    DEFAULT_LOUDNESS_TP,
    DEFAULT_MT_ENGINE,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SUB_FONT_SIZE,
    DEFAULT_SUB_POSITION,
    DEFAULT_TAIL_PAD,
    DEFAULT_VOXCPM2_VOICE,
    MT_ENGINES,
    SUB_POSITIONS,
    SUBTITLES_ONLY,
    TTS_ENGINES,
    VOXCPM2_VOICE_PROFILES,
    JobConfig,
    check_mt_engine,
    check_tts_engine_allowed,
    normalize_mode,
    resolve_asr_model,
    validate_clone_speakers,
    validate_langs,
)
from vidub.manifest import JobManifest
from vidub.pipeline import PipelineCancelled, STAGES, required_stages, run_pipeline
from vidub.stages import export as export_mod
from vidub.stages.ingest import video_hash
from vidub.utils.text_selection import selected_text
from vidub.utils.ytdlp import download, is_url

__all__ = [
    "build_app",
    "launch_ui",
    "form_to_config",
    "resolve_input",
    "new_state",
    "snapshot",
    "poll",
    "cancel",
    "segment_audio",
    "UI_EXTRA_HINT",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
]

log = logging.getLogger(__name__)

UI_EXTRA_HINT = 'pip install -e ".[ui]"'
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
POLL_SECONDS = 1.0
ASR_MODEL_CHOICES = [
    DEFAULT_ASR_MODEL,
    *ASR_PRESETS,
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
]
TTS_ENGINE_CHOICES = list(TTS_ENGINES)
# Underlying mode values (kept stable — the config stores only these).
MODE_CHOICES = ["dub", "voice-over", "subtitles-only"]
# Display labels for the mode dropdown; Gradio maps each to its value. The
# subtitles-only label says up front that no TTS runs (SU-1-T3).
MODE_CHOICES_LABELED = [
    ("Dub (replace the vocals)", "dub"),
    ("Voice-over (keep the original audio ducked)", "voice-over"),
    ("Subtitles only (no TTS)", "subtitles-only"),
]
# Ollama pre-flight probe timeout (seconds).
OLLAMA_TIMEOUT = 3.0
# The Chimera fusion tag the M5 benchmark evaluated; it was measured as a
# *regression*, so the preset offers it for evaluation only.
FUSION_MODEL = "hf.co/mradermacher/Hunyuan-MT-Chimera-7B-GGUF:Q4_K_M"
# "Force full re-run" ticks every stage the force control can offer.
FORCE_ALL_STAGES = [*STAGES, "diarize", "references"]


# --------------------------------------------------------------------- state
def new_state() -> dict:
    """Fresh UI state dict (created once per app and mutated by the worker)."""
    return {
        "status": "idle",
        "message": "Ready.",
        "stage": "",
        "expected_stages": [],
        # SU-1-T3: the run mode + subtitle style the last run used (shown in the
        # form/panel; kept even while idle so the poll can render honestly).
        "mode": "dub",
        "sub_position": DEFAULT_SUB_POSITION,
        "sub_font_size": DEFAULT_SUB_FONT_SIZE,
        "burn": False,
        "manifest_path": None,
        "artifacts_dir": None,
        "stages": {},
        "script_rows": [],
        "segment_files": {},
        "segment_choices": [],
        "segment_spans": {},
        "_dd_choices": [],
        "video": None,
        "download": None,
        "report": None,
        "qc_summary": "",
        "error": None,
        "cancel": False,
        "started_at": None,
        "thread": None,
    }


# ------------------------------------------------------------- form mapping
def resolve_input(video_file, input_url) -> str:
    """Pick the run input: a pasted URL wins over an uploaded file."""
    url = str(input_url or "").strip()
    if url:
        return url
    if video_file:
        return str(video_file)
    raise ValueError("provide a video file to upload or a URL to download")


def form_to_config(
    *,
    mode: str = "dub",
    sub_position: str = DEFAULT_SUB_POSITION,
    sub_font_size: int = DEFAULT_SUB_FONT_SIZE,
    asr_model: str = DEFAULT_ASR_MODEL,
    tts_engine: str = "piper",
    allow_nc_models: bool = False,
    diarize: bool = False,
    clone_speakers: bool = False,
    glossary_file=None,
    loudness_target: float | None = None,
    loudness_tp: float | None = None,
    loudness_lra: float | None = None,
    cache_dir: str | None = None,
    mt_engine: str = DEFAULT_MT_ENGINE,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    hunyuan_model: str = DEFAULT_HUNYUAN_MODEL,
    hunyuan_host: str = DEFAULT_HUNYUAN_HOST,
    hunyuan_refine: bool = False,
    hunyuan_samples: int = 3,
    fusion_model: str = "",
    sentence_units: bool = True,
    restore_punct: bool = False,
    qc_transcript: bool = True,
    qc_subtitles: bool = True,
    qc_translation: bool = True,
    qc_tts: bool = True,
    qc_final: bool = True,
    tts_qc_mos_min: float = 3.0,
    src_lang: str = "en",
    tgt_lang: str = "vi",
    allow_lang_mismatch: bool = False,
    pronounce_path=None,
    tail_pad: float = DEFAULT_TAIL_PAD,
    ref_voice: str | None = None,
    voxcpm2_voice: str = DEFAULT_VOXCPM2_VOICE,
) -> JobConfig:
    """Map the UI form to a validated :class:`JobConfig`.

    Reuses the CLI validators (``resolve_asr_model``, ``check_mt_engine``,
    ``check_tts_engine_allowed``, ``validate_langs``, ``validate_clone_speakers``)
    so the UI fails with the same actionable messages the CLI shows. Every
    M4/M5 engine + QC field maps straight onto the matching :class:`JobConfig`
    field; no defaults are duplicated here beyond the dataclass ones.
    """
    resolve_asr_model(asr_model)  # raises ValueError on an unknown model
    mt_engine = check_mt_engine(mt_engine)  # raises on an unknown engine
    # In subtitles-only no synthesis runs, so the TTS engine config (and its
    # license guard) is irrelevant and must not block the run — mirrors the
    # CLI (`_build_cfg`) and pipeline guards (SU-1-T1).
    if normalize_mode(mode) != SUBTITLES_ONLY:
        check_tts_engine_allowed(tts_engine, allow_nc_models)
    if (
        tts_engine.strip().lower() == "voxcpm2"
        and voxcpm2_voice not in VOXCPM2_VOICE_PROFILES
    ):
        raise ValueError(f"unknown VoxCPM2 voice profile: {voxcpm2_voice!r}")

    cfg = JobConfig(
        mode=mode,
        sub_position=sub_position,
        sub_font_size=sub_font_size,
        asr_model=asr_model,
        tts_engine=tts_engine,
        voxcpm2_voice=voxcpm2_voice,
        allow_nc_models=bool(allow_nc_models),
        diarize=bool(diarize),
        clone_speakers=bool(clone_speakers),
        glossary_path=str(glossary_file) if glossary_file else None,
        loudness_target_i=(
            DEFAULT_LOUDNESS_I if loudness_target is None else float(loudness_target)
        ),
        loudness_tp=DEFAULT_LOUDNESS_TP if loudness_tp is None else float(loudness_tp),
        loudness_lra=(
            DEFAULT_LOUDNESS_LRA if loudness_lra is None else float(loudness_lra)
        ),
        mt_engine=mt_engine,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
        hunyuan_model=hunyuan_model,
        hunyuan_host=hunyuan_host,
        hunyuan_refine=bool(hunyuan_refine),
        hunyuan_samples=int(hunyuan_samples),
        fusion_model=fusion_model,
        sentence_units=bool(sentence_units),
        restore_punct=bool(restore_punct),
        qc_transcript=bool(qc_transcript),
        qc_subtitles=bool(qc_subtitles),
        qc_translation=bool(qc_translation),
        qc_tts=bool(qc_tts),
        qc_final=bool(qc_final),
        tts_qc_mos_min=float(tts_qc_mos_min),
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        allow_lang_mismatch=bool(allow_lang_mismatch),
        pronounce_path=str(pronounce_path) if pronounce_path else None,
        tail_pad=float(tail_pad),
        ref_voice=(
            str(ref_voice)
            if ref_voice and tts_engine.strip().lower() == "voxcpm2"
            else None
        ),
    )
    if cfg.ref_voice and not Path(cfg.ref_voice).is_file():
        raise ValueError(f"voice reference not found: {cfg.ref_voice}")
    if cache_dir and str(cache_dir).strip():
        cfg.cache_dir = str(cache_dir).strip()

    if cfg.mode != SUBTITLES_ONLY:
        validate_langs(cfg.src_lang, cfg.tgt_lang, cfg.tts_engine)
    validate_clone_speakers(cfg)
    return cfg


# ------------------------------------------------------- Ollama pre-flight
def _http_get_json(url: str, timeout: float) -> dict:
    """GET ``url`` and decode a JSON object (stdlib only; raises on failure).

    Thin, monkeypatchable seam so the pre-flight tests never touch the network.
    """
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object from {url}")
    return data


def _ollama_ready(host: str, model: str) -> tuple[bool, str]:
    """Probe ``host`` for the Ollama model ``model``; never raises.

    Ready when an entry in ``{"models": [{"name": ...}]}`` matches ``model``
    exactly **or** shares its base name without the ``:tag`` suffix (the server
    reports tags like ``hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K``). On any failure
    returns ``(False, <actionable detail>)`` naming the host and the model and
    the fix commands (``ollama serve`` / ``ollama run <model>``).
    """
    base = (host or DEFAULT_OLLAMA_HOST).rstrip("/")
    wanted = (model or "").strip()
    try:
        data = _http_get_json(f"{base}/api/tags", OLLAMA_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - any probe failure blocks with advice
        return (
            False,
            f"the local Ollama server at {base} is not reachable ({exc}); start "
            f"it with `ollama serve`, then make the model available with "
            f"`ollama run {wanted}`.",
        )
    names = [
        str(entry.get("name", ""))
        for entry in (data.get("models") or [])
        if isinstance(entry, dict)
    ]
    wanted_base = wanted.split(":", 1)[0]
    if wanted and (
        wanted in names
        or any(name.split(":", 1)[0] == wanted_base for name in names)
    ):
        return True, f"Ollama model {wanted!r} is available at {base}."
    return (
        False,
        f"the Ollama server at {base} does not serve model {wanted!r}; make it "
        f"available with `ollama run {wanted}` and retry.",
    )


def _preflight_error(cfg: JobConfig) -> str | None:
    """Blocking message for a missing Ollama dependency, else ``None``.

    Probes only when the run actually needs Ollama: the Hunyuan engine or the
    refine pass (both use the local Hunyuan generator), and the Ollama engine.
    ``envit5`` with refine off never probes.
    """
    if cfg.mt_engine == "hunyuan" or cfg.hunyuan_refine:
        ready, detail = _ollama_ready(cfg.hunyuan_host, cfg.hunyuan_model)
        if not ready:
            return (
                f"cannot start the run: {detail} The Hunyuan MT engine cannot "
                "translate (and the refine pass would be skipped) without it."
            )
    if cfg.mt_engine == "ollama":
        ready, detail = _ollama_ready(cfg.ollama_host, cfg.ollama_model)
        if not ready:
            return (
                f"cannot start the run: {detail} The Ollama MT engine cannot "
                "translate without it."
            )
    return None


# ------------------------------------------------------------------- presets
def preset_hunyuan() -> tuple:
    """One-click Hunyuan engine preset (form only, no run)."""
    return (
        gr.update(value="hunyuan"),
        gr.update(value=DEFAULT_HUNYUAN_MODEL),
        gr.update(value=DEFAULT_HUNYUAN_HOST),
        gr.update(value=False),
        gr.update(value=""),
        gr.update(
            value="Hunyuan HY-MT1.5-7B selected (local Ollama). Changing the MT "
            "engine re-runs `translate` and everything downstream automatically."
        ),
    )


def preset_refine() -> tuple:
    """One-click refine-pass preset (form only, no run)."""
    return (
        gr.update(value=True),
        gr.update(value=3),
        gr.update(value=""),
        gr.update(
            value="Refine pass on (3 samples). Still-flagged rows are re-sampled "
            "and kept only when strictly better."
        ),
    )


def preset_fusion() -> tuple:
    """One-click Chimera-fusion preset — evaluation only (form only, no run)."""
    return (
        gr.update(value=True),
        gr.update(value=FUSION_MODEL),
        gr.update(
            value="Chimera fusion selected for **evaluation only** — the M5 "
            "benchmark measured it as a *regression*, so it is not recommended."
        ),
    )


def preset_defaults() -> tuple:
    """One-click restore of the default envit5 configuration (form only)."""
    return (
        gr.update(value="envit5"),
        gr.update(value=False),
        gr.update(value=""),
        gr.update(value="Defaults restored: envit5 engine, refine off, no fusion."),
    )


def preset_force_all() -> tuple:
    """One-click "force full re-run": tick every stage the control offers."""
    return (gr.update(value=list(FORCE_ALL_STAGES)),)


# ------------------------------------------------------------- run plumbing
def _handle_run(
    state: dict,
    video_file,
    input_url,
    mode,
    sub_position,
    sub_font_size,
    burn_subtitles,
    asr_model,
    tts_engine,
    allow_nc_models,
    diarize,
    clone_speakers,
    glossary_file,
    loudness_target,
    loudness_tp,
    loudness_lra,
    cache_dir,
    mt_engine: str = DEFAULT_MT_ENGINE,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    hunyuan_model: str = DEFAULT_HUNYUAN_MODEL,
    hunyuan_host: str = DEFAULT_HUNYUAN_HOST,
    hunyuan_refine: bool = False,
    hunyuan_samples: int = 3,
    fusion_model: str = "",
    sentence_units: bool = True,
    restore_punct: bool = False,
    qc_transcript: bool = True,
    qc_subtitles: bool = True,
    qc_translation: bool = True,
    qc_tts: bool = True,
    qc_final: bool = True,
    tts_qc_mos_min: float = 3.0,
    force_stages: list[str] | None = None,
    src_lang: str = "en",
    tgt_lang: str = "vi",
    allow_lang_mismatch: bool = False,
    pronounce_path=None,
    tail_pad: float = DEFAULT_TAIL_PAD,
    ref_voice_path=None,
    voxcpm2_voice: str = DEFAULT_VOXCPM2_VOICE,
) -> tuple:
    """Run-button handler: validate, start the worker thread, return a snapshot.

    Called positionally from the ``inputs=[...]`` lambda, so the parameter order
    here must match the order of that list in :func:`build_app`. ``force_stages``
    names stages to re-run; ``None``/empty keeps today's plain resume behavior.
    """
    existing = state.get("thread")
    if existing is not None and existing.is_alive():
        state["message"] = (
            "A run is already in progress. "
            f"(elapsed {_format_elapsed(_elapsed_seconds(state))})"
        )
        return snapshot(state)
    # No live worker: clear any stale reference/flags so the UI can never lock.
    state["thread"] = None
    state["started_at"] = None

    try:
        video = resolve_input(video_file, input_url)
        cfg = form_to_config(
            mode=mode,
            sub_position=sub_position,
            sub_font_size=sub_font_size,
            asr_model=asr_model,
            tts_engine=tts_engine,
            allow_nc_models=allow_nc_models,
            diarize=diarize,
            clone_speakers=clone_speakers,
            glossary_file=glossary_file,
            loudness_target=loudness_target,
            loudness_tp=loudness_tp,
            loudness_lra=loudness_lra,
            cache_dir=cache_dir,
            mt_engine=mt_engine,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            hunyuan_model=hunyuan_model,
            hunyuan_host=hunyuan_host,
            hunyuan_refine=hunyuan_refine,
            hunyuan_samples=hunyuan_samples,
            fusion_model=fusion_model,
            sentence_units=sentence_units,
            restore_punct=restore_punct,
            qc_transcript=qc_transcript,
            qc_subtitles=qc_subtitles,
            qc_translation=qc_translation,
            qc_tts=qc_tts,
            qc_final=qc_final,
            tts_qc_mos_min=tts_qc_mos_min,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            allow_lang_mismatch=allow_lang_mismatch,
            pronounce_path=pronounce_path,
            tail_pad=tail_pad,
            ref_voice=ref_voice_path,
            voxcpm2_voice=voxcpm2_voice,
        )
    except ValueError as exc:
        state.update(status="error", message=f"error: {exc}", error=str(exc))
        return snapshot(state)

    # Ollama pre-flight: block a run whose engine/refine server or model is
    # missing, with the reason + fix commands, before starting the worker.
    preflight = _preflight_error(cfg)
    if preflight is not None:
        state.update(status="error", message=f"error: {preflight}", error=preflight)
        return snapshot(state)

    # Normalise the force selection: empty means "resume as today" (None).
    force = [str(name).strip() for name in (force_stages or []) if str(name).strip()]
    force_arg = force or None

    state.update(
        status="running",
        message="Starting…",
        stage="",
        error=None,
        report=None,
        qc_summary="",
        manifest_path=None,
        artifacts_dir=None,
        stages={},
        script_rows=[],
        segment_files={},
        segment_choices=[],
        _dd_choices=[],
        video=None,
        download=None,
        cancel=False,
        mode=cfg.mode,
        sub_position=cfg.sub_position,
        sub_font_size=cfg.sub_font_size,
        burn=bool(burn_subtitles),
        started_at=time.monotonic(),
        expected_stages=required_stages(
            cfg.mode, cfg.diarize, cfg.clone_speakers
        ),
    )
    thread = threading.Thread(
        target=_worker,
        args=(state, video, cfg, force_arg, bool(burn_subtitles)),
        daemon=True,
    )
    state["thread"] = thread
    thread.start()
    return snapshot(state)


def _worker(
    state: dict,
    video: str,
    cfg: JobConfig,
    force: list[str] | None = None,
    burn: bool = False,
) -> None:
    """Background thread: download (URL) + run_pipeline with the cancel flag."""
    try:
        if is_url(video):
            state["stage"] = "download"
            local = download(video, cfg.cache_dir)
            pipeline_input = video  # run_pipeline reuses the download
        else:
            local = Path(video)
            pipeline_input = video
        state["manifest_path"] = str(
            Path(cfg.cache_dir) / video_hash(str(local)) / "job.json"
        )
        state["stage"] = ""
        report = run_pipeline(
            pipeline_input,
            cfg,
            force=force,
            burn=burn,
            cancel_cb=lambda: bool(state.get("cancel")),
        )
    except PipelineCancelled:
        state.update(
            status="cancelled",
            message="Cancelled.",
            stage="",
            thread=None,
            started_at=None,
        )
        return
    except Exception as exc:  # noqa: BLE001 - surface as a friendly UI error
        state.update(
            status="error",
            message=f"error: {exc}",
            error=str(exc),
            stage="",
            thread=None,
            started_at=None,
        )
        return

    state["report"] = report
    try:
        _collect(state)
    except Exception:  # noqa: BLE001 - a collect failure must not strand the run
        log.exception("failed to collect artifacts after the run")
    state.update(
        status="done", message="Done.", stage="", thread=None, started_at=None
    )


# -------------------------------------------------------- artifact collection
def _aligned_spans(artifacts_dir: Path, manifest: JobManifest) -> dict[int, dict]:
    """Per script-row aligned span read once from ``timed.json``.

    Returns ``{script row index: {"file", "start", "end", "phrase", "align"}}``
    from each timed row's ``member_spans`` (SA-1-T3 output). ``{}`` when
    ``timed.json`` is missing/broken or no row carries ``member_spans`` (legacy
    ``--no-sentence-units`` runs and every pre-SA-1 cache), so every display
    surface falls back to today's behaviour.
    """
    path = artifacts_dir / manifest.paths.get("timed", "timed.json")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(rows, list):
        return {}
    spans: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        member_spans = row.get("member_spans")
        if not isinstance(member_spans, list) or not member_spans:
            continue
        wav = str(artifacts_dir / str(row.get("file", "")))
        align = str(row.get("align", ""))
        for entry in member_spans:
            if not isinstance(entry, (list, tuple)) or len(entry) < 4:
                continue
            try:
                index = int(entry[0])
                start = float(entry[1])
                end = float(entry[2])
            except (TypeError, ValueError):
                continue
            spans[index] = {
                "file": wav,
                "start": start,
                "end": end,
                "phrase": str(entry[3]),
                "align": align,
            }
    return spans


def _read_script_rows(
    artifacts_dir: Path, spans: dict[int, dict] | None = None
) -> list[list]:
    """``script.json`` rows for the table (``[]`` before translate has run).

    SA-1-T4 appends three columns after the original five (order preserved):
    ``unit`` (the unit audio file name, ``""`` without a span), ``aligned`` (the
    member's aligned range ``"12.4–15.9s"``, ``""`` without a span) and
    ``vi_aligned`` (the phrase actually spoken in that span, or the row's ``vi``
    when there is no span).
    """
    path = artifacts_dir / "script.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    known = spans or {}
    rows: list[list] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        vi = row.get("vi")
        span = known.get(index)
        if isinstance(span, dict):
            unit = Path(str(span.get("file", ""))).name
            aligned = f"{float(span['start']):.1f}–{float(span['end']):.1f}s"
            vi_aligned = str(span.get("phrase", ""))
        else:
            unit = ""
            aligned = ""
            vi_aligned = vi
        rows.append(
            [
                row.get("start"),
                row.get("end"),
                row.get("speaker"),
                row.get("en"),
                vi,
                unit,
                aligned,
                vi_aligned,
            ]
        )
    return rows


def _text_selection_path(artifacts_dir: Path) -> tuple[Path, str]:
    """Use sentence units when present; otherwise edit legacy script rows."""
    units_path = artifacts_dir / "units.json"
    if units_path.is_file():
        return units_path, "units"
    return artifacts_dir / "script.json", "script"


def _read_text_selection_rows(artifacts_dir: Path) -> list[list]:
    """Editable UI rows for spoken and subtitle text, at the synthesis granularity."""
    path, kind = _text_selection_path(artifacts_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    entries = data.get("units") if kind == "units" and isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []

    rows = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        rows.append(
            [
                entry.get("index", entry.get("idx", position)),
                entry.get("start"),
                entry.get("end"),
                entry.get("source", entry.get("en", "")),
                entry.get("vi", ""),
                selected_text(entry, "spoken_vi"),
                selected_text(entry, "subtitle_vi"),
            ]
        )
    return rows


def _write_text_selection_rows(artifacts_dir: Path, edited_rows) -> None:
    """Persist only the two text selections, refusing reordered or partial tables."""
    path, kind = _text_selection_path(artifacts_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc
    entries = data.get("units") if kind == "units" and isinstance(data, dict) else data
    if not isinstance(entries, list) or not all(isinstance(row, dict) for row in entries):
        raise ValueError(f"{path.name} has no editable text rows")

    if hasattr(edited_rows, "values"):
        edited_rows = edited_rows.values.tolist()
    elif hasattr(edited_rows, "tolist"):
        edited_rows = edited_rows.tolist()
    if not isinstance(edited_rows, list) or len(edited_rows) != len(entries):
        raise ValueError("text selection rows must keep the original row count and order")

    for position, (entry, edited) in enumerate(zip(entries, edited_rows)):
        if not isinstance(edited, (list, tuple)) or len(edited) < 7:
            raise ValueError("each text selection row must contain all seven columns")
        expected_id = entry.get("index", entry.get("idx", position))
        try:
            same_id = int(edited[0]) == int(expected_id)
        except (TypeError, ValueError):
            same_id = edited[0] == expected_id
        if not same_id:
            raise ValueError("text selection rows must stay in their original order")
        if not isinstance(edited[5], str) or not isinstance(edited[6], str):
            raise ValueError("spoken and subtitle text must be strings")
        entry["spoken_vi"] = edited[5]
        entry["subtitle_vi"] = edited[6]

    if kind == "units":
        data["units"] = entries
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_text_selection_rows(state: dict) -> list[list]:
    artifacts_dir = state.get("artifacts_dir")
    if not artifacts_dir:
        return []
    return _read_text_selection_rows(Path(artifacts_dir))


def _save_text_selection_rows(state: dict, edited_rows) -> str:
    thread = state.get("thread")
    if (thread is not None and thread.is_alive()) or state.get("status") == "running":
        return "Wait for the current run to finish before saving text choices."
    artifacts_dir = state.get("artifacts_dir")
    if not artifacts_dir:
        return "Run a video first, then load its text choices here."
    try:
        _write_text_selection_rows(Path(artifacts_dir), edited_rows)
    except (OSError, ValueError) as exc:
        return f"Could not save text choices: {exc}"
    return "Saved. Click **Run** to rebuild only stages affected by your edits."


def _timed_unit_files(artifacts_dir: Path, manifest: JobManifest) -> dict[int, str]:
    """``{segment index: unit WAV path}`` from the timed rows' ``members``.

    ``{}`` when the timing stage has not run, ``timed.json`` is missing/broken,
    or the run used the per-segment path (no row carries ``members``). Rows whose
    WAV is missing are skipped, so a stale ``timed.json`` never breaks the player.
    """
    path = artifacts_dir / manifest.paths.get("timed", "timed.json")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(rows, list):
        return {}
    files: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        members = row.get("members")
        if not isinstance(members, list) or not members:
            continue
        wav = artifacts_dir / str(row.get("file", ""))
        if not wav.exists():
            continue
        for member in members:
            try:
                files[int(member)] = str(wav)
            except (TypeError, ValueError):
                continue
    return files


def _segment_map(
    artifacts_dir: Path,
    manifest: JobManifest,
    spans: dict[int, dict] | None = None,
) -> tuple[dict, list]:
    """``{label: wav path}`` for the per-segment player.

    With sentence units on, each segment lists its **unit's** audio file (mapped
    from the timed rows' ``members``); with the opt-out path (no ``members``) it
    falls back to the per-segment ``speech.json`` files exactly as before. The
    label keeps its ``<idx> · <speaker> · <start>s`` form and, when SA-1-T4 gave
    the row an aligned span, appends ``· <start>–<end>s`` so the player shows the
    range it plays.
    """
    files: dict[str, str] = {}
    choices: list[str] = []
    speech_path = artifacts_dir / manifest.paths.get("speech", "speech.json")
    try:
        speech = json.loads(speech_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        speech = None
    speech_rows = speech if isinstance(speech, list) else []

    unit_files = _timed_unit_files(artifacts_dir, manifest)
    for record in speech_rows:
        if not isinstance(record, dict):
            continue
        try:
            idx = int(record.get("idx", 0))
        except (TypeError, ValueError):
            continue
        if unit_files:
            if idx not in unit_files:
                continue
            path = unit_files[idx]
        else:
            candidate = artifacts_dir / str(record.get("file", ""))
            if not candidate.exists():
                continue
            path = str(candidate)
        label = (
            f"{idx} · {record.get('speaker', 'spk0')} · "
            f"{float(record.get('start', 0.0)):.1f}s"
        )
        span = (spans or {}).get(idx)
        if isinstance(span, dict):
            try:
                label += f" · {float(span['start']):.1f}–{float(span['end']):.1f}s"
            except (TypeError, ValueError, KeyError):
                pass
        files[label] = path
        choices.append(label)

    if not choices:  # fall back to any per-segment WAV already on disk
        for path in sorted((artifacts_dir / "speech").glob("seg_*.wav")):
            files[path.stem] = str(path)
            choices.append(path.stem)
    return files, choices


def _read_json_dict(path: Path) -> dict | None:
    """Load a JSON object from ``path``; ``None`` when missing/broken/non-object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _qc_summary(artifacts_dir: Path, report: dict | None) -> str:
    """Compact Markdown summary of the QC artifacts for the results panel.

    Thin and tolerant: it only reads the QC sidecars (``transcript_qc.json``,
    ``translation_qc.json``, ``tts_qc.json``, ``subtitles_qc.json``) plus the
    report's ``quality`` block, never raises, and degrades to ``not run`` /
    ``refine: off`` when an artifact is absent or broken. Called from
    :func:`_collect` so the panel updates on the normal poll timer.
    """
    lines: list[str] = ["### Quality results"]

    # SU-1-T3: state the run mode up front; for subtitles-only, say plainly what
    # was skipped and that no TTS ran. Old artifacts (no report / no mode) keep
    # the previous block untouched.
    run_mode = report.get("mode") if isinstance(report, dict) else None
    if run_mode:
        lines.append(f"- **Mode:** `{run_mode}`")
        if run_mode == SUBTITLES_ONLY:
            lines.append(
                "- **Skipped:** `speak`, `timing`, `mix` — subtitles-only uses no "
                "TTS and keeps the original audio."
            )

    quality = report.get("quality") if isinstance(report, dict) else None
    if isinstance(quality, dict):
        lines.append(f"- **Verdict:** `{quality.get('verdict', 'unknown')}`")
        levels = quality.get("levels")
        if isinstance(levels, dict) and levels:
            rendered = ", ".join(f"{key}={value}" for key, value in levels.items())
            lines.append(f"- **Levels:** {rendered}")
    else:
        lines.append("- **Verdict:** not run")

    # --- transcript QC
    transcript = _read_json_dict(artifacts_dir / "transcript_qc.json")
    lines += ["", "**Transcript QC**"]
    if transcript is None:
        lines.append("- not run")
    else:
        lines.append(f"- verdict: `{transcript.get('verdict', 'unknown')}`")
        ratio = transcript.get("flagged_ratio_unresolved")
        if ratio is not None:
            lines.append(f"- flagged ratio unresolved: {ratio}")
        counts = transcript.get("counts")
        if isinstance(counts, dict):
            nonzero = ", ".join(f"{key}={value}" for key, value in counts.items() if value)
            lines.append(f"- counts (non-zero): {nonzero or 'none'}")

    # --- translation QC (incl. the M5-T2 refine summary)
    translation = _read_json_dict(artifacts_dir / "translation_qc.json")
    lines += ["", "**Translation QC**"]
    if translation is None:
        lines.append("- not run")
    else:
        lines.append(f"- verdict: `{translation.get('verdict', 'unknown')}`")
        counts = translation.get("counts")
        if isinstance(counts, dict):
            rendered = ", ".join(f"{key}={value}" for key, value in counts.items())
            lines.append(f"- counts: {rendered or 'none'}")
            numeric = [v for v in counts.values() if isinstance(v, (int, float))]
            lines.append(
                f"- unresolved flags (battery): {sum(numeric) - counts.get('ok', 0)}"
            )
        refine = translation.get("refine")
        if isinstance(refine, dict):
            attempted = refine.get("attempted", 0)
            improved = refine.get("improved", 0)
            fusion = "yes" if refine.get("fusion_used") else "no"
            detail = (
                f"- refine: attempted {attempted}, improved {improved}, "
                f"fusion used {fusion}"
            )
            rejected = attempted - improved
            if isinstance(rejected, int) and rejected > 0:
                detail += f", rejected {rejected}"
            lines.append(detail)
        else:
            lines.append("- refine: off")

    # --- TTS QC
    tts = _read_json_dict(artifacts_dir / "tts_qc.json")
    lines += ["", "**TTS QC**"]
    if tts is None:
        lines.append("- not run")
    else:
        lines.append(
            "- "
            + ", ".join(
                [
                    f"mos_mean: {tts.get('mos_mean')}",
                    f"mos_min: {tts.get('mos_min')}",
                    f"scored: {tts.get('scored')}",
                    f"low_mos: {tts.get('low_mos')}",
                    f"retried: {tts.get('retried')}",
                    f"resolved: {tts.get('resolved')}",
                ]
            )
        )

    # --- subtitles QC
    subtitles = _read_json_dict(artifacts_dir / "subtitles_qc.json")
    lines += ["", "**Subtitles QC**"]
    if subtitles is None:
        lines.append("- not run")
    else:
        lines.append(
            "- "
            + ", ".join(
                [
                    f"cues_in: {subtitles.get('cues_in')}",
                    f"cues_out: {subtitles.get('cues_out')}",
                    f"split: {subtitles.get('split')}",
                    f"cps_max_observed: {subtitles.get('cps_max_observed')}",
                ]
            )
        )
        violations = subtitles.get("violations")
        if isinstance(violations, dict):
            rendered = ", ".join(f"{key}={value}" for key, value in violations.items())
            lines.append(f"- violations: {rendered or 'none'}")

    # --- sentence units (SA-1-T4): unit-guard flags + display-alignment counts.
    # Both sources are optional, so this block is absent for old artifacts.
    guard_flags = None
    if isinstance(translation, dict):
        units_block = translation.get("units")
        if isinstance(units_block, dict) and isinstance(
            units_block.get("flags"), dict
        ):
            guard_flags = units_block["flags"]
    timing = _read_json_dict(artifacts_dir / "timing.json")
    align = timing.get("align") if isinstance(timing, dict) else None
    has_align = isinstance(align, dict) and bool(align)
    if guard_flags is not None or has_align:
        lines += ["", "**Sentence units**"]
        if guard_flags is not None:
            lines.append(
                "- guard: "
                + ", ".join(
                    f"{key}={guard_flags.get(key, 0)}"
                    for key in (
                        "guard_flagged",
                        "omission",
                        "sentence_loss",
                        "oversized",
                    )
                )
            )
        if has_align:
            lines.append(
                "- align: "
                + ", ".join(f"{key}={value}" for key, value in align.items())
            )

    return "\n".join(lines)


def _collect(state: dict) -> None:
    """Refresh progress/script/segments/video/report from the cache + output."""
    manifest_path = state.get("manifest_path")
    if not manifest_path:
        return
    artifacts_dir = Path(manifest_path).parent
    state["artifacts_dir"] = str(artifacts_dir)
    try:
        manifest = JobManifest.load(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return

    state["stages"] = dict(manifest.stages)
    # SA-1-T4: read the aligned member spans once and share them (script table
    # columns + the segment player's slice) — one timed.json read per collect.
    spans = _aligned_spans(artifacts_dir, manifest)
    state["segment_spans"] = spans
    state["script_rows"] = _read_script_rows(artifacts_dir, spans)
    state["segment_files"], state["segment_choices"] = _segment_map(
        artifacts_dir, manifest, spans
    )

    report = state.get("report")
    dubbed = None
    if isinstance(report, dict):
        relative = (report.get("outputs") or {}).get("dubbed")
        if relative:
            dubbed = Path(relative)
    if dubbed is None:
        dubbed = export_mod.output_dir_for(manifest) / export_mod.DUBBED_NAME
    if dubbed.exists():
        state["video"] = str(dubbed)
        state["download"] = str(dubbed)
    else:
        state["video"] = None
        state["download"] = None

    state["qc_summary"] = _qc_summary(artifacts_dir, report)


# --------------------------------------------------------------- rendering
def _current_stage(state: dict) -> str:
    stages = state.get("stages") or {}
    for name in state.get("expected_stages") or []:
        if stages.get(name) != "done":
            return name
    return ""


def _format_elapsed(seconds: float) -> str:
    """Human elapsed time: ``H:MM:SS`` once past an hour, else ``M:SS``.

    Documented contract (also used by the "already in progress" message):
    ``0:07``, ``3:12``, ``59:59``, then ``1:02:03``.
    """
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _elapsed_seconds(state: dict) -> float:
    """Seconds since the current run started (``0.0`` when not running)."""
    started = state.get("started_at")
    if started is None:
        return 0.0
    return time.monotonic() - started


def _elapsed_marker(state: dict) -> str:
    """`` (elapsed M:SS)`` suffix shown while a run is in progress."""
    return f" (elapsed {_format_elapsed(_elapsed_seconds(state))})"


def _status_text(state: dict) -> str:
    status = state.get("status", "idle")
    if status == "running":
        marker = _elapsed_marker(state)
        if state.get("stage"):
            return f"**{state['stage'].capitalize()}…**{marker}"
        current = _current_stage(state)
        base = f"**Running** — stage `{current}`" if current else "**Running…**"
        return f"{base}{marker}"
    if status == "error":
        return f"**Error** — {state.get('error') or state.get('message')}"
    if status == "cancelled":
        return f"**Cancelled** — {state.get('message')}"
    if status == "done":
        return f"**Done** — {state.get('video') or 'output written'}"
    return "_Ready. Upload a video or paste a URL, then click **Run**._"


def _stage_rows(state: dict) -> list[list]:
    expected = state.get("expected_stages") or []
    stages = state.get("stages") or {}
    report = state.get("report")
    report_stages = (report or {}).get("stages") or {}
    current = _current_stage(state)
    rows: list[list] = []
    for name in expected:
        seconds: object = ""
        if name in report_stages:
            entry = report_stages[name] or {}
            status = str(entry.get("status", ""))
            seconds = entry.get("seconds", "")
        elif name in stages:
            status = str(stages[name])
        else:
            status = "running" if name == current else "pending"
        rows.append([name, status, seconds])
    return rows


def _dropdown_update(state: dict):
    choices = list(state.get("segment_choices") or [])
    if choices != state.get("_dd_choices"):
        state["_dd_choices"] = choices
        return gr.update(choices=choices, value=(choices[0] if choices else None))
    return gr.update()


def snapshot(state: dict) -> tuple:
    """Component values for one poll: status, stages, script, segments, media, QC."""
    return (
        _status_text(state),
        _stage_rows(state),
        state.get("script_rows") or [],
        _dropdown_update(state),
        state.get("video"),
        state.get("download"),
        state.get("report"),
        state.get("qc_summary") or "",
    )


def poll(state: dict) -> tuple:
    """Timer/Refresh handler: collect live artifacts then return a snapshot.

    This handler must **never** propagate: an unexpected raise inside
    :func:`_collect` (or a nested ``_qc_summary``) would otherwise kill the
    EventSource/timer stream and freeze the whole page until a manual reload.
    On failure it logs server-side and returns a best-effort snapshot built from
    the last known state.
    """
    try:
        if state.get("status") == "running" and state.get("manifest_path"):
            _collect(state)
        return snapshot(state)
    except Exception:  # noqa: BLE001 - the poll stream must survive anything
        log.exception("poll failed; returning a degraded snapshot")
        return _degraded_snapshot(state)


def _degraded_snapshot(state: dict) -> tuple:
    """A snapshot that never raises, used when normal rendering fails.

    Same arity as :func:`snapshot`, with component values degraded to the last
    known state (constants for the pieces that cannot be safely rendered).
    """
    try:
        return snapshot(state)
    except Exception:  # noqa: BLE001 - the last-resort snapshot must not raise
        log.exception("snapshot failed; degrading to the last known values")
        return (
            "_Status unavailable._",
            [],
            state.get("script_rows") or [],
            gr.update(),
            state.get("video"),
            state.get("download"),
            state.get("report"),
            state.get("qc_summary") or "",
        )


def cancel(state: dict) -> tuple:
    """Cancel button: raise the cooperative flag; the worker reports cancelled."""
    if state.get("status") == "running":
        state["cancel"] = True
        state["message"] = "Cancelling…"
    return snapshot(state)


# SA-1-T4: the last slice written to the OS temp dir, so repeated clicks reuse
# it and a new slice never leaks the previous file.
_SPAN_SLICE_KEY = "_span_slice"


def _label_index(label: object) -> int | None:
    """The leading script-row index of a segment dropdown label (``None`` else)."""
    head = str(label or "").split(" · ", 1)[0].strip()
    try:
        return int(head)
    except ValueError:
        return None


def _slice_wav(source: Path, start: float, end: float, target: Path) -> int:
    """Write the PCM frames of ``source`` in ``[start, end)`` to ``target``.

    Pure stdlib ``wave``; the frame window is clamped against the file's real
    frame count, so an out-of-range span yields a shorter (or empty) but valid
    WAV rather than reading past the end. Returns the number of frames written.
    """
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        rate = params.framerate or 1
        total = params.nframes
        first = min(max(int(round(float(start) * rate)), 0), total)
        last = min(max(int(round(float(end) * rate)), first), total)
        if last <= first:
            return 0
        reader.setpos(first)
        frames = reader.readframes(last - first)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)
    return last - first


def _slice_span(state: dict, source: str, span: dict) -> str | None:
    """Slice ``source`` to ``span``'s ``[start, end)`` into a cached temp WAV.

    The slice is written under the OS temp directory (never the job artifacts).
    The last slice is cached in ``state`` and the previous one is removed when a
    different slice replaces it. Returns ``None`` when the span is empty, cannot
    be sliced, or lies entirely outside the file (the caller then returns the
    full file).
    """
    start = max(0.0, float(span.get("start", 0.0)))
    end = float(span.get("end", 0.0))
    if end <= start:
        return None
    key = (str(source), round(start, 3), round(end, 3))
    cached = state.get(_SPAN_SLICE_KEY)
    if isinstance(cached, dict) and cached.get("key") == key:
        path = cached.get("path")
        if isinstance(path, str) and Path(path).exists():
            return path
    target = Path(tempfile.gettempdir()) / (
        f"vidub_span_{Path(source).stem}_{int(round(start * 1000))}"
        f"_{int(round(end * 1000))}.wav"
    )
    if _slice_wav(Path(source), start, end, target) <= 0:
        return None
    if isinstance(cached, dict):
        old = cached.get("path")
        if isinstance(old, str) and old != str(target):
            try:
                Path(old).unlink(missing_ok=True)
            except OSError:  # noqa: BLE001 - a stale slice must not break the click
                pass
    state[_SPAN_SLICE_KEY] = {"key": key, "path": str(target)}
    return str(target)


def segment_audio(state: dict, selection):
    """Return a WAV path for the selected segment label (or ``None``).

    SA-1-T4: when the selected script row has an aligned ``member_spans`` span,
    this returns a temporary WAV holding **only that span** of the unit audio
    (sliced with stdlib ``wave`` into the OS temp dir); repeated clicks reuse the
    cached slice. When the row has no span (legacy ``--no-sentence-units`` runs
    and pre-SA-1 caches) it returns the unit's full file exactly as before, and
    any slice failure falls back to that full file — this handler never raises.
    """
    if not selection:
        return None
    full = (state.get("segment_files") or {}).get(selection)
    if not full:
        return None
    index = _label_index(selection)
    span = (state.get("segment_spans") or {}).get(index) if index is not None else None
    if not isinstance(span, dict):
        return full
    try:
        return _slice_span(state, full, span) or full
    except Exception:  # noqa: BLE001 - the player must never raise
        log.exception("segment slice failed; returning the full unit file")
        return full


# ------------------------------------------------------------- error guard
def _noop_updates(count: int) -> tuple:
    """``count`` no-op ``gr.update()`` values (keeps previous component values)."""
    return tuple(gr.update() for _ in range(count))


def _safe(fn, fallback=None):
    """Wrap a Gradio handler so an unexpected exception can never kill the stream.

    A raise from a handler propagates into the Gradio event stream and, once it
    reaches the browser, tears down the page's update/timer channel — the
    reported "controls dead until F5" freeze. This guard logs the error
    server-side with its traceback and returns ``fallback()`` instead (the
    current :func:`snapshot` for snapshot-shaped handlers, a no-op update tuple
    for the form presets, ``None`` for the segment player).

    Deliberate :class:`ValueError` validation is handled *inside* the handlers
    (see :func:`_handle_run` / :func:`form_to_config`) and never reaches this
    guard, so bad input still surfaces to the UI as ``status="error"``.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:  # noqa: BLE001 - handlers must never raise into Gradio
            name = getattr(fn, "__name__", repr(fn))
            log.exception("UI handler %s raised; returning a degraded result", name)
            if fallback is None:
                return None
            try:
                return fallback()
            except Exception:  # noqa: BLE001 - the fallback must also never raise
                log.exception("UI handler %s fallback failed", name)
                return None

    return wrapper


# ----------------------------------------------------------------- app build
def build_app(cfg_defaults: JobConfig | None = None):
    """Build (but do not launch) the Gradio app; return ``(app, state)``."""
    defaults = cfg_defaults or JobConfig()
    state = new_state()

    with gr.Blocks(title="ViDub") as app:
        gr.Markdown(
            "# ViDub — local dubbing UI\n"
            "Upload a video or paste a URL, configure the run, then **Run**. "
            "Long runs happen in the background; progress is polled every second."
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_file = gr.File(
                    label="Upload video",
                    file_types=["video"],
                    type="filepath",
                )
                input_url = gr.Textbox(
                    label="… or paste a URL", placeholder="https://…"
                )
                mode = gr.Dropdown(
                    choices=MODE_CHOICES_LABELED,
                    value=defaults.mode,
                    label="Mode (Subtitles only = vietsub, no TTS)",
                )
                sub_position = gr.Dropdown(
                    choices=list(SUB_POSITIONS),
                    value=defaults.sub_position,
                    label="Subtitle position (subtitles-only / --burn)",
                )
                sub_font_size = gr.Number(
                    value=defaults.sub_font_size,
                    minimum=1,
                    step=1,
                    label="Subtitle font size (subtitles-only / --burn)",
                )
                burn_subtitles = gr.Checkbox(
                    value=False,
                    label=(
                        "Burn subtitles into the video "
                        "(re-encode; subtitles-only / --burn)"
                    ),
                )
                asr_model = gr.Dropdown(
                    choices=ASR_MODEL_CHOICES,
                    value=defaults.asr_model,
                    allow_custom_value=True,
                    label="ASR model",
                )
                tts_engine = gr.Dropdown(
                    choices=TTS_ENGINE_CHOICES,
                    value=defaults.tts_engine,
                    label="TTS engine",
                )
                voice_reference = gr.Audio(
                    value=defaults.ref_voice,
                    type="filepath",
                    format="wav",
                    sources=["upload", "microphone"],
                    label="Voice sample for VoxCPM2 (optional; overrides profile)",
                    visible=defaults.tts_engine == "voxcpm2",
                )
                voxcpm2_voice = gr.Dropdown(
                    choices=[
                        (profile["label"], name)
                        for name, profile in VOXCPM2_VOICE_PROFILES.items()
                    ],
                    value=defaults.voxcpm2_voice,
                    label="VoxCPM2 voice profile",
                    visible=defaults.tts_engine == "voxcpm2",
                )
                allow_nc = gr.Checkbox(
                    value=defaults.allow_nc_models,
                    label="Allow non-commercial models (vixtts)",
                )
                diarize = gr.Checkbox(
                    value=defaults.diarize, label="Diarize speakers"
                )
                clone_speakers = gr.Checkbox(
                    value=defaults.clone_speakers,
                    label="Clone speakers (needs diarize + vixtts)",
                )
                glossary = gr.File(
                    label="Glossary TSV (optional)",
                    file_types=[".tsv"],
                    type="filepath",
                )
                loud_target = gr.Number(
                    value=defaults.loudness_target_i, label="Loudness target (LUFS)"
                )
                loud_tp = gr.Number(
                    value=defaults.loudness_tp, label="True peak (dBTP)"
                )
                loud_lra = gr.Number(
                    value=defaults.loudness_lra, label="Loudness range (LU)"
                )
                cache_dir = gr.Textbox(
                    value=defaults.cache_dir, label="Cache dir"
                )
                with gr.Accordion("Engines & quality (M4/M5)", open=False):
                    gr.Markdown(
                        "`envit5` is the default MT engine; `ollama`/`hunyuan` "
                        "need a local Ollama server. Changing the MT engine or "
                        "the refine settings re-runs `translate` and everything "
                        "downstream automatically. Sentence-level units merge "
                        "ASR fragments into whole sentences before translation "
                        "and split the result back onto the original segments; "
                        "untick to translate ASR segments as-is."
                    )
                    mt_engine = gr.Dropdown(
                        choices=list(MT_ENGINES),
                        value=defaults.mt_engine,
                        label="MT engine",
                    )
                    ollama_model = gr.Textbox(
                        value=defaults.ollama_model, label="Ollama model"
                    )
                    ollama_host = gr.Textbox(
                        value=defaults.ollama_host, label="Ollama host"
                    )
                    hunyuan_model = gr.Textbox(
                        value=defaults.hunyuan_model, label="Hunyuan model"
                    )
                    hunyuan_host = gr.Textbox(
                        value=defaults.hunyuan_host, label="Hunyuan host"
                    )
                    hunyuan_refine = gr.Checkbox(
                        value=defaults.hunyuan_refine,
                        label="Hunyuan refine pass",
                    )
                    hunyuan_samples = gr.Number(
                        value=defaults.hunyuan_samples,
                        minimum=1,
                        maximum=10,
                        step=1,
                        label="Hunyuan refine samples",
                    )
                    fusion_model = gr.Textbox(
                        value=defaults.fusion_model, label="Fusion model"
                    )
                    sentence_units = gr.Checkbox(
                        value=defaults.sentence_units,
                        label="Sentence-level translation units",
                    )
                    restore_punct = gr.Checkbox(
                        value=defaults.restore_punct,
                        label="Restore punctuation (source)",
                    )
                    qc_transcript = gr.Checkbox(
                        value=defaults.qc_transcript, label="QC: transcript"
                    )
                    qc_subtitles = gr.Checkbox(
                        value=defaults.qc_subtitles, label="QC: subtitles"
                    )
                    qc_translation = gr.Checkbox(
                        value=defaults.qc_translation, label="QC: translation"
                    )
                    qc_tts = gr.Checkbox(
                        value=defaults.qc_tts, label="QC: TTS"
                    )
                    qc_final = gr.Checkbox(
                        value=defaults.qc_final, label="QC: final gate"
                    )
                    tts_qc_mos_min = gr.Number(
                        value=defaults.tts_qc_mos_min,
                        minimum=0,
                        maximum=5,
                        step=0.1,
                        label="TTS QC min MOS",
                    )
                force_stages = gr.CheckboxGroup(
                    choices=[*STAGES, "diarize", "references"],
                    value=[],
                    label="Re-run these stages (force)",
                )
                gr.Markdown("_Forcing a stage also re-runs every stage after it._")
                src_lang = gr.Textbox(
                    value=defaults.src_lang, label="Source language"
                )
                tgt_lang = gr.Textbox(
                    value=defaults.tgt_lang, label="Target language"
                )
                allow_lang_mismatch = gr.Checkbox(
                    value=defaults.allow_lang_mismatch,
                    label="Allow language mismatch (override the ASR guard)",
                )
                pronounce = gr.File(
                    label="Pronunciation TSV (changes TTS speech only)",
                    file_types=[".tsv"],
                    type="filepath",
                )
                tail_pad = gr.Number(
                    value=defaults.tail_pad, label="Tail pad (s)"
                )
                gr.Markdown(
                    "**Test presets** — one click to configure the M5 features "
                    "(form only, no run): `Hunyuan engine` selects the opt-in "
                    "engine, `refine pass` turns on re-sampling, `Chimera fusion` "
                    "is **evaluation only**, `defaults` restores envit5. A changed "
                    "MT engine or refine setting re-runs `translate` and everything "
                    "downstream automatically; watch the **Quality results** panel."
                )
                with gr.Row():
                    preset_hunyuan_btn = gr.Button("Preset: Hunyuan engine")
                    preset_refine_btn = gr.Button("Preset: refine pass")
                    preset_fusion_btn = gr.Button(
                        "Preset: Chimera fusion (evaluate-only)"
                    )
                    preset_defaults_btn = gr.Button("Preset: defaults (envit5)")
                    preset_force_btn = gr.Button("Force full re-run")
                preset_hint = gr.Markdown(value="")
                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    cancel_btn = gr.Button("Cancel")
                    refresh_btn = gr.Button("Refresh")
            with gr.Column(scale=2):
                status_md = gr.Markdown(value=_status_text(state))
                stages_df = gr.Dataframe(
                    headers=["stage", "status", "seconds"],
                    label="Stages",
                    interactive=False,
                )
                with gr.Accordion("Review/edit spoken and subtitle text", open=False):
                    gr.Markdown(
                        "Load the translated text, edit only **spoken_vi** and "
                        "**subtitle_vi**, then save. Sentence units are edited as "
                        "one row. Click **Run** afterward to rebuild affected stages."
                    )
                    text_selection_df = gr.Dataframe(
                        headers=[
                            "index",
                            "start",
                            "end",
                            "source (EN)",
                            "accepted (VI)",
                            "spoken_vi",
                            "subtitle_vi",
                        ],
                        value=[],
                        type="array",
                        interactive=True,
                        static_columns=[0, 1, 2, 3, 4],
                        label="Text choices",
                        max_height=360,
                    )
                    with gr.Row():
                        load_text_btn = gr.Button("Load text choices")
                        save_text_btn = gr.Button("Save text choices")
                    text_selection_status = gr.Markdown(value="")
                script_df = gr.Dataframe(
                    headers=[
                        "start",
                        "end",
                        "speaker",
                        "en",
                        "vi",
                        "unit",
                        "aligned",
                        "vi_aligned",
                    ],
                    label="Script",
                    interactive=False,
                )
                seg_dd = gr.Dropdown(choices=[], label="Segment audio")
                seg_audio = gr.Audio(
                    type="filepath", label="Segment player", interactive=False
                )
                video_out = gr.Video(label="Dubbed video")
                download = gr.File(label="Download dubbed video")
                report_json = gr.JSON(label="report.json")
                qc_md = gr.Markdown(value="", label="Quality results")

        outputs = [
            status_md,
            stages_df,
            script_df,
            seg_dd,
            video_out,
            download,
            report_json,
            qc_md,
        ]
        inputs = [
            video_file,
            input_url,
            mode,
            sub_position,
            sub_font_size,
            burn_subtitles,
            asr_model,
            tts_engine,
            allow_nc,
            diarize,
            clone_speakers,
            glossary,
            loud_target,
            loud_tp,
            loud_lra,
            cache_dir,
            mt_engine,
            ollama_model,
            ollama_host,
            hunyuan_model,
            hunyuan_host,
            hunyuan_refine,
            hunyuan_samples,
            fusion_model,
            sentence_units,
            restore_punct,
            qc_transcript,
            qc_subtitles,
            qc_translation,
            qc_tts,
            qc_final,
            tts_qc_mos_min,
            force_stages,
            src_lang,
            tgt_lang,
            allow_lang_mismatch,
            pronounce,
            tail_pad,
            voice_reference,
            voxcpm2_voice,
        ]

        run_btn.click(
            fn=_safe(
                lambda *values: _handle_run(state, *values),
                fallback=lambda: _degraded_snapshot(state),
            ),
            inputs=inputs,
            outputs=outputs,
        )
        tts_engine.change(
            fn=lambda engine: (
                gr.update(visible=engine == "voxcpm2"),
                gr.update(visible=engine == "voxcpm2"),
            ),
            inputs=[tts_engine],
            outputs=[voice_reference, voxcpm2_voice],
        )
        preset_hunyuan_btn.click(
            fn=_safe(preset_hunyuan, fallback=lambda: _noop_updates(6)),
            inputs=None,
            outputs=[
                mt_engine,
                hunyuan_model,
                hunyuan_host,
                hunyuan_refine,
                fusion_model,
                preset_hint,
            ],
        )
        preset_refine_btn.click(
            fn=_safe(preset_refine, fallback=lambda: _noop_updates(4)),
            inputs=None,
            outputs=[hunyuan_refine, hunyuan_samples, fusion_model, preset_hint],
        )
        preset_fusion_btn.click(
            fn=_safe(preset_fusion, fallback=lambda: _noop_updates(3)),
            inputs=None,
            outputs=[hunyuan_refine, fusion_model, preset_hint],
        )
        preset_defaults_btn.click(
            fn=_safe(preset_defaults, fallback=lambda: _noop_updates(4)),
            inputs=None,
            outputs=[mt_engine, hunyuan_refine, fusion_model, preset_hint],
        )
        preset_force_btn.click(
            fn=_safe(preset_force_all, fallback=lambda: _noop_updates(1)),
            inputs=None,
            outputs=[force_stages],
        )
        cancel_btn.click(
            fn=_safe(lambda: cancel(state), fallback=lambda: _degraded_snapshot(state)),
            inputs=None,
            outputs=outputs,
        )
        refresh_btn.click(
            fn=_safe(lambda: poll(state), fallback=lambda: _degraded_snapshot(state)),
            inputs=None,
            outputs=outputs,
        )
        load_text_btn.click(
            fn=_safe(
                lambda: _load_text_selection_rows(state), fallback=lambda: []
            ),
            inputs=None,
            outputs=[text_selection_df],
        )
        save_text_btn.click(
            fn=_safe(
                lambda rows: _save_text_selection_rows(state, rows),
                fallback=lambda: "Could not save text choices.",
            ),
            inputs=[text_selection_df],
            outputs=[text_selection_status],
        )
        seg_dd.change(
            fn=_safe(
                lambda selection: segment_audio(state, selection), fallback=lambda: None
            ),
            inputs=[seg_dd],
            outputs=[seg_audio],
        )
        timer = gr.Timer(POLL_SECONDS)
        timer.tick(
            fn=_safe(lambda: poll(state), fallback=lambda: _degraded_snapshot(state)),
            inputs=None,
            outputs=outputs,
        )

    return app, state


def _silence_connection_reset(original):
    """Wrap ``_ProactorBasePipeTransport._call_connection_lost`` for reset noise.

    On Windows the browser aborts its video/EventSource connections as soon as
    it has enough data; the proactor loop then reports every aborted socket as
    ``ConnectionResetError: [WinError 10054] An existing connection was forcibly
    closed by the remote host`` while tearing the transport down. Those resets
    are harmless and never affect a run, so this wrapper swallows exactly that
    class and re-raises **every other** exception so the loop still prints it
    with its full traceback.
    """

    def _call_connection_lost(self, exc):
        try:
            original(self, exc)
        except ConnectionResetError:
            pass

    _call_connection_lost._vidub_quiet = True
    return _call_connection_lost


def _install_socket_noise_filter() -> bool:
    """Suppress WinError 10054 proactor noise for this process; install once.

    Scoped to the UI process and only called from :func:`launch_ui` (never at
    import time). Returns ``True`` when the patch is (now) in place and
    ``False`` off Windows. Idempotent: re-installing is a no-op.
    """
    if sys.platform != "win32":
        return False
    from asyncio.proactor_events import _ProactorBasePipeTransport

    current = _ProactorBasePipeTransport._call_connection_lost
    if getattr(current, "_vidub_quiet", False):
        return True
    _ProactorBasePipeTransport._call_connection_lost = _silence_connection_reset(
        current
    )
    log.info(
        "Suppressing harmless connection-reset (WinError 10054) socket noise "
        "from the Gradio server."
    )
    return True


def launch_ui(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    cfg_defaults: JobConfig | None = None,
) -> None:
    """Build and launch the local UI (blocking; localhost only, no ``--share``).

    Before the server starts, installs the Windows proactor socket-reset filter
    (:func:`_install_socket_noise_filter`): the browser aborts its video /
    EventSource connections once it has what it needs, and CPython's proactor
    loop logs every abort as ``ConnectionResetError: [WinError 10054]`` —
    harmless noise that otherwise floods the console. Only that exception class
    is hidden; any other error still surfaces with its traceback.
    """
    _install_socket_noise_filter()
    app, _state = build_app(cfg_defaults)
    app.launch(
        server_name=host,
        server_port=int(port),
        share=False,
        show_error=True,
    )

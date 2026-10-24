"""Full ViDub pipeline orchestration with per-stage resume.

Runs S0 ingest → S1 transcribe → (S1.5 diarize, optional) → (S1.6 references,
optional with ``--clone-speakers``) → S2 translate → S3
speak → S4 timing → S5 mix → S6 export, honouring the job mode
(``subtitles-only`` skips speak/timing/mix). ``--diarize`` inserts the optional
``diarize`` stage between transcribe and translate and invalidates translate →
export so the new speaker labels propagate; ``--clone-speakers`` additionally
inserts ``references`` right after diarize.
With ``resume`` (the default) a stage whose manifest status is ``done`` and whose
artifacts are still present is skipped; ``force`` re-runs the named stages *and
every stage after them* in the pipeline order, because a stage's output feeds all
downstream stages (so forcing ``transcribe`` invalidates translate → export).
Mode changes invalidate ``mix``/``export`` because their output depends on the
mode; a TTS-engine or voice-reference change invalidates ``speak`` → ``export``
because the speech audio is engine/profile-specific. A change to the
MT configuration (the engine, the *active* engine's model/host, or the
refine/fusion settings — see :func:`mt_signature`) invalidates ``translate`` →
``export`` so a cached script is re-translated instead of silently reused; an
unchanged MT configuration behaves exactly as before.
"""

from __future__ import annotations

import json
import hashlib
import logging
import time
from pathlib import Path

from vidub.config import (
    JobConfig,
    SUBTITLES_ONLY,
    check_mt_engine,
    check_tts_engine_allowed,
    normalize_mode,
    validate_clone_speakers,
    validate_langs,
)
from vidub.manifest import JobManifest
from vidub.stages import diarize as diarize_mod
from vidub.stages import export as export_mod
from vidub.stages import ingest as ingest_mod
from vidub.stages import mix as mix_mod
from vidub.stages import references as references_mod
from vidub.stages import speak as speak_mod
from vidub.stages import timing as timing_mod
from vidub.stages import transcribe as transcribe_mod
from vidub.stages import translate as translate_mod
from vidub.utils.text_selection import selection_signature
from vidub.utils.ytdlp import download, is_url

__all__ = [
    "run_pipeline",
    "required_stages",
    "stage_artifacts_present",
    "mt_signature",
    "tts_signature",
    "timing_signature",
    "first_forced_index",
    "PipelineCancelled",
    "STAGES",
]

log = logging.getLogger(__name__)

STAGES = export_mod.PIPELINE_STAGES
MODE_SKIPPED = export_mod.MODE_SKIPPED_STAGES
MANIFEST_NAME = "job.json"

# Stages whose output depends on the job mode; a mode change invalidates them.
MODE_DEPENDENT_STAGES = frozenset({"mix", "export"})


class PipelineCancelled(RuntimeError):
    """Raised by :func:`run_pipeline` when ``cancel_cb`` asks to stop.

    The callback is checked at each **stage boundary** (a stage already running
    always finishes), so cancellation is cooperative and leaves the manifest /
    resume state consistent for a later re-run.
    """


def required_stages(
    mode: str, diarize: bool = False, clone_speakers: bool = False
) -> list[str]:
    """Stages that must run for ``mode`` (mode-disabled + optional stages removed).

    The optional ``diarize`` stage is inserted between ``transcribe`` and
    ``translate`` only when ``diarize`` is true; ``references`` (voice cloning)
    is inserted right after ``diarize`` when ``clone_speakers`` is true (which
    requires diarization).
    """
    mode = normalize_mode(mode)
    skipped = MODE_SKIPPED.get(mode, frozenset())
    stages = [stage for stage in STAGES if stage not in skipped]
    if diarize:
        stages.insert(stages.index("transcribe") + 1, "diarize")
    if clone_speakers:
        anchor = "diarize" if "diarize" in stages else "transcribe"
        stages.insert(stages.index(anchor) + 1, "references")
    return stages


def first_forced_index(stages: list[str], force: set[str]) -> int | None:
    """Index of the earliest forced stage in ``stages`` (``None`` if none).

    Forcing a stage invalidates it and everything after it, so the pipeline
    re-runs ``stages[index:]`` for this index.
    """
    indices = [i for i, name in enumerate(stages) if name in force]
    return min(indices) if indices else None


def stage_artifacts_present(stage: str, manifest: JobManifest) -> bool:
    """True when the artifacts a finished ``stage`` should have produced exist."""
    artifacts_dir = manifest.artifacts_dir

    def present(key: str, default: str) -> bool:
        return (artifacts_dir / manifest.paths.get(key, default)).exists()

    if stage == "ingest":
        return present("audio", "audio.wav")
    if stage == "transcribe":
        return present("segments", "segments.json") and present("srt", "draft.srt")
    if stage == "diarize":
        # Diarization rewrites segments.json in place.
        return present("segments", "segments.json")
    if stage == "references":
        # The sidecar is written once every speaker was evaluated.
        return present("references_json", references_mod.REFERENCES_NAME)
    if stage == "translate":
        return present("script", "script.json")
    if stage == "speak":
        return present("speech", "speech.json")
    if stage == "timing":
        return present("timed", "timed.json")
    if stage == "mix":
        return present("dub_track", "dub_track.wav")
    if stage == "export":
        output_dir = export_mod.output_dir_for(manifest)
        return (output_dir / export_mod.DUBBED_NAME).exists() and (
            output_dir / export_mod.SUBTITLES_NAME
        ).exists()
    return False


def mt_signature(cfg: JobConfig) -> str:
    """Deterministic signature of the MT settings that shape ``script.json``.

    Built from the config fields whose change can alter the translated text: the
    lower-cased ``mt_engine`` and — only for the *active* engine — its own model
    and host (``hunyuan_model``/``hunyuan_host`` when the engine is ``hunyuan``,
    ``ollama_model``/``ollama_host`` when it is ``ollama``, nothing extra for
    ``envit5``), plus the engine-independent refinement settings
    ``hunyuan_refine``, ``hunyuan_samples`` and ``fusion_model``, plus the
    translation-mode switch ``sentence_units`` (SA-1: it changes what the engine
    receives and what ``script.json`` holds, so toggling ``--no-sentence-units``
    must invalidate a cached script). The switch value encodes the *pipeline
    version* (``units=2`` with sentence units on, ``units=0`` off), so a cache
    written by a pre-SA-1 sentence-unit pipeline (``units=1``) is always
    re-translated on this path. The ISO-1 isometric budgeting settings
    (``isometric`` and its ``isometric_headroom`` / ``isometric_cps`` knobs) also
    participate, so turning budgeting on/off or retuning it re-translates.
    The glossary content hash and translation-QC policy version participate too,
    including when a glossary is edited without changing its path.

    Settings of engines that are not in use never participate: changing
    ``ollama_model`` while ``mt_engine == "envit5"`` leaves the signature equal.
    ``restore_punct`` and the ``qc_*`` toggles are deliberately out of scope
    (their effect is corrected/measured downstream, not cached — re-run with
    ``--force`` when they change).

    Values are normalised (whitespace stripped, booleans/ints serialised
    stably), so two equivalent configs always produce the same single-line
    string. The pipeline compares it against ``manifest.mt_signature`` to decide
    whether a cached script must be re-translated.
    """
    engine = check_mt_engine(cfg.mt_engine)
    if engine == "hunyuan":
        model = (cfg.hunyuan_model or "").strip()
        host = (cfg.hunyuan_host or "").strip()
    elif engine == "ollama":
        model = (cfg.ollama_model or "").strip()
        host = (cfg.ollama_host or "").strip()
    else:  # envit5, and any other engine without per-engine settings
        model = host = ""
    glossary_digest = (
        hashlib.sha256(Path(cfg.glossary_path).read_bytes()).hexdigest()
        if cfg.glossary_path else "none"
    )
    return "|".join(
        (
            engine,
            model,
            host,
            f"validator={2 if engine in {'hunyuan', 'ollama'} else 0}",
            "tqc=grammar-safety-v2",
            f"glossary={glossary_digest}",
            f"refine={int(bool(cfg.hunyuan_refine))}",
            f"samples={int(cfg.hunyuan_samples)}",
            f"fusion={(cfg.fusion_model or '').strip()}",
            f"units={2 if cfg.sentence_units else 0}",
            # ISO-1 isometric length budgeting: it changes what the engine is
            # asked for (budgets) and what script.json holds, so toggling it or
            # its headroom/pace knobs must invalidate a cached translation.
            f"iso={int(bool(cfg.isometric))}",
            f"iso_head={float(cfg.isometric_headroom):g}",
            f"iso_cps={float(cfg.isometric_cps):g}",
            f"iso_cov={float(cfg.isometric_short_coverage):g}",
            f"iso_win={float(cfg.isometric_short_window):g}",
        )
    )


def tts_signature(cfg: JobConfig) -> str:
    """Sign voice inputs that can change synthesized audio.

    VoxCPM2's selected voice-design profile is included so changing it rebuilds
    speak onward. A supplied reference is hashed by content and overrides that
    profile. ViXTTS references use the same content signature; unrelated engines
    and fields stay out of scope.
    """
    engine = (cfg.tts_engine or "piper").strip().lower()
    if engine == "voxcpm2":
        voice = (
            hashlib.sha256(Path(cfg.ref_voice).read_bytes()).hexdigest()
            if cfg.ref_voice
            else "default"
        )
        profile = "reference" if cfg.ref_voice else cfg.voxcpm2_voice
        return f"{engine}|voice={voice}|profile={profile}"
    if engine != "vixtts" or not cfg.ref_voice:
        return ""
    voice = hashlib.sha256(Path(cfg.ref_voice).read_bytes()).hexdigest()
    return f"{engine}|voice={voice}"


def timing_signature(cfg: JobConfig) -> str:
    """Deterministic signature of the timing settings that shape ``timed.json``.

    Covers only the timing-stage shaping switches whose change alters the placed
    audio or the recorded spans: ``tts_rate_fit`` (per-synthesis rate),
    ``recenter_audio`` (placement shift), ``align_display`` (display alignment)
    and ``sentence_units`` (unit vs per-segment rows). Values are serialised
    stably, so two equivalent configs always produce the same single-line string.

    The pipeline records it on ``manifest.timing_signature`` after a real timing
    run and re-runs timing (and everything downstream) when a non-empty recorded
    value differs, so flipping any of these flags re-runs the stage.
    """
    return "|".join(
        (
            "policy=natural-pace-v1",
            f"rate_fit={int(bool(cfg.tts_rate_fit))}",
            f"recenter={int(bool(cfg.recenter_audio))}",
            f"align={int(bool(cfg.align_display))}",
            f"units={2 if cfg.sentence_units else 0}",
        )
    )


def _sync_config(manifest: JobManifest, cfg: JobConfig) -> None:
    """Record the requested config on the manifest so reports reflect it."""
    manifest.mode = normalize_mode(cfg.mode)
    manifest.src_lang = cfg.src_lang
    manifest.tgt_lang = cfg.tgt_lang
    manifest.asr_model = cfg.asr_model
    manifest.mt_engine = cfg.mt_engine
    manifest.tts_engine = cfg.tts_engine
    manifest.cache_dir = cfg.cache_dir


def text_signatures(manifest: JobManifest) -> tuple[str, str]:
    """Return the effective spoken and subtitle selections for cached artifacts."""
    artifacts_dir = manifest.artifacts_dir
    script_path = artifacts_dir / manifest.paths.get("script", "script.json")
    if not script_path.is_file():
        return "", ""
    try:
        script = json.loads(script_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", ""
    if not isinstance(script, list) or not all(isinstance(row, dict) for row in script):
        return "", ""

    units: list[dict] | None = None
    units_path = artifacts_dir / manifest.paths.get("units", "units.json")
    if units_path.is_file():
        try:
            data = json.loads(units_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        entries = data.get("units") if isinstance(data, dict) else None
        if isinstance(entries, list) and entries and all(
            isinstance(entry, dict) for entry in entries
        ):
            units = entries

    spoken_parts = [selection_signature(script, "spoken_vi")]
    if units is not None:
        spoken_parts.append(selection_signature(units, "spoken_vi"))
    spoken = hashlib.sha256("|".join(spoken_parts).encode("ascii")).hexdigest()
    subtitle_rows = units if units is not None else script
    subtitle = selection_signature(subtitle_rows, "subtitle_vi")
    return spoken, subtitle


def run_pipeline(
    video: str,
    cfg: JobConfig,
    resume: bool = True,
    force: list[str] | None = None,
    burn: bool = False,
    cancel_cb=None,
) -> dict:
    """Run the pipeline for ``video`` and return the ``report.json`` dict.

    ``video`` may be a local media file **or** an http(s) URL: a URL is
    downloaded first (yt-dlp) and then processed exactly like a local file; the
    URL is recorded on the manifest as ``source_url`` and in the report. The
    cache directory is keyed on the downloaded local file, so re-runs of the
    same URL reuse both the download and every finished stage.

    ``cancel_cb`` (default ``None``) is a zero-argument callable checked at each
    stage boundary; when it returns truthy the pipeline raises
    :class:`PipelineCancelled` before starting the next stage. ``None`` keeps the
    behaviour identical to before.

    ``cfg`` is a prebuilt :class:`JobConfig`: ``vidub batch`` reuses a single
    config across every video in a folder, and each video still gets its own
    cache directory (keyed by ``video_hash``). The config is never mutated.
    """
    # Canonical mode spelling (``subtitles`` is an alias for ``subtitles-only``).
    mode = normalize_mode(cfg.mode)
    # In subtitles-only no synthesis runs, so a TTS engine config (and its
    # non-commercial license guard, which needs that engine) is irrelevant and
    # must not block the run. MT validation still applies because translation
    # still runs; dub/voice-over keep both guards unchanged.
    if mode != SUBTITLES_ONLY:
        # License guardrail: refuse non-commercial engines unless opted in.
        check_tts_engine_allowed(cfg.tts_engine, cfg.allow_nc_models)
        # Language guardrail: fail before any work when the TTS engine cannot
        # speak the requested target language (Vietnamese is first-class).
        validate_langs(cfg.src_lang, cfg.tgt_lang, cfg.tts_engine)
    # Cloning guardrail: --clone-speakers needs diarize + vixtts + NC opt-in, and
    # cannot run at all in subtitles-only (an explicit error, never a silent skip).
    validate_clone_speakers(cfg)
    # Diarization guardrail: fail before any work if HF auth/access is missing.
    if cfg.diarize:
        diarize_mod.preflight(cfg)

    # URL input: resolve to a local file before hashing/ingest. The download is
    # name-stable, so the same URL reuses the same file (and cache directory).
    source_url = video if is_url(video) else ""
    if source_url:
        local_video = download(source_url, cfg.cache_dir)
        log.info("source URL %s -> %s", source_url, local_video)
    else:
        local_video = Path(video)

    force_set = {name.strip() for name in (force or []) if name and name.strip()}
    stages = required_stages(
        cfg.mode, diarize=cfg.diarize, clone_speakers=cfg.clone_speakers
    )
    stage_index = {name: i for i, name in enumerate(stages)}
    force_from = first_forced_index(stages, force_set)
    timings: dict[str, float] = {}

    job_hash = ingest_mod.video_hash(local_video)
    manifest_path = Path(cfg.cache_dir) / job_hash / MANIFEST_NAME
    manifest = JobManifest.load(manifest_path) if manifest_path.exists() else None

    current_spoken_signature, current_subtitle_signature = (
        text_signatures(manifest) if manifest is not None else ("", "")
    )

    previous_mode = manifest.mode if manifest is not None else None
    mode_same = previous_mode is None or previous_mode == cfg.mode
    previous_engine = manifest.tts_engine if manifest is not None else None
    engine_same = previous_engine is None or previous_engine == cfg.tts_engine
    current_tts_signature = tts_signature(cfg)
    # A changed MT configuration invalidates the cached translate output and
    # everything downstream of it (mirrors the TTS-engine rule below). An empty
    # recorded signature means "unknown" (a manifest written before this field
    # existed) and never invalidates, preserving backward compatibility;
    # ``manifest is None`` (first run) cannot invalidate either.
    mt_changed = (
        manifest is not None
        and bool(manifest.mt_signature)
        and manifest.mt_signature != mt_signature(cfg)
    )
    if mt_changed:
        log.info("mt settings changed; re-running translate and downstream stages")
    tts_changed = (
        manifest is not None
        and bool(manifest.tts_signature or current_tts_signature)
        and manifest.tts_signature != current_tts_signature
    )
    if tts_changed:
        log.info("tts voice settings changed; re-running speak and downstream stages")
    # A changed timing configuration invalidates the cached timing output and
    # everything downstream of it. Same "empty means unknown, never invalidate"
    # contract as mt_changed (backward compatible with older manifests).
    timing_changed = (
        manifest is not None
        and bool(manifest.timing_signature)
        and manifest.timing_signature != timing_signature(cfg)
    )
    if timing_changed:
        log.info("timing settings changed; re-running timing and downstream stages")
    export_changed = (
        manifest is not None and bool(manifest.export_signature)
        and manifest.export_signature != export_mod.export_signature(cfg, burn)
    )
    spoken_text_changed = (
        manifest is not None
        and bool(manifest.spoken_text_signature)
        and bool(current_spoken_signature)
        and manifest.spoken_text_signature != current_spoken_signature
    )
    subtitle_text_changed = (
        manifest is not None
        and bool(manifest.subtitle_text_signature)
        and bool(current_subtitle_signature)
        and manifest.subtitle_text_signature != current_subtitle_signature
    )
    # A fresh diarization rewrites segments.json -> invalidate translate..export.
    diarize_new = cfg.diarize and (
        manifest is None or manifest.stages.get("diarize") != "done"
    )

    if manifest is not None:
        if source_url:
            manifest.source_url = source_url
        _sync_config(manifest, cfg)
        manifest.save(manifest_path)

    def may_skip(stage: str) -> bool:
        if not resume:
            return False
        if stage == "export" and (export_changed or subtitle_text_changed):
            return False
        index = stage_index.get(stage)
        if index is not None and force_from is not None and index >= force_from:
            # Forcing a stage invalidates it and every stage downstream of it.
            return False
        if diarize_new:
            diarize_index = stage_index.get("diarize")
            if (
                index is not None
                and diarize_index is not None
                and index > diarize_index
            ):
                # Fresh speaker labels change segments.json -> re-run the stages
                # that consumed them (translate/speak/timing/mix/export).
                return False
        if mt_changed:
            translate_index = stage_index.get("translate")
            if (
                index is not None
                and translate_index is not None
                and index >= translate_index
            ):
                # A different MT configuration produced the cached script ->
                # re-run translate and everything that consumed it.
                return False
        if tts_changed:
            speak_index = stage_index.get("speak")
            if index is not None and speak_index is not None and index >= speak_index:
                return False
        if timing_changed:
            timing_index = stage_index.get("timing")
            if (
                index is not None
                and timing_index is not None
                and index >= timing_index
            ):
                # A different timing configuration produced the cached
                # timed.json -> re-run timing and everything after it.
                return False
        if spoken_text_changed:
            speak_index = stage_index.get("speak")
            if index is not None and speak_index is not None and index >= speak_index:
                return False
        if manifest is None or manifest.stages.get(stage) != "done":
            return False
        if stage in MODE_DEPENDENT_STAGES and not mode_same:
            return False
        if not engine_same:
            speak_index = stage_index.get("speak")
            if index is not None and speak_index is not None and index >= speak_index:
                # The speech audio changed engine -> re-run speak and everything
                # that consumed it (timing/mix/export).
                return False
        return stage_artifacts_present(stage, manifest)

    for stage in stages:
        if cancel_cb is not None and cancel_cb():
            # Cooperative cancellation: stop at the boundary, never mid-stage.
            raise PipelineCancelled(f"pipeline cancelled before stage {stage!r}")

        if stage == "ingest":
            if may_skip("ingest") and manifest is not None:
                log.info("skip ingest (already done)")
            else:
                started = time.perf_counter()
                manifest = ingest_mod.ingest(str(local_video), cfg)
                if source_url:
                    # ingest() builds a fresh manifest; record the URL origin.
                    manifest.source_url = source_url
                    manifest.save(manifest.artifacts_dir / MANIFEST_NAME)
                manifest_path = manifest.artifacts_dir / MANIFEST_NAME
                timings["ingest"] = time.perf_counter() - started
            continue

        if manifest is None or not manifest_path.exists():
            raise RuntimeError(
                f"manifest not found for {video!r}; the ingest stage did not run"
            )

        if may_skip(stage):
            log.info("skip %s (already done)", stage)
            continue

        started = time.perf_counter()
        if stage == "transcribe":
            manifest = transcribe_mod.transcribe(str(manifest_path), cfg)
        elif stage == "diarize":
            manifest = diarize_mod.diarize(str(manifest_path), cfg)
        elif stage == "references":
            manifest = references_mod.extract_references(str(manifest_path), cfg)
        elif stage == "translate":
            manifest = translate_mod.translate(str(manifest_path), cfg)
            # Record the MT settings that produced the cached script. A skipped
            # translate never reaches here, so its previous signature is left
            # untouched (see the mt_changed check in may_skip above).
            manifest.mt_signature = mt_signature(cfg)
            manifest.save(manifest_path)
        elif stage == "speak":
            manifest = speak_mod.speak(str(manifest_path), cfg)
            manifest.tts_signature = tts_signature(cfg)
            manifest.save(manifest_path)
        elif stage == "timing":
            manifest = timing_mod.fit(str(manifest_path), cfg)
            # Record the timing settings that produced the cached timed.json. A
            # skipped timing never reaches here, so its previous signature is
            # left untouched (see the timing_changed check in may_skip above).
            manifest.timing_signature = timing_signature(cfg)
            manifest.save(manifest_path)
        elif stage == "mix":
            manifest = mix_mod.mix(str(manifest_path), cfg)
        elif stage == "export":
            manifest = export_mod.export(
                str(manifest_path), cfg, burn=burn, timings=timings
            )
            spoken_signature, manifest.subtitle_text_signature = text_signatures(
                manifest
            )
            # Subtitles-only can export edited captions without running speech,
            # timing, or mix. Do not acknowledge a spoken edit until those audio
            # consumers have completed; a later switch back to dub must rerun them.
            if (
                normalize_mode(manifest.mode) != SUBTITLES_ONLY
                and {"speak", "timing", "mix"}.issubset(timings)
            ):
                manifest.spoken_text_signature = spoken_signature
            manifest.save(manifest_path)
        else:  # pragma: no cover - guarded by STAGES
            raise ValueError(f"unknown stage: {stage}")
        timings[stage] = time.perf_counter() - started

    if manifest is None:  # pragma: no cover - ingest always sets it
        raise RuntimeError("pipeline produced no manifest")

    report_path = export_mod.output_dir_for(manifest) / export_mod.REPORT_NAME
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        # Export was skipped and its report is gone: rebuild it.
        report = export_mod.build_report(manifest, timings)
        report_path.parent.mkdir(parents=True, exist_ok=True)

    if source_url and not report.get("source_url"):
        # build_report canonicalises source_url from the manifest (top level +
        # meta); this injection is only a fallback for a report.json written
        # before that change.
        report["source_url"] = source_url

    if source_url or not report_path.exists():
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report

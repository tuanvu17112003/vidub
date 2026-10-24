"""The timing stage entry point: :func:`fit` + CLI (S4)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from vidub.config import JobConfig
from vidub.engines.asr import build_asr
from vidub.engines.mt.base import TranslationError
from vidub.engines.tts.piper_tts import PiperTTS
from vidub.manifest import JobManifest
from vidub.stages.speak import build_tts, pronunciation_map
from vidub.stages.translate import build_translator
from vidub.utils.audio import wav_duration
from vidub.utils.fit import available_window, plan_rate_fit
from vidub.utils.pronounce import apply_pronunciation
from vidub.utils.sentences import split_translation
from vidub.utils.text_selection import selected_text
from vidub.utils.textclean import strip_fillers

from vidub.stages.timing.align import _align_unit
from vidub.stages.timing.clipqc import _normalized_engine, _synth_variant, run_tts_qc
from vidub.stages.timing.constants import (
    _TGT_LANG_NAMES,
    SCRIPT_NAME,
    SPEECH_COVERAGE_WARN,
    SPEECH_NAME,
    SUBTITLES_ONLY,
    TIMED_NAME,
    TIMING_NAME,
    TTS_QC_NAME,
    UNIT_RETRANSLATE_OFF,
)
from vidub.stages.timing.synth import (
    _assert_end_boundary,
    _assert_no_bleed,
    _speech_rel_dir,
    fit_segment,
)
from vidub.stages.timing.units import (
    _TimingUnit,
    _missing_indices,
    _timing_units,
    compute_coverage,
)

log = logging.getLogger(__name__)


def _check_mode(mode: str) -> None:
    if mode == SUBTITLES_ONLY:
        raise RuntimeError(
            f"skipped: mode {mode!r} does not run the timing stage"
        )



def fit(
    manifest_path: str,
    cfg: JobConfig,
    tts=None,
    translator=None,
    asr=None,
) -> JobManifest:
    """Fit every spoken unit into its window and write ``timed.json``.

    With :attr:`~vidub.config.JobConfig.sentence_units` on (the default) the
    script rows are grouped into sentence units. When the translate stage's
    authoritative ``units.json`` is present and consistent it supplies the unit
    ``indices``, the window ``start``/``end`` and the synthesis ``vi`` verbatim
    (:func:`_timing_units`); otherwise the rows are re-grouped with
    :func:`vidub.utils.sentences.group_units`. Each multi-segment unit is
    synthesized **once** from its unit text into ``speech/unit_XXXX.wav`` and
    fitted as a whole (so no placed-row boundary can fall inside a sentence).
    With ``--no-sentence-units`` the per-segment path runs byte-identically to
    before.

    SA-1-T3: with :attr:`~vidub.config.JobConfig.align_display` on, every
    multi-member unit's FINAL audio is force-aligned with the project ASR and its
    ``timed.json`` row gains ``align`` + ``member_spans`` (a real, word-edge
    snapped span per member). The ASR is built lazily and every error degrades the
    affected unit to the split fallback — never the run.

    ``tts`` / ``translator`` / ``asr`` override the lazily-built engines (used by
    tests).
    """
    manifest = JobManifest.load(manifest_path)
    _check_mode(manifest.mode)

    artifacts_dir = manifest.artifacts_dir

    script_rel = manifest.paths.get("script", SCRIPT_NAME)
    script_path = artifacts_dir / script_rel
    if not script_path.exists():
        raise FileNotFoundError(
            f"script not found: {script_path}. Run the speak stage first, e.g. "
            f"'python -m vidub.stages.speak {manifest_path}'."
        )
    script = json.loads(script_path.read_text(encoding="utf-8"))

    speech_rel = manifest.paths.get("speech", SPEECH_NAME)
    speech_path = artifacts_dir / speech_rel
    if not speech_path.exists():
        raise FileNotFoundError(
            f"speech not found: {speech_path}. Run the speak stage first, e.g. "
            f"'python -m vidub.stages.speak {manifest_path}'."
        )

    speech = json.loads(speech_path.read_text(encoding="utf-8"))
    speech_by_idx = {int(seg.get("idx", 0)): seg for seg in speech}

    # Invariant: the dub must never silently drop a script segment. A missing
    # speech entry means TTS skipped/failed for that segment.
    missing_speech = _missing_indices(len(script), speech_by_idx)
    if missing_speech:
        raise RuntimeError(
            f"speech coverage invariant violated: script segment(s) "
            f"{missing_speech} have no speech.json entry (TTS skipped or "
            f"failed); re-run the speak stage"
        )

    starts = [float(seg.get("start", 0.0)) for seg in script]
    video_end = float(manifest.duration) if manifest.duration > 0 else 0.0
    # The FINAL segment may run past the video end: nothing follows it, so a
    # mild overrun is acceptable mux behavior and prevents a mid-word tail cut.
    tail_pad = max(float(cfg.tail_pad), 0.0)

    def window_for(index: int) -> float:
        slot_start = starts[index]
        slot_end = float(script[index].get("end", 0.0))
        next_start = starts[index + 1] if index + 1 < len(script) else None
        end = video_end if next_start is not None else video_end + tail_pad
        return available_window(slot_start, slot_end, next_start, end)

    boundaries = {
        index: (
            starts[index + 1]
            if index + 1 < len(script)
            else video_end + tail_pad
        )
        for index in range(len(script))
    }
    windows = {index: window_for(index) for index in range(len(script))}

    # UA-1 sentence units (default on). The timing stage uses the translate
    # stage's authoritative ``units.json`` when it is present and consistent,
    # else it re-groups the script rows with the SAME ``group_units`` call the
    # translate stage used. One synthesis and one placed row cover a whole
    # sentence: an intra-sentence pause cannot exist because no row boundary can
    # fall inside a unit. Each unit's window runs from its first member's start
    # to the next unit's first member start (or the video end + tail pad), so the
    # no-bleed invariant holds at unit granularity. With --no-sentence-units this
    # is skipped entirely.
    units: list[_TimingUnit] = (
        _timing_units(script, manifest, artifacts_dir)[0] if cfg.sentence_units else []
    )
    unit_starts: dict[int, float] = {}
    unit_members: dict[int, list[int]] = {}
    unit_texts: dict[int, str] = {}
    unit_windows: dict[int, float] = {}
    unit_boundaries: dict[int, float] = {}
    for unit_index, unit in enumerate(units):
        members = list(unit.members)
        unit_start = float(unit.start)
        slot_end = float(unit.end)
        next_first = (
            units[unit_index + 1].members[0]
            if unit_index + 1 < len(units)
            else None
        )
        next_start = starts[next_first] if next_first is not None else None
        end = video_end if next_start is not None else video_end + tail_pad
        unit_starts[unit_index] = unit_start
        unit_members[unit_index] = members
        unit_texts[unit_index] = unit.text
        unit_windows[unit_index] = available_window(
            unit_start, slot_end, next_start, end
        )
        unit_boundaries[unit_index] = (
            starts[next_first] if next_first is not None else video_end + tail_pad
        )

    # Preserve natural pace for fitting takes; only overflow requests a faster
    # rate. Display fragments cannot define a whole video's speaking baseline.
    rate_base = 1.0
    rate_plans: dict[int, tuple[float, str]] = {}
    n_resynth = 0
    if cfg.tts_rate_fit:
        natural_durs: dict[int, float] = {}
        for index in range(len(script)):
            row = speech_by_idx.get(index)
            rel = str(row.get("file", f"speech/seg_{index:04d}.wav"))
            path = artifacts_dir / rel
            natural_durs[index] = wav_duration(path) if path.exists() else 0.0
        for index in range(len(script)):
            ls_row, decision = plan_rate_fit(
                natural_durs[index], windows[index], rate_base
            )
            rate_plans[index] = (ls_row, decision)
            if decision == "resynth":
                n_resynth += 1
        log.info(
            "rate fit: baseline length_scale %.3f, %d row(s) to re-synthesize",
            rate_base,
            n_resynth,
        )

    # Engines are built at most once, and only when a shortening retry runs.
    tts_holder: dict[str, object] = {"engine": tts}
    translator_holder: dict[str, object] = {"engine": translator}
    # The same TTS-only pronunciation lexicon the speak stage used, loaded once
    # and applied to every re-synthesised retry text too (M3-C3).
    pronunciation = pronunciation_map(cfg)

    def _synth(text: str, out_wav: Path) -> float:
        engine = tts_holder["engine"]
        if engine is None:
            engine = build_tts(manifest, cfg)
            tts_holder["engine"] = engine
        tts_text, _spoken = apply_pronunciation(text, pronunciation)
        return float(engine.synth(tts_text, str(out_wav)))  # type: ignore[attr-defined]

    def _retranslate(text: str) -> str:
        engine = translator_holder["engine"]
        if engine is None:
            engine = build_translator(manifest)
            translator_holder["engine"] = engine
        result = engine.translate(  # type: ignore[attr-defined]
            [text], manifest.src_lang, manifest.tgt_lang
        )
        return str(result[0]) if result else ""

    # Q-1-T3 lever 1: the concise re-translation is only wired when the engine can
    # actually see the instruction (a chat-capable LLM engine: hunyuan / ollama,
    # or an injected stub that exposes ``_chat``). Otherwise the legacy
    # filler-strip retry is used unchanged, so envit5 stays byte-identical.
    mt_engine = (
        (manifest.mt_engine or "").strip().lower()
        or (cfg.mt_engine or "").strip().lower()
        or "envit5"
    )
    use_concise = (
        callable(getattr(translator, "_chat", None))
        if translator is not None
        else mt_engine in {"hunyuan", "ollama"}
    )

    def _concise_retranslate(source: str, target_chars: int) -> str:
        """One dedicated concise translation with an explicit character target.

        Uses the engine's chat interface (hunyuan: ``_chat(user)``; ollama:
        ``_chat(system, user)``) with a length instruction; falls back to the
        plain ``translate`` of the filler-stripped source when no chat interface
        is available. Never raises into the caller (fit_segment wraps it too).
        """
        engine = translator_holder["engine"]
        if engine is None:
            engine = build_translator(manifest)
            translator_holder["engine"] = engine
        language = _TGT_LANG_NAMES.get(
            (manifest.tgt_lang or "").strip().lower(), manifest.tgt_lang
        )
        prompt = (
            f"Translate the following text into {language} concisely: at most "
            f"{int(target_chars)} characters, keep the meaning exactly, and "
            f"output only the translation.\n\n{source}"
        )
        chat = getattr(engine, "_chat", None)
        if callable(chat):
            try:
                out = chat(prompt)
            except TypeError:  # ollama-style _chat(system, user, ...)
                try:
                    out = chat("You are a professional translator.", prompt)
                except Exception:  # noqa: BLE001 - a bad candidate, never fatal
                    out = ""
            except Exception:  # noqa: BLE001 - a bad candidate, never fatal
                out = ""
            if str(out or "").strip():
                return str(out).strip()
        stripped, _removed = strip_fillers(source)
        result = engine.translate(  # type: ignore[attr-defined]
            [stripped], manifest.src_lang, manifest.tgt_lang
        )
        return str(result[0]) if result else ""

    # FIT-1: rate re-synthesis is Piper-only (the ``length_scale`` knob is
    # Piper-specific, like the QC alternate-voice retry). Built when at least one
    # row wants re-synthesis, or when a multi-segment unit needs the baseline
    # length_scale for its single fresh synthesis (UA-1).
    has_multi_units = any(len(members) > 1 for members in unit_members.values())
    resynth_rate_fn: Callable[[str, Path, float], float] | None = None
    if (
        cfg.tts_rate_fit
        and (n_resynth > 0 or has_multi_units)
        and _normalized_engine(manifest) == "piper"
    ):
        engine_for_rate = tts_holder["engine"]
        if engine_for_rate is None:
            engine_for_rate = build_tts(manifest, cfg)
            tts_holder["engine"] = engine_for_rate
        if isinstance(engine_for_rate, PiperTTS):

            def _resynth_rate(text: str, out_wav: Path, scale: float) -> float:
                tts_text, _spoken = apply_pronunciation(text, pronunciation)
                return float(
                    engine_for_rate.synth(  # type: ignore[attr-defined]
                        tts_text, str(out_wav), length_scale=float(scale)
                    )
                )

            resynth_rate_fn = _resynth_rate

    timed: list[dict] = []
    stats: list[dict] = []
    # Per-row ``(vi_text, speaker)`` for the TTS-QC retry: the segment text on the
    # opt-out path, the joined unit text on the sentence-unit path.
    text_by_idx: dict[int, tuple[str, str]] = {}
    prev_stretch = 1.0

    if cfg.sentence_units:
        speech_rel_dir = _speech_rel_dir(speech_by_idx)
        for unit_index, unit in enumerate(units):
            members = unit_members[unit_index]
            first = members[0]
            unit_text = unit_texts[unit_index]
            speaker = str(script[first].get("speaker") or "spk0")
            text_by_idx[unit_index] = (unit_text, speaker)

            if len(members) == 1:
                # Degenerate unit: reuse the member's pass-1 WAV and the exact
                # FIT-1 plan the per-segment path would use (the rate fit may
                # still re-synthesize it, see fit_segment).
                member_row = speech_by_idx.get(first)
                rel = str(
                    member_row.get("file", f"speech/seg_{first:04d}.wav")
                    if member_row
                    else f"speech/seg_{first:04d}.wav"
                )
                src_wav = artifacts_dir / rel
                if not src_wav.exists():
                    raise FileNotFoundError(f"segment audio not found: {src_wav}")
                plan_ls, plan_decision = (
                    plan_rate_fit(wav_duration(src_wav), unit_windows[unit_index])
                    if cfg.tts_rate_fit else (1.0, "off")
                )
                record, info = fit_segment(
                    artifacts_dir,
                    {"idx": unit_index},
                    src_wav,
                    window=unit_windows[unit_index],
                    slot_start=unit_starts[unit_index],
                    prev_stretch=prev_stretch,
                    cfg=cfg,
                    en_text=str(script[first].get("en", "")),
                    retranslate_fn=_retranslate,
                    synth_fn=_synth,
                    resynth_rate_fn=resynth_rate_fn,
                    rate_length_scale=plan_ls,
                    rate_decision=plan_decision,
                    vi_text=unit_text,
                    concise_retranslate_fn=(
                        _concise_retranslate if use_concise else None
                    ),
                    src_lang=manifest.src_lang,
                    tgt_lang=manifest.tgt_lang,
                )
            else:
                # Measure ONE natural take of the full sentence, then apply the
                # same rate policy as a single-member unit. Concise retries are
                # guarded; the legacy fragment translation retry stays disabled.
                src_wav = (
                    artifacts_dir / f"{speech_rel_dir}/unit_{unit_index:04d}.wav"
                )
                new_dur = float(_synth(unit_text, src_wav))
                if new_dur <= 0 or not src_wav.exists():
                    raise RuntimeError(
                        f"unit {unit_index}: synthesis produced no audio for "
                        f"{src_wav}"
                    )
                rate_scale, rate_decision = (
                    plan_rate_fit(new_dur, unit_windows[unit_index])
                    if cfg.tts_rate_fit else (1.0, "off")
                )
                record, info = fit_segment(
                    artifacts_dir,
                    {"idx": unit_index},
                    src_wav,
                    window=unit_windows[unit_index],
                    slot_start=unit_starts[unit_index],
                    prev_stretch=prev_stretch,
                    cfg=cfg,
                    en_text=" ".join(
                        str(script[index].get("en", "")).strip()
                        for index in members
                        if str(script[index].get("en", "")).strip()
                    ),
                    retranslate_fn=None,
                    synth_fn=_synth,
                    resynth_rate_fn=resynth_rate_fn,
                    rate_length_scale=rate_scale,
                    rate_decision=rate_decision,
                    vi_text=unit_text,
                    retranslate_disabled_reason=UNIT_RETRANSLATE_OFF,
                    concise_retranslate_fn=(
                        _concise_retranslate if use_concise else None
                    ),
                    src_lang=manifest.src_lang,
                    tgt_lang=manifest.tgt_lang,
                )
            record["members"] = list(members)
            text_by_idx[unit_index] = (str(record.get("vi", unit_text)), speaker)
            timed.append(record)
            stats.append(info)
            prev_stretch = float(record["stretch"])
    else:
        for seg in sorted(speech, key=lambda row: int(row.get("idx", 0))):
            idx = int(seg.get("idx", 0))
            rel = str(seg.get("file", f"speech/seg_{idx:04d}.wav"))
            src_wav = artifacts_dir / rel
            if not src_wav.exists():
                raise FileNotFoundError(f"segment audio not found: {src_wav}")
            plan_ls, plan_decision = rate_plans.get(idx, (1.0, "off"))
            record, info = fit_segment(
                artifacts_dir,
                seg,
                src_wav,
                window=windows[idx],
                slot_start=starts[idx],
                prev_stretch=prev_stretch,
                cfg=cfg,
                en_text=str(script[idx].get("en", "")),
                retranslate_fn=_retranslate,
                synth_fn=_synth,
                resynth_rate_fn=resynth_rate_fn,
                rate_length_scale=plan_ls,
                rate_decision=plan_decision,
                vi_text=selected_text(script[idx], "spoken_vi"),
                concise_retranslate_fn=(
                    _concise_retranslate if use_concise else None
                ),
                src_lang=manifest.src_lang,
                tgt_lang=manifest.tgt_lang,
            )
            text_by_idx[idx] = (
                str(record.get("vi", selected_text(script[idx], "spoken_vi"))),
                str(script[idx].get("speaker") or "spk0"),
            )
            timed.append(record)
            stats.append(info)
            prev_stretch = float(record["stretch"])

    # Per-clip TTS QC (M4-T7, QUALITY-ROADMAP.md §3.4/P1.7): score every final
    # fitted clip with NISQA-TTS, retry low-MOS clips once with changed settings,
    # keep the better clip and write a review packet when it is still low. Runs
    # after the fit loop (all fitted wavs exist) and BEFORE the bleed asserts.
    # The scorer is optional: an unavailable scorer/weights degrade to skipped
    # counters and never crash the pipeline.
    if cfg.qc_tts:
        engine_name = _normalized_engine(manifest)

        def _qc_resynth(text: str, out_wav: Path, speaker: str) -> float:
            engine = tts_holder["engine"]
            if engine is None:
                engine = build_tts(manifest, cfg)
                tts_holder["engine"] = engine
            tts_text, _spoken = apply_pronunciation(text, pronunciation)
            return _synth_variant(
                engine, tts_text, out_wav, engine_name, cfg, speaker
            )

        tts_qc = run_tts_qc(artifacts_dir, timed, text_by_idx, cfg, _qc_resynth)
        (artifacts_dir / TTS_QC_NAME).write_text(
            json.dumps(tts_qc, ensure_ascii=False), encoding="utf-8"
        )
        manifest.paths["tts_qc"] = TTS_QC_NAME
        log.info(
            "TTS QC: scored=%d low_mos=%d retried=%d resolved=%d skipped=%d "
            "mos_mean=%s (min %.2f)",
            tts_qc["scored"],
            tts_qc["low_mos"],
            tts_qc["retried"],
            tts_qc["resolved"],
            tts_qc["skipped"],
            tts_qc["mos_mean"],
            tts_qc["mos_min"],
        )

    _assert_no_bleed(
        timed, unit_boundaries if cfg.sentence_units else boundaries
    )
    _assert_end_boundary(timed, stats)

    # SA-1-T3 display alignment (sentence-unit path only): transcribe each
    # multi-member unit's FINAL audio with the project ASR and allocate a real,
    # word-edge-snapped span per member row. The ASR is built LAZILY (only when a
    # multi-member unit needs a span and cfg.align_display is on) and every ASR /
    # parse error degrades the affected unit to the split fallback — never the
    # run. Placement (start/end/file) is never touched here.
    align_counts = {"asr": 0, "split": 0, "single": 0, "unavailable": 0}
    if cfg.sentence_units:
        asr_holder: dict[str, object] = {"engine": asr}
        timed_by_idx = {int(row.get("idx", 0)): row for row in timed}

        def _align_engine():
            engine = asr_holder["engine"]
            if engine is None:
                engine = build_asr(manifest.asr_model)
                asr_holder["engine"] = engine
            return engine

        align_engine = None
        if cfg.align_display and has_multi_units:
            try:
                align_engine = _align_engine()
            except Exception as exc:  # noqa: BLE001 - alignment must not fail a run
                log.warning(
                    "display alignment disabled: could not build the ASR (%s)", exc
                )
                align_engine = None

        for unit in units:
            record = timed_by_idx.get(unit.index)
            if record is None:
                continue
            try:
                if len(unit.members) == 1:
                    member = int(unit.members[0])
                    piece = str(split_translation(str(record.get("vi", unit.text)), unit.shares)[0]).strip()
                    if not piece:
                        piece = selected_text(script[member], "spoken_vi").strip()
                    if piece:
                        record["align"] = "single"
                        record["member_spans"] = [
                            [
                                member,
                                float(record["start"]),
                                float(record["end"]),
                                piece,
                            ]
                        ]
                        align_counts["single"] += 1
                    else:
                        record["align"] = "unavailable"
                        record["member_spans"] = []
                        align_counts["unavailable"] += 1
                    continue
                wav_path = artifacts_dir / str(record.get("file", ""))
                align, spans, align_words = _align_unit(
                    record,
                    unit,
                    script,
                    str(record.get("vi", unit.text)),
                    wav_path,
                    align_engine,
                    manifest.tgt_lang,
                )
                record["align"] = align
                record["member_spans"] = spans
                # Q-1-T4: persist the ASR word timeline (sentence-unit path only)
                # so the subtitle builder can split cues on real word edges. Kept
                # off the row when the alignment fell back to the proportional
                # split, so legacy rows keep today's exact shape.
                if align_words:
                    record["align_words"] = align_words
                align_counts[align] = align_counts.get(align, 0) + 1
            except Exception as exc:  # noqa: BLE001 - never fail the run
                log.warning(
                    "unit %s: display alignment failed (%s); recording "
                    "unavailable",
                    unit.index,
                    exc,
                )
                record["align"] = "unavailable"
                record["member_spans"] = []
                align_counts["unavailable"] += 1

    timed_idxs = {int(record.get("idx", 0)) for record in timed}
    if cfg.sentence_units:
        missing_timed = [
            unit_index
            for unit_index in range(len(units))
            if unit_index not in timed_idxs
        ]
    else:
        missing_timed = _missing_indices(len(script), timed_idxs)
    if missing_timed:
        unit_word = "unit(s)" if cfg.sentence_units else "script segment(s)"
        raise RuntimeError(
            f"timing coverage invariant violated: {unit_word} "
            f"{missing_timed} have no timed.json entry"
        )

    stretched = sum(1 for info in stats if info["action"] == "stretch")
    retranslated = sum(1 for info in stats if info["retranslated"])
    # Every clip row (smart or blind) is listed so the report shows *how* it was
    # cut; ``clipped`` counts only the blind cuts, ``smart_trims`` the pause cuts.
    clips = [
        {
            "idx": int(record["idx"]),
            "start": round(float(info["start"]), 3),
            "seconds": round(float(info["seconds_trimmed"]), 3),
            "seconds_trimmed": round(float(info["seconds_trimmed"]), 3),
            "reason": info["clip_reason"],
            "smart_trim": bool(info["smart_trim"]),
        }
        for record, info in zip(timed, stats)
        if info["action"] == "clip"
    ]
    clipped = sum(1 for info in stats if info["clipped"])
    smart_trims = sum(1 for info in stats if info["smart_trim"])
    stretch_values = [float(info["stretch"]) for info in stats]
    max_stretch = max(stretch_values) if stretch_values else 1.0
    max_delta = max(
        (
            abs(curr - prev)
            for prev, curr in zip(stretch_values, stretch_values[1:])
        ),
        default=0.0,
    )
    filler_removals = sum(int(info["filler_removals"]) for info in stats)

    if cfg.sentence_units:
        coverage = compute_coverage(
            script,
            speech_by_idx,
            [unit_members[unit_index] for unit_index in range(len(units))],
        )
    else:
        coverage = compute_coverage(script, speech_by_idx)

    (artifacts_dir / TIMED_NAME).write_text(
        json.dumps(timed, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pacing = {
        "stretched": stretched,
        "retranslated": retranslated,
        "clipped": clipped,
        "smart_trims": smart_trims,
        "clips": clips,
        "max_stretch": round(max_stretch, 3),
        "max_neighbor_delta": round(max_delta, 3),
        "filler_removals": filler_removals,
    }
    # FIT-1: only present when a new unit ran, so an opt-out job's timing.json is
    # byte-identical to before.
    if cfg.tts_rate_fit or cfg.recenter_audio:
        pacing["rate_fit"] = {
            "resynthesized": sum(
                1 for info in stats if info.get("rate_action") == "resynth"
            ),
            "kept": sum(
                1 for info in stats if info.get("rate_action") == "keep"
            ),
            "failed": sum(
                1 for info in stats if info.get("rate_action") == "failed"
            ),
            "global_length_scale": round(rate_base, 3),
            "recentered": sum(1 for info in stats if info.get("recentered")),
        }
    # SA-1-T3 display-alignment summary (sentence-unit path only, so an opt-out
    # job's timing.json stays byte-identical).
    if cfg.sentence_units:
        pacing["align"] = dict(align_counts)
    (artifacts_dir / TIMING_NAME).write_text(
        json.dumps(pacing, ensure_ascii=False), encoding="utf-8"
    )

    if clipped:
        log.warning(
            "%d segment(s) clipped at the window end (last resort): %s",
            clipped,
            ", ".join(str(clip["idx"]) for clip in clips if not clip["smart_trim"]),
        )
    if smart_trims:
        log.info(
            "%d segment(s) smart-trimmed at a silence: %s",
            smart_trims,
            ", ".join(str(clip["idx"]) for clip in clips if clip["smart_trim"]),
        )

    manifest.speech_coverage = round(coverage, 2)
    manifest.stretched_segments = stretched
    if coverage < SPEECH_COVERAGE_WARN:
        log.warning(
            "speech coverage %.2f is below %.2f: the dub only fills %s of the "
            "spoken time, so long stretches will be silent. Likely causes: "
            "wrong-language ASR (short/hallucinated segments — use the default "
            "auto/large-v3 model for English audio), very short TTS output, or "
            "a TTS segment that failed. Check script.tsv / segments.json.",
            coverage,
            SPEECH_COVERAGE_WARN,
            f"{coverage:.0%}",
        )

    manifest.paths["timed"] = TIMED_NAME
    manifest.paths["timing"] = TIMING_NAME
    manifest.stages["timing"] = "done"
    if tts is None:
        engine = tts_holder["engine"]
        close = getattr(engine, "close", None)
        if callable(close):
            close()
    manifest.save(artifacts_dir / "job.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m vidub.stages.timing <job.json>``."""
    parser = argparse.ArgumentParser(
        prog="vidub.stages.timing",
        description="S4 timing: fit speech segments into their video windows.",
    )
    parser.add_argument("job", help="path to the job.json manifest")
    parser.add_argument(
        "--cache-dir", default=None, help="cache root (default: manifest's)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = JobConfig()
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir

    try:
        manifest = fit(args.job, cfg)
    except (FileNotFoundError, RuntimeError, TranslationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(str(manifest.artifacts_dir / TIMED_NAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

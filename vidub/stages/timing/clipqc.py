"""Per-clip TTS QC (M4-T7): NISQA scoring, one retry, review packets."""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Callable
from pathlib import Path

from vidub.config import JobConfig
from vidub.engines.tts.piper_tts import PiperTTS
from vidub.manifest import JobManifest
from vidub.stages.speak import PIPER_VOICES, resolve_voice
from vidub.utils import ttsqc

from vidub.stages.timing.constants import (
    _FIT_SUFFIX,
    TTS_QC_REVIEW_SUBDIR,
    VXTTS_RETRY_TEMPERATURE,
    VXTTS_RETRY_TOP_K,
)
from vidub.stages.timing.synth import _fit_clip, _fit_to_window

log = logging.getLogger(__name__)


def _normalized_engine(manifest: JobManifest) -> str:
    """Lowercased TTS engine name (``piper`` aliases normalised)."""
    name = (manifest.tts_engine or "piper").strip().lower()
    return "piper" if name in {"piper", "piper-tts", "piper_tts"} else name


def _piper_voice_files(engine: PiperTTS, voice_path: str) -> tuple[Path, Path] | None:
    """Cached ``(model, config)`` paths for an alternate Piper voice.

    ``PiperTTS`` downloads only the *default* voice's config, and its load branch
    fires (overwriting the config) whenever ``model_path`` is ``None``. So both
    alternate files are fetched here and set together. ``None`` on failure lets
    ``PiperTTS`` fall back to its own download.
    """
    try:
        from huggingface_hub import hf_hub_download

        model = Path(hf_hub_download(engine.repo_id, voice_path))
        config = Path(hf_hub_download(engine.repo_id, voice_path + ".json"))
        return model, config
    except Exception as exc:  # noqa: BLE001 - optional alternate-voice files
        log.warning("could not fetch the alternate Piper voice files (%s)", exc)
        return None


def _synth_variant(
    engine,
    text: str,
    out_wav: Path,
    engine_name: str,
    cfg: JobConfig,
    speaker: str,
) -> float:
    """Re-synthesize one QC retry with changed generation settings.

    * ``vixtts`` -> higher sampling ``temperature``/``top_k`` (temporarily
      overrides the engine module's ``_GEN_KWARGS``, restored afterwards).
    * ``piper``  -> the *other* shipped voice from :data:`DEFAULT_VOICE_MAP`
      (``25hours`` when the active voice is ``vais1000``, else ``vais1000``) by
      reconfiguring the ``PiperTTS`` instance, restored afterwards.

    Test stub engines are neither, so they are simply asked to synth again.
    """
    if engine_name == "vixtts":
        try:
            from vidub.engines.tts import vixtts as vixtts_mod
        except Exception:  # noqa: BLE001 - optional engine module
            vixtts_mod = None
        saved = getattr(vixtts_mod, "_GEN_KWARGS", None) if vixtts_mod else None
        if saved is not None:
            vixtts_mod._GEN_KWARGS = {
                **saved,
                "temperature": VXTTS_RETRY_TEMPERATURE,
                "top_k": VXTTS_RETRY_TOP_K,
            }
            try:
                return float(engine.synth(text, str(out_wav)))  # type: ignore[attr-defined]
            finally:
                vixtts_mod._GEN_KWARGS = saved
        return float(engine.synth(text, str(out_wav)))  # type: ignore[attr-defined]

    if engine_name == "piper" and isinstance(engine, PiperTTS):
        active = resolve_voice(cfg.voice_map or {}, speaker)
        alternate = "25hours" if active == "vais1000" else "vais1000"
        alternate_path = PIPER_VOICES.get(alternate, PIPER_VOICES["vais1000"])
        state = (
            engine.voice_path,
            engine.model_path,
            engine.config_path,
            engine._voice,
        )
        engine.voice_path = alternate_path
        engine._voice = None
        alternate_files = _piper_voice_files(engine, alternate_path)
        if alternate_files is not None:
            engine.model_path, engine.config_path = alternate_files
        else:
            engine.model_path = None
            engine.config_path = None
        try:
            return float(engine.synth(text, str(out_wav)))
        finally:
            (
                engine.voice_path,
                engine.model_path,
                engine.config_path,
                engine._voice,
            ) = state

    return float(engine.synth(text, str(out_wav)))  # type: ignore[attr-defined]


def _write_tts_review(
    review_dir: Path,
    idx: int,
    record: dict,
    vi_text: str,
    original_mos: float,
    retry_mos: float | None,
    clip_path: Path,
) -> None:
    """Write ``seg-<idx>.txt`` + a copy of the final clip for human review."""
    review_dir.mkdir(parents=True, exist_ok=True)
    retry_str = "n/a" if retry_mos is None else f"{float(retry_mos):.3f}"
    lines = [
        f"segment: {idx}",
        f"window: {record.get('window')}",
        f"start: {record.get('start')}",
        f"end: {record.get('end')}",
        f"action: {record.get('action')}",
        f"mos original: {float(original_mos):.3f}",
        f"mos retry: {retry_str}",
        "",
        "vi text:",
        vi_text,
    ]
    (review_dir / f"seg-{idx}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    try:
        if clip_path.exists():
            shutil.copyfile(clip_path, review_dir / f"seg-{idx}.wav")
    except OSError as exc:  # noqa: BLE001 - a review copy must not fail the run
        log.warning("could not copy TTS QC review clip %s (%s)", clip_path, exc)


def run_tts_qc(
    artifacts_dir: Path,
    timed: list[dict],
    text_by_idx: dict[int, tuple[str, str]],
    cfg: JobConfig,
    resynth: Callable[[str, Path, str], float],
) -> dict:
    """Score every fitted clip, retry low-MOS ones once; return the sidecar dict.

    ``text_by_idx`` maps each timed row's ``idx`` to ``(vi_text, speaker)`` — the
    segment text on the per-segment path, the joined **unit text** on the
    sentence-unit path. ``resynth(text, out_wav, speaker)`` re-synthesizes that
    text with changed settings (provided by :func:`fit`). A clip whose final MOS
    is still below ``cfg.tts_qc_mos_min`` gets a ``review/tts/seg-<idx>.txt`` +
    ``.wav`` packet (``<idx>`` is the unit index on the unit path).
    """
    scored = low_mos = retried = resolved = skipped = 0
    final_mos: list[float] = []
    review_dir = artifacts_dir / TTS_QC_REVIEW_SUBDIR
    # These are generated summaries of the current fitted clips, not a history.
    # Preserve unrelated notes while discarding packets from an earlier run.
    for packet in review_dir.glob("seg-*"):
        if packet.is_file() and re.fullmatch(r"seg-\d+\.(?:txt|wav)", packet.name):
            packet.unlink()
    min_mos = float(cfg.tts_qc_mos_min)

    for record in timed:
        idx = int(record.get("idx", 0))
        clip_path = artifacts_dir / str(record.get("file", ""))
        original_mos = ttsqc.score_wav(clip_path)
        if original_mos is None:
            skipped += 1
            continue
        scored += 1
        vi_text, speaker = text_by_idx.get(idx, ("", "spk0"))

        kept_mos = float(original_mos)
        kept_path = clip_path
        retry_mos: float | None = None

        if original_mos < min_mos:
            low_mos += 1
            retried += 1
            retry_raw = clip_path.with_name(
                f"{clip_path.stem}_qc{clip_path.suffix}"
            )
            try:
                duration = float(resynth(vi_text, retry_raw, speaker))
            except Exception as exc:  # noqa: BLE001 - a retry must not crash
                log.warning(
                    "segment %s: TTS QC retry synthesis failed (%s)", idx, exc
                )
                duration = 0.0
            if duration > 0 and retry_raw.exists():
                retry_mos = None
                try:
                    stretch = float(record.get("stretch", 1.0)) or 1.0
                    window = float(record.get("window", 0.0))
                    fitted = retry_raw.with_name(
                        f"{retry_raw.stem}{_FIT_SUFFIX}{retry_raw.suffix}"
                    )
                    if str(record.get("action")) == "clip":
                        _fit_clip(retry_raw, fitted, stretch, window)
                    else:
                        _fit_to_window(retry_raw, fitted, stretch, window)
                    retry_mos = ttsqc.score_wav(fitted)
                except Exception as exc:  # noqa: BLE001 - a retry must not crash
                    log.warning(
                        "segment %s: TTS QC retry re-fit failed (%s)", idx, exc
                    )
                # keep the better clip; ties keep the original
                if retry_mos is not None and float(retry_mos) > kept_mos:
                    record["file"] = fitted.relative_to(artifacts_dir).as_posix()
                    record["tts_qc_retry"] = True
                    kept_mos = float(retry_mos)
                    kept_path = fitted

            if kept_mos >= min_mos:
                resolved += 1
            else:
                _write_tts_review(
                    review_dir,
                    idx,
                    record,
                    vi_text,
                    original_mos,
                    retry_mos,
                    kept_path,
                )

        final_mos.append(kept_mos)

    mos_mean = round(sum(final_mos) / len(final_mos), 2) if final_mos else None
    return {
        "enabled": True,
        "mos_min": min_mos,
        "scored": scored,
        "low_mos": low_mos,
        "retried": retried,
        "resolved": resolved,
        "skipped": skipped,
        "mos_mean": mos_mean,
    }

"""Per-clip TTS quality QC with NISQA-TTS (QUALITY-ROADMAP.md P1.7, D5).

Scores every FINAL fitted clip (the audio that actually goes into the mix) with
the non-intrusive **NISQA v2.0** MOS model via the TorchMetrics wrapper
(:class:`torchmetrics.audio.nisqa.NonIntrusiveSpeechQualityAssessment`), so a
garbled/silent/low-naturalness synthesis cannot reach the mix unnoticed.

This module is deliberately defensive: :func:`score_wav` returns ``None`` on ANY
failure (missing file, non-audio file, missing ``torchmetrics``, weight download
or inference error) instead of raising, so the timing gate can degrade to
"skipped" counters and the pipeline continues.

``torchmetrics`` is the only dependency this module adds (``librosa`` and
``requests`` -- the metric's other requirements -- are already installed). The
NISQA weights (~1 MB ``nisqa.tar``) download once on first use into
``~/.torchmetrics/NISQA``; unit tests stub the scorer so they never download.

Note: on torchmetrics >= 1.4 the class is exported as
``NonIntrusiveSpeechQualityAssessment``; :func:`_load_nisqa_class` also accepts
the older ``NISQA`` spelling when present.
"""

from __future__ import annotations

import logging
import math
import wave
from pathlib import Path

__all__ = [
    "MOS_MIN",
    "SAMPLE_RATE",
    "score_wav",
    "nisqa_available",
]

log = logging.getLogger(__name__)

# The metric's expected sample rate (NISQA is trained at 16 kHz).
SAMPLE_RATE = 16000
# MOS (1-5) below which a clip is flagged for a re-synthesis retry.
MOS_MIN = 3.0

# Module-level lazy singleton: the scorer (and its weights) load once.
_SCORER = None
_SCORER_FAILED = False


def _load_nisqa_class():
    """Return the TorchMetrics NISQA class, or ``None`` when unavailable."""
    try:
        from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

        return NonIntrusiveSpeechQualityAssessment
    except ImportError:
        pass
    try:
        from torchmetrics.audio.nisqa import NISQA  # older/newer spelling

        return NISQA
    except ImportError as exc:
        log.warning(
            "TTS QC: NISQA unavailable (%s); install 'torchmetrics' to enable it",
            exc,
        )
        return None


def _get_scorer():
    """Lazily build (once) the NISQA scorer, forcing the weights to load.

    Returns ``None`` when ``torchmetrics``/NISQA cannot be loaded *or* when the
    first (warm-up) inference fails, so the caller degrades to skipped counters.
    The warm-up makes a download/load failure a one-time cost instead of a
    per-clip retry.
    """
    global _SCORER, _SCORER_FAILED
    if _SCORER is not None:
        return _SCORER
    if _SCORER_FAILED:
        return None

    cls = _load_nisqa_class()
    if cls is None:
        _SCORER_FAILED = True
        return None

    try:
        scorer = cls(SAMPLE_RATE)
    except Exception as exc:  # noqa: BLE001 - construction must never crash a run
        log.warning("TTS QC: could not construct the NISQA scorer (%s)", exc)
        _SCORER_FAILED = True
        return None

    try:
        import torch

        with torch.no_grad():
            scorer(torch.zeros(SAMPLE_RATE * 4))
    except Exception as exc:  # noqa: BLE001 - weights/inference unavailable
        log.warning(
            "TTS QC: NISQA weights unavailable (%s); skipping per-clip scoring", exc
        )
        _SCORER_FAILED = True
        return None

    _SCORER = scorer
    log.info("TTS QC: NISQA scorer ready on cpu (%d Hz)", SAMPLE_RATE)
    return _SCORER


def reset_scorer() -> None:
    """Forget the cached scorer/state (used by tests)."""
    global _SCORER, _SCORER_FAILED
    _SCORER = None
    _SCORER_FAILED = False


def nisqa_available() -> tuple[bool, str]:
    """``(importable, torchmetrics_version)`` for the ``vidub doctor`` line.

    Only checks that the Python package imports -- it never downloads weights or
    constructs the model, so the doctor stays fast and side-effect free.
    """
    if _load_nisqa_class() is None:
        return False, ""
    try:
        import torchmetrics

        return True, str(getattr(torchmetrics, "__version__", "?"))
    except Exception:  # noqa: BLE001 - metadata best-effort
        return True, "?"


def _read_wav(path: str | Path) -> tuple["object", int]:
    """Read a mono 16-bit PCM WAV into a float32 numpy array at its native rate."""
    import numpy as np

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if width != 2:
        raise ValueError(f"unsupported sample width {width} (expected 16-bit PCM)")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples), int(rate)


def _resample(samples, orig_rate: int, target_rate: int):
    """Linear-interpolation resample (stdlib numpy only)."""
    import numpy as np

    if orig_rate == target_rate or samples.size == 0:
        return samples
    duration = samples.size / float(orig_rate)
    n_target = max(1, int(round(duration * target_rate)))
    x_old = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_target, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def score_wav(path: str | Path) -> float | None:
    """Return the NISQA overall MOS (1-5) for ``path``, or ``None`` on failure.

    Any problem -- scorer unavailable, missing/corrupt audio, too-short clip,
    inference error -- is logged and returns ``None`` so the caller counts the
    clip as skipped instead of crashing the pipeline.
    """
    scorer = _get_scorer()
    if scorer is None:
        return None
    try:
        import torch

        samples, rate = _read_wav(path)
        if samples.size == 0:
            return None
        samples = _resample(samples, rate, SAMPLE_RATE)
        tensor = torch.from_numpy(samples).float()
        with torch.no_grad():
            output = scorer(tensor)
        # NISQA returns [overall MOS, noisiness, discontinuity, coloration, loudness].
        mos = float(output.reshape(-1)[0].item())
        if not math.isfinite(mos):
            return None
        return mos
    except Exception as exc:  # noqa: BLE001 - scoring must never crash a run
        log.warning("TTS QC: could not score %s (%s)", path, exc)
        return None

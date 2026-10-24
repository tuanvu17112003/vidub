"""Tests for per-clip TTS QC scoring (M4-T7).

The NISQA scorer is always stubbed here, so no test downloads weights (the one
live test is skipped unless ``QC_TTS_LIVE=1``).
"""

from __future__ import annotations

import os
import math
import struct
import wave
from pathlib import Path

import pytest

from vidub.utils import ttsqc

RATE = 16000


def _write_wav(
    path: Path, seconds: float, rate: int = RATE, channels: int = 1
) -> None:
    """Write a mono/stereo 16-bit sine WAV of ``seconds`` duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for i in range(round(seconds * rate)):
        value = int(0.3 * 32767 * math.sin(2.0 * math.pi * 440.0 * i / rate))
        frames += struct.pack("<h", value) * channels
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(bytes(frames))


class _FakeScorer:
    """Stand-in for the NISQA metric: records input lengths, returns fixed MOS."""

    def __init__(self, mos: float = 4.2) -> None:
        self.mos = mos
        self.lengths: list[int] = []

    def __call__(self, tensor):  # noqa: ANN001 - torch tensor
        import torch

        self.lengths.append(int(tensor.shape[-1]))
        return torch.tensor([self.mos, 1.0, 1.0, 1.0, 1.0])


# ------------------------------------------------------------------ failures
def test_score_wav_missing_file_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ttsqc, "_get_scorer", lambda: _FakeScorer())

    assert ttsqc.score_wav(tmp_path / "nope.wav") is None


def test_score_wav_non_audio_file_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ttsqc, "_get_scorer", lambda: _FakeScorer())
    bad = tmp_path / "not_audio.wav"
    bad.write_text("this is not a wav file", encoding="utf-8")

    assert ttsqc.score_wav(bad) is None


def test_score_wav_scorer_unavailable_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ttsqc, "_get_scorer", lambda: None)
    wav = tmp_path / "clip.wav"
    _write_wav(wav, 1.0)

    assert ttsqc.score_wav(wav) is None


def test_get_scorer_none_when_class_unavailable(monkeypatch) -> None:
    ttsqc.reset_scorer()
    try:
        monkeypatch.setattr(ttsqc, "_load_nisqa_class", lambda: None)
        assert ttsqc._get_scorer() is None
        # the failure is remembered -> later calls stay cheap/None
        assert ttsqc._get_scorer() is None
    finally:
        ttsqc.reset_scorer()


# ------------------------------------------------------------------ scoring
def test_score_wav_uses_stubbed_scorer_at_16k(monkeypatch, tmp_path) -> None:
    scorer = _FakeScorer(4.2)
    monkeypatch.setattr(ttsqc, "_get_scorer", lambda: scorer)
    wav = tmp_path / "clip.wav"
    _write_wav(wav, 2.0, rate=16000)

    mos = ttsqc.score_wav(wav)

    assert mos == pytest.approx(4.2)
    assert scorer.lengths == [32000]  # no resample needed


def test_score_wav_resamples_22k_to_16k(monkeypatch, tmp_path) -> None:
    scorer = _FakeScorer(3.5)
    monkeypatch.setattr(ttsqc, "_get_scorer", lambda: scorer)
    wav = tmp_path / "clip22.wav"
    _write_wav(wav, 1.0, rate=22050)

    mos = ttsqc.score_wav(wav)

    assert mos == pytest.approx(3.5)
    assert scorer.lengths == [16000]  # 1.0 s resampled to 16 kHz


def test_score_wav_downmixes_stereo(monkeypatch, tmp_path) -> None:
    scorer = _FakeScorer(3.9)
    monkeypatch.setattr(ttsqc, "_get_scorer", lambda: scorer)
    wav = tmp_path / "clip_stereo.wav"
    _write_wav(wav, 1.0, rate=16000, channels=2)

    assert ttsqc.score_wav(wav) == pytest.approx(3.9)
    assert scorer.lengths == [16000]  # 2 s of frames -> 1 s mono


def test_resample_length() -> None:
    import numpy as np

    out = ttsqc._resample(np.zeros(22050, dtype=np.float32), 22050, 16000)

    assert out.shape[0] == 16000


def test_nisqa_available_returns_tuple_without_download() -> None:
    available, version = ttsqc.nisqa_available()

    assert isinstance(available, bool)
    assert isinstance(version, str)
    # torchmetrics is a declared dependency, so it must import on this stack
    assert available is True
    assert version


def test_mos_min_constant() -> None:
    assert ttsqc.MOS_MIN == 3.0


# --------------------------------------------------------------- live smoke
@pytest.mark.skipif(
    os.environ.get("QC_TTS_LIVE") != "1",
    reason="set QC_TTS_LIVE=1 to score with the real NISQA weights",
)
def test_live_nisqa_score(tmp_path) -> None:
    ttsqc.reset_scorer()
    wav = tmp_path / "tone.wav"
    _write_wav(wav, 5.0, rate=16000)

    mos = ttsqc.score_wav(wav)

    assert mos is not None
    assert 1.0 <= mos <= 5.0

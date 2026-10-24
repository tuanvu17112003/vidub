"""Tests for Piper's per-synthesis rate control (FIT-1).

The voice is stubbed so no model is downloaded and no network is touched; the
tests assert the two-argument ``synth`` call path is unchanged and that a
non-default ``length_scale`` builds a ``SynthesisConfig``.
"""

from __future__ import annotations

import wave
from types import SimpleNamespace
from typing import Any

from piper.config import SynthesisConfig

from vidub.engines.tts.piper_tts import PiperTTS


class _FakeVoice:
    """Minimal Piper voice stub recording every ``synthesize_wav`` call."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(sample_rate=22050)
        self.calls: list[dict[str, Any]] = []

    def synthesize_wav(
        self,
        text: str,
        wav_file: wave.Wave_write,
        syn_config: SynthesisConfig | None = None,
        set_wav_format: bool = True,
    ) -> None:
        self.calls.append(
            {
                "text": text,
                "wav_file": wav_file,
                "syn_config": syn_config,
                "set_wav_format": set_wav_format,
            }
        )


def _stub_tts() -> tuple[PiperTTS, _FakeVoice]:
    tts = PiperTTS()
    voice = _FakeVoice()
    tts._voice = voice  # bypass the lazy download/load
    return tts, voice


def test_default_call_path_has_no_syn_config(tmp_path) -> None:
    tts, voice = _stub_tts()
    duration = tts.synth("xin chào", str(tmp_path / "a.wav"))
    assert duration == 0.0
    assert len(voice.calls) == 1
    call = voice.calls[0]
    assert call["syn_config"] is None
    assert call["set_wav_format"] is False
    assert call["text"] == "xin chào"


def test_explicit_none_is_byte_identical_to_today(tmp_path) -> None:
    tts, voice = _stub_tts()
    tts.synth("xin chào", str(tmp_path / "a.wav"), length_scale=None)
    call = voice.calls[0]
    assert call["syn_config"] is None
    assert call["set_wav_format"] is False
    # same positional shape as the original two-arg call
    assert call["text"] == "xin chào"


def test_length_scale_one_uses_voice_default(tmp_path) -> None:
    tts, voice = _stub_tts()
    tts.synth("xin chào", str(tmp_path / "a.wav"), length_scale=1.0)
    assert voice.calls[0]["syn_config"] is None


def test_length_scale_passes_synthesis_config(tmp_path) -> None:
    tts, voice = _stub_tts()
    tts.synth("xin chào", str(tmp_path / "a.wav"), length_scale=1.15)
    cfg = voice.calls[0]["syn_config"]
    assert isinstance(cfg, SynthesisConfig)
    assert cfg.length_scale == 1.15


def test_length_scale_below_one_speaks_faster(tmp_path) -> None:
    tts, voice = _stub_tts()
    tts.synth("xin chào", str(tmp_path / "a.wav"), length_scale=0.8)
    cfg = voice.calls[0]["syn_config"]
    assert isinstance(cfg, SynthesisConfig)
    assert cfg.length_scale == 0.8

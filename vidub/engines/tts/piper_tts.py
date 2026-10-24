"""Piper TTS engine (``piper-tts``, voice ``vi_VN-vais1000-medium``).

The voice ONNX model + config are fetched once from ``rhasspy/piper-voices`` via
``huggingface_hub.hf_hub_download`` (default HF cache) and loaded lazily on the
first :meth:`synth`. Output is 22.05 kHz mono 16-bit WAV; the duration is
measured from the written file with the stdlib :mod:`wave` module.

piper-tts >= 1.8 API (verified against the installed package):
``PiperVoice.load(model_path, config_path=...)`` and
``voice.synthesize_wav(text, wav_file)``.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

from vidub.engines.tts.base import BaseTTS, TTSError

__all__ = ["PiperTTS"]

log = logging.getLogger(__name__)

REPO_ID = "rhasspy/piper-voices"
VOICE_PATH = "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx"
VOICE_CONFIG_PATH = VOICE_PATH + ".json"


class PiperTTS(BaseTTS):
    """Piper-backed Vietnamese TTS, loading its voice lazily."""

    def __init__(
        self,
        repo_id: str = REPO_ID,
        voice_path: str = VOICE_PATH,
        use_cuda: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.voice_path = voice_path
        self.use_cuda = use_cuda
        self.model_path: Path | None = None
        self.config_path: Path | None = None
        self._voice = None

    # -------------------------------------------------------------- lazy loading
    def _ensure_loaded(self) -> None:
        if self._voice is not None:
            return

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise TTSError(
                "huggingface_hub is not installed; install it with "
                "'pip install huggingface_hub'"
            ) from exc

        if self.model_path is None or self.config_path is None:
            try:
                self.model_path = Path(
                    hf_hub_download(self.repo_id, self.voice_path)
                )
                self.config_path = Path(
                    hf_hub_download(self.repo_id, VOICE_CONFIG_PATH)
                )
            except Exception as exc:  # noqa: BLE001 - download failure
                raise TTSError(
                    f"could not download piper voice '{self.voice_path}' from "
                    f"'{self.repo_id}': {exc}. Check your network connection."
                ) from exc

        try:
            from piper import PiperVoice
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise TTSError(
                "piper-tts is not installed; install it with 'pip install piper-tts'"
            ) from exc

        try:
            self._voice = PiperVoice.load(
                str(self.model_path),
                config_path=str(self.config_path),
                use_cuda=self.use_cuda,
            )
        except Exception as exc:  # noqa: BLE001 - load failure
            raise TTSError(
                f"could not load piper voice '{self.model_path}': {exc}"
            ) from exc

        log.info(
            "piper voice loaded: %s (config %s, %d Hz)",
            self.model_path,
            self.config_path,
            self._voice.config.sample_rate,
        )

    # ------------------------------------------------------------- synthesis
    def synth(
        self,
        text: str,
        out_wav: str,
        *,
        length_scale: float | None = None,
    ) -> float:
        """Synthesise ``text`` to ``out_wav`` and return its duration in seconds.

        ``length_scale`` controls Piper's speaking rate per synthesis: ``> 1``
        speaks slower, ``< 1`` faster, and ``1.0``/``None`` uses the voice's
        default. It is keyword-only so the existing two-argument call is
        unchanged; the synthesis config is only built for a non-default value.
        """
        self._ensure_loaded()
        voice = self._voice
        assert voice is not None

        target = Path(out_wav)
        target.parent.mkdir(parents=True, exist_ok=True)

        syn_config = None
        if length_scale is not None and float(length_scale) != 1.0:
            from piper.config import SynthesisConfig

            syn_config = SynthesisConfig(length_scale=float(length_scale))

        try:
            with wave.open(str(target), "wb") as wav_file:
                # Set the format explicitly so an empty text still yields a valid
                # (silent) WAV header.
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(voice.config.sample_rate)
                if syn_config is None:
                    voice.synthesize_wav(text, wav_file, set_wav_format=False)
                else:
                    voice.synthesize_wav(
                        text,
                        wav_file,
                        syn_config=syn_config,
                        set_wav_format=False,
                    )
        except Exception as exc:  # noqa: BLE001 - synthesis failure
            raise TTSError(f"piper synthesis failed for {target}: {exc}") from exc

        with wave.open(str(target), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
        return frames / float(rate) if rate else 0.0

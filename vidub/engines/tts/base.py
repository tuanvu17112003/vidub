"""Text-to-speech engine interface.

Pipeline stages depend only on :class:`BaseTTS`, so the concrete engine (Piper in
M1, ViXTTS/F5 in M2/M4) can be swapped without touching stage code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["BaseTTS", "TTSError"]


class TTSError(RuntimeError):
    """Raised when a TTS engine cannot load its voice or synthesise audio."""


class BaseTTS(ABC):
    """Interface every TTS engine must implement."""

    @abstractmethod
    def synth(self, text: str, out_wav: str) -> float:
        """Synthesise ``text`` into ``out_wav`` and return the duration (s).

        Implementations write a WAV file to ``out_wav``.
        """
        raise NotImplementedError

"""Audio helpers shared by the S4 timing and S5 mix stages."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

from vidub.utils.proc import find_tool, run

__all__ = ["FFmpegError", "wav_duration", "run_ffmpeg", "ensure_wav_pcm16"]


class FFmpegError(RuntimeError):
    """Raised when ffmpeg is unavailable or a command fails."""


def _require_ffmpeg() -> str:
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise FFmpegError(
            "ffmpeg is required but was not found on PATH. Install it with "
            "'winget install --id Gyan.FFmpeg -e --accept-source-agreements "
            "--accept-package-agreements'."
        )
    return ffmpeg


def wav_duration(path: str | Path) -> float:
    """Return the duration in seconds of the WAV at ``path``."""
    with wave.open(str(path), "rb") as wav_file:
        rate = wav_file.getframerate()
        frames = wav_file.getnframes()
    return frames / float(rate) if rate else 0.0


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    """Run ``ffmpeg <args>`` (thin wrapper over :func:`vidub.utils.proc.run`)."""
    return run([_require_ffmpeg(), *args], capture_output=True)


def ensure_wav_pcm16(path: str | Path) -> Path:
    """Return ``path`` as a 16-bit PCM WAV, re-encoding in place when needed."""
    target = Path(path)
    with wave.open(str(target), "rb") as wav_file:
        already_pcm16 = wav_file.getsampwidth() == 2
    if already_pcm16:
        return target

    tmp = target.with_name(f"{target.stem}.pcm16{target.suffix}")
    run_ffmpeg(["-y", "-i", str(target), "-c:a", "pcm_s16le", str(tmp)])
    tmp.replace(target)
    return target

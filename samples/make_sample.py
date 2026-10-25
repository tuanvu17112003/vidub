"""Generate the English TTS sample clip(s) for the test-suite.

``--which 1`` (default) builds ``samples/demo_en.mp4``; ``--which 2`` builds
``samples/demo_en2.mp4`` from a different two-sentence narration.

Pipeline (per clip):
  1. PowerShell ``System.Speech`` synthesises the fixed English script to a
     temporary ``samples/_voice*.wav``.
  2. ffmpeg muxes a 640x360 ``testsrc2`` video with that voice track
     (``-shortest``), H.264 + AAC, yuv420p.
  3. The temporary voice WAV is deleted.

Idempotent: skips generation when the video is newer than its voice file (or
when the voice file is already gone). Pass ``--force`` to regenerate.

Run::

    ./.venv/Scripts/python.exe samples/make_sample.py --which 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vidub.stages.ingest import probe
from vidub.utils.proc import find_tool, run

__all__ = ["SCRIPT", "SCRIPT_1", "SCRIPT_2", "generate", "main"]

# Fixed narrations — do not change without updating expectations elsewhere.
SCRIPT_1 = (
    "This is a demo clip for the vidub tool. "
    "The weather is sunny today. "
    "We are testing automatic speech recognition. "
    "Vidub will translate and dub this video into Vietnamese. "
    "Thank you for listening."
)
# Sample 2: two sentences, different content, ~12-25 s of narration.
SCRIPT_2 = (
    "This is the second ViDub demonstration clip for batch processing. "
    "It uses a different narration than the first sample and still translates "
    "cleanly into natural Vietnamese speech."
)
# Backwards-compatible alias: script 1 was the only narration before ``--which``.
SCRIPT = SCRIPT_1

SAMPLES_DIR = Path(__file__).resolve().parent
VOICE_WAV = SAMPLES_DIR / "_voice.wav"
DEMO_MP4 = SAMPLES_DIR / "demo_en.mp4"
VOICE_WAV_2 = SAMPLES_DIR / "_voice2.wav"
DEMO_MP4_2 = SAMPLES_DIR / "demo_en2.mp4"

SCRIPTS: dict[int, str] = {1: SCRIPT_1, 2: SCRIPT_2}
VIDEOS: dict[int, Path] = {1: DEMO_MP4, 2: DEMO_MP4_2}
VOICES: dict[int, Path] = {1: VOICE_WAV, 2: VOICE_WAV_2}


def _powershell() -> str:
    for candidate in ("powershell", "pwsh"):
        found = find_tool(candidate)
        if found:
            return found
    raise RuntimeError("PowerShell was not found on PATH")


def synthesize_voice(text: str, out_wav: Path) -> None:
    """Synthesise ``text`` to ``out_wav`` using Windows System.Speech."""
    ps = _powershell()
    wav_arg = str(out_wav).replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        # Default rate delivers ~14 s; -3 slows it to ~20 s so the demo clip
        # lands inside the required 15-30 s window.
        "$synth.Rate = -3; "
        f"$synth.SetOutputToWaveFile('{wav_arg}'); "
        f"$synth.Speak('{text}'); "
        "$synth.Dispose()"
    )
    run(
        [ps, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
    )


def mux(video_out: Path, voice_wav: Path) -> None:
    """Mux the testsrc2 video with the voice track into ``video_out``."""
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required but was not found on PATH. "
            "Install it with: winget install --id Gyan.FFmpeg -e "
            "--accept-source-agreements --accept-package-agreements"
        )
    run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=25",
            "-i",
            str(voice_wav),
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video_out),
        ],
        capture_output=True,
    )


def _is_up_to_date(video_out: Path, voice_wav: Path) -> bool:
    if not video_out.exists():
        return False
    if not voice_wav.exists():
        return True
    return video_out.stat().st_mtime >= voice_wav.stat().st_mtime


def generate(force: bool = False, which: int = 1) -> Path:
    """Generate sample ``which`` (1 → ``demo_en.mp4``, 2 → ``demo_en2.mp4``)."""
    if which not in SCRIPTS:
        raise ValueError(f"unknown sample {which!r}: choose 1 or 2")
    video_out = VIDEOS[which]
    voice_wav = VOICES[which]

    if not force and _is_up_to_date(video_out, voice_wav):
        print(f"up to date: {video_out}")
        return video_out

    print(f"synthesising voice -> {voice_wav}")
    synthesize_voice(SCRIPTS[which], voice_wav)

    try:
        print(f"muxing video     -> {video_out}")
        mux(video_out, voice_wav)
    finally:
        if voice_wav.exists():
            voice_wav.unlink()

    info = probe(video_out)
    print(f"wrote {video_out} (duration {info['duration']:.2f}s, "
          f"{info['width']}x{info['height']} @ {info['fps']:.2f} fps)")
    return video_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="regenerate even if up to date"
    )
    parser.add_argument(
        "--which",
        type=int,
        choices=sorted(SCRIPTS),
        default=1,
        help="which sample to build: 1=demo_en.mp4 (default), 2=demo_en2.mp4",
    )
    args = parser.parse_args(argv)
    generate(force=args.force, which=args.which)
    return 0


if __name__ == "__main__":
    sys.exit(main())

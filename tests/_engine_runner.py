"""Standalone helpers run in a subprocess by the heavy (real-model) tests.

Native runtimes (CTranslate2, torch, demucs, onnxruntime) are pinned to a fresh
process per test so that a crash or hang can never wedge the pytest session; the
parent asserts on the small JSON result written here.

Invoked as::

    python tests/_engine_runner.py <command> <out.json> [args...]

Commands: ``asr`` (ASR; optional model id), ``mt`` (envit5 En→Vi), ``tts``
(piper or vixtts), ``diarize`` (pyannote; auth-gated), ``demucs`` (source
separation).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Same OpenMP workaround as the package, set before any model import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _write(path: str, payload: object) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def cmd_asr(out: str, wav: str, model: str = "tiny", language: str = "en") -> int:
    """Transcribe ``wav`` with the ASR engine for ``model`` (default ``tiny``).

    Writes an object with the *resolved* device + compute type (so callers can
    assert whether CUDA actually worked), the requested model, and the segment
    dicts including optional word-level timing.
    """
    from vidub.engines.asr import build_asr

    engine = build_asr(model)
    segments = engine.transcribe(wav, language or None)
    _write(
        out,
        {
            "model": model,
            "device": engine.device,
            "requested_device": engine.requested_device,
            "compute_type": engine.compute_type,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "speaker": s.speaker,
                    "words": s.words,
                }
                for s in segments
            ],
        },
    )
    return 0


def cmd_mt(out: str, text: str) -> int:
    """Translate ``text`` En→Vi with envit5; write the resolved device + list.

    The device is recorded so callers can assert envit5 actually ran on CUDA
    when the torch build exposes a GPU.
    """
    from vidub.engines.mt.envit5_mt import Envit5Translator

    engine = Envit5Translator()
    translations = engine.translate([text], "en", "vi")
    _write(out, {"device": engine.device, "translations": translations})
    return 0


def cmd_tts(
    out: str,
    wav: str,
    text: str,
    engine: str = "piper",
    ref_voice: str | None = None,
) -> int:
    """Synthesise ``text`` to ``wav`` with ``engine``; write duration + format."""
    if engine == "vixtts":
        from vidub.engines.tts.vixtts import VixttsTTS

        tts = VixttsTTS(ref_voice=ref_voice)
    else:
        from vidub.engines.tts.piper_tts import PiperTTS

        tts = PiperTTS()

    duration = tts.synth(text, wav)

    import wave

    with wave.open(wav, "rb") as wav_file:
        channels = wav_file.getnchannels()
        rate = wav_file.getframerate()
    _write(
        out,
        {
            "duration": duration,
            "wav": wav,
            "engine": engine,
            "channels": channels,
            "rate": rate,
            "device": getattr(tts, "device", None),
        },
    )
    return 0


def cmd_demucs(
    out: str, audio: str, sep_root: str, device: str | None = None
) -> int:
    """Separate ``audio`` (Demucs htdemucs); write the stems dir + device used.

    ``device`` is passed through to Demucs (``cuda``/``cpu``); when omitted the
    mix stage's own probe (``select_sep_device``) decides, and the resolved value
    is echoed back so callers can assert GPU separation.
    """
    from vidub.stages.mix import run_separation, select_sep_device

    resolved = device or select_sep_device()
    track_dir = run_separation(Path(audio), Path(sep_root), device=resolved)
    _write(out, {"track_dir": str(track_dir), "device": resolved})
    return 0


def cmd_diarize(
    out: str, wav: str, model: str = "pyannote/speaker-diarization-3.1"
) -> int:
    """Attempt pyannote diarization; write ``{ok, turns}`` or an error JSON.

    Auth/gating failures are reported as data (never crash) so the parent test
    can assert the actionable message. The real model path only runs with a
    valid ``HF_TOKEN``.
    """
    from vidub.engines.diarize.pyannote_diarizer import (
        DiarizeAuthError,
        PyannoteDiarizer,
    )

    try:
        diarizer = PyannoteDiarizer(model=model)
        diarizer.check_ready()
        turns = diarizer.diarize(wav)
    except DiarizeAuthError as exc:
        _write(
            out,
            {"ok": False, "error_type": "DiarizeAuthError", "error": str(exc)},
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - report instead of crashing
        _write(out, {"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        return 0

    _write(out, {"ok": True, "turns": turns})
    return 0


_COMMANDS = {
    "asr": cmd_asr,
    "mt": cmd_mt,
    "tts": cmd_tts,
    "demucs": cmd_demucs,
    "diarize": cmd_diarize,
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = _COMMANDS.get(args[0]) if args else None
    if command is None:
        names = ",".join(_COMMANDS)
        print(f"usage: {Path(__file__).name} {{{names}}} <out.json> [args...]",
              file=sys.stderr)
        return 2
    return command(*args[1:])


if __name__ == "__main__":
    raise SystemExit(main())

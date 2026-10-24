"""Opt-in VoxCPM2 Vietnamese TTS, isolated in its own Python process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
from vidub.engines.tts.base import BaseTTS, TTSError
from vidub.config import DEFAULT_VOXCPM2_VOICE, VOXCPM2_VOICE_PROFILES

__all__ = ["VoxCPM2TTS"]

_VOICE_SEED = 4200


class VoxCPM2TTS(BaseTTS):
    """Keep the model loaded across utterances without importing its stack in ViDub."""

    def __init__(
        self,
        python_path: str | Path | None = None,
        model_cache: str | Path | None = None,
        ref_voice: str | Path | None = None,
        voice_profile: str = DEFAULT_VOXCPM2_VOICE,
    ) -> None:
        isolated_python = (
            _ROOT / "eval-cache/quality-rework/voxcpm2-env/Scripts/python.exe"
        )
        self.python_path = str(
            python_path
            or os.environ.get("VIDUB_VOXCPM2_PYTHON")
            or (isolated_python if isolated_python.is_file() else sys.executable)
        )
        self.model_cache = str(
            model_cache
            or os.environ.get("VIDUB_VOXCPM2_MODEL_CACHE")
            or _ROOT / "eval-cache/quality-rework/voxcpm2-models"
        )
        self.ref_voice = str(Path(ref_voice).resolve()) if ref_voice else None
        if self.ref_voice and not Path(self.ref_voice).is_file():
            raise TTSError(f"reference voice not found: {self.ref_voice}")
        if voice_profile not in VOXCPM2_VOICE_PROFILES:
            raise TTSError(f"unknown VoxCPM2 voice profile: {voice_profile!r}")
        self.voice_profile = voice_profile
        self._process: subprocess.Popen | None = None
        self._sample_rate = 0

    def _read_response(self) -> dict:
        process = self._process
        if process is None or process.stdout is None:
            raise TTSError("VoxCPM2 worker is not running")
        line = process.stdout.readline()
        if not line:
            code = process.poll()
            raise TTSError(f"VoxCPM2 worker exited without a response (code {code})")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TTSError(f"VoxCPM2 worker returned invalid JSON: {line!r}") from exc
        if not isinstance(response, dict):
            raise TTSError("VoxCPM2 worker returned an invalid response")
        if response.get("error"):
            raise TTSError(str(response["error"]))
        return response

    def _start(self) -> None:
        if self._process is not None:
            return
        env = os.environ.copy()
        env.update(
            HF_HOME=self.model_cache,
            HF_HUB_OFFLINE="1",
            PYTHONIOENCODING="utf-8",
            TRANSFORMERS_OFFLINE="1",
        )
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(_ROOT), env.get("PYTHONPATH", "")) if part
        )
        worker_path = Path(__file__).resolve()
        try:
            self._process = subprocess.Popen(
                [self.python_path, "-u", str(worker_path), "--worker"],
                cwd=str(_ROOT),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise TTSError(
                f"cannot start VoxCPM2 Python at {self.python_path!r}: {exc}. "
                "Set VIDUB_VOXCPM2_PYTHON or install the optional VoxCPM2 extra."
            ) from exc

        try:
            response = self._read_response()
            if response.get("ready") is not True:
                raise TTSError("VoxCPM2 worker did not finish loading the model")
            self._sample_rate = int(response.get("sample_rate", 0))
            if self._sample_rate <= 0:
                raise TTSError("VoxCPM2 worker reported an invalid sample rate")
        except Exception:
            self.close()
            raise

    def synth(self, text: str, out_wav: str) -> float:
        clean = " ".join(str(text).split())
        if not clean:
            raise TTSError("VoxCPM2 received empty text")
        self._start()
        process = self._process
        if process is None or process.stdin is None:
            raise TTSError("VoxCPM2 worker is not accepting requests")
        request = {
            "text": clean,
            "output_path": str(Path(out_wav).resolve()),
            "seed": _VOICE_SEED,
            "voice_profile": self.voice_profile,
        }
        if self.ref_voice:
            request["reference_wav_path"] = self.ref_voice
        try:
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            response = self._read_response()
        except (OSError, BrokenPipeError) as exc:
            self.close()
            raise TTSError(f"VoxCPM2 worker communication failed: {exc}") from exc
        if response.get("sample_rate", self._sample_rate) != self._sample_rate:
            self.close()
            raise TTSError("VoxCPM2 changed its sample rate during synthesis")
        try:
            duration = float(response["duration"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TTSError("VoxCPM2 worker returned no valid audio duration") from exc
        if duration <= 0 or not Path(out_wav).is_file():
            raise TTSError("VoxCPM2 worker did not write a non-empty WAV")
        return duration

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None and process.poll() is None:
                process.stdin.write('{"command":"close"}\n')
                process.stdin.flush()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise during garbage collection
            pass


def _worker_main() -> int:
    """Model-side JSON-lines worker; kept in this module for a single-file adapter."""
    import contextlib
    import io

    def send(payload: dict) -> None:
        sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.__stdout__.flush()

    try:
        import torch
        from voxcpm import VoxCPM

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            model = VoxCPM.from_pretrained(
                "openbmb/VoxCPM2", load_denoiser=False, local_files_only=True
            )
        sample_rate = int(model.tts_model.sample_rate)
    except Exception as exc:  # noqa: BLE001 - report model/dependency failures to caller
        send(
            {
                "error": f"VoxCPM2 failed to load: {exc}. Check the optional "
                "dependency and local model cache. Install the tts-voxcpm2 "
                "extra in the worker Python and set "
                "VIDUB_VOXCPM2_MODEL_CACHE if the checkpoint is elsewhere.",
            }
        )
        return 1

    send({"ready": True, "sample_rate": sample_rate})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                return 0
            torch.manual_seed(int(request["seed"]))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(request["seed"]))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                samples = np.asarray(
                    _generate_audio(
                        model,
                        request["text"],
                        request.get("reference_wav_path"),
                        request.get("voice_profile", DEFAULT_VOXCPM2_VOICE),
                    ),
                    dtype=np.float32,
                ).reshape(-1)
            if samples.size == 0:
                raise ValueError("VoxCPM2 returned empty audio")
            output = Path(request["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
            with wave.open(str(output), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm.tobytes())
            send(
                {
                    "duration": len(samples) / sample_rate,
                    "sample_rate": sample_rate,
                }
            )
        except Exception as exc:  # noqa: BLE001 - report per-utterance failure
            send({"error": f"VoxCPM2 synthesis failed: {exc}"})
    return 0


def _generate_audio(
    model, text: str, ref_voice: str | None, voice_profile: str = DEFAULT_VOXCPM2_VOICE
):
    """Generate with one reference sample or a selected voice-design profile."""
    kwargs = {
        "text": (
            text
            if ref_voice
            else f"{VOXCPM2_VOICE_PROFILES[voice_profile]['style']} {text}"
        ),
        "cfg_value": 2.0,
        "inference_timesteps": 10,
    }
    if ref_voice:
        kwargs["reference_wav_path"] = ref_voice
    return model.generate(**kwargs)


if __name__ == "__main__":
    raise SystemExit(_worker_main())

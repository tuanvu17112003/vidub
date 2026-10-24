"""Run reproducible, ID-matched ASR comparisons on benchmark audio windows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Callable

__all__ = ["run_asr_cases", "main"]


def run_asr_cases(
    cases: list[dict], transcribe: Callable[[Path, float, float], str], engine_id: str
) -> list[dict]:
    """Run one transcription per case and retain its stable ID and source window."""
    rows: list[dict] = []
    seen: set[str] = set()
    for position, case in enumerate(cases):
        case_id = str(case.get("id", "")).strip()
        if not case_id:
            raise ValueError(f"case {position} has no ID")
        if case_id in seen:
            raise ValueError(f"duplicate case ID {case_id!r}")
        seen.add(case_id)
        try:
            audio = Path(str(case["audio"]))
            start, end = float(case["start"]), float(case["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"case {case_id}: missing audio path or time window") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"case {case_id}: invalid time window [{start}, {end}]")
        if not audio.is_file():
            raise FileNotFoundError(f"case {case_id}: audio file not found: {audio}")

        began = time.perf_counter()
        try:
            text = transcribe(audio, start, end)
        except Exception as exc:
            raise RuntimeError(f"ASR failed for case {case_id}: {exc}") from exc
        elapsed = time.perf_counter() - began
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"ASR returned blank text for case {case_id}")
        rows.append(
            {
                "id": case_id,
                "engine": engine_id,
                "text": text.strip(),
                "elapsed_seconds": elapsed,
                **({"audio_sha256": case["audio_sha256"]} if case.get("audio_sha256") else {}),
            }
        )
    return rows


def _read_cases(path: Path) -> list[dict]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read cases from {path}: {exc}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected one or more JSON objects in {path}")
    return rows


def _verify_audio_hashes(cases: list[dict]) -> None:
    """Check each unique source once before running against the frozen corpus."""
    checked: dict[Path, str] = {}
    for case in cases:
        audio = Path(str(case["audio"])).resolve()
        case_id = str(case.get("id", "<unknown>"))
        expected = str(case.get("audio_sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"case {case_id}: audio_sha256 must be 64 hexadecimal characters")
        actual = checked.get(audio)
        if actual is None:
            digest = hashlib.sha256()
            try:
                with audio.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
            except OSError as exc:
                raise FileNotFoundError(f"case {case['id']}: cannot read audio: {audio}") from exc
            actual = digest.hexdigest()
            checked[audio] = actual
        if actual != expected:
            raise ValueError(f"audio hash changed for case {case_id}: {audio}")


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(data: dict, path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _clip_audio(source: Path, start: float, end: float, destination: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to clip benchmark audio")
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(source),
            "-t",
            str(end - start),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if not destination.is_file() or destination.stat().st_size <= 44:
        raise RuntimeError(f"ffmpeg produced an empty audio clip for {source}")
    try:
        with wave.open(str(destination), "rb") as clip:
            if clip.getnframes() <= 0 or clip.getnchannels() != 1:
                raise RuntimeError(f"ffmpeg produced an invalid audio clip: {destination}")
    except (wave.Error, EOFError) as exc:
        raise RuntimeError(f"ffmpeg produced an unreadable audio clip: {destination}") from exc
    return destination


class _GpuMemorySampler:
    """Poll nvidia-smi while inference runs; report peak total used memory in MiB."""

    def __init__(self) -> None:
        self.executable = shutil.which("nvidia-smi")
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        if not self.executable:
            return
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [self.executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
                if values:
                    used = sum(values)
                    self.peak_mib = max(self.peak_mib or 0, used)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(0.25)

    def __enter__(self) -> "_GpuMemorySampler":
        if self.executable:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


def _load_faster_whisper(model_name: str):
    from huggingface_hub import snapshot_download

    from vidub.config import resolve_asr_model
    from vidub.engines.asr.faster_whisper_asr import FasterWhisperASR

    model_id = resolve_asr_model(model_name)
    try:
        local_path = snapshot_download(model_id, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"model {model_id!r} is not cached locally; refusing a network download"
        ) from exc
    # resolve_asr_model accepts repo IDs and preset names; normalize this local
    # Windows path to forward slashes so the wrapper treats it as a path.
    engine = FasterWhisperASR(Path(local_path).as_posix())
    return engine, {
        "model": model_id,
        "checkpoint_revision": Path(local_path).name,
        "settings": {
            "language": "en",
            "beam_size": 1,
            "word_timestamps": False,
            "vad_filter": True,
            "clip_sample_rate": 16000,
            "clip_channels": 1,
        },
        "device": engine.device,
        "compute_type": engine.compute_type,
        "runtime_packages": _package_versions(("faster-whisper", "ctranslate2", "huggingface-hub")),
    }


def _load_qwen3(model_name: str, revision: str | None):
    try:
        from huggingface_hub import snapshot_download
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3 ASR is missing from the isolated .eval-venv-qwen3 environment"
        ) from exc
    try:
        local_path = snapshot_download(
            model_name, revision=revision, local_files_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"Qwen3 model {model_name!r} is not cached locally; refusing a network download"
        ) from exc
    model = Qwen3ASRModel.from_pretrained(
        local_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    return model, {
        "model": model_name,
        "checkpoint_revision": Path(local_path).name,
        "settings": {
            "language": "English",
            "dtype": "bfloat16",
            "device_map": "cuda:0",
            "clip_sample_rate": 16000,
            "clip_channels": 1,
        },
        "runtime_packages": _package_versions(("qwen-asr", "torch", "transformers", "huggingface-hub")),
    }


def _run_qwen3_in_isolated_venv(argv: list[str]) -> int:
    python = Path(__file__).resolve().parents[1] / ".eval-venv-qwen3" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError(
            f"isolated Qwen environment not found: {python}; create it and install qwen-asr"
        )
    command = [str(python), str(Path(__file__).resolve()), *argv]
    return subprocess.run(command, check=False).returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--engine", choices=("faster-whisper", "qwen3"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", help="optional immutable model revision")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    sidecar = args.out.with_suffix(".run.json")
    if args.out.exists() or sidecar.exists():
        print(f"error: output or run metadata already exists: {args.out} / {sidecar}", file=sys.stderr)
        return 1
    try:
        cases = _read_cases(args.cases)
        _verify_audio_hashes(cases)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.engine == "qwen3" and Path(sys.executable).resolve() != (
            Path(__file__).resolve().parents[1] / ".eval-venv-qwen3" / "Scripts" / "python.exe"
        ).resolve():
            return _run_qwen3_in_isolated_venv(raw_argv)

        if args.engine == "faster-whisper":
            model, metadata = _load_faster_whisper(args.model)

            def transcribe(audio: Path, start: float, end: float) -> str:
                with tempfile.TemporaryDirectory(prefix="vidub-asr-") as scratch:
                    clip = _clip_audio(audio, start, end, Path(scratch) / "clip.wav")
                    segments = model.transcribe(str(clip), language="en", word_timestamps=False)
                    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())

        else:
            model, metadata = _load_qwen3(args.model, args.revision)

            def transcribe(audio: Path, start: float, end: float) -> str:
                with tempfile.TemporaryDirectory(prefix="vidub-asr-") as scratch:
                    clip = _clip_audio(audio, start, end, Path(scratch) / "clip.wav")
                    result = model.transcribe(audio=str(clip), language="English")
                    return str(result[0].text if result else "")

        with _GpuMemorySampler() as gpu:
            rows = run_asr_cases(cases, transcribe, args.engine)
        command = [sys.executable, str(Path(__file__).resolve()), *raw_argv]
        run = {
            **metadata,
            "engine": args.engine,
            "command": command,
            "case_count": len(rows),
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
            "peak_gpu_memory_mib": gpu.peak_mib,
            "gpu_memory_status": "available" if gpu.peak_mib is not None else "unavailable",
        }
        _write_jsonl(rows, args.out)
        _write_json(run, sidecar)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} ASR result(s): {args.out}")
    print(f"wrote run metadata: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Automatic speech recognition engine adapters.

:func:`build_asr` picks the adapter that can load a given model:

* standard faster-whisper sizes and CTranslate2 repo ids -> :class:`FasterWhisperASR`;
* the ``vinai/PhoWhisper-*`` repos (transformers/PyTorch weights only, no
  ``model.bin``) -> :class:`PhoWhisperASR` via the already-installed transformers
  pipeline.

Both adapters share the same interface (``device`` / ``compute_type``
attributes and a ``transcribe(path, language, word_timestamps)`` method).

The Level-1 transcript QC layer additionally needs an alternative hypothesis
for a flagged window; only :class:`FasterWhisperASR` exposes
``transcribe_window(path, start, end, language)`` (a ``clip_timestamps``-scoped
re-run with changed beam/temperature settings). The PhoWhisper path has no such
method, so QC falls back to text-only statistics there.
"""

from __future__ import annotations

import logging

from vidub.config import resolve_asr_model
from vidub.engines.asr.faster_whisper_asr import (
    MIN_SEGMENT_SECONDS,
    FasterWhisperASR,
)
from vidub.manifest import Segment

__all__ = ["build_asr", "FasterWhisperASR", "PhoWhisperASR"]

log = logging.getLogger(__name__)

# Repos that only ship transformers (PyTorch) weights: verified 2026-09-28 that
# vinai/PhoWhisper-* have no CTranslate2 `model.bin`, so faster-whisper cannot
# load them; they run through the transformers pipeline instead.
_TRANSFORMERS_ASR_PREFIXES = ("vinai/PhoWhisper",)

# Grouping for engines that only return word-level timestamps.
_WORD_SEGMENT_GAP = 0.7
_WORD_SEGMENT_MAX = 8.0
_TARGET_RATE = 16000


def build_asr(model: str, device: str = "auto"):
    """Return the ASR engine that can load ``model`` (preset/size/repo id)."""
    resolved = resolve_asr_model(model)
    if any(resolved.startswith(prefix) for prefix in _TRANSFORMERS_ASR_PREFIXES):
        return PhoWhisperASR(resolved, device=device)
    return FasterWhisperASR(resolved, device=device)


class PhoWhisperASR:
    """Vietnamese PhoWhisper models via the transformers ASR pipeline.

    Same interface as :class:`FasterWhisperASR`: ``device`` / ``compute_type``
    attributes and ``transcribe(path, language=None, word_timestamps=True)``
    returning :class:`~vidub.manifest.Segment`\\ s (with ``words`` when
    requested). Runs on CPU because the M2 torch wheel is CPU-only.
    """

    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = resolve_asr_model(model_id)
        self.model_size = model_id
        self.requested_device = device
        self.device, self.compute_type = self._resolve_device(device)
        self.pipeline = self._load()

    # ------------------------------------------------------------------ loading
    def _resolve_device(self, requested: str) -> tuple[str, str]:
        """Map the requested device to (device, compute_type) for torch."""
        cuda = False
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - missing/broken torch -> CPU
            cuda = False

        if requested == "cpu":
            return "cpu", "float32"
        if cuda:
            return "cuda", "float16"
        if requested == "cuda":
            log.warning(
                "device=cuda requested but torch has no CUDA; using CPU float32"
            )
        return "cpu", "float32"

    def _load(self):
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "transformers is required for PhoWhisper; install it with "
                "'pip install transformers'"
            ) from exc

        device_index = 0 if self.device == "cuda" else -1
        log.info(
            "loading PhoWhisper '%s' via transformers (device=%s, compute_type=%s)",
            self.model_id,
            self.device,
            self.compute_type,
        )
        try:
            loaded = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=device_index,
            )
        except Exception as exc:  # noqa: BLE001 - clear, actionable error
            raise RuntimeError(
                f"could not load PhoWhisper model '{self.model_id}': {exc}. The "
                "model is downloaded from Hugging Face on first use; check the "
                "network or pick another model."
            ) from exc
        log.info("PhoWhisper '%s' loaded on %s", self.model_id, self.device)
        return loaded

    # -------------------------------------------------------------- transcribing
    def transcribe(
        self,
        wav_path: str,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> list[Segment]:
        """Transcribe ``wav_path`` with PhoWhisper; ``language=None`` autodetects."""
        audio, rate = _read_wav(wav_path)
        generate_kwargs = {"language": language} if language else {}
        result = self.pipeline(
            {"array": audio, "sampling_rate": rate},
            return_timestamps="word" if word_timestamps else True,
            generate_kwargs=generate_kwargs,
        )

        chunks = result.get("chunks") or []
        if word_timestamps:
            segments = _group_words(_chunks_to_words(chunks))
        else:
            segments = _segments_from_chunks(chunks)

        if not segments:
            text = (result.get("text") or "").strip()
            if text:
                end = float(chunks[-1]["timestamp"][1]) if chunks else 0.0
                segments = [
                    Segment(start=0.0, end=end, text=text, speaker="spk0")
                ]
        log.info("transcribed %d segment(s)", len(segments))
        return segments


def _read_wav(path: str) -> tuple["object", int]:
    """Read a WAV file as mono float32 at 16 kHz (numpy array, rate)."""
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:  # pragma: no cover - ingest always writes 16-bit PCM
        raise RuntimeError(f"unsupported WAV sample width: {width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if rate != _TARGET_RATE and len(audio):
        duration = len(audio) / float(rate)
        target_len = max(int(round(duration * _TARGET_RATE)), 1)
        audio = np.interp(
            np.linspace(0.0, duration, target_len, endpoint=False),
            np.linspace(0.0, duration, len(audio), endpoint=False),
            audio,
        ).astype(np.float32)
        rate = _TARGET_RATE
    return audio, rate


def _chunks_to_words(chunks) -> list[dict[str, float | str]]:
    """Convert transformers word-level chunks to ``{start,end,word}`` dicts."""
    words: list[dict[str, float | str]] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        timestamp = chunk.get("timestamp") or (None, None)
        start = timestamp[0] if len(timestamp) > 0 else None
        end = timestamp[1] if len(timestamp) > 1 else None
        start = float(start) if start is not None else 0.0
        end = float(end) if end is not None else start
        if end < start:
            end = start
        words.append({"start": start, "end": end, "word": text})
    return words


def _segment_from_words(
    words: list[dict[str, float | str]],
) -> Segment | None:
    start = float(words[0]["start"])
    end = max(float(words[-1]["end"]), start)
    text = " ".join(str(word["word"]) for word in words).strip()
    if not text or (end - start) < MIN_SEGMENT_SECONDS:
        return None
    return Segment(start=start, end=end, text=text, speaker="spk0", words=list(words))


def _group_words(words: list[dict[str, float | str]]) -> list[Segment]:
    """Group word chunks into segments at pauses / a max duration."""
    segments: list[Segment] = []
    current: list[dict[str, float | str]] = []
    for word in words:
        if current and (
            float(word["start"]) - float(current[-1]["end"]) > _WORD_SEGMENT_GAP
            or float(word["end"]) - float(current[0]["start"]) > _WORD_SEGMENT_MAX
        ):
            segment = _segment_from_words(current)
            if segment is not None:
                segments.append(segment)
            current = []
        current.append(word)
    if current:
        segment = _segment_from_words(current)
        if segment is not None:
            segments.append(segment)
    return segments


def _segments_from_chunks(chunks) -> list[Segment]:
    """Segment-level chunks (``return_timestamps=True``) -> segments, no words."""
    segments: list[Segment] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        timestamp = chunk.get("timestamp") or (None, None)
        start = timestamp[0] if len(timestamp) > 0 else None
        end = timestamp[1] if len(timestamp) > 1 else None
        start = float(start) if start is not None else 0.0
        end = float(end) if end is not None else start
        if end < start:
            end = start
        segments.append(Segment(start=start, end=end, text=text, speaker="spk0"))
    return segments

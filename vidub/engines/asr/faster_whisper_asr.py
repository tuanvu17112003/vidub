"""faster-whisper ASR engine wrapper.

Backed by CTranslate2 (no PyTorch). Device policy:

* ``"cpu"``  -> load on CPU with ``compute_type="int8"``;
* ``"auto"`` -> pick CUDA **only** when the GPU probe passes
  (:func:`vidub.utils.cuda.gpu_whisper_available`), otherwise CPU;
* ``"cuda"`` -> attempt CUDA, fall back on failure.

On CUDA the compute type is ``float16``, on CPU ``int8``; the resolved device +
compute type are logged (and exposed as ``.device`` / ``.compute_type``). A
missing CUDA runtime (e.g. ``cublas64_12.dll``) can surface at load *or* at
encode time, so CPU fallback is applied in both places and the wrapper never
crashes on a missing/broken CUDA install. A genuine model-download failure
surfaces as a clear ``RuntimeError``.
"""

from __future__ import annotations

import logging

from vidub.config import resolve_asr_model
from vidub.manifest import Segment
from vidub.utils.cuda import gpu_whisper_available

__all__ = ["FasterWhisperASR"]

log = logging.getLogger(__name__)

MIN_SEGMENT_SECONDS = 0.3
COMPUTE_TYPE_CUDA = "float16"
COMPUTE_TYPE_CPU = "int8"


class FasterWhisperASR:
    """Transcribe a 16 kHz mono WAV into :class:`~vidub.manifest.Segment`\\ s.

    ``model_size`` accepts a preset/size/repo id and is resolved via
    :func:`vidub.config.resolve_asr_model`; any Hugging Face repo id that ships
    a CTranslate2 model works (faster-whisper downloads it on first use).
    """

    def __init__(self, model_size: str, device: str = "auto") -> None:
        self.model_size = model_size
        self.model_id = resolve_asr_model(model_size)
        self.requested_device = device
        self.device = device
        self.compute_type = "auto"
        self.model = None
        self._model_cls = None
        self._load()

    # ------------------------------------------------------------------ loading
    def _import_model_cls(self):  # noqa: ANN202 - fastpath, optional dependency
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "faster-whisper is not installed; install it with "
                "'pip install faster-whisper'"
            ) from exc
        return WhisperModel

    def _load(self) -> None:
        self._model_cls = self._import_model_cls()

        if self.requested_device == "cpu":
            self._load_cpu()
            return

        target = self.requested_device
        if target == "auto":
            # Only commit to CUDA when the GPU actually works; otherwise load
            # the CPU model straight away instead of racing a doomed init.
            gpu_ok, reason = gpu_whisper_available()
            if not gpu_ok:
                log.info("CUDA probe failed (%s); loading on CPU int8", reason)
                self._load_cpu()
                return
            target = "cuda"

        compute_type = (
            COMPUTE_TYPE_CUDA if target == "cuda" else COMPUTE_TYPE_CPU
        )
        try:
            log.info(
                "loading faster-whisper '%s' on device=%s (compute_type=%s)",
                self.model_id,
                target,
                compute_type,
            )
            self.model = self._model_cls(
                self.model_id, device=target, compute_type=compute_type
            )
            self.device = target
            self.compute_type = compute_type
            log.info(
                "faster-whisper resolved device=%s compute_type=%s",
                self.device,
                self.compute_type,
            )
            return
        except Exception as exc:  # noqa: BLE001 - any CUDA failure -> CPU
            log.warning(
                "ASR load on device=%s compute_type=%s failed (%s); "
                "falling back to CPU int8",
                target,
                compute_type,
                exc,
            )

        self._load_cpu()

    def _load_cpu(self) -> None:
        try:
            log.info(
                "loading faster-whisper '%s' on device=cpu (compute_type=%s)",
                self.model_id,
                COMPUTE_TYPE_CPU,
            )
            self.model = self._model_cls(
                self.model_id, device="cpu", compute_type=COMPUTE_TYPE_CPU
            )
            self.device = "cpu"
            self.compute_type = COMPUTE_TYPE_CPU
            log.info(
                "faster-whisper resolved device=cpu compute_type=%s",
                COMPUTE_TYPE_CPU,
            )
        except Exception as exc:  # noqa: BLE001 - re-raise with a clear message
            raise RuntimeError(
                f"could not load faster-whisper model '{self.model_id}' on GPU "
                f"or CPU: {exc}. Check your network connection (the model is "
                "downloaded from Hugging Face on first use) or pick another "
                "model size."
            ) from exc

    # -------------------------------------------------------------- transcribing
    def transcribe(
        self,
        wav_path: str,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> list[Segment]:
        """Transcribe ``wav_path``; ``language=None`` auto-detects.

        ``word_timestamps`` asks CTranslate2 for per-word timing (filled into
        ``Segment.words`` when available). CUDA library problems can surface
        only here, so a failed GPU pass is retried once on CPU before giving up.
        """
        try:
            return self._transcribe_once(wav_path, language, word_timestamps)
        except Exception as exc:  # noqa: BLE001 - GPU runtime failure -> CPU
            if self.device == "cpu":
                raise
            log.warning(
                "inference on device=%s failed (%s); reloading on CPU int8",
                self.device,
                exc,
            )
            self._load_cpu()
            return self._transcribe_once(wav_path, language, word_timestamps)

    def _transcribe_once(
        self, wav_path: str, language: str | None, word_timestamps: bool
    ) -> list[Segment]:
        if self.model is None:  # pragma: no cover - defensive
            raise RuntimeError("ASR model is not loaded")

        segments_iter, info = self.model.transcribe(
            wav_path,
            language=language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            beam_size=1,
            word_timestamps=word_timestamps,
        )
        log.info(
            "detected language=%s (probability=%.2f)",
            getattr(info, "language", "?"),
            getattr(info, "language_probability", 0.0),
        )

        segments: list[Segment] = []
        for raw in segments_iter:  # the generator is consumed here
            text = (raw.text or "").strip()
            start = float(raw.start)
            end = float(raw.end)
            if not text or (end - start) < MIN_SEGMENT_SECONDS:
                continue
            segments.append(
                Segment(
                    start=start,
                    end=end,
                    text=text,
                    speaker="spk0",
                    words=_collect_words(raw) if word_timestamps else None,
                    meta=_collect_meta(raw),
                )
            )
        log.info("transcribed %d segment(s)", len(segments))
        return segments

    # ------------------------------------------------------------- window retry
    def transcribe_window(
        self, wav_path: str, start: float, end: float, language: str | None
    ) -> Segment | None:
        """Re-transcribe just ``[start, end]`` s with changed settings.

        The Level-1 QC layer uses this to produce an alternative hypothesis for
        a flagged (hallucination/garbled) segment: beam search
        (``beam_size=5``/``best_of=5``) and a temperature fallback
        (``[0.0, 0.2]``) instead of the greedy first pass. ``clip_timestamps``
        bounds the work and, per faster-whisper, the returned segment times are
        absolute (the clip offset is already applied). Returns the first
        produced :class:`~vidub.manifest.Segment` (with meta/words) or ``None``
        when the window yields nothing. As in :meth:`transcribe`, a failed GPU
        pass is retried once on CPU.
        """
        try:
            return self._transcribe_window_once(wav_path, start, end, language)
        except Exception as exc:  # noqa: BLE001 - GPU runtime failure -> CPU
            if self.device == "cpu":
                raise
            log.warning(
                "window inference on device=%s failed (%s); reloading on CPU int8",
                self.device,
                exc,
            )
            self._load_cpu()
            return self._transcribe_window_once(wav_path, start, end, language)

    def _transcribe_window_once(
        self, wav_path: str, start: float, end: float, language: str | None
    ) -> Segment | None:
        if self.model is None:  # pragma: no cover - defensive
            raise RuntimeError("ASR model is not loaded")

        segments_iter, _info = self.model.transcribe(
            wav_path,
            language=language,
            clip_timestamps=[float(start), float(end)],
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2],
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        for raw in segments_iter:  # the generator is consumed here
            text = (raw.text or "").strip()
            if not text:
                continue
            return Segment(
                start=float(raw.start),
                end=float(raw.end),
                text=text,
                speaker="spk0",
                words=_collect_words(raw),
                meta=_collect_meta(raw),
            )
        return None


def _collect_meta(raw) -> dict[str, float]:
    """Per-segment confidence signals from a faster-whisper segment."""
    return {
        "avg_logprob": float(raw.avg_logprob),
        "compression_ratio": float(raw.compression_ratio),
        "no_speech_prob": float(raw.no_speech_prob),
        "temperature": float(raw.temperature),
    }


def _collect_words(raw) -> list[dict[str, float | str]] | None:
    """Word-level timing from a faster-whisper segment, or ``None``."""
    raw_words = getattr(raw, "words", None)
    if not raw_words:
        return None
    words: list[dict[str, float | str]] = []
    for word in raw_words:
        text = (getattr(word, "word", "") or "").strip()
        if not text:
            continue
        words.append(
            {
                "start": float(getattr(word, "start", 0.0) or 0.0),
                "end": float(getattr(word, "end", 0.0) or 0.0),
                "word": text,
            }
        )
    return words or None

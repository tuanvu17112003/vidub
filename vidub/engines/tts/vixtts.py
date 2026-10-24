"""ViXTTS TTS engine — natural Vietnamese speech + zero-shot voice cloning.

Opt-in only: the ``capleaf/viXTTS`` weights are licensed under the **Coqui
Public Model License (CPML, non-commercial)**, so this engine is refused unless
``--allow-nc-models`` / ``JobConfig.allow_nc_models`` is set (see
:mod:`vidub.config`).

Design notes
------------
* **Lazy + subprocess-safe:** nothing heavyweight (torch, coqui-tts, the 1.9 GB
  checkpoint) is imported or downloaded at module import; the model loads on the
  first :meth:`VixttsTTS.synth`.
* **Model files** come from ``capleaf/viXTTS`` (``model.pth``, ``config.json``,
  ``vocab.json``) plus ``speakers_xtts.pth`` from ``coqui/XTTS-v2`` — exactly the
  set the upstream ``thinhlpg/vixtts-demo`` reference downloads. Everything is
  cached in the default Hugging Face cache.
* **Device** follows the installed torch build: the model is moved to ``cuda``
  when ``torch.cuda.is_available()`` (a move/inference failure falls back to
  ``cpu``), and the resolved device is exposed as :attr:`VixttsTTS.device`.
* **torchaudio/torchcodec compatibility:** coqui-tts' XTTS imports
  ``torchaudio`` and ``TTS/__init__`` requires ``torchcodec`` on torch>=2.9.
  This environment has no matching ``torchaudio`` wheel (no torchaudio is
  installed for torch 2.14.0) and torchcodec is a heavy binary extra, so we
  register a small ``torchaudio`` shim built on ``soundfile``/``librosa``/``torch``
  and satisfy the import guards. XTTS inference only needs ``load`` /
  ``functional.resample`` / ``transforms.MelSpectrogram``. Output is written with
  the stdlib ``wave`` module (no torchaudio.save).
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
import tempfile
import types
import wave
from pathlib import Path

from vidub.engines.tts.base import BaseTTS, TTSError
from vidub.engines.tts.vixtts_norm import normalize_text

__all__ = ["VixttsTTS"]

log = logging.getLogger(__name__)

MODEL_REPO = "capleaf/viXTTS"
BASE_REPO = "coqui/XTTS-v2"
SPEAKER_FILE = "speakers_xtts.pth"
MODEL_FILES = ("model.pth", "config.json", "vocab.json")
# Reference samples shipped in the model repo (6-30 s of Vietnamese speech).
REF_CANDIDATES = ("vi_sample.wav", "samples/nu-nhe-nhang.wav")
LANGUAGE = "vi"
SAMPLE_RATE = 24000
MAX_CHARS = 250

# viXTTS-tuned sampling settings (from the thinhlpg/vixtts-demo reference).
# The engine's own sentence splitter is disabled: this coqui-tts build has no
# char limit for "vi", and we chunk the text ourselves (_split_chunks).
_GEN_KWARGS = {
    "temperature": 0.3,
    "length_penalty": 1.0,
    "repetition_penalty": 10.0,
    "top_k": 30,
    "top_p": 0.85,
    "enable_text_splitting": False,
}

_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


class VixttsTTS(BaseTTS):
    """Coqui XTTS (viXTTS checkpoint) with lazy loading."""

    def __init__(
        self,
        ref_voice: str | None = None,
        model_repo: str = MODEL_REPO,
        base_repo: str = BASE_REPO,
    ) -> None:
        self.ref_voice = ref_voice
        self.model_repo = model_repo
        self.base_repo = base_repo
        self.model_dir: Path | None = None
        self.speaker_file: Path | None = None
        self.ref_path: Path | None = None
        # Resolved inference device ("cuda"/"cpu"); set on first load.
        self.device: str | None = None
        self._config = None
        self._model = None
        self._latents = None

    # -------------------------------------------------------------- lazy loading
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        self.model_dir, self.speaker_file = self._download_model()
        self.ref_path = self._resolve_ref_voice()

        XttsConfig, Xtts = _import_xtts()
        config_path = self.model_dir / "config.json"
        config = XttsConfig()
        config.load_json(str(config_path))

        log.info("loading viXTTS '%s' from %s", self.model_repo, self.model_dir)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=str(self.model_dir),
            speaker_file_path=str(self.speaker_file),
            use_deepspeed=False,
        )
        model.eval()
        _patch_vi_tokenizer(model)

        # Move the model to the GPU when the torch build exposes one (coqui XTTS
        # reads ``self.device`` off the parameter device, so this switches the
        # whole inference path). A failed move falls back to CPU.
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model.to(device)
        except Exception as exc:  # noqa: BLE001 - never lose the CPU path
            log.warning(
                "could not move viXTTS to %s (%s); falling back to cpu", device, exc
            )
            device = "cpu"
            model.to(device)
        self.device = device

        self._config = config
        self._model = model
        # Conditioning runs on the model's device, so a CUDA failure here must
        # fall back to the CPU path instead of propagating a raw exception.
        try:
            latents = _conditioning_latents(model, config, self.ref_path)
        except Exception as exc:  # noqa: BLE001 - CUDA failure -> CPU fallback
            if device != "cuda":
                raise
            log.warning(
                "viXTTS conditioning on cuda failed (%s); retrying on cpu", exc
            )
            device = "cpu"
            model.to(device)
            self.device = device
            latents = _conditioning_latents(model, config, self.ref_path)
        self._latents = latents
        log.info("viXTTS ready on device=%s (reference voice: %s)", device, self.ref_path)

    def _download_model(self) -> tuple[Path, Path]:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise TTSError(
                "huggingface_hub is not installed; install it with "
                "'pip install huggingface_hub'"
            ) from exc

        try:
            config_file = Path(hf_hub_download(self.model_repo, "config.json"))
            for name in MODEL_FILES:
                hf_hub_download(self.model_repo, name)
            speaker_file = Path(hf_hub_download(self.base_repo, SPEAKER_FILE))
        except Exception as exc:  # noqa: BLE001 - download failure
            raise TTSError(
                f"could not download viXTTS files from '{self.model_repo}' / "
                f"'{self.base_repo}': {exc}. Check your network connection."
            ) from exc
        return config_file.parent, speaker_file

    def _resolve_ref_voice(self) -> Path:
        if self.ref_voice:
            path = Path(self.ref_voice)
            if not path.exists():
                raise TTSError(f"reference voice not found: {path}")
            return path

        from huggingface_hub import hf_hub_download

        for name in REF_CANDIDATES:
            try:
                return Path(hf_hub_download(self.model_repo, name))
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                log.debug("reference sample %s unavailable: %s", name, exc)

        log.warning(
            "no shipped viXTTS reference sample could be downloaded; falling "
            "back to a Piper-synthesised reference voice"
        )
        return _piper_reference()

    # -------------------------------------------------------------- synthesis
    def synth(self, text: str, out_wav: str) -> float:
        """Synthesise ``text`` to a 24 kHz mono 16-bit WAV, returning its duration."""
        self._ensure_loaded()

        import numpy as np

        model = self._model
        latents = self._latents
        assert model is not None and latents is not None

        clean = normalize_text(text) if text and text.strip() else ""
        chunks = _split_chunks(clean, MAX_CHARS)
        try:
            waveforms = self._render(model, latents, chunks)
        except Exception as exc:  # noqa: BLE001 - CUDA failure -> CPU fallback
            if self.device != "cuda":
                raise TTSError(f"viXTTS synthesis failed for {out_wav!r}: {exc}") from exc
            log.warning(
                "viXTTS synthesis on cuda failed (%s); retrying on cpu", exc
            )
            model.to("cpu")
            self.device = "cpu"
            try:
                waveforms = self._render(model, latents, chunks)
            except Exception as retry_exc:  # noqa: BLE001 - synthesis failure
                raise TTSError(
                    f"viXTTS synthesis failed for {out_wav!r}: {retry_exc}"
                ) from retry_exc

        audio = (
            np.concatenate(waveforms) if waveforms else np.zeros(1, dtype=np.float32)
        )
        _write_wav_16bit(out_wav, audio, SAMPLE_RATE)
        return float(len(audio)) / SAMPLE_RATE

    @staticmethod
    def _render(model, latents, chunks: list[str]) -> list:
        """Run XTTS inference for each chunk; return the flat waveforms."""
        import numpy as np

        gpt_cond_latent, speaker_embedding = latents
        waveforms: list["np.ndarray"] = []
        for chunk in chunks:
            result = model.inference(
                text=chunk,
                language=LANGUAGE,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                **_GEN_KWARGS,
            )
            waveforms.append(np.asarray(result["wav"], dtype=np.float32).reshape(-1))
        return waveforms


def _conditioning_latents(model, config, ref_path) -> tuple:
    """Compute the speaker conditioning latents for ``ref_path``.

    Split out of :meth:`VixttsTTS._ensure_loaded` so the call can be retried on
    CPU after a CUDA failure without repeating the argument list.
    """
    return model.get_conditioning_latents(
        audio_path=str(ref_path),
        gpt_cond_len=config.gpt_cond_len,
        max_ref_length=config.max_ref_len,
        sound_norm_refs=config.sound_norm_refs,
    )


def _vi_clean(text: str) -> str:
    """Clean Vietnamese input for the viXTTS BPE tokenizer.

    coqui-tts dropped the "vi" branch from ``VoiceBpeTokenizer.preprocess_text``
    (it raises ``NotImplementedError`` for our only supported target language),
    so we route "vi" through ViXTTS's own normalization: NFC unicode, then
    number/symbol normalization from :mod:`vidub.engines.tts.vixtts_norm`, then
    whitespace collapse. The upstream ``thinhlpg/vixtts-demo`` patches the same
    gap with ``vinorm``.
    """
    import unicodedata

    text = unicodedata.normalize("NFC", text or "")
    text = normalize_text(text)
    return " ".join(text.split())


def _patch_vi_tokenizer(model) -> None:
    """Teach the model's tokenizer about "vi" (char limit + preprocessor)."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:  # pragma: no cover - defensive
        return
    try:
        tokenizer.char_limits["vi"] = MAX_CHARS
    except Exception:  # noqa: BLE001 - char_limits is an internal dict
        log.debug("could not register the 'vi' char limit", exc_info=True)

    original = tokenizer.preprocess_text

    def preprocess_text(txt, lang):
        if lang == "vi":
            return _vi_clean(txt)
        return original(txt, lang)

    tokenizer.preprocess_text = preprocess_text


def _split_chunks(text: str, limit: int) -> list[str]:
    """Split ``text`` at sentence boundaries into chunks of at most ``limit`` chars."""
    chunks: list[str] = []
    for sentence in _SENTENCE_RE.split(text or ""):
        sentence = sentence.strip()
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            chunks.append(sentence)
    return [chunk for chunk in chunks if chunk]


def _write_wav_16bit(path: str, audio, rate: int) -> None:
    """Write a mono float waveform as 16-bit PCM with the stdlib ``wave`` module."""
    import numpy as np

    samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm.tobytes())


def _piper_reference() -> Path:
    """Fallback reference voice: a ~12 s Piper-synthesised Vietnamese clip."""
    from vidub.engines.tts.piper_tts import PiperTTS

    sentence = (
        "Xin chào, đây là giọng nói tham chiếu được tạo tự động cho mô hình "
        "chuyển đổi văn bản thành giọng nói tiếng Việt. "
    )
    target = Path(tempfile.gettempdir()) / "vidub_vixtts_ref.wav"
    if not target.exists():
        PiperTTS().synth(sentence * 3, str(target))
    return target


# --------------------------------------------------------------------- compat
def _register_torchaudio_shim() -> None:
    """Register a minimal ``torchaudio`` module backed by soundfile/librosa/torch."""
    import librosa
    import numpy as np
    import soundfile as sf
    import torch

    spec = importlib.util.spec_from_loader("torchaudio", loader=None)

    ta = types.ModuleType("torchaudio")
    ta.__spec__ = spec
    ta.__version__ = "0.0.0+vidub-shim"

    def load(uri, frame_offset=0, num_frames=-1, normalize=True, **kwargs):
        frames = num_frames if num_frames and num_frames > 0 else -1
        data, sr = sf.read(
            str(uri), dtype="float32", always_2d=True,
            start=int(frame_offset), frames=frames,
        )
        return torch.from_numpy(np.ascontiguousarray(data.T)), sr

    def save(uri, src, sample_rate, **kwargs):
        array = src.detach().cpu().numpy()
        if array.ndim == 2 and array.shape[0] <= 2:
            array = array.T
        sf.write(str(uri), array, int(sample_rate))

    def info(uri):
        meta = sf.info(str(uri))
        return types.SimpleNamespace(
            num_frames=meta.frames,
            sample_rate=meta.samplerate,
            channels=meta.channels,
            duration=meta.duration,
        )

    ta.load = load
    ta.save = save
    ta.info = info

    functional = types.ModuleType("torchaudio.functional")
    functional.__spec__ = importlib.util.spec_from_loader(
        "torchaudio.functional", loader=None
    )

    def resample(waveform, orig_freq, new_freq, **kwargs):
        if float(orig_freq) == float(new_freq):
            return waveform
        array = waveform.detach().cpu().numpy()
        out = librosa.resample(
            array, orig_sr=float(orig_freq), target_sr=float(new_freq),
            axis=-1, res_type="soxr_hq",
        )
        return torch.from_numpy(np.ascontiguousarray(out)).to(
            dtype=waveform.dtype, device=waveform.device
        )

    functional.resample = resample
    ta.functional = functional

    transforms = types.ModuleType("torchaudio.transforms")
    transforms.__spec__ = importlib.util.spec_from_loader(
        "torchaudio.transforms", loader=None
    )

    class _Spectrogram(torch.nn.Module):
        """STFT magnitude spectrogram (torchaudio buffer layout)."""

        def __init__(
            self,
            n_fft: int,
            win_length: int,
            hop_length: int,
            power: float,
            normalized: bool,
        ) -> None:
            super().__init__()
            self.n_fft = n_fft
            self.win_length = win_length
            self.hop_length = hop_length
            self.power = power
            self.normalized = normalized
            self.register_buffer("window", torch.hann_window(win_length))

        def forward(self, waveform):
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            spec = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                pad_mode="reflect",
                normalized=self.normalized,
                return_complex=True,
            )
            return spec.abs().pow(self.power)

    class _MelScale(torch.nn.Module):
        """Linear mel filter bank (torchaudio's ``mel_scale.fb`` buffer)."""

        def __init__(
            self,
            n_mels: int,
            sample_rate: int,
            f_min: float,
            f_max: float,
            n_stft: int,
            norm: str | None,
            mel_scale: str,
        ) -> None:
            super().__init__()
            mel_bank = librosa.filters.mel(
                sr=sample_rate,
                n_fft=(n_stft - 1) * 2,
                n_mels=n_mels,
                fmin=f_min,
                fmax=f_max,
                htk=mel_scale == "htk",
                norm=norm,
            )
            # torchaudio stores the bank as (n_freqs, n_mels) and transposes.
            self.register_buffer(
                "fb", torch.tensor(mel_bank.T.copy(), dtype=torch.float32)
            )

        def forward(self, spec):
            return torch.matmul(self.fb.transpose(-1, -2), spec)

    class MelSpectrogram(torch.nn.Module):
        """torchaudio-equivalent mel spectrogram (matching module/buffer names)."""

        def __init__(
            self,
            n_fft: int = 400,
            hop_length: int | None = None,
            win_length: int | None = None,
            power: float = 2.0,
            normalized: bool = False,
            sample_rate: int = 16000,
            f_min: float = 0.0,
            f_max: float | None = None,
            n_mels: int = 128,
            norm: str | None = None,
            mel_scale: str = "htk",
            **kwargs,
        ) -> None:
            super().__init__()
            win_length = win_length or n_fft
            hop_length = hop_length or n_fft // 2
            self.spectrogram = _Spectrogram(
                n_fft, win_length, hop_length, power, normalized
            )
            self.mel_scale = _MelScale(
                n_mels,
                sample_rate,
                f_min,
                f_max if f_max is not None else sample_rate / 2.0,
                n_fft // 2 + 1,
                norm,
                mel_scale,
            )

        def forward(self, waveform):
            return self.mel_scale(self.spectrogram(waveform))

    transforms.MelSpectrogram = MelSpectrogram
    ta.transforms = transforms

    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = functional
    sys.modules["torchaudio.transforms"] = transforms


def _patch_transformers_compat() -> None:
    """Restore transformers symbols removed in v5 that coqui-tts still imports."""
    import torch
    import transformers.pytorch_utils as tu

    if not hasattr(tu, "isin_mps_friendly"):

        def isin_mps_friendly(elements, test_elements):
            if elements.device.type != "mps":
                return torch.isin(elements, test_elements)
            return (elements.unsqueeze(-1) == test_elements).any(-1)

        tu.isin_mps_friendly = isin_mps_friendly


def _import_xtts():
    """Import coqui-tts' XTTS, shimming torchaudio and the torchcodec guard."""
    if importlib.util.find_spec("torchaudio") is None:
        _register_torchaudio_shim()
    _patch_transformers_compat()

    import transformers.utils.import_utils as iu

    patched: dict[str, object] = {}
    for name in ("is_torchaudio_available", "is_torchcodec_available"):
        check = getattr(iu, name, None)
        try:
            available = bool(check()) if check is not None else True
        except Exception:  # noqa: BLE001
            available = True
        if check is not None and not available:
            patched[name] = check
            setattr(iu, name, lambda: True)
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise TTSError(
            "coqui-tts is not installed (ViXTTS is an optional engine); install "
            "it with 'pip install coqui-tts'. "
            f"Original error: {exc}"
        ) from exc
    finally:
        for name, check in patched.items():
            setattr(iu, name, check)
    return XttsConfig, Xtts

"""pyannote speaker-diarization wrapper (optional ``[diarize]`` extra).

The pyannote models are **gated on the Hugging Face Hub**: they need a token and
the user must have accepted the model conditions. This module therefore fails
with an actionable :class:`DiarizeAuthError` for *any* of the three realistic
failure modes (missing token, no access, pyannote not installed) instead of a
raw traceback. The default pipeline never imports this unless ``--diarize``.

Everything is lazy: importing :mod:`vidub.engines.diarize` pulls in no torch or
pyannote code, and the model is only fetched on the first :meth:`diarize`.
"""

from __future__ import annotations

import logging

__all__ = ["DiarizeAuthError", "PyannoteDiarizer", "DIARIZATION_SETUP"]

log = logging.getLogger(__name__)

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
# Turns shorter than this are noise for voice assignment.
MIN_TURN_SECONDS = 0.5

DIARIZATION_SETUP = (
    "speaker diarization needs a Hugging Face token with access to the gated "
    "pyannote models:\n"
    "  1. create a token:  https://huggingface.co/settings/tokens\n"
    "  2. accept the conditions:  "
    "https://hf.co/pyannote/speaker-diarization-3.1 and "
    "https://hf.co/pyannote/segmentation-3.0\n"
    "  3. expose it:  set HF_TOKEN=hf_...  (or run 'huggingface-cli login')\n"
    "  4. install the optional engine:  pip install -e \".[diarize]\""
)


class DiarizeAuthError(RuntimeError):
    """Raised when diarization cannot run (no token/access or missing pyannote)."""


def _env_token() -> str | None:
    """HF token from the environment or the huggingface_hub token store."""
    try:
        from huggingface_hub import get_token
    except Exception:  # noqa: BLE001 - fall back to the bare env vars
        import os

        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        return get_token()
    except Exception:  # noqa: BLE001 - a broken token store means "no token"
        return None


class PyannoteDiarizer:
    """Diarize a 16 kHz mono WAV into ``[{start, end, speaker}]`` turns."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "cpu",
        token: str | None = None,
    ) -> None:
        self.model = model
        # pyannote must run on CPU: torch here is the CPU-only wheel.
        self.device = device
        self.token = token
        self._pipeline = None

    # ------------------------------------------------------------------ guards
    def check_ready(self) -> None:
        """Fail fast with an actionable error when diarization cannot run.

        The token check comes *first* so a missing credential always produces
        the setup message even if the optional package cannot be imported.
        """
        if self.token is None and not _env_token():
            raise DiarizeAuthError(DIARIZATION_SETUP)
        self._import_pipeline_class()

    def _import_pipeline_class(self):  # noqa: ANN202 - optional dependency
        try:
            from pyannote.audio import Pipeline
        except Exception as exc:  # noqa: BLE001 - missing/broken optional extra
            raise DiarizeAuthError(
                "pyannote.audio is not available; install the optional engine "
                'with  pip install -e ".[diarize]".  '
                f"({type(exc).__name__}: {exc})"
            ) from exc
        return Pipeline

    # ----------------------------------------------------------------- loading
    def _load(self):
        if self._pipeline is not None:
            return self._pipeline

        Pipeline = self._import_pipeline_class()
        token = self.token or _env_token()
        if not token:
            raise DiarizeAuthError(DIARIZATION_SETUP)

        try:
            try:
                pipeline = Pipeline.from_pretrained(self.model, token=token)
            except TypeError:
                # older pyannote/hub releases use `use_auth_token`
                pipeline = Pipeline.from_pretrained(self.model, use_auth_token=token)
        except Exception as exc:  # noqa: BLE001 - 401/403/gated/offline
            raise DiarizeAuthError(
                f"could not load the gated diarization model {self.model!r} "
                f"({type(exc).__name__}: {exc}).\n{DIARIZATION_SETUP}"
            ) from exc

        try:
            import torch

            pipeline.to(torch.device(self.device))
        except Exception:  # noqa: BLE001 - CPU is the default anyway
            log.debug("could not move the diarization pipeline to %s", self.device)

        self._pipeline = pipeline
        return pipeline

    # -------------------------------------------------------------- diarizing
    def diarize(self, wav_path: str) -> list[dict[str, float | str]]:
        """Return cleaned diarization turns for ``wav_path``.

        Turns shorter than :data:`MIN_TURN_SECONDS` are dropped. Every failure
        (auth, gated model, offline) is wrapped in :class:`DiarizeAuthError`.
        """
        pipeline = self._load()
        try:
            annotation = pipeline(str(wav_path))
        except Exception as exc:  # noqa: BLE001 - keep the failure actionable
            raise DiarizeAuthError(
                f"diarization failed for {wav_path!r} ({type(exc).__name__}: "
                f"{exc}).\n{DIARIZATION_SETUP}"
            ) from exc

        turns: list[dict[str, float | str]] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start = float(turn.start)
            end = float(turn.end)
            if end - start < MIN_TURN_SECONDS:
                continue
            turns.append({"start": start, "end": end, "speaker": str(speaker)})

        log.info(
            "diarization: %d turn(s), %d speaker(s)",
            len(turns),
            len({str(turn["speaker"]) for turn in turns}),
        )
        return turns

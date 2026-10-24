"""Optional speaker-diarization engines (pyannote 3.1 via the Hugging Face Hub)."""

from vidub.engines.diarize.pyannote_diarizer import (
    DIARIZATION_SETUP,
    DiarizeAuthError,
    PyannoteDiarizer,
)

__all__ = ["DiarizeAuthError", "PyannoteDiarizer", "DIARIZATION_SETUP"]

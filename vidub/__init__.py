"""ViDub — video translation + Vietnamese dubbing CLI (in development)."""

from __future__ import annotations

import os

# CTranslate2 (faster-whisper) and torch (demucs/envit5) bundle clashing OpenMP
# runtimes; loading both in one process can deadlock on Windows. This must be
# set before either library is imported, so it goes above every other import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# CTranslate2 (faster-whisper) resolves cuBLAS/cuDNN through the process DLL
# search path, not the package directory, so the nvidia pip wheels must be wired
# before any engine import. Best-effort: never raises.
from vidub.utils.cuda import setup_cuda_dlls

setup_cuda_dlls()

from vidub.config import JobConfig
from vidub.manifest import JobManifest, Segment

__version__ = "0.11.0"

__all__ = ["__version__", "JobConfig", "JobManifest", "Segment"]

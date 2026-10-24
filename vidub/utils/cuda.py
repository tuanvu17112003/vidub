"""Wire the NVIDIA pip CUDA DLLs so CTranslate2 can load them.

CTranslate2 (the backend of faster-whisper) resolves its CUDA dependencies
(``cublas64_12.dll`` / ``cudnn64_9.dll``) through the *process* DLL search path,
not through the package directory. Installing faster-whisper alone therefore
leaves ViDub on a CPU-only CTranslate2 even on a machine with a working NVIDIA
driver — the exact failure M1 hit (``cublas64_12.dll`` missing at encode time).
The ``nvidia-*`` wheels ship those DLLs under
``site-packages/nvidia/<lib>/bin``, which is not on ``PATH``, so this module
prepends every such directory to ``os.environ["PATH"]`` *and* registers it with
:func:`os.add_dll_directory` before any CTranslate2 import happens.

Everything here is best-effort: a missing wheel, a non-Windows platform or a
failed registration is reported in the returned status string and never raises.
"""

from __future__ import annotations

import logging
import os
import site
import sys
from pathlib import Path

__all__ = ["setup_cuda_dlls", "cuda_available", "gpu_whisper_available"]

log = logging.getLogger(__name__)

# Sub-directories of the nvidia wheels that ship the runtime DLLs, in load
# order (cuDNN sits on top of cuBLAS/cuDART, so cuBLAS comes first).
_NVIDIA_LIBS = ("cublas", "cudnn", "cuda_runtime")

# DLLs the CUDA CTranslate2 build actually needs at encode time (Windows names).
_CUDA_RUNTIME_DLLS = ("cublas64_12.dll", "cudnn64_9.dll")

# Loaded DLL modules and add_dll_directory handles must stay referenced for as
# long as the process lives, otherwise the entries are dropped when GC'd.
_DLL_HANDLES: list[object] = []


def _site_packages_dirs() -> list[Path]:
    """Candidate ``site-packages`` directories for this interpreter, de-duped."""
    candidates: list[Path] = []
    try:
        candidates.extend(Path(path) for path in site.getsitepackages())
    except Exception:  # pragma: no cover - site is unusual on frozen builds
        pass
    candidates.append(Path(sys.prefix) / "Lib" / "site-packages")

    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _nvidia_bin_dirs() -> list[Path]:
    """Existing ``site-packages/nvidia/<lib>/bin`` directories, de-duped."""
    found: list[Path] = []
    seen: set[str] = set()
    for site_packages in _site_packages_dirs():
        for lib in _NVIDIA_LIBS:
            bin_dir = site_packages / "nvidia" / lib / "bin"
            key = str(bin_dir).lower()
            if bin_dir.is_dir() and key not in seen:
                seen.add(key)
                found.append(bin_dir)
    return found


def _wire_dir(bin_dir: Path) -> None:
    """Prepend ``bin_dir`` to PATH and register it as a DLL directory."""
    text = str(bin_dir)

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if text not in path_entries:
        os.environ["PATH"] = text + os.pathsep + os.environ.get("PATH", "")

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        try:
            _DLL_HANDLES.append(add_dll_directory(text))
        except OSError as exc:  # pragma: no cover - unusual, but never fatal
            log.debug("os.add_dll_directory(%s) failed: %s", text, exc)


def setup_cuda_dlls() -> str:
    """Wire every NVIDIA pip DLL directory; return a doctor-friendly status.

    Idempotent and safe to call repeatedly (and before any ctranslate2 import).
    """
    bin_dirs = _nvidia_bin_dirs()
    if not bin_dirs:
        return "no nvidia pip packages found"

    for bin_dir in bin_dirs:
        _wire_dir(bin_dir)
    return "cuda dlls wired at " + os.pathsep.join(str(path) for path in bin_dirs)


def cuda_available() -> tuple[bool, str]:
    """Probe (via CTranslate2) whether a usable CUDA device is visible.

    Returns ``(available, reason)``; the reason carries the DLL error text when
    the probe raises, so ``vidub doctor`` can explain a CPU-only fallback.
    """
    setup_cuda_dlls()
    try:
        import ctranslate2
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        return False, f"ctranslate2 import failed: {exc}"

    try:
        count = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:  # noqa: BLE001 - e.g. a missing CUDA DLL
        return False, f"ctranslate2 cuda probe failed: {exc}"

    if count <= 0:
        return False, "no CUDA device reported by CTranslate2"
    return True, f"{count} CUDA device(s) reported by CTranslate2"


def _missing_runtime_dlls() -> list[str]:
    """CUDA runtime DLLs that cannot be loaded into this process (Windows)."""
    if sys.platform != "win32":
        return []

    import ctypes

    missing: list[str] = []
    for name in _CUDA_RUNTIME_DLLS:
        try:
            _DLL_HANDLES.append(ctypes.WinDLL(name))
        except OSError as exc:
            missing.append(f"{name} ({exc})")
    return missing


def gpu_whisper_available() -> tuple[bool, str]:
    """Probe whether faster-whisper can actually run on the GPU.

    Goes one step beyond :func:`cuda_available`: a driver-visible device is not
    enough, CTranslate2 also needs cuBLAS/cuDNN to load. Loading those DLLs is
    the cheap, offline-safe way to catch the M1 ``cublas64_12.dll`` failure.
    Returns ``(available, reason)`` and never raises.
    """
    ok, reason = cuda_available()
    if not ok:
        return False, reason

    try:
        import faster_whisper  # noqa: F401 - presence check only
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        return False, f"faster-whisper import failed: {exc}"

    missing = _missing_runtime_dlls()
    if missing:
        return False, "missing CUDA runtime: " + ", ".join(missing)
    return True, "CUDA device and runtime DLLs available"

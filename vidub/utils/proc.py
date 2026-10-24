"""Subprocess helpers shared across ViDub stages."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

__all__ = ["find_tool", "run"]

log = logging.getLogger(__name__)


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Run ``cmd``, log it, and return the completed process.

    Raises :class:`RuntimeError` on a non-zero exit code unless ``check=False``
    is passed. Extra keyword arguments are forwarded to
    :func:`subprocess.run` (e.g. ``capture_output=True``, ``cwd=...``).
    """
    check = kwargs.pop("check", True)
    log.info("run: %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, **kwargs)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {' '.join(cmd)}"
        )
    return proc


def find_tool(name: str) -> str | None:
    """Return the full path to ``name`` on ``PATH``, or ``None`` if missing."""
    return shutil.which(name)

"""YouTube (and any yt-dlp-supported) URL download helper.

``vidub run <URL>`` and ``vidub batch <URL>`` resolve a URL to a local media
file before ingest. yt-dlp is an **optional** dependency (Unlicense), so this
module never imports it: it is invoked as ``python -m yt_dlp`` in a subprocess so
a broken/missing install can never wedge the process that imports ViDub.

Downloads land in ``<cache_dir>/downloads/<extractor>-<id>.<ext>`` — a stable
name, so re-running the same URL reuses the existing file (an ``index.json``
maps URL -> filename to make the reuse check network-free). Audio+video are
merged to mp4 (mkv fallback) and video height is capped so URL runs stay light.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlsplit

from vidub.utils.proc import run

__all__ = ["YtDlpError", "is_url", "download", "DOWNLOADS_DIR", "INDEX_NAME"]

log = logging.getLogger(__name__)

DOWNLOADS_DIR = "downloads"
INDEX_NAME = "index.json"
# yt-dlp's default merge container: prefer mp4, fall back to mkv when the
# streams cannot be muxed into mp4.
MERGE_FORMAT = "mp4/mkv"


class YtDlpError(RuntimeError):
    """Raised when URL input cannot be downloaded (missing yt-dlp, etc.)."""


def is_url(text: str | None) -> bool:
    """True when ``text`` is an ``http(s)`` URL (a yt-dlp-supported input)."""
    if not text:
        return False
    candidate = str(text).strip()
    if not candidate:
        return False
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _yt_dlp_available() -> bool:
    """Whether the ``yt_dlp`` package is importable (checked before any network)."""
    try:
        return importlib.util.find_spec("yt_dlp") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def _require_yt_dlp() -> None:
    if not _yt_dlp_available():
        raise YtDlpError(
            "yt-dlp is not installed; URL input needs the optional 'youtube' "
            'extra. Install it with: pip install -e ".[youtube]"'
        )


def _load_index(down_dir: Path) -> dict[str, str]:
    """URL -> filename map written alongside the downloads (or ``{}``)."""
    try:
        data = json.loads((down_dir / INDEX_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _save_index(down_dir: Path, index: dict[str, str]) -> None:
    (down_dir / INDEX_NAME).write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _newest_media(down_dir: Path) -> Path | None:
    """Newest non-index file in ``down_dir`` (a fallback when yt-dlp prints none)."""
    media = [
        path
        for path in down_dir.iterdir()
        if path.is_file()
        and path.name != INDEX_NAME
        and not path.name.endswith((".part", ".ytdl"))
    ]
    return max(media, key=lambda path: path.stat().st_mtime) if media else None


def download(
    url: str,
    cache_dir: str | Path,
    max_height: int = 720,
    progress_cb=None,
) -> Path:
    """Download ``url`` into ``<cache_dir>/downloads`` and return the local path.

    Reuses an existing completed download for the same URL. ``max_height`` caps
    the selected video stream (default 720p); ``progress_cb``, when given, is
    called with the final :class:`~pathlib.Path` once the file is ready.

    Raises :class:`YtDlpError` (a :class:`RuntimeError`) with an actionable
    message when yt-dlp is missing or the download fails.
    """
    _require_yt_dlp()

    url = str(url).strip()
    if not is_url(url):
        raise ValueError(f"not an http(s) URL: {url!r}")

    cache = Path(cache_dir)
    down_dir = cache / DOWNLOADS_DIR
    down_dir.mkdir(parents=True, exist_ok=True)

    index = _load_index(down_dir)
    existing = index.get(url)
    if existing:
        candidate = down_dir / existing
        if candidate.is_file():
            log.info("reusing downloaded media for %s -> %s", url, candidate)
            if progress_cb is not None:
                progress_cb(candidate)
            return candidate

    outtmpl = str(down_dir / "%(extractor)s-%(id)s.%(ext)s")
    # Prefer <=max_height video+audio; fall back to a single best file (direct
    # URLs / generic extractor formats often carry no height metadata).
    fmt = (
        f"bv*[height<={int(max_height)}]+ba/"
        f"b[height<={int(max_height)}]/b"
    )
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-progress",
        "-f",
        fmt,
        "--merge-output-format",
        MERGE_FORMAT,
        "-o",
        outtmpl,
        "--print",
        "after_move:filepath",
        url,
    ]
    log.info("downloading %s", url)
    proc = run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = " | ".join(tail[-5:]) if tail else "no output"
        raise YtDlpError(f"yt-dlp failed for {url}: {detail}")

    printed = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    final = Path(printed[-1]) if printed else None
    if final is None or not final.is_file():
        final = _newest_media(down_dir)
    if final is None or not final.is_file():
        raise YtDlpError(f"yt-dlp reported success but produced no file for {url}")

    index[url] = final.name
    _save_index(down_dir, index)
    log.info("downloaded %s -> %s", url, final)
    if progress_cb is not None:
        progress_cb(final)
    return final

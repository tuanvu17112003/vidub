"""Resolve versioned spoken/subtitle text while keeping legacy ``vi`` rows readable."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

__all__ = ["selected_text", "selection_signature"]

_SELECTIONS = frozenset({"spoken_vi", "subtitle_vi"})


def selected_text(row: dict, field: str) -> str:
    """Return an explicit text selection, falling back to the legacy ``vi``."""
    if field not in _SELECTIONS:
        raise ValueError("text field must be 'spoken_vi' or 'subtitle_vi'")
    value = row.get(field)
    if isinstance(value, str) and value.strip():
        return value
    legacy = row.get("vi")
    return str(legacy) if legacy is not None else ""


def selection_signature(rows: Sequence[dict], field: str) -> str:
    """Hash each row's identity and effective selection for cache invalidation."""
    payload = json.dumps(
        [
            {
                "id": row.get("indices", row.get("index", row.get("idx", position))),
                "text": selected_text(row, field),
            }
            for position, row in enumerate(rows)
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

"""Shared artifact names and thresholds for the translate stage."""

from __future__ import annotations

SEGMENTS_NAME = "segments.json"
SCRIPT_NAME = "script.json"
# Glossary counters sidecar for report.json ("glossary": {...}); script.json rows
# stay unchanged, so the loaded/applied counts are persisted here instead.
GLOSSARY_NAME = "glossary.json"
# Level-2 translation QC artifact (QUALITY-ROADMAP.md §3.3, M4-T3).
TRANSLATION_QC_NAME = "translation_qc.json"
# SA-1-T2 unit map sidecar: one record per sentence-level unit with the unit's
# authoritative locked span and its member row indices (only on the unit path).
UNITS_NAME = "units.json"
LENGTH_RATIO = 2.2
# A source segment this Vietnamese-looking is already Vietnamese: asking envit5
# to translate it just echoes the input (and can corrupt it), so copy it through.
PASSTHROUGH_RATIO = 0.5
# Below this many source words an output without diacritics is not meaningful.
LOW_CONFIDENCE_MIN_WORDS = 3
_SENTENCE_END = (".", "!", "?")

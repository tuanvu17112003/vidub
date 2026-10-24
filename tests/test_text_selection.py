import hashlib

import pytest

from vidub.utils.text_selection import selected_text, selection_signature


def test_selected_text_uses_explicit_variant_and_falls_back_to_vi():
    row = {"vi": "Legacy text.", "spoken_vi": "Spoken text.", "subtitle_vi": "Caption text."}

    assert selected_text(row, "spoken_vi") == "Spoken text."
    assert selected_text(row, "subtitle_vi") == "Caption text."
    assert selected_text({"vi": "Legacy text."}, "spoken_vi") == "Legacy text."
    assert selected_text({"vi": "Legacy text.", "subtitle_vi": "  "}, "subtitle_vi") == "Legacy text."


def test_selection_signature_tracks_effective_text_and_order():
    rows = [
        {"vi": "First.", "subtitle_vi": "Caption first."},
        {"vi": "Second.", "subtitle_vi": "Caption second."},
    ]

    signature = selection_signature(rows, "subtitle_vi")

    assert signature == selection_signature(rows, "subtitle_vi")
    assert signature != selection_signature(rows, "spoken_vi")
    assert signature != selection_signature(list(reversed(rows)), "subtitle_vi")
    assert signature != selection_signature(
        [{**rows[0], "subtitle_vi": "Edited caption."}, rows[1]], "subtitle_vi"
    )
    assert len(signature) == hashlib.sha256().digest_size * 2
    assert selection_signature(
        [{"indices": [0], "vi": "Same."}], "spoken_vi"
    ) != selection_signature(
        [{"indices": [1], "vi": "Same."}], "spoken_vi"
    )


def test_selected_text_rejects_unknown_selection_field():
    with pytest.raises(ValueError, match="spoken_vi.*subtitle_vi"):
        selected_text({"vi": "Legacy text."}, "subtitle")

"""ViDub pipeline stages (S0 ingest … S6 export).

Implemented:

* S0 ``ingest``     — video → 16 kHz mono ``audio.wav`` + ``job.json``.
* S1 ``transcribe`` — audio → ``segments.json`` + draft source ``.srt``.
* S2 ``translate``  — segments → Vietnamese dubbing script ``script.json``.
* S3 ``speak``      — script → per-segment Vietnamese WAVs + ``speech.json``.
* S4 ``timing``     — fit each segment into its slot → ``timed.json``.
* S5 ``mix``        — Demucs bed + placed segments + loudnorm → ``dub_track.wav``.
* S6 ``export``     — mux dubbed audio + Vietnamese subtitles → ``dubbed.vi.mp4``.

Each stage is runnable standalone via ``python -m vidub.stages.<stage>``; the
whole sequence is orchestrated by :mod:`vidub.pipeline` (``vidub run``).
"""

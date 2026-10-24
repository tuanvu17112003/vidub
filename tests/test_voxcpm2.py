from __future__ import annotations

import io
import json

import pytest

from vidub.config import JobConfig, TTS_ENGINES, check_tts_engine_allowed, validate_langs
from vidub.engines.tts.base import TTSError
from vidub.engines.tts.voxcpm2 import VoxCPM2TTS
from vidub.manifest import JobManifest
from vidub.stages.speak import _VoiceResolver, build_tts


class _FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self.lines = iter(lines)

    def readline(self) -> str:
        return next(self.lines, "")


class _FakeProcess:
    def __init__(self, lines: list[str]) -> None:
        self.stdout = _FakeStdout(lines)
        self.stdin = io.StringIO()
        self.waited = False
        self.terminated = False

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def terminate(self):
        self.terminated = True


def test_voxcpm2_is_registered_as_commercial_vietnamese_tts() -> None:
    assert TTS_ENGINES["voxcpm2"] == {"license": "Apache-2.0", "nc": False}
    check_tts_engine_allowed("voxcpm2", allow_nc=False)
    validate_langs("en", "vi", "voxcpm2")


def test_voxcpm2_reuses_worker_and_returns_each_duration(monkeypatch, tmp_path) -> None:
    process = _FakeProcess(
        [
            '{"ready": true, "sample_rate": 48000}\n',
            '{"duration": 1.25}\n',
            '{"duration": 1.5}\n',
        ]
    )
    launches = []

    def fake_popen(args, **kwargs):
        launches.append((args, kwargs))
        return process

    monkeypatch.setattr("vidub.engines.tts.voxcpm2.subprocess.Popen", fake_popen)
    engine = VoxCPM2TTS(python_path="worker-python", model_cache=tmp_path)
    (tmp_path / "one.wav").write_bytes(b"wav")
    (tmp_path / "two.wav").write_bytes(b"wav")

    assert engine.synth("Xin chào.", str(tmp_path / "one.wav")) == 1.25
    assert engine.synth("Tạm biệt.", str(tmp_path / "two.wav")) == 1.5
    engine.close()

    assert len(launches) == 1
    assert launches[0][1]["env"]["PYTHONIOENCODING"] == "utf-8"
    requests = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
    assert requests == [
        {
            "text": "Xin chào.",
            "output_path": str(tmp_path / "one.wav"),
            "seed": 4200,
            "voice_profile": "vi_female_warm",
        },
        {
            "text": "Tạm biệt.",
            "output_path": str(tmp_path / "two.wav"),
            "seed": 4200,
            "voice_profile": "vi_female_warm",
        },
        {"command": "close"},
    ]
    assert process.waited


def test_voxcpm2_reuses_selected_reference_voice_for_every_segment(
    monkeypatch, tmp_path
) -> None:
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"reference audio")
    process = _FakeProcess(
        [
            '{"ready": true, "sample_rate": 48000}\n',
            '{"duration": 1.0}\n',
            '{"duration": 1.0}\n',
        ]
    )
    monkeypatch.setattr(
        "vidub.engines.tts.voxcpm2.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    engine = VoxCPM2TTS(
        python_path="worker-python", model_cache=tmp_path, ref_voice=reference
    )
    first = tmp_path / "one.wav"
    second = tmp_path / "two.wav"
    first.write_bytes(b"wav")
    second.write_bytes(b"wav")

    engine.synth("Xin chào.", str(first))
    engine.synth("Tạm biệt.", str(second))

    requests = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
    assert [request["reference_wav_path"] for request in requests[:2]] == [
        str(reference),
        str(reference),
    ]
    assert [request["seed"] for request in requests[:2]] == [4200, 4200]


def test_voxcpm2_worker_conditions_on_selected_voice_sample() -> None:
    from vidub.engines.tts.voxcpm2 import _generate_audio

    class FakeModel:
        calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [0.0]

    model = FakeModel()
    _generate_audio(model, "Xin chào.", "voice.wav")

    assert model.calls == [
        {
            "text": "Xin chào.",
            "cfg_value": 2.0,
            "inference_timesteps": 10,
            "reference_wav_path": "voice.wav",
        }
    ]


def test_voxcpm2_worker_uses_selected_voice_design_profile() -> None:
    from vidub.config import VOXCPM2_VOICE_PROFILES
    from vidub.engines.tts.voxcpm2 import _generate_audio

    class FakeModel:
        calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [0.0]

    model = FakeModel()
    _generate_audio(model, "Xin chào.", None, "vi_male_calm")

    assert model.calls[0]["text"] == (
        f"{VOXCPM2_VOICE_PROFILES['vi_male_calm']['style']} Xin chào."
    )


def test_voxcpm2_sends_selected_voice_profile_to_worker(monkeypatch, tmp_path) -> None:
    process = _FakeProcess(
        ['{"ready": true, "sample_rate": 48000}\n', '{"duration": 1.0}\n']
    )
    monkeypatch.setattr(
        "vidub.engines.tts.voxcpm2.subprocess.Popen", lambda *_args, **_kwargs: process
    )
    engine = VoxCPM2TTS(
        python_path="worker-python", model_cache=tmp_path, voice_profile="vi_male_calm"
    )
    output = tmp_path / "one.wav"
    output.write_bytes(b"wav")

    engine.synth("Xin chào.", str(output))

    request = json.loads(process.stdin.getvalue().splitlines()[0])
    assert request["voice_profile"] == "vi_male_calm"


def test_voxcpm2_surfaces_worker_startup_error(monkeypatch, tmp_path) -> None:
    process = _FakeProcess(['{"error": "local model files are missing"}\n'])
    monkeypatch.setattr(
        "vidub.engines.tts.voxcpm2.subprocess.Popen", lambda *_args, **_kwargs: process
    )
    engine = VoxCPM2TTS(python_path="worker-python", model_cache=tmp_path)

    with pytest.raises(TTSError, match="local model files are missing"):
        engine.synth("Xin chào.", str(tmp_path / "one.wav"))


def test_voxcpm2_builds_through_the_standard_engine_factory(monkeypatch) -> None:
    created = []

    class FakeVoxCPM2:
        def __init__(self, ref_voice=None, voice_profile=None):
            created.append((self, ref_voice, voice_profile))

    monkeypatch.setattr(
        "vidub.engines.tts.voxcpm2.VoxCPM2TTS", FakeVoxCPM2
    )
    manifest = JobManifest(
        cache_dir="cache", video_hash="hash", tts_engine="voxcpm2"
    )
    cfg = JobConfig(
        tts_engine="voxcpm2", ref_voice="voice.wav", voxcpm2_voice="vi_male_calm"
    )

    assert isinstance(build_tts(manifest, cfg), FakeVoxCPM2)
    assert created[0][1] == "voice.wav"
    assert created[0][2] == "vi_male_calm"


def test_voxcpm2_resolver_reuses_one_engine_across_speakers(monkeypatch) -> None:
    created = []

    class FakeVoxCPM2:
        def __init__(self, ref_voice=None, voice_profile=None):
            self.closed = False
            self.ref_voice = ref_voice
            self.voice_profile = voice_profile
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "vidub.engines.tts.voxcpm2.VoxCPM2TTS", FakeVoxCPM2
    )
    manifest = JobManifest(
        cache_dir="cache", video_hash="hash", tts_engine="voxcpm2"
    )
    resolver = _VoiceResolver(
        manifest,
        JobConfig(
            tts_engine="voxcpm2", ref_voice="voice.wav", voxcpm2_voice="vi_male_calm"
        ),
    )

    assert resolver.for_speaker("spk0") is resolver.for_speaker("spk1")
    resolver.close()

    assert len(created) == 1
    assert created[0].ref_voice == "voice.wav"
    assert created[0].voice_profile == "vi_male_calm"
    assert created[0].closed


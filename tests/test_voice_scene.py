import asyncio
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from voice_scene import VoiceSceneError, _chunks, render_voice_scene, validate_segments

A = "a" * 32
B = "b" * 32


def tone(freq):
    return subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        f"sine=frequency={freq}:duration=0.12", "-c:a", "libopus", "-f", "ogg", "pipe:1",
    ])


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg tools required")
def test_composes_tones_and_pause_to_single_opus():
    samples = {A: tone(440), B: tone(880)}
    calls = []

    async def synth(text, reference_id):
        calls.append((text, reference_id))
        return samples[reference_id]

    result = asyncio.run(render_voice_scene([
        {"text": "first voice", "voice_id": A, "pause_ms": 250},
        {"text": "second voice", "voice_id": B},
    ], synth))
    assert result.startswith(b"OggS")
    assert len(calls) == 2
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "scene.ogg"
        audio.write_bytes(result)
        probe = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels",
            "-show_entries", "format=duration", "-of", "json", str(audio),
        ])
    info = json.loads(probe)
    assert info["streams"][0]["codec_name"] == "opus"
    assert info["streams"][0]["sample_rate"] == "48000"
    assert info["streams"][0]["channels"] == 1
    assert 0.48 <= float(info["format"]["duration"]) <= 0.65


@pytest.mark.parametrize("segments", [
    [],
    [{"text": "x", "voice_id": "bad"}],
    [{"text": "   ", "voice_id": A}],
    [{"text": "x", "voice_id": A, "pause_ms": 3001}],
    [{"text": "x" * 100001, "voice_id": A}],
])
def test_validation_rejects_bad_scene(segments):
    async def synth(*args, **kwargs):
        raise AssertionError("must validate before synthesis")
    with pytest.raises(VoiceSceneError):
        asyncio.run(render_voice_scene(segments, synth))


def test_chunks_preserve_every_character_and_stay_near_target():
    text = ("alpha beta\n" * 1000) + "END"
    parts = _chunks(text)
    assert "".join(parts) == text
    assert all(len(part) <= 1501 for part in parts)
    assert all(part for part in parts)


def test_public_validation_preserves_role_and_normalizes_id():
    validated = validate_segments([{"text": "hello", "voice_id": "A" * 32, "role": "narrator"}])
    assert validated[0]["role"] == "narrator"
    assert validated[0]["voice_id"] == "a" * 32


def test_long_segment_synthesized_without_loss():
    text = "0123456789 " * 350
    received = []
    blob = tone(440)
    async def synth(chunk, reference_id):
        received.append(chunk)
        return blob
    result = asyncio.run(render_voice_scene([{"text": text, "voice_id": A}], synth))
    assert result.startswith(b"OggS")
    assert "".join(received) == text
    assert len(received) > 1


def test_synthesis_failure_aborts_whole_render():
    async def synth(text, reference_id):
        if reference_id == B:
            raise RuntimeError("private provider detail")
        return tone(440)
    with pytest.raises(VoiceSceneError, match="synthesis failed") as error:
        asyncio.run(render_voice_scene([
            {"text": "ok", "voice_id": A}, {"text": "failure", "voice_id": B}
        ], synth))
    assert "private provider detail" not in str(error.value)

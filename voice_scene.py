"""Standalone bounded compositor for multi-voice OGG Opus scenes."""
from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import math
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Sequence

VOICE_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")
CHUNK_CHARS = 1500
MAX_SEGMENTS = 100
MAX_TEXT_CHARS = 100_000
MAX_FINAL_BYTES = 45 * 1024 * 1024
MAX_RAW_CHUNK_BYTES = MAX_FINAL_BYTES
MAX_DURATION_SECONDS = 7200
SUBPROCESS_TIMEOUT_SECONDS = 180
SAMPLE_RATE = 48000
CHANNELS = 1


class VoiceSceneError(ValueError):
    """Invalid scene or failed/unsafe audio composition."""


def validate_segments(segments: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if not isinstance(segments, (list, tuple)) or not 1 <= len(segments) <= MAX_SEGMENTS:
        raise VoiceSceneError("segments must contain 1..100 entries")
    clean: list[dict[str, object]] = []
    total = 0
    for i, item in enumerate(segments):
        if not isinstance(item, Mapping):
            raise VoiceSceneError(f"segment {i} must be an object")
        text, voice_id = item.get("text"), item.get("voice_id")
        if not isinstance(text, str) or not text.strip():
            raise VoiceSceneError(f"segment {i} text must be non-empty")
        if not isinstance(voice_id, str) or not VOICE_ID_RE.fullmatch(voice_id):
            raise VoiceSceneError(f"segment {i} voice_id must be 32 hexadecimal characters")
        total += len(text)
        if total > MAX_TEXT_CHARS:
            raise VoiceSceneError("total text exceeds 100000 characters")
        pause = item.get("pause_ms", 0)
        if isinstance(pause, bool) or not isinstance(pause, int) or not 0 <= pause <= 3000:
            raise VoiceSceneError(f"segment {i} pause_ms must be an integer from 0 to 3000")
        entry = {"text": text, "voice_id": voice_id.lower(), "pause_ms": pause}
        if "role" in item:
            if not isinstance(item["role"], str):
                raise VoiceSceneError(f"segment {i} role must be a string")
            entry["role"] = item["role"]
        clean.append(entry)
    return clean


_validate = validate_segments


def _chunks(text: str) -> list[str]:
    """Split near 1500 chars at whitespace while preserving every source char."""
    result: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + CHUNK_CHARS, n)
        if end < n:
            # Prefer a whitespace boundary near the target; retain the delimiter.
            boundary = max(text.rfind(" ", start, end + 1), text.rfind("\n", start, end + 1), text.rfind("\t", start, end + 1))
            if boundary > start + CHUNK_CHARS // 2:
                end = boundary + 1
        result.append(text[start:end])
        start = end
    return result


async def _run(*args: str, timeout: int = SUBPROCESS_TIMEOUT_SECONDS) -> bytes:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        if 'proc' in locals() and proc.returncode is None:
            proc.kill()
            await proc.wait()
        raise VoiceSceneError("audio tool timed out") from exc
    except OSError as exc:
        raise VoiceSceneError("audio tool could not be started") from exc
    if proc.returncode:
        raise VoiceSceneError("audio tool failed")
    return stdout


async def render_voice_scene(
    segments: Sequence[Mapping[str, object]],
    synthesize: Callable[..., Awaitable[bytes]],
) -> bytes:
    """Synthesize each voice chunk and return one bounded OGG Opus byte stream.

    synthesize is awaited as ``synthesize(text, reference_id=voice_id)`` and must
    return OGG audio bytes. Any synthesis/validation/encoding failure aborts the
    whole operation; callers must not send partial results.
    """
    scene = validate_segments(segments)
    if not callable(synthesize):
        raise VoiceSceneError("synthesize must be callable")
    with tempfile.TemporaryDirectory(prefix="voice-scene-") as tmp:
        root = Path(tmp)
        normalized: list[Path] = []
        segment_inputs: list[list[Path]] = []
        cumulative_duration = 0.0
        for si, segment in enumerate(scene):
            inputs: list[Path] = []
            for ci, text in enumerate(_chunks(segment["text"])):
                try:
                    blob = await synthesize(text, reference_id=segment["voice_id"])
                except Exception as exc:
                    raise VoiceSceneError(f"synthesis failed for segment {si}, chunk {ci}") from exc
                if not isinstance(blob, (bytes, bytearray)) or not blob:
                    raise VoiceSceneError(f"synthesis returned invalid audio for segment {si}, chunk {ci}")
                if len(blob) > MAX_RAW_CHUNK_BYTES:
                    raise VoiceSceneError("synthesis chunk exceeds 45 MB")
                src = root / f"src-{si:03d}-{ci:04d}.ogg"
                src.write_bytes(blob)
                dst = root / f"part-{si:03d}-{ci:04d}.ogg"
                await _run("ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
                           "-map", "0:a:0", "-vn", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
                           "-c:a", "libopus", "-b:a", "64k", "-f", "ogg", str(dst))
                if not dst.is_file() or dst.stat().st_size > MAX_FINAL_BYTES:
                    raise VoiceSceneError("normalized audio chunk exceeds 45 MB")
                probe = await _run("ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "json", str(dst))
                try:
                    chunk_duration = float(json.loads(probe).get("format", {}).get("duration", "nan"))
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise VoiceSceneError("could not verify chunk duration") from exc
                # Bound aggregate duration before concat; pauses are bounded by input validation.
                if not math.isfinite(chunk_duration) or not 0 <= chunk_duration <= MAX_DURATION_SECONDS:
                    raise VoiceSceneError("audio chunk exceeds 7200 seconds or has invalid duration")
                cumulative_duration += chunk_duration
                if cumulative_duration + sum(int(s["pause_ms"]) for s in scene[:-1]) / 1000 > MAX_DURATION_SECONDS:
                    raise VoiceSceneError("combined audio exceeds 7200 seconds")
                inputs.append(dst)
                normalized.append(dst)
            segment_inputs.append(inputs)

        # Assemble via a disk-backed concat demuxer manifest, avoiding a large argv.
        ordered: list[Path] = []
        for si, inputs in enumerate(segment_inputs):
            ordered.extend(inputs)
            pause = int(scene[si]["pause_ms"]) if si < len(scene) - 1 else 0
            if pause:
                silence = root / f"pause-{si:03d}.ogg"
                await _run("ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                           "-i", f"anullsrc=r={SAMPLE_RATE}:cl={CHANNELS and 'mono'}",
                           "-t", f"{pause / 1000:.3f}", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
                           "-c:a", "libopus", "-b:a", "32k", "-f", "ogg", str(silence))
                ordered.append(silence)
        manifest = root / "concat.txt"
        manifest.write_text("".join("file '" + p.name + "'\n" for p in ordered), encoding="utf-8")
        output = root / "scene.ogg"
        await _run("ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "1",
                   "-i", str(manifest), "-map", "0:a:0", "-vn", "-c:a", "libopus", "-b:a", "64k",
                   "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE), "-f", "ogg", str(output))
        if not output.is_file() or output.stat().st_size > MAX_FINAL_BYTES:
            raise VoiceSceneError("final audio exceeds 45 MB or is missing")
        try:
            probe = await _run("ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "json", str(output))
            duration = float(json.loads(probe).get("format", {}).get("duration", "nan"))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise VoiceSceneError("could not verify final duration") from exc
        if not 0 <= duration <= MAX_DURATION_SECONDS:
            raise VoiceSceneError("final audio exceeds 7200 seconds or has invalid duration")
        return output.read_bytes()


__all__ = ["render_voice_scene", "validate_segments", "VoiceSceneError"]

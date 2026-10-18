"""Cliproxy native image generation adapter (OpenAI chat-completions image modality)."""
from __future__ import annotations
import base64
import re
import requests

MODEL = "gemini-3.1-flash-image"


def generate(base_url: str, api_key: str, prompt: str, input_images_b64: list[str] | None = None,
             timeout: tuple[int, int] = (20, 180)) -> tuple[bytes, str]:
    """Generate an image via Cliproxy's native /chat/completions endpoint."""
    if not base_url or not api_key:
        raise RuntimeError("Cliproxy image fallback is not configured")
    content = [{"type": "text", "text": prompt}]
    for value in input_images_b64 or []:
        raw = value.split(",", 1)[-1] if value.startswith("data:") else value
        # Cliproxy vision input accepts OpenAI image_url data URLs.
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + raw}})
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "modalities": ["image", "text"]}
    response = requests.post(base_url.rstrip("/") + "/chat/completions", json=body,
                             headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                             timeout=timeout)
    if not response.ok:
        response.raise_for_status()
    result = response.json()
    message = (result.get("choices") or [{}])[0].get("message") or {}
    for item in message.get("images") or []:
        url = ((item.get("image_url") or {}).get("url") or item.get("url") or "")
        match = re.fullmatch(r"data:([^;,]+);base64,(.+)", url, flags=re.S)
        if match:
            return base64.b64decode(match.group(2), validate=True), match.group(1)
        if url.startswith(("https://", "http://")):
            image = requests.get(url, timeout=timeout)
            image.raise_for_status()
            return image.content, image.headers.get("Content-Type", "image/png").split(";")[0]
    # Compatibility with OpenAI-style text content carrying a data URL.
    text = message.get("content")
    if isinstance(text, str):
        match = re.search(r"data:([^;,\s]+);base64,([A-Za-z0-9+/=]+)", text)
        if match:
            return base64.b64decode(match.group(2), validate=True), match.group(1)
    raise RuntimeError("Cliproxy returned no usable image")

__all__ = ["MODEL", "generate"]

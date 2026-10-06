"""Honest image payload validation and provider capability metadata."""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

@dataclass(frozen=True)
class ProviderProfile:
    name: str
    native_sizes: tuple[tuple[int, int], ...] = ()
    aspect_ratios: tuple[str, ...] = ()
    supports_refs: bool = False
    supports_masks: bool = False
    max_bytes: int = 20_000_000
    max_images: int = 1
    timeout_seconds: float = 120.0

@dataclass(frozen=True)
class ValidatedImage:
    data: bytes
    mime_type: str
    pixel_size: tuple[int, int]
    format: str
    byte_size: int

def validate_image(data: bytes, *, max_bytes: int = 20_000_000, max_pixels: int = 40_000_000, expected_mime: Optional[str] = None) -> ValidatedImage:
    """Decode and verify actual image bytes; sizes are measured, never inferred from requested tier."""
    if not isinstance(data, (bytes, bytearray)) or not data: raise ValueError("empty or non-byte image response")
    raw = bytes(data)
    if len(raw) > max_bytes: raise ValueError(f"image exceeds byte limit ({len(raw)} > {max_bytes})")
    from PIL import Image, ImageOps
    try:
        with Image.open(BytesIO(raw)) as im:
            im.verify()
        with Image.open(BytesIO(raw)) as im:
            fmt = (im.format or "").upper()
            if im.width * im.height > max_pixels:
                raise ValueError('image dimensions exceed pixel limit')
            oriented = ImageOps.exif_transpose(im)
            oriented.load()
            size = oriented.size
            if size[0] < 1 or size[1] < 1 or size[0] * size[1] > max_pixels: raise ValueError("image dimensions exceed pixel limit")
            if im.getexif().get(274, 1) != 1:
                normalized = BytesIO()
                oriented.save(normalized, format='PNG')
                raw = normalized.getvalue()
                fmt = 'PNG'
                if len(raw) > max_bytes:
                    raise ValueError('normalized image exceeds byte limit')
    except ValueError: raise
    except Exception as exc: raise ValueError("provider returned invalid or unsupported image data") from exc
    mimes = {"PNG":"image/png", "JPEG":"image/jpeg", "WEBP":"image/webp", "GIF":"image/gif", "TIFF":"image/tiff"}
    mime = mimes.get(fmt)
    if not mime: raise ValueError(f"unsupported decoded image format: {fmt or 'unknown'}")
    if expected_mime and expected_mime.split(";")[0].strip().lower() != mime: raise ValueError(f"MIME mismatch: header says {expected_mime}, bytes are {mime}")
    return ValidatedImage(raw, mime, size, fmt, len(raw))

def classify_provider_error(status_code: Optional[int], code: str = "", message: str = ""):
    """Map errors conservatively; only explicit moderation indicators are moderation."""
    from gen_contracts import ProviderError
    text = f"{code} {message}".lower()
    if any(x in text for x in ("content_policy", "safety_block", "moderation")): kind = "moderation"
    elif status_code in (401, 403): kind = "authentication"
    elif status_code == 429: kind = "rate_limit"
    elif status_code == 408 or status_code == 504: kind = "timeout"
    elif status_code == 402 or "quota" in text: kind = "quota"
    elif status_code == 400 or status_code == 422: kind = "invalid_request"
    elif status_code is not None and status_code >= 500: kind = "transient"
    else: kind = "transient"
    return ProviderError(kind, "Image provider request failed", status_code, kind in {"rate_limit", "timeout", "transient"})

GATEWAY_PROFILE = ProviderProfile('chatgpt2api', ((1024,1024),(1536,1024),(1024,1536)), ('1:1','3:2','2:3'), True, False, 4_500_000, 10, 600)

def gateway_dimensions(aspect):
    # Native API capability is explicit; requested ratios are not relabeled.
    if aspect in ('16:9','4:3','3:2','horizontal'): return (1536,1024)
    if aspect in ('9:16','3:4','2:3','vertical'): return (1024,1536)
    return (1024,1024)

def capability_notice(requested_aspect, requested_tier, actual_size):
    notices=[]
    if requested_aspect and ':' in requested_aspect:
        a,b=map(int,requested_aspect.split(':'))
        if actual_size[0]*b != actual_size[1]*a:
            notices.append(f'Формат {requested_aspect} недоступен нативно; фактически {actual_size[0]}×{actual_size[1]}, без обрезки')
    if requested_tier in ('2K','4K') and max(actual_size)<int(requested_tier[0])*1000:
        notices.append(f'{requested_tier} не получен нативно; апскейл не применялся')
    return notices

__all__ = ["ProviderProfile", "ValidatedImage", "validate_image", "classify_provider_error", "GATEWAY_PROFILE", "gateway_dimensions", "capability_notice"]

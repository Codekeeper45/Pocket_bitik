"""Immutable reference registry and safe Telegram link parsing for image generation."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
from typing import Iterable, Optional
from urllib.parse import parse_qs, unquote, urlsplit


@dataclass(frozen=True)
class Reference:
    ref_id: str
    chat_id: int
    message_id: int
    digest: str
    data: bytes
    subject: Optional[str] = None
    role: str = "reference"
    caption: str = ""
    api_index: Optional[int] = None
    priority: int = 0
    pixel_size: Optional[tuple[int, int]] = None
    source: str = "user"


def make_reference(chat_id: int, message_id: int, data: bytes, **metadata) -> Reference:
    if not isinstance(chat_id, int) or isinstance(chat_id, bool) or not isinstance(message_id, int) or isinstance(message_id, bool):
        raise ValueError("chat_id/message_id must be integers")
    raw = bytes(data)
    digest = sha256(raw).hexdigest()
    rid = sha256(f"{chat_id}:{message_id}:{digest}".encode()).hexdigest()[:24]
    size = None
    try:
        from PIL import Image, ImageOps
        with Image.open(BytesIO(raw)) as im:
            im = ImageOps.exif_transpose(im)
            size = im.size
    except Exception:
        pass
    return Reference(rid, chat_id, message_id, digest, raw, pixel_size=size, **metadata)


class ReferenceRegistry:
    """Insertion-ordered, immutable snapshots; identity is source tuple OR content hash."""
    def __init__(self, refs: Iterable[Reference] = ()):
        self._refs: tuple[Reference, ...] = ()
        for ref in refs:
            self.add(ref)

    @property
    def refs(self):
        return self._refs

    def add(self, ref: Reference) -> Reference:
        for old in self._refs:
            if (old.chat_id, old.message_id) == (ref.chat_id, ref.message_id) or old.digest == ref.digest:
                return old
        self._refs += (ref,)
        return ref

    def selected(self, ref_ids: Iterable[str], max_refs: int = 16) -> tuple[Reference, ...]:
        if max_refs < 0:
            raise ValueError("max_refs must be nonnegative")
        ids = tuple(dict.fromkeys(ref_ids))
        missing = [rid for rid in ids if not any(r.ref_id == rid for r in self._refs)]
        if missing:
            raise KeyError(f"selected references unavailable: {', '.join(missing)}")
        chosen = sorted((r for r in self._refs if r.ref_id in ids), key=lambda r: (-r.priority, ids.index(r.ref_id)))
        if len(chosen) > max_refs:
            raise ValueError(f"{len(chosen)} required references exceed limit {max_refs}")
        return tuple(replace(r, api_index=i) for i, r in enumerate(chosen, 1))


def resize_image(data: bytes, max_side: int = 1536, max_bytes: int = 2_000_000) -> bytes:
    """EXIF-normalize and bounded JPEG encode; reject non-images and invalid budgets."""
    if max_side < 1 or max_bytes < 128:
        raise ValueError("invalid image budget")
    from PIL import Image, ImageOps
    with Image.open(BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    quality = 90
    while True:
        out = BytesIO()
        image.save(out, "JPEG", quality=quality, optimize=True)
        value = out.getvalue()
        if len(value) <= max_bytes:
            return value
        if quality > 35:
            quality -= 10
        elif max(image.size) > 128:
            image.thumbnail((max(128, int(image.width * .8)), max(128, int(image.height * .8))), Image.Resampling.LANCZOS)
            quality = 75
        else:
            raise ValueError("cannot fit image within byte budget")


@dataclass(frozen=True)
class TelegramLink:
    chat: str
    message_id: int
    topic_id: Optional[int] = None
    caption: str = ""


def parse_telegram_link(url: str, caption: str = "") -> TelegramLink:
    p = urlsplit(url.strip())
    if p.scheme not in ("https", "http") or p.hostname not in ("t.me", "www.t.me"):
        raise ValueError("unsupported Telegram link")
    parts = [unquote(x) for x in p.path.strip("/").split("/") if x]
    if parts and parts[0] == "s":
        parts.pop(0)
    if len(parts) < 2:
        raise ValueError("Telegram link must identify a message")
    chat = parts[0]
    topic = None
    if chat == 'c':
        if len(parts) not in (3, 4) or not parts[1].isdigit():
            raise ValueError('invalid private Telegram link')
        chat = '-100' + parts[1]
        if len(parts) == 4:
            topic = int(parts[2])
    elif len(parts) not in (2, 3):
        raise ValueError('invalid public Telegram link')
    if chat.startswith("+") or chat == "joinchat":
        raise ValueError("invite links do not identify a message")
    try:
        mid = int(parts[-1])
    except ValueError as e:
        raise ValueError("invalid Telegram message id") from e
    if mid <= 0:
        raise ValueError("invalid Telegram message id")
    if not chat.startswith('-100') and len(parts) == 3:
        topic = int(parts[1])
    q = parse_qs(p.query)
    vals = q.get("thread") or q.get("topic") or []
    if vals:
        try: topic = int(vals[0])
        except ValueError as e: raise ValueError("invalid topic id") from e
    return TelegramLink(chat, mid, topic, str(caption))


def filter_refs(refs: Iterable[Reference], *, chat_id=None, topic_ids=None, include_ids=None, exclude_ids=None, author_ids=None):
    """Fail-closed scope filtering; explicit exclusions always win."""
    includes = None if include_ids is None else set(include_ids)
    excludes = set(exclude_ids or ())
    authors = None if author_ids is None else set(author_ids)
    return tuple(r for r in refs if (chat_id is None or r.chat_id == chat_id)
                 and (includes is None or r.message_id in includes)
                 and r.message_id not in excludes
                 and (authors is None or r.subject in authors))


def remap_roles(refs: Iterable[Reference]):
    ordered = tuple(refs)
    return ordered, {i: r.role for i, r in enumerate(ordered, 1)}


def rewrite_ref_numbers(prompt: str, mapping: dict[int, int]) -> str:
    import re
    return re.sub(r"\bREF\s*#?(\d+)\b", lambda m: f"REF #{mapping[int(m.group(1))]}" if int(m.group(1)) in mapping else m.group(0), prompt, flags=re.I)


def provider_inputs(prompt, inputs, profile, *, references=()):
    """Validate and serialize provider images in stable order; fail rather than drop refs."""
    import base64
    import json
    from gen_provider import validate_image

    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    if isinstance(profile.max_images, bool) or not isinstance(profile.max_images, int) or profile.max_images < 0:
        raise ValueError("provider max_images must be a nonnegative integer")
    if isinstance(profile.max_bytes, bool) or not isinstance(profile.max_bytes, int) or profile.max_bytes < 1:
        raise ValueError("provider max_bytes must be a positive integer")
    values = tuple(inputs)
    if len(values) > profile.max_images:
        raise ValueError(f"image count {len(values)} exceeds provider limit {profile.max_images}")
    if references and len(references) != len(values):
        raise ValueError("reference/input count mismatch; refusing to alter image indices")
    packed = []
    for index, value in enumerate(values, 1):
        if isinstance(value, str):
            try:
                raw = base64.b64decode(value, validate=True)
            except Exception as exc:
                raise ValueError(f"input image {index} has invalid base64") from exc
        elif isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
        else:
            raise ValueError(f"input image {index} must be bytes or base64 text")
        image = validate_image(raw, max_bytes=profile.max_bytes)
        packed.append(f"data:{image.mime_type};base64,{base64.b64encode(image.data).decode('ascii')}")
    payload = json.dumps({"prompt": prompt, "images": packed}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > profile.max_bytes:
        raise ValueError(f"provider payload exceeds byte limit ({len(payload)} > {profile.max_bytes})")
    return prompt, tuple(packed)


__all__ = ["Reference", "ReferenceRegistry", "make_reference", "resize_image", "TelegramLink", "parse_telegram_link", "filter_refs", "remap_roles", "rewrite_ref_numbers", "provider_inputs"]

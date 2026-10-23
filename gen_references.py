"""Immutable reference registry and safe Telegram link parsing for image generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from io import BytesIO
from typing import Iterable, Optional
from urllib.parse import parse_qs, unquote, urlsplit


REFERENCE_ROLES = ("subject", "style", "clothing", "background", "pose", "composition", "object")
ROLE_ALIASES = {"identity": "subject", "primary": "subject", "matching": "subject",
                "submitted": "subject", "template": "composition"}


def normalize_roles(roles=(), *, legacy_role="reference") -> tuple[str, ...]:
    values = (roles,) if isinstance(roles, str) else tuple(roles)
    if not values:
        values = (legacy_role,) if legacy_role != "reference" else ()
    normalized = tuple(dict.fromkeys(ROLE_ALIASES.get(role, role) for role in values))
    if any(role not in REFERENCE_ROLES for role in normalized):
        raise ValueError("invalid reference role")
    return normalized


@dataclass(frozen=True)
class ReferenceOrigin:
    ref_id: str
    source_type: str
    chat_id: int
    message_id: int
    digest: str
    roles: tuple[str, ...] = ()
    caption: str = ""
    subject: Optional[str] = None
    image_url: Optional[str] = None
    source_page_url: Optional[str] = None
    title: str = ""
    author: str = ""
    source_status: str = "unverified"

    def __post_init__(self):
        object.__setattr__(self, "roles", normalize_roles(self.roles))


def _merge_origins(origins):
    merged = {}
    for origin in origins:
        key = replace(origin, roles=())
        previous = merged.get(key)
        merged[key] = origin if previous is None else replace(
            previous, roles=tuple(dict.fromkeys((*previous.roles, *origin.roles))))
    return tuple(merged.values())


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
    source: str = "catalog"
    source_type: str = "catalog"
    mime_type: str = "application/octet-stream"
    size_bytes: int = 0
    roles: tuple[str, ...] = ()
    provenance: tuple[ReferenceOrigin, ...] = ()
    aliases: tuple[str, ...] = ()
    image_url: Optional[str] = None
    source_page_url: Optional[str] = None
    title: str = ""
    author: str = ""
    source_status: str = "unverified"

    def __post_init__(self):
        if not isinstance(self.data, (bytes, bytearray)):
            raise ValueError("reference data must be bytes")
        raw = bytes(self.data)
        object.__setattr__(self, "data", raw)
        object.__setattr__(self, "digest", sha256(raw).hexdigest())
        object.__setattr__(self, "size_bytes", len(raw))
        roles = normalize_roles(self.roles, legacy_role=self.role)
        object.__setattr__(self, "roles", roles)
        if self.role == "reference" and roles:
            object.__setattr__(self, "role", roles[0])
        object.__setattr__(self, "aliases", tuple(dict.fromkeys((self.ref_id, *self.aliases))))
        origins = tuple(ReferenceOrigin(**origin) if isinstance(origin, dict) else origin
                        for origin in self.provenance)
        if not origins:
            origins = (ReferenceOrigin(self.ref_id, self.source_type, self.chat_id, self.message_id,
                                       self.digest, roles, self.caption, self.subject, self.image_url,
                                       self.source_page_url, self.title, self.author, self.source_status),)
        object.__setattr__(self, "provenance", _merge_origins(origins))


@dataclass(frozen=True)
class WorkingSet:
    refs: tuple[Reference, ...]
    overflow_ids: tuple[str, ...] = ()
    missing_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceIssue:
    ref_id: str
    reason: str


@dataclass(frozen=True)
class PackedReferences:
    refs: tuple[Reference, ...]
    prompt: str
    status: str
    issues: tuple[ReferenceIssue, ...] = ()
    required_missing: tuple[str, ...] = ()


def _nonnegative_limit(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _submitted(ref):
    return any(origin.source_type == "submitted" for origin in ref.provenance)


def make_reference(chat_id: int, message_id: int, data: bytes, **metadata) -> Reference:
    if not isinstance(chat_id, int) or isinstance(chat_id, bool) or not isinstance(message_id, int) or isinstance(message_id, bool):
        raise ValueError("chat_id/message_id must be integers")
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("reference data must be bytes")
    raw = bytes(data)
    digest = sha256(raw).hexdigest()
    source_type = metadata.get("source_type", "catalog")
    namespace = {"catalog": "chat", "avatar": "chat"}.get(source_type, source_type)
    if namespace not in {"submitted", "chat", "web"}:
        raise ValueError("invalid reference source_type")
    legacy_id = sha256(f"{chat_id}:{message_id}:{digest}".encode()).hexdigest()[:24]
    source_key = metadata.get("image_url") or ""
    rid = namespace + ":" + sha256(f"{source_type}:{chat_id}:{message_id}:{source_key}:{digest}".encode()).hexdigest()[:24]
    if source_type == "catalog":
        metadata["aliases"] = tuple(dict.fromkeys((*metadata.get("aliases", ()), legacy_id)))
    size = None
    try:
        from PIL import Image, ImageOps
        with Image.open(BytesIO(raw)) as im:
            im = ImageOps.exif_transpose(im)
            size = im.size
    except Exception:
        pass
    mime = "application/octet-stream"
    try:
        from PIL import Image
        with Image.open(BytesIO(raw)) as image:
            mime = Image.MIME.get(image.format, mime)
    except Exception:
        pass
    metadata.setdefault("source_type", "catalog")
    metadata.setdefault("mime_type", mime)
    metadata.setdefault("size_bytes", len(raw))
    return Reference(rid, chat_id, message_id, digest, raw, pixel_size=size, **metadata)


class ReferenceRegistry:
    """Unbounded source registry; bounded views never delete registered evidence.

    Register only sources already authorized for this invocation. Content deduplication
    is not an access-control decision. Legacy IDs remain aliases, never API indices.
    """
    def __init__(self, refs: Iterable[Reference] = ()):
        self._refs: tuple[Reference, ...] = ()
        for ref in refs:
            self.add(ref)

    @property
    def refs(self):
        return self._refs

    def add(self, ref: Reference) -> Reference:
        for old in self._refs:
            if set(old.aliases).intersection(ref.aliases) and old.digest != ref.digest:
                raise ValueError("reference ID reused for different bytes")
        for index, old in enumerate(self._refs):
            if old.digest == ref.digest:
                merged = replace(old, roles=tuple(dict.fromkeys((*old.roles, *ref.roles))),
                                 provenance=_merge_origins((*old.provenance, *ref.provenance)),
                                 aliases=tuple(dict.fromkeys((*old.aliases, *ref.aliases))),
                                 priority=max(old.priority, ref.priority), api_index=None)
                if merged == old:
                    return old
                self._refs = self._refs[:index] + (merged,) + self._refs[index + 1:]
                return merged
        ref = replace(ref, api_index=None) if ref.api_index is not None else ref
        self._refs += (ref,)
        return ref

    def resolve(self, ref_id: str) -> Optional[Reference]:
        return next((ref for ref in self._refs if ref_id in ref.aliases), None)

    def selected(self, ref_ids: Iterable[str], max_refs: int = 16, *, optional_ids=()) -> tuple[Reference, ...]:
        _nonnegative_limit(max_refs, "max_refs")
        ids = tuple(dict.fromkeys(ref_ids))
        optional = set(optional_ids)
        missing = [rid for rid in ids if self.resolve(rid) is None and rid not in optional]
        if missing:
            raise KeyError(f"selected references unavailable: {', '.join(missing)}")
        resolved = {ref.ref_id: ref for rid in ids if (ref := self.resolve(rid)) is not None}
        chosen = sorted(resolved.values(), key=lambda ref: (not _submitted(ref), -ref.priority))
        if len(chosen) > max_refs:
            raise ValueError(f"{len(chosen)} required references exceed limit {max_refs}")
        return tuple(replace(r, api_index=i) for i, r in enumerate(chosen, 1))

    @staticmethod
    def api_mapping(refs: Iterable[Reference]) -> dict[str, int]:
        return {alias: ref.api_index for ref in refs if ref.api_index is not None for alias in ref.aliases}

    @staticmethod
    def manifest(refs: Iterable[Reference]) -> list[dict]:
        return [{"ref_id": ref.ref_id, "aliases": list(ref.aliases), "api_index": ref.api_index,
                 "roles": list(ref.roles), "digest": ref.digest, "mime": ref.mime_type,
                 "role": ref.role, "source_type": ref.source_type, "source": ref.source,
                 "chat_id": ref.chat_id, "message_id": ref.message_id, "caption": ref.caption,
                 "size_bytes": ref.size_bytes, "pixel_size": ref.pixel_size,
                 "provenance": [asdict(origin) for origin in ref.provenance]} for ref in refs]

    def working_set(self, max_refs: int, *, selected_ids=(), inspected_ids=()) -> WorkingSet:
        """Protect user/selected refs; prefer recent inspections over weak candidates.

        If protected refs alone overflow, report them rather than silently forgetting
        them. The caller can request a compact selection or show multiple batches.
        """
        _nonnegative_limit(max_refs, "max_refs")
        selected = tuple(selected_ids)
        inspected = tuple(inspected_ids)
        protected = {ref.ref_id for rid in selected if (ref := self.resolve(rid)) is not None}
        recent = {ref.ref_id for rid in inspected if (ref := self.resolve(rid)) is not None}
        protected.update(ref.ref_id for ref in self._refs if _submitted(ref))
        ranked = sorted(self._refs, key=lambda ref: (ref.ref_id not in protected, not _submitted(ref),
                                                    ref.ref_id not in recent, -ref.priority))
        visible = tuple(ranked[:max_refs])
        return WorkingSet(visible, tuple(ref.ref_id for ref in ranked[max_refs:] if ref.ref_id in protected | recent),
                          tuple(dict.fromkeys(rid for rid in (*selected, *inspected) if self.resolve(rid) is None)))

    def pack(self, ref_ids: Iterable[str], profile, *, prompt="", required_ids=(), max_side=1536) -> PackedReferences:
        """Offer a validated compact set, or text-only fallback, without changing intent.

        Payload size uses provider_inputs' expanded JSON budget, not raw image bytes.
        A needs_reselection result must be shown to the agent before accepting losses.
        An unavailable result means even text exceeds the real provider budget.
        """
        import json
        from gen_provider import validate_image

        _nonnegative_limit(profile.max_images, "provider max_images")
        _nonnegative_limit(profile.max_bytes, "provider max_bytes")
        if profile.max_bytes < 1:
            raise ValueError("provider max_bytes must be a positive integer")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be text")
        _nonnegative_limit(max_side, "max_side")
        if max_side < 1:
            raise ValueError("max_side must be positive")
        required = tuple(dict.fromkeys(required_ids))
        ids = tuple(dict.fromkeys((*ref_ids, *required)))
        issues = []
        resolved = {}
        for rid in ids:
            ref = self.resolve(rid)
            if ref is None:
                issues.append(ReferenceIssue(rid, "unavailable"))
            else:
                resolved[ref.ref_id] = ref
        mandatory = {ref.ref_id for rid in required if (ref := self.resolve(rid)) is not None}
        mandatory.update(ref.ref_id for ref in resolved.values() if _submitted(ref))
        ordered = sorted(resolved.values(), key=lambda ref: (ref.ref_id not in mandatory, not _submitted(ref), -ref.priority))
        packed = []
        try:
            provider_inputs(prompt, (), profile)
        except ValueError:
            return PackedReferences((), prompt, "unavailable", (ReferenceIssue("", "prompt_budget"),), required)
        valid = []
        for ref in ordered:
            try:
                validated = validate_image(ref.data, max_bytes=max(len(ref.data), profile.max_bytes))
            except ValueError:
                issues.append(ReferenceIssue(ref.ref_id, "invalid_image"))
                continue
            valid.append(replace(ref, data=validated.data, mime_type=validated.mime_type, pixel_size=validated.pixel_size))
        candidates = valid[:profile.max_images]
        issues.extend(ReferenceIssue(ref.ref_id, "count_cap") for ref in valid[profile.max_images:])
        try:
            provider_inputs(prompt, [ref.data for ref in candidates], profile)
            compact = False
        except ValueError:
            compact = True
        for position, ref in enumerate(candidates):
            candidate = ref
            _, encoded = provider_inputs(prompt, [item.data for item in packed], profile)
            used = len(json.dumps({"prompt": prompt, "images": encoded}, ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8"))
            remaining_slots = len(candidates) - position
            byte_budget = int(((profile.max_bytes - used) / remaining_slots - 32) * .75)
            try:
                if compact and len(candidate.data) > byte_budget:
                    raise ValueError("reserve capacity for remaining references")
                provider_inputs(prompt, [item.data for item in (*packed, candidate)], profile)
            except ValueError:
                try:
                    resized = resize_image(ref.data, max_side=max_side, max_bytes=byte_budget)
                    normalized = validate_image(resized, max_bytes=profile.max_bytes)
                    candidate = replace(ref, data=normalized.data, mime_type=normalized.mime_type,
                                        pixel_size=normalized.pixel_size)
                    provider_inputs(prompt, [item.data for item in (*packed, candidate)], profile)
                except ValueError:
                    issues.append(ReferenceIssue(ref.ref_id, "byte_cap"))
                    continue
            packed.append(candidate)
        packed_ids = {ref.ref_id for ref in packed}
        missing = tuple(dict.fromkeys([rid for rid in required if self.resolve(rid) is None or
                                      self.resolve(rid).ref_id not in packed_ids] +
                                     [ref.ref_id for ref in ordered if _submitted(ref) and ref.ref_id not in packed_ids]))
        final = tuple(replace(ref, api_index=index) for index, ref in enumerate(packed, 1))
        status = "text_only" if not final else "needs_reselection" if issues else "ready"
        return PackedReferences(final, prompt, status, tuple(issues), missing)


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


__all__ = ["Reference", "ReferenceRegistry", "make_reference", "resize_image", "TelegramLink", "parse_telegram_link", "filter_refs", "remap_roles", "rewrite_ref_numbers", "provider_inputs",
           "REFERENCE_ROLES", "normalize_roles", "ReferenceOrigin", "WorkingSet", "PackedReferences", "ReferenceIssue"]

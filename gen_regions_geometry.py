"""Geometry and compositing helpers for image-region repair.

Bounding boxes use normalized (x1, y1, x2, y2) coordinates in [0, 1],
with x2/y2 exclusive. PIL images are RGB/RGBA or other modes supported by
Pillow; all pixel coordinates use the image's native size.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFilter

BBox = Sequence[float]


def bbox_to_pixels(bbox: BBox, image_size: tuple[int, int], margin: float = 0.0) -> tuple[int, int, int, int]:
    """Validate normalized bbox, expand by normalized margin, return clamped pixel box."""
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    if len(image_size) != 2 or any(not isinstance(n, int) or n <= 0 for n in image_size):
        raise ValueError("image_size must be positive integer (width, height)")
    vals = tuple(float(v) for v in bbox)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("bbox coordinates must be finite")
    x1, y1, x2, y2 = vals
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("bbox must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    w, h = image_size
    return (max(0, math.floor((x1-margin)*w)), max(0, math.floor((y1-margin)*h)),
            min(w, math.ceil((x2+margin)*w)), min(h, math.ceil((y2+margin)*h)))


def merge_overlapping_boxes(boxes: Iterable[Sequence[int]], *, touching: bool = True) -> list[tuple[int, int, int, int]]:
    """Union overlapping (optionally edge-touching) pixel boxes, transitively."""
    result = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError("each box must have four coordinates")
        b = tuple(int(v) for v in box)
        if b[0] >= b[2] or b[1] >= b[3]:
            raise ValueError("boxes must have positive area")
        result.append(b)
    changed = True
    while changed:
        changed = False
        out = []
        while result:
            a = result.pop()
            i = 0
            while i < len(result):
                b = result[i]
                gap = 0 if touching else 1
                if a[0] <= b[2]-gap and b[0] <= a[2]-gap and a[1] <= b[3]-gap and b[1] <= a[3]-gap:
                    a = (min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3]))
                    result.pop(i)
                    changed = True
                    i = 0
                else:
                    i += 1
            out.append(a)
        result = out
    return sorted(result, key=lambda b: (b[1], b[0], b[3], b[2]))


def feather_mask(size: tuple[int, int], feather: int) -> Image.Image:
    """Create L-mode mask with a feathered interior and exact zero perimeter."""
    w, h = size
    if w <= 0 or h <= 0 or not isinstance(feather, int) or feather < 0:
        raise ValueError("size must be positive and feather a nonnegative integer")
    mask = Image.new("L", (w, h), 255)
    if feather:
        # Blur a solid interior, then explicitly force the patch edge to zero.
        inset = min(feather, (w-1)//2, (h-1)//2)
        if inset > 0:
            core = Image.new("L", (w, h), 0)
            ImageDraw.Draw(core).rectangle((inset, inset, w-inset-1, h-inset-1), fill=255)
            mask = core.filter(ImageFilter.GaussianBlur(radius=max(0.5, feather/2)))
    if w:
        mask.putpixel((0, 0), 0)
        for x in range(w):
            mask.putpixel((x, 0), 0)
            mask.putpixel((x, h-1), 0)
    for y in range(h):
        mask.putpixel((0, y), 0)
        mask.putpixel((w-1, y), 0)
    return mask


def paste_repair(original: Image.Image, repaired_patch: Image.Image, box: Sequence[int], *, feather: int = 0) -> Image.Image:
    """Composite repaired patch into original; all pixels outside box stay identical."""
    if len(box) != 4:
        raise ValueError("box must have four coordinates")
    x1, y1, x2, y2 = (int(v) for v in box)
    if not (0 <= x1 < x2 <= original.width and 0 <= y1 < y2 <= original.height):
        raise ValueError("box must be a nonempty region within original")
    if repaired_patch.size != (x2-x1, y2-y1):
        raise ValueError("repaired_patch size must equal box dimensions")
    if not isinstance(feather, int) or feather < 0:
        raise ValueError("feather must be a nonnegative integer")
    base = original.convert("RGBA")
    patch = repaired_patch.convert("RGBA")
    if feather:
        base.alpha_composite(Image.new("RGBA", base.size, (0, 0, 0, 0)))
        base.paste(patch, (x1, y1), feather_mask(patch.size, feather))
    else:
        base.paste(patch, (x1, y1))
    return base.convert(original.mode)


def validate_bbox(bbox: object) -> list[float] | None:
    """Return a normalized valid bbox as a list, otherwise None."""
    try:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        vals = [float(v) for v in bbox]
        if not all(math.isfinite(v) for v in vals):
            return None
        x1, y1, x2, y2 = vals
        return vals if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 else None
    except (TypeError, ValueError):
        return None


def plan_regions(findings: Iterable[dict], size: tuple[int, int], margin: float = .35) -> list[dict]:
    """Convert finding bboxes to pixels and group overlap components."""
    rows = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        bbox = finding.get("bbox", finding.get("box"))
        valid = validate_bbox(bbox)
        if valid is None:
            continue
        x0,y0,x1,y1 = valid
        dx,dy = (x1-x0)*margin, (y1-y0)*margin
        expanded = [max(0,x0-dx),max(0,y0-dy),min(1,x1+dx),min(1,y1+dy)]
        rows.append((bbox_to_pixels(expanded, size, 0), finding))
    groups: list[dict] = []
    for box, finding in rows:
        matches = []
        for i, group in enumerate(groups):
            b = group["box"]
            if box[0] <= b[2] and b[0] <= box[2] and box[1] <= b[3] and b[1] <= box[3]:
                matches.append(i)
        if not matches:
            groups.append({"box": box, "findings": [finding]})
        else:
            first = matches[0]
            group = groups[first]
            group["box"] = (min(group["box"][0], box[0]), min(group["box"][1], box[1]), max(group["box"][2], box[2]), max(group["box"][3], box[3]))
            group["findings"].append(finding)
            for idx in reversed(matches[1:]):
                other = groups.pop(idx)
                group["box"] = (min(group["box"][0], other["box"][0]), min(group["box"][1], other["box"][1]), max(group["box"][2], other["box"][2]), max(group["box"][3], other["box"][3]))
                group["findings"].extend(other["findings"])
    return groups


def blend_patch(canvas: Image.Image, repaired: Image.Image, box: Sequence[int]) -> Image.Image:
    """Feather-blend a patch into canvas with exact outside preservation."""
    target = (int(box[2])-int(box[0]), int(box[3])-int(box[1]))
    repaired = repaired.resize(target, Image.Resampling.LANCZOS)
    feather = max(1, min(target) // 12)
    return paste_repair(canvas, repaired, box, feather=feather)


__all__ = ["bbox_to_pixels", "merge_overlapping_boxes", "feather_mask", "paste_repair", "validate_bbox", "plan_regions", "blend_patch"]

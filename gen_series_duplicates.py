"""Conservative image-level guard; reject copies, not merely similar characters."""
from io import BytesIO
from hashlib import sha256
from PIL import Image, ImageChops, ImageStat

def _thumb(raw):
    with Image.open(BytesIO(raw)) as im:
        return im.convert('RGB').resize((96,96), Image.Resampling.LANCZOS)

def duplicate_page(candidate: bytes, previous: list[bytes]) -> bool:
    current = None
    for old in previous:
        if sha256(candidate).digest() == sha256(old).digest():
            return True
        try:
            if current is None: current = _thumb(candidate)
            other = _thumb(old)
        except (OSError, ValueError):
            continue
        # Very low threshold: allows continuity, catches recompression and tiny edits.
        diff = ImageChops.difference(current, other)
        mean = sum(ImageStat.Stat(diff).mean) / 3
        changed = sum(1 for px in diff.getdata() if max(px) > 18) / (96*96)
        if mean < 2.5 and changed < .025:
            return True
    return False

def assert_distinct_page(candidate: bytes, previous: list[bytes]) -> None:
    if duplicate_page(candidate, previous):
        raise ValueError('duplicate series image: new page repeats an earlier output')

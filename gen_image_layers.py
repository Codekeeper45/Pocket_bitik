"""Validated independent RGBA layer planning and compositing for /gen."""
import json
import math
from PIL import Image, ImageFilter

MAX_LAYERS = 32
MAX_ASSET_PIXELS = 16_000_000

class LayerError(ValueError):
    pass

def parse_plan(text, width, height, user_prompt):
    """Parse strict JSON plan; malformed/incomplete plans fail closed."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:].strip()
    try:
        plan = json.loads(text)
    except Exception as exc:
        raise LayerError("layer planner did not return valid JSON") from exc
    if not isinstance(plan, dict) or not isinstance(plan.get("background"), dict):
        raise LayerError("plan requires a background object")
    layers = plan.get("layers")
    if not isinstance(layers, list) or not layers or len(layers) > MAX_LAYERS:
        raise LayerError(f"plan must contain 1..{MAX_LAYERS} foreground layers")
    if not plan.get("style") or not plan.get("lighting") or not plan.get("perspective"):
        raise LayerError("plan must define global style, lighting, and perspective")
    if not isinstance(plan['background'].get('prompt'), str) or not plan['background']['prompt'].strip():
        raise LayerError('background prompt missing')
    names = str(user_prompt or "")
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict) or not all(k in layer for k in ("prompt", "x", "y", "w", "h")):
            raise LayerError(f"layer {i+1} missing prompt or normalized placement")
        for k in ("x", "y", "w", "h"):
            v = layer[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise LayerError(f"layer {i+1}: {k} must be finite")
        x,y,w,h = (float(layer[k]) for k in ("x","y","w","h"))
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > 1 or y+h > 1:
            raise LayerError(f"layer {i+1} placement must fit normalized canvas")
        group = layer.get("identities", [])
        if not isinstance(group, list):
            raise LayerError(f"layer {i+1} identities must be a list")
    identities = [str(n) for l in layers for n in l.get("identities", [])]
    requested = plan.get("requested_identities", [])
    if not isinstance(requested, list):
        raise LayerError("requested_identities must be a list")
    # Planner must preserve every explicitly requested name/count verbatim in its declared roster.
    for item in requested:
        if not isinstance(item, str) or not item.strip():
            raise LayerError("requested_identities entries must be nonempty strings")
        if item not in identities:
            raise LayerError(f"requested identity omitted from layers: {item}")
    plan["layers"] = layers
    return plan

def composite_layers(background, layers, size, blur_background=False):
    """Composite genuine alpha assets; reject opaque generated foregrounds."""
    width, height = map(int, size)
    if width <= 0 or height <= 0 or width*height > MAX_ASSET_PIXELS:
        raise LayerError("canvas exceeds pixel budget")
    bg = Image.open(background).convert("RGBA") if not isinstance(background, Image.Image) else background.convert("RGBA")
    bg = bg.resize((width,height), Image.Resampling.LANCZOS)
    if blur_background:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=max(1, min(width,height)*0.006)))
    for asset, spec in layers:
        im = Image.open(asset).convert("RGBA") if not isinstance(asset, Image.Image) else asset.convert("RGBA")
        alpha = im.getchannel("A")
        amin, amax = alpha.getextrema()
        if amin >= 250 or amax <= 8:
            raise LayerError("foreground asset has no usable genuine transparency; refusing opaque rectangle compositing")
        x,y,w,h = [float(spec[k]) for k in ("x","y","w","h")]
        box = (round(x*width), round(y*height), max(1,round(w*width)), max(1,round(h*height)))
        if box[2]*box[3] > MAX_ASSET_PIXELS:
            raise LayerError("layer pixel budget exceeded")
        # Preserve anatomy proportions; fit assets, never stretch their faces/bodies.
        scale = min(box[2] / im.width, box[3] / im.height)
        im = im.resize((max(1, round(im.width*scale)), max(1, round(im.height*scale))), Image.Resampling.LANCZOS)
        bg.alpha_composite(im, dest=(box[0] + (box[2]-im.width)//2, box[1] + box[3]-im.height))
    return bg


def make_planner_prompt(user_prompt, width, height):
    return ("Return JSON only with keys style, lighting, perspective, background, requested_identities, layers. "
            "background: {prompt}. layers: array of independently renderable character/group objects, each with prompt, identities array, "
            "and normalized x,y,w,h (0..1). Preserve EVERY person/count/name explicitly requested; do not cap named characters. "
            "Use as many layers/groups as budget permits, group only when identity/count remains explicit. Keep one coherent style, light direction, "
            "camera/perspective and ground plane across assets. Foreground prompts MUST request isolated subjects on genuinely transparent RGBA alpha, "
            "no backdrop, no text labels unless explicitly requested. No blur is mandatory. Limit to 31 foreground layers; report limitations rather than omit requested people. "
            f"Canvas {width}x{height}. User request: {user_prompt}")

__all__ = ["LayerError", "parse_plan", "composite_layers", "make_planner_prompt", "MAX_LAYERS"]

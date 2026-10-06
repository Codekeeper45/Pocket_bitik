"""Typed scene-contract extraction, validation and compact prompt serialization."""
from __future__ import annotations
from dataclasses import dataclass, field
import json


@dataclass(frozen=True)
class SceneContract:
    task_type: str
    prompt: str
    participants: tuple[str, ...] = ()
    participant_count: int | None = None
    known_appearance: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    action: str = ""
    composition: str = ""
    required_text: tuple[str, ...] = ()
    style: str = ""
    aspect: str | None = None
    refs: tuple[dict, ...] = ()
    immutable_requirements: tuple[str, ...] = ()
    fictional_interpretation: bool = False


class ContractError(ValueError):
    pass


def extract_contract(value) -> SceneContract:
    if isinstance(value, str):
        text = value.strip()
        if "{" in text and "}" in text:
            text = text[text.find("{"):text.rfind("}")+1]
        try: value = json.loads(text)
        except json.JSONDecodeError as exc: raise ContractError("expected structured JSON scene contract") from exc
    if not isinstance(value, dict): raise ContractError("scene contract must be an object")
    allowed = {"task_type", "prompt", "participants", "participant_count", "known_appearance", "hypotheses", "action", "composition", "required_text", "style", "aspect", "refs", "immutable_requirements", "fictional_interpretation"}
    unknown = set(value) - allowed
    if unknown: raise ContractError(f"unknown fields: {', '.join(sorted(unknown))}")
    def strings(key):
        v = value.get(key, ())
        if not isinstance(v, (list, tuple)) or any(not isinstance(x, str) or not x.strip() for x in v):
            raise ContractError(f"{key} must be an array of nonempty strings")
        return tuple(x.strip() for x in v)
    task = value.get("task_type", "creation")
    prompt = value.get("prompt")
    if task not in {"creation", "edit", "series"}: raise ContractError("invalid task_type")
    if not isinstance(prompt, str) or not prompt.strip(): raise ContractError("prompt is required")
    count = value.get("participant_count")
    if count is not None:
        if isinstance(count, str) and count.strip().isdigit():
            count = int(count.strip())
        elif not isinstance(count, int) or count < 0:
            raise ContractError("participant_count must be a nonnegative integer")
    participants = strings("participants")
    if count is not None and participants and count != len(participants):
        raise ContractError("participant_count conflicts with participants")
    aspect = value.get("aspect")
    if aspect not in (None, "1:1", "9:16", "16:9", "4:3", "3:4"): raise ContractError("unsupported aspect")
    refs = value.get("refs", ())
    if not isinstance(refs, (list, tuple)) or any(not isinstance(r, dict) or not r.get("ref_id") or not r.get("role") for r in refs): raise ContractError("refs require ref_id and role")
    fiction = value.get("fictional_interpretation", False)
    if type(fiction) is not bool: raise ContractError("fictional_interpretation must be boolean")
    return SceneContract(task, prompt.strip(), participants, count, strings("known_appearance"), strings("hypotheses"), str(value.get("action", "")), str(value.get("composition", "")), strings("required_text"), str(value.get("style", "")), aspect, tuple(dict(r) for r in refs), strings("immutable_requirements"), fiction)


def contract_to_dict(c: SceneContract) -> dict:
    from dataclasses import asdict
    return asdict(c)


def render_prompt(c: SceneContract, *, user_aspect: str | None = None, refs=None) -> str:
    """Serialize one compact, unambiguous prompt; explicit user aspect overrides model output."""
    if user_aspect is not None and user_aspect not in ("1:1", "9:16", "16:9", "4:3", "3:4"):
        raise ContractError("unsupported user aspect")
    data = contract_to_dict(c)
    data["aspect"] = user_aspect or c.aspect
    if refs is not None:
        valid = {r.ref_id for r in refs}
        requested = [r["ref_id"] for r in c.refs]
        missing = [rid for rid in requested if rid not in valid]
        if missing: raise ContractError("missing references: " + ", ".join(missing))
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def build_contract_prompt(user_prompt: str, *, task_type="creation", participants=(), participant_count=None, required_text=(), aspect=None, refs=(), immutable_requirements=(), known_appearance=(), hypotheses=(), action="", composition="", style="", fictional_interpretation=False) -> SceneContract:
    values = locals().copy()
    values["prompt"] = values.pop("user_prompt")
    return extract_contract(values)

__all__ = ["SceneContract", "ContractError", "extract_contract", "contract_to_dict", "render_prompt", "build_contract_prompt"]

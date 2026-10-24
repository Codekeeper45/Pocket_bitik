"""Typed scene-contract extraction, validation and compact prompt serialization."""
from __future__ import annotations
from dataclasses import asdict, dataclass
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

INTENT_PLAN_INSTRUCTIONS = """Return one JSON object, not a prose answer or a second contract.
Fields: task_type (creation/edit/series), output_kind (single_image/independent_images/series),
output_count (positive integer files), prompt (scene description), panels_per_image (positive integer or null),
explicit_text (verbatim strings), refs ([{ref_id: stable source ID, roles: [subject/style/clothing/background/pose/composition/object]}]),
aspect (null/1:1/9:16/16:9/4:3/3:4), constraints (strings), layers (boolean, default false).
idea (one concise human-readable sentence describing the scene, ideally under 180 characters; never copy the user request or rendering instructions),
ideas (one distinct human-readable caption per file for multiple outputs).
Optional: scenes (one distinct scene string per file), participant_count (nonnegative integer).
Understand the user's meaning and negations; panels are not files. Distinguish independent images from a continuous series.
Explicit command parameters override your choices. Preserve the scene, exact text and source identities.
Search finds candidates; inspect pixels to confirm observations. Keep captions, observed text and visual descriptions distinct.
Separate facts from hypotheses; do not invent unreadable text or claim an unverified source is canonical.
For a named ensemble, establish the actual cast before writing the rendering scene: use chat_list_participants for chat accounts, original source images for show characters, and chat_get_avatar/chat_inspect_image for appearance. An account name or caption alone does not establish a character design. Do not recycle previous generated outputs as evidence of canon.
For large casts, use chat_get_avatar_sheet in batches to inspect many account avatars within the tool and reference budget. A sheet is identity evidence only, not the requested scene layout; keep cell-to-account mapping and use actual exact names for final captions, not account ID numbers.
Maintain an explicit per-character mapping in your scene description: exact label, source ref_id or evidenced source message, observed distinguishing anchors, unique action and spatial placement. Keep different characters separate; do not fill requested counts by duplicating names, alternate costumes or unsupported invented identities. If you cannot substantiate the whole cast, state that limitation honestly in idea, while still rendering the best available interpretation. Never describe an approximate crowd as an exact verified roster.
Preserve exact spelling of each label. A shared reference can contain several characters, but identify which visible character belongs to each name and do not borrow another character's appearance. Use spatial groups to keep labels connected and silhouettes distinct, not a generic instruction to make everyone unique.
Finish with the available references and an honest interpretation rather than refusing for imperfect sources."""


@dataclass(frozen=True)
class IntentPlan:
    task_type: str
    output_kind: str
    output_count: int
    prompt: str
    panels_per_image: int | None = None
    explicit_text: tuple[str, ...] = ()
    refs: tuple[dict, ...] = ()
    aspect: str | None = None
    constraints: tuple[str, ...] = ()
    layers: bool = False
    scenes: tuple[str, ...] = ()
    participant_count: int | None = None
    source_prompt: str = ""
    fallback_used: bool = False
    raw: bool = False
    idea: str = ""
    ideas: tuple[str, ...] = ()


def _intent_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def _intent_strings(value, name):
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ContractError(f"{name} must be an array of nonempty strings")
    return tuple(value)


def _intent_refs(value):
    if not isinstance(value, (list, tuple)):
        raise ContractError("refs must be an array")
    merged = {}
    for item in value:
        if not isinstance(item, dict):
            raise ContractError("refs must contain objects")
        ref_id = item.get("ref_id")
        if not isinstance(ref_id, str) or not ref_id.strip():
            raise ContractError("refs require a stable ref_id")
        roles = _intent_strings(item.get("roles", ()), "roles")
        if "role" in item:
            roles += _intent_strings([item["role"]], "role")
        if not roles or any(role not in ("subject", "style", "clothing", "background", "pose", "composition", "object") for role in roles):
            raise ContractError("unsupported reference role")
        target = merged.setdefault(ref_id, {"ref_id": ref_id, "roles": []})
        for role in roles:
            if role not in target["roles"]:
                target["roles"].append(role)
    return tuple(merged.values())


def _intent_object(value):
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.partition("\n")[2].rsplit("```", 1)[0].strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ContractError("expected a JSON intent plan") from exc
    if not isinstance(value, dict):
        raise ContractError("intent plan must be an object")
    allowed = {"task_type", "output_kind", "output_count", "prompt", "panels_per_image", "explicit_text", "refs", "aspect", "constraints", "layers", "scenes", "participant_count", "idea", "ideas"}
    if set(value) - allowed:
        raise ContractError("unknown intent plan fields")
    return dict(value)


def parse_intent_plan(out, user_request=None, explicit_count=None, edit_mode=False, *,
                      user_aspect=None, task_type=None, raw=False, repair=None,
                      fallback_prompt=None, fallback_refs=(), source_prompt=None, output_count=None) -> IntentPlan:
    """Parse the agent's sole response, without interpreting natural language or calling APIs.

    Explicit parameters are already parsed command flags. A synchronous
    repair(out, error) may run once on schema failure; otherwise preserve the
    source request and any caller-supplied legacy footer prompt/refs. The caller
    owns asynchronous model calls, resource limits and actual reference lookup.
    source_prompt/output_count are keyword aliases for user_request/explicit_count.
    """
    source = user_request if user_request is not None else source_prompt
    if not isinstance(source, str) or not source.strip():
        raise ContractError("user_request is required")
    if explicit_count is not None and output_count is not None and explicit_count != output_count:
        raise ContractError("conflicting explicit counts")
    count_override = explicit_count if explicit_count is not None else output_count
    if count_override is not None:
        _intent_integer(count_override, "explicit_count")
    if user_aspect not in (None, "1:1", "9:16", "16:9", "4:3", "3:4"):
        raise ContractError("unsupported user aspect")
    if task_type not in (None, "creation", "edit", "series"):
        raise ContractError("invalid explicit task_type")
    if type(edit_mode) is not bool or type(raw) is not bool:
        raise ContractError("edit_mode and raw must be boolean")
    task_override = "edit" if edit_mode else task_type

    def parse(value):
        data = _intent_object(value)
        task = task_override or data.get("task_type", "creation")
        if task not in ("creation", "edit", "series"):
            raise ContractError("invalid task_type")
        count = _intent_integer(count_override if count_override is not None else data.get("output_count", 1), "output_count")
        kind = data.get("output_kind", "single_image" if count == 1 else "independent_images")
        if kind not in ("single_image", "independent_images", "series"):
            raise ContractError("invalid output_kind")
        if kind == "single_image" and count > 1:
            if count_override is None:
                raise ContractError("single_image requires output_count=1")
            kind = "independent_images"
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ContractError("scene prompt is required")
        panels = data.get("panels_per_image")
        if panels is not None:
            _intent_integer(panels, "panels_per_image")
        participants = data.get("participant_count")
        if participants is not None:
            _intent_integer(participants, "participant_count", 0)
        aspect = user_aspect or data.get("aspect")
        if aspect not in (None, "1:1", "9:16", "16:9", "4:3", "3:4"):
            raise ContractError("unsupported aspect")
        layers = data.get("layers", False)
        if type(layers) is not bool:
            raise ContractError("layers must be boolean")
        scenes = _intent_strings(data.get("scenes", ()), "scenes")
        idea = data.get("idea", "")
        if not isinstance(idea, str):
            raise ContractError("idea must be a string")
        ideas = _intent_strings(data.get("ideas", ()), "ideas")
        if ideas and len(ideas) != count:
            if count_override is None:
                raise ContractError("ideas must match output_count")
            ideas = ()
        if scenes and len(scenes) != count:
            if count_override is None:
                raise ContractError("scenes must match output_count")
            scenes = ()
        return IntentPlan(task_type=task, output_kind=kind, output_count=count, prompt=prompt,
                          panels_per_image=panels, explicit_text=_intent_strings(data.get("explicit_text", ()), "explicit_text"),
                          refs=_intent_refs(data.get("refs", ())), aspect=aspect,
                          constraints=_intent_strings(data.get("constraints", ()), "constraints"),
                          layers=layers, scenes=scenes, participant_count=participants, source_prompt=source,
                          idea=idea.strip(), ideas=ideas)

    if not raw:
        try:
            return parse(out)
        except ContractError as original:
            if repair is not None:
                try:
                    repaired = repair(out, str(original))
                except Exception:
                    repaired = None
                try:
                    return parse(repaired)
                except ContractError:
                    pass
    if fallback_prompt is not None and (not isinstance(fallback_prompt, str) or not fallback_prompt.strip()):
        raise ContractError("fallback_prompt must be nonempty")
    count = count_override or 1
    return IntentPlan(task_type=task_override or "creation",
                      output_kind="single_image" if count == 1 else "independent_images",
                      output_count=count, prompt=source if raw else (fallback_prompt or source),
                      refs=_intent_refs(fallback_refs), aspect=user_aspect, source_prompt=source,
                      fallback_used=not raw, raw=raw)


def intent_plan_to_dict(plan: IntentPlan) -> dict:
    """Return detached, JSON-serializable state for CURRENT.timings or diagnostics."""
    return asdict(plan)


def intent_idea(plan: IntentPlan, output_number: int = 1) -> str:
    # A schema fallback preserves the request for rendering, not as a public caption.
    if plan.fallback_used and not plan.idea and not plan.ideas:
        return ''
    if plan.ideas and output_number <= len(plan.ideas):
        return plan.ideas[output_number - 1]
    if plan.idea and plan.output_count == 1:
        return plan.idea
    if plan.scenes and output_number <= len(plan.scenes):
        return plan.scenes[output_number - 1]
    return plan.idea or plan.prompt


def intent_instructions(plan: IntentPlan) -> str:
    """Compact renderer guidance selected only by structured plan fields."""
    sections = ["Preserve the user's intent and constraints. Search finds candidates; inspect pixels to confirm observations. Separate observed facts from hypotheses; use available sources without claiming invented details are canonical."]
    if plan.panels_per_image is not None:
        sections.append(f"Panels: compose exactly {plan.panels_per_image} panels within each image, not separate files.")
    if plan.task_type == "edit":
        sections.append("Edit: preserve the source image and all content outside the requested changes.")
    if plan.explicit_text:
        sections.append("Text: reproduce explicit text verbatim, preserving language, Cyrillic, punctuation and legibility; do not invent unreadable source text.")
    if plan.participant_count is not None and plan.participant_count > 1:
        sections.append(f"Group: retain all {plan.participant_count} participants, distinct identities and readable interactions.")
    if plan.output_count > 1:
        if plan.output_kind == "series":
            sections.append("Series: advance the requested sequence with distinct scenes; keep identities, clothing and visual continuity consistent.")
        else:
            sections.append("Independent images: render separate requested results, not panels or an invented sequential story.")
    if plan.layers:
        sections.append("Layers: use the planned layered composition only when enabled by the caller; preserve the complete scene if rendered directly.")
    return "\n".join(sections)


def render_intent_prompt(plan: IntentPlan, *, output_number: int | None = None) -> str:
    """Render scene instructions, never an internal JSON document or ref-number rewrite."""
    if output_number is not None:
        _intent_integer(output_number, "output_number")
        if output_number > plan.output_count:
            raise ContractError("output_number exceeds output_count")
    if plan.raw:
        return plan.source_prompt
    if output_number is None and plan.output_count == 1 and plan.scenes:
        output_number = 1
    sections = [intent_instructions(plan), "Scene:\n" + plan.prompt]
    if output_number is not None:
        sections.append(f"Output: {output_number} of {plan.output_count} separate files.")
        if plan.scenes:
            sections.append("This image:\n" + plan.scenes[output_number - 1])
    if plan.source_prompt:
        sections.append("Original user request (verbatim):\n" + plan.source_prompt + "\nEnd original user request.")
    if plan.explicit_text:
        sections.append("Explicit text:\n" + "\n".join(plan.explicit_text))
    if plan.constraints:
        sections.append("Constraints:\n" + "\n".join(plan.constraints))
    if plan.aspect:
        sections.append("Aspect: " + plan.aspect)
    if plan.refs:
        sections.append("Reference roles:\n" + "\n".join(ref["ref_id"] + ": " + ", ".join(ref["roles"]) for ref in plan.refs))
    return "\n\n".join(sections)


__all__ = ["SceneContract", "ContractError", "extract_contract", "contract_to_dict", "render_prompt", "build_contract_prompt",
           "IntentPlan", "INTENT_PLAN_INSTRUCTIONS", "parse_intent_plan", "intent_plan_to_dict", "intent_idea", "intent_instructions", "render_intent_prompt"]

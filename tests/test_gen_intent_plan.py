import asyncio
import json

import pytest

import gen_prompt
import gen_series


def model_plan(**changes):
    return dict(task_type="creation", output_kind="single_image", output_count=1,
                prompt="A quiet scene", **changes)


def test_model_count_not_natural_language():
    for source, count, panels in [
        ("Создай 3 отдельных изображения", 3, None),
        ("Комикс с 3 панелями", 1, 3),
        ("Не делай комикс на три страницы, нужен один постер", 1, None),
    ]:
        payload = model_plan(panels_per_image=panels)
        payload["output_count"] = count
        payload["output_kind"] = "single_image" if count == 1 else "independent_images"
        plan = gen_prompt.parse_intent_plan(payload, source_prompt=source)
        assert plan.output_count == count
        assert plan.panels_per_image == panels
        assert plan.source_prompt == source


def test_flags_override_model_and_preserve_source():
    source = '  Keep image #1 and REF #7; "Привет!"\n'
    plan = gen_prompt.parse_intent_plan(
        model_plan(scenes=["One scene"]), source_prompt=source,
        output_count=3, user_aspect="9:16", task_type="edit",
    )
    assert (plan.output_count, plan.aspect, plan.task_type) == (3, "9:16", "edit")
    assert plan.scenes == ()
    assert source in gen_prompt.render_intent_prompt(plan)


def test_one_repair_then_source_preserving_fallback():
    calls = []

    def repair(value, error):
        calls.append((value, error))
        return "still not JSON"

    source = "  Не три страницы. Оставь надпись «два».\n"
    plan = gen_prompt.parse_intent_plan("bad", source_prompt=source, repair=repair)
    assert len(calls) == 1
    assert plan.prompt == source
    assert plan.output_count == 1
    assert plan.fallback_used
    assert plan.source_prompt == source


def test_repair_success_and_exception():
    plan = gen_prompt.parse_intent_plan("bad", source_prompt="source",
                                        repair=lambda *_: model_plan())
    assert not plan.fallback_used

    def unavailable(*args):
        raise RuntimeError("offline")

    plan = gen_prompt.parse_intent_plan(None, source_prompt="source",
                                        output_count=2, repair=unavailable)
    assert plan.fallback_used and plan.output_count == 2


@pytest.mark.parametrize("changes", [
    {"output_count": True}, {"output_count": 0}, {"output_count": "3"},
    {"panels_per_image": False}, {"panels_per_image": -1},
    {"task_type": []}, {"aspect": []}, {"output_kind": "unknown"},
    {"explicit_text": [3]}, {"constraints": [3]},
    {"refs": [{"ref_id": "source:1", "roles": ["unknown"]}]},
    {"scenes": ["first", "second"]}, {"surprise": True},
])
def test_bad_model_schema_falls_back(changes):
    payload = model_plan()
    payload.update(changes)
    plan = gen_prompt.parse_intent_plan(payload, source_prompt="original")
    assert plan.fallback_used
    assert plan.prompt == "original"


@pytest.mark.parametrize("flags", [
    {"output_count": True}, {"output_count": 0}, {"user_aspect": "bad"},
    {"task_type": "bad"},
])
def test_bad_explicit_flags_are_not_silently_ignored(flags):
    with pytest.raises(gen_prompt.ContractError):
        gen_prompt.parse_intent_plan(model_plan(), source_prompt="source", **flags)


def test_refs_merge_roles_without_mutation_or_missing_source_block():
    refs = [{"ref_id": "submitted:1", "role": "subject"},
            {"ref_id": "submitted:1", "roles": ["clothing", "subject"]},
            {"ref_id": "catalog:1", "roles": ["style"]}]
    original = json.dumps(refs)
    plan = gen_prompt.parse_intent_plan(model_plan(refs=refs), source_prompt="source")
    assert plan.refs[0]["roles"] == ["subject", "clothing"]
    assert len(plan.refs) == 2
    assert json.dumps(refs) == original
    rendered = gen_prompt.render_intent_prompt(plan)
    assert "submitted:1" in rendered and "catalog:1" in rendered


def test_compact_instructions_are_plan_conditional():
    plan = gen_prompt.parse_intent_plan(model_plan(), source_prompt="source")
    plain = gen_prompt.intent_instructions(plan)
    assert "inspect" in plain.lower()
    for section in ("Panels:", "Edit:", "Text:", "Group:", "Series:"):
        assert section not in plain
    payload = model_plan(panels_per_image=3, explicit_text=["  Привет!  "], participant_count=8)
    payload.update(task_type="edit", output_count=2, output_kind="series")
    plan = gen_prompt.parse_intent_plan(payload, source_prompt="source")
    instructions = gen_prompt.intent_instructions(plan)
    for section in ("Panels:", "Edit:", "Text:", "Group:", "Series:"):
        assert section in instructions
    assert "  Привет!  " in gen_prompt.render_intent_prompt(plan)


def test_valid_json_and_raw_bypass():
    payload = json.dumps(model_plan(), ensure_ascii=False)
    plan = gen_prompt.parse_intent_plan(f"```json\n{payload}\n```", source_prompt="source")
    assert not plan.fallback_used
    raw = gen_prompt.parse_intent_plan(None, source_prompt=" exact raw\n", raw=True,
                                      repair=lambda *_: pytest.fail("raw called repair"))
    assert gen_prompt.render_intent_prompt(raw) == " exact raw\n"


def test_legacy_contract_interface_is_unchanged():
    contract = gen_prompt.build_contract_prompt("legacy", required_text=["Hello"])
    assert gen_prompt.extract_contract(gen_prompt.render_prompt(contract)) == contract


def test_series_adapter_checkpoint_resume_and_distinct_scenes():
    payload = model_plan(scenes=["Arrival", "Departure"], constraints=["Same coat"],
                         refs=[{"ref_id": "submitted:1", "roles": ["subject", "clothing"]}])
    payload["output_count"] = 2
    payload["output_kind"] = "series"
    intent = gen_prompt.parse_intent_plan(payload, source_prompt="two images", user_aspect="3:4")
    series = gen_series.series_plan_from_intent(intent)
    assert len(series.pages) == 2
    assert "Arrival" in series.pages[0].prompt
    assert "Departure" in series.pages[1].prompt
    assert all(page.refs == ("submitted:1",) and page.aspect == "3:4" for page in series.pages)
    assert all("Same coat" in page.prompt for page in series.pages)
    assert series.plan_id == gen_series.series_plan_from_intent(intent).plan_id
    generated, delivered = [], []

    def generate(**kwargs):
        generated.append(kwargs["page"].number)
        return f"artifact-{kwargs['page'].number}"

    state = gen_series.SeriesCheckpoint(series.plan_id, completed={1: "artifact-1"}, delivered={1})
    state = asyncio.run(gen_series.execute_series(
        series, generate=generate, checkpoint=state,
        deliver=lambda page, result: delivered.append(page.number)))
    assert generated == delivered == [2]
    assert state.delivered == {1, 2}


def test_panels_do_not_expand_into_files_and_fallback_count_is_explicit():
    intent = gen_prompt.parse_intent_plan(model_plan(panels_per_image=3), source_prompt="comic")
    assert len(gen_series.series_plan_from_intent(intent).pages) == 1
    fallback = gen_prompt.parse_intent_plan("bad", source_prompt="original", output_count=3)
    series = gen_series.series_plan_from_intent(fallback)
    assert len(series.pages) == 3
    assert all("original" in page.prompt for page in series.pages)


def test_legacy_series_parser_still_accepts_saved_plans():
    plan = gen_series.parse_multipage_plan("PAGE 1: Arrival\nPAGE 2: Departure", 2)
    assert [page.prompt for page in plan.pages] == ["Arrival", "Departure"]


def test_integration_signature_layers_and_legacy_footer_fallback():
    payload = model_plan(layers=True)
    plan = gen_prompt.parse_intent_plan(payload, "source", explicit_count=2, edit_mode=True)
    assert plan.output_kind == "independent_images"
    assert plan.task_type == "edit"
    assert plan.layers
    state = gen_prompt.intent_plan_to_dict(plan)
    assert json.loads(json.dumps(state))["output_count"] == 2
    fallback = gen_prompt.parse_intent_plan(
        "legacy prose footer", "original", fallback_prompt="Parsed legacy scene",
        fallback_refs=[{"ref_id": "submitted:1", "role": "subject"}], edit_mode=True)
    assert fallback.fallback_used and not fallback.layers
    assert fallback.prompt == "Parsed legacy scene"
    assert fallback.refs[0]["roles"] == ["subject"]
    assert "original" in gen_prompt.render_intent_prompt(fallback)


def test_plan_round_trip_uses_legacy_checkpoint_identity():
    payload = model_plan(scenes=["Arrival", "Departure"])
    payload.update(output_kind="series", output_count=2)
    intent = gen_prompt.parse_intent_plan(payload, "  two scenes\n")
    series = gen_series.series_plan_from_intent(intent)
    saved = json.dumps({"pages": [page.__dict__ for page in series.pages],
                        "bible": series.bible, "version": series.version})
    reloaded = gen_series.parse_multipage_plan(saved, 2)
    assert reloaded.plan_id == series.plan_id
    state = gen_series.SeriesCheckpoint(series.plan_id, completed={1: b"image"}, delivered={1})
    resumed = gen_series.SeriesCheckpoint.from_json(state.to_json(), plan=reloaded)
    assert resumed.completed == state.completed
    assert resumed.delivered == {1}


def test_independent_images_do_not_get_story_continuity():
    payload = model_plan()
    payload.update(output_kind="independent_images", output_count=2)
    plan = gen_prompt.parse_intent_plan(payload, "two unrelated pictures")
    assert "Series:" not in gen_prompt.intent_instructions(plan)
    assert gen_series.series_plan_from_intent(plan).bible == {}


@pytest.mark.parametrize("changes", [{"layers": "true"}, {"participant_count": True},
                                         {"output_kind": "single_image", "output_count": 2}])
def test_additional_schema_guards(changes):
    payload = model_plan()
    payload.update(changes)
    assert gen_prompt.parse_intent_plan(payload, "source").fallback_used


def test_valid_plan_does_not_call_repair():
    plan = gen_prompt.parse_intent_plan(model_plan(), "source",
                                        repair=lambda *_: pytest.fail("extra model call"))
    assert not plan.fallback_used


def test_explicit_parameters_replace_invalid_model_values():
    payload = model_plan()
    payload.update(output_count=False, aspect="bad", task_type="bad")
    plan = gen_prompt.parse_intent_plan(payload, "source", explicit_count=2,
                                        edit_mode=True, user_aspect="1:1")
    assert not plan.fallback_used
    assert (plan.output_count, plan.aspect, plan.task_type) == (2, "1:1", "edit")


def test_unusable_model_count_is_not_salvaged_or_inferred():
    payload = model_plan()
    payload.update(output_count=7, prompt=None, output_kind="series")
    plan = gen_prompt.parse_intent_plan(payload, "seven images")
    assert plan.fallback_used and plan.output_count == 1


def test_layers_are_model_controlled_and_single_scene_is_rendered():
    plan = gen_prompt.parse_intent_plan(model_plan(scenes=["A specific action"]), "use layers")
    assert not plan.layers
    assert "Layers:" not in gen_prompt.intent_instructions(plan)
    assert "A specific action" in gen_prompt.render_intent_prompt(plan)
    payload = model_plan(layers=True)
    plan = gen_prompt.parse_intent_plan(payload, "source")
    assert "Layers:" in gen_prompt.intent_instructions(plan)


def test_diagnostic_refs_are_detached():
    plan = gen_prompt.parse_intent_plan(model_plan(refs=[{"ref_id": "source:1", "roles": ["subject"]}]), "source")
    data = gen_prompt.intent_plan_to_dict(plan)
    data["refs"][0]["roles"].append("style")
    assert plan.refs[0]["roles"] == ["subject"]

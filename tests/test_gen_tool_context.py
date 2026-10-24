"""Offline integration checks for the real planner loop and command routing."""
import asyncio
import base64
from collections import Counter
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from PIL import Image
import pytest

import bot_new as bot
from gen_runtime import CURRENT, Invocation


def plan_response(**changes):
    plan = {"task_type": "creation", "output_kind": "single_image", "output_count": 1,
            "prompt": "A fox beside a blue door", "explicit_text": [], "refs": [],
            "aspect": "1:1", "constraints": [], "layers": False}
    plan.update(changes)
    return response(json.dumps(plan, ensure_ascii=False))


def response(content=None, calls=()):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=content, tool_calls=list(calls)))])


def tool_call(name, arguments, call_id):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps(arguments)))


def png_bytes(color):
    stream = BytesIO()
    Image.new("RGB", (32, 32), color).save(stream, "PNG")
    return stream.getvalue()


def image_urls(message):
    content = message.get("content")
    return [part["image_url"]["url"] for part in content if part.get("type") == "image_url"] if isinstance(content, list) else []


@pytest.fixture
def planner(monkeypatch):
    replies, requests = [], []

    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        assert replies, "unexpected additional planner call"
        return replies.pop(0)

    create_mock = Mock(side_effect=create)
    llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
    invocation = Invocation(job_id="intent-context-offline")
    token = CURRENT.set(invocation)
    monkeypatch.setattr(bot, "get_gen_planner_model", lambda: (llm, "offline-planner", "test"))
    monkeypatch.setattr(bot, "get_gen_prompt_model", lambda: (llm, "offline-planner", "test"))
    monkeypatch.setattr(bot, "openrouter_client", llm)
    monkeypatch.setattr(bot, "active_model_supports_vision", lambda: True)
    monkeypatch.setattr(bot, "MODEL_TOOLS_SUPPORT", {})
    monkeypatch.setattr(bot, "log", Mock())
    monkeypatch.setattr(bot, "tavily_api_key", "")
    monkeypatch.setattr(bot, "client", SimpleNamespace(
        get_messages=AsyncMock(side_effect=AssertionError("unmocked Telegram lookup"))))
    monkeypatch.setattr(bot, "_run_chat_search", AsyncMock(return_value="No candidates"))
    monkeypatch.setattr(bot, "_run_chat_ant_search", AsyncMock(return_value="No candidates"))
    monkeypatch.setattr(bot, "_run_chat_read_context", AsyncMock(return_value="No context"))
    monkeypatch.setattr(bot, "_run_chat_inspect_image", AsyncMock(return_value="Observed blue clothing; text unreadable."))
    monkeypatch.setattr(bot, "_gen_is_own_generation", lambda message: False)
    monkeypatch.setattr(bot, "_is_attached_photo", lambda message: True)
    monkeypatch.setattr(bot, "_downscale_img", AsyncMock(side_effect=lambda raw: raw))
    monkeypatch.setattr(bot, "_label_for", lambda *args: "test participant")
    try:
        yield SimpleNamespace(replies=replies, requests=requests, create=create_mock, invocation=invocation)
    finally:
        CURRENT.reset(token)


def test_valid_plan_uses_one_model_call_and_records_renderable_intent(planner):
    planner.replies.append(plan_response(
        explicit_text=["Привет!"], refs=[{"ref_id": "catalog:1", "roles": ["subject", "clothing"]}]))
    catalog = [{"idx": 1, "mid": 11, "bytes": png_bytes("blue"), "caption": "unverified caption"}]
    result = asyncio.run(bot._build_gen_prompt('Keep the exact sign "Привет!"', catalog=catalog, chat_id=99))
    assert planner.create.call_count == 1
    assert not planner.replies
    assert set(result[1]) == {(1, "subject"), (1, "clothing")}
    assert "Привет!" in result[0]
    assert not result[0].lstrip().startswith("{")
    assert result[3] == "1:1"
    assert not planner.invocation.timings["intent_plan"]["fallback_used"]
    assert planner.invocation.timings["planner_calls"] == 1
    system = planner.requests[0]["messages"][0]["content"]
    assert "output_count" in system and "panels_per_image" in system
    assert "inspect" in system and "message_caption" in system


@pytest.mark.parametrize("malformed", ["{bad JSON", "```json\n{bad JSON\n```", "not JSON at all"],
                         ids=["open-json", "fenced-json", "non-json"])
def test_bad_json_gets_exactly_one_schema_repair(planner, malformed):
    planner.replies.extend([response(malformed), plan_response(prompt="Repaired fox scene")])
    result = asyncio.run(bot._build_gen_prompt("original fox scene", catalog=[]))
    assert planner.create.call_count == 2
    assert not planner.replies
    assert "Repaired fox scene" in result[0]
    assert not planner.invocation.timings["intent_plan"]["fallback_used"]
    repair_request = planner.requests[1]
    assert "schema" in repair_request["messages"][-1]["content"].lower()
    assert "original fox scene" in json.dumps(repair_request["messages"])


def test_failed_schema_repair_stops_and_preserves_source(planner):
    planner.replies.extend([response('{"output_count": false}'), response("{still invalid")])
    source = '  Keep image #1 and the exact text "Не три страницы!"\n'
    result = asyncio.run(bot._build_gen_prompt(source, catalog=[]))
    assert planner.create.call_count == 2
    assert source in result[0]
    assert planner.invocation.timings["intent_plan"]["fallback_used"]
    assert planner.invocation.timings["intent_plan"]["output_count"] == 1


def test_repair_call_is_included_in_planner_call_telemetry(planner):
    planner.replies.extend([response("{bad JSON"), plan_response()])
    asyncio.run(bot._build_gen_prompt("fox", catalog=[]))
    assert planner.invocation.timings["planner_calls"] == planner.create.call_count == 2


def test_legacy_footer_fallback_does_not_add_extraction_call(planner):
    planner.replies.append(response("IDEA: fox\nASPECT: 1:1\nREFS:\nPROMPT: A fox beside a blue door"))
    result = asyncio.run(bot._build_gen_prompt("fox", catalog=[]))
    assert planner.create.call_count == 1
    assert result[2:] == ("fox", "1:1")
    assert "A fox beside a blue door" in result[0]


def test_tool_failure_is_returned_to_agent_and_planning_continues(planner, monkeypatch):
    search = AsyncMock(side_effect=RuntimeError("private provider response"))
    monkeypatch.setattr(bot, "_run_chat_search", search)
    planner.replies.extend([
        response(calls=[tool_call("chat_search", {"query": "fox"}, "failed-search")]),
        plan_response(prompt="Fox from the original description"),
    ])
    result = asyncio.run(bot._build_gen_prompt("fox", catalog=[], chat_id=99, msg_by_id={}))
    search.assert_awaited_once()
    assert planner.create.call_count == 2
    feedback = [message for message in planner.requests[1]["messages"] if message["role"] == "tool"]
    assert len(feedback) == 1 and feedback[0]["tool_call_id"] == "failed-search"
    assert "RuntimeError" in feedback[0]["content"]
    assert "private provider response" not in feedback[0]["content"]
    assert "Fox from the original description" in result[0]
    assert not planner.invocation.timings["intent_plan"]["fallback_used"]


def inspected_message(number, color):
    return SimpleNamespace(id=number, sender_id=7, raw_text="caption is not OCR", sender=None,
                           date=None, download_media=AsyncMock(return_value=png_bytes(color)))


def test_new_user_turns_add_only_new_images_without_repeating_submitted_refs(planner):
    submitted = base64.b64encode(png_bytes("red")).decode("ascii")
    catalog = [{"idx": 1, "mid": 11, "bytes": png_bytes("blue")}]
    messages = {22: inspected_message(22, "green"), 33: inspected_message(33, "yellow")}
    planner.replies.extend([
        response(calls=[tool_call("chat_inspect_image", {"message_id": 22}, "inspect-22")]),
        response(calls=[tool_call("chat_inspect_image", {"message_id": 33}, "inspect-33")]),
        plan_response(refs=[{"ref_id": "catalog:3", "roles": ["subject"]}]),
    ])
    result = asyncio.run(bot._build_gen_prompt("fox", catalog=catalog, chat_id=99,
                                              initial_refs=[submitted], msg_by_id=messages))
    assert planner.create.call_count == 3
    turns = [message for message in planner.requests[-1]["messages"] if message["role"] == "user"]
    assert [len(image_urls(message)) for message in turns] == [2, 1, 1]
    counts = Counter(url for message in turns for url in image_urls(message))
    assert len(counts) == 4 and set(counts.values()) == {1}
    assert planner.invocation.timings["planner_image_parts"] == 4
    assert result[1] == [(3, "subject")]
    assert all(message.download_media.await_count == 1 for message in messages.values())


def test_separate_builder_invocations_do_not_reuse_previous_images(planner):
    planner.replies.extend([plan_response(), plan_response()])
    asyncio.run(bot._build_gen_prompt("first", catalog=[{"idx": 1, "mid": 11, "bytes": png_bytes("red")}]))
    asyncio.run(bot._build_gen_prompt("second", catalog=[]))
    assert sum(len(image_urls(message)) for message in planner.requests[0]["messages"]) == 1
    assert not any(image_urls(message) for message in planner.requests[1]["messages"])


def test_inspected_cap_plus_four_candidates_remain_selectable(planner, monkeypatch):
    cap = 2
    monkeypatch.setattr(bot, "GEN_CTX_IMG_MAX", cap)
    catalog = [{"idx": 1, "mid": 11, "bytes": png_bytes("red")},
               {"idx": 2, "mid": 12, "bytes": png_bytes("blue")}]
    discovered = {number: inspected_message(number, color) for number, color in
                  [(21, "green"), (22, "yellow"), (23, "purple"), (24, "orange")]}
    for number in discovered:
        planner.replies.append(response(calls=[tool_call("chat_inspect_image", {"message_id": number}, f"inspect-{number}")]))
    planner.replies.append(plan_response(refs=[{"ref_id": "catalog:6", "roles": ["subject", "clothing"]}]))
    result = asyncio.run(bot._build_gen_prompt("use the last inspected source", catalog=catalog,
                                              chat_id=99, msg_by_id=discovered))
    assert len(catalog) == cap + 4
    assert {item["mid"] for item in catalog} == {11, 12, *discovered}
    assert bot._run_chat_inspect_image.await_count == 4
    assert set(result[1]) == {(cap + 4, "subject"), (cap + 4, "clothing")}
    assert planner.create.call_count == 5
    images = [url for message in planner.requests[-1]["messages"] for url in image_urls(message)]
    assert len(images) == len(set(images)) <= cap
    latest = base64.b64encode(png_bytes("orange")).decode("ascii")
    assert any(url.endswith(latest) for url in images)


@pytest.mark.parametrize("source,kind,count,panels,flags,expected", [
    ("Создай 3 отдельных изображения", "independent_images", 3, None, "", 3),
    ("Комикс с 3 панелями на одном листе", "single_image", 1, 3, "", 1),
    ("Не делай комикс на три страницы, нужен один постер", "single_image", 1, None, "", 1),
    ("A short visual story", "series", 2, None, "", 2),
    ("A fox", "single_image", 1, None, "-p3", 3),
], ids=["independent-images", "panels-one-file", "negated-series", "continuous-series", "explicit-count"])
def test_real_handler_routes_model_file_count_with_safe_doubles(
        planner, monkeypatch, source, kind, count, panels, flags, expected):
    status = SimpleNamespace(edit=AsyncMock(), delete=AsyncMock())
    event = SimpleNamespace(out=True, sender_id=77, chat_id=9001, id=501, reply_to=None,
                            sender=SimpleNamespace(first_name="Owner"), delete=AsyncMock(),
                            get_reply_message=AsyncMock(return_value=None), reply=AsyncMock(),
                            pattern_match=SimpleNamespace(group=lambda number: {1: "", 2: flags, 3: "", 4: source}[number]))
    monkeypatch.setattr(bot, "openrouter_client", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=planner.create))))
    monkeypatch.setattr(bot, "client", SimpleNamespace(send_message=AsyncMock(return_value=status)))
    monkeypatch.setattr(bot, "_slash_for_other_bot", AsyncMock(return_value=False))
    monkeypatch.setattr(bot, "_gen_fetch_link_refs", AsyncMock(side_effect=lambda event, prompt: (prompt, [], 0)))
    monkeypatch.setattr(bot, "_gen_collect_input_images", AsyncMock(return_value=([], 0)))
    monkeypatch.setattr(bot, "_get_topic_id", lambda event: None)
    monkeypatch.setattr(bot, "_is_real_reply", lambda event: (False, None))
    monkeypatch.setattr(bot, "_owner_label", lambda: "owner")
    monkeypatch.setattr(bot, "_build_multipage_prompts", AsyncMock(side_effect=AssertionError("second planner must not run")))
    artifacts, saves = {}, []

    def save_artifact(key, raw, suffix):
        path = Path("/offline-memory") / (key + suffix)
        artifacts[str(path)] = raw
        return path

    def save_checkpoint(path, payload):
        saves.append(deepcopy(payload))

    monkeypatch.setattr("gen_runtime.ARTIFACTS", SimpleNamespace(save=save_artifact))
    monkeypatch.setattr("gen_jobs.JobStore", lambda path: SimpleNamespace(path=SimpleNamespace(chmod=Mock())))
    monkeypatch.setattr("gen_series.save_series_state", save_checkpoint)
    monkeypatch.setattr(Path, "read_bytes", lambda path: artifacts[str(path)])
    images = iter([png_bytes("red"), png_bytes("blue"), png_bytes("green")])

    async def render(prompt, *args):
        return next(images), "image/png", prompt, False

    render_mock = AsyncMock(side_effect=render)
    delivery = AsyncMock(return_value=SimpleNamespace(id=700))
    monkeypatch.setattr(bot, "_gen_render_image", render_mock)
    monkeypatch.setattr(bot, "_gen_send_image", delivery)
    planner.replies.append(plan_response(output_kind=kind, output_count=count, panels_per_image=panels))
    asyncio.run(bot.gen_command.__wrapped__(event))
    assert planner.create.call_count == 1
    assert render_mock.await_count == expected, status.edit.await_args_list
    assert delivery.await_count == expected
    assert planner.invocation.timings["intent_plan"]["output_count"] == expected
    assert planner.invocation.timings["intent_plan"]["panels_per_image"] == panels
    bot._build_multipage_prompts.assert_not_awaited()
    assert status.delete.await_count == 1
    if expected > 1:
        checkpoint = json.loads(saves[-1]["checkpoint"])
        assert set(checkpoint["delivered"]) == set(range(1, expected + 1))
        assert len(artifacts) == expected
        if kind == "independent_images":
            assert all(call.args[1] == [] for call in render_mock.await_args_list)
        elif kind == "series":
            assert render_mock.await_args_list[1].args[1]

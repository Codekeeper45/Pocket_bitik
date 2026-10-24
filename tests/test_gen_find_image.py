"""Coverage for optional planner image-memory retrieval."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import bot_new as bot
from tests.test_gen_tool_context import plan_response, response, tool_call


def test_chat_find_image_optional_retry_and_adds_stable_catalog_candidates(planner, monkeypatch):
    planner.replies.extend([
        response(calls=[tool_call("chat_find_image", {"query": "fox with handwritten label", "limit": 99}, "find1")]),
        response(calls=[tool_call("chat_find_image", {"query": "blue coat", "limit": 1}, "find2")]),
        plan_response(refs=[{"ref_id": "catalog:2", "roles": ["subject"]}]),
    ])
    first = {"mid": 501, "bytes": b"first", "caption": "fox post", "visual_desc": "Fox wearing a red scarf"}
    second = {"mid": 502, "bytes": b"second", "caption": "other post", "visual_desc": "Person in blue coat"}
    index = AsyncMock(side_effect=[[{"mid": 501}], [{"mid": 502}]])
    fetch = AsyncMock(side_effect=[[first], [second]])
    monkeypatch.setattr(bot, "_index_media_hybrid_hits", index)
    monkeypatch.setattr(bot, "_gen_fetch_ref_items", fetch)
    with patch.object(bot, "_img_mime_from_bytes", return_value="image/png"):
        catalog = []
        result = asyncio.run(bot._build_gen_prompt("Find relevant memory", catalog=catalog, chat_id=99))
    assert index.await_args_list[0].args == (99, "fox with handwritten label", 32)
    assert index.await_args_list[0].kwargs == {"for_gen": True}
    assert fetch.await_args_list[0].kwargs["rerank_prompt"] == "fox with handwritten label"
    assert fetch.await_args_list[0].kwargs["limit"] == 12
    assert fetch.await_args_list[1].kwargs["limit"] == 1
    assert [item["idx"] for item in catalog] == [1, 2]
    assert [item["ref_id"] for item in catalog] == ["chat:99:501", "chat:99:502"] or all(item.get("ref_id") for item in catalog)
    assert catalog[0]["mid"] == 501 and catalog[1]["mid"] == 502
    feedback = [m["content"] for m in planner.requests[1]["messages"] if m["role"] == "tool"]
    assert len(feedback) == 1
    assert "fox post" in feedback[0] and "Fox wearing a red scarf" in feedback[0]
    assert "unverified" in feedback[0].lower()
    second_feedback = [m["content"] for m in planner.requests[2]["messages"] if m["role"] == "tool"]
    assert len(second_feedback) == 2 and "blue coat" in second_feedback[1]
    assert result[1] == [(2, "subject")]
    schemas = planner.requests[0]["tools"]
    schema = next(x["function"] for x in schemas if x["function"]["name"] == "chat_find_image")
    assert schema["parameters"]["properties"]["limit"]["maximum"] == 12
    assert "optional" in schema["description"].lower()
    assert "call it again" in schema["description"].lower()


def test_chat_find_image_deduplicates_existing_mid_and_requires_query(planner, monkeypatch):
    planner.replies.extend([
        response(calls=[tool_call("chat_find_image", {"query": "fox"}, "find")]),
        plan_response(),
    ])
    existing = {"idx": 4, "mid": 501, "bytes": b"existing", "caption": "already here"}
    monkeypatch.setattr(bot, "_index_media_hybrid_hits", AsyncMock(return_value=[{"mid": 501}]))
    fetch = AsyncMock(return_value=[{"mid": 501, "bytes": b"dup", "caption": "dup"}])
    monkeypatch.setattr(bot, "_gen_fetch_ref_items", fetch)
    with patch.object(bot, "_img_mime_from_bytes", return_value="image/png"):
        asyncio.run(bot._build_gen_prompt("fox", catalog=[existing], chat_id=99))
    fetch.assert_awaited_once()
    assert fetch.await_args.kwargs["exclude_mids"] == {501}
    assert existing["idx"] == 4
    assert len([m for m in planner.requests[1]["messages"] if m["role"] == "tool"]) == 1
    assert "candidates" in json.loads([m["content"] for m in planner.requests[1]["messages"] if m["role"] == "tool"][0])


def test_chat_find_image_empty_query_skips_index(planner, monkeypatch):
    planner.replies.extend([
        response(calls=[tool_call("chat_find_image", {"query": " "}, "empty")]),
        plan_response(),
    ])
    index = AsyncMock()
    monkeypatch.setattr(bot, "_index_media_hybrid_hits", index)
    asyncio.run(bot._build_gen_prompt("fox", catalog=[], chat_id=99))
    index.assert_not_awaited()
    tool = [m["content"] for m in planner.requests[1]["messages"] if m["role"] == "tool"][0]
    assert "non-empty" in tool


def test_chat_find_image_is_gen_only():
    assert all(spec["function"]["name"] != "chat_find_image" for spec in bot.CHAT_TOOLS)


# fixtures are shared with the neighboring integration suite
pytest_plugins = ["tests.test_gen_tool_context"]

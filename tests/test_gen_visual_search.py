import pytest

from gen_visual_search import extract_query, rank_description, search_visual_descriptions, tokenize


def test_tokenize_and_russian_inflection_prefix():
    assert "rabbit" in tokenize("A rabbit, quietly")
    r = rank_description("серое платье", "Серая зайчиха в розовом платье")
    assert r["score"] > 0
    assert r["visual_provenance"] == "visual_desc"


def test_long_instruction_removes_common_words_and_phrase_hits():
    q = extract_query('Please find the picture where a small red bird sits on a branch')
    assert "please" not in q["tokens"]
    result = rank_description(q, "small red bird perched on branch")
    assert result["score"] > 0.4
    assert "small red" in result["phrase_hits"]


def test_quoted_caption_is_separate_not_visual_score():
    q = extract_query('Find the photo captioned "summer picnic" with a blue blanket')
    a = rank_description(q, "a dog running through snow", "Summer Picnic")
    b = rank_description(q, "a dog running through snow", "other words")
    assert a["score"] == b["score"] == 0
    assert a["caption_provenance"] == "caption"
    assert b["caption_provenance"] is None


def test_irrelevant_query_has_zero_score():
    assert rank_description("volcano telescope", "a quiet garden with flowers")["score"] == 0


@pytest.mark.asyncio
async def test_async_search_bounded_parameterized_and_media_hits_shape():
    calls = []
    rows = [
        {"chat_id": 4, "msg_id": 9, "file_uid": "abc", "visual_description": "a tiny green turtle crossing sand", "image_description": None},
        {"chat_id": 4, "msg_id": 8, "file_uid": "def", "visual_description": "winter forest", "image_description": None},
    ]

    async def read(sql, params):
        calls.append((sql, params))
        return rows

    hits = await search_visual_descriptions(read, 4, "green turtle", limit=5)
    assert len(calls) == 1
    assert "visual_description IS NOT NULL" in calls[0][0]
    assert calls[0][1] == (4, 10000)
    assert len(hits) == 1
    assert hits[0]["source"] == "visual_lexical"
    assert hits[0]["score"] > 0
    assert hits[0]["diagnostics"]["visual_provenance"] == "visual_desc"


def test_no_query_terms_never_returns_matches():
    assert rank_description("show me the picture", "picture of a fox")["score"] == 0

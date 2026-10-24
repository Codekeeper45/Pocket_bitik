"""Generic lexical retrieval over persisted visual descriptions.

This supplements (never replaces) vector retrieval. Caption text is tracked as
separate evidence and is never included in visual-description scores.
"""
from __future__ import annotations

import re
import shlex
from collections import Counter
from typing import Any, Callable, Iterable, Mapping

MAX_ROWS = 10_000

# Small multilingual function-word list; content vocabulary remains language agnostic.
_STOPWORDS = set("a an and are as at be by for from in into is it of on or that the this to with what which who where when find show me image picture photo pic please looking search about have has wearing holding near against over under рядом около изображение фото картинка покажи найти ищу где что кто который которая которое это вот мне пожалуйста на в во и или с со из для по под над между у к от за как есть был была были быть его её их он она они оно это тот та те такой такая такие".split())
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase Unicode words, retaining only searchable content words."""
    return [t for t in _TOKEN.findall((text or "").casefold()) if len(t) > 1 and t not in _STOPWORDS]


def extract_query(text: str) -> dict[str, Any]:
    """Return lexical terms and exact quoted-caption candidates, kept separate."""
    quoted = []
    for m in re.finditer(r'"([^"\n]{2,})"|«([^»\n]{2,})»|“([^”\n]{2,})”', text or ""):
        phrase = next(g for g in m.groups() if g is not None).strip()
        if phrase:
            quoted.append(phrase)
    unquoted = re.sub(r'"[^"\n]*"|«[^»\n]*»|“[^”\n]*”', " ", text or "")
    tokens = tokenize(unquoted)
    return {"tokens": tokens, "phrases": _ngrams(tokens, 2, 4), "caption_candidates": quoted}


def _ngrams(tokens: list[str], low: int = 2, high: int = 3) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for n in range(high, low-1, -1) for i in range(max(0, len(tokens)-n+1))]


def _stem(token: str) -> str:
    # Deliberately rough suffix trimming supports common Russian/English variants.
    suffixes = ("иями", "ями", "ами", "ого", "ему", "ыми", "ими", "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ой", "ом", "ем", "ах", "ях", "ов", "ев", "ей", "ам", "ям", "ы", "и", "а", "я", "у", "ю", "е", "о", "ь", "s", "es")
    for suffix in suffixes:
        if len(token) - len(suffix) >= 3 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


def rank_description(query: str | Mapping[str, Any], visual_description: str, caption: str = "") -> dict[str, Any]:
    """Score visual description only; report caption matches as distinct evidence."""
    q = extract_query(query) if isinstance(query, str) else dict(query)
    qtokens = list(dict.fromkeys(q.get("tokens", [])))
    dtokens = tokenize(visual_description or "")
    dset = set(dtokens)
    stems = {_stem(t) for t in dtokens}
    matched = [t for t in qtokens if t in dset or _stem(t) in stems or any((len(_stem(t)) >= 4 and (_stem(t).startswith(s) or s.startswith(_stem(t)))) for s in stems)]
    # Coverage is primary; adjacent query n-grams add modest phrase evidence.
    coverage = len(matched) / max(1, len(qtokens))
    qgrams = _ngrams(qtokens, 2, 3)
    dgrams = set(_ngrams(dtokens, 2, 3))
    phrase_hits = [g for g in qgrams if g in dgrams]
    score = min(1.0, coverage * 0.82 + min(0.18, 0.06 * len(phrase_hits))) if qtokens else 0.0
    caption_candidates = q.get("caption_candidates", [])
    caption_hits = [p for p in caption_candidates if p.casefold() in (caption or "").casefold()]
    return {"score": round(score, 6), "matched_terms": matched, "phrase_hits": [" ".join(g) for g in phrase_hits], "visual_provenance": "visual_desc", "caption_provenance": "caption" if caption_hits else None, "caption_matches": caption_hits}


async def search_visual_descriptions(db_read: Callable[..., Any], chat_id: int, query: str, *, limit: int = 30) -> list[dict[str, Any]]:
    """Read at most 10k descriptions and return media_hits-compatible lexical matches."""
    rows = await db_read(
        "SELECT chat_id, msg_id, file_uid, visual_description, image_description FROM media_assets "
        "WHERE chat_id=%s AND visual_description IS NOT NULL ORDER BY msg_id DESC LIMIT %s",
        (chat_id, MAX_ROWS),
    )
    hits = []
    q = extract_query(query)
    for row in rows or []:
        desc = row.get("visual_description") or ""
        result = rank_description(q, desc, row.get("image_description") or "")
        if result["score"] <= 0:
            continue
        hits.append({"chat_id": row.get("chat_id", chat_id), "msg_id": row.get("msg_id"), "file_uid": row.get("file_uid"), "visual_description": desc, "source": "visual_lexical", "score": result["score"], "diagnostics": result})
    hits.sort(key=lambda h: (h["score"], h.get("msg_id") or 0), reverse=True)
    return hits[:max(0, min(limit, MAX_ROWS))]


# Alias convenient for callers integrating with existing media search code.
async def visual_lexical_search(db_read: Callable[..., Any], chat_id: int, query: str, *, limit: int = 30) -> list[dict[str, Any]]:
    return await search_visual_descriptions(db_read, chat_id, query, limit=limit)

__all__ = ["tokenize", "extract_query", "rank_description", "search_visual_descriptions", "visual_lexical_search", "MAX_ROWS"]

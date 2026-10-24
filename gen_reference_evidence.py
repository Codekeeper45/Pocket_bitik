"""Pure, auditable evidence helpers for generation reference retrieval."""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any, Mapping

_SECRET_KEYS = re.compile(r"(token|password|secret|api[_-]?key|authorization|cookie)", re.I)
_SAFE_ARGS = {"query", "message_id", "limit", "offset", "question", "decision_only", "scope"}

@dataclass(frozen=True)
class EvidenceFailure:
    code: str
    message: str
    verified: bool = False

@dataclass(frozen=True)
class QueryIntent:
    kind: str
    query: str
    exact: bool = False
    negations: tuple[str, ...] = ()

def selected_args(args: Any) -> dict[str, Any]:
    """Return allowlisted scalar arguments only; arbitrary and secret fields are omitted."""
    if not isinstance(args, Mapping): return {}
    return {str(k): v for k, v in args.items() if str(k) in _SAFE_ARGS and not _SECRET_KEYS.search(str(k)) and isinstance(v, (str, int, float, bool, type(None)))}

def _source_ids(value: Any) -> list[str]:
    found = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in {"message_id", "source_id", "ref_id", "mid"} and child is not None: found.append(str(child))
            else: found.extend(_source_ids(child))
    elif isinstance(value, (list, tuple)):
        for child in value: found.extend(_source_ids(child))
    return list(dict.fromkeys(found))

def retrieval_trace(tool_name: str, args: Any, result: Any, *, status: str = "ok", preview_chars: int = 400) -> dict[str, Any]:
    """Structured retrieval record with bounded preview and extracted source IDs."""
    text = result if isinstance(result, str) else repr(result)
    return {"tool": str(tool_name), "args": selected_args(args), "status": str(status), "source_ids": _source_ids(result), "result_preview": text[:max(0, int(preview_chars))]}

def evidence_record(*, source_id: Any, caption: str = "", visual_result: Any = None, visual_status: str = "not_inspected") -> dict[str, Any]:
    """Keep caption evidence distinct from successful evidence derived from pixels."""
    visual = visual_result if visual_status == "ok" and isinstance(visual_result, str) and visual_result.strip() else None
    return {"source_id": str(source_id) if source_id is not None else None, "caption_evidence": caption or "", "visual_evidence": visual, "visual_status": str(visual_status), "verified": visual is not None}

def require_visual_evidence(record: Mapping[str, Any]) -> EvidenceFailure | None:
    """Typed failure prevents unverified inputs being described as visually verified."""
    if not isinstance(record, Mapping) or not record.get("source_id"):
        return EvidenceFailure("source_missing", "No source ID was established.")
    if record.get("visual_status") != "ok" or not record.get("visual_evidence"):
        return EvidenceFailure("visual_evidence_unavailable", "Source pixels were not successfully inspected.")
    return None

def extract_query_intent(text: str) -> QueryIntent:
    """Classify caption quotes vs explicit pixel-text requests; preserve Russian negation tokens."""
    text = str(text or "")
    quoted = re.search(r'[«“\"]([^»”\"]+)[»”\"]', text)
    neg = tuple(dict.fromkeys(re.findall(r"(?iu)\b(?:не|нет|без|нельзя|ни)\b", text)))
    pixel = bool(re.search(r"(?iu)(?:на\s+)?(?:сам(?:ом|ой)?\s+)?(?:фото|картинк\w*|изображени\w*|пиксел\w*).{0,70}(?:текст|надпис\w*|написан\w*)|(?:текст|надпис\w*).{0,70}(?:на\s+фото|на\s+картинк\w*|в\s+пиксел\w*|видим\w*)", text))
    if pixel: return QueryIntent("pixel_text", quoted.group(1) if quoted else text.strip(), True, neg)
    if quoted: return QueryIntent("caption_quote", quoted.group(1), False, neg)
    return QueryIntent("general", text.strip(), False, neg)

def verified_source_claim(record: Mapping[str, Any]) -> bool:
    return require_visual_evidence(record) is None

__all__ = ["EvidenceFailure", "QueryIntent", "selected_args", "retrieval_trace", "evidence_record", "require_visual_evidence", "extract_query_intent", "verified_source_claim"]

if __name__ == "__main__":
    import unittest
    unittest.main(module="test_gen_reference_evidence", verbosity=0)

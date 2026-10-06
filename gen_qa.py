"""QA finding validation and paired candidate acceptance."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable
from gen_contracts import QAResult

CATEGORIES = {"fidelity", "technical", "anatomy", "composition", "identity", "text", "seam", "style", "other"}
SEVERITIES = {"low", "medium", "high", "critical"}

@dataclass(frozen=True)
class Acceptance:
    accepted: bool
    reason: str
    target_improved: bool = False

def parse_findings(payload: Any, *, limit: int = 12) -> QAResult:
    """Validate structured QA payload; malformed/empty responses are never clean."""
    if limit < 1: raise ValueError("limit must be positive")
    if isinstance(payload, str):
        import json
        try: payload = json.loads(payload)
        except Exception: return QAResult("unavailable", error="malformed QA JSON")
    if not isinstance(payload, dict) or not isinstance(payload.get("findings"), list): return QAResult("unavailable", error="invalid QA schema")
    if set(payload) - {'findings', 'status'}: return QAResult('unavailable', error='unknown QA fields')
    findings = []
    for item in payload["findings"]:
        if not isinstance(item, dict): return QAResult("unavailable", error="invalid finding")
        required = {"id", "category", "severity", "confidence", "bbox", "affected_subject", "requirement_id", "description", "location"}
        if not required.issubset(item): return QAResult("unavailable", error="finding missing required fields")
        if set(item) - required: return QAResult('unavailable', error='unknown finding fields')
        if type(item['severity']) is not str or type(item['category']) is not str:
            return QAResult('unavailable', error='invalid enum')
        if any(f['id'] == item['id'] for f in findings):
            return QAResult('unavailable', error='duplicate finding id')
        if payload.get('status') not in (None, 'pass', 'warnings', 'fail', 'unavailable'):
            return QAResult('unavailable', error='invalid QA status')
        if not isinstance(item["id"], str) or not item["id"].strip() or not isinstance(item["description"], str) or not item["description"].strip() or not isinstance(item["location"], str) or not item["location"].strip(): return QAResult("unavailable", error="invalid finding text fields")
        if item["affected_subject"] is not None and not isinstance(item["affected_subject"], str): return QAResult("unavailable", error="invalid affected_subject")
        if item["requirement_id"] is not None and not isinstance(item["requirement_id"], str): return QAResult("unavailable", error="invalid requirement_id")
        if item["category"] not in CATEGORIES or item["severity"] not in SEVERITIES: return QAResult("unavailable", error="unknown finding category/severity")
        if not isinstance(item["bbox"], (list, tuple)) and item["bbox"] is not None: return QAResult("unavailable", error="invalid bbox")
        if item["bbox"] is None and not (item["category"] in {"fidelity", "composition", "other"} and item["location"].lower() in {"global", "всё изображение", "вся сцена"}): return QAResult("unavailable", error="localized finding requires bbox")
        if item["bbox"] is not None and len(item["bbox"]) == 4 and not item["location"]: return QAResult("unavailable", error="bbox requires location")
        if type(item["severity"]) is not str or type(item["category"]) is not str: return QAResult("unavailable", error="invalid enum")

        conf = item["confidence"]
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1: return QAResult("unavailable", error="invalid confidence")
        box = item["bbox"]
        if box is not None:
            if not isinstance(box, (list, tuple)) or len(box) != 4 or any(isinstance(x, bool) or not isinstance(x,(int,float)) for x in box) or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1): return QAResult("unavailable", error="invalid bbox")
        findings.append(dict(item))
    overflow = len(findings) > limit
    findings = tuple(findings)  # Overflow is flagged, never hides a later critical finding.
    if payload.get("status") == "unavailable": status = "unavailable"
    elif any(x["severity"] in {"high", "critical"} for x in findings): status = "fail"
    elif findings or overflow: status = "warnings"
    else: status = "pass"
    return QAResult(status, findings, overflow)

def _high(findings: Iterable[dict]) -> set[tuple]:
    return {(f.get("category"), f.get("affected_subject"), f.get("requirement_id")) for f in findings if f.get("severity") in {"high", "critical"}}

def accept_repair(before: QAResult, after: QAResult, *, target_finding_id: str, protected_requirements: Iterable[str] = ()) -> Acceptance:
    """Require actual target improvement and reject newly severe or protected changes."""
    if before.status == "unavailable" or after.status == "unavailable": return Acceptance(False, "QA unavailable")
    target_before = [f for f in before.findings if f.get("id") == target_finding_id]
    target_after = [f for f in after.findings if f.get("id") == target_finding_id]
    if not target_before: return Acceptance(False, "target finding absent from baseline")
    if target_after and ['low', 'medium', 'high', 'critical'].index(target_after[0].get("severity")) >= ['low', 'medium', 'high', 'critical'].index(target_before[0].get("severity")): return Acceptance(False, "target finding not reduced")
    if _high(after.findings) - _high(before.findings): return Acceptance(False, "new high/critical finding")
    protected = set(protected_requirements)
    if any(f.get("requirement_id") in protected for f in after.findings if f.get("severity") in {"high", "critical"}): return Acceptance(False, "protected requirement affected")
    return Acceptance(True, "target improved without new severe findings", True)

__all__ = ["Acceptance", "parse_findings", "accept_repair"]

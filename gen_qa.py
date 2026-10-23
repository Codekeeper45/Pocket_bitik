"""QA finding validation and paired candidate acceptance."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable
from gen_contracts import QAResult

CATEGORIES = {"fidelity", "technical", "anatomy", "composition", "identity", "text", "seam", "style", "other"}
SEVERITIES = {"low", "medium", "high", "critical"}

@dataclass(frozen=True)
class Acceptance:
    accepted: bool
    reason: str
    target_improved: bool = False

def parse_findings(payload: Any, *, limit: int = 12, image_bytes: bytes | None = None) -> QAResult:
    """Validate model output and optionally bind it to the bytes actually inspected."""
    result = _parse_findings(payload, limit=limit)
    if image_bytes is None:
        return result
    from dataclasses import replace
    return replace(result, image_digest=sha256(image_bytes).hexdigest())


def qa_for_image(result: QAResult | None, image_bytes: bytes) -> QAResult:
    """Never reuse a baseline assessment for different or unbound bytes."""
    digest = sha256(image_bytes).hexdigest()
    if result is None or result.image_digest != digest:
        return QAResult('unavailable', error='QA not bound to these image bytes', image_digest=digest)
    return result


def _parse_findings(payload: Any, *, limit: int = 12) -> QAResult:
    """Validate structured QA payload; malformed/empty responses are never clean."""
    if limit < 1: raise ValueError("limit must be positive")
    if isinstance(payload, str):
        import json
        try: payload = json.loads(payload)
        except Exception: return QAResult("unavailable", error="malformed QA JSON")
    if not isinstance(payload, dict) or not isinstance(payload.get("findings"), list): return QAResult("unavailable", error="invalid QA schema")
    if set(payload) - {'findings', 'status'}: return QAResult('unavailable', error='unknown QA fields')
    status = payload.get('status')
    if 'status' in payload and status not in ('pass', 'warnings', 'fail', 'unavailable'):
        return QAResult('unavailable', error='invalid QA status')
    if status == 'unavailable':
        return QAResult('unavailable', error='QA reported unavailable')
    if status in ('warnings', 'fail') and not payload['findings']:
        return QAResult('unavailable', error='QA status requires findings')
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
    if payload.get("status") == "fail": status = "fail"
    elif any(x["severity"] in {"high", "critical"} for x in findings): status = "fail"
    elif findings or overflow: status = "warnings"
    else: status = "pass"
    return QAResult(status, findings, overflow)

def _finding_key(finding):
    return (finding.get('category'), finding.get('affected_subject'), finding.get('requirement_id'),
            tuple(finding['bbox']) if finding.get('bbox') is not None else finding.get('location'))


def _severity(finding):
    return ('low', 'medium', 'high', 'critical').index(finding['severity'])


def _same_target(finding, target):
    if finding['id'] == target['id']:
        return True
    if any(finding[key] != target[key] for key in ('category', 'affected_subject', 'requirement_id')):
        return False
    first_box, second_box = finding['bbox'], target['bbox']
    if first_box is None or second_box is None:
        return first_box is second_box and finding['location'] == target['location']
    overlap = max(0, min(first_box[2], second_box[2]) - max(first_box[0], second_box[0])) * max(
        0, min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]))
    first_area = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
    second_area = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])
    return overlap / (first_area + second_area - overlap) >= .5

def accept_repair(before: QAResult, after: QAResult, *, target_finding_id: str, protected_requirements: Iterable[str] = (), before_image: bytes | None = None, after_image: bytes | None = None) -> Acceptance:
    """Require actual target improvement and reject newly severe or protected changes."""
    if before_image is not None: before = qa_for_image(before, before_image)
    if after_image is not None: after = qa_for_image(after, after_image)
    if before.status == "unavailable" or after.status == "unavailable": return Acceptance(False, "QA unavailable")
    if before.error or after.error: return Acceptance(False, 'QA unavailable')
    if before.overflow or after.overflow: return Acceptance(False, 'QA findings overflow')
    for result in (before, after):
        validated = parse_findings({'status': result.status, 'findings': list(result.findings)},
                                   limit=max(12, len(result.findings)))
        if validated.status == 'unavailable': return Acceptance(False, 'invalid QA evidence')
    target_before = [f for f in before.findings if f.get("id") == target_finding_id]
    if not target_before: return Acceptance(False, "target finding absent from baseline")
    target = target_before[0]
    target_after = [finding for finding in after.findings if _same_target(finding, target)]
    if any(_severity(finding) >= _severity(target) for finding in target_after):
        return Acceptance(False, 'target finding not reduced')
    for finding in after.findings:
        previous = [item for item in before.findings if _finding_key(item) == _finding_key(finding)]
        if _severity(finding) >= 2 and (not previous or _severity(finding) > max(map(_severity, previous))):
            return Acceptance(False, 'new or worsened high/critical finding')
    protected = set(protected_requirements)
    if any(f.get("requirement_id") in protected for f in after.findings if f.get("severity") in {"high", "critical"}): return Acceptance(False, "protected requirement affected")
    return Acceptance(True, "target improved without new severe findings", True)

__all__ = ["Acceptance", "parse_findings", "qa_for_image", "accept_repair"]

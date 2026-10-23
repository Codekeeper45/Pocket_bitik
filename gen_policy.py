"""Bounded automatic generation/QA/repair policy and explicit accounting."""
from dataclasses import dataclass

MEDIUM_TARGET_CATEGORIES = {'fidelity', 'identity', 'text', 'anatomy', 'technical', 'seam'}

class BudgetExceeded(RuntimeError): pass

@dataclass
class GenerationBudget:
    max_generation_calls: int = 2
    max_qa_calls: int = 4
    max_repair_calls: int = 2
    max_wall_seconds: float = 300.0
    max_refs: int = 8
    max_repair_area: float = 0.25
    generation_calls: int = 0
    qa_calls: int = 0
    repair_calls: int = 0
    repair_area: float = 0.0
    def __post_init__(self):
        if min(self.max_generation_calls,self.max_qa_calls,self.max_repair_calls,self.max_wall_seconds,self.max_refs,self.max_repair_area) < 0: raise ValueError("budget limits must be nonnegative")
    def consume(self, kind: str, count: int = 1):
        if type(count) is not int or count < 0: raise ValueError("count must be a nonnegative integer")
        field, maximum = {"generation":("generation_calls","max_generation_calls"),"qa":("qa_calls","max_qa_calls"),"repair":("repair_calls","max_repair_calls")}.get(kind,(None,None))
        if field is None: raise ValueError("unknown budget kind")
        if getattr(self,field)+count > getattr(self,maximum): raise BudgetExceeded(f"{kind} call budget exhausted")
        setattr(self,field,getattr(self,field)+count)
    def add_repair_area(self, fraction: float):
        if isinstance(fraction,bool) or not 0 <= fraction <= 1: raise ValueError("repair area fraction must be in [0,1]")
        if self.repair_area + fraction > self.max_repair_area: raise BudgetExceeded("repair area budget exhausted")
        self.repair_area += fraction
    def check_refs(self, count: int):
        if count < 0 or count > self.max_refs: raise BudgetExceeded("reference budget exhausted")
    def remaining_calls(self, kind: str):
        if kind not in ('generation', 'qa', 'repair'): raise ValueError('unknown budget kind')
        return max(0, getattr(self, 'max_' + kind + '_calls') - getattr(self, kind + '_calls'))


def select_repair_targets(qa, *, limit=3, min_confidence=.75):
    """Select significant, evidenced findings, never words in their descriptions."""
    from gen_contracts import QAResult
    from gen_qa import parse_findings
    if type(limit) is not int or limit < 1: raise ValueError('limit must be a positive integer')
    if type(min_confidence) not in (int, float) or not 0 <= min_confidence <= 1:
        raise ValueError('invalid confidence threshold')
    if not isinstance(qa, QAResult) or qa.status == 'unavailable' or qa.error or qa.overflow:
        return ()
    validated = parse_findings({'status': qa.status, 'findings': list(qa.findings)}, limit=max(12, len(qa.findings)))
    if validated.status == 'unavailable': return ()
    targets = [finding for finding in validated.findings if finding['confidence'] >= min_confidence and
               (finding['severity'] in ('high', 'critical') or
                (finding['severity'] == 'medium' and finding['category'] in MEDIUM_TARGET_CATEGORIES))]
    ranks = {'critical': 3, 'high': 2, 'medium': 1, 'low': 0}
    return tuple(sorted(targets, key=lambda finding: (ranks[finding['severity']], finding['confidence']), reverse=True)[:limit])


def repair_instructions(targets, *, immutable_requirements=()):
    """Carry selected evidence and original constraints into a repair/regeneration."""
    import json
    return 'Correct only these selected defects; preserve the original request and immutable requirements.\n' + json.dumps(
        {'targets': list(targets), 'immutable_requirements': list(immutable_requirements)}, ensure_ascii=False)

def action_for(findings):
    """Return conservative action label from structured findings."""
    items = list(findings)
    severe = [f for f in items if (type(f.get('confidence')) in (int, float) and .75 <= f['confidence'] <= 1) and (f.get("severity") in {"high","critical"} or
              (f.get('category') in MEDIUM_TARGET_CATEGORIES and f.get('severity') == 'medium'))]
    if not severe: return "keep"
    local = all(f.get("bbox") is not None and f.get('category') in {'anatomy','technical','other','text','seam'} for f in severe)
    return "repair" if local else "regenerate_once"

__all__=["GenerationBudget","BudgetExceeded","action_for", "select_repair_targets", "repair_instructions"]

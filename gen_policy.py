"""Bounded automatic generation/QA/repair policy and explicit accounting."""
from dataclasses import dataclass

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
        if isinstance(count,bool) or count < 0: raise ValueError("count must be nonnegative")
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

def action_for(findings):
    """Return conservative action label from structured findings."""
    items = list(findings)
    severe = [f for f in items if f.get("severity") in {"high","critical"} or
              (f.get('category') in {'fidelity','identity','text'} and f.get('severity') == 'medium' and f.get('confidence',0) >= .75)]
    if not severe: return "keep"
    local = all(f.get("bbox") is not None and f.get('category') in {'anatomy','technical','other'} for f in severe)
    return "repair" if local else "regenerate_once"

__all__=["GenerationBudget","BudgetExceeded","action_for"]

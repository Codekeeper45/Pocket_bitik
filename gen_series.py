"""Strict multipage plans and dependency-injected resumable series execution."""
from __future__ import annotations
from dataclasses import dataclass, field
from hashlib import sha256
import json
import re

class SeriesPlanError(ValueError): pass

@dataclass(frozen=True)
class SeriesPage:
    number: int
    prompt: str
    aspect: str | None = None
    refs: tuple[str, ...] = ()

@dataclass(frozen=True)
class SeriesPlan:
    pages: tuple[SeriesPage, ...]
    bible: dict = field(default_factory=dict)
    version: str = "1"
    plan_id: str = ""


def parse_multipage_plan(output: str, pages_count: int, *, user_aspect=None, repair=None) -> SeriesPlan:
    if type(pages_count) is not int or pages_count < 1: raise SeriesPlanError("pages_count must be positive")
    def parse(text):
        text = (text or "").strip()
        if not text: raise SeriesPlanError("empty plan")
        # Accept a structured JSON form or explicit PAGE N blocks, not legacy duplicated fallback.
        try:
            obj = json.loads(text)
        except json.JSONDecodeError: obj = None
        if obj is not None:
            if not isinstance(obj, dict) or set(obj) - {"pages", "bible", "version"} or not isinstance(obj.get("pages"), list): raise SeriesPlanError("invalid structured plan")
            raw_pages, bible, version = obj["pages"], obj.get("bible", {}), str(obj.get("version", "1"))
        else:
            matches = list(re.finditer(r"(?im)^\s*PAGE\s+(\d+)\s*:\s*", text))
            if not matches or text[:matches[0].start()].strip(): raise SeriesPlanError("expected PAGE N blocks")
            raw_pages, bible, version = [], {}, "1"
            for i, m in enumerate(matches):
                raw_pages.append({"number": int(m.group(1)), "prompt": text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)].strip()})
        if len(raw_pages) != pages_count: raise SeriesPlanError(f"expected exactly {pages_count} pages, got {len(raw_pages)}")
        pages=[]; seen=set()
        for item in raw_pages:
            if not isinstance(item, dict) or type(item.get("number")) is not int: raise SeriesPlanError("each page needs an integer number")
            n=item["number"]; p=item.get("prompt")
            if n in seen: raise SeriesPlanError(f"duplicate page {n}")
            if not isinstance(p,str) or not p.strip(): raise SeriesPlanError(f"page {n} prompt is empty")
            normalized=" ".join(p.split()).casefold()
            if normalized in {" ".join(x.prompt.split()).casefold() for x in pages}: raise SeriesPlanError("duplicate page story prompt")
            aspect = user_aspect or item.get("aspect")
            if aspect not in (None,"1:1","9:16","16:9","4:3","3:4"): raise SeriesPlanError(f"invalid aspect on page {n}")
            refs=item.get("refs",[])
            if not isinstance(refs,list) or any(not isinstance(r,str) or not r for r in refs): raise SeriesPlanError(f"invalid refs on page {n}")
            pages.append(SeriesPage(n,p.strip(),aspect,tuple(dict.fromkeys(refs)))); seen.add(n)
        if seen != set(range(1,pages_count+1)): raise SeriesPlanError("pages must be numbered 1..N without gaps")
        if not isinstance(bible,dict) or any(not isinstance(k, str) for k in bible): raise SeriesPlanError("bible must be an object with string keys")
        # Snapshot canonical JSON data so downstream page generation cannot mutate
        # the planner-owned object and silently change series continuity.
        try:
            bible = json.loads(json.dumps(bible, ensure_ascii=False, sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise SeriesPlanError("bible must contain JSON-compatible values") from exc
        pages.sort(key=lambda p:p.number)
        canonical=json.dumps({"pages":[p.__dict__ for p in pages],"bible":bible,"version":version},sort_keys=True,ensure_ascii=False)
        return SeriesPlan(tuple(pages),dict(bible),version,sha256(canonical.encode()).hexdigest())
    try: return parse(output)
    except SeriesPlanError as original:
        if repair is None: raise
        try: fixed=repair(output, str(original))
        except Exception as e: raise SeriesPlanError(f"schema repair failed: {type(e).__name__}") from e
        try: return parse(fixed)
        except SeriesPlanError as e: raise SeriesPlanError(f"invalid plan after one repair: {e}") from e


@dataclass
class SeriesCheckpoint:
    plan_id: str
    completed: dict = field(default_factory=dict)
    failures: dict = field(default_factory=dict)
    delivered: set = field(default_factory=set)
    chat_id: str = ""
    owner_id: str = ""
    job_id: str = ""
    pages_meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize checkpoint metadata/artifact references; bytes become base64."""
        import base64
        def encode(v):
            if isinstance(v, bytes): return {"$bytes": base64.b64encode(v).decode("ascii")}
            if isinstance(v, dict): return {str(k): encode(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)): return [encode(x) for x in v]
            if v is None or type(v) in (str, int, float, bool): return v
            raise SeriesPlanError(f"unsupported checkpoint value: {type(v).__name__}")
        return json.dumps({"schema":1,"plan_id":self.plan_id,"completed":encode(self.completed),"failures":encode(self.failures),"delivered":sorted(self.delivered),"chat_id":self.chat_id,"owner_id":self.owner_id,"job_id":self.job_id,"pages_meta":self.pages_meta}, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str, *, plan: SeriesPlan):
        import base64
        def decode(v):
            if isinstance(v, dict) and set(v) == {"$bytes"}:
                try: return base64.b64decode(v["$bytes"], validate=True)
                except Exception as exc: raise SeriesPlanError("invalid checkpoint artifact encoding") from exc
            if isinstance(v, dict): return {k: decode(x) for k, x in v.items()}
            if isinstance(v, list): return [decode(x) for x in v]
            return v
        try: data=json.loads(raw)
        except Exception as exc: raise SeriesPlanError("invalid checkpoint JSON") from exc
        if not isinstance(data, dict) or data.get("schema") != 1 or data.get("plan_id") != plan.plan_id:
            raise SeriesPlanError("checkpoint schema/plan mismatch")
        completed=decode(data.get("completed",{})); failures=decode(data.get("failures",{}))
        valid={p.number for p in plan.pages}
        if not isinstance(completed,dict) or not isinstance(failures,dict): raise SeriesPlanError("invalid checkpoint maps")
        try:
            completed={int(k):v for k,v in completed.items()}; failures={int(k):v for k,v in failures.items()}
        except (TypeError, ValueError) as exc: raise SeriesPlanError("invalid checkpoint page number") from exc
        if not (set(completed)|set(failures)) <= valid: raise SeriesPlanError("checkpoint contains an unknown page")
        delivered=data.get("delivered", [])
        if not isinstance(delivered,list) or any(type(n) is not int for n in delivered) or not set(delivered) <= valid or not set(delivered) <= set(completed):
            raise SeriesPlanError("invalid checkpoint delivery markers")
        return cls(plan.plan_id, completed, failures, set(delivered), str(data.get("chat_id", "")), str(data.get("owner_id", "")), str(data.get("job_id", "")), data.get("pages_meta", {}))

async def execute_series(plan: SeriesPlan, *, generate, checkpoint: SeriesCheckpoint | None = None, save_checkpoint=None, deliver=None, max_input_bytes=None, anchor_selector=None):
    """Run missing pages only. Persist generated artifacts before delivery and resume delivery separately."""
    import inspect
    state=checkpoint or SeriesCheckpoint(plan.plan_id)
    if state.plan_id != plan.plan_id: raise SeriesPlanError("checkpoint belongs to a different immutable plan")
    results=dict(state.completed)
    anchor=anchor_selector(results) if anchor_selector and results else None
    for page in plan.pages:
        if page.number in results:
            state.failures.pop(page.number, None)
            if deliver and page.number not in state.delivered:
                try:
                    v=deliver(page, results[page.number])
                    if inspect.isawaitable(v): await v
                    state.delivered.add(page.number)
                    if save_checkpoint:
                        v=save_checkpoint(state)
                        if inspect.isawaitable(v): await v
                except Exception as exc:
                    state.failures[page.number]=f"delivery {type(exc).__name__}"
                    if save_checkpoint:
                        v=save_checkpoint(state)
                        if inspect.isawaitable(v): await v
                    return state
            continue
        try:
            kwargs={"page":page,"bible":plan.bible,"anchor":anchor,"max_input_bytes":max_input_bytes}
            result=generate(**kwargs)
            if inspect.isawaitable(result): result=await result
            results[page.number]=result
            state.completed=dict(results); state.failures.pop(page.number,None)
            if anchor is None and anchor_selector: anchor=anchor_selector(results)
            if save_checkpoint:
                v=save_checkpoint(state)
                if inspect.isawaitable(v): await v
            if deliver:
                v=deliver(page,result)
                if inspect.isawaitable(v): await v
                state.delivered.add(page.number)
                if save_checkpoint:
                    v=save_checkpoint(state)
                    if inspect.isawaitable(v): await v
        except Exception as exc:
            phase = 'delivery ' if page.number in results else 'generation '
            state.completed=dict(results); state.failures[page.number]=f"{phase}{type(exc).__name__}"
            if save_checkpoint:
                v=save_checkpoint(state)
                if inspect.isawaitable(v): await v
            return state
    return state

__all__=["SeriesPlanError","SeriesPage","SeriesPlan","parse_multipage_plan","SeriesCheckpoint","execute_series"]

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
        if not isinstance(bible,dict): raise SeriesPlanError("bible must be an object")
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

async def execute_series(plan: SeriesPlan, *, generate, checkpoint: SeriesCheckpoint | None = None, save_checkpoint=None, deliver=None, max_input_bytes=None, anchor_selector=None):
    """Run missing pages only. Callbacks are async-compatible and fully injected."""
    import inspect
    state=checkpoint or SeriesCheckpoint(plan.plan_id)
    if state.plan_id != plan.plan_id: raise SeriesPlanError("checkpoint belongs to a different immutable plan")
    results=dict(state.completed)
    anchor=anchor_selector(results) if anchor_selector and results else None
    for page in plan.pages:
        if page.number in results: continue
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
        except Exception as exc:
            state.completed=dict(results); state.failures[page.number]=f"{type(exc).__name__}: {exc}"
            if save_checkpoint:
                v=save_checkpoint(state)
                if inspect.isawaitable(v): await v
            return state
    return state

__all__=["SeriesPlanError","SeriesPage","SeriesPlan","parse_multipage_plan","SeriesCheckpoint","execute_series"]

"""Strict multipage plans and dependency-injected resumable series execution."""
from __future__ import annotations
from dataclasses import dataclass, field
from hashlib import sha256
from contextlib import asynccontextmanager
import asyncio
import json
import re

class SeriesPlanError(ValueError): pass


_SERIES_JOB_LOCKS = {}


@asynccontextmanager
async def series_job_lock(job_id: str):
    """Share a lock across initial execution and resume in the same event loop.

    Acquire using the original job ID BEFORE loading its checkpoint, and hold
    through reconciliation, generation, delivery and the final save. Execution
    does not acquire this non-reentrant lock itself. Separate processes require
    external coordination. Entries are removed once all holders/waiters leave.
    """
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
        raise SeriesPlanError("invalid series job id")
    key = (asyncio.get_running_loop(), job_id)
    entry = _SERIES_JOB_LOCKS.setdefault(key, [asyncio.Lock(), 0])
    entry[1] += 1
    try:
        async with entry[0]:
            yield
    finally:
        entry[1] -= 1
        if not entry[1]:
            del _SERIES_JOB_LOCKS[key]


def validate_series_job(record: dict, *, job_id: str, owner_id: str, chat_id: str) -> dict:
    """Validate identity-bound persisted state before a caller resumes or delivers it."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", str(job_id or "")):
        raise SeriesPlanError("invalid series job id")
    if not isinstance(record, dict) or record.get("job_id") != job_id:
        raise SeriesPlanError("series job id mismatch")
    if str(record.get("owner_id", "")) != str(owner_id) or str(record.get("chat_id", "")) != str(chat_id):
        raise SeriesPlanError("series job owner/chat mismatch")
    return record


def series_state_path(root, job_id: str):
    from pathlib import Path
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", str(job_id or "")):
        raise SeriesPlanError("invalid series job id")
    return Path(root) / "gen_series_state" / f"{job_id}.json"


def save_series_state(path, payload: dict) -> None:
    """Atomic mode-0600 JSON state file, isolated by validated job id."""
    import os, tempfile
    path = __import__('pathlib').Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_series_state(root, *, job_id: str, owner_id: str, chat_id: str) -> dict:
    path = series_state_path(root, job_id)
    try: record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc: raise SeriesPlanError("series job not found") from exc
    return validate_series_job(record, job_id=job_id, owner_id=owner_id, chat_id=chat_id)

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


def series_plan_from_intent(intent) -> SeriesPlan:
    """Adapt the single agent plan to the existing checkpoint/execution interface.

    Only output_count determines file count. Panels never expand into pages.
    Independent outputs carry no continuity bible. Raw plans use raw-1 metadata
    to preserve prompt whitespace and permit explicitly independent repeats.
    No planner/API calls occur.
    """
    from gen_prompt import IntentPlan, intent_plan_to_dict, render_intent_prompt
    if not isinstance(intent, IntentPlan):
        raise SeriesPlanError("expected an IntentPlan")
    pages = tuple(SeriesPage(number, render_intent_prompt(intent, output_number=number), intent.aspect,
                  tuple(ref['ref_id'] for ref in intent.refs if 'outputs' not in ref or number in ref['outputs']))
                  for number in range(1, intent.output_count + 1))
    data = intent_plan_to_dict(intent)
    bible = {"constraints": data["constraints"], "refs": data["refs"]} if intent.output_kind == "series" else {}
    version = "1"
    if intent.raw:
        if intent.output_kind not in ("single_image", "independent_images") or (intent.output_kind == "single_image" and intent.output_count != 1):
            raise SeriesPlanError("raw plans require single or independent images")
        bible = {"raw": True, "output_kind": intent.output_kind}
        version = "raw-1"
    canonical = json.dumps({"pages": [page.__dict__ for page in pages], "bible": bible, "version": version},
                           sort_keys=True, ensure_ascii=False)
    return SeriesPlan(pages, bible, version, sha256(canonical.encode()).hexdigest())


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
        if not isinstance(bible,dict) or any(not isinstance(key, str) for key in bible): raise SeriesPlanError("bible must be an object with string keys")
        raw_mode = version == "raw-1"
        if raw_mode and (bible.get("raw") is not True or bible.get("output_kind") not in ("single_image", "independent_images")
                         or (bible.get("output_kind") == "single_image" and pages_count != 1)):
            raise SeriesPlanError("invalid raw plan metadata")
        allow_duplicate_prompts = raw_mode and bible["output_kind"] == "independent_images"
        pages=[]; seen=set()
        for item in raw_pages:
            if not isinstance(item, dict) or type(item.get("number")) is not int: raise SeriesPlanError("each page needs an integer number")
            n=item["number"]; p=item.get("prompt")
            if n in seen: raise SeriesPlanError(f"duplicate page {n}")
            if not isinstance(p,str) or not p.strip(): raise SeriesPlanError(f"page {n} prompt is empty")
            normalized=" ".join(p.split()).casefold()
            if not allow_duplicate_prompts and normalized in {" ".join(x.prompt.split()).casefold() for x in pages}: raise SeriesPlanError("duplicate page story prompt")
            aspect = user_aspect or item.get("aspect")
            if aspect not in (None,"1:1","9:16","16:9","4:3","3:4"): raise SeriesPlanError(f"invalid aspect on page {n}")
            refs=item.get("refs",[])
            if not isinstance(refs,list) or any(not isinstance(r,str) or not r for r in refs): raise SeriesPlanError(f"invalid refs on page {n}")
            pages.append(SeriesPage(n,p if raw_mode else p.strip(),aspect,tuple(dict.fromkeys(refs)))); seen.add(n)
        if seen != set(range(1,pages_count+1)): raise SeriesPlanError("pages must be numbered 1..N without gaps")
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
    delivery_phases: dict = field(default_factory=dict)

    def delivery_phase(self, number: int) -> str:
        """Legacy completed pages without a phase are unknown, never unsent."""
        if number in self.delivered:
            return "delivered"
        return self.delivery_phases.get(number, "unknown")

    def to_json(self) -> str:
        """Serialize checkpoint metadata/artifact references; bytes become base64."""
        import base64
        def encode(v):
            if isinstance(v, bytes): return {"$bytes": base64.b64encode(v).decode("ascii")}
            if isinstance(v, dict): return {str(k): encode(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)): return [encode(x) for x in v]
            if v is None or type(v) in (str, int, float, bool): return v
            raise SeriesPlanError(f"unsupported checkpoint value: {type(v).__name__}")
        phases = dict(self.delivery_phases)
        phases.update({number: "delivered" for number in self.delivered})
        return json.dumps({"schema":1,"plan_id":self.plan_id,"completed":encode(self.completed),"failures":encode(self.failures),"delivered":sorted(self.delivered),"chat_id":self.chat_id,"owner_id":self.owner_id,"job_id":self.job_id,"pages_meta":self.pages_meta,"delivery_phases":encode(phases)}, sort_keys=True)

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
        pages_meta = data.get("pages_meta", {})
        if not isinstance(pages_meta, dict): raise SeriesPlanError("invalid page metadata")
        raw_phases = data.get("delivery_phases", {})
        if not isinstance(raw_phases, dict): raise SeriesPlanError("invalid delivery phases")
        phases = {}
        for number, phase in raw_phases.items():
            if not re.fullmatch(r"[1-9][0-9]*", number) or int(number) not in completed:
                raise SeriesPlanError("delivery phase requires a completed page")
            if phase not in ("not_started", "sending", "delivered"):
                raise SeriesPlanError("invalid delivery phase")
            number = int(number)
            if (phase == "delivered") != (number in delivered):
                raise SeriesPlanError("conflicting delivery phase and marker")
            phases[number] = phase
        phases.update({number: "delivered" for number in delivered})
        return cls(plan.plan_id, completed, failures, set(delivered), str(data.get("chat_id", "")), str(data.get("owner_id", "")), str(data.get("job_id", "")), pages_meta, phases)

async def execute_series(plan: SeriesPlan, *, generate, checkpoint: SeriesCheckpoint | None = None, save_checkpoint=None, deliver=None, max_input_bytes=None, anchor_selector=None, reconcile_delivery=None):
    """Run missing pages, persisting not_started -> sending -> delivered.

    Hold series_job_lock around load and this call. For sending/legacy unknown
    pages, reconcile_delivery(page, artifact) must return True (already sent),
    False (confirmed absent), or None (uncertain). No callback means no resend.
    Callbacks may be synchronous or asynchronous. Cancellation propagates;
    ordinary failures return a checkpoint with failures and preserved artifacts.
    """
    import inspect
    state=checkpoint or SeriesCheckpoint(plan.plan_id)
    if state.plan_id != plan.plan_id: raise SeriesPlanError("checkpoint belongs to a different immutable plan")
    async def invoke(callback, *args, **kwargs):
        value = callback(*args, **kwargs)
        return await value if inspect.isawaitable(value) else value

    async def persist():
        if save_checkpoint:
            await invoke(save_checkpoint, state)

    anchor=anchor_selector(state.completed) if anchor_selector and state.completed else None
    for page in plan.pages:
        stage = "generation"
        try:
            if page.number not in state.completed:
                result = await invoke(generate, page=page, bible=plan.bible, anchor=anchor, max_input_bytes=max_input_bytes)
                state.completed[page.number] = result
                state.delivery_phases[page.number] = "not_started"
                state.failures.pop(page.number, None)
                stage = "checkpoint"
                await persist()
                if anchor is None and anchor_selector:
                    anchor = anchor_selector(state.completed)
            result = state.completed[page.number]
            if page.number in state.delivered:
                retry_save = bool(state.failures.get(page.number))
                state.failures.pop(page.number, None)
                if retry_save:
                    stage = "checkpoint"
                    await persist()
                continue
            if not deliver:
                state.failures.pop(page.number, None)
                continue
            stage = "delivery reconciliation"
            phase = state.delivery_phase(page.number)
            if phase != "not_started":
                confirmed = await invoke(reconcile_delivery, page, result) if reconcile_delivery else None
                if confirmed is not True and confirmed is not False:
                    state.failures[page.number] = "delivery reconciliation required"
                    stage = "checkpoint"
                    await persist()
                    return state
                if confirmed is True:
                    state.delivered.add(page.number)
                    state.delivery_phases[page.number] = "delivered"
                    state.failures.pop(page.number, None)
                    stage = "checkpoint"
                    await persist()
                    continue
                state.delivery_phases[page.number] = "not_started"
                stage = "checkpoint"
                await persist()
            state.delivery_phases[page.number] = "sending"
            stage = "checkpoint"
            await persist()
            stage = "delivery"
            await invoke(deliver, page, result)
            state.delivered.add(page.number)
            state.delivery_phases[page.number] = "delivered"
            state.failures.pop(page.number, None)
            stage = "checkpoint"
            await persist()
        except Exception as exc:
            state.failures[page.number] = f"{stage} {type(exc).__name__}"
            if stage != "checkpoint":
                try:
                    await persist()
                except Exception as save_error:
                    state.failures[page.number] += f"; checkpoint {type(save_error).__name__}"
            return state
    return state

__all__=["SeriesPlanError","SeriesPage","SeriesPlan","parse_multipage_plan","series_plan_from_intent","SeriesCheckpoint","execute_series","series_job_lock","validate_series_job","series_state_path","save_series_state","load_series_state"]

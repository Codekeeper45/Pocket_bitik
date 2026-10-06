"""Process-local generation job coordinator with atomic recovery metadata.

No provider or Telegram dependency: callers inject work and delivery callbacks.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import tempfile
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional


@dataclass
class JobRecord:
    job_id: str
    chat_id: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    heartbeat_at: float = field(default_factory=time.time)
    boot_id: str = ""
    artifact_path: Optional[str] = None
    delivered: bool = False
    attempts: int = 0
    error: Optional[str] = None


class JobStore:
    """Atomic JSON snapshot store. Intended for small local job registries."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def load(self) -> dict[str, Any]:
        async with self._lock:
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {"boot_id": "", "jobs": {}}

    async def save(self, data: dict[str, Any]) -> None:
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, sort_keys=True)
                    f.flush(); os.fsync(f.fileno())
                os.replace(tmp, self.path)
                try:
                    dfd = os.open(self.path.parent, os.O_DIRECTORY)
                    try: os.fsync(dfd)
                    finally: os.close(dfd)
                except (AttributeError, OSError):
                    pass
            finally:
                if os.path.exists(tmp): os.unlink(tmp)


class GlobalRateGate:
    """Shared interval-based rate gate, independent of concurrency semaphore."""
    def __init__(self, max_concurrent: int = 2, min_interval: float = 0.0):
        if max_concurrent < 1 or min_interval < 0:
            raise ValueError("invalid rate gate limits")
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.min_interval = float(min_interval)
        self._lock = asyncio.Lock()
        self._next_at = 0.0
        self._inflight_threads: set[asyncio.Future] = set()

    async def enter_rate(self) -> None:
        async with self._lock:
            delay = max(0.0, self._next_at - time.monotonic())
            self._next_at = max(self._next_at, time.monotonic()) + self.min_interval
        if delay:
            await asyncio.sleep(delay)

    async def run(self, fn: Callable[..., Any], *args: Any, timeout: Optional[float] = None) -> Any:
        await self.semaphore.acquire()
        try:
            await self.enter_rate()
            if asyncio.iscoroutinefunction(fn):
                call = fn(*args)
                return await asyncio.wait_for(call, timeout) if timeout else await call
            # Shield the thread future: task cancellation/timeout does not release real capacity
            # until the blocking provider call actually exits.
            fut = asyncio.create_task(asyncio.to_thread(fn, *args))
            self._inflight_threads.add(fut)
            fut.add_done_callback(self._inflight_threads.discard)
            try:
                return await asyncio.wait_for(asyncio.shield(fut), timeout) if timeout else await asyncio.shield(fut)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                # Caller exits promptly; unfinished blocking work retains its slot.
                held_by_future = True
                def release_when_done(done):
                    self.semaphore.release()
                    if not done.cancelled():
                        done.exception()  # retrieve late provider error
                fut.add_done_callback(release_when_done)
                raise
        finally:
            if not locals().get('held_by_future', False):
                self.semaphore.release()

    @property
    def inflight_threads(self) -> int:
        return sum(not f.done() for f in self._inflight_threads)


class ContextBudget:
    def __init__(self, max_refs: int = 8, max_bytes: int = 20_000_000, max_calls: int = 8):
        if min(max_refs, max_bytes, max_calls) < 0: raise ValueError("budgets must be nonnegative")
        self.max_refs, self.max_bytes, self.max_calls = max_refs, max_bytes, max_calls
        self.refs = self.bytes = self.calls = 0

    def add_context(self, refs: int = 0, byte_count: int = 0) -> None:
        if refs < 0 or byte_count < 0 or self.refs + refs > self.max_refs or self.bytes + byte_count > self.max_bytes:
            raise BudgetExceeded("context budget exceeded")
        self.refs += refs; self.bytes += byte_count

    def consume_call(self, count: int = 1) -> None:
        if count < 0 or self.calls + count > self.max_calls: raise BudgetExceeded("call budget exceeded")
        self.calls += count


class BudgetExceeded(RuntimeError): pass


class ArtifactStore:
    def __init__(self, root: str | Path, ttl_seconds: float = 86400, max_bytes: int = 1_000_000_000):
        self.root, self.ttl_seconds, self.max_bytes = Path(root), ttl_seconds, max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, job_id: str, data: bytes, suffix: str = ".bin") -> Path:
        if not data or len(data) > self.max_bytes: raise ValueError("artifact size outside configured bounds")
        if not job_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in job_id) or '/' in suffix or '\\' in suffix:
            raise ValueError('unsafe artifact name')
        path = self.root / f"{job_id}{suffix}"
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".artifact-")
        try:
            with os.fdopen(fd, "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        self.prune()
        return path

    def prune(self, now: Optional[float] = None) -> list[Path]:
        now = time.time() if now is None else now
        files = sorted((p for p in self.root.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime)
        removed = []
        for p in files:
            if now - p.stat().st_mtime > self.ttl_seconds:
                p.unlink(missing_ok=True); removed.append(p)
        remaining = sorted((p for p in self.root.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in remaining)
        for p in remaining:
            if total <= self.max_bytes: break
            size = p.stat().st_size; p.unlink(missing_ok=True); total -= size; removed.append(p)
        return removed


class GenerationJobs:
    def __init__(self, *, store: JobStore, gate: GlobalRateGate, max_queue: int = 100,
                 default_deadline: float = 300, boot_id: Optional[str] = None):
        if max_queue < 1 or default_deadline <= 0: raise ValueError("invalid queue/deadline")
        self.store, self.gate = store, gate
        self.max_queue, self.default_deadline = max_queue, default_deadline
        self.boot_id = boot_id or uuid.uuid4().hex
        self.jobs: dict[str, JobRecord] = {}
        self._waiting: dict[str, deque[str]] = {}
        self._chat_rotation: deque[str] = deque()
        self._lock = asyncio.Lock()
        self._accepting = True
        self._active: set[asyncio.Task] = set()
        self._tasks: dict[str, asyncio.Task] = {}
        self._changed = asyncio.Event()
        self._started = False
        self._startup_lock = asyncio.Lock()
        self._reserved = 0

    async def startup(self) -> list[str]:
        async with self._startup_lock:
            return await self._startup_once()

    async def _startup_once(self) -> list[str]:
        if self._started:
            return []
        data = await self.store.load()
        old = data.get("jobs", {})
        interrupted = []
        for jid, raw in old.items():
            if raw.get("status") in {"queued", "running", "delivering"}:
                raw["status"] = "interrupted"; raw["error"] = "process restarted; explicit resume required"
                raw["updated_at"] = raw["heartbeat_at"] = time.time(); interrupted.append(jid)
            self.jobs[jid] = JobRecord(**{k: raw[k] for k in JobRecord.__dataclass_fields__ if k in raw})
        await self._persist()
        self._started = True
        return interrupted

    async def _persist(self) -> None:
        await self.store.save({"boot_id": self.boot_id, "jobs": {k: asdict(v) for k, v in self.jobs.items()}})

    async def submit(self, chat_id: str | int, work: Callable[..., Any], *args: Any,
                     artifact_store: Optional[ArtifactStore] = None, context_budget: Optional[ContextBudget] = None,
                     deadline: Optional[float] = None) -> JobRecord:
        if not self._started:
            await self.startup()
        chat = str(chat_id)
        async with self._lock:
            if not self._accepting: raise RuntimeError("job intake is draining")
            if sum(r.status in {'queued','running'} for r in self.jobs.values()) >= self.max_queue:
                raise RuntimeError("generation queue full")
            self._reserved += 1
            rec = JobRecord(job_id=uuid.uuid4().hex, chat_id=chat, boot_id=self.boot_id)
            self.jobs[rec.job_id] = rec
            q = self._waiting.setdefault(chat, deque()); q.append(rec.job_id)
            if chat not in self._chat_rotation: self._chat_rotation.append(chat)
            self._changed.set()
            await self._persist()
        task = asyncio.create_task(self._execute(rec, work, args, artifact_store, context_budget, deadline or self.default_deadline))
        self._active.add(task); task.add_done_callback(self._active.discard)
        self._tasks[rec.job_id] = task
        return rec

    async def _fair_turn(self, rec: JobRecord) -> None:
        # Round-robin by chat, preserving FIFO within each chat.
        while True:
            async with self._lock:
                q = self._waiting.get(rec.chat_id, deque())
                if q and q[0] == rec.job_id and self._chat_rotation and self._chat_rotation[0] == rec.chat_id:
                    self._chat_rotation.popleft()
                    q.popleft()
                    if q:
                        self._chat_rotation.append(rec.chat_id)
                    else: self._waiting.pop(rec.chat_id, None)
                    self._reserved -= 1
                    self._changed.set()
                    return
                self._changed.clear()
            await self._changed.wait()

    async def run_event(self, chat_id: str | int, work: Callable[..., Any], *args: Any, **kwargs: Any) -> JobRecord:
        """Enqueue and await one event's terminal record; suitable for handler hooks."""
        rec = await self.submit(chat_id, work, *args, **kwargs)
        task = self._tasks.get(rec.job_id)
        if task:
            await asyncio.shield(task)
        return rec

    async def _execute(self, rec: JobRecord, work: Callable[..., Any], args: tuple[Any, ...],
                       artifact_store: Optional[ArtifactStore], budget: Optional[ContextBudget], deadline: float) -> None:
        expires = time.monotonic() + deadline
        try:
            await asyncio.wait_for(self._fair_turn(rec), timeout=deadline)
            rec.status = "running"; rec.updated_at = rec.heartbeat_at = time.time(); rec.attempts += 1; await self._persist()
            if budget: budget.consume_call()
            remaining = max(.001, expires - time.monotonic())
            result = await asyncio.wait_for(self.gate.run(work, *args, timeout=remaining), timeout=remaining)
            if isinstance(result, bytes) and artifact_store:
                rec.artifact_path = str(artifact_store.save(rec.job_id, result))
            rec.status = "completed"
            from gen_runtime import CURRENT
            invocation = CURRENT.get()
            if invocation and invocation.result is not None:
                recorded_error = getattr(invocation.result, 'error', None)
                if recorded_error:
                    rec.status = 'failed'; rec.error = recorded_error.kind
        except asyncio.CancelledError:
            rec.status = "cancelled"; rec.error = "cancelled (provider thread may have completed)"; raise
        except Exception as exc:
            rec.status = "failed"; rec.error = type(exc).__name__
        finally:
            async with self._lock:
                q = self._waiting.get(rec.chat_id)
                if q and rec.job_id in q:
                    q.remove(rec.job_id)
                    self._reserved = max(0, self._reserved-1)
                    if not q:
                        self._waiting.pop(rec.chat_id, None)
                        try: self._chat_rotation.remove(rec.chat_id)
                        except ValueError: pass
                    self._changed.set()
            rec.updated_at = rec.heartbeat_at = time.time(); await self._persist()

    async def retry_delivery(self, job_id: str, deliver: Callable[[Path], Awaitable[Any] | Any], *, attempts: int = 3) -> Any:
        rec = self.jobs[job_id]
        if rec.status not in {"completed", "delivery_failed"} or not rec.artifact_path:
            raise RuntimeError("no saved artifact available for delivery")
        path = Path(rec.artifact_path)
        for n in range(max(1, attempts)):
            rec.status = "delivering"; rec.attempts += 1; rec.heartbeat_at = time.time(); await self._persist()
            try:
                value = deliver(path)
                if inspect.isawaitable(value): value = await value
                rec.status = "delivered"; rec.delivered = True; rec.error = None; rec.updated_at = time.time(); await self._persist()
                return value
            except Exception as exc:
                rec.status = "delivery_failed"; rec.error = type(exc).__name__; rec.updated_at = time.time(); await self._persist()
                if n + 1 == max(1, attempts): raise
                await asyncio.sleep(min(2 ** n, 8))

    async def cancel(self, job_id: str) -> bool:
        rec = self.jobs[job_id]
        if rec.status not in {"queued", "running"}: return False
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        async with self._lock:
            q = self._waiting.get(rec.chat_id)
            if q and job_id in q:
                q.remove(job_id); self._reserved = max(0, self._reserved - 1)
                if not q:
                    self._waiting.pop(rec.chat_id, None)
                    try: self._chat_rotation.remove(rec.chat_id)
                    except ValueError: pass
                self._changed.set()
        rec.status = "cancelled"; rec.updated_at = time.time(); await self._persist()
        return True

    async def drain(self, timeout: Optional[float] = None) -> bool:
        """Stop intake and wait for work. Returns false if deadline expires."""
        self._accepting = False
        try:
            waiter = asyncio.gather(*tuple(self._active), return_exceptions=True)
            await asyncio.wait_for(asyncio.shield(waiter), timeout) if timeout else await waiter
            return True
        except asyncio.TimeoutError:
            return False

    async def resume_intake(self) -> None: self._accepting = True

    def queue_position(self, job_id: str) -> Optional[int]:
        rec = self.jobs[job_id]
        q = self._waiting.get(rec.chat_id, ())
        try: return list(q).index(job_id) + 1
        except ValueError: return None

    def heartbeat_stale(self, job_id: str, *, max_age: float, now: Optional[float] = None) -> bool:
        rec = self.jobs[job_id]; return (time.time() if now is None else now) - rec.heartbeat_at > max_age

__all__ = ["ArtifactStore", "BudgetExceeded", "ContextBudget", "GenerationJobs", "GlobalRateGate", "JobRecord", "JobStore"]

import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path

from gen_jobs import ArtifactStore, BudgetExceeded, ContextBudget, GenerationJobs, GlobalRateGate, JobStore


class GenerationJobsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.jobs = GenerationJobs(store=JobStore(Path(self.tmp.name) / "jobs.json"), gate=GlobalRateGate(2), default_deadline=2)
        await self.jobs.startup()

    async def test_global_concurrency_limit(self):
        active = peak = 0; lock = threading.Lock()
        def work():
            nonlocal active, peak
            with lock: active += 1; peak = max(peak, active)
            time.sleep(.04)
            with lock: active -= 1
            return b"ok"
        await asyncio.gather(*(self.jobs.submit(str(i), work) for i in range(7)))
        self.assertTrue(await self.jobs.drain(3))
        self.assertLessEqual(peak, 2)
        self.assertTrue(all(j.status == "completed" for j in self.jobs.jobs.values()))

    async def test_budget_and_retained_artifact(self):
        b = ContextBudget(1, 10, 1); b.add_context(1, 10); b.consume_call()
        with self.assertRaises(BudgetExceeded): b.add_context(1, 0)
        store = ArtifactStore(Path(self.tmp.name) / "artifacts", ttl_seconds=50, max_bytes=20)
        rec = await self.jobs.submit("c", lambda: b"image bytes", artifact_store=store)
        await self.jobs.drain(2)
        self.assertEqual(Path(rec.artifact_path).read_bytes(), b"image bytes")

    async def test_startup_marks_inflight_interrupted(self):
        rec = await self.jobs.submit("c", lambda: b"x"); await self.jobs.drain(2)
        data = await self.jobs.store.load(); data["jobs"][rec.job_id]["status"] = "running"
        await self.jobs.store.save(data)
        restarted = GenerationJobs(store=self.jobs.store, gate=GlobalRateGate(), boot_id="new")
        self.assertEqual(await restarted.startup(), [rec.job_id])
        self.assertEqual(restarted.jobs[rec.job_id].status, "interrupted")

    async def test_rate_gate_and_drain(self):
        gate = GlobalRateGate(3, .025); times = []
        await asyncio.gather(*(gate.run(lambda: times.append(time.monotonic())) for _ in range(3)))
        times.sort(); self.assertGreaterEqual(times[1] - times[0], .015)
        self.assertTrue(await self.jobs.drain(1))
        with self.assertRaisesRegex(RuntimeError, "draining"): await self.jobs.submit("c", lambda: 1)

    async def test_chat_round_robin_fifo(self):
        gate = GlobalRateGate(1)
        jobs = GenerationJobs(store=JobStore(Path(self.tmp.name) / "fair.json"), gate=gate)
        order = []
        started = asyncio.Event()
        def work(label):
            order.append(label)
            if label == "a1": started_loop.call_soon_threadsafe(started.set)
            time.sleep(.01)
            return label
        started_loop = asyncio.get_running_loop()
        first = await jobs.submit("A", work, "a1")
        await started.wait()
        await jobs.submit("A", work, "a2")
        await jobs.submit("B", work, "b1")
        self.assertTrue(await jobs.drain(2))
        # FIFO within A is guaranteed; B arrives after A2 has already claimed
        # the next fair turn, so arrival interleaving is intentionally deterministic.
        self.assertEqual(order, ["a1", "a2", "b1"])
        self.assertEqual(first.status, "completed")

    async def test_run_event_lazily_starts_and_records_delivery_failure(self):
        from gen_jobs import GenerationJobs, JobStore, GlobalRateGate
        jobs = GenerationJobs(store=JobStore(Path(self.tmp.name) / "lazy.json"), gate=GlobalRateGate(1))
        rec = await jobs.run_event("chat", lambda: b"image", artifact_store=ArtifactStore(Path(self.tmp.name) / "art"))
        self.assertEqual(rec.status, "completed")
        async def fail_delivery(path): raise OSError("private payload must not persist")
        with self.assertRaises(OSError): await jobs.retry_delivery(rec.job_id, fail_delivery, attempts=1)
        data = await jobs.store.load()
        self.assertEqual(data["jobs"][rec.job_id]["status"], "delivery_failed")
        self.assertEqual(data["jobs"][rec.job_id]["error"], "OSError")

if __name__ == "__main__": unittest.main()

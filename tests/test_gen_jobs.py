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

if __name__ == "__main__": unittest.main()

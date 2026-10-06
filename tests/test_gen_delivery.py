import os
import tempfile
import time
import unittest
from pathlib import Path

from gen_jobs import ArtifactStore, GenerationJobs, GlobalRateGate, JobStore


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_delivery_retries_saved_artifact_without_regeneration(self):
        with tempfile.TemporaryDirectory() as d:
            artifacts = ArtifactStore(Path(d) / "artifacts")
            jobs = GenerationJobs(store=JobStore(Path(d) / "jobs.json"), gate=GlobalRateGate())
            await jobs.startup(); generated = []; sent = []
            rec = await jobs.submit("chat", lambda: (generated.append(1), b"fixture-output-bytes")[1], artifact_store=artifacts)
            await jobs.drain(2)
            def deliver(path):
                sent.append(path.read_bytes())
                if len(sent) == 1: raise TimeoutError()
                return "accepted"
            self.assertEqual(await jobs.retry_delivery(rec.job_id, deliver, attempts=2), "accepted")
            self.assertEqual(generated, [1])
            self.assertEqual(sent, [b"fixture-output-bytes"] * 2)
            self.assertTrue(rec.delivered)

    async def test_ttl_and_quota_retention(self):
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(d, ttl_seconds=1, max_bytes=5)
            old = store.save("old", b"123")
            os.utime(old, (time.time() - 10, time.time() - 10)); store.prune()
            self.assertFalse(old.exists())
            first = store.save("one", b"12345"); store.save("two", b"abcde")
            self.assertFalse(first.exists())

if __name__ == "__main__": unittest.main()

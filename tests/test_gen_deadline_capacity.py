import asyncio
import threading
import unittest

from gen_jobs import GlobalRateGate


class DeadlineCapacity(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_returns_but_real_slot_stays_busy(self):
        gate = GlobalRateGate(1)
        started = threading.Event()
        release = threading.Event()
        done = threading.Event()
        second_started = threading.Event()

        def work():
            started.set()
            try:
                if not release.wait(5):
                    raise AssertionError('test watchdog: worker never released')
            finally:
                done.set()

        # Warm this loop's executor before testing provider deadlines. Thread-pool
        # startup under a full-suite load is not the provider cancellation contract.
        await asyncio.to_thread(lambda: None)
        first = asyncio.create_task(gate.run(work, timeout=.05))
        next_call = None
        try:
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(first), timeout=1)
            # Distinguish gate deadline from the outer watchdog timing out.
            self.assertTrue(first.done())
            self.assertIsInstance(first.exception(), asyncio.TimeoutError)
            self.assertTrue(started.is_set())
            self.assertFalse(done.is_set())
            self.assertEqual(gate.inflight_threads, 1)
            next_call = asyncio.create_task(gate.run(second_started.set))
            await asyncio.sleep(.025)
            self.assertFalse(second_started.is_set())
            self.assertEqual(gate.inflight_threads, 1)
            release.set()
            await asyncio.wait_for(next_call, timeout=1)
            self.assertTrue(done.is_set())
            self.assertTrue(second_started.is_set())
            self.assertEqual(gate.inflight_threads, 0)
        finally:
            release.set()
            await asyncio.gather(first, *([next_call] if next_call else []), return_exceptions=True)

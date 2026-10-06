import asyncio,time,unittest,threading
from gen_jobs import GlobalRateGate
class DeadlineCapacity(unittest.IsolatedAsyncioTestCase):
 async def test_deadline_returns_but_real_slot_stays_busy(self):
  gate=GlobalRateGate(1);done=threading.Event();second_started=threading.Event()
  def work():time.sleep(.12);done.set()
  t=time.monotonic()
  with self.assertRaises(asyncio.TimeoutError):await gate.run(work,timeout=.015)
  self.assertLess(time.monotonic()-t,.08)
  next_call=asyncio.create_task(gate.run(lambda:second_started.set()))
  await asyncio.sleep(.025);self.assertFalse(second_started.is_set());self.assertEqual(gate.inflight_threads,1)
  await next_call;self.assertTrue(done.is_set());self.assertTrue(second_started.is_set())

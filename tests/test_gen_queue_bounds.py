import asyncio,tempfile,unittest
from pathlib import Path
from gen_jobs import GenerationJobs,JobStore,GlobalRateGate
class QueueBounds(unittest.IsolatedAsyncioTestCase):
 async def test_running_jobs_count_toward_capacity(self):
  with tempfile.TemporaryDirectory() as d:
   jobs=GenerationJobs(store=JobStore(Path(d)/'jobs.json'),gate=GlobalRateGate(1),max_queue=1)
   release=asyncio.Event()
   async def work():await release.wait()
   await jobs.submit('a',work)
   await asyncio.sleep(.01)
   with self.assertRaisesRegex(RuntimeError,'queue full'):await jobs.submit('b',work)
   release.set();self.assertTrue(await jobs.drain(1))
 async def test_queued_deadline_includes_gate_wait(self):
  with tempfile.TemporaryDirectory() as d:
   jobs=GenerationJobs(store=JobStore(Path(d)/'jobs.json'),gate=GlobalRateGate(1),max_queue=3)
   release=asyncio.Event()
   async def work():await release.wait()
   await jobs.submit('a',work)
   await asyncio.sleep(.01)
   rec=await jobs.run_event('b',work,deadline=.04)
   self.assertEqual(rec.status,'failed');self.assertEqual(rec.error,'TimeoutError')
   release.set();self.assertTrue(await jobs.drain(1))

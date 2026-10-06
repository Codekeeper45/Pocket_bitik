import asyncio, tempfile, unittest
from pathlib import Path
from gen_jobs import GenerationJobs, JobStore, GlobalRateGate, ArtifactStore
from gen_references import parse_telegram_link

class RuntimeRegressions(unittest.IsolatedAsyncioTestCase):
 async def test_cancel_one_does_not_cancel_other(self):
  with tempfile.TemporaryDirectory() as d:
   jobs=GenerationJobs(store=JobStore(Path(d)/'jobs.json'),gate=GlobalRateGate(2))
   await jobs.startup()
   done=asyncio.Event()
   async def slow(): await asyncio.sleep(.05);done.set();return b'ok'
   a=await jobs.submit(1,slow);b=await jobs.submit(2,slow)
   await asyncio.sleep(.01);await jobs.cancel(a.job_id)
   await jobs.drain(1)
   self.assertTrue(done.is_set());self.assertEqual(jobs.jobs[b.job_id].status,'completed')
 async def test_drain_timeout_does_not_cancel_work(self):
  with tempfile.TemporaryDirectory() as d:
   jobs=GenerationJobs(store=JobStore(Path(d)/'jobs.json'),gate=GlobalRateGate(1))
   await jobs.startup()
   async def work():await asyncio.sleep(.05);return b'ok'
   rec=await jobs.submit(1,work)
   self.assertFalse(await jobs.drain(.01))
   self.assertTrue(await jobs.drain(1));self.assertEqual(rec.status,'completed')
 def test_private_topic_link(self):
  link=parse_telegram_link('https://t.me/c/4455207724/42/11088?single')
  self.assertEqual(link.chat,'-1004455207724');self.assertEqual(link.topic_id,42);self.assertEqual(link.message_id,11088)
 def test_artifact_path_escape(self):
  with tempfile.TemporaryDirectory() as d:
   store=ArtifactStore(d)
   with self.assertRaises(ValueError):store.save('../bad',b'data')

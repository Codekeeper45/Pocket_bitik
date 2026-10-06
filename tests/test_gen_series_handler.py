import asyncio
import tempfile
import unittest
from pathlib import Path

from gen_series import (parse_multipage_plan, execute_series, SeriesCheckpoint,
    SeriesPlanError, series_state_path, save_series_state, load_series_state)


class IncrementalSeriesHandlerTests(unittest.TestCase):
    def test_per_job_state_is_private_and_owner_chat_bound(self):
        with tempfile.TemporaryDirectory() as td:
            for jid, owner, chat in (("job-a", "u1", "c1"), ("job-b", "u2", "c2")):
                path = series_state_path(td, jid)
                save_series_state(path, {"job_id": jid, "owner_id": owner, "chat_id": chat})
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertNotEqual(series_state_path(td, "job-a"), series_state_path(td, "job-b"))
            self.assertEqual(load_series_state(td, job_id="job-a", owner_id="u1", chat_id="c1")["job_id"], "job-a")
            for owner, chat in (("u2", "c1"), ("u1", "c2")):
                with self.assertRaises(SeriesPlanError):
                    load_series_state(td, job_id="job-a", owner_id=owner, chat_id=chat)
            with self.assertRaises(SeriesPlanError): series_state_path(td, "../escape")

    def test_persists_before_delivery_and_resumes_delivery_without_regeneration(self):
        plan = parse_multipage_plan('{"pages":[{"number":1,"prompt":"one"},{"number":2,"prompt":"two"}]}', 2)
        calls, sends = [], []
        state = SeriesCheckpoint(plan.plan_id)
        with tempfile.TemporaryDirectory() as td:
            async def save(cp):
                Path(td, 'checkpoint.json').write_text(cp.to_json())
            async def generate(page, **_):
                calls.append(page.number)
                return f'image-{page.number}'.encode()
            async def deliver(page, raw):
                sends.append(page.number)
                if page.number == 1 and len(sends) == 1:
                    raise RuntimeError('mock delivery down')
            first = asyncio.run(execute_series(plan, generate=generate, checkpoint=state, save_checkpoint=save, deliver=deliver))
            self.assertEqual(calls, [1])
            self.assertEqual(first.completed[1], b'image-1')
            self.assertIn('delivery RuntimeError', first.failures[1])
            reloaded = SeriesCheckpoint.from_json(Path(td, 'checkpoint.json').read_text(), plan=plan)
            second = asyncio.run(execute_series(plan, generate=generate, checkpoint=reloaded, save_checkpoint=save, deliver=deliver))
            self.assertEqual(calls, [1, 2])
            self.assertEqual(set(second.delivered), {1, 2})
            self.assertEqual(sends, [1, 1, 2])

    def test_generation_failure_is_explicit_and_prior_page_not_regenerated(self):
        plan = parse_multipage_plan('{"pages":[{"number":1,"prompt":"one"},{"number":2,"prompt":"two"}]}', 2)
        calls = []
        async def gen(page, **_):
            calls.append(page.number)
            if page.number == 2:
                raise ValueError('provider unavailable')
            return b'page-one'
        first = asyncio.run(execute_series(plan, generate=gen))
        self.assertEqual(first.completed, {1: b'page-one'})
        self.assertIn('ValueError', first.failures[2])
        second = asyncio.run(execute_series(plan, generate=gen, checkpoint=first))
        self.assertEqual(calls, [1, 2, 2])
        self.assertEqual(second.completed[1], b'page-one')


if __name__ == '__main__':
    unittest.main()

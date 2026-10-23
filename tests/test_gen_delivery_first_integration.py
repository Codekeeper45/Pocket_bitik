"""Offline delivery/jobs regressions; bot functions run without importing the bot."""
import ast
import asyncio
import base64
import copy
from contextlib import chdir
import io
import json
import os
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

import gen_delivery as delivery
import gen_runtime as runtime
import gen_series as series
from gen_jobs import ArtifactStore, ContextBudget, GenerationJobs, GlobalRateGate, JobStore
from gen_series import SeriesCheckpoint, SeriesPage, SeriesPlan, execute_series, parse_multipage_plan


def load_bot_function(name, namespace):
    source = Path(__file__).resolve().parents[1] / 'bot_new.py'
    if '_bot_ast' not in namespace:
        namespace['_bot_ast'] = ast.parse(source.read_text())
    node = next(node for node in namespace['_bot_ast'].body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


class DeliveryFirstIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = ArtifactStore(self.root / 'gen_artifacts')
        self.jobs = GenerationJobs(store=JobStore(self.root / 'jobs.json'), gate=GlobalRateGate(1))
        self.enterContext(patch.object(delivery, 'ROOT', self.root / 'records'))

    async def saved_job(self):
        return await self.jobs.run_event('chat', lambda: b'image', artifact_store=self.artifacts)

    async def run_raw_handler(self, *, send_error=None):
        context = await self.run_handler(send_error=send_error)
        return (context.record, context.provider, context.client, context.status,
                context.messages, context.raw)

    async def run_handler(self, *, send_error=None, prompt='A blue square', intent_plan=None,
                          selected_refs=(), provider_results=None, flags=None, namespace_overrides=None):
        buffer = io.BytesIO()
        Image.new('RGB', (64, 64), 'blue').save(buffer, format='PNG')
        raw = buffer.getvalue()
        provider = Mock(return_value=(raw, 'image/png'), side_effect=provider_results)
        status = SimpleNamespace(edit=AsyncMock(), delete=AsyncMock(side_effect=RuntimeError('cleanup')))
        async def confirmed_send(*args, **kwargs):
            return SimpleNamespace(id=41 + client.send_file.await_count)
        client = SimpleNamespace(get_me=AsyncMock(return_value=SimpleNamespace(id=42)),
                                 send_message=AsyncMock(return_value=status),
                                 send_file=AsyncMock(side_effect=confirmed_send if send_error is None else send_error))
        messages = []
        invocations = []
        async def build_plan(user_prompt, *args, **kwargs):
            invocation = runtime.CURRENT.get()
            invocations.append(invocation)
            invocation.timings['intent_plan'] = copy.deepcopy(intent_plan)
            return user_prompt, list(selected_refs), None, None
        builder = AsyncMock(side_effect=build_plan)
        namespace = dict(
            asyncio=asyncio, base64=base64, io=io, json=json, os=os, re=re, time=time, client=client,
            _GEN_ACTIVE_JOBS={}, save_json=lambda *args: None,
            log=lambda *args: messages.append(args),
            _slash_for_other_bot=AsyncMock(return_value=False), openrouter_client=object(),
            CHATGPT2API_AUTH_KEY='', _get_topic_id=lambda event: None,
            _is_real_reply=lambda event: (False, None), _owner_label=lambda: 'test owner',
            _gen_fetch_link_refs=AsyncMock(return_value=(prompt, [], [])),
            _gen_collect_input_images=AsyncMock(return_value=([], 0)),
            _build_gen_prompt=builder, get_active_model=lambda: (None, 'fake-model', 'fake model'),
            _collapsed_entities=lambda text, **kwargs: (text, []), add_surrogate=lambda text: text,
            send_long=AsyncMock(), GEN_CTX_REF_MAX=8, GEN_IMAGE_MAX_INPUT=20_000_000, OWNER_ID=42,
            GEN_IMAGE_MODEL='test-image', GEN_IMAGE_RES='2K', GEN_IMAGE_INPUT=True,
            OPENROUTER_IMAGE_FALLBACK='', _clamp_resolution=lambda size, maximum: size,
            _GEN_PROVIDER_GATE=GlobalRateGate(1), _sync_generate_image=provider,
            _gen_rate_gate=AsyncMock(), _gen_visual_qa=AsyncMock(return_value=None),
            active_model_supports_vision=lambda: True,
            _index_memory_allowed=lambda owner: False, describe_image=AsyncMock(return_value=None),
            GEN_CTX_IMG_MAX=8,
        )
        for name in ('_gen_provider_call', '_gen_one_image_impl', '_gen_one_image_recorded',
                     '_gen_one_image', '_gen_render_image', '_gen_send_image',
                     '_gen_remap_selected_refs', '_gen_role_number_remap', '_gen_actual_role_instruction',
                     '_gen_page_reference_snapshot', '_gen_series_refs_line', '_gen_refs_line',
                     '_detect_pages_count', 'gen_command', 'gen_resume', '_track_gen_activity'):
            load_bot_function(name, namespace)
        namespace.update(namespace_overrides or {})
        event = SimpleNamespace(
            out=True, sender_id=42, chat_id=123, id=10, reply_to=None,
            delete=AsyncMock(), reply=AsyncMock(),
            pattern_match=SimpleNamespace(group=lambda number: {
                1: None, 2: flags if flags is not None else '-raw -noimg' if intent_plan is None else '-noimg',
                3: None, 4: prompt}[number]))
        with patch.object(runtime, 'ARTIFACTS', self.artifacts), \
                patch.object(runtime, 'run_event', self.jobs.run_event), \
                patch.dict(os.environ, {'GEN_LAYERS_ENABLED': '0'}), chdir(self.root):
            rec = await namespace['_track_gen_activity'](namespace['gen_command'])(event)
        return SimpleNamespace(record=rec, provider=provider, client=client, status=status,
                               messages=messages, raw=raw, namespace=namespace, event=event,
                               builder=builder, invocations=invocations)

    async def resume_handler(self, context, job_id):
        event = SimpleNamespace(out=True, sender_id=42, chat_id=123, id=11, reply=AsyncMock(),
                                pattern_match=SimpleNamespace(group=lambda number: job_id))
        with patch.object(runtime, 'ARTIFACTS', self.artifacts), \
                patch.object(runtime, 'run_event', self.jobs.run_event), \
                patch.dict(os.environ, {'GEN_LAYERS_ENABLED': '0'}), chdir(self.root):
            rec = await context.namespace['_track_gen_activity'](context.namespace['gen_resume'])(event)
        return rec, event

    def saved_series(self):
        paths = list((self.root / 'gen_series_state').glob('*.json'))
        self.assertEqual(len(paths), 1)
        record = json.loads(paths[0].read_text())
        plan = parse_multipage_plan(json.dumps({
            'pages': record['plan'], 'bible': record['bible'], 'version': record['version']}),
            len(record['plan']))
        return paths[0], record, SeriesCheckpoint.from_json(record['checkpoint'], plan=plan)

    def distinct_provider_results(self):
        results = []
        for color in ('red', 'green', 'blue'):
            buffer = io.BytesIO()
            Image.new('RGB', (64, 64), color).save(buffer, format='PNG')
            results.append((buffer.getvalue(), 'image/png'))
        return results

    def independent_plan(self):
        return dict(task_type='creation', output_kind='independent_images', output_count=3,
                    prompt='Three independent landscapes', scenes=['Sunrise', 'Forest', 'Ocean'])

    async def test_ai_model_three_independent_images_reach_provider_and_delivery(self):
        outputs = self.distinct_provider_results()
        context = await self.run_handler(
            prompt='Make three independent images: sunrise, forest, ocean.',
            intent_plan=self.independent_plan(), provider_results=outputs)
        self.assertEqual(context.provider.call_count, 3, context.messages)
        self.assertEqual(context.client.send_file.await_count, 3, context.messages)
        self.assertEqual(context.record.status, 'delivered', context.messages)
        context.builder.assert_awaited_once()
        self.assertTrue(all(call.args[1] is None for call in context.provider.call_args_list))
        _, record, checkpoint = self.saved_series()
        self.assertEqual(record['generation']['intent_plan']['output_kind'], 'independent_images')
        self.assertEqual(checkpoint.delivered, {1, 2, 3})
        self.assertEqual(set(checkpoint.completed), {1, 2, 3})
        self.assertFalse(checkpoint.failures)
        self.assertEqual(len(list(delivery.ROOT.glob('*.json'))), 3)
        self.assertEqual({result['message_id'] for result in checkpoint.completed.values()}, {42, 43, 44})
        for number, result in checkpoint.completed.items():
            self.assertEqual(Path(result['artifact']).read_bytes(), outputs[number - 1][0])

    async def test_ai_three_panels_stay_one_provider_call_and_one_file(self):
        context = await self.run_handler(prompt='One image with three panels, not separate files.',
            intent_plan=dict(task_type='creation', output_kind='single_image', output_count=1,
                             panels_per_image=3, prompt='One triptych landscape'))
        context.provider.assert_called_once()
        context.client.send_file.assert_awaited_once()
        self.assertEqual(context.record.status, 'delivered', context.messages)
        self.assertIn('exactly 3 panels', context.provider.call_args.args[0])
        self.assertFalse((self.root / 'gen_series_state').exists())

    async def test_ai_negated_three_images_stays_one_output(self):
        prompt = 'Do not make three images. Make one image of three friends.'
        context = await self.run_handler(prompt=prompt,
            intent_plan=dict(task_type='creation', output_kind='single_image', output_count=1,
                             participant_count=3, prompt='Three friends together in one image'))
        context.provider.assert_called_once()
        context.client.send_file.assert_awaited_once()
        self.assertEqual(context.record.status, 'delivered', context.messages)
        self.assertIn(prompt, context.provider.call_args.args[0])
        self.assertFalse((self.root / 'gen_series_state').exists())

    async def test_ai_malformed_plan_falls_back_to_original_request_and_delivers(self):
        prompt = 'A blue landscape with the exact text HELLO.'
        context = await self.run_handler(prompt=prompt,
            intent_plan={'output_count': 'three', 'prompt': ['invalid scene']})
        context.provider.assert_called_once()
        context.client.send_file.assert_awaited_once()
        context.builder.assert_awaited_once()
        self.assertEqual(context.record.status, 'delivered', context.messages)
        self.assertIn(prompt, context.provider.call_args.args[0])
        self.assertTrue(context.invocations[0].timings['intent_plan']['fallback_used'])

    async def test_ai_missing_selected_reference_still_delivers_best_effort(self):
        context = await self.run_handler(prompt='Draw the character using a source if available.',
            intent_plan=dict(output_count=1, output_kind='single_image', prompt='Character portrait',
                             refs=[{'ref_id': 'catalog:99', 'roles': ['subject']}]),
            selected_refs=[(99, 'subject')])
        context.provider.assert_called_once()
        context.client.send_file.assert_awaited_once()
        self.assertEqual(context.record.status, 'delivered', context.messages)
        self.assertIsNone(context.provider.call_args.args[1])
        registry = context.invocations[0].timings['reference_registry']
        self.assertEqual(registry['input_count'], 0)
        self.assertIn({'ref_id': 'catalog:99', 'reason': 'unavailable'}, registry['issues'])

    async def test_actual_ai_partial_series_resumes_only_missing_pages(self):
        outputs = self.distinct_provider_results()
        context = await self.run_handler(prompt='Make three independent landscapes.',
            intent_plan=self.independent_plan(), provider_results=[outputs[0], TimeoutError(), *outputs[1:]])
        self.assertEqual(context.record.status, 'partial', context.messages)
        self.assertEqual(context.provider.call_count, 2)
        context.client.send_file.assert_awaited_once()
        path, record, checkpoint = self.saved_series()
        self.assertEqual(checkpoint.delivered, {1})
        self.assertEqual(set(checkpoint.completed), {1})
        self.assertIn(2, checkpoint.failures)
        first_artifact = dict(checkpoint.completed[1])
        resumed, event = await self.resume_handler(context, record['job_id'])
        self.assertEqual(resumed.status, 'delivered', event.reply.call_args_list)
        self.assertEqual(context.provider.call_count, 4)
        self.assertEqual(context.client.send_file.await_count, 3)
        context.builder.assert_awaited_once()
        resumed_path, _, restored = self.saved_series()
        self.assertEqual(path, resumed_path)
        self.assertEqual(restored.delivered, {1, 2, 3})
        self.assertEqual(restored.completed[1], first_artifact)
        self.assertFalse(restored.failures)

    async def test_mixed_original_and_anchor_qa_initial_and_resume(self):
        outputs = self.distinct_provider_results()
        original = base64.b64encode(outputs[2][0]).decode()
        observations = []
        async def qa(*args, **kwargs):
            observations.append(copy.deepcopy(runtime.CURRENT.get().qa_references))
            return None
        plan = self.independent_plan()
        plan['output_kind'] = 'series'
        plan['refs'] = [{'ref_id': 'submitted:1', 'roles': ['subject']}]
        context = await self.run_handler(intent_plan=plan, flags='',
            provider_results=[outputs[0], outputs[1], TimeoutError(), outputs[2]],
            namespace_overrides={'_gen_collect_input_images': AsyncMock(return_value=([original], 0)),
                                 '_gen_visual_qa': qa})
        self.assertEqual(context.record.status, 'partial', context.messages)
        _, record, _ = self.saved_series()
        resumed, event = await self.resume_handler(context, record['job_id'])
        self.assertEqual(resumed.status, 'delivered', event.reply.call_args_list)
        self.assertEqual(len(observations), 3)
        from gen_references import resize_image
        for refs in observations[1:]:
            self.assertEqual([ref['ref_id'] for ref in refs], [observations[0][0]['ref_id'], 'series:anchor'])
            self.assertEqual(refs[0]['data'], outputs[2][0])
            self.assertEqual(refs[1]['data'], resize_image(outputs[0][0]))
            self.assertEqual(refs[1]['roles'], ['subject', 'style'])

    async def test_batch_real_builder_preserves_captured_reference_mapping(self):
        import bot_new as bot
        originals = [base64.b64encode(raw).decode() for raw, mime in self.distinct_provider_results()[:2]]
        payload = dict(task_type='creation', output_kind='single_image', output_count=1,
                       prompt='Two distinct characters', refs=[
                           {'ref_id': 'submitted:1', 'roles': ['subject']},
                           {'ref_id': 'submitted:2', 'roles': ['subject']}])
        changed = {**payload, 'refs': [{'ref_id': 'web:unattached', 'roles': ['subject']}]}
        def plan_response(**kwargs):
            plan = payload if planner.call_count == 1 else changed
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(plan), tool_calls=[]))])
        planner = Mock(side_effect=plan_response)
        llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=planner)))
        observed = []
        async def render(prompt, inputs, *args):
            observed.append((prompt, inputs, copy.deepcopy(runtime.CURRENT.get().timings['reference_registry'])))
            self.assertNotIn('web:unattached', str(runtime.CURRENT.get().timings['intent_plan']))
            return self.distinct_provider_results()[0] + (prompt, False)
        with patch.object(bot, 'get_active_model', return_value=(llm, 'fake', 'fake')), \
                patch.object(bot, 'active_model_supports_vision', return_value=False):
            context = await self.run_handler(flags='-x3', namespace_overrides={
                'GEN_BATCH_MAX': 4, 'GEN_BATCH_CONCURRENCY': 2,
                '_gen_collect_input_images': AsyncMock(return_value=(originals, 0)),
                '_build_gen_prompt': bot._build_gen_prompt, '_gen_render_image': render})
        self.assertEqual(len(observed), 3, context.messages)
        for prompt, inputs, registry in observed:
            self.assertEqual(inputs, originals)
            self.assertIn('Stable source to API attachment mapping: ', prompt)
            mapping = prompt.split('Stable source to API attachment mapping: ')[-1]
            self.assertEqual([ref['ref_id'] for ref in json.loads(mapping)],
                             [ref['ref_id'] for ref in observed[0][2]['references']])
            self.assertEqual([ref['api_index'] for ref in json.loads(mapping)], [1, 2])
            self.assertEqual(registry['input_count'], 2)
        for call in planner.call_args_list[1:]:
            self.assertNotIn('tools', call.kwargs)
            self.assertIn('submitted:2', str(call.kwargs['messages']))

    async def test_presend_resolution_failure_can_retry_without_history(self):
        context = await self.run_handler()
        context.client.get_me.side_effect = TimeoutError()
        invocation = runtime.Invocation()
        token = runtime.CURRENT.set(invocation)
        try:
            with patch.object(runtime, 'ARTIFACTS', self.artifacts), chdir(self.root):
                with self.assertRaises(TimeoutError):
                    await context.namespace['_gen_send_image']('me', context.raw, 'image/png', 'test', False, None)
                key = invocation.timings['delivery_key']
                entry = json.loads((delivery.ROOT / (key + '.json')).read_text())
                self.assertEqual(entry['status'], 'not_started')
                context.client.get_me.side_effect = None
                context.client.send_file.reset_mock()
                async def send(*args, **kwargs):
                    self.assertEqual(json.loads((delivery.ROOT / (key + '.json')).read_text())['status'], 'sending')
                    return SimpleNamespace(id=99)
                context.client.send_file.side_effect = send
                retry = load_bot_function('gen_retry_file', context.namespace)
                event = SimpleNamespace(out=True, sender_id=42, chat_id=42, reply=AsyncMock(),
                    pattern_match=SimpleNamespace(group=lambda number: key))
                with patch.object(delivery, 'reconcile', AsyncMock(side_effect=AssertionError('history queried'))):
                    await retry(event)
                    await retry(event)
                context.client.send_file.assert_awaited_once()
                self.assertEqual(json.loads((delivery.ROOT / (key + '.json')).read_text())['status'], 'delivered')
        finally:
            runtime.CURRENT.reset(token)

    async def test_retry_unknown_sends_remain_conservative(self):
        client = SimpleNamespace(send_file=AsyncMock(), get_messages=AsyncMock())
        retry = load_bot_function('gen_retry_file', dict(client=client, OWNER_ID=42))
        artifact = self.artifacts.save('unknown', b'image', '.png')
        with chdir(self.root):
            for status in ('pending', 'sending', 'ambiguous'):
                with self.subTest(status=status):
                    delivery.record(status, chat=123, reply_to=None, path=artifact, status=status)
                    delivery.record(status, chat=123, reply_to=None, path=artifact, status='not_started')
                    self.assertEqual(delivery.lookup(status, 123)['status'], status)
                    event = SimpleNamespace(out=True, sender_id=42, chat_id=123, reply=AsyncMock(),
                        pattern_match=SimpleNamespace(group=lambda number: status))
                    await retry(event)
                    client.send_file.assert_not_awaited()
                    event.reply.assert_awaited_once()

    async def test_retry_cancellation_never_restores_not_started(self):
        artifact = self.artifacts.save('cancelled', b'image', '.png')
        delivery.record('cancelled', chat=123, reply_to=None, path=artifact, status='not_started')
        async def cancel_send(*args, **kwargs):
            self.assertEqual(delivery.lookup('cancelled', 123)['status'], 'sending')
            raise asyncio.CancelledError()
        client = SimpleNamespace(send_file=AsyncMock(side_effect=cancel_send))
        retry = load_bot_function('gen_retry_file', dict(client=client, OWNER_ID=42))
        event = SimpleNamespace(out=True, sender_id=42, chat_id=123, reply=AsyncMock(),
            pattern_match=SimpleNamespace(group=lambda number: 'cancelled'))
        with chdir(self.root):
            with self.assertRaises(asyncio.CancelledError):
                await retry(event)
            self.assertEqual(delivery.lookup('cancelled', 123)['status'], 'ambiguous')
            await retry(event)
        client.send_file.assert_awaited_once()

    async def test_actual_ai_resume_stops_on_uncertain_page_delivery(self):
        context = await self.run_handler(prompt='Make three independent landscapes.',
            intent_plan=self.independent_plan(), provider_results=self.distinct_provider_results(),
            send_error=[SimpleNamespace(id=42), TimeoutError(), SimpleNamespace(id=44)])
        self.assertEqual(context.record.status, 'partial', context.messages)
        self.assertEqual(context.provider.call_count, 2)
        self.assertEqual(context.client.send_file.await_count, 2)
        path, record, checkpoint = self.saved_series()
        self.assertEqual(checkpoint.delivered, {1})
        self.assertEqual(set(checkpoint.completed), {1, 2})
        before = path.read_bytes()
        resumed, event = await self.resume_handler(context, record['job_id'])
        self.assertFalse(resumed.delivered)
        self.assertEqual(context.provider.call_count, 2)
        self.assertEqual(context.client.send_file.await_count, 2)
        self.assertEqual(path.read_bytes(), before)
        self.assertIn('Повторная отправка остановлена', event.reply.call_args.args[0])

    async def test_concurrent_actual_resume_loads_checkpoint_under_shared_lock(self):
        outputs = self.distinct_provider_results()
        context = await self.run_handler(prompt='Three independent images.',
            intent_plan=self.independent_plan(), provider_results=[outputs[0], TimeoutError()])
        self.assertEqual(context.record.status, 'partial', context.messages)
        _, record, checkpoint = self.saved_series()
        first_page = dict(checkpoint.completed[1])
        started, release, second_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original_render = context.namespace['_gen_render_image']
        async def paused_render(*args, **kwargs):
            started.set()
            await release.wait()
            return await original_render(*args, **kwargs)
        context.namespace['_gen_render_image'] = paused_render
        context.provider.side_effect = outputs[1:]
        events = [SimpleNamespace(out=True, sender_id=42, chat_id=123, id=number, reply=AsyncMock(),
                  pattern_match=SimpleNamespace(group=lambda number: record['job_id'])) for number in (11, 12)]
        async def second_resume():
            second_started.set()
            return await context.namespace['gen_resume'](events[1])
        with patch.object(runtime, 'ARTIFACTS', self.artifacts), \
                patch.object(series, 'load_series_state', wraps=series.load_series_state) as load, \
                patch.dict(os.environ, {'GEN_LAYERS_ENABLED': '0'}), chdir(self.root):
            first = asyncio.create_task(context.namespace['gen_resume'](events[0]))
            second = None
            try:
                await asyncio.wait_for(started.wait(), 1)
                second = asyncio.create_task(second_resume())
                await asyncio.wait_for(second_started.wait(), 1)
                self.assertEqual(load.call_count, 1)
                self.assertFalse(second.done())
                release.set()
                results = await asyncio.wait_for(asyncio.gather(first, second), 3)
            finally:
                release.set()
                for task in (first, second):
                    if task is not None and not task.done(): task.cancel()
                await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)
        self.assertEqual(load.call_count, 2)
        self.assertTrue(all(result.delivered == {1, 2, 3} for result in results))
        self.assertEqual(context.provider.call_count, 4)
        self.assertEqual(context.client.send_file.await_count, 3)
        _, _, restored = self.saved_series()
        self.assertEqual(restored.completed[1], first_page)
        self.assertEqual(restored.delivered, {1, 2, 3})
        self.assertFalse(restored.failures)

    async def test_actual_handler_provider_delivery_survives_unavailable_qa_and_cleanup(self):
        rec, provider, client, status, messages, raw = await self.run_raw_handler()
        provider.assert_called_once()
        self.assertEqual(client.send_file.await_count, 1, messages)
        self.assertEqual(client.send_file.await_args.args[1].getvalue(), raw)
        status.delete.assert_awaited_once()
        entry = json.loads(next(delivery.ROOT.glob('*.json')).read_text())
        self.assertEqual(entry['status'], 'delivered')
        self.assertEqual(entry['message_id'], 42)
        self.assertEqual(Path(entry['path']).read_bytes(), raw)
        self.assertNotEqual(rec.status, 'failed', messages)

    async def test_actual_handler_send_timeout_retains_bytes_without_duplicate(self):
        rec, provider, client, status, messages, raw = await self.run_raw_handler(send_error=TimeoutError())
        provider.assert_called_once()
        self.assertEqual(client.send_file.await_count, 1, messages)
        entry = json.loads(next(delivery.ROOT.glob('*.json')).read_text())
        self.assertEqual(entry['status'], 'ambiguous')
        self.assertEqual(Path(entry['path']).read_bytes(), raw)
        self.assertFalse(rec.delivered)

    async def test_gate_deadline_includes_queue_and_rate_wait(self):
        gate = GlobalRateGate(1)
        await gate.semaphore.acquire()
        work = AsyncMock()
        pending = asyncio.create_task(gate.run(work, timeout=.02))
        await asyncio.sleep(.06)
        expired_in_queue = pending.done()
        gate.semaphore.release()
        await asyncio.gather(pending, return_exceptions=True)
        self.assertTrue(expired_in_queue)
        work.assert_not_awaited()
        gate._next_at = time.monotonic() + .06
        with self.assertRaises(TimeoutError):
            await gate.run(work, timeout=.01)
        work.assert_not_awaited()

    async def test_zero_and_invocation_deadlines_do_not_start_provider(self):
        work = AsyncMock(return_value=b'image')
        rec = await self.jobs.run_event('chat', work, deadline=0)
        self.assertEqual(rec.status, 'failed')
        work.assert_not_awaited()
        token = runtime.CURRENT.set(runtime.Invocation(max_seconds=0))
        try:
            rec = await self.jobs.run_event('chat', work)
        finally:
            runtime.CURRENT.reset(token)
        self.assertEqual(rec.status, 'failed')
        work.assert_not_awaited()

    async def test_handler_return_without_result_is_not_completion(self):
        rec = await self.jobs.run_event('chat', AsyncMock(return_value=None))
        self.assertEqual(rec.status, 'incomplete')
        self.assertFalse(rec.delivered)

    async def test_persistence_wait_consumes_submission_deadline(self):
        await self.jobs.startup()
        original_save = self.jobs.store.save
        async def delayed_save(data):
            await asyncio.sleep(.03)
            await original_save(data)
        work = AsyncMock(return_value=b'image')
        with patch.object(self.jobs.store, 'save', delayed_save):
            rec = await self.jobs.run_event('chat', work, deadline=.01)
        self.assertEqual(rec.status, 'failed')
        work.assert_not_awaited()

    async def test_queue_timeout_does_not_spend_provider_call_budget(self):
        budget = ContextBudget(max_calls=1)
        work = AsyncMock(return_value=b'image')
        await self.jobs.gate.semaphore.acquire()
        try:
            rec = await self.jobs.run_event('chat', work, deadline=.02, context_budget=budget)
        finally:
            self.jobs.gate.semaphore.release()
        self.assertEqual(rec.status, 'failed')
        self.assertEqual(budget.calls, 0)
        work.assert_not_awaited()

    async def test_thread_timeout_keeps_real_capacity_until_thread_exits(self):
        gate = GlobalRateGate(1)
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        def provider():
            loop.call_soon_threadsafe(started.set)
            release.wait(2)
        task = asyncio.create_task(gate.run(provider, timeout=.03))
        try:
            await started.wait()
            with self.assertRaises(TimeoutError):
                await task
            self.assertEqual(gate.inflight_threads, 1)
            waiting = AsyncMock()
            with self.assertRaises(TimeoutError):
                await gate.run(waiting, timeout=.01)
            waiting.assert_not_awaited()
        finally:
            release.set()
            await asyncio.gather(*tuple(gate._inflight_threads), return_exceptions=True)
        self.assertEqual(gate.semaphore._value, 1)

    async def test_explicit_confirmed_outcome_survives_qa_error(self):
        invocation = runtime.Invocation(error='qa_unavailable')
        token = runtime.CURRENT.set(invocation)
        try:
            rec = await self.jobs.run_event('chat', AsyncMock(return_value={'status': 'delivered'}))
        finally:
            runtime.CURRENT.reset(token)
        self.assertEqual(rec.status, 'delivered')
        self.assertTrue(rec.delivered)

    async def test_uncertain_send_never_blindly_retries_even_after_restart(self):
        rec = await self.saved_job()
        send = AsyncMock(side_effect=TimeoutError('private payload'))
        with self.assertRaises(TimeoutError):
            await self.jobs.retry_delivery(rec.job_id, send, attempts=3)
        self.assertEqual(send.await_count, 1)
        restarted = GenerationJobs(store=self.jobs.store, gate=GlobalRateGate())
        await restarted.startup()
        with self.assertRaises(RuntimeError):
            await restarted.retry_delivery(rec.job_id, send)
        self.assertEqual(send.await_count, 1)
        self.assertEqual(Path(rec.artifact_path).read_bytes(), b'image')
        self.assertNotIn('private payload', self.jobs.store.path.read_text())

    async def test_verified_absence_allows_retry_without_regeneration(self):
        rec = await self.saved_job()
        send = AsyncMock(side_effect=[TimeoutError(), 'sent'])
        reconcile = AsyncMock(return_value=None)
        result = await self.jobs.retry_delivery(rec.job_id, send, attempts=2, reconcile=reconcile)
        self.assertEqual(result, 'sent')
        self.assertEqual(send.await_count, 2)
        reconcile.assert_awaited_once()
        self.assertTrue(rec.delivered)

    async def test_cancellation_during_send_requires_reconciliation(self):
        rec = await self.saved_job()
        started = asyncio.Event()
        async def send(path):
            started.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(self.jobs.retry_delivery(rec.job_id, send))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(rec.status, 'delivery_uncertain')
        retry = AsyncMock()
        existing = SimpleNamespace(id=42)
        self.assertIs(await self.jobs.retry_delivery(rec.job_id, retry,
                      reconcile=AsyncMock(return_value=existing)), existing)
        retry.assert_not_awaited()
        self.assertTrue(rec.delivered)

    async def test_restart_during_delivery_never_resends_without_readback(self):
        rec = await self.saved_job()
        rec.status = 'delivering'
        await self.jobs._persist()
        restarted = GenerationJobs(store=self.jobs.store, gate=GlobalRateGate())
        self.assertEqual(await restarted.startup(), [rec.job_id])
        self.assertEqual(restarted.jobs[rec.job_id].status, 'delivery_uncertain')
        send = AsyncMock()
        with self.assertRaises(RuntimeError):
            await restarted.retry_delivery(rec.job_id, send)
        send.assert_not_awaited()

    async def test_concurrent_delivery_requests_send_once(self):
        rec = await self.saved_job()
        send = AsyncMock(return_value='sent')
        await asyncio.gather(self.jobs.retry_delivery(rec.job_id, send),
                             self.jobs.retry_delivery(rec.job_id, send))
        send.assert_awaited_once()

    async def test_successful_send_persist_failure_never_retries_delivery(self):
        rec = await self.saved_job()
        original_save = self.jobs.store.save
        async def fail_confirmation(data):
            if data['jobs'][rec.job_id]['status'] == 'delivered':
                raise OSError('confirmation persistence unavailable')
            await original_save(data)
        sent = SimpleNamespace(id=42)
        send = AsyncMock(return_value=sent)
        reconcile = AsyncMock(return_value=sent)
        with patch.object(self.jobs.store, 'save', side_effect=fail_confirmation):
            with self.assertRaises(OSError):
                await self.jobs.retry_delivery(rec.job_id, send, attempts=3, reconcile=reconcile)
        self.assertTrue(rec.delivered)
        self.assertEqual(rec.status, 'delivered')
        await self.jobs.retry_delivery(rec.job_id, send)
        send.assert_awaited_once()
        reconcile.assert_not_awaited()
        restarted = GenerationJobs(store=self.jobs.store, gate=GlobalRateGate())
        await restarted.startup()
        self.assertEqual(restarted.jobs[rec.job_id].status, 'delivery_uncertain')
        with self.assertRaises(RuntimeError):
            await restarted.retry_delivery(rec.job_id, send)
        self.assertIs(await restarted.retry_delivery(rec.job_id, send, reconcile=reconcile), sent)
        send.assert_awaited_once()
        reconcile.assert_awaited_once()
        saved = await restarted.store.load()
        self.assertTrue(saved['jobs'][rec.job_id]['delivered'])

    async def test_cancellation_after_send_during_confirmation_does_not_resend(self):
        rec = await self.saved_job()
        original_save = self.jobs.store.save
        confirming = asyncio.Event()
        async def wait_confirmation(data):
            if data['jobs'][rec.job_id]['status'] == 'delivered':
                confirming.set()
                await asyncio.Event().wait()
            await original_save(data)
        send = AsyncMock(return_value=SimpleNamespace(id=42))
        with patch.object(self.jobs.store, 'save', side_effect=wait_confirmation):
            task = asyncio.create_task(self.jobs.retry_delivery(rec.job_id, send))
            await asyncio.wait_for(confirming.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(rec.delivered)
        self.assertEqual(rec.status, 'delivered')
        await self.jobs.retry_delivery(rec.job_id, send)
        send.assert_awaited_once()

    async def test_batch_parent_cancellation_joins_started_child(self):
        started = asyncio.Event()
        children, parents = [], []
        async def render(*args, **kwargs):
            children.append(asyncio.current_task())
            self.assertIs(runtime.CURRENT.get().parent, parents[0])
            started.set()
            await asyncio.Event().wait()
        async def build(*args, **kwargs):
            parents.append(runtime.CURRENT.get())
            if len(parents) == 1:
                return 'First image', [], None, None
            await asyncio.wait_for(started.wait(), 1)
            raise asyncio.CancelledError()
        send = AsyncMock()
        try:
            with self.assertRaises(asyncio.CancelledError):
                await self.run_handler(flags='-x3 -noimg', namespace_overrides=dict(
                    GEN_BATCH_MAX=4, GEN_BATCH_CONCURRENCY=2,
                    _build_gen_prompt=AsyncMock(side_effect=build),
                    _gen_render_image=AsyncMock(side_effect=render), _gen_send_image=send))
            self.assertEqual(len(children), 1)
            self.assertTrue(children[0].done())
            self.assertTrue(children[0].cancelled())
            send.assert_not_awaited()
            self.assertEqual(next(iter(self.jobs.jobs.values())).status, 'cancelled')
        finally:
            for child in children:
                if not child.done(): child.cancel()
            await asyncio.gather(*children, return_exceptions=True)

    async def test_saved_messages_resolution_requires_authenticated_self(self):
        client = SimpleNamespace(get_me=AsyncMock(return_value=SimpleNamespace(id=42)))
        self.assertEqual(await delivery.resolve_chat(client, 123), 123)
        client.get_me.assert_not_awaited()
        self.assertEqual(await delivery.resolve_chat(client, 'me'), 42)
        client.get_me.assert_awaited_once()
        for invalid_id in (None, True, -42, '42'):
            client.get_me.return_value = SimpleNamespace(id=invalid_id)
            with self.assertRaises(RuntimeError):
                await delivery.resolve_chat(client, 'me')

    def test_legacy_saved_messages_lookup_is_scoped_to_verified_self(self):
        artifact = self.artifacts.save('legacy', b'image')
        delivery.record('legacy', chat='me', reply_to=None, path=artifact, status='ambiguous')
        with chdir(self.root):
            with self.assertRaises(PermissionError):
                delivery.lookup('legacy', 42)
            with self.assertRaises(PermissionError):
                delivery.lookup('legacy', 123, self_id=42)
            entry = delivery.lookup('legacy', 42, self_id=42)
            self.assertEqual(entry['key'], 'legacy')
        updated = delivery.record('legacy', chat=42, reply_to=None, path=artifact,
                                  status='delivered', message_id=7, self_id=42)
        self.assertEqual(updated['chat'], 42)
        with chdir(self.root):
            self.assertEqual(delivery.lookup('legacy', 42)['message_id'], 7)

    async def test_legacy_saved_messages_reconcile_checks_numeric_self_scope(self):
        message = SimpleNamespace(id=7, out=True, chat_id=42, reply_to_msg_id=None, media=object())
        client = SimpleNamespace(get_me=AsyncMock(return_value=SimpleNamespace(id=42)),
                                 get_messages=AsyncMock(return_value=message))
        entry = dict(key='legacy', chat='me', reply_to=None, message_id=7)
        self.assertIs(await delivery.reconcile(client, entry), message)
        client.get_messages.assert_awaited_once_with(42, ids=7)
        message.chat_id = 123
        with self.assertRaises(RuntimeError):
            await delivery.reconcile(client, entry)
        client.get_messages.reset_mock()
        client.get_me.side_effect = TimeoutError()
        with self.assertRaises(TimeoutError):
            await delivery.reconcile(client, entry)
        client.get_messages.assert_not_awaited()

    def test_legacy_saved_messages_provenance_is_not_a_scope_wildcard(self):
        delivery.record_output(chat='me', message_id=7)
        self.assertFalse(delivery.known_output(42, 7))
        self.assertTrue(delivery.known_output(42, 7, self_id=42))
        self.assertFalse(delivery.known_output(123, 7, self_id=42))

    async def test_client_lookup_resolves_only_legacy_me_scope(self):
        artifact = self.artifacts.save('lookup', b'image')
        client = SimpleNamespace(get_me=AsyncMock(return_value=SimpleNamespace(id=42)))
        delivery.record('numeric', chat=42, reply_to=None, path=artifact, status='pending')
        delivery.record('legacy', chat='me', reply_to=None, path=artifact, status='pending')
        with chdir(self.root):
            self.assertEqual((await delivery.lookup_for_client(client, 'numeric', 42))['chat'], 42)
            with self.assertRaises(PermissionError):
                await delivery.lookup_for_client(client, 'numeric', 123)
            client.get_me.assert_not_awaited()
            entry = await delivery.lookup_for_client(client, 'legacy', 42)
            self.assertEqual(entry['chat'], 'me')
            client.get_me.assert_awaited_once()
            with self.assertRaises(PermissionError):
                await delivery.lookup_for_client(client, 'legacy', 123)
            client.get_me.side_effect = TimeoutError()
            with self.assertRaises(TimeoutError):
                await delivery.lookup_for_client(client, 'legacy', 42)

    async def test_client_lookup_never_authenticates_unsafe_path_or_missing_record(self):
        client = SimpleNamespace(get_me=AsyncMock(return_value=SimpleNamespace(id=42)))
        delivery.record('unsafe', chat=42, reply_to=None, path=self.root / 'outside.png', status='pending')
        with chdir(self.root):
            with self.assertRaises(PermissionError):
                await delivery.lookup_for_client(client, 'unsafe', 42)
            with self.assertRaises(FileNotFoundError):
                await delivery.lookup_for_client(client, 'missing', 42)
        client.get_me.assert_not_awaited()

    async def test_delivery_lock_covers_readback_send_and_confirmation(self):
        artifact = self.artifacts.save('locked', b'image')
        delivery.record('locked', chat=42, reply_to=None, path=artifact,
                        status='ambiguous', caption=delivery.marker('locked'))
        started, release = asyncio.Event(), asyncio.Event()
        async def send_file(*args, **kwargs):
            started.set()
            await release.wait()
            return SimpleNamespace(id=7)
        client = SimpleNamespace(get_me=AsyncMock(), get_messages=AsyncMock(return_value=[]),
                                 send_file=AsyncMock(side_effect=send_file))
        async def retry():
            async with delivery.delivery_lock('locked'):
                entry = await delivery.lookup_for_client(client, 'locked', 42)
                if entry['status'] == 'delivered': return
                self.assertIsNone(await delivery.reconcile(client, entry))
                sent = await client.send_file(42, entry['path'])
                delivery.record('locked', chat=entry['chat'], reply_to=None, path=entry['path'],
                                status='delivered', message_id=sent.id)
        with chdir(self.root):
            first = asyncio.create_task(retry())
            second = None
            try:
                await asyncio.wait_for(started.wait(), 1)
                second = asyncio.create_task(retry())
                await asyncio.sleep(0)
                client.send_file.assert_awaited_once()
                release.set()
                await asyncio.wait_for(asyncio.gather(first, second), 1)
            finally:
                release.set()
                for task in (first, second):
                    if task is not None and not task.done(): task.cancel()
                await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)
            self.assertEqual(delivery.lookup('locked', 42)['message_id'], 7)
        client.get_messages.assert_awaited_once()
        client.send_file.assert_awaited_once()
        client.get_me.assert_not_awaited()

    async def test_delivery_lock_cancelled_waiter_does_not_release_holder(self):
        entered = asyncio.Event()
        async def waiter():
            entered.set()
            async with delivery.delivery_lock('key'):
                self.fail('cancelled waiter acquired the lock')
        async with delivery.delivery_lock('key'):
            task = asyncio.create_task(waiter())
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            async with delivery.delivery_lock('different'):
                pass
        async with delivery.delivery_lock('key'):
            pass
        with self.assertRaises(ValueError):
            async with delivery.delivery_lock('../unsafe'):
                pass

    async def test_saved_messages_resolved_send_records_numeric_scope(self):
        raw = self.distinct_provider_results()[0][0]
        client = SimpleNamespace(get_me=AsyncMock(return_value=SimpleNamespace(id=42)),
                                 send_file=AsyncMock(return_value=SimpleNamespace(id=7)))
        send = load_bot_function('_gen_send_image', dict(
            io=io, json=json, client=client, log=lambda *args: None))
        target = await delivery.resolve_chat(client, 'me')
        with patch.object(runtime, 'ARTIFACTS', self.artifacts):
            await send(target, raw, 'image/png', 'A landscape', False, None)
        self.assertEqual(client.send_file.await_args.args[0], 42)
        entry = json.loads(next(delivery.ROOT.glob('*.json')).read_text())
        self.assertEqual(entry['chat'], 42)
        with chdir(self.root):
            self.assertEqual(delivery.lookup(entry['key'], 42)['message_id'], 7)
            with self.assertRaises(PermissionError):
                delivery.lookup(entry['key'], 123, self_id=42)
        self.assertTrue(delivery.known_output(42, 7))

    async def test_partial_series_retains_delivered_page_and_resumes_missing(self):
        plan = SeriesPlan((SeriesPage(1, 'first'), SeriesPage(2, 'second')))
        state = SeriesCheckpoint(plan.plan_id)
        generate = AsyncMock(side_effect=[b'first', b'second'])
        send = AsyncMock(side_effect=[None, TimeoutError()])
        checkpoint = self.root / 'series.json'
        def save(current):
            checkpoint.write_text(current.to_json())
        async def work():
            return await execute_series(plan, generate=generate, checkpoint=state,
                                        save_checkpoint=save, deliver=send)
        rec = await self.jobs.run_event('chat', work)
        self.assertEqual(rec.status, 'partial')
        self.assertFalse(rec.delivered)
        restored = SeriesCheckpoint.from_json(checkpoint.read_text(), plan=plan)
        self.assertEqual(restored.delivered, {1})
        readback = SimpleNamespace(get_messages=AsyncMock(return_value=[]))
        async def reconcile_page(page, artifact):
            self.assertEqual(page.number, 2)
            self.assertEqual(artifact, b'second')
            existing = await delivery.reconcile(readback, dict(
                key='second', chat='chat', caption=delivery.marker('second')))
            return existing is not None
        resumed_send = AsyncMock()
        resumed = await execute_series(plan, generate=generate, checkpoint=restored,
                                       save_checkpoint=save, deliver=resumed_send,
                                       reconcile_delivery=reconcile_page)
        self.assertEqual(resumed.delivered, {1, 2})
        self.assertEqual(generate.await_count, 2)
        self.assertEqual(resumed_send.await_args.args[0].number, 2)
        readback.get_messages.assert_awaited_once()

    async def test_cancelled_series_preserves_delivered_checkpoint(self):
        plan = SeriesPlan((SeriesPage(1, 'first'), SeriesPage(2, 'second')))
        state = SeriesCheckpoint(plan.plan_id)
        started_second = asyncio.Event()
        checkpoint = self.root / 'cancelled-series.json'
        async def generate(*, page, **kwargs):
            if page.number == 2:
                started_second.set()
                await asyncio.Event().wait()
            return b'first'
        def save(current):
            checkpoint.write_text(current.to_json())
        async def work():
            return await execute_series(plan, generate=generate, checkpoint=state,
                                        save_checkpoint=save, deliver=AsyncMock())
        rec = await self.jobs.submit('chat', work)
        await started_second.wait()
        self.assertTrue(await self.jobs.cancel(rec.job_id))
        await self.jobs.drain(1)
        restored = SeriesCheckpoint.from_json(checkpoint.read_text(), plan=plan)
        self.assertEqual(restored.delivered, {1})
        self.assertEqual(restored.completed, {1: b'first'})
        self.assertEqual(rec.status, 'cancelled')
        with self.assertRaises(RuntimeError):
            await self.jobs.retry_delivery(rec.job_id, AsyncMock())

    def test_confirmed_delivery_cannot_be_downgraded_by_cleanup(self):
        args = dict(chat=123, reply_to=10, path=self.root / 'image.png')
        delivery.record('key', **args, status='delivered', message_id=42)
        entry = delivery.record('key', **args, status='ambiguous')
        self.assertEqual(entry['status'], 'delivered')
        self.assertEqual(entry['message_id'], 42)

    async def test_reconcile_rejects_prefix_collision_and_wrong_reply(self):
        entry = dict(key='key', chat=123, reply_to=10, caption=delivery.marker('key'))
        for text, reply in [(delivery.marker('key_extra'), 10), (delivery.marker('key'), 11)]:
            client = SimpleNamespace(get_messages=AsyncMock(return_value=[
                SimpleNamespace(id=42, out=True, raw_text=text, reply_to_msg_id=reply, media=object())]))
            with self.assertRaises(RuntimeError):
                await delivery.reconcile(client, entry)

    async def test_reconcile_requires_valid_readback_and_saved_message_identity(self):
        entry = dict(key='key', chat=123, reply_to=10, caption=delivery.marker('key'))
        client = SimpleNamespace(get_messages=AsyncMock(return_value=None))
        with self.assertRaises(RuntimeError):
            await delivery.reconcile(client, entry)
        entry['message_id'] = 42
        client.get_messages.return_value = SimpleNamespace(
            id=43, out=True, chat_id=123, reply_to_msg_id=10, media=object())
        with self.assertRaises(RuntimeError):
            await delivery.reconcile(client, entry)
        client.get_messages.return_value.id = 42
        self.assertEqual((await delivery.reconcile(client, entry)).id, 42)

    async def test_ast_retry_uncertain_readback_never_sends(self):
        client = SimpleNamespace(get_me=AsyncMock(return_value=SimpleNamespace(id=42)),
                                 send_file=AsyncMock(), get_messages=AsyncMock(side_effect=TimeoutError()))
        retry = load_bot_function('gen_retry_file', dict(client=client, OWNER_ID=42))
        event = SimpleNamespace(out=True, sender_id=42, chat_id=123, reply=AsyncMock(),
                                pattern_match=SimpleNamespace(group=lambda number: 'key'))
        entry = dict(key='key', chat=123, reply_to=10, status='ambiguous', path='saved',
                     caption=delivery.marker('key'))
        with patch.object(delivery, 'lookup', return_value=entry):
            await retry(event)
        client.send_file.assert_not_awaited()
        event.reply.assert_awaited_once()

    async def test_ast_wrapper_queue_uses_invocation_deadline(self):
        track = load_bot_function('_track_gen_activity', dict(
            asyncio=asyncio, os=os, time=time, _GEN_ACTIVE_JOBS={},
            save_json=lambda *args: None, log=lambda *args: None))
        event = SimpleNamespace(chat_id=123, id=10, reply=AsyncMock())
        handler = AsyncMock(return_value={'status': 'delivered'})
        invocation = runtime.Invocation(max_seconds=.02)
        await self.jobs.gate.semaphore.acquire()
        try:
            with patch.object(runtime, 'Invocation', return_value=invocation), \
                    patch.object(runtime, 'run_event', self.jobs.run_event):
                rec = await track(handler)(event)
        finally:
            self.jobs.gate.semaphore.release()
        self.assertEqual(rec.status, 'failed')
        handler.assert_not_awaited()

    async def test_ast_wrapper_does_not_invent_success_for_bare_return(self):
        track = load_bot_function('_track_gen_activity', dict(
            asyncio=asyncio, os=os, time=time, _GEN_ACTIVE_JOBS={},
            save_json=lambda *args: None, log=lambda *args: None))
        event = SimpleNamespace(chat_id=123, id=10, reply=AsyncMock())
        with patch.object(runtime, 'run_event', self.jobs.run_event):
            rec = await track(AsyncMock(return_value=None))(event)
        self.assertEqual(rec.status, 'incomplete')
        self.assertFalse(rec.delivered)

    async def test_ast_send_preserves_image_when_auxiliary_caption_fails(self):
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8), 'blue').save(buffer, format='PNG')
        client = SimpleNamespace(send_file=AsyncMock(return_value=SimpleNamespace(id=42)),
                                 send_message=AsyncMock(side_effect=RuntimeError()))
        send = load_bot_function('_gen_send_image', dict(
            io=io, json=json, client=client, send_long=AsyncMock(side_effect=RuntimeError()),
            log=lambda *args: None))
        with patch.object(runtime, 'ARTIFACTS', self.artifacts):
            result = await send(123, buffer.getvalue(), 'image/png', 'prompt' * 300,
                                True, 10, refs_line='refs')
        self.assertEqual(result.id, 42)
        client.send_file.assert_awaited_once()
        entry = json.loads(next(delivery.ROOT.glob('*.json')).read_text())
        self.assertEqual(entry['status'], 'delivered')
        self.assertEqual(Path(entry['path']).read_bytes(), buffer.getvalue())

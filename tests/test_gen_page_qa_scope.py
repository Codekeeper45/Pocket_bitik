"""Offline command/resume -> renderer -> actual vision payload regressions."""
import io
import json
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

import bot_new as bot
from gen_jobs import ArtifactStore
from gen_runtime import CURRENT, Invocation
from gen_series import SeriesCheckpoint, parse_multipage_plan, save_series_state
from tests import test_gen_command_e2e as command_tests
from tests.test_gen_qa_status_integrity import finding


def png(color):
    output = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(output, format='PNG')
    return output.getvalue()


def response(payload):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])


class PageQAScopeTests(unittest.IsolatedAsyncioTestCase):
    setUp = command_tests.GenCommandE2ETests.setUp
    _stop_patches = command_tests.GenCommandE2ETests._stop_patches

    async def test_full_ensemble_prompt_reaches_visual_qa(self):
        create = Mock(return_value=response({'findings': []}))
        route = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        invocation = Invocation(qa_route=(route, 'offline-vision'))
        prompt = 'Source mappings and poses. ' * 300 + 'LAST_IDENTITY_SOURCE_ANCHOR'
        token = CURRENT.set(invocation)
        try:
            await bot._gen_visual_qa_impl(png('blue'), 'Check all participants', prompt)
        finally:
            CURRENT.reset(token)
        payload = create.call_args.kwargs['messages'][0]['content'][0]['text']
        self.assertIn(prompt, payload)

    async def test_initial_and_resumed_series_send_page_local_qa_with_global_constraints(self):
        source = ('Два отдельных файла: первый — чаепитие; второй — шахматы. '
                  'На обоих подписи "Flowery" и "Ториэль"; Flowery носит красный шарф. '
                  + 'Сохрани стиль. ' * 250 + 'На каждой странице надпись "Конец".')
        scenes = ['Tea party with Flowery and Ториэль.', 'Chess match with Flowery and Ториэль.']
        plan_data = dict(task_type='creation', output_kind='independent_images', output_count=2,
                         prompt='Two different scenes', scenes=scenes, explicit_text=['Flowery', 'Ториэль'],
                         constraints=['Flowery wears a red scarf.'])
        create = Mock(return_value=response({'findings': []}))
        route = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        parent = Invocation(qa_route=(route, 'offline-vision'), max_generation_calls=12)

        async def build(*args, **kwargs):
            CURRENT.get().timings['intent_plan'] = plan_data
            return 'Two different scenes', [], 'Two scenes', None

        token = CURRENT.set(parent)
        try:
            with tempfile.TemporaryDirectory() as temporary, chdir(temporary), \
                 patch.object(bot, 'OWNER_ID', 77), \
                 patch('gen_runtime.ARTIFACTS', ArtifactStore(Path(temporary) / 'artifacts')), \
                 patch.object(bot, '_build_gen_prompt', AsyncMock(side_effect=build)), \
                 patch.object(bot, '_gen_provider_call', AsyncMock(side_effect=[
                     (png(color), 'image/png') for color in ('blue', 'green') * 3])), \
                 patch.object(bot, '_gen_rate_gate', AsyncMock()), \
                 patch.object(bot, '_gen_send_image', AsyncMock(return_value=SimpleNamespace(id=700))) as send, \
                 patch.dict('os.environ', {'GEN_LAYERS_ENABLED': '0'}):
                await bot.gen_command.__wrapped__(command_tests.FakeEvent('-p2', source))
                self.assertEqual(send.await_count, 2)
                state_path = Path('gen_series_state') / f'{parent.job_id}.json'
                record = json.loads(state_path.read_text())
                for legacy in (False, True):
                    if legacy:
                        record['generation'].pop('intent_plan')
                        record['bible'] = {'characters': [{'name': 'Flowery', 'appearance': 'red scarf'}]}
                        for index, page in enumerate(record['plan']):
                            page['prompt'] = scenes[index]
                            record['pages_meta'][str(index + 1)]['prompt'] = scenes[index]
                    plan = parse_multipage_plan(json.dumps({
                        'pages': record['plan'], 'bible': record['bible'], 'version': record['version']}), 2)
                    checkpoint = SeriesCheckpoint(plan.plan_id, job_id=parent.job_id, chat_id='9001', owner_id='77')
                    record['checkpoint'] = checkpoint.to_json()
                    save_series_state(state_path, record)
                    event = SimpleNamespace(out=True, sender_id=77, chat_id=9001,
                                            pattern_match=SimpleNamespace(group=lambda _: parent.job_id), reply=AsyncMock())
                    result = await bot.gen_resume.__wrapped__(event)
                    self.assertFalse(result.failures)
                    self.assertEqual(result.delivered, {1, 2})
                self.assertEqual(send.await_count, 6)
        finally:
            CURRENT.reset(token)
        self.assertEqual(create.call_count, 6)
        for index, call in enumerate(create.call_args_list):
            request = call.kwargs['messages'][0]['content'][0]['text']
            self.assertIn('Evaluate ONLY the current page/file', request)
            self.assertIn('never report their absent scenes', request)
            self.assertIn('explicitly required on every page', request)
            self.assertIn('Mixed Latin/Cyrillic names are not a defect', request)
            document, _ = json.JSONDecoder().raw_decode(request[request.index('{"current_page"'):])
            self.assertEqual(document['current_page']['page'], index % 2 + 1)
            self.assertEqual(document['current_page']['prompt'], scenes[index % 2])
            self.assertNotIn(scenes[1 - index % 2], json.dumps(document['current_page']))
            self.assertEqual(document['global_request_context'], source)
            if index < 4:
                self.assertEqual(document['shared_constraints'], plan_data['constraints'])
                self.assertEqual(document['explicit_text_spellings_not_a_per_page_checklist'], ['Flowery', 'Ториэль'])
            else:
                self.assertEqual(document['series_bible']['characters'][0]['name'], 'Flowery')

    async def test_literal_script_rule_survives_schema_retry_without_hiding_real_text_defects(self):
        defect = finding(category='text', requirement_id='literal-name', affected_subject='Flowery',
                         description='Requested Flowery was rendered as Флауэри')
        create = Mock(side_effect=[response({'findings': 'invalid'}), response({'findings': [defect]})])
        route = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        parent = Invocation(qa_route=(route, 'offline-vision'))
        token = CURRENT.set(parent)
        try:
            with patch.object(bot, '_gen_provider_call', AsyncMock(return_value=(png('blue'), 'image/png'))), \
                 patch.object(bot, '_gen_rate_gate', AsyncMock()):
                await bot._gen_one_image('Names in Cyrillic', [], '2K', '1:1', False,
                                         'Подписи строго "Flowery" и "Ториэль".')
        finally:
            CURRENT.reset(token)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(parent.result.qa.status, 'fail')
        self.assertEqual(parent.result.qa.findings[0]['description'], defect['description'])
        for call in create.call_args_list:
            request = call.kwargs['messages'][0]['content'][0]['text']
            self.assertIn('"Flowery" и "Ториэль"', request)
            self.assertIn('even when the generation prompt differs', request)
            self.assertIn('Do not invent a uniform Cyrillic', request)
            self.assertIn('Report actual deviations', request)

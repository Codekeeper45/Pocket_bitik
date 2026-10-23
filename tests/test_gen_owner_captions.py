"""Offline planner/handler/delivery coverage for owner-facing captions and sources."""
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, chdir
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

import bot_new as bot
from gen_jobs import ArtifactStore
from gen_prompt import IntentPlan, intent_idea, intent_plan_to_dict, parse_intent_plan
from gen_runtime import CURRENT, Invocation
from gen_series import save_series_state
from tests import test_gen_command_e2e as command_tests
from tests.test_gen_command_e2e import FakeEvent


def png(color):
    output = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(output, 'PNG')
    return output.getvalue()


class OwnerCaptionTests(unittest.IsolatedAsyncioTestCase):
    _stop_patches = command_tests.GenCommandE2ETests._stop_patches

    def setUp(self):
        self.collect_inputs = bot._gen_collect_input_images
        command_tests.GenCommandE2ETests.setUp(self)

    async def run_delivery(self, kind, *, resume=False):
        multiple = kind in ('series', 'independent_images')
        ideas = ['Лиса встречает рассвет', 'Лиса смотрит на звёзды']
        payload = {'prompt': 'A fox at different times of day\n\nKeep the distant mountains.', 'idea': ideas[0],
                   'output_kind': kind if multiple else 'single_image',
                   'output_count': 2 if multiple else 1,
                   'refs': [{'ref_id': f'catalog:{index}', 'roles': ['subject' if index == 1 else 'style']}
                            for index in (1, 2, 3)]}
        if multiple:
            payload.update(ideas=ideas, scenes=['Fox at sunrise\n\nKeep the distant mountains.', 'Fox under stars\n\nKeep the distant mountains.'])
        model_message = SimpleNamespace(content=json.dumps(payload), tool_calls=[])
        create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=model_message)]))
        route = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        catalog = [
            {'idx': 1, 'bytes': png('red'), 'profile_url': 'https://t.me/alice', 'participant_id': 123},
            {'idx': 2, 'bytes': png('green'), 'source_url': 'https://example.org/fox.jpg',
             'source_page_url': 'https://example.org/fox', 'web_ref': True},
            {'idx': 3, 'bytes': png('blue'), 'mid': 456},
            {'idx': 4, 'bytes': png('yellow'), 'source_url': 'https://example.org/unused'},
        ]
        real_builder = bot._build_gen_prompt

        async def build(*args, **kwargs):
            supplied = args[5]
            if supplied is not None:
                supplied.extend(dict(item) for item in catalog)
            if kind == 'batch' and create.call_count:
                model_message.content = json.dumps({**payload, 'idea': ideas[1]})
            return await real_builder(*args, **kwargs)

        render_count = 0

        async def render(prompt, inputs, *args):
            nonlocal render_count
            render_count += 1
            self.assertEqual(len(inputs), 4 if render_count == 1 or kind != 'series' else 5)
            if resume and render_count == 2:
                return None, 'transport', prompt, False
            return png('purple' if render_count == 1 else 'orange'), 'image/png', prompt, False

        flags = '-x2' if kind == 'batch' else '-p2' if multiple else ''
        event = FakeEvent(flags, 'Нарисуй лису')
        event.get_chat = AsyncMock(return_value=SimpleNamespace(id=9001, username='source_chat'))
        event.message = SimpleNamespace(id=event.id)
        event.reply_to = 88
        event.get_reply_message = AsyncMock(return_value=SimpleNamespace(id=88, photo=True, raw_text='',
            download_media=AsyncMock(return_value=png('cyan')), get_chat=event.get_chat))
        self.client.get_me = AsyncMock(return_value=SimpleNamespace(id=77))
        self.client.send_file.side_effect = [SimpleNamespace(id=701), SimpleNamespace(id=702)]
        invocation = Invocation()
        token = CURRENT.set(invocation)
        try:
            with tempfile.TemporaryDirectory() as directory, chdir(directory), ExitStack() as stack:
                stack.enter_context(patch('gen_runtime.ARTIFACTS', ArtifactStore(Path('artifacts'))))
                stack.enter_context(patch('gen_delivery.ROOT', Path('delivery')))
                stack.enter_context(patch.object(bot, 'OWNER_ID', 77))
                stack.enter_context(patch.object(bot, 'get_active_model', return_value=(route, 'offline', 'offline')))
                stack.enter_context(patch.object(bot, 'active_model_supports_vision', return_value=True))
                stack.enter_context(patch.object(bot, '_build_gen_prompt', side_effect=build))
                stack.enter_context(patch.object(bot, '_gen_render_image', side_effect=render))
                stack.enter_context(patch.object(bot, '_gen_collect_input_images', self.collect_inputs))
                stack.enter_context(patch.object(bot, 'describe_image', AsyncMock(return_value='Reference image')))
                await bot.gen_command.__wrapped__(event)
                if resume:
                    self.assertEqual(self.client.send_file.await_count, 1, self.status.edits)
                    state_path = Path('gen_series_state') / f'{invocation.job_id}.json'
                    state = json.loads(state_path.read_text())
                    self.assertEqual(state['pages_meta']['2']['idea'], ideas[1])
                    self.assertIn('source_chat/456', state['generation']['gen_refs_line'])
                    state['pages_meta']['2'].pop('idea')
                    save_series_state(state_path, state)
                    resume_event = SimpleNamespace(out=True, sender_id=77, chat_id=9001,
                        pattern_match=SimpleNamespace(group=lambda _: invocation.job_id), reply=AsyncMock())
                    result = await bot.gen_resume.__wrapped__(resume_event)
                    self.assertFalse(result.failures)
                    self.assertEqual(result.delivered, {1, 2})
                count = 2 if multiple or kind == 'batch' else 1
                self.assertEqual(self.client.send_file.await_count, count, self.status.edits)
                for index, call in enumerate(self.client.send_file.await_args_list):
                    caption = call.kwargs['caption']
                    self.assertIn(ideas[index], caption)
                    self.assertNotIn('Preserve the user', caption)
                    self.assertNotIn('Reference attachment mapping', caption)
                    self.assertNotIn('Original user request', caption)
                    self.assertIn('💡', caption)
                    self.assertIn('Keep the distant mountains.', caption)
                sources = [call for call in self.client.send_message.await_args_list
                           if len(call.args) > 1 and call.args[1].startswith('📎')]
                self.assertEqual(len(sources), count)
                for index, call in enumerate(sources):
                    self.assertEqual(call.kwargs['reply_to'], 701 + index)
                    self.assertEqual(call.kwargs['parse_mode'], 'md')
                    self.assertIn('[Аватар 123](https://t.me/alice)', call.args[1])
                    self.assertIn('https://example.org/fox)', call.args[1])
                    self.assertIn('https://t.me/source_chat/456)', call.args[1])
                    self.assertIn('https://t.me/source_chat/88)', call.args[1])
                    self.assertNotIn('unused', call.args[1])
                    if kind == 'series' and index == 1:
                        self.assertIn('[Страница 1](https://t.me/source_chat/701)', call.args[1])
                    else:
                        self.assertNotIn('[Страница 1]', call.args[1])
                records = [json.loads(path.read_text()) for path in Path('delivery').glob('*.json')]
                self.assertEqual(len(records), count)
                self.assertTrue(all(record['status'] == 'delivered' for record in records))
                self.assertTrue(all('💡' in record['caption'] for record in records))
        finally:
            CURRENT.reset(token)

    async def test_single_actual_delivery_metadata(self):
        await self.run_delivery('single_image')

    async def test_batch_actual_delivery_metadata(self):
        await self.run_delivery('batch')

    async def test_series_actual_delivery_metadata(self):
        await self.run_delivery('series')

    async def test_independent_images_actual_delivery_metadata(self):
        await self.run_delivery('independent_images')

    async def test_resume_actual_delivery_metadata(self):
        await self.run_delivery('series', resume=True)

    def test_ideas_survive_parse_and_persisted_round_trip(self):
        payload = {'prompt': 'Full scene', 'idea': 'Short caption', 'output_count': 2,
                   'output_kind': 'series', 'ideas': ['First scene', 'Second scene']}
        plan = parse_intent_plan(payload, 'User request')
        self.assertFalse(plan.fallback_used)
        restored = IntentPlan(**json.loads(json.dumps(intent_plan_to_dict(plan))))
        self.assertEqual([intent_idea(restored, number) for number in (1, 2)], payload['ideas'])
        self.assertEqual(restored.idea, payload['idea'])

    def test_idea_schema_validation_and_legacy_scene_fallback(self):
        for invalid in ({'idea': 7}, {'ideas': 'caption'}, {'ideas': ['First', 'Second']}):
            with self.subTest(invalid=invalid):
                self.assertTrue(parse_intent_plan({'prompt': 'Scene', **invalid}, 'Request').fallback_used)
        plan = parse_intent_plan({'prompt': 'Shared scene', 'output_count': 2,
                                 'output_kind': 'series', 'scenes': ['Sunrise', 'Sunset']}, 'Request')
        self.assertEqual([intent_idea(plan, number) for number in (1, 2)], ['Sunrise', 'Sunset'])

    def test_only_packed_and_selected_page_sources_are_displayed(self):
        invocation = Invocation()
        token = CURRENT.set(invocation)
        try:
            catalog = [{'idx': 1, 'bytes': png('blue'), 'mid': 56},
                       {'idx': 2, 'mid': 57},
                       {'idx': 3, 'bytes': png('red'), 'source_url': 'https://example.org/used'}]
            inputs, roles, used, mapping = bot._gen_remap_selected_refs([], [], catalog,
                                                                       [(1, 'subject'), (2, 'style'), (3, 'style')])
            self.assertEqual(used, [1, 3])
            chat = SimpleNamespace(id=1, username='real_chat')
            generation = {'reference_registry': invocation.timings['reference_registry'],
                          'reference_sources': {str(mapping[index]): bot._gen_refs_line(chat, catalog, [index])
                                                for index in used}}
            line = bot._gen_series_refs_line(generation, ['catalog:3', 'catalog:2', 'series:anchor'])
            self.assertIn('https://example.org/used', line)
            self.assertNotIn('/56', line)
            self.assertNotIn('/57', line)
            self.assertIsNone(bot._gen_series_refs_line(generation, ['catalog:2', 'series:anchor']))
        finally:
            CURRENT.reset(token)

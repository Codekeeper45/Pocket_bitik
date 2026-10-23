import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import tests
import bot_new as bot
from gen_runtime import CURRENT, Invocation


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))])


class BuilderContractHookTests(unittest.IsolatedAsyncioTestCase):
    async def build(self, outputs):
        create = Mock(side_effect=[response(output) for output in outputs])
        llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        invocation = Invocation()
        token = CURRENT.set(invocation)
        try:
            with patch.object(bot, 'get_active_model', return_value=(llm, 'model-x', 'test')), \
                 patch.object(bot, 'active_model_supports_vision', return_value=False), \
                 patch.object(bot, 'MODEL_TOOLS_SUPPORT', {bot.ACTIVE_MODEL: False}):
                result = await bot._build_gen_prompt('Mira and exact text "OPEN LATE"', catalog=[])
        finally:
            CURRENT.reset(token)
        return result, create, invocation

    async def test_one_structured_plan_without_second_extractor(self):
        plan = {'task_type': 'creation', 'output_kind': 'single_image', 'output_count': 1,
                'prompt': 'Mira beside a bicycle', 'explicit_text': ['OPEN LATE'], 'aspect': '1:1'}
        result, create, invocation = await self.build([json.dumps(plan)])
        self.assertEqual(create.call_count, 1)
        self.assertEqual(result[3], '1:1')
        self.assertIn('Mira beside a bicycle', result[0])
        self.assertIn('OPEN LATE', result[0])
        self.assertEqual(invocation.timings['intent_plan']['output_count'], 1)
        self.assertFalse(result[0].startswith('{'))

    async def test_invalid_json_schema_gets_one_repair(self):
        result, create, invocation = await self.build(['{"prompt": 4}', '{"prompt":"Mira", "output_count":1}'])
        self.assertEqual(create.call_count, 2)
        self.assertIn('Mira and exact text "OPEN LATE"', result[0])
        self.assertFalse(invocation.timings['intent_plan']['fallback_used'])

    async def test_legacy_footer_is_read_without_an_extraction_call(self):
        result, create, invocation = await self.build(['IDEA: scene\nASPECT: 1:1\nPROMPT: Mira'])
        self.assertEqual(create.call_count, 1)
        self.assertEqual(result[2:], ('scene', '1:1'))
        self.assertIn('Mira and exact text "OPEN LATE"', result[0])
        self.assertTrue(invocation.timings['intent_plan']['fallback_used'])

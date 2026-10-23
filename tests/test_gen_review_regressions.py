import unittest
import base64
from io import BytesIO
from unittest.mock import AsyncMock, patch

from PIL import Image

import bot_new as bot


class ReviewRegressions(unittest.TestCase):
    def test_negated_natural_language_does_not_override_plan(self):
        self.assertEqual(bot._detect_pages_count('Не делай комикс на три страницы, нужен один постер'), 1)

    def test_multiple_roles_survive_selection(self):
        self.assertEqual(bot._gen_validated_selection([{'idx': 1}], [(1, 'subject'), (1, 'clothing')]),
                         [(1, 'subject'), (1, 'clothing')])

    def test_verbatim_request_numbers_are_not_rewritten(self):
        prompt = 'Write exactly "image #4" on the sign'
        self.assertEqual(bot._gen_role_number_remap(prompt, {4: 2}), prompt)

    def test_actual_gateway_count_not_looser_catalog_cap(self):
        from gen_provider import GATEWAY_PROFILE
        from gen_runtime import CURRENT, Invocation
        catalog = []
        for number in range(GATEWAY_PROFILE.max_images + 2):
            stream = BytesIO()
            Image.new('RGB', (16, 16), (number, 10, 40)).save(stream, 'PNG')
            catalog.append({'idx': number + 1, 'bytes': stream.getvalue()})
        invocation = Invocation()
        token = CURRENT.set(invocation)
        try:
            with patch.object(bot, 'GEN_IMAGE_MODEL', 'gpt-image-test'):
                packed = bot._gen_remap_selected_refs([], [], catalog, [(item['idx'], 'subject') for item in catalog])
            self.assertEqual(len(packed[0]), GATEWAY_PROFILE.max_images)
            self.assertEqual(len(invocation.timings['reference_registry']['issues']), 2)
        finally:
            CURRENT.reset(token)


class GatewayRetryRegression(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_internal_retries_are_not_repeated_outside(self):
        provider = AsyncMock(side_effect=bot.GenTransient('upstream unavailable'))
        with patch.object(bot, 'GEN_IMAGE_MODEL', 'gpt-image-test'), \
             patch.object(bot, 'CHATGPT2API_AUTH_KEY', 'inert-offline-test'), \
             patch.object(bot, '_gen_rate_gate', AsyncMock()), \
             patch.object(bot, '_gen_provider_call', provider):
            result = await bot._gen_one_image('scene', [], '2K', None, False, 'scene')
        self.assertIsNone(result[0])
        provider.assert_awaited_once()

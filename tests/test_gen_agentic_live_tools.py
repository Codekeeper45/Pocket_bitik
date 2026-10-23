"""Unit tests for agentic live Telegram tools inside /gen."""
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from types import SimpleNamespace
import json
import bot_new as bot

class TestGenAgenticLiveTools(unittest.IsolatedAsyncioTestCase):
    def test_tools_contain_live_chat_capabilities(self):
        tool_names = {t['function']['name'] for t in bot.GEN_AGENTIC_TOOLS}
        self.assertIn('chat_search', tool_names)
        self.assertIn('chat_read_context', tool_names)
        self.assertIn('chat_ant_search', tool_names)
        self.assertIn('character', tool_names)
        self.assertIn('search', tool_names)
        self.assertIn('connections', tool_names)

    async def test_agentic_enrich_calls_chat_search_and_collects_images(self):
        class FakeToolCall:
            def __init__(self, call_id, name, args):
                self.id = call_id
                self.type = 'function'
                self.function = SimpleNamespace(name=name, arguments=json.dumps(args))

        class FakeImageMsg:
            def __init__(self, mid, text=''):
                self.id = mid
                self.raw_text = text
                self.photo = True
                self.document = None

        fake_msg = FakeImageMsg(777, 'Вот свежий стикер Заи')
        tool_call_1 = FakeToolCall('tc_1', 'chat_search', {'query': 'Зая'})

        resp_with_tools = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[tool_call_1],
                reasoning_content=None
            ))]
        )
        resp_done = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='Готово',
                tool_calls=None,
                reasoning_content=None
            ))]
        )

        mock_llm = MagicMock()
        mock_llm.chat.completions.create.side_effect = [resp_with_tools, resp_done]

        async def mock_run_chat_search(chat_id, args, msg_by_id=None, include_ids=None, exclude_ids=None):
            if msg_by_id is not None:
                msg_by_id[fake_msg.id] = fake_msg
            return "Нашел сообщение #777 со стикером"

        with patch.object(bot, 'get_active_model', return_value=(mock_llm, 'test-model', 'test')), \
             patch.object(bot, 'MODEL_TOOLS_SUPPORT', {bot.ACTIVE_MODEL: True}), \
             patch.object(bot, '_run_chat_search', side_effect=mock_run_chat_search), \
             patch.object(bot, '_index_is_image_msg', return_value=True), \
             patch.object(bot, '_gen_fetch_ref_items', AsyncMock(return_value=[{'idx': 1, 'mid': 777}])) as mock_fetch:
            items, ctx = await bot._gen_agentic_enrich(chat_id=-100, user_prompt='нарисуй Заю', seed_names=[])

        mock_fetch.assert_awaited_once()
        call_args = mock_fetch.call_args[0]
        # Verify that mid 777 was passed to fetch_ref_items
        ref_hits = call_args[1]
        self.assertTrue(any(h.get('msg_id') == 777 for h in ref_hits))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['mid'], 777)

if __name__ == '__main__':
    unittest.main()

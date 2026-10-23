"""Clean regression tests for reference caption disambiguation and live chat precedence."""
import unittest
import bot_new as bot

class TestReferenceDisambiguationClean(unittest.TestCase):
    def test_catalog_ref_label_clearly_marks_message_caption(self):
        item = {
            'idx': 1,
            'mid': 331450,
            'caption': 'Лучшая девочка Нейромафии, ну или просто Зайка #фанарт',
            'desc': 'Арт крольчихи с ушами'
        }
        label = bot._gen_catalog_ref_label(item)
        self.assertIn('message_caption', label)
        self.assertIn('НЕ внутри картинки', label)

    def test_gen_ref_rules_warn_against_blind_caption_selection(self):
        self.assertIn('message_caption', bot._GEN_REF_RULES)
        self.assertIn('chat_inspect_image', bot._GEN_REF_RULES)

    def test_quote_detection_and_exact_pixel_match(self):
        quote = bot._gen_source_quote('Рефер это изображение где она говорит Я зая но можно просто зайка а другого героя придумай')
        self.assertEqual(quote, 'Я зая но можно просто зайка')
        self.assertTrue(bot._gen_quote_matches(quote, 'Я Зая,\nно можно просто Зайка.'))
        self.assertFalse(bot._gen_quote_matches(quote, 'ну или просто Зайка'))
        self.assertIsNone(bot._gen_source_quote('Напиши на картинке Привет'))

    def test_catalog_combination_keeps_fresh_chat_media_over_stale_index(self):
        fresh_items = [{'mid': 100 + i, 'idx': i} for i in range(1, 9)]
        stale_index = [{'mid': 200 + i, 'idx': i + 8} for i in range(1, 9)]
        cap = 10
        combined = fresh_items + stale_index
        seen = set()
        deduped = []
        for it in combined:
            m = it.get('mid')
            if m not in seen:
                seen.add(m)
                deduped.append(it)
        result = deduped[:cap]
        # First 8 must be fresh items, only remaining 2 are filled by index
        self.assertEqual([x['mid'] for x in result[:8]], [101, 102, 103, 104, 105, 106, 107, 108])
        self.assertEqual([x['mid'] for x in result[8:]], [201, 202])

class SourceGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_unverified_model_selection_is_rejected(self):
        from unittest.mock import patch, Mock
        from types import SimpleNamespace as NS
        llm = NS(chat=NS(completions=NS(create=Mock(return_value=NS(choices=[NS(message=NS(content='PROMPT: hug\nREFS: 1:subject\nIDEA: hug\nASPECT: 1:1', tool_calls=[]))])))))
        with patch.object(bot, 'get_active_model', return_value=(llm, 'fake', 'fake')), patch.object(bot, 'active_model_supports_vision', return_value=False):
            with self.assertRaises(bot.GenSourceUnconfirmed):
                await bot._build_gen_prompt('Стикер с фразой «Я Зая но можно просто Зайка»', catalog=[{'idx':1,'mid':11,'caption':'Я Зая но можно просто Зайка','desc':'wrong art'}], chat_id=99)

if __name__ == '__main__':
    unittest.main()

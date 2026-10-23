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


class SourceGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_unverified_model_selection_is_rejected(self):
        from unittest.mock import patch, Mock
        from types import SimpleNamespace as NS
        llm = NS(chat=NS(completions=NS(create=Mock(return_value=NS(choices=[NS(message=NS(content='PROMPT: hug\nREFS: 1:subject\nIDEA: hug\nASPECT: 1:1', tool_calls=[]))])))))
        with patch.object(bot, 'get_active_model', return_value=(llm, 'fake', 'fake')), patch.object(bot, 'active_model_supports_vision', return_value=False):
            with self.assertRaises(bot.GenSourceUnconfirmed):
                await bot._build_gen_prompt('Стикер с фразой «Я Зая но можно просто Зайка»', catalog=[{'idx':1,'mid':11,'caption':'Я Зая но можно просто Зайка','desc':'wrong art'}], chat_id=99)

    async def test_verified_tool_source_survives_full_catalog(self):
        import json
        from unittest.mock import patch, Mock, AsyncMock
        from types import SimpleNamespace as NS
        tc = NS(id='inspect1', type='function', function=NS(name='chat_inspect_image', arguments='{"message_id":77}'))
        def response(msg): return NS(choices=[NS(message=msg)])
        calls = [0]
        def create(**kw):
            if 'response_format' in kw:
                return response(NS(content=json.dumps({'prompt':'hug','immutable_requirements':['hug']})))
            calls[0] += 1
            return response(NS(content=None,tool_calls=[tc]) if calls[0]==1 else NS(content='IDEA: hug\nASPECT: 1:1\nREFS: 2:subject\nPROMPT: hug',tool_calls=[]))
        llm = NS(chat=NS(completions=NS(create=create)))
        image = NS(id=77,sender_id=2,raw_text='',date=None,sender=None,download_media=AsyncMock(return_value=b'pixels'))
        catalog=[{'idx':1,'mid':11,'caption':'wrong','desc':'wrong'}]
        with patch.object(bot,'get_active_model',return_value=(llm,'fake','fake')), patch.object(bot,'active_model_supports_vision',return_value=False), patch.object(bot,'MODEL_TOOLS_SUPPORT',{}), patch.object(bot,'GEN_CTX_IMG_MAX',1), patch.object(bot,'_gen_is_own_generation',return_value=False), patch.object(bot,'_is_attached_photo',return_value=True), patch.object(bot,'_downscale_img',AsyncMock(return_value=b'thumb')), patch.object(bot,'_label_for',return_value='user'), patch.object(bot,'_run_chat_inspect_image',AsyncMock(return_value='Я Зая, но можно просто Зайка.')) as inspect:
            result=await bot._build_gen_prompt('Стикер с фразой «Я Зая но можно просто Зайка»',catalog=catalog,chat_id=99,msg_by_id={77:image})
        self.assertIn((2,'subject'), result[1])
        self.assertEqual(catalog[-1]['mid'],77)
        self.assertTrue(inspect.await_args.args[1]['pixels_only'])

if __name__ == '__main__':
    unittest.main()

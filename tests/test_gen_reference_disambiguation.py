import unittest
from unittest.mock import AsyncMock, patch
import base64
import gen_reference_evidence as evidence
from gen_reference_evidence import SourceCriterion, SourceEvidenceUnresolved
import bot_new as bot

class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        evidence._CRITERIA_CACHE.clear(); evidence._PIXEL_CACHE.clear()

    def test_normalization_and_caption_collision(self):
        self.assertEqual(evidence.verify_image_text('Я герой,\nно можно просто Гость!','я герой но можно просто гость'),'match')
        self.assertEqual(evidence.verify_image_text('ну или просто Гость','Я герой но можно просто Гость'),'mismatch')
        self.assertEqual(evidence.verify_image_text('', 'фраза'),'unknown')

    def test_schema_rejects_invented_quote(self):
        with self.assertRaises(ValueError):
            evidence.parse_criterion({'kind':'image_text','subject':'герой','quote':'not in request'},'другой текст')

    async def test_semantic_unquoted_source_and_none_generated_text(self):
        data={'kind':'image_text','subject':'герой','quote':'привет друзья'}
        with patch.object(evidence,'json_call',AsyncMock(return_value=data)):
            c=await evidence.extract_criterion('рефер героя где он говорит привет друзья а второго придумай',object(),'model')
        self.assertEqual(c.quote,'привет друзья')
        with patch.object(evidence,'json_call',AsyncMock(return_value={'kind':'none','subject':'','quote':''})):
            self.assertIsNone(await evidence.extract_criterion('добавь текст привет друзья',object(),'model'))

    async def test_caption_is_not_evidence(self):
        c=SourceCriterion('герой','image_text','привет друзья')
        with patch.object(evidence,'inspect_pixels',AsyncMock(return_value={'visible_text':'','appearance':'animal','source_form':'artwork'})):
            selected,trace=await evidence.confirm_source(c,[{'mid':10,'idx':1,'bytes':b'image','caption':'привет друзья'}],[],None,'model')
        self.assertIsNone(selected);self.assertEqual(trace[0]['verdict'],'unknown')

    async def test_pixels_positive_dedup_and_initial_refs(self):
        c=SourceCriterion('герой','image_text','привет друзья')
        obs={'visible_text':'Привет, друзья!','appearance':'human','source_form':'sticker'}
        with patch.object(evidence,'inspect_pixels',AsyncMock(return_value=obs)) as inspect:
            selected,trace=await evidence.confirm_source(c,[],[base64.b64encode(b'img').decode()],None,'model')
        self.assertTrue(selected['explicit']);self.assertEqual(selected['idx'],1);inspect.assert_awaited_once()

    async def test_vision_failure_unknown(self):
        with patch.object(evidence,'inspect_pixels',AsyncMock(side_effect=TimeoutError)):
            selected,trace=await evidence.confirm_source(SourceCriterion('x','image_text','hello'),[{'mid':1,'bytes':b'x'}],[],None,'m')
        self.assertIsNone(selected);self.assertEqual(trace[0]['verdict'],'unknown')

    async def test_runtime_rejects_before_unchecked_builder(self):
        with patch.object(bot,'_gen_source_preflight',AsyncMock(side_effect=SourceEvidenceUnresolved('уточни'))),patch.object(bot,'_build_gen_prompt_unchecked',AsyncMock()) as build:
            with self.assertRaises(SourceEvidenceUnresolved):await bot._build_gen_prompt('request')
        build.assert_not_awaited()

    async def test_runtime_preserves_confirmed_ref_even_when_planner_omits(self):
        chosen={'idx':2,'mid':22,'bytes':b'x','source_pixel_evidence':{'appearance':'human face','visible_text':'hello'}}
        cat=[{'idx':1,'mid':11},chosen]
        with patch.object(bot,'_gen_source_preflight',AsyncMock(return_value=chosen)),patch.object(bot,'_build_gen_prompt_unchecked',AsyncMock(return_value=('prompt',[],None,None))):
            result=await bot._build_gen_prompt('request',catalog=cat)
        self.assertEqual(result[1],[(2,'subject')]);self.assertIn(chosen,cat)

    async def test_other_subject_reference_preserved_but_wrong_identity_dropped(self):
        chosen={'idx':2,'mid':22,'bytes':b'x','source_pixel_evidence':{'appearance':'human','visible_text':'hello'}}
        cat=[{'idx':1,'mid':11},chosen,{'idx':3,'mid':33}]
        with patch.object(bot,'_gen_source_preflight',AsyncMock(return_value=chosen)),patch.object(bot,'_build_gen_prompt_unchecked',AsyncMock(return_value=('two subjects',[(1,'subject'),(3,'subject')],None,None))),patch.object(evidence,'json_call',AsyncMock(return_value={'keep_ids':[3]})):
            result=await bot._build_gen_prompt('two subjects',catalog=cat)
        self.assertEqual(result[1],[(3,'subject'),(2,'subject')])

    async def test_preflight_recovery_uses_phrase_image_space(self):
        c=SourceCriterion('hero','image_text','hello friends')
        msg=type('Message',(),{'id':12,'sender_id':7,'document':type('Doc',(),{'id':5})(),'raw_text':'caption','download_media':AsyncMock(return_value=b'pixels')})()
        chosen={'idx':1,'mid':12,'bytes':b'pixels','source_pixel_evidence':{'visible_text':'hello friends','appearance':'human'}}
        with patch.object(bot,'get_image_desc_client',return_value=(None,'m')),patch.object(evidence,'extract_criterion',AsyncMock(return_value=c)),patch.object(evidence,'confirm_source',AsyncMock(side_effect=[(None,[]),(chosen,[])])),patch.object(bot,'_index_embed_query',AsyncMock(return_value='vector')) as embed,patch.object(bot,'_index_vector_search',AsyncMock(return_value=[{'key':12}])) as search,patch.object(bot.client,'get_messages',AsyncMock(return_value=[msg])),patch.object(bot,'_index_is_image_msg',return_value=True),patch.object(evidence,'append_trace'):
            cat=[];result=await bot._gen_source_preflight('full scene request',cat,chat_id=99)
        embed.assert_awaited_once_with('hello friends',image_space=True)
        search.assert_awaited_once_with(99,'media_image','vector',48)
        self.assertEqual(result['mid'],12)

if __name__=='__main__':unittest.main()

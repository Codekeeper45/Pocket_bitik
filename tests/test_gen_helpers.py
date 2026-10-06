import asyncio
import unittest
from PIL import Image
from io import BytesIO

from gen_references import ReferenceRegistry, make_reference, parse_telegram_link, resize_image, filter_refs, rewrite_ref_numbers
from gen_prompt import build_contract_prompt, extract_contract, render_prompt, ContractError
from gen_series import parse_multipage_plan, SeriesPlanError, execute_series, SeriesCheckpoint


def image_bytes(size=(1200, 800), color='red'):
    b=BytesIO(); Image.new('RGB', size, color).save(b, 'PNG'); return b.getvalue()

class ReferencesTest(unittest.TestCase):
    def test_dedup_scoped_and_stable_ids_chat_pair(self):
        raw=image_bytes(); a=make_reference(10,7,raw,role='subject',priority=4)
        reg=ReferenceRegistry(); self.assertIs(reg.add(a),a)
        self.assertIs(reg.add(make_reference(20,7,raw)),a) # hash duplicate
        other=reg.add(make_reference(20,7,image_bytes(color='blue')))
        self.assertEqual(len(reg.refs),2)
        self.assertEqual([r.chat_id for r in filter_refs(reg.refs,chat_id=20)],[20])
        self.assertEqual(reg.selected([a.ref_id])[0].api_index,1)
        with self.assertRaises(KeyError): reg.selected(['absent'])
        self.assertNotEqual(a.ref_id,other.ref_id)
    def test_resize_and_links(self):
        data=resize_image(image_bytes((2200,1600)),max_side=640,max_bytes=150000)
        im=Image.open(BytesIO(data)); self.assertLessEqual(max(im.size),640); self.assertLessEqual(len(data),150000)
        link=parse_telegram_link('https://t.me/s/channel_name/42?thread=8#x','untrusted caption')
        self.assertEqual((link.chat,link.message_id,link.topic_id,link.caption),('channel_name',42,8,'untrusted caption'))
        link=parse_telegram_link('https://t.me/c/123456/88'); self.assertEqual(link.message_id,88)
        with self.assertRaises(ValueError): parse_telegram_link('https://t.me/+invite')
    def test_role_mapping(self): self.assertEqual(rewrite_ref_numbers('REF #2 and REF 3',{2:1}), 'REF #1 and REF 3')

class PromptTest(unittest.TestCase):
    def test_required_fields_aspect_and_refs(self):
        c=build_contract_prompt('Draw Ada',participants=('Ada',),participant_count=1,required_text=('HELLO',),aspect='1:1',refs=({'ref_id':'r','role':'subject'},))
        self.assertEqual(extract_contract(render_prompt(c,user_aspect='16:9')).aspect,'16:9') # explicit user override serialized
        self.assertIn('16:9',render_prompt(c,user_aspect='16:9'))
        with self.assertRaises(ContractError): render_prompt(c,refs=[])
        with self.assertRaises(ContractError): extract_contract({'task_type':'creation','prompt':'x','participant_count':1,'participants':['a','b']})

class SeriesTest(unittest.TestCase):
    def test_strict_parser(self):
        p=parse_multipage_plan('PAGE 1:\nA walks.\nPAGE 2:\nB arrives.',2,user_aspect='16:9')
        self.assertEqual([x.number for x in p.pages],[1,2]); self.assertEqual(p.pages[0].aspect,'16:9')
        for txt in ('PAGE 1:\nA\nPAGE 1:\nB','PAGE 1:\nA', 'PAGE 1:\nA\nPAGE 2:\nA'):
            with self.subTest(txt=txt), self.assertRaises(SeriesPlanError): parse_multipage_plan(txt,2)
        with self.assertRaises(SeriesPlanError): parse_multipage_plan('bad',2)
    def test_checkpoint_resume_and_delivery(self):
        plan=parse_multipage_plan('PAGE 1:\nOne\nPAGE 2:\nTwo\nPAGE 3:\nThree',3)
        calls=[]; delivered=[]
        async def gen(page,**kw):
            calls.append(page.number)
            if page.number==2: raise RuntimeError('provider failure')
            return f'img{page.number}'
        async def deliver(page,result): delivered.append((page.number,result))
        state=asyncio.run(execute_series(plan,generate=gen,deliver=deliver))
        self.assertEqual(state.completed,{1:'img1'}); self.assertEqual(delivered,[(1,'img1')]); self.assertIn(2,state.failures)
        async def gen2(page,**kw): calls.append(page.number); return f'img{page.number}'
        state2=asyncio.run(execute_series(plan,generate=gen2,checkpoint=state,deliver=deliver))
        self.assertEqual(state2.completed,{1:'img1',2:'img2',3:'img3'}); self.assertEqual(calls,[1,2,2,3])
        with self.assertRaises(SeriesPlanError): asyncio.run(execute_series(plan,generate=gen2,checkpoint=SeriesCheckpoint('other')))

if __name__=='__main__': unittest.main()

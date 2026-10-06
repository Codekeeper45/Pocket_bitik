import io,json,unittest
from types import SimpleNamespace
from PIL import Image
from gen_pair_qa import compare_images
class PairQA(unittest.IsolatedAsyncioTestCase):
 def png(self):
  buf=io.BytesIO();Image.new('RGB',(20,20),'blue').save(buf,format='PNG');return buf.getvalue()
 def response(self,data):
  return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])
 def client(self,data):
  return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:self.response(data))))
 def passing(self):
  return dict(target_evidence=[dict(target='hand',original_evidence='six fingers visible',candidate_evidence='five fingers visible',improved=True)],identity_preserved=True,requirements_preserved=True,no_new_defects=True,no_seams=True,confidence=.95)
 async def test_uncertain_or_regression_rejected(self):
  raw=self.png()
  for bad in ['identity_preserved','requirements_preserved','no_new_defects','no_seams']:
   data=self.passing();data[bad]=False
   self.assertFalse(await compare_images(raw,raw,'fix hand',client=self.client(data),model='mock'))
 async def test_requires_complete_nonempty_target_evidence(self):
  raw=self.png()
  for evidence in ([],[{'target':'hand','original_evidence':'bad','candidate_evidence':'fixed'}],[{'target':'hand','original_evidence':'bad','candidate_evidence':'fixed','improved':False}],[{'target':' ','original_evidence':'bad','candidate_evidence':'fixed','improved':True}]):
   data=self.passing();data['target_evidence']=evidence
   self.assertFalse(await compare_images(raw,raw,'fix hand',client=self.client(data),model='mock'))
 async def test_true_result_uses_pinned_model_and_two_real_pngs(self):
  from unittest.mock import Mock
  raw=self.png(); data=self.passing(); reply=self.response(data); create=Mock(return_value=reply)
  client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
  self.assertTrue(await compare_images(raw,raw,'fix hand',client=client,model='pinned-model'))
  kw=create.call_args.kwargs;self.assertEqual(kw['model'],'pinned-model')
  content=kw['messages'][0]['content'];self.assertIn('fix hand',content[0]['text'])
  self.assertEqual([x['text'] for x in content if x['type']=='text'][1:3],['ORIGINAL','CANDIDATE'])
  self.assertTrue(all(x['image_url']['url'].startswith('data:image/png;base64,') for x in content if x['type']=='image_url'))
 async def test_unknown_fields_fail_closed(self):
  # The contract currently validates required fields/evidence and is tolerant of extensions.
  # Unknown-field rejection is exercised by the strict findings parser, not this comparator.
  raw=self.png();data=self.passing();data['unexpected']=True
  self.assertTrue(await compare_images(raw,raw,'fix hand',client=self.client(data),model='mock'))
 async def test_missing_choices_is_not_clean(self):
  buf=io.BytesIO();Image.new('RGB',(20,20)).save(buf,format='PNG');raw=buf.getvalue()
  client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:SimpleNamespace(choices=[]))))
  self.assertFalse(await compare_images(raw,raw,'fix',client=client,model='mock'))

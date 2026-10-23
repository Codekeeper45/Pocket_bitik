import os,io,unittest
from unittest.mock import patch,AsyncMock
from PIL import Image
from gen_jobs import GlobalRateGate
os.environ.setdefault('api_id','12345');os.environ.setdefault('api_hash','deadbeefdeadbeefdeadbeefdeadbeef')
import bot_new as bot

def raw(color):
 b=io.BytesIO();Image.new('RGB',(1024,1024),color).save(b,format='PNG');return b.getvalue()
def finding(box):return {'id':'face-1','category':'anatomy','severity':'medium','confidence':.9,'description':'bad face','location':'head','bbox':box,'affected_subject':'person','requirement_id':'face'}

def global_finding(description, location='global'):
 return {'id':'global-1','category':'composition','severity':'medium','confidence':.9,'description':description,'location':location,'bbox':None,'affected_subject':None,'requirement_id':None}
class RegionIntegration(unittest.IsolatedAsyncioTestCase):
 def setUp(self):
  gate=patch.object(bot,'_GEN_PROVIDER_GATE',GlobalRateGate(max_concurrent=2,min_interval=0))
  gate.start()
  self.addCleanup(gate.stop)
 async def test_multiple_zones_one_detection_pass(self):
  first=raw('blue'); fixed=raw('green');a=finding([.1,.1,.2,.2]);b=dict(finding([.7,.7,.8,.8]),id='face-2');qa={'findings':[a,b]}
  inspect=AsyncMock(side_effect=[{'findings':[]},{'findings':[b]},{'findings':[]},{'findings':[global_finding('background spacing','global')]}])
  with patch('gen_pair_qa.compare_images',AsyncMock(return_value=True)),patch.object(bot,'get_image_desc_client',return_value=(object(),'qa')),patch.object(bot,'_sync_generate_image',return_value=(fixed,'image/png')) as gen,patch.object(bot,'_gen_visual_qa',inspect),patch.object(bot,'_gen_rate_gate',AsyncMock()):
   result,_=await bot._gen_repair_regions(first,'image/png',qa,'user','original','gpt-image-2.5-sunburst','2K')
  self.assertEqual(gen.call_count,2);self.assertNotEqual(result,first)
  im=Image.open(io.BytesIO(result));self.assertEqual(im.getpixel((500,500)),(0,0,255))
  self.assertTrue(inspect.call_args.args[2].startswith('original'))
 async def test_crop_qa_compares_blended_candidate_that_will_be_committed(self):
  original=raw('blue');f=finding([.1,.1,.2,.2]);seen=[]
  async def pair(before,after,*args,**kwargs):
   seen.append((Image.open(io.BytesIO(before)).convert('RGB'),Image.open(io.BytesIO(after)).convert('RGB')))
   return True
  with patch('gen_pair_qa.compare_images',side_effect=pair),patch.object(bot,'get_image_desc_client',return_value=(object(),'qa')),patch.object(bot,'_sync_generate_image',return_value=(raw('green'),'image/png')),patch.object(bot,'_gen_visual_qa',AsyncMock(return_value={'findings':[]})),patch.object(bot,'_gen_rate_gate',AsyncMock()):
   result,_=await bot._gen_repair_regions(original,'image/png',{'findings':[f]},'user','original','model','2K')
  self.assertNotEqual(result,original)
  # Feathering preserves original edge pixels; the compared candidate must match
  # the accepted composite exactly, rather than the raw green generated patch.
  self.assertEqual(seen[0][1].getpixel((0,0)),seen[0][0].getpixel((0,0)))
  self.assertNotEqual(seen[0][1].getpixel((seen[0][1].width//2,seen[0][1].height//2)),seen[0][0].getpixel((seen[0][1].width//2,seen[0][1].height//2)))
 async def test_seam_and_identity_categories_reject_including_critical(self):
  original=raw('blue');target=finding([.1,.1,.2,.2])
  for category in ('seam','identity'):
   for severity in ('medium','high','critical'):
    with self.subTest(category=category,severity=severity):
     defect=dict(target,id='new-defect',category=category,severity=severity,description='境界で色が急変する')
     with patch('gen_pair_qa.compare_images',AsyncMock(return_value=True)) as pair,patch.object(bot,'get_image_desc_client',return_value=(object(),'qa')),patch.object(bot,'_sync_generate_image',return_value=(raw('green'),'image/png')) as generate,patch.object(bot,'_gen_visual_qa',AsyncMock(side_effect=[{'findings':[]},{'findings':[defect]}])),patch.object(bot,'_gen_rate_gate',AsyncMock()):
      result,_=await bot._gen_repair_regions(original,'image/png',{'findings':[target]},'user','original','model','2K')
     self.assertEqual(generate.call_count,1)
     self.assertEqual(pair.await_count,2)
     self.assertEqual(result,original)
 async def test_unrelated_minor_findings_and_seam_words_do_not_veto_target(self):
  original=raw('blue');target=dict(finding([.1,.1,.2,.2]),description='可視の形状の欠陥')
  for description in ('visible seam halo','шов вокруг героя','背景の余白'):
   with self.subTest(description=description):
    unrelated=global_finding(description)
    with patch('gen_pair_qa.compare_images',AsyncMock(return_value=True)) as pair,patch.object(bot,'get_image_desc_client',return_value=(object(),'qa')),patch.object(bot,'_sync_generate_image',return_value=(raw('green'),'image/png')) as generate,patch.object(bot,'_gen_visual_qa',AsyncMock(side_effect=[{'findings':[]},{'findings':[unrelated]}])),patch.object(bot,'_gen_rate_gate',AsyncMock()):
     result,_=await bot._gen_repair_regions(original,'image/png',{'findings':[target]},'user','original','model','2K')
    self.assertEqual(generate.call_count,1)
    self.assertNotEqual(result,original)
    self.assertTrue(all(call.kwargs['targets']==['face-1'] for call in pair.await_args_list))
 async def test_failed_qa_or_missing_bbox_preserves_bytes(self):
  original=raw('blue')
  for valid,answer in [(True,None),(True,{'findings':[finding([.1,.1,.2,.2])]}),(False,{'findings':[]})]:
   f=finding([.1,.1,.2,.2]);
   if not valid:f.pop('bbox')
   with patch('gen_pair_qa.compare_images',AsyncMock(return_value=True)),patch.object(bot,'get_image_desc_client',return_value=(object(),'qa')),patch.object(bot,'_sync_generate_image',return_value=(raw('green'),'image/png')),patch.object(bot,'_gen_visual_qa',AsyncMock(return_value=answer)),patch.object(bot,'_gen_rate_gate',AsyncMock()):
    result,_=await bot._gen_repair_regions(original,'image/png',{'findings':[f]},'user','original','model','2K')
   self.assertEqual(result,original)
if __name__=='__main__':unittest.main()

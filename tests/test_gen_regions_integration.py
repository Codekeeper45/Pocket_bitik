import os,io,unittest
from unittest.mock import patch,AsyncMock
from PIL import Image
os.environ.setdefault('api_id','12345');os.environ.setdefault('api_hash','deadbeefdeadbeefdeadbeefdeadbeef')
import bot_new as bot

def raw(color):
 b=io.BytesIO();Image.new('RGB',(1024,1024),color).save(b,format='PNG');return b.getvalue()
def finding(box):return {'severity':'medium','confidence':.9,'issue':'bad face','location':'head','bbox':box}
class RegionIntegration(unittest.IsolatedAsyncioTestCase):
 async def test_multiple_zones_one_detection_pass(self):
  first=raw('blue'); fixed=raw('green');a=finding([.1,.1,.2,.2]);b=finding([.7,.7,.8,.8]);qa={'findings':[a,b]}
  inspect=AsyncMock(side_effect=[{'findings':[]},{'findings':[b]},{'findings':[]},{'findings':[]}])
  with patch('gen_pair_qa.compare_images',AsyncMock(return_value=True)),patch.object(bot,'get_image_desc_client',return_value=(object(),'qa')),patch.object(bot,'_sync_generate_image',return_value=(fixed,'image/png')) as gen,patch.object(bot,'_gen_visual_qa',inspect),patch.object(bot,'_gen_rate_gate',AsyncMock()):
   result,_=await bot._gen_repair_regions(first,'image/png',qa,'user','original','gpt-image-2.5-sunburst','2K')
  self.assertEqual(gen.call_count,2);self.assertNotEqual(result,first)
  im=Image.open(io.BytesIO(result));self.assertEqual(im.getpixel((500,500)),(0,0,255))
  self.assertTrue(inspect.call_args.args[2].startswith('original'))
 async def test_seam_rejects_otherwise_improved_composite(self):
  original=raw('blue');f=finding([.1,.1,.2,.2])
  seam={'severity':'medium','confidence':.9,'issue':'visible seam halo','location':'patch edge'}
  with patch.object(bot,'_sync_generate_image',return_value=(raw('green'),'image/png')),patch.object(bot,'_gen_visual_qa',AsyncMock(side_effect=[{'findings':[]},{'findings':[seam]}])),patch.object(bot,'_gen_rate_gate',AsyncMock()):
   result,_=await bot._gen_repair_regions(original,'image/png',{'findings':[f]},'user','original','model','2K')
  self.assertEqual(result,original)
 async def test_failed_qa_or_missing_bbox_preserves_bytes(self):
  original=raw('blue')
  for valid,answer in [(True,None),(True,{'findings':[finding([.1,.1,.2,.2])]}),(False,{'findings':[]})]:
   f=finding([.1,.1,.2,.2]);
   if not valid:f.pop('bbox')
   with patch.object(bot,'_sync_generate_image',return_value=(raw('green'),'image/png')),patch.object(bot,'_gen_visual_qa',AsyncMock(return_value=answer)),patch.object(bot,'_gen_rate_gate',AsyncMock()):
    result,_=await bot._gen_repair_regions(original,'image/png',{'findings':[f]},'user','original','model','2K')
   self.assertEqual(result,original)
if __name__=='__main__':unittest.main()

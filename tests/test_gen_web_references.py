import unittest,asyncio,io,socket
from PIL import Image
from gen_web_references import _validate_url,UnsafeURL,inspect_image,search_and_fetch
class WebRefTests(unittest.TestCase):
 def test_reject_private(self):
  for u in ['http://127.0.0.1/a','http://169.254.169.254/a','file:///etc/passwd','https://user:pass@example.org/a','http://localhost/a','http://[::1]/a']:
   with self.assertRaises(UnsafeURL):_validate_url(u)
 def test_reject_private_dns(self):
  with self.assertRaises(UnsafeURL):_validate_url('http://example.org/a',lambda *a,**kw:[(2,1,6,'',('10.0.0.1',80))])
 def test_image_validation(self):
  with self.assertRaises(Exception):inspect_image(b'<html>no image</html>')
 def test_tavily_images_shape(self):
  b=io.BytesIO();Image.new('RGB',(10,10)).save(b,'PNG')
  r=asyncio.run(search_and_fetch('public style',lambda q:{'images':['https://example.org/a.png']},fetch_fn=lambda u:(b.getvalue(),u)))
  self.assertEqual(len(r),1);self.assertEqual(r[0]['source_status'],'unverified search result')
 def test_no_results_is_not_fake(self):
  self.assertEqual(asyncio.run(search_and_fetch('public style',lambda q:{'images':[]})),[])

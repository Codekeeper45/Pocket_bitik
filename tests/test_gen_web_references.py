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
  queries=[]
  def search(query):
   queries.append(query)
   return {'images':['https://example.org/a.png']}
  r=asyncio.run(search_and_fetch('public style',search,fetch_fn=lambda u:(b.getvalue(),u),public_terms=('public style',)))
  self.assertEqual(queries,['public style'])
  self.assertEqual(len(r),1);self.assertEqual(r[0]['source_status'],'unverified search result')
  self.assertEqual(r[0]['image_url'],r[0]['source_url']);self.assertIsNone(r[0]['source_page_url'])
 def test_no_results_is_not_fake(self):
  self.assertEqual(asyncio.run(search_and_fetch('public style',lambda q:{'images':[]},public_terms=('public style',))),[])
 def test_unapproved_query_never_reaches_search(self):
  queries=[]
  self.assertEqual(asyncio.run(search_and_fetch('our private inside joke',lambda q:queries.append(q))),[])
  self.assertEqual(queries,[])

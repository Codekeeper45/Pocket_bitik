import unittest
from io import BytesIO
from PIL import Image,ImageDraw
from gen_series_duplicates import duplicate_page,assert_distinct_page

def sample(side,fmt='PNG'):
 im=Image.new('RGB',(256,256),'white');d=ImageDraw.Draw(im)
 d.rectangle((side,30,side+65,220),fill='purple');b=BytesIO();im.save(b,fmt);return b.getvalue()
class DuplicateTests(unittest.TestCase):
 def test_exact(self):self.assertTrue(duplicate_page(sample(10),[sample(10)]))
 def test_recompressed(self):self.assertTrue(duplicate_page(sample(10,'JPEG'),[sample(10)]))
 def test_same_character_new_composition(self):self.assertFalse(duplicate_page(sample(150),[sample(10)]))
 def test_reject(self):
  with self.assertRaisesRegex(ValueError,'duplicate'):assert_distinct_page(sample(10),[sample(10)])
if __name__=='__main__':unittest.main()

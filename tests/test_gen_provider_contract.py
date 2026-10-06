import unittest
from gen_provider import gateway_dimensions,capability_notice,classify_provider_error
class HonestProvider(unittest.TestCase):
 def test_native_ratios_not_requested_ratios(self):
  self.assertEqual(gateway_dimensions('16:9'),(1536,1024))
  self.assertTrue(capability_notice('16:9','4K',(1536,1024)))
  self.assertFalse(capability_notice('1:1','1K',(1024,1024)))
 def test_exif_normalizes_pixels_and_bytes(self):
  import io
  from PIL import Image
  from gen_provider import validate_image
  b=io.BytesIO();im=Image.new('RGB',(30,10));exif=im.getexif();exif[274]=6;im.save(b,format='JPEG',exif=exif)
  result=validate_image(b.getvalue())
  self.assertEqual(result.pixel_size,(10,30))
  self.assertEqual(Image.open(io.BytesIO(result.data)).size,result.pixel_size)
  self.assertEqual(result.mime_type,'image/png')
 def test_authentication_is_not_moderation(self):
  self.assertEqual(classify_provider_error(401).kind,'authentication')
  self.assertEqual(classify_provider_error(400).kind,'invalid_request')
  self.assertEqual(classify_provider_error(400,'content_policy_violation').kind,'moderation')

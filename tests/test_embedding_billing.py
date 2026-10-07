import tests
import unittest,requests
from unittest.mock import Mock,patch
import bot_new as bot
class Billing(unittest.TestCase):
 def test_billing_does_not_retry_text_or_image(self):
  for fn,arg in [(bot._sync_embed_texts,['probe']),(bot._sync_embed_image,b'probe')]:
   resp=Mock();resp.status_code=402;resp.raise_for_status.side_effect=requests.HTTPError(response=resp)
   with patch.object(bot.requests,'post',return_value=resp) as post,patch.object(bot.time,'sleep') as sleep:
    with self.assertRaises(requests.HTTPError):fn(arg)
    self.assertEqual(post.call_count,1);sleep.assert_not_called()

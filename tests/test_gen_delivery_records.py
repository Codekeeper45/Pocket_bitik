import tests
import os,io,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,AsyncMock
from PIL import Image
import bot_new as bot
import gen_delivery as delivery
from gen_runtime import ARTIFACTS
class DeliveryRecovery(unittest.IsolatedAsyncioTestCase):
 async def test_send_timeout_not_duplicated_and_recorded(self):
  buffer=io.BytesIO();Image.new('RGB',(32,32),'blue').save(buffer,format='PNG')
  with tempfile.TemporaryDirectory() as d,patch.object(delivery,'ROOT',Path(d)/'records'),patch.object(ARTIFACTS,'root',Path(d)),patch.object(bot.client,'send_file',AsyncMock(side_effect=TimeoutError('private-secret'))) as send:
   with self.assertRaises(TimeoutError):await bot._gen_send_image(123,buffer.getvalue(),'image/png','prompt',True,10)
   self.assertEqual(send.await_count,1)
   entries=list((Path(d)/'records').glob('*.json'));self.assertEqual(len(entries),1)
   import json
   obj=json.loads(entries[0].read_text());self.assertEqual(obj['status'],'ambiguous')
   self.assertNotIn('private-secret',entries[0].read_text());self.assertIn(delivery.marker(obj['key']),obj['caption'])
 async def test_retry_failed_readback_never_sends(self):
  event=SimpleNamespace(out=True,sender_id=42,chat_id=123,id=1,pattern_match=SimpleNamespace(group=lambda n:'valid'),reply=AsyncMock())
  entry={'key':'valid','chat':123,'status':'ambiguous','path':'x','reply_to':1}
  with patch.object(bot,'OWNER_ID',42),patch.object(delivery,'lookup',return_value=entry),patch.object(delivery,'reconcile',AsyncMock(side_effect=TimeoutError())),patch.object(bot.client,'send_file',AsyncMock()) as send:
   await bot.gen_retry_file(event)
   send.assert_not_awaited()
 async def test_reconcile_matches_own_exact_marker(self):
  client=SimpleNamespace(get_messages=AsyncMock(return_value=[SimpleNamespace(out=False,raw_text=delivery.marker('key'),id=1),SimpleNamespace(out=True,raw_text=delivery.marker('key'),id=2)]))
  msg=await delivery.reconcile(client,{'chat':123,'key':'key'});self.assertEqual(msg.id,2)

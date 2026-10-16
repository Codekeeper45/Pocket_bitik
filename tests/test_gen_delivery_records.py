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
 def test_output_ledger_and_conservative_provenance(self):
  with tempfile.TemporaryDirectory() as d,patch.object(delivery,'ROOT',Path(d)):
   delivery.record_output(chat=123,message_id=456,media_id=789,artifact_key='abc')
   self.assertTrue(delivery.known_output('123',456))
   self.assertTrue(delivery.known_output(123,999,789))
   self.assertFalse(delivery.known_output(124,456))
   self.assertTrue(bot._gen_is_own_generation(SimpleNamespace(out=True,chat_id=123,id=456,media=None,raw_text='')))
   self.assertFalse(bot._gen_is_own_generation(SimpleNamespace(out=True,chat_id=123,id=457,media=None,raw_text='')))
   self.assertFalse(bot._gen_is_own_generation(SimpleNamespace(out=True,chat_id=123,id=458,media=None,raw_text='ordinary image caption')))

 def test_catalog_label_distinguishes_provenance(self):
  base={'idx':1}
  self.assertIn('сгенерировано этим ботом',bot._gen_catalog_ref_label({**base,'provenance':'generated'}))
  self.assertIn('загружено пользователем',bot._gen_catalog_ref_label({**base,'provenance':'upload'}))
  self.assertIn('источник неизвестен',bot._gen_catalog_ref_label(base))

 async def test_send_timeout_not_duplicated_and_recorded(self):
  buffer=io.BytesIO();Image.new('RGB',(32,32),'blue').save(buffer,format='PNG')
  with tempfile.TemporaryDirectory() as d,patch.object(delivery,'ROOT',Path(d)/'records'),patch.object(ARTIFACTS,'root',Path(d)),patch.object(bot.client,'send_file',AsyncMock(side_effect=TimeoutError('private-secret'))) as send:
   with self.assertRaises(TimeoutError):await bot._gen_send_image(123,buffer.getvalue(),'image/png','prompt',True,10)
   self.assertEqual(send.await_count,1)
   entries=list((Path(d)/'records').glob('*.json'));self.assertEqual(len(entries),1)
   import json
   obj=json.loads(entries[0].read_text());self.assertEqual(obj['status'],'ambiguous')
   self.assertNotIn('private-secret',entries[0].read_text());self.assertNotIn('gen-artifact:',obj['caption']);self.assertNotIn('QA',obj['caption'])
 async def test_retry_failed_readback_never_sends(self):
  event=SimpleNamespace(out=True,sender_id=42,chat_id=123,id=1,pattern_match=SimpleNamespace(group=lambda n:'valid'),reply=AsyncMock())
  entry={'key':'valid','chat':123,'status':'ambiguous','path':'x','reply_to':1}
  with patch.object(bot,'OWNER_ID',42),patch.object(delivery,'lookup',return_value=entry),patch.object(delivery,'reconcile',AsyncMock(side_effect=TimeoutError())),patch.object(bot.client,'send_file',AsyncMock()) as send:
   await bot.gen_retry_file(event)
   send.assert_not_awaited()
 async def test_reconcile_matches_own_exact_marker(self):
  client=SimpleNamespace(get_messages=AsyncMock(return_value=[SimpleNamespace(out=False,raw_text=delivery.marker('key'),id=1),SimpleNamespace(out=True,raw_text=delivery.marker('key'),id=2)]))
  msg=await delivery.reconcile(client,{'chat':123,'key':'key','caption':delivery.marker('key')});self.assertEqual(msg.id,2)

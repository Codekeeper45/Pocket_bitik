import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from gen_avatar import fetch_avatar
class AvatarTests(unittest.IsolatedAsyncioTestCase):
 async def test_current_main_not_history(self):
  entity=SimpleNamespace(id=123,username='tester',photo=SimpleNamespace(photo_id=987))
  c=SimpleNamespace(get_entity=AsyncMock(return_value=entity),get_permissions=AsyncMock(),get_profile_photos=AsyncMock(),download_profile_photo=AsyncMock(return_value=b'current-main'))
  r=await fetch_avatar(c,99,'123',7,lambda raw:(raw,'image/jpeg',10,10))
  self.assertEqual(c.get_entity.await_args_list[0].args,(123,))
  c.download_profile_photo.assert_awaited_once_with(entity,file=bytes,download_big=True)
  c.get_profile_photos.assert_not_awaited()
  self.assertEqual(r['bytes'],b'current-main');self.assertEqual(r['avatar_photo_id'],987)
  self.assertEqual(r['profile_url'],'https://t.me/tester')
 async def test_nonmember_never_downloaded(self):
  c=SimpleNamespace(get_entity=AsyncMock(return_value=SimpleNamespace(id=1)),get_permissions=AsyncMock(side_effect=ValueError('not member')),download_profile_photo=AsyncMock())
  with self.assertRaises(ValueError):await fetch_avatar(c,99,'123',1,None)
  c.download_profile_photo.assert_not_awaited()
 async def test_missing_main(self):
  c=SimpleNamespace(get_entity=AsyncMock(return_value=SimpleNamespace(id=1)),get_permissions=AsyncMock(),download_profile_photo=AsyncMock(return_value=None))
  with self.assertRaises(ValueError):await fetch_avatar(c,99,'1',1,None)

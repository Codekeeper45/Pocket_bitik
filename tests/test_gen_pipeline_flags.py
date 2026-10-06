import os, unittest
from unittest.mock import patch, AsyncMock
os.environ.setdefault('api_id','12345');os.environ.setdefault('api_hash','deadbeefdeadbeefdeadbeefdeadbeef')
import bot_new as bot

class Flags(unittest.IsolatedAsyncioTestCase):
 async def test_layers_default_off(self):
  with patch.dict(os.environ,{'GEN_LAYERS_ENABLED':'0'}),patch.object(bot,'_gen_one_image',new=AsyncMock(return_value=(b'image','image/png',None,False))) as generate:
   result=await bot._gen_render_image('по слоям',[], '1K',None,False,'по слоям')
  generate.assert_awaited_once();self.assertEqual(result[0],b'image')
 async def test_marker_overrides_enabled(self):
  with patch.dict(os.environ,{'GEN_LAYERS_ENABLED':'1'}),patch('pathlib.Path.exists',return_value=True),patch.object(bot,'_gen_one_image',new=AsyncMock(return_value=(b'image','image/png',None,False))) as generate:
   await bot._gen_render_image('по слоям',[], '1K',None,False,'по слоям')
  generate.assert_awaited_once()

import os,io,json,unittest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
from PIL import Image
os.environ.setdefault('api_id','12345')
os.environ.setdefault('api_hash','deadbeefdeadbeefdeadbeefdeadbeef')
import bot_new as bot

def png(alpha=False):
    im=Image.new('RGBA' if alpha else 'RGB',(64,64),(0,0,0,0) if alpha else (20,30,40))
    if alpha:
        for x in range(20,44):
            for y in range(10,60): im.putpixel((x,y),(200,50,40,255))
    b=io.BytesIO();im.save(b,format='PNG');return b.getvalue()

class LayerIntegration(unittest.IsolatedAsyncioTestCase):
    def planner(self):
        plan={'style':'2D','lighting':'left','perspective':'eye level','background':{'prompt':'cafe'},'requested_identities':['A','B'], 'layers':[{'prompt':'two guests A B','identities':['A','B'],'x':0.1,'y':0.1,'w':0.8,'h':0.9}]}
        result=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(plan)))])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:result)))
    async def test_ordinary_generation_unchanged(self):
        f=AsyncMock(return_value=(b'a','image/png','p',False))
        with patch.object(bot,'_gen_one_image',f):
            self.assertEqual((await bot._gen_render_image('p',[],'2K',None,True,'cat'))[0],b'a')
        f.assert_awaited_once()
    async def asyncSetUp(self):
        self.layer_flag = patch.dict(os.environ, {'GEN_LAYERS_ENABLED': '1'})
        self.layer_flag.start()
        self.addCleanup(self.layer_flag.stop)

    async def test_disabled_layers_use_one_generation(self):
        f = AsyncMock(return_value=(b'one', 'image/png', 'p', False))
        with patch.dict(os.environ, {'GEN_LAYERS_ENABLED': '0'}), patch.object(bot, '_gen_one_image', f), patch.object(bot, 'get_active_model') as planner:
            result = await bot._gen_render_image('p', [], '2K', None, True, 'по слоям A B')
        self.assertEqual(result[0], b'one')
        f.assert_awaited_once()
        planner.assert_not_called()

    async def test_layers_make_separate_calls_and_final_qa(self):
        f=AsyncMock(side_effect=[(png(),'image/png','bg',False),(png(True),'image/png','asset',False)])
        qa=AsyncMock(return_value={'findings':[]})
        with patch.object(bot,'get_active_model',return_value=(self.planner(),'model','test')),patch.object(bot,'_gen_one_image',f),patch.object(bot,'_gen_visual_qa',qa):
            raw,mime,_,_=await bot._gen_render_image('A B',[],'2K','16:9',True,'по слоям A B')
        self.assertEqual(mime,'image/png'); self.assertEqual(f.await_count,2)
        self.assertEqual(Image.open(io.BytesIO(raw)).size,(1536,1024))
        qa.assert_awaited_once()
    async def test_opaque_asset_and_unavailable_qa_fail_closed(self):
        for alpha,qa_result in [(False,{'findings':[]}),(True,None)]:
            with self.subTest(alpha=alpha):
                f=AsyncMock(side_effect=[(png(),'image/png','bg',False),(png(alpha),'image/png','asset',False)])
                with patch.object(bot,'get_active_model',return_value=(self.planner(),'model','test')),patch.object(bot,'_gen_one_image',f),patch.object(bot,'_gen_visual_qa',AsyncMock(return_value=qa_result)):
                    result=await bot._gen_render_image('A B',[],'2K',None,True,'по слоям A B')
                self.assertIsNone(result[0]);self.assertEqual(result[1],'layers_failed')
if __name__=='__main__': unittest.main()

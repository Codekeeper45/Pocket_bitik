import tests
import io,json,unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock,AsyncMock
from PIL import Image
import bot_new as bot

def png():
    b=io.BytesIO();Image.new('RGB',(64,64),'blue').save(b,'PNG');return b.getvalue()

class DeliveryFirst(unittest.IsolatedAsyncioTestCase):
    async def test_no_prompt_service_still_preserves_request(self):
        with patch.object(bot,'get_active_model',return_value=(None,'offline','offline')),patch.object(bot,'_sync_image_prompt',side_effect=RuntimeError('offline')):
            result=await bot._build_gen_prompt('Two friends and exact text HELLO',catalog=[])
        contract=json.loads(result[0])
        self.assertIn('Two friends and exact text HELLO',contract['prompt'])
        self.assertEqual(contract['immutable_requirements'],['Two friends and exact text HELLO'])

    async def test_qa_route_failure_cannot_block_generated_bytes(self):
        raw=png()
        with patch.object(bot,'_gen_provider_call',new=AsyncMock(return_value=(raw,'image/png'))),patch.object(bot,'get_image_desc_client',side_effect=RuntimeError('QA offline')):
            result=await bot._gen_one_image('scene',[], '2K','1:1',True,'scene')
        self.assertEqual(result[0],raw)

    async def test_qa_malformed_retries_once_then_returns_unavailable(self):
        response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='not json'))])
        create=Mock(return_value=response)
        route=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(bot,'get_image_desc_client',return_value=(route,'vision')):
            self.assertIsNone(await bot._gen_visual_qa(png(),'scene','scene'))
        self.assertEqual(create.call_count,2)

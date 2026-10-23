import tests
import json,unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import bot_new as bot
class ContractFailures(unittest.IsolatedAsyncioTestCase):
 async def test_invalid_contract_stops_after_one_retry(self):
  response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"prompt":4}',tool_calls=None))])
  create=Mock(return_value=response)
  llm=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
  from gen_prompt import ContractError
  with patch.object(bot,'get_active_model',return_value=(llm,'offline-model','test')),patch.object(bot,'active_model_supports_vision',return_value=False),patch.object(bot,'MODEL_TOOLS_SUPPORT',{bot.ACTIVE_MODEL:False}):
   result=await bot._build_gen_prompt('Preserve Mira and exact text "OPEN"',catalog=[])
  self.assertIn('Preserve Mira and exact text "OPEN"',result[0])
  self.assertIn('Original user request (verbatim):',result[0])
  self.assertEqual(create.call_count,2)

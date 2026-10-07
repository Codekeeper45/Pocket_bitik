import tests
import json,unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import bot_new as bot
class ContractFailures(unittest.IsolatedAsyncioTestCase):
 async def test_invalid_contract_stops_after_one_retry(self):
  response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='PROMPT: incomplete',tool_calls=None))])
  create=Mock(return_value=response)
  llm=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
  from gen_prompt import ContractError
  with patch.object(bot,'get_active_model',return_value=(llm,'offline-model','test')),patch.object(bot,'active_model_supports_vision',return_value=False),patch.object(bot,'MODEL_TOOLS_SUPPORT',{bot.ACTIVE_MODEL:False}):
   result=await bot._build_gen_prompt('Preserve Mira and exact text "OPEN"',catalog=[])
  contract=json.loads(result[0])
  self.assertIn('Preserve Mira and exact text "OPEN"',contract['immutable_requirements'])
  self.assertIn('Preserve Mira and exact text "OPEN"',contract['prompt'])
  self.assertEqual(create.call_count,3)
  self.assertEqual(sum('response_format' in c.kwargs for c in create.call_args_list),2)

import os
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("api_id", "12345")
os.environ.setdefault("api_hash", "deadbeefdeadbeefdeadbeefdeadbeef")
import bot_new as bot


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))])


class BuilderContractHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_builder_calls_bounded_extractor_and_returns_rendered_tuple(self):
        contract = {"task_type": "creation", "prompt": "Mira standing beside a red bicycle; sign says \"OPEN LATE\".",
                   "participants": ["Mira"], "participant_count": 1, "required_text": ["OPEN LATE"],
                   "immutable_requirements": ['Mira beside a bicycle, include the exact sign text "OPEN LATE"'], "refs": []}
        llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=unittest.mock.Mock(
            side_effect=[response("IDEA: scene\nASPECT: 1:1\nREFS: \nPROMPT: Mira standing beside a red bicycle; sign says \"OPEN LATE\"."), response(json.dumps(contract))]))))
        with patch.object(bot, "get_active_model", return_value=(llm, "model-x", "test")), \
             patch.object(bot, "active_model_supports_vision", return_value=False), \
             patch.object(bot, "MODEL_TOOLS_SUPPORT", {bot.ACTIVE_MODEL: False}):
            result = await bot._build_gen_prompt('Mira beside a bicycle, include the exact sign text "OPEN LATE"', catalog=[])
        self.assertEqual(len(result), 4)
        self.assertEqual(result[1:], ([], "scene", "1:1"))
        rendered = json.loads(result[0])
        self.assertEqual(rendered["participant_count"], 1)
        self.assertEqual(rendered["participants"], ["Mira"])
        self.assertIn('OPEN LATE', rendered['required_text'])
        self.assertEqual(llm.chat.completions.create.call_count, 2)
        self.assertNotIn("response_format", llm.chat.completions.create.call_args_list[0].kwargs)
        self.assertEqual(llm.chat.completions.create.call_args_list[1].kwargs["max_tokens"], min(bot.ASK_MAX_TOKENS, 1200))


if __name__ == "__main__":
    unittest.main()
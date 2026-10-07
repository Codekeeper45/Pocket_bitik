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
        self.assertEqual(llm.chat.completions.create.call_args_list[1].kwargs["max_tokens"], 4000)

    async def test_cliproxy_contract_has_explicit_low_effort_without_adapter(self):
        contract = {"task_type": "creation", "prompt": "Morning cat", "immutable_requirements": ["Morning cat"], "refs": []}
        raw = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=unittest.mock.Mock(return_value=response(json.dumps(contract))))))
        llm = object.__new__(bot._CliproxyReasoningClient)
        llm._c = raw
        llm.chat = SimpleNamespace(completions=SimpleNamespace(create=unittest.mock.Mock(return_value=response("IDEA: cat\nASPECT: 1:1\nREFS: \nPROMPT: Morning cat"))))
        with patch.object(bot, "get_active_model", return_value=(llm, "gemini-3.8-flash-high", "test")), \
             patch.object(bot, "active_model_supports_vision", return_value=False), \
             patch.object(bot, "MODEL_TOOLS_SUPPORT", {bot.ACTIVE_MODEL: False}):
            await bot._build_gen_prompt("Morning cat", catalog=[])
        self.assertEqual(raw.chat.completions.create.call_args.kwargs['reasoning_effort'], 'low')
        self.assertEqual(raw.chat.completions.create.call_args.kwargs['max_tokens'], 4000)
        self.assertEqual(llm.chat.completions.create.call_count, 1)

    async def test_invalid_contract_retries_and_logs_safe_reason(self):
        contract = {"task_type": "creation", "prompt": "Morning cat", "immutable_requirements": ["Morning cat"], "refs": []}
        llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=unittest.mock.Mock(side_effect=[
            response("IDEA: cat\nASPECT: 1:1\nREFS: \nPROMPT: Morning cat"), response(""), response(json.dumps(contract))]))))
        with patch.object(bot, "get_active_model", return_value=(llm, "model-x", "test")), \
             patch.object(bot, "active_model_supports_vision", return_value=False), \
             patch.object(bot, "MODEL_TOOLS_SUPPORT", {bot.ACTIVE_MODEL: False}), patch.object(bot, 'log') as log:
            await bot._build_gen_prompt("Morning cat", catalog=[])
        self.assertEqual(llm.chat.completions.create.call_count, 3)
        self.assertTrue(any('empty contract response' in str(c) for c in log.call_args_list))


if __name__ == "__main__":
    unittest.main()
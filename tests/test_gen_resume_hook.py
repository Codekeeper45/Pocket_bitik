import asyncio
import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import bot_new as bot
from gen_series import SeriesCheckpoint, parse_multipage_plan, save_series_state


class GenResumeHookTests(unittest.TestCase):
    def test_generates_and_delivers_only_missing_page_from_saved_plan(self):
        plan = parse_multipage_plan(json.dumps({"pages": [
            {"number": 1, "prompt": "one", "refs": ["1"]},
            {"number": 2, "prompt": "two", "refs": ["1"]}], "bible": {"hero": "fox"}}), 2, user_aspect="1:1")
        checkpoint = SeriesCheckpoint(plan.plan_id, chat_id="42", owner_id="7", job_id="job-a")
        checkpoint.completed[1] = {"artifact": "", "mime": "image/png", "prompt": "one", "idea": "one"}
        checkpoint.delivered.add(1)
        event = SimpleNamespace(out=True, sender_id=7, chat_id=42,
            pattern_match=SimpleNamespace(group=lambda _n: "job-a"), reply=AsyncMock())
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            existing = artifact_dir / "page1.png"
            existing.write_bytes(b"existing")
            checkpoint.completed[1]["artifact"] = str(existing)
            state_path = root / "gen_series_state" / "job-a.json"
            state = {"job_id": "job-a", "chat_id": "42", "owner_id": "7",
                "plan": [{"number": p.number, "prompt": p.prompt, "aspect": p.aspect, "refs": list(p.refs)} for p in plan.pages],
                "bible": plan.bible, "version": plan.version, "checkpoint": checkpoint.to_json(),
                "pages_meta": {"2": {"idea": "two"}},
                "generation": {"input_b64s": [base64.b64encode(b"ref").decode()], "input_roles": ["hero"],
                    "image_size": "2K", "aspect_ratio": "1:1", "user_prompt": "story", "reply_target_id": 99}}
            save_series_state(state_path, state)
            class ArtifactStore:
                ttl_seconds = 86400
                def save(self, key, data, suffix):
                    path = artifact_dir / (key + suffix)
                    path.write_bytes(data)
                    return path
            generated, sent = [], AsyncMock(return_value=SimpleNamespace(id=123))
            async def render(prompt, inputs, size, aspect, repair, user_prompt, status):
                generated.append((prompt, inputs, size, aspect, user_prompt))
                return b"page-two", "image/png", "rendered prompt", False
            import os
            cwd = os.getcwd()
            try:
                os.chdir(root)
                with patch.object(bot, "OWNER_ID", 7), patch("gen_runtime.ARTIFACTS", ArtifactStore()), \
                     patch.object(bot, "_gen_render_image", new=render), patch.object(bot, "_gen_send_image", new=sent), \
                     patch("gen_provider.GATEWAY_PROFILE", SimpleNamespace(max_bytes=1000)):
                    asyncio.run(bot.gen_resume.__wrapped__(event))
            finally:
                os.chdir(cwd)
            self.assertEqual(len(generated), 1, event.reply.await_args_list)
            self.assertIn("Series bible", generated[0][0])
            self.assertEqual(sent.await_count, 1)
            updated = json.loads(state_path.read_text())
            final = SeriesCheckpoint.from_json(updated["checkpoint"], plan=plan)
            self.assertEqual(set(final.completed), {1, 2})
            self.assertEqual(final.delivered, {1, 2})

    def test_non_owner_event_is_ignored(self):
        event = SimpleNamespace(out=False, sender_id=8, chat_id=42)
        with patch.object(bot, "OWNER_ID", 7):
            asyncio.run(bot.gen_resume.__wrapped__(event))

    def test_unconfirmed_saved_delivery_is_not_replayed(self):
        plan = parse_multipage_plan('{"pages":[{"number":1,"prompt":"one"}]}', 1)
        checkpoint = SeriesCheckpoint(plan.plan_id, chat_id="42", owner_id="7", job_id="job-a")
        checkpoint.completed[1] = {"artifact": "", "mime": "image/png", "prompt": "one"}
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "gen_series_state" / "job-a.json"
            save_series_state(state_path, {"job_id":"job-a","chat_id":"42","owner_id":"7",
                "plan":[{"number":1,"prompt":"one","aspect":None,"refs":[]}],"bible":{},"version":"1",
                "checkpoint":checkpoint.to_json(),"generation":{"input_b64s":[],"input_roles":[],
                "image_size":"2K","aspect_ratio":None,"user_prompt":"story","reply_target_id":None}})
            event = SimpleNamespace(out=True, sender_id=7, chat_id=42,
                pattern_match=SimpleNamespace(group=lambda _n:"job-a"), reply=AsyncMock())
            import os
            cwd = os.getcwd()
            try:
                os.chdir(td)
                with patch.object(bot,"OWNER_ID",7), patch("gen_runtime.ARTIFACTS",SimpleNamespace(ttl_seconds=86400)):
                    asyncio.run(bot.gen_resume.__wrapped__(event))
            finally:
                os.chdir(cwd)
            event.reply.assert_awaited_once()
            self.assertIn("Автоповтор остановлен", event.reply.await_args.args[0])


if __name__ == "__main__":
    unittest.main()

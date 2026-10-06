"""Offline mocked end-to-end checks for the real /gen command coroutine."""
import asyncio
import base64
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

import bot_new as bot


def png_bytes():
    out = io.BytesIO()
    Image.new("RGB", (12, 9), "navy").save(out, format="PNG")
    return out.getvalue()


class FakeEvent:
    out = True
    sender_id = 77
    chat_id = 9001
    id = 501
    reply_to = None
    sender = SimpleNamespace(first_name="Owner")

    def __init__(self, flags="-raw", prompt="draw a fox"):
        self.pattern_match = SimpleNamespace(group=lambda n: {1: "", 2: flags, 3: "", 4: prompt}[n])
        self.replies = []

    async def reply(self, text):
        self.replies.append(text)
        return FakeStatus()

    async def get_reply_message(self):
        return None


class FakeStatus:
    def __init__(self):
        self.edits, self.deleted = [], False

    async def edit(self, text):
        self.edits.append(text)

    async def delete(self):
        self.deleted = True


class GenCommandE2ETests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        bot._GEN_PROVIDER_GATE.min_interval = 0
        self.status = FakeStatus()
        self.client = SimpleNamespace(
            send_message=AsyncMock(return_value=self.status),
            send_file=AsyncMock(return_value=SimpleNamespace(id=700)),
        )
        self.patches = [patch.object(bot, "openrouter_client", object()),
                        patch.object(bot, "client", self.client),
                        patch.object(bot, "_slash_for_other_bot", AsyncMock(return_value=False)),
                        patch.object(bot, "_gen_fetch_link_refs", AsyncMock(side_effect=lambda e, p: (p, [], 0))),
                        patch.object(bot, "_gen_collect_input_images", AsyncMock(return_value=([], 0))),
                        patch.object(bot, "_get_topic_id", return_value=None),
                        patch.object(bot, "_is_real_reply", return_value=(False, None)),
                        patch.object(bot, "_owner_label", return_value="owner"),
                        patch.object(bot, "log")]
        for p in self.patches: p.start()
        self.addCleanup(self._stop_patches)

    def _stop_patches(self):
        for p in reversed(self.patches): p.stop()

    async def _invoke(self, event):
        # Exercise the registered queue/activity wrapper as well as the actual coroutine.
        async def run_event(chat, work, deadline):
            await work()
            return SimpleNamespace(status="completed", job_id="fake-job")
        with patch("gen_runtime.run_event", run_event), patch("gen_runtime.JOBS"):
            # Invoke the undecorated handler in the same queue context; wrapper is separately
            # checked to ensure it delegates this coroutine through run_event.
            return await bot.gen_command.__wrapped__(event)

    async def test_raw_runs_real_command_keeps_event_and_sends_image_caption(self):
        event = FakeEvent("-raw -sq -1k", 'literal "keep exactly"')
        png = png_bytes()
        render = AsyncMock(return_value=(png, "image/png", 'literal "keep exactly"', False))
        with patch.object(bot, "_gen_render_image", render), \
             patch.object(bot, "_build_gen_prompt", new=AsyncMock(side_effect=AssertionError("raw invoked prompt builder"))), \
             patch.object(bot, "_gen_send_image", wraps=bot._gen_send_image) as delivery:
            await self._invoke(event)
        render.assert_awaited_once()
        args = render.await_args.args
        self.assertEqual(args[0], 'literal "keep exactly"')
        self.assertEqual(args[1:4], ([], "1K", "1:1"))
        self.assertFalse(args[4])  # optional layers/repair disabled for literal raw prompt
        self.assertEqual(delivery.await_args.args[:6], (9001, png, "image/png", 'literal "keep exactly"', False, None))
        self.assertTrue(self.status.deleted)
        self.assertNotIn("delete", event.__dict__)  # command message retained

    async def test_nonraw_uses_structured_prompt_contract_and_delivers_actual_caption(self):
        event = FakeEvent("", "a fox under stars")
        structured = ("A carefully specified fox under stars", [], "Night fox", "16:9")
        builder = AsyncMock(return_value=structured)
        with patch.object(bot, "_build_gen_prompt", builder), \
             patch.object(bot, "_gen_render_image", AsyncMock(return_value=(png_bytes(), "image/png", structured[0], False))), \
             patch("gen_runtime.ARTIFACTS.save") as save:
            await self._invoke(event)
        builder.assert_awaited_once()
        self.assertEqual(builder.await_args.args[0], "a fox under stars")
        self.client.send_file.assert_awaited_once()
        sent = self.client.send_file.await_args
        self.assertEqual(sent.args[0], 9001)
        self.assertIn("Night fox", sent.kwargs["caption"])
        self.assertIn(structured[0], sent.kwargs["caption"])
        self.assertEqual(sent.kwargs["reply_to"], None)
        save.assert_called_once()
        self.assertTrue(self.status.deleted)

    async def test_failure_reports_type_without_exception_secret(self):
        event = FakeEvent("-raw", "test")
        secret = "PRIVATE_PROVIDER_RESPONSE_BODY"
        with patch.object(bot, "_gen_render_image", AsyncMock(side_effect=RuntimeError(secret))):
            await self._invoke(event)
        feedback = "\n".join(self.status.edits)
        self.assertIn("RuntimeError", feedback)
        self.assertNotIn(secret, feedback)


if __name__ == "__main__":
    unittest.main()

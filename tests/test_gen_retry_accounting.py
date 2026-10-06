"""Regression checks for bounded provider retries and conservative fallbacks."""
import ast
import inspect
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def impl_source():
    source = (ROOT / "bot_new.py").read_text()
    start = source.index("async def _gen_one_image_impl(")
    end = source.index("\n\nasync def _gen_send_image", start)
    return source[start:end]


class RetryAccountingTests(unittest.TestCase):
    def test_sdk_retry_default_and_image_transport_retry_layer(self):
        from openai import OpenAI
        self.assertEqual(inspect.signature(OpenAI).parameters["max_retries"].default, 2)
        source = (ROOT / "bot_new.py").read_text()
        start = source.index("def _sync_generate_image")
        end = source.index("async def _webp_to_png", start)
        image_transport = source[start:end]
        self.assertIn("requests.post(endpoint", image_transport)
        self.assertNotIn("max_retries", image_transport)

    def test_retry_budget_not_reset_on_fallback_and_attempt_accounted(self):
        source = impl_source()
        self.assertIn("transient_left = 0 if gateway_pool else 1", source)
        self.assertNotIn("transient_left = 2", source)
        self.assertIn("invocation.generation_calls >= invocation.max_generation_calls", source)
        self.assertIn("ensure_budget()", source)

    def test_deadline_checked_before_calls_and_retries(self):
        source = impl_source()
        self.assertIn("deadline = (invocation.started + min(900, invocation.max_seconds))", source)
        self.assertLess(source.index("ensure_budget()"), source.index("_gen_provider_call(fp, input_b64s"))
        self.assertIn("if remaining() <= wait:", source)

    def test_auth_invalid_request_never_use_fallback_branches(self):
        source = impl_source()
        self.assertIn("except (GenTransient, requests.exceptions.RequestException) as e:", source)
        self.assertNotIn("except ProviderError", source)
        self.assertIn("except Exception as exc", source)  # repair QA is fail-safe; provider errors aren't caught at top level

    def test_moderation_repair_is_separate_from_invalid_request(self):
        source = impl_source()
        self.assertIn("except GenRejected as e:", source)
        self.assertIn("_sync_repair_image_prompt", source)
        self.assertIn("except (GenTransient, requests.exceptions.RequestException) as e:", source)

    def test_original_candidate_promoted_only_after_pair_qa(self):
        source = impl_source()
        condition = source.index("if newer_qa is not None and route and await compare_images")
        promotion = source.index("raw, mime, qa = newer, newer_mime, newer_qa", condition)
        self.assertLess(condition, promotion)
        self.assertIn("except Exception as exc", source)

    def test_provider_categories(self):
        from gen_provider import classify_provider_error
        self.assertEqual(classify_provider_error(401).kind, "authentication")
        self.assertEqual(classify_provider_error(403).kind, "authentication")
        self.assertEqual(classify_provider_error(400).kind, "invalid_request")
        self.assertEqual(classify_provider_error(429).kind, "rate_limit")
        self.assertEqual(classify_provider_error(500).kind, "transient")


if __name__ == "__main__":
    unittest.main()

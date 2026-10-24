import unittest
from gen_reference_evidence import (
    EvidenceFailure, evidence_record, extract_query_intent, require_visual_evidence,
    retrieval_trace, selected_args, verified_source_claim,
)

class EvidenceHelpersTests(unittest.TestCase):
    def test_trace_tool_and_allowlisted_args(self):
        trace = retrieval_trace("chat_search", {"query": "cat", "limit": 4, "extra": "drop"}, "done")
        self.assertEqual(trace["tool"], "chat_search")
        self.assertEqual(trace["args"], {"query": "cat", "limit": 4})

    def test_secret_args_omitted(self):
        self.assertEqual(selected_args({"query": "q", "api_key": "secret", "authorization": "x"}), {"query": "q"})

    def test_result_source_ids_extracted(self):
        t = retrieval_trace("read", {}, {"message_id": 71, "nested": [{"ref_id": "r2"}]})
        self.assertEqual(t["source_ids"], ["71", "r2"])

    def test_preview_is_bounded(self):
        self.assertEqual(retrieval_trace("x", {}, "abcdef", preview_chars=3)["result_preview"], "abc")

    def test_caption_alone_is_not_visual_evidence(self):
        record = evidence_record(source_id=8, caption="caption claims text is present")
        self.assertIsNone(record["visual_evidence"])
        self.assertFalse(record["verified"])
        self.assertIsInstance(require_visual_evidence(record), EvidenceFailure)

    def test_visual_success_can_verify_source(self):
        record = evidence_record(source_id=8, visual_result="Pixels show an inscription", visual_status="ok")
        self.assertTrue(verified_source_claim(record))

    def test_visual_error_cannot_verify_even_with_text(self):
        record = evidence_record(source_id=8, visual_result="caption says yes", visual_status="error")
        self.assertFalse(verified_source_claim(record))

    def test_quoted_caption_extracted(self):
        intent = extract_query_intent('Найди сообщение с подписью «Зайка»')
        self.assertEqual((intent.kind, intent.query), ("caption_quote", "Зайка"))

    def test_pixel_text_quote_is_exact_visual_intent(self):
        intent = extract_query_intent('Проверь, есть ли на фото надпись «Я Зая»')
        self.assertEqual((intent.kind, intent.query, intent.exact), ("pixel_text", "Я Зая", True))

    def test_russian_negation_is_preserved(self):
        intent = extract_query_intent('На фото нет текста «Зайка»')
        self.assertIn("нет", intent.negations)
        self.assertEqual(intent.kind, "pixel_text")

if __name__ == "__main__":
    unittest.main(verbosity=0)

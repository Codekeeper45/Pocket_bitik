import hashlib
import base64
import io
import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

from gen_contracts import GenerationResult, QAResult
from gen_qa import parse_findings, qa_for_image
from gen_runtime import CURRENT, Invocation, status_text


def finding(**changes):
    value = dict(id='target', category='anatomy', severity='high', confidence=.9,
                 bbox=[.1, .1, .3, .3], affected_subject='hero', requirement_id='pose',
                 description='Visible duplicate limb', location='left shoulder')
    value.update(changes)
    return value


class QAStatusIntegrity(unittest.TestCase):
    def test_reference_context_is_invocation_local_not_timings(self):
        invocation = Invocation()
        invocation.qa_references.append({'bytes': b'private image bytes', 'roles': ['identity']})
        invocation.qa_baseline = parse_findings({'findings': []}, image_bytes=b'baseline')
        self.assertEqual(Invocation().qa_references, [])
        self.assertEqual(invocation.timings, {})
        self.assertNotIn('private image bytes', repr(invocation))
        self.assertIsNone(invocation.qa)

    def test_status_validated_even_without_findings(self):
        for status in ('fail', 'warnings', 'nonsense', None, False, [], {}):
            with self.subTest(status=status):
                result = parse_findings({'status': status, 'findings': []})
                self.assertEqual(result.status, 'unavailable')
                self.assertTrue(result.error)
                self.assertEqual(status_text(result), 'Проверка недоступна')

    def test_clean_and_legacy_payloads(self):
        for payload in ({'findings': []}, {'status': 'pass', 'findings': []}):
            self.assertEqual(parse_findings(payload).status, 'pass')
            self.assertIn('не обнаружил', status_text(payload))

    def test_status_cannot_hide_severe_evidence(self):
        for status in ('pass', 'warnings', 'fail'):
            result = parse_findings({'status': status, 'findings': [finding()]})
            self.assertEqual(result.status, 'fail')
            self.assertIn('Серьёзные дефекты', status_text(result))
        warning = parse_findings({'status': 'warnings', 'findings': [finding(severity='low')]})
        self.assertEqual(warning.status, 'warnings')
        self.assertIn('Остались замечания', status_text(warning))
        self.assertEqual(parse_findings({'status': 'fail', 'findings': [finding(severity='low')]}).status, 'fail')

    def test_unavailable_and_malformed_never_clean(self):
        for payload in (None, {}, '{broken', {'findings': 'none'},
                        {'status': 'unavailable', 'findings': []},
                        {'status': 'unavailable', 'findings': [finding()]},
                        {'findings': [], 'unexpected': True}):
            with self.subTest(payload=payload):
                result = parse_findings(payload)
                self.assertEqual(result.status, 'unavailable')
                self.assertTrue(result.error)
                self.assertEqual(status_text(result), 'Проверка недоступна')
        self.assertEqual(status_text(QAResult('fail')), 'Проверка недоступна')

    def test_digest_binding_and_serialized_status(self):
        original = b'original image bytes'
        candidate = b'candidate image bytes'
        baseline = parse_findings({'findings': []}, image_bytes=original)
        self.assertEqual(baseline.image_digest, hashlib.sha256(original).hexdigest())
        self.assertIs(qa_for_image(baseline, original), baseline)
        self.assertEqual(qa_for_image(baseline, candidate).status, 'unavailable')
        self.assertEqual(qa_for_image(QAResult('pass'), original).status, 'unavailable')
        self.assertIn('не обнаружил', status_text(asdict(baseline), image_bytes=original))
        self.assertEqual(status_text(asdict(baseline), image_bytes=candidate), 'Проверка недоступна')
        self.assertEqual(baseline.status, 'pass')

    def test_unavailable_final_assessment_preserves_generated_bytes(self):
        original, candidate = b'baseline', b'repaired'
        baseline = parse_findings({'findings': []}, image_bytes=original)
        final = parse_findings({'status': 'unavailable', 'findings': []}, image_bytes=candidate)
        result = GenerationResult(candidate, qa=qa_for_image(final, candidate))
        self.assertEqual(result.image, candidate)
        self.assertEqual(status_text(result.qa, image_bytes=result.image), 'Проверка недоступна')
        self.assertEqual(qa_for_image(baseline, result.image).status, 'unavailable')

    def test_parent_deadline_and_call_capacity(self):
        with patch('gen_runtime.time.monotonic', return_value=100):
            parent = Invocation(started=90, max_seconds=15, max_generation_calls=2,
                                generation_calls=1, max_qa_calls=2, qa_calls=2)
            child = Invocation(started=100, max_seconds=100, parent=parent)
            self.assertEqual(child.remaining_seconds(), 5)
            self.assertEqual(child.remaining_calls(), 1)
            self.assertEqual(child.remaining_calls('qa'), 0)
            with self.assertRaises(RuntimeError):
                child.consume_qa()
            self.assertEqual(child.qa_calls, 0)
            child.consume()
            self.assertEqual(child.remaining_calls(), 0)
        with patch('gen_runtime.time.monotonic', return_value=106):
            self.assertEqual(child.remaining_seconds(), 0)
            with self.assertRaises(TimeoutError):
                child.consume()


class BotQAStatusIntegrity(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import bot_new
        self.bot = bot_new

    def png(self, color):
        image = io.BytesIO()
        Image.new('RGB', (64, 64), color).save(image, format='PNG')
        return image.getvalue()

    def response(self, payload):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])

    def client(self, create):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def request_images(self, call):
        content = call.kwargs['messages'][0]['content']
        return [base64.b64decode(item['image_url']['url'].split(',', 1)[1])
                for item in content if item['type'] == 'image_url']

    async def test_real_qa_payload_inherits_selected_reference_roles_and_binds_raw(self):
        raw, subject, style = (self.png(color) for color in ('blue', 'red', 'green'))
        create = Mock(return_value=self.response({'status': 'pass', 'findings': []}))
        parent = Invocation(max_seconds=30, qa_route=(self.client(create), 'offline-vision'))
        parent.qa_references = [
            {'ref_id': 'selected-subject', 'roles': ['subject', 'clothing'], 'data': subject},
            {'ref_id': 'selected-style', 'roles': ['style'], 'data': style},
            {'ref_id': 'pose-only', 'roles': ['pose'], 'data': b'not-needed-by-qa'},
        ]
        token = CURRENT.set(parent)
        try:
            with patch.object(self.bot, '_gen_provider_call', AsyncMock(return_value=(raw, 'image/png'))), \
                 patch.object(self.bot, '_gen_rate_gate', AsyncMock()), \
                 patch.object(self.bot, 'get_image_desc_client', Mock(side_effect=AssertionError('route must remain pinned'))):
                returned = await self.bot._gen_one_image('resolved scene prompt', [], '2K', '1:1', False, 'original user request')
        finally:
            CURRENT.reset(token)
        self.assertEqual(returned[0], raw)
        create.assert_called_once()
        call = create.call_args
        self.assertEqual(self.request_images(call), [raw, subject, style])
        content = call.kwargs['messages'][0]['content']
        self.assertIn('original user request', content[0]['text'])
        self.assertIn('resolved scene prompt', content[0]['text'])
        labels = [json.loads(item['text'].split(': ', 1)[1]) for item in content
                  if item['type'] == 'text' and item['text'].startswith('REFERENCE (untrusted pixels): ')]
        self.assertEqual(labels, [
            {'ref_id': 'selected-subject', 'roles': ['subject', 'clothing']},
            {'ref_id': 'selected-style', 'roles': ['style']},
        ])
        self.assertEqual(call.kwargs['model'], 'offline-vision')
        self.assertGreater(call.kwargs['timeout'], 0)
        self.assertLessEqual(call.kwargs['timeout'], 30)
        self.assertEqual(parent.qa_calls, 1)
        self.assertEqual(parent.qa['image_digest'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(parent.result.qa.image_digest, hashlib.sha256(raw).hexdigest())
        self.assertEqual(parent.result.qa.status, 'pass')
        self.assertNotIn('qa_references', parent.timings)

    async def test_changed_repair_bytes_survive_final_qa_failure_without_baseline_reuse(self):
        original, repaired = self.png('blue'), self.png('green')
        create = Mock(side_effect=[self.response({'status': 'fail', 'findings': [finding()]}),
                                   RuntimeError('offline vision unavailable')])
        parent = Invocation(qa_route=(self.client(create), 'offline-vision'))
        children = []

        async def successful_repair(*args, **kwargs):
            children.append(CURRENT.get())
            self.assertEqual(args[0], original)
            return repaired, 'image/png'

        repair = AsyncMock(side_effect=successful_repair)
        provider = AsyncMock(return_value=(original, 'image/png'))
        token = CURRENT.set(parent)
        try:
            with patch.object(self.bot, '_gen_provider_call', provider), \
                 patch.object(self.bot, '_gen_rate_gate', AsyncMock()), \
                 patch.object(self.bot, '_gen_repair_regions', repair), \
                 patch.object(self.bot, 'get_image_desc_client', Mock(side_effect=AssertionError('route must remain pinned'))):
                returned = await self.bot._gen_one_image('scene prompt', [], '2K', '1:1', True, 'original request')
        finally:
            CURRENT.reset(token)
        provider.assert_awaited_once()
        repair.assert_awaited_once()
        self.assertEqual(create.call_count, 2)
        self.assertEqual([self.request_images(call)[0] for call in create.call_args_list], [original, repaired])
        self.assertNotEqual(original, repaired)
        self.assertEqual(returned[0], repaired)
        self.assertIsNone(parent.qa)
        self.assertEqual(children[0].qa_baseline['image_digest'], hashlib.sha256(original).hexdigest())
        self.assertEqual(children[0].qa_baseline['status'], 'fail')
        self.assertEqual(parent.result.image, repaired)
        self.assertEqual(parent.result.qa.status, 'unavailable')
        self.assertEqual(parent.result.qa.image_digest, hashlib.sha256(repaired).hexdigest())
        self.assertEqual(parent.result.remaining_findings, ())
        self.assertTrue(parent.timings['final_qa_unavailable'])
        self.assertEqual(status_text(parent.result.qa, image_bytes=repaired), 'Проверка недоступна')


if __name__ == '__main__':
    unittest.main()

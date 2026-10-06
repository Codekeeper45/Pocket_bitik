import io
import json
import unittest
from types import SimpleNamespace
from PIL import Image
from gen_pair_qa import compare_images


class PairTargets(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        buf = io.BytesIO()
        Image.new('RGB', (20, 20), 'blue').save(buf, format='PNG')
        self.raw = buf.getvalue()

    def evidence(self, target):
        return dict(target=target, original_evidence='visible issue',
                    candidate_evidence='visible improvement', improved=True)

    def payload(self, targets=('hand',)):
        return dict(target_evidence=[self.evidence(name) for name in targets],
                    identity_preserved=True, requirements_preserved=True,
                    no_new_defects=True, no_seams=True, confidence=.95)

    def client(self, payload):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(payload)))])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: response)))

    async def check(self, payload, targets=None):
        return await compare_images(self.raw, self.raw, 'repair',
                                   client=self.client(payload), model='mock', targets=targets)

    async def test_exact_target_coverage(self):
        self.assertTrue(await self.check(self.payload(('hand', 'sleeve')), ['hand', 'sleeve']))
        self.assertFalse(await self.check(self.payload(('hand',)), ['hand', 'sleeve']))
        self.assertFalse(await self.check(self.payload(('hand', 'extra')), ['hand']))

    async def test_duplicate_and_unknown_targets_rejected(self):
        self.assertFalse(await self.check(self.payload(('hand', 'hand')), ['hand']))
        self.assertFalse(await self.check(self.payload(('alien',)), ['hand']))
        self.assertFalse(await self.check(self.payload(), []))

    async def test_strict_schema_and_confidence(self):
        data = self.payload(); data['extra'] = True
        self.assertFalse(await self.check(data))
        data = self.payload(); data['target_evidence'][0]['extra'] = 'x'
        self.assertFalse(await self.check(data))
        data = self.payload(); data['confidence'] = True
        self.assertFalse(await self.check(data))

    async def test_legacy_call_requires_single_evidence_target(self):
        self.assertTrue(await compare_images(self.raw, self.raw, 'repair',
                    client=self.client(self.payload()), model='mock'))
        self.assertFalse(await compare_images(self.raw, self.raw, 'repair',
                    client=self.client(self.payload(('hand', 'sleeve'))), model='mock'))


if __name__ == '__main__':
    unittest.main()

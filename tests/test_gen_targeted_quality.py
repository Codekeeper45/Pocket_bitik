import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from gen_pair_qa import compare_images
from gen_contracts import QAResult
from gen_policy import GenerationBudget, action_for, repair_instructions, select_repair_targets
from gen_qa import accept_repair, parse_findings
from gen_runtime import CURRENT, Invocation


def finding(**changes):
    value = dict(id='target', category='anatomy', severity='high', confidence=.9,
                 bbox=[.1, .1, .3, .3], affected_subject='hero', requirement_id='pose',
                 description='Visible duplicate limb', location='left shoulder')
    value.update(changes)
    return value


class TargetedQuality(unittest.TestCase):
    def qa(self, *findings, image=b'image'):
        return parse_findings({'findings': list(findings)}, image_bytes=image)

    def test_selection_is_structured_not_keyword_based(self):
        for description in ('face hand шов рук', 'Несовпадение', '形状の不一致'):
            qa = self.qa(finding(description=description, category='other', severity='medium'))
            self.assertEqual(select_repair_targets(qa), ())
            qa = self.qa(finding(description=description, category='text', severity='medium'))
            self.assertEqual([item['id'] for item in select_repair_targets(qa)], ['target'])
            self.assertEqual(action_for(select_repair_targets(qa)), 'repair')

    def test_select_only_significant_confident_targets(self):
        qa = self.qa(finding(id='minor', severity='low'), finding(id='uncertain', confidence=.2),
                     finding(id='high'), finding(id='critical', severity='critical'))
        self.assertEqual([item['id'] for item in select_repair_targets(qa, limit=1)], ['critical'])
        self.assertEqual(action_for([finding(confidence=.2)]), 'keep')
        self.assertEqual(select_repair_targets(parse_findings({'status': 'unavailable', 'findings': []})), ())
        overflow = parse_findings({'findings': [finding(id='one'), finding(id='two')]}, limit=1)
        self.assertEqual(select_repair_targets(overflow), ())

    def test_medium_local_defects_remain_targeted_without_keywords(self):
        for category in ('anatomy', 'technical', 'seam'):
            targets = select_repair_targets(self.qa(finding(category=category, severity='medium', description='可視の欠陥')))
            self.assertEqual(len(targets), 1)
            self.assertEqual(action_for(targets), 'repair')

    def test_regeneration_receives_evidence_and_constraints(self):
        targets = select_repair_targets(self.qa(finding(category='identity')))
        self.assertEqual(action_for(targets), 'regenerate_once')
        instructions = repair_instructions(targets, immutable_requirements=['Keep the pose'])
        payload = json.loads(instructions.split('\n', 1)[1])
        self.assertEqual(payload['targets'][0]['description'], targets[0]['description'])
        self.assertEqual(payload['immutable_requirements'], ['Keep the pose'])

    def test_minor_findings_need_not_all_improve(self):
        minor = finding(id='minor', category='style', severity='low')
        before = self.qa(finding(), minor, image=b'before')
        after = self.qa(finding(severity='low'), minor, image=b'after')
        decision = accept_repair(before, after, target_finding_id='target',
                                 before_image=b'before', after_image=b'after')
        self.assertTrue(decision.accepted)
        self.assertFalse(accept_repair(before, after, target_finding_id='target',
                                      after_image=b'wrong').accepted)

    def test_renumbering_target_is_not_improvement(self):
        before = self.qa(finding())
        after = self.qa(finding(id='renumbered', bbox=[.11, .1, .31, .3]))
        self.assertFalse(accept_repair(before, after, target_finding_id='target').accepted)

    def test_unrelated_same_category_target_need_not_improve(self):
        unrelated = finding(id='other-limb', bbox=[.6, .6, .8, .8], location='right shoulder')
        before = self.qa(finding(), unrelated)
        self.assertTrue(accept_repair(before, self.qa(unrelated), target_finding_id='target').accepted)

    def test_invalid_or_unavailable_assessment_cannot_accept_repair(self):
        before = self.qa(finding())
        for after in (QAResult('fail'), QAResult('pass', error='vision failed'), QAResult('unavailable'),
                      QAResult('pass', overflow=True), QAResult('warnings', ({'id': 'broken'},))):
            self.assertFalse(accept_repair(before, after, target_finding_id='target').accepted)

    def test_new_worsened_and_protected_defects_rejected(self):
        other = finding(id='other', category='identity', requirement_id='identity')
        before = self.qa(finding(), other)
        for after in (self.qa(dict(other, severity='critical')),
                      self.qa(other, finding(id='new', category='text')),
                      self.qa(other, dict(other, id='new', bbox=[.6, .6, .8, .8]))):
            self.assertFalse(accept_repair(before, after, target_finding_id='target').accepted)
        self.assertFalse(accept_repair(before, self.qa(other), target_finding_id='target',
                                      protected_requirements=['identity']).accepted)

    def test_remaining_budget_and_integer_accounting(self):
        budget = GenerationBudget(max_repair_calls=1)
        budget.consume('repair')
        self.assertEqual(budget.remaining_calls('repair'), 0)
        for count in (True, .5, -1):
            with self.assertRaises(ValueError):
                budget.consume('qa', count)


class PairedTargetEvidence(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        image = io.BytesIO()
        Image.new('RGB', (20, 20), 'blue').save(image, format='PNG')
        self.raw = image.getvalue()
        self.payload = dict(target_evidence=[dict(target='target', original_evidence='duplicate limb visible',
                            candidate_evidence='single limb visible', improved=True)],
                            identity_preserved=True, requirements_preserved=True,
                            no_new_defects=True, no_seams=True, confidence=.95)

    async def compare(self, payload, targets=('target',)):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
        self.create = Mock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        return await compare_images(self.raw, self.raw, 'repair chosen target only',
                                    client=client, model='offline-double', targets=targets)

    async def test_only_selected_targets_need_evidence(self):
        self.assertTrue(await self.compare(self.payload))
        self.assertFalse(await self.compare(self.payload, targets=('target', 'minor')))
        self.payload['target_evidence'][0]['target'] = ' target '
        self.assertFalse(await self.compare(self.payload))

    async def test_evidence_and_preservation_are_required(self):
        for field in ('original_evidence', 'candidate_evidence'):
            payload = dict(self.payload, target_evidence=[dict(self.payload['target_evidence'][0], **{field: ''})])
            self.assertFalse(await self.compare(payload))
        self.assertFalse(await self.compare(dict(self.payload, no_new_defects=False)))
        self.assertFalse(await self.compare(dict(self.payload, identity_preserved=False)))

    async def test_exhausted_parent_never_calls_client(self):
        parent = Invocation(max_qa_calls=0)
        child = Invocation(parent=parent)
        token = CURRENT.set(child)
        try:
            self.assertFalse(await self.compare(self.payload))
            self.create.assert_not_called()
            self.assertEqual(child.qa_calls, 0)
        finally:
            CURRENT.reset(token)

    async def test_invalid_image_is_unavailable_not_exception(self):
        client = Mock()
        self.assertFalse(await compare_images(b'bad', self.raw, 'repair', client=client,
                                             model='offline-double', targets=['target']))
        client.chat.completions.create.assert_not_called()


if __name__ == '__main__':
    unittest.main()

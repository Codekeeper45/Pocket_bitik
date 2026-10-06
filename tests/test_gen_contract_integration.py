import unittest
import sys
sys.path.insert(0, '..')
from gen_prompt import extract_contract, render_prompt, ContractError

class SceneContractIntegrationTests(unittest.TestCase):
    def test_structured_contract_validated_and_rendered(self):
        c = extract_contract({'task_type':'creation','prompt':'A scene','participants':['Ada'],'participant_count':1,'required_text':['"HELLO"'],'immutable_requirements':['Ada','"HELLO"']})
        rendered = render_prompt(c)
        self.assertEqual(extract_contract(rendered), c)
        self.assertIn('Ada', rendered)
        import json
        self.assertEqual(json.loads(rendered)['required_text'], ['"HELLO"'])

    def test_mandatory_fields_and_count_conflicts_fail(self):
        with self.assertRaises(ContractError): extract_contract({'prompt':'scene','participants':['Ada'],'participant_count':2})
        with self.assertRaises(ContractError): extract_contract({'task_type':'creation','participants':['Ada']})

if __name__ == '__main__': unittest.main()
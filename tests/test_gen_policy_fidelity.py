import unittest
from gen_policy import action_for
class FidelityPolicy(unittest.TestCase):
 def test_scene_failure_not_anatomy_patch(self):
  self.assertEqual(action_for([{'category':'fidelity','severity':'medium','confidence':.8,'bbox':[.2,.2,.5,.5]}]),'regenerate_once')
 def test_uncertain_issue_keeps_best(self):
  self.assertEqual(action_for([{'category':'fidelity','severity':'medium','confidence':.4,'bbox':None}]),'keep')
 def test_local_anatomy_repairs(self):
  self.assertEqual(action_for([{'category':'anatomy','severity':'high','confidence':.9,'bbox':[.2,.2,.4,.4]}]),'repair')

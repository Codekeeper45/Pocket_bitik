import unittest
from gen_identity_guard import missing_identity_source
class IdentityGuardTests(unittest.TestCase):
 def test_missing_rustam(self):
  self.assertTrue(missing_identity_source('portrait','Настоящее исходное фото Рустама в чате не найдено, поэтому создаётся собирательный реалистичный портрет',False))
 def test_missing_with_irrelevant_refs(self):
  self.assertTrue(missing_identity_source('','Исходное фото не найдено',True))
 def test_regular_character(self):
  self.assertFalse(missing_identity_source('original anime character','новый персонаж',False))
 def test_reference_edit(self):
  self.assertFalse(missing_identity_source('same person as reference','Рустам с макияжем',True))

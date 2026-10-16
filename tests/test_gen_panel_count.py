import ast,re,unittest
from pathlib import Path
class PanelCountTests(unittest.TestCase):
 def test_panels_are_not_pages(self):
  tree=ast.parse(Path('bot_new.py').read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_detect_pages_count');ns={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),'x','exec'),ns);f=ns['_detect_pages_count']
  self.assertEqual(f('Комикс на 3 кадра'),1)
  self.assertEqual(f('Комикс на три кадра'),1)
  self.assertEqual(f('Комикс на 3 страницы'),3)
  self.assertEqual(f('Комикс', ['-p2']),2)

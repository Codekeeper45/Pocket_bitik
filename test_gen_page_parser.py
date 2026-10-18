import ast,json,re,unittest
from pathlib import Path
s=Path(__file__).with_name('bot_new.py').read_text()
n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='_parse_gen_multipage_out')
ns={'json':json,'re':re};exec(ast.get_source_segment(s,n),ns)
parse=ns['_parse_gen_multipage_out']
class Tests(unittest.TestCase):
 def test_pages(self):
  text='\n'.join(f'PAGE {i}:\nIDEA: scene {i}\nASPECT: 9:16\nPROMPT: Distinct scene number {i}' for i in range(1,6))
  p=parse(text,5);self.assertEqual(len(p),5);self.assertEqual(p[0]['prompt'],'Distinct scene number 1');self.assertEqual(p[-1]['aspect'],'9:16')
 def test_reject_unstructured(self):
  with self.assertRaises(ValueError):parse('No structured plan',2)
if __name__=='__main__':unittest.main()

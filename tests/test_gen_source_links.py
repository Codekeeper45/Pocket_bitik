import ast,unittest
from pathlib import Path
class SourceLinksTests(unittest.TestCase):
 def test_only_used_web_links(self):
  tree=ast.parse(Path('bot_new.py').read_text())
  node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_gen_refs_line')
  ns={'build_msg_link':lambda c,m:f'https://t.me/test/{m}'}
  exec(compile(ast.Module(body=[node],type_ignores=[]),'source','exec'),ns)
  result=ns['_gen_refs_line'](object(),[{'idx':1,'source_url':'https://example.org/a.jpg'},{'idx':2,'source_url':'https://example.org/not-used.jpg'}],[1],{1:'style'})
  self.assertIn('https://example.org/a.jpg',result)
  self.assertNotIn('not-used',result)
  self.assertIn('style',result)

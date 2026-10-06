import ast,collections,unittest
from pathlib import Path
class ProductionDefinitionIntegrity(unittest.TestCase):
 def test_no_shadowed_generation_functions(self):
  tree=ast.parse(Path('bot_new.py').read_text());counts=collections.Counter(n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and (n.name.startswith('_gen') or n.name=='get_image_desc_client'))
  self.assertEqual({k:v for k,v in counts.items() if v>1},{})
 def test_render_has_real_dispatch(self):
  tree=ast.parse(Path('bot_new.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='_gen_render_image')
  self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_gen_one_image' for n in ast.walk(node)))

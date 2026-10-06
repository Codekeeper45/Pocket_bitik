import ast
import unittest
from pathlib import Path

class FilterGuards(unittest.TestCase):
    def test_failed_initial_filter_returns_before_reference_lookup(self):
        tree = ast.parse(Path('bot_new.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'gen_command')
        guard = next(n for n in fn.body if isinstance(n, ast.If) and 'failed_filters' in ast.unparse(n.test))
        self.assertTrue(any(isinstance(n, ast.Return) for n in guard.body))
        lookup = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_gen_fetch_link_refs')
        self.assertLess(guard.lineno, lookup.lineno)
    def test_user_command_is_not_deleted(self):
        tree = ast.parse(Path('bot_new.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'gen_command')
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and ast.unparse(n.func) == 'event.delete' for n in ast.walk(fn)))

if __name__ == '__main__': unittest.main()

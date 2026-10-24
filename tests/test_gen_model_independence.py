import ast
import unittest
from pathlib import Path


class GenModelIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path('bot_new.py').read_text()
        cls.tree = ast.parse(cls.source)

    def test_state_has_independent_persisted_field(self):
        self.assertIn('"gen_prompt_model": GEN_PROMPT_MODEL', self.source)
        self.assertIn('GEN_PROMPT_MODEL = _model_state.get("gen_prompt_model")', self.source)
        self.assertIn('deepseek/deepseek-v4.1-flash-1017', self.source)

    def test_gen_builder_never_reads_or_mutates_ask_active_model(self):
        node = next(n for n in self.tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_build_gen_prompt')
        attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        self.assertNotIn('ACTIVE_MODEL', names)
        self.assertNotIn('get_active_model', names)
        self.assertNotIn('MODEL_TOOLS_SUPPORT', names)
        self.assertIn('get_gen_planner_model', names)
        self.assertIn('GEN_PROMPT_TOOLS_SUPPORT', names)

    def test_command_dispatch_preserves_image_gen_branch(self):
        node = next(n for n in self.tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'model_command')
        segment = ast.get_source_segment(self.source, node)
        self.assertIn('arg.lower() == "genprompt"', segment)
        self.assertIn('arg.lower() == "gen" or arg.lower().startswith("gen ")', segment)
        self.assertIn('GEN_IMAGE_MODEL = mid', segment)

    def test_all_gen_llm_routes_are_independent_of_ask(self):
        expected = {
            '_sync_image_prompt': 'get_gen_prompt_model',
            '_gen_public_web_query': 'get_gen_planner_model',
            '_build_gen_prompt': 'get_gen_planner_model',
            '_build_multipage_prompts': 'get_gen_planner_model',
            '_gen_rerank_index_items': 'get_gen_planner_model',
        }
        for func_name, getter in expected.items():
            node = next(n for n in self.tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func_name)
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            self.assertIn(getter, names, func_name)
            self.assertNotIn('get_active_model', names, func_name)
            self.assertNotIn('_llm_create', names, func_name)
        node = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sync_image_prompt')
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        self.assertIn('get_gen_prompt_model', names)
        self.assertNotIn('get_active_model', names)

    def test_gen_planner_repair_uses_same_selected_route(self):
        fn = next(n for n in self.tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_build_gen_prompt')
        source = ast.get_source_segment(self.source, fn)
        self.assertIn('repair_client = llm', source)
        self.assertIn('repair_model = model_id', source)
        self.assertNotIn('google/gemini-3.5-flash', source)
        self.assertNotIn('ACTIVE_MODEL', source)

    def test_gen_rerank_is_not_ask_route(self):
        fn = next(n for n in self.tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_gen_rerank_index_items')
        source = ast.get_source_segment(self.source, fn)
        self.assertIn('get_gen_planner_model()', source)
        self.assertNotIn('_llm_create(', source)
        self.assertNotIn('ACTIVE_MODEL', source)

    def test_multipage_planning_uses_gen_model(self):
        fn = next(n for n in self.tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_build_multipage_prompts')
        source = ast.get_source_segment(self.source, fn)
        self.assertIn('get_gen_planner_model()', source)
        self.assertNotIn('active_model_supports_vision', source)
        self.assertNotIn('get_active_model()', source)

    def test_image_prompt_call_uses_separate_model_getter(self):
        node = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sync_image_prompt')
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        self.assertIn('get_gen_prompt_model', names)
        self.assertNotIn('get_active_model', names)


if __name__ == '__main__':
    unittest.main()

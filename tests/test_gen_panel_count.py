import ast
import re
import unittest
from pathlib import Path

from gen_prompt import parse_intent_plan
from gen_series import series_plan_from_intent


class PanelCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(Path('bot_new.py').read_text())
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == '_detect_pages_count')
        namespace = {'re': re}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'bot_new.py', 'exec'), namespace)
        cls.detect_explicit_count = staticmethod(namespace['_detect_pages_count'])

    def test_natural_language_never_determines_file_count_in_code(self):
        for request in ('Комикс на 3 кадра', 'Комикс на три кадра', 'Комикс на 3 страницы',
                        'Не три страницы, один постер', 'Создай 3 отдельных изображения'):
            with self.subTest(request=request):
                self.assertEqual(self.detect_explicit_count(request), 1)

    def test_explicit_count_flags_remain_supported(self):
        for flags, expected in [(['-p2'], 2), (['--pages', '3'], 3), (['--pages=4'], 4),
                                (['-p', '5'], 5), (['-pages', '2'], 2), (['-p99'], 10)]:
            with self.subTest(flags=flags):
                self.assertEqual(self.detect_explicit_count('Комикс', flags), expected)

    def test_model_panels_are_not_separate_files(self):
        plan = parse_intent_plan({'prompt': 'A comic sheet', 'output_kind': 'single_image',
                                  'output_count': 1, 'panels_per_image': 3}, 'Комикс на три кадра')
        self.assertEqual(plan.panels_per_image, 3)
        self.assertEqual(len(series_plan_from_intent(plan).pages), 1)

    def test_model_file_count_and_explicit_override_are_independent_of_panels(self):
        payload = {'prompt': 'Separate comic sheets', 'output_kind': 'independent_images',
                   'output_count': 3, 'panels_per_image': 2}
        plan = parse_intent_plan(payload, 'Три изображения')
        self.assertEqual(len(series_plan_from_intent(plan).pages), 3)
        override = parse_intent_plan(payload, 'Три изображения', explicit_count=1)
        self.assertEqual(len(series_plan_from_intent(override).pages), 1)
        self.assertEqual(override.panels_per_image, 2)

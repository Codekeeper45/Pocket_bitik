"""Clean regression tests for reference caption disambiguation and live chat precedence."""
import unittest
import bot_new as bot

class TestReferenceDisambiguationClean(unittest.TestCase):
    def test_catalog_ref_label_clearly_marks_message_caption(self):
        item = {
            'idx': 1,
            'mid': 331450,
            'caption': 'Лучшая девочка Нейромафии, ну или просто Зайка #фанарт',
            'desc': 'Арт крольчихи с ушами'
        }
        label = bot._gen_catalog_ref_label(item)
        self.assertIn('message_caption', label)
        self.assertIn('НЕ внутри картинки', label)

    def test_gen_ref_rules_warn_against_blind_caption_selection(self):
        self.assertIn('message_caption', bot._GEN_REF_RULES)
        self.assertIn('chat_inspect_image', bot._GEN_REF_RULES)

    def test_catalog_combination_keeps_fresh_chat_media_over_stale_index(self):
        fresh_items = [{'mid': 100 + i, 'idx': i} for i in range(1, 9)]
        stale_index = [{'mid': 200 + i, 'idx': i + 8} for i in range(1, 9)]
        cap = 10
        combined = fresh_items + stale_index
        seen = set()
        deduped = []
        for it in combined:
            m = it.get('mid')
            if m not in seen:
                seen.add(m)
                deduped.append(it)
        result = deduped[:cap]
        # First 8 must be fresh items, only remaining 2 are filled by index
        self.assertEqual([x['mid'] for x in result[:8]], [101, 102, 103, 104, 105, 106, 107, 108])
        self.assertEqual([x['mid'] for x in result[8:]], [201, 202])

if __name__ == '__main__':
    unittest.main()

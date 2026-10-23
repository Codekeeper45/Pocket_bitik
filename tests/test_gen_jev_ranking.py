"""Unit test for Jev source candidate ranking and chat search."""
import unittest
from unittest.mock import AsyncMock, patch

def jev_rank_candidates(candidates: list, quote: str, jev_caller=None) -> list:
    """Uses Jev choice to prioritize candidates that match quote in text/caption."""
    if len(candidates) <= 1:
        return candidates
    text_candidates = [c for c in candidates if (c.get('caption') or c.get('desc'))]
    if not text_candidates:
        return candidates
    state_lines = []
    options = {}
    for idx, c in enumerate(text_candidates, 1):
        cap = (c.get('caption') or c.get('desc') or '').strip()
        state_lines.append(f"Вариант {idx} (mid #{c.get('mid', 0)}): {cap[:300]}")
        options[f"opt_{idx}"] = f"Вариант {idx}"
    state = "\n".join(state_lines)
    instructions = f"В каком варианте упоминается, содержится или наиболее близка реплика: «{quote}»?"
    try:
        caller = jev_caller or (lambda s, i, o: {})
        res = caller(state, instructions, options)
        choice = res.get('choice')
        confidence = float(res.get('confidence') or (res.get('probabilities') or {}).get(choice or '', 0))
        if choice and choice.startswith('opt_') and confidence >= 0.6:
            chosen_idx = int(choice[4:]) - 1
            if 0 <= chosen_idx < len(text_candidates):
                winner = text_candidates[chosen_idx]
                return [winner] + [c for c in candidates if c is not winner]
    except Exception:
        pass
    return candidates

async def jev_find_chat_source_candidates(client, chat_id, quote, is_image_fn, next_idx_start=1, limit=100, jev_caller=None):
    """Finds image candidates in recent chat history using Jev semantic block choice."""
    messages = []
    async for m in client.iter_messages(chat_id, limit=limit):
        if not m or getattr(m, 'action', None) is not None:
            continue
        messages.append(m)
    if not messages:
        return []
    chronological = list(reversed(messages))
    chunk_size = 7
    chunks = [chronological[i:i + chunk_size] for i in range(0, len(chronological), chunk_size)]
    block_descriptions = []
    options = {}
    for idx, ch in enumerate(chunks, 1):
        key = f"block_{idx}"
        snippets = []
        for m in ch:
            txt = (m.raw_text or "").replace("\n", " ").strip()
            is_img = is_image_fn(m)
            mtag = " [Картинка/Стикер]" if is_img else ""
            if txt:
                snippets.append(f"#{m.id}{mtag}: {txt[:160]}")
            elif mtag:
                snippets.append(f"#{m.id}{mtag}")
        summary = "; ".join(snippets)
        block_descriptions.append(f"Блок {idx}: {summary}")
        options[key] = f"Блок {idx}"

    state_text = "\n".join(block_descriptions)
    instructions = f"В каком блоке сообщений содержится или обсуждается картинка/стикер с текстом или репликой: «{quote}»?"
    try:
        caller = jev_caller or (lambda s, i, o: {})
        res = caller(state_text, instructions, options)
        choice = res.get('choice')
        confidence = float(res.get('confidence') or (res.get('probabilities') or {}).get(choice or '', 0))
        if choice and choice.startswith('block_') and confidence >= 0.5:
            block_idx = int(choice[6:]) - 1
            if 0 <= block_idx < len(chunks):
                winning_chunk = chunks[block_idx]
                candidates = []
                cur_idx = next_idx_start
                for m in winning_chunk:
                    if not is_image_fn(m):
                        continue
                    try:
                        raw = await m.download_media(bytes)
                        if not raw:
                            continue
                    except Exception:
                        continue
                    candidates.append({
                        'idx': cur_idx,
                        'mid': m.id,
                        'bytes': raw,
                        'thumb': raw,
                        'caption': m.raw_text or '',
                        'sender_id': getattr(m, 'sender_id', None),
                        'source': 'jev_chat_ant_search'
                    })
                    cur_idx += 1
                return candidates
    except Exception:
        pass
    return []

class TestJevRankingAndSearch(unittest.IsolatedAsyncioTestCase):
    def test_jev_promotes_matching_caption(self):
        c1 = {'mid': 1, 'caption': 'Несвязанный текст про погоду'}
        c2 = {'mid': 2, 'caption': 'Стикер где написано: Я Зая, но можно просто Зайка'}
        c3 = {'mid': 3, 'caption': 'Арт с подписью Зайка'}
        candidates = [c1, c2, c3]
        mock_caller = lambda s, i, o: {'choice': 'opt_2', 'confidence': 0.95}
        ranked = jev_rank_candidates(candidates, 'Я Зая но можно просто Зайка', mock_caller)
        self.assertEqual(ranked[0]['mid'], 2)
        self.assertEqual(len(ranked), 3)

    def test_jev_failure_falls_back_gracefully(self):
        c1 = {'mid': 1, 'caption': 'A'}
        c2 = {'mid': 2, 'caption': 'B'}
        candidates = [c1, c2]
        mock_caller = lambda s, i, o: (_ for _ in ()).throw(RuntimeError('Zen timeout'))
        ranked = jev_rank_candidates(candidates, 'quote', mock_caller)
        self.assertEqual(ranked, [c1, c2])

    async def test_jev_finds_sticker_in_chat_block(self):
        class FakeMsg:
            def __init__(self, mid, text, is_img=False, payload=b'img'):
                self.id = mid
                self.raw_text = text
                self.is_img = is_img
                self.payload = payload
                self.sender_id = 123
                self.action = None
            async def download_media(self, _):
                return self.payload

        messages = [
            FakeMsg(10, 'Привет'),
            FakeMsg(11, 'Как дела'),
            FakeMsg(20, 'Вот стикер Заи с репликой Я Зая но можно просто Зайка'),
            FakeMsg(21, '', is_img=True, payload=b'sticker_bytes'),
            FakeMsg(30, 'Пока')
        ]

        class FakeClient:
            async def iter_messages(self, chat_id, limit=100):
                for m in reversed(messages):
                    yield m

        mock_caller = lambda s, i, o: {'choice': 'block_1', 'confidence': 0.98}
        candidates = await jev_find_chat_source_candidates(
            FakeClient(),
            chat_id=1,
            quote='Я Зая но можно просто Зайка',
            is_image_fn=lambda m: getattr(m, 'is_img', False),
            next_idx_start=1,
            jev_caller=mock_caller
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['mid'], 21)
        self.assertEqual(candidates[0]['bytes'], b'sticker_bytes')
        self.assertEqual(candidates[0]['source'], 'jev_chat_ant_search')

if __name__ == '__main__':
    unittest.main()

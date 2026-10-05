"""AST-isolated regression checks for optional retrieval failure paths."""
import ast
import asyncio
import time
import datetime
import unittest
from pathlib import Path
from unittest.mock import patch
import requests

SOURCE = Path(__file__).resolve().parents[1] / 'bot_new.py'

def load(name, namespace):
    node = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace[name]

class ReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_402_stops_repeated_requests(self):
        ns = dict(asyncio=asyncio, time=time, requests=requests,
                  _INDEX_EMBED_UNAVAILABLE_UNTIL=0., INDEX_EMBED_IMAGE_MODEL='i',
                  INDEX_EMBED_TEXT_MODEL='t', INDEX_EMBED_IMAGE_DIM=8,
                  INDEX_EMBED_TEXT_DIM=8, _INDEX_QEMB_CACHE={},
                  OPENROUTER_BASE_URL='http://test', openrouter_api_key='test',
                  log=lambda *args: None)
        fn = load('_index_embed_query', ns)
        error = requests.HTTPError('402')
        error.response = type('Response', (), {'status_code': 402})()
        with patch.object(requests, 'post', side_effect=error) as post:
            self.assertIsNone(await fn('first'))
            self.assertIsNone(await fn('different'))
            self.assertEqual(post.call_count, 1)

    async def test_invalid_media_filter_preserves_search(self):
        class Client:
            def __init__(self): self.calls = []
            def iter_messages(self, chat, **kwargs):
                self.calls.append((chat, kwargs))
                async def iterator():
                    if 'filter' in kwargs:
                        raise Exception('The search query filter is invalid')
                    if False: yield None
                return iterator()
        client = Client()
        import sys, types
        types_mod = types.ModuleType('telethon.tl.types')
        types_mod.InputMessagesFilterPhotos = lambda: object()
        tl_mod = types.ModuleType('telethon.tl'); tl_mod.types = types_mod
        telethon_mod = types.ModuleType('telethon'); telethon_mod.tl = tl_mod
        with patch.dict(sys.modules, {'telethon': telethon_mod, 'telethon.tl': tl_mod, 'telethon.tl.types': types_mod}):
            fn = load('_run_chat_search', dict(client=client, log=lambda *args: None))
            result = await fn(123, {'query': 'Зая', 'filter': 'photo'})
        self.assertIn('ничего не найдено', result)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0][0], client.calls[1][0])
        self.assertEqual(client.calls[0][1]['search'], client.calls[1][1]['search'])
        self.assertNotIn('filter', client.calls[1][1])

    async def test_unresolved_sender_filter_never_returns_unfiltered_hits(self):
        class Client:
            def iter_messages(self, chat, **kwargs):
                async def iterator():
                    yield type('Message', (), {'id': 1, 'sender': type('Sender', (), {'first_name': 'Alice', 'last_name': 'Smith', 'username': 'alice'})(), 'sender_id': 7, 'date': None, 'out': False, 'raw_text': '', 'photo': None, 'document': None, 'reply_to_msg_id': None})()
                return iterator()
        fn = load('_run_chat_search', dict(client=Client(), datetime=datetime.datetime, MEDIA_CACHE={}, log=lambda *args: None,
                                           _forward_src=lambda m: '', _fmt_date=lambda d: '', _media_tag=lambda m: '',
                                           _label_for=lambda m, s: 'Alice Smith',
                                           _preview=lambda text, n: text))
        result = await fn(123, {'query': 'photo', 'from_user': 'missing-person'})
        self.assertIn('ничего не найдено', result)

    async def test_sender_match_is_exact_not_substring(self):
        class Client:
            def iter_messages(self, chat, **kwargs):
                async def iterator():
                    yield type('Message', (), {'id': 1, 'sender': type('Sender', (), {'first_name': 'Alice Smith', 'last_name': '', 'username': 'alice'})(), 'sender_id': 7, 'date': None, 'out': False, 'raw_text': '', 'photo': None, 'document': None, 'reply_to_msg_id': None})()
                return iterator()
        fn = load('_run_chat_search', dict(client=Client(), datetime=datetime.datetime, MEDIA_CACHE={}, log=lambda *args: None,
                                           _forward_src=lambda m: '', _fmt_date=lambda d: '', _media_tag=lambda m: '',
                                           _label_for=lambda m, s: 'Alice Smith',
                                           _preview=lambda text, n: text))
        result = await fn(123, {'query': 'photo', 'from_user': 'Alice'})
        self.assertIn('Найдено 1 сообщений', result)  # Exact full-name matching remains supported.

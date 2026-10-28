import asyncio
from unittest.mock import AsyncMock
import tests
import bot_new as bot


def test_short_quote_awaits_parser(monkeypatch):
    class Client:
        _parse_message_text = AsyncMock(return_value=('hello', []))
        __call__ = AsyncMock()
    client = Client()
    monkeypatch.setattr(bot, 'client', client)
    monkeypatch.setattr(bot, 'REPLY_COLLAPSE', 1000)
    assert asyncio.run(bot._send_quote_reply(99, 5, '<b>hello</b>', 'target', '😀 target'))
    client._parse_message_text.assert_awaited_once_with('<b>hello</b>', 'html')
    request = client.__call__.call_args.args[0]
    assert request.message == 'hello'
    assert request.reply_to.quote_offset == 3
    assert request.reply_to.quote_text == 'target'


def test_missing_quote_does_not_send(monkeypatch):
    client = AsyncMock()
    monkeypatch.setattr(bot, 'client', client)
    assert not asyncio.run(bot._send_quote_reply(99, 5, 'hello', 'absent', 'source'))
    client.assert_not_awaited()

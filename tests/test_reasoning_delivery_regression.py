import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import tests
import bot_new as bot
from tests.test_voice_agent_integration import setup, response, call

def test_empty_reply_preserves_tool_loop(monkeypatch):
    empty=response()
    empty.choices[0].message.reasoning_content='PRIVATE_SENTINEL'
    create=setup(monkeypatch,[empty,response(calls=[call('send_chat_responses',{'responses':[{'text':'Проверенный результат','mode':'text'}]})]),response('Итог')])
    fallback=AsyncMock(return_value='Wrong fallback')
    monkeypatch.setattr(bot,'generate_ask_reply',fallback)
    result=asyncio.run(bot.ask_agentic('source','answer',chat_id=99,msg_by_id={},memory_allowed=False))
    assert result=='Итог'
    fallback.assert_not_called()
    assert bot.send_long.await_count==1
    assert all('tools' in c.kwargs for c in create.call_args_list)

def test_plain_tool_preamble_is_not_final(monkeypatch):
    create=setup(monkeypatch,[response('Сейчас найду',calls=[call('chat_search',{'query':'test'})]),response('Готовый ответ')])
    result=asyncio.run(bot._llm_create(messages=[{'role':'user','content':'test'}]))
    assert result=='Готовый ответ'
    assert create.call_count==2

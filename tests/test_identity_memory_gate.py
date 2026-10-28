import asyncio
from unittest.mock import AsyncMock
from tests.test_voice_agent_integration import setup, response, call
import bot_new as bot
from identity_policy import IDENTITY_POLICY, IMAGE_IDENTITY_POLICY
from voice_agent_tools import VOICE_HINT

def test_disabled_before_dependency_checks(monkeypatch):
    monkeypatch.delenv('INDEX_ENABLED',raising=False)
    assert 'отключена' in bot._index_available()
    assert not bot._index_memory_allowed(True)

def test_disabled_memory_tools_never_queried(monkeypatch):
    setup(monkeypatch,[response('Личность не установлена')])
    monkeypatch.setenv('INDEX_ENABLED','false')
    probe=AsyncMock(side_effect=AssertionError('must not touch DB'))
    monkeypatch.setattr(bot,'_index_ready_memory_tools',probe)
    asyncio.run(bot.ask_agentic('ctx','Кто такой Неизвестный?',chat_id=99,memory_allowed=True))
    probe.assert_not_awaited()

def test_shared_policy():
    assert IDENTITY_POLICY in bot.ASK_SYSTEM_PROMPT
    assert IDENTITY_POLICY in bot.AUTO_REPLY_SYSTEM_PROMPT
    assert IDENTITY_POLICY in VOICE_HINT
    assert IMAGE_IDENTITY_POLICY in bot._gen_unified_system(False,False)

def test_auto_can_ground_identity_with_chat_search(monkeypatch):
    setup(monkeypatch,[response(calls=[call('chat_search',{'query':'Ника'})]),response('Ника найдена')])
    search=AsyncMock(return_value='Ника: участница переписки')
    monkeypatch.setattr(bot,'_run_chat_search',search)
    result=asyncio.run(bot.generate_auto_reply('Кто Ника?',chat_id=99,msg_by_id={}))
    assert result=='Ника найдена'
    search.assert_awaited_once()

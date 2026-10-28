import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import tests
import bot_new as bot
from tests.test_voice_agent_integration import setup, response, call


def test_ask_empty_after_delivery_recovers(monkeypatch):
    create=setup(monkeypatch,[response(calls=[call('send_chat_responses',{'responses':[{'text':'replica','mode':'text'}]})]),response(''),response('separate final')])
    fallback=AsyncMock();monkeypatch.setattr(bot,'generate_ask_reply',fallback)
    result=asyncio.run(bot.ask_agentic('source','answer',chat_id=99,msg_by_id={},memory_allowed=False))
    assert result=='separate final'
    assert 'tools' not in create.call_args.kwargs
    assert bot.send_long.await_count==1
    fallback.assert_not_called()


def test_ask_empty_choices_after_delivery_recovers(monkeypatch):
    setup(monkeypatch,[response(calls=[call('send_chat_responses',{'responses':[{'text':'replica','mode':'text'}]})]),NS(choices=[]),response('final')])
    assert asyncio.run(bot.ask_agentic('source','answer',chat_id=99,msg_by_id={},memory_allowed=False))=='final'


def test_auto_empty_after_delivery_recovers(monkeypatch):
    setup(monkeypatch,[response(calls=[call('send_chat_responses',{'responses':[{'text':'replica','mode':'text'}]})]),response(''),response('auto final')])
    ledger=[]
    assert asyncio.run(bot.generate_auto_reply('source',chat_id=99,msg_by_id={},delivery=ledger))=='auto final'
    assert len(ledger)==1


def test_long_text_no_old_limit(monkeypatch):
    setup(monkeypatch,[])
    turn=bot.VoiceTurn(bot,99)
    import json
    result=json.loads(asyncio.run(turn.run('send_chat_responses',{'responses':[{'text':'x'*20000,'mode':'text'}]})))
    assert result['results'][0]['status']=='sent'
    assert len(bot.send_long.call_args.args[1])==20000


def test_long_voice_no_old_limit(monkeypatch):
    setup(monkeypatch,[])
    turn=bot.VoiceTurn(bot,99)
    import json
    result=json.loads(asyncio.run(turn.run('send_chat_responses',{'responses':[{'text':'x'*8000,'mode':'voice'}]})))
    assert result['results'][0]['status']=='sent'
    assert len(bot._send_agent_voice.call_args.args[1])==8000

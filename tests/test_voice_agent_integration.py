import asyncio,json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock,Mock
import tests
import bot_new as bot
from voice_agent_tools import VOICE_TOOLS
A='a'*32

def response(text=None,calls=()):
 return NS(choices=[NS(message=NS(content=text,tool_calls=list(calls)),finish_reason='stop')],usage=None)
def call(name,args):return NS(id='t1',type='function',function=NS(name=name,arguments=json.dumps(args)))

def setup(monkeypatch,replies):
 create=Mock(side_effect=replies)
 llm=NS(chat=NS(completions=NS(create=create)))
 monkeypatch.setattr(bot,'get_active_model',lambda:(llm,'model','test'))
 monkeypatch.setattr(bot,'fish_available',True)
 monkeypatch.setattr(bot,'FISH_VOICE',A)
 monkeypatch.setattr(bot,'_voice_chat_defaults',{})
 monkeypatch.setattr(bot,'_sync_fish_get',lambda v:{'_id':v,'title':'test'})
 monkeypatch.setattr(bot,'_send_agent_voice',AsyncMock(return_value=NS(id=42)))
 monkeypatch.setattr(bot,'send_long',AsyncMock())
 monkeypatch.setattr(bot,'_set_tools_support',Mock())
 monkeypatch.setattr(bot,'get_tracked',lambda:[])
 monkeypatch.setattr(bot,'tavily_api_key',None)
 monkeypatch.setattr(bot,'_index_available',lambda:'disabled')
 monkeypatch.setattr(bot.client,'get_entity',AsyncMock(side_effect=RuntimeError('no entity')))
 return create

def test_ask_delivered_no_final_duplicate(monkeypatch):
 create=setup(monkeypatch,[response(calls=[call('send_chat_responses',{'responses':[{'text':'Hello','mode':'voice','message_id':5}]})]),response('Duplicate summary')])
 result=asyncio.run(bot.ask_agentic('source #5','answer',chat_id=99,msg_by_id={5:NS(raw_text='source')},memory_allowed=False))
 assert result=='';bot._send_agent_voice.assert_awaited_once_with(99,'Hello',A,5)
 assert 'fish_voice_search' in [x['function']['name'] for x in create.call_args_list[0].kwargs['tools']]

def test_auto_delivered_ledger_and_no_duplicate(monkeypatch):
 setup(monkeypatch,[response(calls=[call('send_chat_responses',{'responses':[{'text':'Hello','mode':'voice','message_id':5}]})]),response('Duplicate')])
 ledger=[];result=asyncio.run(bot.generate_auto_reply('source #5',chat_id=99,msg_by_id={5:NS(raw_text='source')},delivery=ledger))
 assert result=='';assert ledger[0]['text']=='Hello';assert ledger[0]['reply_to']==5

def test_auto_error_after_send_not_redelivered(monkeypatch):
 setup(monkeypatch,[response(calls=[call('send_chat_responses',{'responses':[{'text':'Hello','mode':'voice'}]})]),RuntimeError('network')])
 ledger=[];result=asyncio.run(bot.generate_auto_reply('source',chat_id=99,msg_by_id={},delivery=ledger))
 assert result=='';assert len(ledger)==1;assert bot._send_agent_voice.await_count==1

def test_ask_error_after_send_no_fallback(monkeypatch):
 setup(monkeypatch,[response(calls=[call('send_chat_responses',{'responses':[{'text':'Hello','mode':'voice'}]})]),RuntimeError('network')])
 fallback=AsyncMock(return_value='duplicate');monkeypatch.setattr(bot,'generate_ask_reply',fallback)
 result=asyncio.run(bot.ask_agentic('source','answer',chat_id=99,msg_by_id={},memory_allowed=False))
 assert result=='';fallback.assert_not_called()

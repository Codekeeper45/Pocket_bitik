import asyncio,json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock,Mock
import pytest
from voice_agent_tools import VoiceTurn, valid_voice
A='a'*32;B='b'*32

def bot():
 return NS(REPLY_MAX=10,REPLY_COLLAPSE=300,FISH_VOICE=A,FISH_FAVORITES=[{'id':A,'title':'First'}],_voice_chat_defaults={},_save_voice_chat_defaults=Mock(),_sync_fish_get=lambda v:{'_id':v,'title':v},_send_agent_voice=AsyncMock(),send_long=AsyncMock(),_html_clean_markdown=lambda t:t,_send_quote_reply=AsyncMock(return_value=False))
def run(t,n,a):return json.loads(asyncio.run(t.run(n,a)))
def test_mixed_replies():
 b=bot();t=VoiceTurn(b,99,{1:NS(raw_text='a'),2:NS(raw_text='b')})
 r=run(t,'send_chat_responses',{'responses':[{'text':'Hello','mode':'voice','message_id':1,'voice_id':B},{'text':'Link','mode':'text','message_id':2}]})
 assert [x['status'] for x in r['results']]==['sent','sent'];b._send_agent_voice.assert_awaited_once_with(99,'Hello',B,1);assert b.send_long.call_args.kwargs['reply_to']==2;assert b.FISH_VOICE==A
@pytest.mark.parametrize('row',[{'text':'x','mode':'voice','voice_id':'invalid'},{'text':'x','mode':'text','message_id':999},{'text':'','mode':'voice'},{'text':'x','mode':'wrong'}])
def test_invalid_not_sent(row):
 b=bot();t=VoiceTurn(b,99,{1:NS()});r=run(t,'send_chat_responses',{'responses':[row]});assert r['results'][0]['status']=='error';assert not t.sent;b.send_long.assert_not_called();b._send_agent_voice.assert_not_called()
def test_select_scoped_and_next_override():
 b=bot();t=VoiceTurn(b,99);assert run(t,'fish_voice_select',{'voice_id':B})['status']=='selected';assert b._voice_chat_defaults=={'99':B};assert b.FISH_VOICE==A
 run(t,'send_chat_responses',{'responses':[{'text':'Hi','mode':'voice','voice_id':A}]});assert b._send_agent_voice.call_args.args[2]==A;assert b._voice_chat_defaults['99']==B

def test_failed_send_no_success():
 b=bot();b._send_agent_voice.side_effect=RuntimeError('provider down');t=VoiceTurn(b,99);r=run(t,'send_chat_responses',{'responses':[{'text':'Hi','mode':'voice'}]});assert r['results'][0]['status']=='error';assert not t.sent

def test_concurrent_different_voices_and_limit():
 async def scenario():
  b=bot();b.REPLY_MAX=0.5;t=VoiceTurn(b,99)
  await asyncio.gather(t.run('send_chat_responses',{'responses':[{'text':'one','mode':'voice','voice_id':A}]}),t.run('send_chat_responses',{'responses':[{'text':'two','mode':'voice','voice_id':B}]}))
  assert len(t.sent)==1;assert b._send_agent_voice.await_count==1
 asyncio.run(scenario())

def test_dedupe():
 b=bot();t=VoiceTurn(b,99);a={'responses':[{'text':'one','mode':'voice','voice_id':A}]};run(t,'send_chat_responses',a);r=run(t,'send_chat_responses',a);assert r['results'][0]['status']=='already_sent';assert b._send_agent_voice.await_count==1

def test_force_voice():
 b=bot();t=VoiceTurn(b,99,force_voice=True);run(t,'send_chat_responses',{'responses':[{'text':'Hello','mode':'text'}]});b._send_agent_voice.assert_awaited_once();b.send_long.assert_not_called()

def test_favorites():
 b=bot();r=run(VoiceTurn(b,99),'fish_voice_search',{'source':'favorites'});assert r['voices'][0]['voice_id']==A


def test_weighted_twenty_budget_forty_voices():
 b=bot();b.REPLY_MAX=20;t=VoiceTurn(b,99)
 r=run(t,'send_chat_responses',{'responses':[{'text':str(i),'mode':'voice'} for i in range(40)]})
 assert all(x['status']=='sent' for x in r['results']);assert t.counter[0]==20;assert len(t.sent)==40
 r=run(t,'send_chat_responses',{'responses':[{'text':'overflow','mode':'voice'}]});assert r['results'][0]['status']=='error'

def test_weighted_mixed_half_remaining():
 b=bot();b.REPLY_MAX=20;t=VoiceTurn(b,99)
 rows=[{'text':str(i),'mode':'text'} for i in range(19)]+[{'text':'half','mode':'voice'}]
 run(t,'send_chat_responses',{'responses':rows});assert t.counter[0]==19.5
 assert run(t,'send_chat_responses',{'responses':[{'text':'too much','mode':'text'}]})['results'][0]['status']=='error'
 assert t.counter[0]==19.5
 assert run(t,'send_chat_responses',{'responses':[{'text':'fits','mode':'voice'}]})['results'][0]['status']=='sent'
 assert t.counter[0]==20

import asyncio, json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from tests.test_voice_agent_tools import bot, A
from voice_agent_tools import VoiceTurn, VOICE_TOOLS
B='b'*32

def test_scene_single_delivery_and_dedupe():
    b=bot();b._send_agent_voice_scene=AsyncMock(return_value=NS(id=123));t=VoiceTurn(b,99)
    args={'segments':[{'text':'hello','voice_id':A,'role':'one'},{'text':'world','voice_id':B,'role':'two','pause_ms':200}]}
    first=json.loads(asyncio.run(t.run('send_voice_scene',args)))
    second=json.loads(asyncio.run(t.run('send_voice_scene',args)))
    assert first['status']=='sent' and first['voices']==2
    assert second['status']=='already_sent'
    assert b._send_agent_voice_scene.await_count==1
    assert t.counter[0]==0.5


def test_scene_failure_not_recorded_as_delivery():
    b=bot();b._send_agent_voice_scene=AsyncMock(side_effect=RuntimeError('tts failed'));t=VoiceTurn(b,99)
    r=json.loads(asyncio.run(t.run('send_voice_scene',{'segments':[{'text':'hello','voice_id':A}]})))
    assert r['status']=='error' and not t.sent and t.counter[0]==0


def test_scene_tool_registered():
    assert 'send_voice_scene' in [t['function']['name'] for t in VOICE_TOOLS]

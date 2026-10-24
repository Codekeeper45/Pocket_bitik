import json
from types import SimpleNamespace
import pytest
from gen_participants import discover_participants

@pytest.mark.asyncio
async def test_actual_scoped_accounts_not_invented_cast():
    class Message:
        def __init__(self, uid, mid, bot=False):
            self.sender_id,self.id,self.bot=uid,mid,bot
        async def get_sender(self):
            return SimpleNamespace(first_name='Name',last_name='',username=None,bot=self.bot,photo=None)
    class Client:
        async def iter_messages(self, chat, limit):
            assert chat == -123
            for m in [Message(1,10),Message(1,9),Message(2,8),Message(3,7,True),Message(-8,6)]:
                yield m
    result=json.loads(await discover_participants(Client(),-123,exclude_ids={2}))
    assert result['returned_count']==1
    assert result['participants'][0]['user_id']==1
    assert result['participants'][0]['message_count']==2
    assert 'not inspected' in result['participants'][0]['identity_status']

import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace as NS
import bot_new as bot

class ThreadContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_distant_reply_not_lost_outside_numeric_window(self):
        async def empty(*a,**kw):
            if False:yield None
        def message(mid,parent=None,sender=2):
            return NS(id=mid,sender_id=sender,raw_text='source',out=True,date=None,sender=None,reply_to_msg_id=parent,reply_to=None,photo=None,document=None,voice=None,video=None,audio=None,video_note=None,sticker=None,gif=None,fwd_from=None,media=None,action=None)
        center=message(100);child=message(800,100)
        async def get(chat,ids):return [child] if isinstance(ids,list) else center
        fake=NS(iter_messages=empty,get_messages=get,get_entity=AsyncMock(return_value=NS(username='chat')))
        scope={}
        with patch.object(bot,'client',fake),patch.object(bot,'db_read',AsyncMock(return_value=[{'msg_id':800}])),patch.object(bot,'_label_for',return_value='user'),patch.object(bot,'_fmt_date',return_value='date'),patch.object(bot,'_forward_src',return_value=None),patch.object(bot,'_media_tag',return_value=''):
            result=await bot._run_chat_read_context(99,{'message_id':100},scope)
            self.assertIn('#800',result)
            self.assertIn(800,scope)
            result=await bot._run_chat_read_context(99,{'message_id':100},{},exclude_ids={2})
            self.assertNotIn('#800',result)

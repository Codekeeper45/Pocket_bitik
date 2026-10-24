import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace as NS
import bot_new as bot

class HistoricalJevTests(unittest.IsolatedAsyncioTestCase):
    async def test_historical_candidate_survives_recent_window_and_scopes(self):
        async def recent(*args, **kwargs):
            if False: yield None
        old=NS(id=71,sender_id=2,raw_text='',photo=True,document=None,out=True,date=None,sender=None)
        fake=NS(iter_messages=recent,get_messages=AsyncMock(return_value=[old]),get_entity=AsyncMock(return_value=NS(username='chat')))
        scope={}
        with patch.object(bot,'client',fake), patch.object(bot,'_index_media_hybrid_hits',AsyncMock(return_value=[{'msg_id':71,'visual_description':'purple bird'}])), patch.object(bot,'_label_for',return_value='user'), patch.object(bot,'_fmt_date',return_value='date'):
            result=await bot._run_chat_ant_search(99,{'query':'purple bird'},scope,include_ids={2})
            self.assertIn('#71',result)
            self.assertIn(71,scope)
            result=await bot._run_chat_ant_search(99,{'query':'purple bird'},{},exclude_ids={2})
            self.assertNotIn('#71',result)

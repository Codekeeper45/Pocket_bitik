import asyncio
from types import SimpleNamespace as NS
from unittest.mock import patch,AsyncMock
import bot_new as b

def test_completed_reference_survives_other_download_timeout():
    async def slow(*args):
        await asyncio.sleep(5)
        return b'slow'
    one=NS(id=1,raw_text='source',out=False,sender_id=7,date=None,download_media=AsyncMock(return_value=b'fast'))
    two=NS(id=2,raw_text='other',out=False,sender_id=7,date=None,download_media=slow)
    with patch.object(b,'client',NS(get_messages=AsyncMock(return_value=[one,two]))),patch.object(b,'_index_is_image_msg',return_value=True),patch.object(b,'_gen_is_own_generation',return_value=False),patch.object(b,'_downscale_img',AsyncMock(side_effect=lambda x:x)),patch.object(b,'_media_key',side_effect=lambda x:str(x.id)),patch.object(b,'_gen_nearby_text_from_db',AsyncMock(return_value='')),patch.object(b,'_gen_describe_candidates_bounded',AsyncMock(return_value=1)),patch.object(b,'GEN_CATALOG_TIMEOUT',0.03):
        rows=asyncio.run(b._gen_fetch_ref_items(99,[{'msg_id':1,'visual_description':'blue cat'},{'msg_id':2}],limit=2))
    assert [r['mid'] for r in rows]==[1]
    assert rows[0]['bytes']==b'fast'

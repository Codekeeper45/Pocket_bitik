import asyncio
from unittest.mock import AsyncMock,patch
import bot_new as b

def test_exact_caption_is_candidate_not_pixel_proof():
    async def db(sql,args):
        if 'm.txt LIKE' in sql:return [{'msg_id':123,'txt':'Rare quoted label','visual_description':'blue cat'}]
        return []
    with patch.object(b,'db_read',side_effect=db),patch.object(b,'_index_embed_query',AsyncMock(return_value=None)):
        rows=asyncio.run(b._index_media_hybrid_hits(99,'Find «Rare quoted label»',3,for_gen=True))
    assert rows[0]['msg_id']==123
    assert rows[0]['caption_match']
    assert not rows[0].get('verified')

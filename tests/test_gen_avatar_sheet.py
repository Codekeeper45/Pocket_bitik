from io import BytesIO
from PIL import Image
import pytest
import gen_avatar_sheet

@pytest.mark.asyncio
async def test_sheet_preserves_unique_member_mapping_and_partial_success(monkeypatch):
    async def fake(client, chat, user, idx, inspect):
        if user=='missing':raise ValueError('missing')
        b=BytesIO();Image.new('RGB',(80,80),'red').save(b,format='PNG')
        return {'bytes':b.getvalue(),'participant_id':int(user),'profile_url':'tg://user?id='+user,'avatar_photo_id':int(user)+1}
    monkeypatch.setattr(gen_avatar_sheet,'fetch_avatar',fake)
    result=await gen_avatar_sheet.fetch_avatar_sheet(None,-1,['12','12','missing','13'],7,None)
    assert [m['participant_id'] for m in result['members']]==[12,13]
    assert result['unavailable']==[{'user':'missing','error':'ValueError'}]
    assert result['idx']==7 and result['from_tool']
    assert all(m['label'] is None for m in result['members'])
    image=Image.open(BytesIO(result['bytes']))
    assert image.size==(640,360)

@pytest.mark.asyncio
async def test_numeric_lookup_retains_resolved_nickname(monkeypatch):
    async def fake(client, chat, user, idx, inspect):
        b=BytesIO(); Image.new('RGB',(80,80),'blue').save(b,format='PNG')
        return {'bytes':b.getvalue(),'participant_id':123,'profile_url':'tg://user?id=123',
                'avatar_photo_id':1,'label':'Тимур Замурий','display_name':'Тимур Замурий','username':None}
    monkeypatch.setattr(gen_avatar_sheet,'fetch_avatar',fake)
    result=await gen_avatar_sheet.fetch_avatar_sheet(None,-1,['123'],7,None)
    assert result['members'][0]['label']=='Тимур Замурий'
    assert result['members'][0]['user']=='123'

@pytest.mark.asyncio
async def test_empty_sheet_never_claims_references():
    with pytest.raises(ValueError):
        await gen_avatar_sheet.fetch_avatar_sheet(None,-1,[],1,None)

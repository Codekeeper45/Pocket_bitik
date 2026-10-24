"""Labeled avatar contact sheets for bounded ensemble reference acquisition."""
import asyncio
from io import BytesIO
from PIL import Image, ImageOps, ImageDraw
from gen_avatar import fetch_avatar

async def fetch_avatar_sheet(client, chat_id, users, idx, inspect_image):
    if not isinstance(users, list) or not 1 <= len(users) <= 12:
        raise ValueError('provide 1 to 12 participant identifiers')
    users = list(dict.fromkeys(str(user).strip() for user in users))
    semaphore = asyncio.Semaphore(3)
    async def fetch(user):
        async with semaphore:
            try:
                item = await asyncio.wait_for(fetch_avatar(client, chat_id, user, 0, inspect_image), timeout=20)
                return user, item, None
            except Exception as exc:
                return user, None, type(exc).__name__
    results = await asyncio.gather(*(fetch(user) for user in users))
    available = [(user, item) for user, item, error in results if item]
    if not available:
        raise ValueError('no accessible avatars in this batch')
    columns = min(4, len(available))
    sheet = Image.new('RGB', (columns * 320, ((len(available)+columns-1)//columns)*360), 'white')
    draw = ImageDraw.Draw(sheet)
    members = []
    for position, (user, item) in enumerate(available):
        x, y = (position % columns)*320, (position//columns)*360
        image = Image.open(BytesIO(item['bytes'])).convert('RGB')
        image = ImageOps.contain(image, (306, 306))
        sheet.paste(image, (x+(320-image.width)//2, y+(306-image.height)//2))
        # Numeric labels avoid transliteration/truncation of names; mapping below is exact.
        draw.text((x+10, y+315), str(item['participant_id']), fill='black')
        members.append({'cell': position+1, 'user': user, 'participant_id': item['participant_id'],
            'profile_url': item['profile_url'], 'avatar_photo_id': item['avatar_photo_id']})
    output = BytesIO(); sheet.save(output, format='PNG')
    return {'idx': idx, 'bytes': output.getvalue(), 'thumb': output.getvalue(), 'mime': 'image/png',
        'caption': 'Avatar contact sheet. Cell numeric labels identify accounts, NOT final character captions.',
        'provenance': 'profile_avatar_sheet', 'from_tool': True, 'members': members,
        'visual_desc': 'Inspect each avatar cell separately. Preserve account-to-cell mapping. Avatars are not proof of real appearance.',
        'unavailable': [{'user': user, 'error': error} for user, item, error in results if not item]}


def tool_schema():
    return {'type':'function','function':{'name':'chat_get_avatar_sheet',
        'description':'Fetch up to 120 current-chat participant avatars, automatically divided into labeled sheets of 12. Supply the complete selected roster in one call using account IDs from chat_list_participants. Each sheet is a provider reference. Inspect each cell; preserve identity mappings, not the grid layout.',
        'parameters':{'type':'object','properties':{'users':{'type':'array','items':{'type':'string'},'minItems':1,'maxItems':120}},'required':['users'],'additionalProperties':False}}}

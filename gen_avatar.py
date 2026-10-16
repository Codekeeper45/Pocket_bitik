"""Current-chat avatar reference acquisition, independent of prompt text."""
async def fetch_avatar(client, chat_id, target, idx, inspect_image):
    target = str(target or '').strip()
    if not target:
        raise ValueError('participant identifier required')
    entity = await client.get_entity(int(target) if target.lstrip('-').isdigit() else target)
    await client.get_permissions(chat_id, entity)
    # Resolve fresh profile state: entity.photo is the currently displayed main avatar.
    entity = await client.get_entity(entity)
    raw = await client.download_profile_photo(entity, file=bytes, download_big=True)
    if not raw:
        raise ValueError('participant has no accessible current main avatar')
    photo_id = getattr(getattr(entity, 'photo', None), 'photo_id', None)
    blob, mime, width, height = inspect_image(raw)
    username = getattr(entity, 'username', None)
    url = 'https://t.me/' + username if username else 'tg://user?id=' + str(entity.id)
    return {'idx': idx, 'bytes': blob, 'thumb': blob, 'mime': mime,
            'caption': 'Current participant avatar ' + target,
            'provenance': 'profile_avatar', 'profile_url': url,
            'participant_id': entity.id, 'avatar_photo_id': photo_id, 'avatar_kind': 'current_main',
            'visual_desc': 'Inspect this actual avatar. It may be a drawing, mascot or photo; account association is not proof of real-life appearance.'}

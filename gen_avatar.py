"""Current-chat avatar reference acquisition, independent of prompt text."""
async def fetch_avatar(client, chat_id, target, idx, inspect_image):
    target = str(target or '').strip()
    if not target:
        raise ValueError('participant identifier required')
    entity = await client.get_entity(int(target) if target.lstrip('-').isdigit() else target)
    try:
        await client.get_permissions(chat_id, entity)
    except Exception as exc:
        if type(exc).__name__ != 'ChatAdminRequiredError':
            raise
        # Hidden member lists require admin rights. A real message by this sender
        # in the exact current chat is sufficient scoped evidence, not a global lookup.
        messages = await client.get_messages(chat_id, from_user=entity, limit=1)
        if not any(getattr(m, 'sender_id', None) == entity.id for m in messages):
            raise ValueError('participant not evidenced in current chat') from exc
    # Resolve fresh profile state: entity.photo is the currently displayed main avatar.
    entity = await client.get_entity(entity)
    raw = await client.download_profile_photo(entity, file=bytes, download_big=True)
    if not raw:
        raise ValueError('participant has no accessible current main avatar')
    photo_id = getattr(getattr(entity, 'photo', None), 'photo_id', None)
    blob, mime, width, height = inspect_image(raw)
    username = getattr(entity, 'username', None)
    url = 'https://t.me/' + username if username else 'tg://user?id=' + str(entity.id)
    display_name = ' '.join(str(x) for x in (getattr(entity, 'first_name', ''), getattr(entity, 'last_name', '')) if x).strip()
    label = username or display_name or None
    return {'username': username, 'display_name': display_name, 'label': label,
            'idx': idx, 'bytes': blob, 'thumb': blob, 'mime': mime,
            'caption': 'Current participant avatar ' + target,
            'provenance': 'profile_avatar', 'profile_url': url,
            'participant_id': entity.id, 'avatar_photo_id': photo_id, 'avatar_kind': 'current_main',
            'visual_desc': 'Inspect this actual avatar. It may be a drawing, mascot or photo; account association is not proof of real-life appearance.'}

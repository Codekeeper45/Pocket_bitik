"""Plain text extraction for regular Telegram messages and RichMessage articles."""


def _value(node, name, default=None):
    return node.get(name, default) if isinstance(node, dict) else getattr(node, name, default)


def _text(node):
    if node is None:
        return ''
    if isinstance(node, str):
        return node
    parts = _value(node, 'texts')
    if parts is not None:
        return ''.join(_text(item) for item in parts)
    text = _value(node, 'text')
    if text is not None:
        return _text(text)
    return _value(node, 'alt', '') or ''


def _blocks(blocks):
    parts = []
    for block in blocks or []:
        text = _text(_value(block, 'text'))
        if text:
            parts.append(text)
        caption = _value(block, 'caption')
        if caption:
            caption_text = _text(_value(caption, 'text'))
            if caption_text:
                parts.append(caption_text)
        for name in ('blocks', 'items'):
            children = _value(block, name)
            if children:
                parts.extend(_blocks(children))
    return parts


def message_text(message):
    """Return article text without interpreting it as commands or HTML."""
    if message is None:
        return ''
    plain = _value(message, 'raw_text') or _value(message, 'message') or ''
    rich = _value(message, 'rich_message')
    article = '\n\n'.join(_blocks(_value(rich, 'blocks', []))) if rich else ''
    if article:
        return (plain + '\n\n' + article).strip() if plain and plain != article else article
    return plain

"""Fail closed when prompt planning explicitly reports a missing identity source."""
import re

def missing_identity_source(prompt: str, idea: str | None, has_references: bool) -> bool:
    text = (str(idea or '') + '\n' + str(prompt or '')).lower()
    absent = bool(re.search(r'(?:исходн\w*|настоящ\w*|оригинальн\w*)[^\n.]{0,90}(?:не найден|не удалось найти|отсутству)', text))
    absent |= bool(re.search(r'(?:original|actual|real|source)\s+(?:photo|image|reference)[^\n.]{0,60}(?:not found|unavailable|missing)', text))
    substitute = any(x in text for x in ('собирательный', 'generic portrait', 'generic young', 'вымышленный портрет', 'не найдены'))
    return absent or (substitute and not has_references)

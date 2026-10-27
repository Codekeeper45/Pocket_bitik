import asyncio
import base64
import io
import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('api_id', '12345')
os.environ.setdefault('api_hash', 'deadbeefdeadbeefdeadbeefdeadbeef')
import bot_new as bot
from gen_prompt import ContractError
from PIL import Image


def test_gpt_edit_preserves_inputs_and_compacts_wire_payload():
    src = io.BytesIO(); Image.effect_noise((1280, 1280), 100).convert('RGB').save(src, 'PNG')
    value = base64.b64encode(src.getvalue()).decode()
    captured = {}
    def post(url, **kwargs):
        captured.update(kwargs)
        for _, (_, file, mime) in kwargs['files']:
            raw = file.getvalue()
            assert len(raw) <= 100_000
            assert mime == 'image/jpeg'
            Image.open(io.BytesIO(raw)).verify()
        result = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'data': [{'b64_json': base64.b64encode(b'result').decode()}]})
        return result
    with patch.object(bot.requests, 'post', side_effect=post):
        raw, mime = bot._sync_generate_image('scene', [value, value], 'gpt-image-2', '2K', '1:1')
    assert raw == b'result' and mime == 'image/png'
    assert len(captured['files']) == 2
    assert captured['timeout'] == (120, 600)

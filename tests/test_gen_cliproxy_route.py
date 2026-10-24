import base64
from unittest.mock import patch, Mock
import bot_new as b

def test_prompt_cliproxy():
    with patch.object(b,'GEN_PROMPT_PROVIDER','cliproxy'), patch.object(b,'GEN_PROMPT_MODEL','gemini-3.8-flash'):
        assert b.get_gen_planner_model()[:2] == (b.cliproxy_client,'gemini-3.8-flash')

def test_image_generation_route():
    r=Mock();r.json.return_value={'data':[{'b64_json':base64.b64encode(b'image').decode()}]}
    with patch.object(b.requests,'post',return_value=r) as p,patch.object(b,'cliproxy_api_key','test'):
        assert b._sync_generate_image('cup',model='gpt-image-2') == (b'image','image/png')
        assert p.call_args.args[0].endswith('/images/generations')

def test_image_ref_route():
    r=Mock();r.json.return_value={'data':[{'b64_json':base64.b64encode(b'image').decode()}]}
    with patch.object(b.requests,'post',return_value=r) as p,patch.object(b,'cliproxy_api_key','test'):
        b._sync_generate_image('cup',[base64.b64encode(b'ref').decode()],model='gpt-image-2')
        assert p.call_args.args[0].endswith('/images/edits')
        assert len(p.call_args.kwargs['files'])==1

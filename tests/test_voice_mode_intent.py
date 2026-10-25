import ast,re
from pathlib import Path
import pytest
s=Path(__file__).resolve().parents[1].joinpath('bot_new.py').read_text()
t=ast.parse(s)
pattern=None
for n in ast.walk(t):
 if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='explicit_voice_req' for x in n.targets):
  pattern=next(x.args[0].value for x in ast.walk(n.value) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='search')
@pytest.mark.parametrize('text',['проверь voice_id invalid-test-id','выбери сам текст или voice','не отвечай voice: только текст'])
def test_voice_mention_not_force(text):assert not re.search(pattern,text)
@pytest.mark.parametrize('text',['ответь голосом','пришли мне голосовым сообщением','скажи голосовухой'])
def test_explicit_force(text):assert re.search(pattern,text)

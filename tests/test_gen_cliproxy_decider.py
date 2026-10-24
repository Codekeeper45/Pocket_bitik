import json
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import pytest
import bot_new as b

@pytest.mark.parametrize('choice',['a','not_found'])
def test_typed_search_uses_cliproxy_without_paid_network(choice):
    create=Mock(return_value=NS(choices=[NS(message=NS(content=json.dumps({'choice':choice})))]))
    cl=NS(chat=NS(completions=NS(create=create)))
    with patch.object(b,'GEN_PROMPT_PROVIDER','cliproxy'),patch.object(b,'GEN_PROMPT_MODEL','gemini-3.8-flash'),patch.object(b,'cliproxy_client',cl),patch.object(b.urllib.request,'urlopen',side_effect=AssertionError('Paid network forbidden')):
        assert b._jev_sync_choice('evidence','pick',{'a':'block'})['choice']==choice
        assert create.call_args.kwargs['model']=='gemini-3.8-flash'

def test_typed_search_rejects_unsupplied_key():
    cl=NS(chat=NS(completions=NS(create=Mock(return_value=NS(choices=[NS(message=NS(content='{"choice":"invented"}'))])))))
    with patch.object(b,'GEN_PROMPT_PROVIDER','cliproxy'),patch.object(b,'cliproxy_client',cl):
        with pytest.raises(ValueError):b._jev_sync_choice('data','pick',{'a':'block'})

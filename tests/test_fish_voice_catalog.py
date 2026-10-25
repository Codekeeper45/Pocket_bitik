import asyncio,json
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from voice_agent_tools import VoiceTurn
A='a'*32;B='b'*32

def turn(pages):
 get=Mock(side_effect=[NS(raise_for_status=lambda:None,json=lambda p=p:p) for p in pages])
 b=NS(requests=NS(get=get),fish_audio_api_key='test',FISH_MODELS_URL='https://test/model',FISH_FAVORITES=[],_sync_fish_get=lambda v:dict(_id=v,title='Exact',description='Detailed',like_count=5))
 return VoiceTurn(b,1),get

def run(t,args):return json.loads(asyncio.run(t.run('fish_voice_search',args)))
def test_description_filters_and_sort():
 t,get=turn([{'items':[{'_id':A,'title':'A','description':'Warm voice','languages':['ru'],'tags':['male'],'like_count':3,'task_count':10},{'_id':B,'description':'Warm calm','languages':['ru'],'tags':['male'],'like_count':9,'task_count':20}], 'has_more':True,'total':1000,'total_is_exact':False}])
 r=run(t,{'description':'warm','language':'ru','tags':['male'],'min_likes':4,'min_uses':15,'sort_by':'like_count'})
 assert r['voices'][0]['voice_id']==B;assert r['items_scanned']==2;assert r['upstream_total_is_exact'] is False;assert r['sort_scope']=='scanned_pages_only';assert get.call_args.kwargs['params']['tag']==['male']
def test_id_lookup_and_metadata():
 t,get=turn([]);r=run(t,{'query':A});assert r['scope']=='exact_id';assert r['voices'][0]['description']=='Detailed';assert r['voices'][0]['popularity']['score'] is None;get.assert_not_called()
def test_pagination_empty_and_more():
 t,get=turn([{'items':[{'_id':A,'description':'a'}],'has_more':True},{'items':[{'_id':B,'description':'b'}],'has_more':False}]);r=run(t,{'description':'missing','scan_pages':2});assert r['voices']==[];assert r['pages_scanned']==2;assert r['next_page'] is None;assert 'absence' in r['description_scope']
@pytest.mark.parametrize('args',[{'scan_pages':6},{'page':0},{'sort_by':'invented'},{'min_likes':-1},{'tags':'male'}])
def test_invalid_filters(args):
 t,get=turn([]);assert run(t,args)['status']=='error';get.assert_not_called()

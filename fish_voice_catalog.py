"""Fish catalog discovery with explicit upstream vs scanned-page filters."""
import asyncio
import re

SEARCH_SCHEMA = {
 'query':{'type':'string','description':'Title search, or exact Fish voice ID'},
 'voice_id':{'type':'string','description':'Exact voice ID; returns full metadata'},
 'title':{'type':'string','description':'Upstream title search, combined with local description filter'},
 'description':{'type':'string','description':'Case-insensitive substring in descriptions of scanned catalog pages only. Not a global Fish full-text search.'},
 'language':{'type':'string'}, 'tags':{'type':'array','items':{'type':'string'}},
 'author_id':{'type':'string'},'licensed':{'type':'boolean'},
 'sort_by':{'type':'string','enum':['score','task_count','created_at','like_count','mark_count']},
 'min_likes':{'type':'integer','minimum':0},'min_uses':{'type':'integer','minimum':0},
 'page':{'type':'integer','minimum':1},'scan_pages':{'type':'integer','minimum':1,'maximum':5},
 'source':{'type':'string','enum':['catalog','favorites']},'limit':{'type':'integer','minimum':1,'maximum':20}}
GET_TOOL={'type':'function','function':{'name':'fish_voice_get','description':'Inspect exact Fish voice ID: title, description, tags, language, author, popularity counters, available quality metadata and sample links. Counts are popularity, not a star rating.','parameters':{'type':'object','properties':{'voice_id':{'type':'string'}},'required':['voice_id']}}}

def meta(row):
 def num(k):
  v=row.get(k);return v if isinstance(v,(int,float)) and not isinstance(v,bool) else None
 author=row.get('author') or {}
 samples=[]
 for sample in (row.get('samples') or [])[:3]:
  if isinstance(sample,str):samples.append(sample)
  elif isinstance(sample,dict):samples.append({k:sample[k] for k in ['audio','url','audio_url','text'] if k in sample})
 return {'voice_id':row.get('_id') or row.get('id'),'title':str(row.get('title') or '')[:240],
 'description':str(row.get('description') or '')[:4000],'languages':row.get('languages') or [],'tags':row.get('tags') or [],
 'popularity':{k:num(k) for k in ['like_count','mark_count','shared_count','task_count','score']},
 'rating_note':'No star rating is provided. score is a provider ranking if present, not a quality guarantee.',
 'author':{k:author[k] for k in ['_id','nickname'] if k in author},
 'licensed':row.get('licensed'),'state':row.get('state'),'visibility':row.get('visibility'),
 'created_at':row.get('created_at'),'updated_at':row.get('updated_at'),
 'quality':row.get('quality'),'samples':samples,'url':'https://fish.audio/m/'+str(row.get('_id') or row.get('id') or '')}

async def search(turn,args):
 b=turn.bot;limit=int(args.get('limit',6));page=int(args.get('page',1));scan=int(args.get('scan_pages',1))
 if not 1<=limit<=20 or page<1 or not 1<=scan<=5:raise ValueError('limit1..20, page>=1, scan_pages1..5 required')
 query=args.get('query','')
 vid=args.get('voice_id') or (query if isinstance(query,str) and re.fullmatch('[a-fA-F0-9]{32}',query) else None)
 if vid:return {'status':'ok','voices':[await turn.metadata(vid)],'scope':'exact_id'}
 sort=args.get('sort_by','score')
 if sort not in ['score','task_count','created_at','like_count','mark_count']:raise ValueError('Unsupported sort_by')
 for key in ['min_likes','min_uses']:
  if key in args and (isinstance(args[key],bool) or not isinstance(args[key],int) or args[key]<0):raise ValueError('Invalid '+key)
 title=args.get('title',query) or '';description=str(args.get('description') or '').casefold()
 tags=args.get('tags') or []
 if not isinstance(tags,list) or any(not isinstance(x,str) for x in tags):raise ValueError('tags must be string array')
 rows=[];pages=0;last={};scanned=0;source=args.get('source','catalog')
 if source not in ['catalog','favorites']:raise ValueError('Unsupported source')
 if source=='favorites':
  favorites=b.FISH_FAVORITES
  # Hydrate requested slice to expose real metadata rather than invented favorite counters.
  chosen=favorites[(page-1)*limit:(page-1)*limit+limit]
  for x in chosen:rows.append(await turn.metadata(x.get('id') or x.get('_id')))
  scanned=len(rows);last={'has_more':page*limit<len(favorites),'total':len(favorites),'total_is_exact':True}
 else:
  params={'title':str(title)[:160],'sort_by':sort if sort in ['score','task_count','created_at'] else 'score','page_size':max(limit,50) if description or tags or 'min_likes' in args or 'min_uses' in args or sort in ['like_count','mark_count'] else limit}
  for key in ['language','author_id']:
   if args.get(key):params[key]=str(args[key])[:100]
  if tags:params['tag']=tags
  if 'licensed' in args:
   if not isinstance(args['licensed'],bool):raise ValueError('licensed must be boolean')
   params['licensed']=str(args['licensed']).lower()
  for current in range(page,page+scan):
   params['page_number']=current
   def fetch():
    r=b.requests.get(b.FISH_MODELS_URL,headers={'Authorization':'Bearer '+b.fish_audio_api_key},params=params.copy(),timeout=30);r.raise_for_status();return r.json()
   last=await asyncio.to_thread(fetch);pages+=1
   items=last.get('items',[]);scanned+=len(items);rows.extend(meta(x) for x in items)
   if not items or last.get('has_more') is False:break
 def matches(row):
  if source=='favorites' and str(title).casefold() not in row['title'].casefold():return False
  if description not in row['description'].casefold():return False
  if args.get('language') and args['language'] not in row['languages']:return False
  if tags and not set(tags).issubset(set(row['tags'])):return False
  if args.get('author_id') and row['author'].get('_id')!=args['author_id']:return False
  if args.get('licensed') is True and row.get('licensed') is not True:return False
  for param,field in [('min_likes','like_count'),('min_uses','task_count')]:
   if param in args and (row['popularity'][field] is None or row['popularity'][field]<args[param]):return False
  return True
 filtered=[x for x in rows if matches(x)]
 if sort in ['like_count','mark_count']:filtered.sort(key=lambda x:x['popularity'].get(sort) if x['popularity'].get(sort) is not None else -1,reverse=True)
 for row in filtered:turn.known[row['voice_id']]=row
 return {'status':'ok','voices':filtered[:limit],'scope':'scanned_catalog_pages' if source=='catalog' else 'favorites_page',
 'pages_scanned':pages,'items_scanned':scanned,'matches_in_scanned_pages':len(filtered),'returned':min(limit,len(filtered)),
 'page':page,'next_page':page+pages if last.get('has_more') else None,
 'upstream_has_more':last.get('has_more'),'upstream_total':last.get('total'),'upstream_total_is_exact':last.get('total_is_exact'),
 'sort_scope':'scanned_pages_only' if sort in ['like_count','mark_count'] else 'upstream',
 'description_scope':'scanned_pages_only; absence of matches is not absence from full catalog',
 'window_limited':last.get('window_limited')}

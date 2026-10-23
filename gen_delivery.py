"""Scoped durable delivery records; uncertainty never triggers automatic resend."""
import asyncio,json,os,tempfile,time,re
from contextlib import asynccontextmanager
from pathlib import Path
ROOT=Path('gen_delivery_records')
_DELIVERY_LOCKS={}

@asynccontextmanager
async def delivery_lock(key):
 """Hold before lookup through reconciliation, send and persistence; non-reentrant.

 Shared by initial delivery and retries in one event loop, not across processes.
 Acquire a series lock first when both are required. Cancelled waiters are removed.
 """
 marker(key)
 scope=(asyncio.get_running_loop(),key)
 entry=_DELIVERY_LOCKS.setdefault(scope,[asyncio.Lock(),0])
 entry[1]+=1
 try:
  async with entry[0]:yield
 finally:
  entry[1]-=1
  if not entry[1]:del _DELIVERY_LOCKS[scope]

def normalize_chat(chat,*,self_id=None):
 """Compare scopes; self_id must come from the authenticated client, not a request."""
 if str(chat)!='me' or self_id is None:return str(chat)
 if type(self_id) is not int or self_id<=0:raise ValueError('invalid authenticated self id')
 return str(self_id)

async def resolve_chat(client,chat):
 """Resolve Saved Messages before sending or recording; fail closed if unavailable."""
 if str(chat)!='me':return chat
 own=await client.get_me()
 self_id=getattr(own,'id',None)
 if type(self_id) is not int or self_id<=0:raise RuntimeError('Saved Messages identity unavailable')
 return self_id

def marker(key):
 if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',key):raise ValueError('unsafe key')
 return 'gen-artifact:'+key

def record(key,*,chat,reply_to,path,status,message_id=None,caption=None,self_id=None):
 marker(key);ROOT.mkdir(parents=True,exist_ok=True)
 target=ROOT/(key+'.json')
 try:old=json.loads(target.read_text())
 except FileNotFoundError:old={}
 if old and normalize_chat(old['chat'],self_id=self_id)!=normalize_chat(chat,self_id=self_id):raise PermissionError('artifact belongs to another chat')
 if old.get('status')=='delivered' and old.get('message_id') is not None:
  return old
 if status=='not_started' and old and old.get('status')!='not_started':
  return old
 if message_id is None:message_id=old.get('message_id')
 if str(chat)=='me' and self_id is not None:chat=int(normalize_chat(chat,self_id=self_id))
 obj={**old,'key':key,'chat':chat,'reply_to':reply_to,'path':str(path),'status':status,'message_id':message_id,'updated':time.time()}
 if caption is not None:obj['caption']=caption
 fd,tmp=tempfile.mkstemp(dir=ROOT)
 try:
  with os.fdopen(fd,'w') as f:json.dump(obj,f);f.flush();os.fsync(f.fileno())
  os.replace(tmp,target)
  try:
   directory=os.open(ROOT,os.O_DIRECTORY)
   try:os.fsync(directory)
   finally:os.close(directory)
  except (AttributeError,OSError):pass
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
 for p in ROOT.glob('*.json'):
  if time.time()-p.stat().st_mtime>86400:p.unlink(missing_ok=True)
 return obj

def _ledger_path():
 return ROOT/'outputs.jsonl'

def record_output(*,chat,message_id,media_id=None,artifact_key=None):
 """Append confirmed output identity; records are durable across artifact cleanup."""
 if message_id is None:return None
 ROOT.mkdir(parents=True,exist_ok=True)
 obj={'chat':str(chat),'message_id':str(message_id),'media_id':str(media_id) if media_id is not None else None,'artifact_key':artifact_key,'updated':time.time(),'provenance':'generated'}
 with open(_ledger_path(),'a',encoding='utf-8') as f:
  f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
 return obj

def known_output(chat,message_id,media_id=None,*,self_id=None):
 if message_id is None:return False
 try:
  with open(_ledger_path(),encoding='utf-8') as f:
   for line in f:
    try:o=json.loads(line)
    except (ValueError,TypeError):continue
    if normalize_chat(o.get('chat'),self_id=self_id)==normalize_chat(chat,self_id=self_id) and (str(o.get('message_id'))==str(message_id) or (media_id is not None and o.get('media_id') is not None and str(o['media_id'])==str(media_id))):return True
 except FileNotFoundError:pass
 return False

def lookup(key,chat,*,self_id=None):
 marker(key);obj=json.loads((ROOT/(key+'.json')).read_text())
 if normalize_chat(obj['chat'],self_id=self_id)!=normalize_chat(chat,self_id=self_id):raise PermissionError('artifact belongs to another chat')
 if time.time()-obj['updated']>86400:raise FileNotFoundError('artifact expired')
 path=Path(obj['path']).resolve();root=Path('gen_artifacts').resolve()
 if not path.is_relative_to(root):raise PermissionError('unsafe artifact path')
 if not path.is_file():raise FileNotFoundError('artifact expired')
 return obj

async def lookup_for_client(client,key,chat):
 """Resolve only a legacy Saved Messages scope mismatch using authenticated self.

 Return the stored scope unchanged: use entry['chat'] for record updates and the
 validated request chat for sending. Caller must hold delivery_lock during retries.
 """
 try:return lookup(key,chat)
 except PermissionError:
  marker(key)
  entry=json.loads((ROOT/(key+'.json')).read_text())
  if entry.get('chat')!='me' or str(chat)=='me':raise
  self_id=await resolve_chat(client,'me')
  return lookup(key,chat,self_id=self_id)

async def reconcile(client,entry):
 """Read exact caption marker. Raise on uncertain query, never pretend absent."""
 chat=await resolve_chat(client,entry['chat'])
 def matches_scope(msg):
  if not getattr(msg,'out',False):return False
  if 'reply_to' in entry:
   if getattr(msg,'reply_to_msg_id',None)!=entry['reply_to']:return False
   if not getattr(msg,'media',None):return False
  if str(entry['chat'])=='me' and getattr(msg,'chat_id',None) is None:return False
  if getattr(msg,'chat_id',None) is not None and normalize_chat(msg.chat_id)!=normalize_chat(chat):return False
  return True
 if entry.get('message_id'):
  msg=await client.get_messages(chat,ids=entry['message_id'])
  if msg and str(getattr(msg,'id',None))==str(entry['message_id']) and matches_scope(msg):return msg
  raise RuntimeError('saved delivery cannot be verified')
 # Legacy captions had a searchable marker. New clean captions cannot prove
 # that a timed-out send was absent, so never risk sending a duplicate.
 exact_marker=re.compile(r'(?<![A-Za-z0-9_:-])'+re.escape(marker(entry['key']))+r'(?![A-Za-z0-9_-])')
 if not exact_marker.search(entry.get('caption', '')):
  raise RuntimeError('ambiguous unmarked delivery requires manual verification')
 found=await client.get_messages(chat,search=marker(entry['key']),limit=100)
 if not isinstance(found,(list,tuple)):raise RuntimeError('delivery search unavailable')
 for msg in found:
  if matches_scope(msg) and exact_marker.search(getattr(msg,'raw_text','') or getattr(msg,'message','') or ''):
   return msg
 if found or getattr(found,'total',0):raise RuntimeError('delivery search cannot establish absence')
 return None

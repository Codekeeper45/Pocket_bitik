"""Scoped durable delivery records; uncertainty never triggers automatic resend."""
import json,os,tempfile,time,re
from pathlib import Path
ROOT=Path('gen_delivery_records')
def marker(key):
 if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',key):raise ValueError('unsafe key')
 return 'gen-artifact:'+key

def record(key,*,chat,reply_to,path,status,message_id=None,caption=None):
 marker(key);ROOT.mkdir(parents=True,exist_ok=True)
 target=ROOT/(key+'.json')
 try:old=json.loads(target.read_text())
 except FileNotFoundError:old={}
 obj={**old,'key':key,'chat':chat,'reply_to':reply_to,'path':str(path),'status':status,'message_id':message_id,'updated':time.time()}
 if caption is not None:obj['caption']=caption
 fd,tmp=tempfile.mkstemp(dir=ROOT)
 try:
  with os.fdopen(fd,'w') as f:json.dump(obj,f);f.flush();os.fsync(f.fileno())
  os.replace(tmp,target)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
 for p in ROOT.glob('*.json'):
  if time.time()-p.stat().st_mtime>86400:p.unlink(missing_ok=True)
 return obj

def lookup(key,chat):
 marker(key);obj=json.loads((ROOT/(key+'.json')).read_text())
 if str(obj['chat'])!=str(chat):raise PermissionError('artifact belongs to another chat')
 if time.time()-obj['updated']>86400:raise FileNotFoundError('artifact expired')
 path=Path(obj['path']).resolve();root=Path('gen_artifacts').resolve()
 if not path.is_relative_to(root):raise PermissionError('unsafe artifact path')
 if not path.is_file():raise FileNotFoundError('artifact expired')
 return obj

async def reconcile(client,entry):
 """Read exact caption marker. Raise on uncertain query, never pretend absent."""
 if entry.get('message_id'):
  msg=await client.get_messages(entry['chat'],ids=entry['message_id'])
  if msg and getattr(msg,'out',False):return msg
  raise RuntimeError('saved delivery cannot be verified')
 # Legacy captions had a searchable marker. New clean captions cannot prove
 # that a timed-out send was absent, so never risk sending a duplicate.
 if marker(entry['key']) not in entry.get('caption', ''):
  raise RuntimeError('ambiguous unmarked delivery requires manual verification')
 found=await client.get_messages(entry['chat'],search=marker(entry['key']),limit=100)
 for msg in found:
  if getattr(msg,'out',False) and marker(entry['key']) in (getattr(msg,'raw_text','') or getattr(msg,'message','') or ''):
   return msg
 return None

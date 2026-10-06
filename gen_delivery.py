"""Scoped artifact delivery records; never regenerate on send failure."""
import json,os,tempfile,time
from pathlib import Path

ROOT=Path('gen_delivery_records')
def record(key,*,chat,reply_to,path,status,message_id=None):
 if not key or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in key):raise ValueError('unsafe key')
 ROOT.mkdir(exist_ok=True)
 obj={'key':key,'chat':chat,'reply_to':reply_to,'path':str(path),'status':status,'message_id':message_id,'updated':time.time()}
 fd,tmp=tempfile.mkstemp(dir=ROOT)
 with os.fdopen(fd,'w') as f:json.dump(obj,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,ROOT/(key+'.json'))
 return obj

def lookup(key,chat):
 if not key or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in key):raise ValueError('unsafe key')
 obj=json.loads((ROOT/(key+'.json')).read_text())
 if str(obj['chat'])!=str(chat):raise PermissionError('artifact belongs to another chat')
 path=Path(obj['path']).resolve();root=Path('gen_artifacts').resolve()
 if not path.is_relative_to(root):raise PermissionError('unsafe artifact path')
 if not path.is_file():raise FileNotFoundError('artifact expired')
 return obj

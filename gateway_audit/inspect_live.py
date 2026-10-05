"""Read-only sanitized gateway account/error evidence. Never prints credentials."""
import collections,json
from pathlib import Path
import requests
cfg={}
for line in Path('/home/hermes/.hermes/secrets/chatgpt2api.env').read_text().splitlines():
    if '=' in line and not line.startswith('#'):
        k,v=line.split('=',1);cfg[k]=v.strip().strip('"').strip("'")
key=next(v for k,v in cfg.items() if 'KEY' in k or 'TOKEN' in k)
s=requests.Session();s.headers['Authorization']='Bearer '+key
base='https://tess-unchary-lenore.ngrok-free.dev'
def get(path,**params):
    r=s.get(base+path,params=params,timeout=30);r.raise_for_status();return r.json()
accounts=get('/api/accounts',page_size=100)
rows=accounts['items']
assert len(rows)==accounts['total'], 'Account pagination incomplete'
identity_counts=collections.Counter(x.get('user_id') or x['id'] for x in rows)
groups={identity:'identity_'+str(i+1) for i,identity in enumerate(identity_counts)}
pool=[dict({k:x.get(k) for k in ['id','available','status_category','quota_remaining','quota_unknown','quota_reset_at','last_remote_check_result','last_remote_checked_at']}, identity=groups[x.get('user_id') or x['id']]) for x in rows]
logs=get('/api/logs',limit=40,offset=0)
evidence=[]
for x in logs['items']:
    if not x.get('endpoint') or x.get('outcome')=='success':continue
    d=get('/api/logs/'+x['id'])
    safe={k:d.get(k) for k in ['id','time','endpoint','error_code','status_code','outcome','duration_ms','attempt_count','switch_count']}
    identity_by_email={x.get('email'):groups[x.get('user_id') or x['id']] for x in rows}
    safe['attempts']=[dict({k:a.get(k) for k in ['attempt','status','outcome','duration_ms','error_code']},identity=identity_by_email.get(a.get('account_email')),instant_limit='Instant limit' in str(a.get('upstream_error') or '')) for a in d.get('attempts',[])]
    evidence.append(safe)
report={'records':len(rows),'identities':len(identity_counts),'statuses':dict(collections.Counter(x.get('status_category') for x in rows)),'accounts':pool,'recent_failures':evidence}
path=Path(__file__).with_name('live_evidence.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:report[k] for k in ['records','identities','statuses']},ensure_ascii=False));print('Failure records',len(evidence));print('Evidence',path)

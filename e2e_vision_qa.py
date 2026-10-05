import json, urllib.request, base64, sys
from pathlib import Path
cfg={}
for line in Path('/home/hermes/.hermes/.env').read_text().splitlines():
 if '=' in line and not line.startswith('#'):
  k,v=line.split('=',1);cfg[k]=v.strip().strip('\"').strip("'")
content=[{'type':'text','text':sys.argv[1]}]
for path in sys.argv[2:]:
 b=base64.b64encode(Path(path).read_bytes()).decode()
 content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+b}})
payload={'model':'gemini-3.8-flash-high','messages':[{'role':'user','content':content}], 'max_tokens':1800}
r=urllib.request.Request('http://127.0.0.1:8317/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+cfg['CLIPROXY_API_KEY'],'Content-Type':'application/json'})
d=json.load(urllib.request.urlopen(r,timeout=120));print(d['choices'][0]['message']['content'])

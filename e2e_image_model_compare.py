"""Bounded gateway comparison; keeps credentials out of output."""
import json,base64,time,sys
from pathlib import Path
import requests
cfg={}
for line in Path('/home/hermes/.hermes/secrets/chatgpt2api.env').read_text().splitlines():
 if '=' in line and not line.startswith('#'):
  k,v=line.split('=',1);cfg[k]=v.strip().strip('\"').strip("'")
key=next((v for k,v in cfg.items() if 'KEY' in k or 'TOKEN' in k),None)
if not key: raise RuntimeError('gateway key missing')
root=Path('qa_artifacts/model_compare');root.mkdir(parents=True,exist_ok=True)
prompt='A square color anime comic page, exactly four panels in a 2x2 grid. Four consistent human anime characters: Asya wearing a teal scarf, Тайлер wearing a red jacket, Тимур holding a sketchbook, Вера wearing a cream cardigan. Print these exact names on their badges. Panel 1: missing artwork in a workshop. Panel 2: the friends follow glowing breadcrumbs. Panel 3: a tiny robot toaster prints the missing artworks on bread. Panel 4: the friends hold an art-toast exhibition. Readable short Russian speech bubbles, distinct expressive faces, clear panel borders, no other lettering or crowds.'
for model in sys.argv[1:]:
 start=time.monotonic();r=requests.post('https://tess-unchary-lenore.ngrok-free.dev/v1/images/generations',headers={'Authorization':f'Bearer {key}'},json={'model':model,'prompt':prompt,'size':'1024x1024','n':1,'response_format':'b64_json'},timeout=180)
 row={'model_requested':model,'status':r.status_code,'seconds':round(time.monotonic()-start,1)}
 try:
  data=r.json();items=data.get('data') or []
  if items and items[0].get('b64_json'):
   raw=base64.b64decode(items[0]['b64_json']);p=root/f'{model}.png';p.write_bytes(raw);row.update(path=str(p.resolve()),bytes=len(raw))
  else:row['error']=str((data.get('error') or {}).get('message','no image'))[:250]
 except Exception as e:row['error']=type(e).__name__
 (root/f'{model}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
 print(json.dumps(row,ensure_ascii=False),flush=True)

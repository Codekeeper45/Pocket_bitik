"""Exercise real prompt model using AST-isolated Gen functions, without starting bot."""
import ast, asyncio, base64, json, re, sys
from pathlib import Path
from openai import OpenAI
cfg={}
for line in Path('/home/hermes/.hermes/.env').read_text().splitlines():
 if '=' in line and not line.startswith('#'):
  k,v=line.split('=',1);cfg[k]=v.strip().strip('"').strip("'")
llm=OpenAI(base_url='http://127.0.0.1:8317/v1',api_key=cfg['CLIPROXY_API_KEY'],timeout=90)
s=Path('bot_new.py').read_text();tree=ast.parse(s)
names={'_gen_unified_system','_build_gen_prompt','_parse_gen_prompt_out','_gen_validated_selection','_gen_role_number_remap','_gen_remap_selected_refs','_gen_actual_role_instruction'}
nodes=[n for n in tree.body if (isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('_GEN_') and isinstance(n.value,ast.Constant) for t in n.targets))]
ns=dict(asyncio=asyncio,base64=base64,json=json,re=re,GEN_CTX_REF_MAX=16,GEN_CTX_IMG_MAX=20,ASK_MAX_TOKENS=8192,ACTIVE_MODEL='cp-gemini-3.8-flash-high',MODEL_TOOLS_SUPPORT={},CHAT_TOOLS=[],get_active_model=lambda:(llm,'gemini-3.8-flash-high','Gemini'),active_model_supports_vision=lambda:False,_strip_think=lambda x:x,_extract_content=lambda msg:msg.content or '',log=lambda *x:print(*x),GEN_IMAGE_MAX_INPUT=3*1024*1024)
async def fallback(*args): return args[0]
ns['_sync_image_prompt']=lambda *args:args[0]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'gen_isolated','exec'),ns)
result=asyncio.run(ns['_build_gen_prompt'](sys.argv[1],catalog=[]))
print(json.dumps(result,ensure_ascii=False,indent=2))

"""Owner-requested live Gen probe. Does not remove messages or expose credentials."""
import asyncio, json, os, sys
from pathlib import Path
from telethon import TelegramClient

async def main():
    cfg = {}
    for line in Path('/home/hermes/.hermes/secrets/telegram.env').read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k,v=line.split('=',1); cfg[k]=v.strip().strip('\"').strip("'")
    import sqlite3
    from telethon.sessions import MemorySession
    from telethon.crypto import AuthKey
    conn=sqlite3.connect('file:/home/hermes/integrations/telegram-user/user.session?mode=ro', uri=True)
    dc, address, port, auth = conn.execute('SELECT dc_id, server_address, port, auth_key FROM sessions').fetchone()
    conn.close()
    session=MemorySession(); session.set_dc(dc,address,port); session.auth_key=AuthKey(auth)
    client=TelegramClient(session, int(cfg['TELEGRAM_API_ID']), cfg['TELEGRAM_API_HASH'])
    await client.connect()
    try:
        action=sys.argv[1]
        target=os.environ.get('GEN_QA_CHAT','me')
        if target.lstrip('-').isdigit(): target=int(target)
        if action=='send':
            reply=int(sys.argv[3]) if len(sys.argv)>3 else None
            m=await client.send_message(target,sys.argv[2],reply_to=reply)
            print(json.dumps({'id':m.id,'text':m.raw_text},ensure_ascii=False))
        elif action=='seed':
            path=sys.argv[2]; caption=sys.argv[3]
            m=await client.send_file(target,path,caption=caption)
            print(json.dumps({'id':m.id,'caption':caption},ensure_ascii=False))
        elif action=='read':
            since=int(sys.argv[2]); rows=[]
            async for m in client.iter_messages(target,limit=30,min_id=since):
                row={'id':m.id,'text':(m.raw_text or '')[:5000],'photo':bool(m.photo),'document':bool(m.document),'reply':getattr(m.reply_to,'reply_to_msg_id',None)}
                row['sender_id']=m.sender_id
                row['reactions']=m.reactions.to_dict() if m.reactions else None
                if m.photo or (m.document and str(getattr(m.file,'mime_type','')).startswith('image/')):
                    out=Path('/home/hermes/projects/Pocket_bitik/qa_artifacts'); out.mkdir(exist_ok=True)
                    row['path']=await client.download_media(m,file=str(out/f'{m.id}'))
                rows.append(row)
            print(json.dumps(rows,ensure_ascii=False,default=str))
    finally:
        await client.disconnect()

if __name__=='__main__': asyncio.run(main())

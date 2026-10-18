#!/usr/bin/env python3
"""Interactive portable setup; credentials never appear in command arguments."""
import getpass
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit
ROOT = Path(__file__).resolve().parent

def quote(value):
    if any(c in value for c in ('\n','\r','\x00')) or '${' in value:
        raise ValueError('Multiline values and dotenv interpolation are not supported')
    return '"'+value.replace('\\','\\\\').replace('"','\\"')+'"'

def write_env(values, path, overwrite=False):
    path=Path(path)
    content='# Private configuration. Never commit.\n'+''.join(k+'='+quote(v)+'\n' for k,v in values.items())
    if path.is_symlink():raise ValueError('Refusing symlink')
    flags=os.O_WRONLY|os.O_CREAT|getattr(os,'O_NOFOLLOW',0)
    flags|=os.O_TRUNC if overwrite else os.O_EXCL
    fd=os.open(path,flags,0o600)
    with os.fdopen(fd,'w') as stream:
        if hasattr(os,'fchmod'):os.fchmod(stream.fileno(),0o600)
        stream.write(content)

def validate(values):
    if not values['api_id'].isdigit() or int(values['api_id'])<=0:raise ValueError('Invalid Telegram api_id')
    import re
    if not re.fullmatch('[a-fA-F0-9]{32}',values['api_hash']):raise ValueError('api_hash must contain 32 hexadecimal characters')
    u=urlsplit(values['OPENAI_COMPAT_BASE_URL'])
    if u.scheme not in ('http','https') or not u.hostname or u.username or u.password or u.query or u.fragment:raise ValueError('Invalid API base URL')
    if u.scheme=='http' and u.hostname not in ('localhost','127.0.0.1','::1'):raise ValueError('Use HTTPS for remote providers')
    if not values['OPENAI_COMPAT_MODEL'] or not values['OPENAI_COMPAT_API_KEY']:raise ValueError('Model and key are required (local servers may use a dummy key)')
    for v in values.values():quote(v)

def ask(label,default=''):
    return input(label+(f' [{default}]' if default else '')+': ').strip() or default

def main():
    if sys.version_info<(3,11):raise SystemExit('Python 3.11+ required')
    os.chdir(ROOT)
    print('Pocket Bitik setup. Secrets are hidden; .env and Telegram session stay private.')
    if sys.prefix==sys.base_prefix and '--in-venv' not in sys.argv:
        if ask('Create .venv and install dependencies? y/N','N').lower()=='y':
            subprocess.run([sys.executable,'-m','venv',str(ROOT/'.venv')],check=True)
            py=ROOT/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            subprocess.run([str(py),'-m','pip','install','-r',str(ROOT/'requirements.txt')],check=True)
            raise SystemExit(subprocess.call([str(py),str(Path(__file__)),'--in-venv']))
    p=ROOT/'.env'
    if p.exists() and ask('Replace existing .env? Type OVERWRITE')!='OVERWRITE':raise SystemExit('Existing config preserved')
    values={'api_id':ask('Telegram api_id from my.telegram.org'),'api_hash':getpass.getpass('Telegram api_hash: ').strip(),
      'TELEGRAM_SESSION':str(ROOT/'session_name'),'OPENAI_COMPAT_ENABLED':'true',
      'OPENAI_COMPAT_BASE_URL':ask('API base URL including /v1','https://api.openai.com/v1'),
      'OPENAI_COMPAT_MODEL':ask('Exact remote model ID'),'OPENAI_COMPAT_API_KEY':getpass.getpass('Provider key (local server: dummy key): ').strip(),
      'OPENAI_COMPAT_VISION':'true' if ask('Does this model accept images? y/N','N').lower()=='y' else 'false'}
    validate(values);write_env(values,p,overwrite=p.exists())
    print('Saved private .env. For an existing bot select /model custom-openai.')
    if ask('Log into Telegram now? y/N','N').lower()=='y':
        from telethon.sync import TelegramClient
        client=TelegramClient(values['TELEGRAM_SESSION'],int(values['api_id']),values['api_hash'])
        client.start(phone=lambda:input('Phone: '),code_callback=lambda:getpass.getpass('Telegram login code: '),password=lambda:getpass.getpass('Telegram 2FA: '))
        client.disconnect()
        session=Path(values['TELEGRAM_SESSION']+'.session')
        if session.exists():session.chmod(0o600)
    print('Run using this Python: '+sys.executable+' bot_new.py')
    print('Image generation, audio and database indexing require separate compatible services; chat API alone does not provide them.')

if __name__=='__main__':main()

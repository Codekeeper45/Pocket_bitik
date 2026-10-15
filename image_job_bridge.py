#!/usr/bin/env python3
"""Local asynchronous OpenAI-compatible image job bridge."""
from __future__ import annotations
import hashlib, json, os, secrets, sqlite3, threading, time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import requests

UPSTREAM = 'http://127.0.0.1:8317/v1'
MAX_BODY = 8 * 1024 * 1024
TTL = 3600
ACTIVE_LIMIT = 4
QUEUE_LIMIT = 8

class JobService:
    def __init__(self, state_dir=None, upstream=UPSTREAM, session=None, max_workers=ACTIVE_LIMIT):
        self.state_dir = Path(state_dir or os.environ.get('IMAGE_JOB_STATE_DIR', '~/.local/state/pocket-image-jobs')).expanduser()
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state_dir, 0o700)
        self.db = self.state_dir / 'jobs.sqlite3'
        self.upstream = upstream.rstrip('/')
        self.session = session or requests
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='image-job')
        self.keys = {}
        with self._connect() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, idem TEXT NOT NULL, fingerprint TEXT NOT NULL, endpoint TEXT NOT NULL, body TEXT NOT NULL, status TEXT NOT NULL, http_status INTEGER, response TEXT, created REAL NOT NULL, updated REAL NOT NULL, UNIQUE(owner,idem))''')
            c.execute("UPDATE jobs SET status='failed',http_status=503,response=? WHERE status IN ('queued','running')", (json.dumps({'error':{'message':'job interrupted by process restart'}}),))
        try: os.chmod(self.db, 0o600)
        except OSError: pass

    def _connect(self):
        c=sqlite3.connect(self.db, timeout=30)
        c.row_factory=sqlite3.Row
        return c

    @staticmethod
    def owner_hash(token): return hashlib.sha256(token.encode()).hexdigest()

    def validate(self, token):
        try:
            r=self.session.get(self.upstream+'/models', headers={'Authorization':'Bearer '+token}, timeout=10)
            return r.status_code == 200
        except requests.RequestException: return False

    def submit(self, token, idem, endpoint, body):
        if endpoint not in ('/images/edits','/images/generations'): return 400, {'error':'unsupported endpoint'}
        if not isinstance(body,dict): return 400, {'error':'body must be an object'}
        model=body.get('model')
        if not isinstance(model,str) or not model.startswith('gpt-image-'): return 400, {'error':'model must be gpt-image-*'}
        # Image references may only be inline data URLs; never let upstream fetch arbitrary URLs.
        for field in ('image','mask','images'):
            val=body.get(field)
            vals=val if isinstance(val,list) else [val]
            for x in vals:
                if x is None:
                    continue
                url = x.get('image_url', '') if isinstance(x, dict) else x
                if isinstance(url, dict):
                    url = url.get('url', '')
                if not isinstance(url, str) or not url.startswith('data:image/'):
                    return 400, {'error':field+' must use data:image URLs'}
        owner=self.owner_hash(token); encoded=json.dumps(body,sort_keys=True,separators=(',',':'))
        fingerprint=hashlib.sha256((endpoint+'\0'+encoded).encode()).hexdigest()
        with self.lock, self._connect() as c:
            now=time.time()
            c.execute('DELETE FROM jobs WHERE created < ?', (now-TTL,))
            row=c.execute('SELECT * FROM jobs WHERE owner=? AND idem=?',(owner,idem)).fetchone()
            if row:
                if row['fingerprint'] != fingerprint: return 409, {'error':'Idempotency-Key reused with different payload'}
                return 202, {'id':row['id'],'status':row['status']}
            active=c.execute("SELECT COUNT(*) n FROM jobs WHERE status IN ('queued','running')").fetchone()['n']
            if active >= ACTIVE_LIMIT+QUEUE_LIMIT: return 429, {'error':'image job queue full'}
            if not self.validate(token): return 401, {'error':'upstream authorization failed'}
            job_id=secrets.token_hex(18)
            c.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,'queued',NULL,NULL,?,?)",(job_id,owner,idem,fingerprint,endpoint,encoded,now,now))
            self.keys[job_id]=token
            self.pool.submit(self._run,job_id,token,endpoint,encoded)
            return 202, {'id':job_id,'status':'queued'}

    def _run(self, job_id, token, endpoint, encoded):
        with self.lock, self._connect() as c:
            c.execute("UPDATE jobs SET status='running',updated=? WHERE id=?",(time.time(),job_id))
        try:
            response=self.session.post(self.upstream+endpoint,headers={'Authorization':'Bearer '+token},json=json.loads(encoded),timeout=600)
            try: data=response.json()
            except (ValueError,requests.exceptions.JSONDecodeError): data={'error':{'message':response.text}}
            status=response.status_code
        except Exception as e:
            status=502; data={'error':{'message':str(e)}}
        finally:
            self.keys.pop(job_id,None)
        with self.lock, self._connect() as c:
            c.execute("UPDATE jobs SET status='done',http_status=?,response=?,body='',updated=? WHERE id=? AND status='running'",(status,json.dumps(data),time.time(),job_id))

    def get(self, token, job_id):
        with self._connect() as c: row=c.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
        if row is None: return 404, {'error':'job not found'}
        if not secrets.compare_digest(row['owner'],self.owner_hash(token)): return 404, {'error':'job not found'}
        if row['status'] in ('queued','running'): return 202, {'id':job_id,'status':row['status']}
        try: result=json.loads(row['response'])
        except (ValueError,TypeError): result={'error':{'message':'job result unavailable'}}
        return row['http_status'] or 503, result

    def close(self): self.pool.shutdown(wait=True)

class Handler(BaseHTTPRequestHandler):
    service=None
    def log_message(self, *_): pass
    def _send(self,status,data):
        raw=json.dumps(data,separators=(',',':')).encode(); self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def _token(self):
        value=self.headers.get('Authorization','')
        return value[7:] if value.startswith('Bearer ') and len(value)>7 else None
    def do_GET(self):
        if self.path=='/health': return self._send(200,{'status':'ok'})
        prefix='/v1/image-jobs/'
        if self.path.startswith(prefix):
            token=self._token()
            if not token: return self._send(401,{'error':'Bearer token required'})
            status,data=self.service.get(token,self.path[len(prefix):].split('?',1)[0]); return self._send(status,data)
        self._send(404,{'error':'not found'})
    def do_POST(self):
        if self.path!='/v1/image-jobs': return self._send(404,{'error':'not found'})
        token=self._token()
        if not token: return self._send(401,{'error':'Bearer token required'})
        idem=self.headers.get('Idempotency-Key')
        if not idem or len(idem)>256: return self._send(400,{'error':'Idempotency-Key required'})
        try: size=int(self.headers.get('Content-Length','0'))
        except ValueError: return self._send(400,{'error':'invalid Content-Length'})
        if size<1 or size>MAX_BODY: return self._send(413 if size>MAX_BODY else 400,{'error':'invalid request size'})
        try: payload=json.loads(self.rfile.read(size))
        except (ValueError,UnicodeDecodeError): return self._send(400,{'error':'invalid JSON'})
        if not isinstance(payload,dict) or set(payload)-{'endpoint','body'} or 'endpoint' not in payload or 'body' not in payload: return self._send(400,{'error':'expected endpoint and body'})
        status,data=self.service.submit(token,idem,payload['endpoint'],payload['body']); self._send(status,data)

def serve():
    service=JobService(); Handler.service=service
    server=ThreadingHTTPServer(('127.0.0.1',8318),Handler)
    try: server.serve_forever()
    finally: server.server_close(); service.close()
if __name__=='__main__': serve()

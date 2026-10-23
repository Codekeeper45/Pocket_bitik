import hashlib
import json
import sqlite3
import tempfile
import threading
import time
import unittest

from image_job_bridge import JobService

class Response:
    def __init__(self,status=200,data=None,text=''):
        self.status_code=status; self._data=data if data is not None else {}; self.text=text
    def json(self): return self._data

class FakeUpstream:
    def __init__(self,delay=0): self.delay=delay; self.calls=[]; self.lock=threading.Lock(); self.started=threading.Event()
    def get(self,url,**kw): return Response(200,{'data':[{'id':'gpt-image-1'}]})
    def post(self,url,**kw):
        with self.lock: self.calls.append((url,kw))
        self.started.set()
        if self.delay: time.sleep(self.delay)
        return Response(429,{'error':{'message':'quota exhausted'}} if kw['json']['prompt']=='error' else {'created':True})

class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.fake=FakeUpstream(.12)
        self.service=JobService(self.tmp.name,'http://fake/v1',self.fake)
    def tearDown(self): self.service.close(); self.tmp.cleanup()
    def submit(self,key='k',body=None,token='secret-token'):
        return self.service.submit(token,key,'/images/generations',body or {'model':'gpt-image-1','prompt':'hi'})
    def test_async_poll_result_and_duplicate_no_repeat(self):
        status,data=self.submit(); self.assertEqual(status,202); jid=data['id']
        self.fake.started.wait(1)
        start=time.monotonic(); poll=self.service.get('secret-token',jid); elapsed=time.monotonic()-start
        self.assertEqual(poll[0],202); self.assertLess(elapsed,.1)
        duplicate=self.submit(); self.assertEqual(duplicate[0],202); self.assertEqual(duplicate[1]['id'],jid)
        for _ in range(100):
            result=self.service.get('secret-token',jid)
            if result[0]!=202: break
            time.sleep(.01)
        self.assertEqual(result,(429,{'created':True})); self.assertEqual(len(self.fake.calls),1)
    def test_auth_idempotency_payload_mismatch_and_endpoint_allowlist(self):
        _,data=self.submit(); self.assertEqual(self.service.get('other',data['id'])[0],404)
        self.assertEqual(self.submit(body={'model':'gpt-image-1','prompt':'different'})[0],409)
        self.assertEqual(self.service.submit('secret-token','bad','/v1/models',{'model':'gpt-image-1'})[0],400)
        self.assertEqual(self.submit(body={'model':'gpt-image-1','prompt':'','image':'http://169.254.169.254/'})[0],400)
    def test_upstream_error_status_body_preserved(self):
        _,data=self.submit(body={'model':'gpt-image-1','prompt':'error'}); jid=data['id']
        for _ in range(100):
            result=self.service.get('secret-token',jid)
            if result[0]!=202: break
            time.sleep(.01)
        self.assertEqual(result,(429,{'error':{'message':'quota exhausted'}}))
    def test_restart_marks_pending_failed(self):
        self.service.close()
        self.service=JobService(self.tmp.name,'http://fake/v1',self.fake)
        body={'model':'gpt-image-1','prompt':'hi'}; encoded=json.dumps(body,sort_keys=True,separators=(',',':'))
        owner=self.service.owner_hash('secret-token'); fp=hashlib.sha256(('/images/generations\0'+encoded).encode()).hexdigest(); jid='deadbeef'
        with sqlite3.connect(self.service.db) as db: db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,'queued',NULL,NULL,?,?)",(jid,owner,'crash',fp,'/images/generations',encoded,time.time(),time.time()))
        restarted=JobService(self.tmp.name,'http://fake/v1',self.fake)
        try: self.assertEqual(restarted.get('secret-token',jid),(503,{'error':{'message':'job interrupted by process restart'}}))
        finally: restarted.close()
    def test_concurrency_cap(self):
        # Keep work pending deterministically; a sleep races fast test hosts.
        release=threading.Event()
        original=self.fake.post
        def blocked(*args,**kwargs):
            release.wait(5)
            return original(*args,**kwargs)
        self.fake.post=blocked
        try:
            jobs=[self.submit(key=str(i))[1]['id'] for i in range(12)]
            self.assertEqual(len(set(jobs)),12)
            self.assertEqual(self.submit(key='overflow')[0],429)
        finally:
            release.set()
    def test_auth_validation_and_persisted_state_contains_no_token(self):
        self.fake.get=lambda *a,**k: Response(401,{})
        self.assertEqual(self.submit(key='bad')[0],401)
        self.fake.get=lambda *a,**k: Response(200,{})
        _,data=self.submit(key='persist'); self.service.pool.shutdown(wait=True)
        with open(self.service.db,'rb') as f: raw=f.read()
        self.assertNotIn(b'secret-token',raw)

if __name__=='__main__': unittest.main()

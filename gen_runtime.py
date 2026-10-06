"""Invocation-scoped generation budgets, durable delivery artifacts and heartbeat."""
import contextvars
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from gen_jobs import ArtifactStore, GenerationJobs, GlobalRateGate, JobStore

@dataclass
class Invocation:
 job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
 started: float = field(default_factory=time.monotonic)
 max_seconds: float = 900
 max_generation_calls: int = 4
 generation_calls: int = 0
 qa: dict | None = None
 result: object | None = None
 qa_route: tuple | None = None
 qa_calls: int = 0
 max_qa_calls: int = 12
 timings: dict = field(default_factory=dict)
 parent: object | None = None
 def consume_qa(self):
  if self.remaining_seconds() <= 0: raise TimeoutError('QA deadline reached')
  if self.qa_calls >= self.max_qa_calls: raise RuntimeError('QA budget reached')
  if self.parent is not None: self.parent.consume_qa()
  self.qa_calls += 1
 def consume(self, count=1):
  if count < 0: raise ValueError('count must be nonnegative')
  if time.monotonic()-self.started >= self.max_seconds: raise TimeoutError('generation deadline reached')
  if self.generation_calls + count > self.max_generation_calls: raise RuntimeError('generation call budget reached')
  if self.parent is not None: self.parent.consume(count)
  self.generation_calls += count
 def remaining_seconds(self):
  return max(0.0, self.max_seconds - (time.monotonic()-self.started))

CURRENT = contextvars.ContextVar('gen_invocation', default=None)
ARTIFACTS = ArtifactStore('gen_artifacts', ttl_seconds=86400,max_bytes=200_000_000)
JOBS = GenerationJobs(store=JobStore(Path('gen_jobs.json')), gate=GlobalRateGate(max_concurrent=2), max_queue=100)

async def run_event(chat_id, work, *args, **kwargs):
 """Run handler body as a fair, bounded durable job (lazy startup)."""
 return await JOBS.run_event(chat_id, work, *args, artifact_store=ARTIFACTS, **kwargs)

def status_text(qa):
 if qa is None: return 'Проверка недоступна'
 findings=qa.get('findings',[])
 if findings:
  serious = any(f.get('severity') in ('high','critical') for f in findings)
  return ('Серьёзные дефекты: ' if serious else 'Остались замечания: ') + str(len(findings))
 return 'QA не обнаружил существенных дефектов'

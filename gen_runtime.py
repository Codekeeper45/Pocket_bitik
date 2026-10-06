"""Invocation-scoped generation budgets, durable delivery artifacts and heartbeat."""
import contextvars
import time
import uuid
from dataclasses import dataclass, field
from gen_jobs import ArtifactStore

@dataclass
class Invocation:
 job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
 started: float = field(default_factory=time.monotonic)
 max_seconds: float = 900
 max_generation_calls: int = 4
 generation_calls: int = 0
 qa: dict | None = None
 timings: dict = field(default_factory=dict)
 def consume(self):
  if time.monotonic()-self.started >= self.max_seconds: raise TimeoutError('generation deadline reached')
  if self.generation_calls >= self.max_generation_calls: raise RuntimeError('generation call budget reached')
  self.generation_calls += 1

CURRENT = contextvars.ContextVar('gen_invocation', default=None)
ARTIFACTS = ArtifactStore('gen_artifacts', ttl_seconds=86400,max_bytes=200_000_000)

def status_text(qa):
 if qa is None: return 'Проверка недоступна'
 findings=qa.get('findings',[])
 if findings: return f'Остались замечания: {len(findings)}'
 return 'Проверено'

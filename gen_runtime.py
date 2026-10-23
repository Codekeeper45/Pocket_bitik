"""Invocation-scoped generation budgets, durable delivery artifacts and heartbeat."""
import contextvars
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from gen_contracts import QAResult
from gen_jobs import ArtifactStore, GenerationJobs, GlobalRateGate, JobStore

@dataclass
class Invocation:
 job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
 started: float = field(default_factory=time.monotonic)
 max_seconds: float = 900
 max_generation_calls: int = 4
 generation_calls: int = 0
 qa: dict | QAResult | None = None
 qa_baseline: dict | QAResult | None = None
 qa_references: list[dict] = field(default_factory=list, repr=False)
 result: object | None = None
 error: str | None = None
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
  if type(count) is not int or count < 0: raise ValueError('count must be a nonnegative integer')
  if self.remaining_seconds() <= 0: raise TimeoutError('generation deadline reached')
  if self.generation_calls + count > self.max_generation_calls: raise RuntimeError('generation call budget reached')
  if self.parent is not None: self.parent.consume(count)
  self.generation_calls += count
 def remaining_seconds(self):
  remaining = max(0.0, self.max_seconds - (time.monotonic()-self.started))
  return min(remaining, self.parent.remaining_seconds()) if self.parent is not None else remaining
 def remaining_calls(self, kind='generation'):
  if kind not in ('generation', 'qa'): raise ValueError('unknown call kind')
  remaining = max(0, getattr(self, 'max_' + kind + '_calls') - getattr(self, kind + '_calls'))
  return min(remaining, self.parent.remaining_calls(kind)) if self.parent is not None else remaining

CURRENT = contextvars.ContextVar('gen_invocation', default=None)
ARTIFACTS = ArtifactStore('gen_artifacts', ttl_seconds=86400,max_bytes=200_000_000)
JOBS = GenerationJobs(store=JobStore(Path('gen_jobs.json')), gate=GlobalRateGate(max_concurrent=2), max_queue=100)

async def run_event(chat_id, work, *args, **kwargs):
 """Run handler body as a fair, bounded durable job (lazy startup)."""
 return await JOBS.run_event(chat_id, work, *args, artifact_store=ARTIFACTS, **kwargs)

def status_text(qa, *, image_bytes=None):
 from gen_contracts import QAResult
 from gen_qa import parse_findings, qa_for_image
 if isinstance(qa, dict):
  digest = qa.get('image_digest')
  canonical = {key: value for key, value in qa.items() if key not in ('image_digest', 'error', 'overflow')}
  if isinstance(canonical.get('findings'), (list, tuple)):
   canonical['findings'] = [{key: value for key, value in finding.items() if key != 'issue'}
                            if isinstance(finding, dict) else finding for finding in canonical['findings']]
  parsed = parse_findings(canonical)
  qa = QAResult(parsed.status, parsed.findings, parsed.overflow or bool(qa.get('overflow')), qa.get('error') or parsed.error, digest)
 if not isinstance(qa, QAResult): return 'Проверка недоступна'
 if image_bytes is not None: qa = qa_for_image(qa, image_bytes)
 if qa.status == 'unavailable' or qa.error: return 'Проверка недоступна'
 findings=qa.findings
 if findings:
  serious = any(f.get('severity') in ('high','critical') for f in findings)
  return ('Серьёзные дефекты: ' if serious else 'Остались замечания: ') + str(len(findings))
 if qa.status != 'pass' or qa.overflow: return 'Проверка недоступна'
 return 'QA не обнаружил существенных дефектов'

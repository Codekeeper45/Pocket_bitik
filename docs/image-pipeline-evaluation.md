# Image pipeline evaluation (offline by default)

`tests/fixtures/gen_eval/manifest.json` is a prompt/criteria catalogue only. It has no personal references, image fixtures, credentials, or purported provider outputs.

## Deterministic run

From the repository root:

```sh
/home/hermes/integrations/telegram-user/venv/bin/python qa_artifacts/evaluate_pipeline.py
```

The runner validates manifest structure and emits `defined_not_run` per case with `live_calls: 0`, `images_generated: 0`. It does not measure image quality and makes no provider claims.

## Optional bounded callback

Import `load_manifest` and `run_bounded_live` from `qa_artifacts.evaluate_pipeline`. Supply an explicitly selected async/sync callback that accepts one case dictionary. The helper allows at most 3 cases and at most 180 seconds per awaitable; it has no provider configuration and does not make network calls itself. The callback owner is responsible for permission, provider route, costs, artifacts, and actual-output verification. Never interpret callback success as evidence of an image's quality. Retain actual output images separately for human review; do not put private images in the manifest or source control.

## Review protocol

For stochastic scenes use at least three runs, compare the same requirement checklist and record actual provider/model, dimensions, stage durations, and calls from real run metadata. Review fidelity, anatomy, identity, seams, exact text, latency, and calls separately. Provide candidate images to a human reviewer; an LLM score is not acceptance. Compare multi-candidate generation against generation plus repair only in a separately approved, bounded experiment. Keep best-of routing, OCR/text rendering, multi-scale QA, mask edits, upscaling, and model routing disabled absent measured results and owner approval.

## Runtime job API (integration reference)

`GenerationJobs(store=JobStore(path), gate=GlobalRateGate(max_concurrent=N, min_interval=seconds), max_queue=..., default_deadline=...)`; call `await startup()` once at process boot. Submit provider work via `await submit(chat_id, callable, *args, artifact_store=ArtifactStore(...), context_budget=ContextBudget(...), deadline=seconds)`. It returns a `JobRecord` immediately; observe `record.status` or drain at shutdown with `await drain(timeout)`, which closes intake first. `await resume_intake()` reopens it. Delivery is deliberately separate: `await retry_delivery(job_id, deliver_callback, attempts=...)` reads the retained artifact path and never regenerates. Provider callbacks should be synchronous when wrapping a blocking SDK; `GlobalRateGate` tracks a worker thread through its actual completion even after cancellation/timeout. Use one shared gate instance for all generation/repair/series calls in this process.

The coordinator is process-local (not distributed); JSON snapshots are atomically replaced and stale queued/running/delivering jobs become `interrupted` on boot and require explicit recovery. Do not automatically rerun paid work. No Telegram adapter is wired by this module.

## Runtime limits / caveats

Choose queue/call/context limits and artifact TTL/quota from real usage, not this deterministic harness. Artifacts may contain user data: configure a project-isolated private directory, enforce TTL/quota, and restrict permissions/backups. The current `drain` waits for running tasks; if its timeout expires, keep the process alive or apply an explicit shutdown policy. A Python thread cannot be forcibly stopped; the semaphore remains held until the callback returns.

This evaluation task intentionally has not called Telegram or any image-generation API.

## Measured results

No live generation, delivery, benchmark, or human visual assessment was run for this implementation. Offline manifest validation and local unit tests only.

## Delivery/retention guarantees

A successful byte result can be written atomically by `ArtifactStore`; delivery retries reference that saved file. Pruning applies age first and then oldest-first total quota. The caller must ensure artifact paths survive long enough for retry and remove expired job metadata separately according to its retention policy.

## Limitations

Concurrency coordination is in-memory per process; multi-process deployments need a shared queue/lock backend before claiming global limits across processes. The interval rate gate is global only within the shared object/process. Fairness is FIFO per chat, while concurrently submitted chats contend for available slots normally.

No execution above enables or changes the disabled layered-generation path.

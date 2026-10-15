# Image generation transport

## Routing
`gpt-image-2` is the default. Configured GPT image models stay on the authenticated local CliProxy pool rather than silently using OpenRouter when a saved input-capability bit is stale.

Bot image requests use `GEN_JOB_BASE_URL` or hot-loaded `image_job_transport.json` (`base_url`, HTTPS). The bridge at localhost:8318 submits to localhost:8317/v1. Existing chat model tunnel is unchanged.

POST `/v1/image-jobs` receives endpoint `/images/edits` or `/images/generations`, body, bearer authorization and an idempotency key. It immediately returns 202 and a job ID. GET polls return 202 while running, then the upstream JSON and status. Repeated submit attempts reuse the same key. Poll failures never re-submit generation. Local upstream deadline 600 sec, client deadline 660 sec, no unknown-outcome automatic outer retry.

## Deployment
User units: `pocket-image-jobs.service`, `pocket-image-jobs-tunnel.service`, `pocket-image-jobs-sync.timer`. Tunnel URL rotation is copied to Serverix `/image_job_transport.json` by `sync_image_job_tunnel.py`; the bot hot-loads it on each image request. Quick tunnels still depend on Cloudflare availability; no claim of guaranteed uptime is made.

Private job state is in `~/.local/state/pocket-image-jobs/jobs.sqlite3` (600, parent 700), bearer keys stay in memory. Jobs are owner-scoped, payload collisions reject with 409, max four active workers and eight queued tasks, cleanup on new submissions after one hour. Interrupted jobs are marked failed at startup, never replayed. Only inline image references accepted to avoid arbitrary upstream URL fetches.

## Regression checks
`python3 -m pytest tests -q`

Tests include reference object schema, bounded retries, transport idempotency, polling outages, authorization rejection, exact upstream status/body preservation, queue capacity, private state and interrupted-job handling. Full test result and live probe evidence are private under `qa_artifacts/`.

## Limitations
Model API availability, provider quotas and external networks can still fail. The bot reports typed errors rather than promising every request succeeds. Image content/count/names require visual verification; successful transport alone is not content QA.

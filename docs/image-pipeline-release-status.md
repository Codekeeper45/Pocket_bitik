# Image pipeline release verification

Local regression: 135 unittest tests passed with credential-free bootstrap via `/home/hermes/integrations/telegram-user/venv/bin/python -m unittest discover -s tests -q`. Static verification: `python3 qa_artifacts/verify_release_local.py`; `git diff --check` passed.

Integrated: mandatory scene contract with one bounded retry, invocation call/deadline limits, bounded persistent queue, submitted/catalog reference provenance, exact-target paired QA, crop geometry and conservative rollback, per-job series checkpoints and owner-scoped genresume, durable ambiguous delivery records and genretry/genfile. Layers remain disabled in the production command path. Optional sampling/OCR/mask/upscale evaluation helpers do not enable experiments in production.

Real API evidence and limitations are in pipeline-v2-live-check.md. Independent vision detected remaining hand/perspective defects. Actual crop QA timeout caused rejection and preservation of original pixels, not unverified promotion. A defect-free image or an accepted visually clean crop repair has not been demonstrated. No measured multi-scene experiment or production Telegram end-to-end activation is claimed.

No Serverix activation or restart for this release. Owner explicitly prohibited restart; idle status does not override that prohibition. Release is saved in git, not active production behavior. Full plan acceptance remains incomplete until real visual benchmarks and permitted production E2E are completed.

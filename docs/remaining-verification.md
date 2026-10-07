# Remaining verification evidence

## Release
- eb551d8: explicit low reasoning for local Gemini QA, timeout 90 seconds.
- e3ae5cb: stop embedding retries on HTTP 401/402/403.
- Full suite: 141 tests pass.
- Production Serverix 74e81a81 revision and startup readback verified.

## Embeddings
Production credential probe: HTTP 402, zero vectors. Billing remains external blocker.
Lexical Telegram search exercised live with semantic cooldown active: 5 matching Socrate Store messages and linked reply parent returned. No replacement vector model introduced into existing embedding space.

## Historical scenes
Exact existing generated prompts recovered from Telegram, not original deleted commands:
- Socrate 11089: 115 crowd, four named foreground figures, Disney-Pixar digital metropolis.
- Toster 593435: ten named astronauts fixing toaster reactor.
Other historical Toster prompts 593437, 593445, 593450 also recovered.
Recovering a generated prompt is not equivalent to recovering every original reference/command.
Artifacts and per-scene verdicts: qa_artifacts/originals_live_ledger.json. Both replay runs completed with real image outputs. Socrate: three requested tags read correctly, exact 115 people not verified; background faces and hands retain defects. Toster: ten named subjects visible, rod law badge reads rodlam and Vera badge partly obscured; hands retain artifacts. These are failed full-fidelity acceptance, not clean successes.

## Repair
Live source pipeline_v2_live.png: one explicit hand edit generated but failed verification (QA timeout), original kept. No successful live repair asserted.
Repeated probes remain bounded by max_generation_calls=4, max_qa_calls=12, max_seconds=900. Lack of paired evidence means reject candidate, not discard source image.

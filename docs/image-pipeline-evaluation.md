# Offline image-pipeline evaluation

This harness evaluates **recorded evidence only**. It does not invoke image generation, a vision model, OCR, or network services. Synthetic fixture values are executable contract examples, not model-quality findings.

## Run

From the repository root:

```sh
/home/hermes/integrations/telegram-user/venv/bin/python -m unittest tests.test_gen_evaluation -v
/home/hermes/integrations/telegram-user/venv/bin/python gen_evaluation.py tests/fixtures/gen_eval/manifest.json
```

The manifest must use `schema_version: 1`. Every run records a unique `id`, `variant`, non-empty quality dimensions scored by a declared evaluator (0..1), measured `cost_usd`, measured per-stage `timing_ms`, integer per-stage `calls`, and artifact references. Missing measurements are rejected, not imputed. Keep repeated stochastic runs and inspect individual artifacts; means are descriptive only and not a significance test. Human visual review remains necessary for fidelity, identity, anatomy, composition, text, and repair seams.

## Experimental controls

- Candidate sampling defaults to one. A count greater than one requires explicit `enabled=True`; no production integration is performed by this module.
- OCR is dependency-injected (`inspect_required_text`); absent adapter is explicitly unavailable. Adapter output is not a universal OCR benchmark and should be retained with its implementation/version in experiment records.
- Upscale reports source/output dimensions and method separately. It explicitly makes no detail-improvement claim; assess faces/text/details by blind visual comparison.
- Mask experiments are rejected unless the selected `ProviderProfile.supports_masks` is true. Capability declaration alone does not imply mask quality.
- Record prompt/schema/provider/model revision and actual provider costs/timing/artifact locations alongside each experiment (extend the run manifest as needed); never store credentials or private reference images in fixtures.

Fixture runs use `synthetic://` artifact identifiers and plausible dummy measurements solely to exercise validation and aggregation. They must not be cited as measured production results. No user images or external generation are part of the harness.

## Reading comparisons

Compare variants on the same scene set and multiple stochastic seeds. Report per-dimension quality alongside cost, latency, call count, and sample count. A higher mean quality with higher spend/latency is a tradeoff, not an automatic win. Never select or enable routing, best-of-N, repair, OCR rendering, mask editing, or upscaling from a single favorable sample. Preserve raw evaluator notes and show candidates to a human reviewer.

CLI rejects malformed/incomplete manifests with a non-zero error; the Python API raises `EvaluationError`.

### Current harness scope

Implemented: strict measured-evidence validation, per-variant descriptive aggregation, opt-in candidate-count policy helper, injectable optional OCR adapter, explicit upscale measurement labels, and profile-gated mask option. Not implemented: actual generation, visual scoring, OCR engine, model routing, upscaler, or provider mask request. No external generation was run for this task.

### Synthetic fixture summary

The fixture contains 2 `single` runs and 1 `two-candidates` run, with intentionally illustrative quality/cost/timing/call fields. These are schema-test data only, not empirical findings.

## Future evidence report template

For a real approved experiment, add a separate report containing date/runtime, dataset version and permission status, model/provider identity, scene-level blind assessments, n and seed, artifact references, measured quality dimensions, call count, latency and cost, OCR exact-match where applicable, and explicit limitations. Do not overwrite synthetic fixture numbers with claims unless they are actually measured.

**Decision:** no experimental behavior enabled by this offline harness. Further production integration requires separate review and evidence.

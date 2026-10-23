# Generation quality implementation — October 23, 2026

## Scope and baseline

Implemented approved steps 1–9 of `2026-10-23_215952-generation-quality-flexible-agent.md` in this isolated worktree. Read `CONSTITUTION.md`, the October 23 generation review, and the parent's runtime/integration handoffs. No production inspection, secrets inspection, live API requests, deployment, push, or writes to the original checkout/project memory were performed. Parent-owned handoff Markdown files are excluded from the commit.

Baseline HEAD: `caf6fa5d4326ed8823d50e60b5cdf9734f3dbf32`. Initial tracked working tree was clean. Baseline full suite: **205 passed, 21 subtests passed, 2 Pillow deprecation warnings, 38.70 seconds**.

Baseline file SHA-256 values:

| Component | SHA-256 |
| --- | --- |
| `bot_new.py` | `36b24605ecfb079456e559f3c259078482d9b3c52491b7738763d48909492314` |
| `gen_prompt.py` | `67d689db867806508c21d62b22cef7dbc570adc6c5260a93e02a3fa2a0bbcce5` |
| `gen_references.py` | `8f901e62b4441b45e2f2ae5b1112030e7fc45876f6c1cf8942dfe6d439ba2fa0` |
| `gen_qa.py` | `a14625a2cd9fa8eb38f3fa349569bfe7c9f5f4d27fdfb8a79f7b44f90eaa3526` |
| `gen_runtime.py` | `8b2e466532e6bb03342f0913b6946b0884dac93fe61295371eb35b6b90eec3c6` |
| `gen_delivery.py` | `c84ab4f118d1400194cbcb20a0b24af8093cb8dd068ee8bc2e54880a52170b08` |
| `gen_jobs.py` | `56deb9d7ea360c600f6c3363cbf7ab7095d7118c803fa81ae29553fd048edba2` |
| `gen_series.py` | `c6cbb764b97eb085b82c4e5bf2ecdc752c985ba5b1c454a35369a87007d34ae7` |
| `gen_web_references.py` | `97290a76247001b3915b84087788f8bbcb44aa65bbcdf2adbda9e546e2a5ff39` |

Only source-defined routing was inspected: configured generation/media routes remain authoritative, layers remain off by default, gateway provider profile remains 10 images/4.5 MB, and generation input preparation retains a conservative 3 MB payload budget. Production model selection, queue, tunnel, and delivery state were deliberately not queried; parent owns those checks.

## Changed runtime components

| Component | Implemented changes |
| --- | --- |
| `bot_new.py` | One structured agent plan drives single/series routing; file count comes from the model or explicit flags, never natural-language regex. Removed obligatory post-prompt contract extraction. One schema repair then verbatim-source fallback. Registered IDs, actual attachment manifests, role merging, capacity reselection/fallback, bounded image context and tool deltas. Tool errors stay in the conversation. Selected subject/style inputs reach QA; QA binds exact image digest; final unavailable QA never reuses an old assessment. Targeted regeneration receives evidence and invariants; category-based seam checks replace language substrings. Model-selected optional layers preserve complete composites or fall back to ordinary rendering. Remaining deadlines cover provider waiting. Batch tasks cancel/join on interruption. Shared series and delivery locks, durable delivery outcomes, Saved Messages normalization, stable resume delivery keys. |
| `gen_prompt.py` | `IntentPlan`, compact schema instructions, structural validation, explicit overrides, source-preserving fallback, multi-role refs and conditional edit/text/group/panel/series renderer instructions. Legacy scene contracts remain readable. |
| `gen_references.py` | Namespaced stable IDs plus legacy aliases, merged provenance/roles for identical bytes, bounded protected working sets, actual-payload packing with compression and explicit missing/capacity diagnostics. Attachment indices assigned after packing; source IDs survive normalization. |
| `gen_web_references.py` | Separate image/page URLs, author/title and unverified provenance, exact sanitized-query authorization, bounded source-page reader using pinned SSRF-safe transport, page tool and multi-role selection schemas. |
| `gen_contracts.py` | Optional QA image digest, preserving existing positional fields. |
| `gen_qa.py` | Status validation before findings, honest unavailable results, digest binding, selected-target comparison and rejection of new/worsened substantial defects. |
| `gen_policy.py` | Evidence/category/severity/confidence-based targets, bounded repairs, explicit repair instructions and original constraints. |
| `gen_pair_qa.py` | Selected-target evidence and substantial-defect preservation, bounded remaining timeout and prerequisite validation. |
| `gen_runtime.py` | Parent-aware remaining budgets, transient selected QA inputs, separate baseline QA, honest status rendering and digest checks. |
| `gen_jobs.py` | Deadline includes persistence/queue/rate wait; generation budget consumed immediately before actual execution. Timed-out blocking workers retain capacity. Truthful incomplete/partial/delivered states; uncertain retries require reconciliation; successful sends are not repeated after persistence failure. |
| `gen_delivery.py` | Exact marker/scope/readback reconciliation, authenticated Saved Messages compatibility, shared delivery locks and durable confirmation preservation. |
| `gen_series.py` | Existing executor/checkpoints adapted to unified intent; shared per-original-job lock; persisted `not_started`/`sending`/`delivered` phases, safe crash recovery and explicit reconciliation for legacy unknown sends. Versioned explicit raw repeats preserve original bytes and checkpoint compatibility. |

No framework, service, parallel generation architecture, credential changes, or constitution changes were introduced. Existing access checks, technical URL/ID validation, explicit flags, cancellation and old checkpoint reading remain. Unavailable explicitly linked Telegram sources retain the existing access/source safety stop; they are not substituted with unauthorized sources.

## Regression tests

New files:

- `tests/test_gen_review_regressions.py`: initially reproduced three failures (negated count, lost second role, changed verbatim image number); gateway capacity and non-stacked retry regressions.
- `tests/test_gen_intent_plan.py`: structural model count, panels/files, negation mock plans, flags, malformed schema, raw/edit/source preservation, multi-role refs and series adaptation.
- `tests/test_gen_reference_registry.py`: namespaced IDs, duplicate-byte provenance/roles, aliases, optional missing inputs and structured manifests.
- `tests/test_gen_reference_capacity.py`: protected working set, overflow, compression, invalid bytes, real payload limits and explicit fallback diagnostics.
- `tests/test_gen_qa_status_integrity.py`: malformed/contradictory QA status, image hashes, inherited selected reference inputs, repaired bytes plus unavailable final QA.
- `tests/test_gen_targeted_quality.py`: evidence-driven targets, language-independent categories, targeted instructions and rejection of new substantial defects.
- `tests/test_gen_tool_context.py`: real builder call counts and tool ordering, one schema repair, tool-error recovery, nonrepeated image parts, cap+4 inspected candidates, actual command routing for independent files/panels/negation/series/flags.
- `tests/test_gen_web_provenance.py`: image/article separation, unverified official captions, private quotations without handles, sanitizer denial/failure, only sanitized queries reaching mocked web transport, source pages, SSRF/redirect/rebinding/MIME/size checks.
- `tests/test_gen_delivery_first_integration.py`: actual activity wrapper → command → provider wrappers → safe fake provider → real delivery wrapper; malformed planner, missing refs, unavailable QA, cleanup/send failures, partial/resumed series, batch cancellation, concurrent resume, queue deadlines, worker capacity, exact reconciliation and Saved Messages scopes.
- `tests/test_gen_series_lifecycle.py`: crash windows, durable delivery phases, legacy unknown sends, lock concurrency/cancellation, raw checkpoint identity and explicit reconciliation.

Updated existing files: `tests/test_gen_actual_reference_registry.py`, `tests/test_gen_contract_builder_hook.py`, `tests/test_gen_contract_fail_closed.py`, `tests/test_gen_delivery.py`, `tests/test_gen_delivery_first.py`, `tests/test_gen_helpers.py`, `tests/test_gen_layers_integration.py`, `tests/test_gen_panel_count.py`, `tests/test_gen_reference_caption_disambiguation.py`, `tests/test_gen_regions_integration.py`, `tests/test_gen_retry_accounting.py`, `tests/test_gen_series_handler.py`, `tests/test_gen_unified.py`, `tests/test_gen_web_references.py`.

Changed assertions intentionally replace obsolete requirements (semantic regex count, second extractor call, all instruction blocks on every request, missing optional refs raising, blind delivery retry). The builder-only caption test is renamed to describe prompt preparation rather than claim delivered generation. Existing transport/security tests remain exercised.

## Verification and measured context

- Final full-suite result (`python -m pytest tests -q`): **411 passed, 57 subtests passed, 19 warnings, 34.68 seconds**. Warnings are the existing Pillow `getdata` deprecation exercised by more integration cases.
- Focused lifecycle verification: **83 passed**, covering actual handler execution and durable recovery.
- `python -m py_compile bot_new.py gen_*.py` and `git diff --check` pass.
- Valid structured plans require one model completion instead of the old prompt completion plus mandatory extraction. A schema error permits one correction; capacity reselection and public-query sanitization are optional extra calls, recorded separately.
- Context tests measure image parts in actual mocked request messages. Existing images are not appended again after tools; retained visual history is bounded and the latest inspected source remains selectable while its source record is retained. `planner_calls`, `planner_image_parts`, `planner_image_transmissions`, and `web_privacy_calls` are available per invocation. Repeated API requests still retransmit retained history; this is not represented as zero image cost.
- Public-query sanitization uses the already authorized model, only when web search is requested, with an invocation-local cache. Structural approval prevents bypassing that boundary; semantic privacy is still a model decision, not a mathematical guarantee or a keyword whitelist.

Exact offline measurements against `caf6fa5`, using AST-extracted baseline functions and mocked model/tool responses (Unicode characters, not tokens):

| Measurement | Baseline | Implemented |
| --- | ---: | ---: |
| Creation builder system message including web instructions | 21,216 characters | 2,706 characters |
| Successful no-tools planner completions | 2 | 1 |
| Two submitted images plus one inspected image: image parts appended in successive user turns | `[2, 3]` | `[2, 1]` |
| Final-context image parts / unique images | 5 / 3 | 3 / 3 |
| Image transmissions across the two planner requests | 7 | 5 |

The final-context image-part reduction is 40%; transmissions decrease 28.57% in that synthetic scenario. These measurements do not establish token billing, production latency, or visual-quality improvements.

## Remaining acceptance and limitations

Steps 10–11 are not performed here: parent independent review, live visual/model-language acceptance, production route/queue checks, deployment/rollback/readback, and project-memory updates. Passing offline mocks does not establish real negation understanding, canonical identity, artistic quality, text legibility, provider availability, latency savings, or live Telegram correctness. No result is claimed for a live test in this worktree.

Locks coordinate one bot process/event loop; operating a second process against the same account/state still requires external operational coordination. A cancelled/timed-out blocking provider worker may finish remotely; capacity is retained until it exits, but the late output is not promised as delivered. Ambiguous unmarked Telegram sends remain manual/readback-only rather than risking duplication. Missing optional references and unavailable QA preserve useful generated output; genuine access failures, external refusals, exhausted budgets and cancellation remain explicit.

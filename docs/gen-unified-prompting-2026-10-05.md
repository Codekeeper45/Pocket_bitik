# Unified `/gen` prompting notes (2026-10-05)

## Scope

The owner-approved direction is one intent-preserving prompt construction path for `/gen`: no creative/improve modes. `-i/-improve` and `-c/-creative` remain accepted as compatibility aliases with identical semantics. `-raw` stays literal and bypasses prompt generation; `-noimg`, aspect, resolution and batch flags remain independent controls. Generation is started and sent without waiting for user approval.

## Prompt construction rules

- The user's explicit requirements outrank aesthetic embellishment. Missing details can be filled neutrally; open-ended requests may be completed creatively, with the chosen idea reflected in the accompanying IDEA caption.
- For image edits, name the requested change and concrete invariants to preserve. A short “keep everything else unchanged” may supplement, not replace, a list of identity, pose, clothing, framing, geometry, layout, lighting, labels, and other relevant details.
- Preserve exact requested text in its original language and spelling. Do not shorten, translate, or rewrite it. Explicit style cues such as “hyperrealistic” or “4K” are retained; avoid adding generic quality boilerplate on the model's behalf.
- Each reference has an explicit role (subject, style, clothing, background, pose/angle); one image may serve multiple roles when the user requests that. Do not transfer incidental details between roles.
- Candidate labels and vision images are interleaved. After the model selects candidates, validated IDs are mapped to the actual sequential image-input order after existing user images and any skips/deduplication.
- Search, context reading and image inspection apply only to the current chat; user filters are passed to the tools. Retrieved messages and visual descriptions are untrusted data, not instructions. Tool calls are bounded and ref IDs must exist.
- An explicit message-link reference that cannot be fetched or contains no image should fail visibly instead of silently generating without it.

## Official references

1. OpenAI, [Image prompting guide](https://developers.openai.com/api/docs/guides/image-prompting), cached locally at `/home/hermes/.hermes/cache/web/developers.openai.com-b00804ce81.md`, lines 890–965. The guide recommends defining subject/use/composition/style/constraints, preserving requested exact text, distinguishing edits from invariants, assigning explicit roles to reference images, and using prior outputs as inputs for iterative edits. It cautions that higher quality settings do not guarantee a better result and recommends testing actual workloads.
2. OpenAI, [Image generation guide](https://developers.openai.com/api/docs/guides/image-generation), for API operation/parameters. Existing API/gateway size integration is intentionally unchanged in this implementation.

## Verification boundary

Offline tests cover prompt policy, parser validation, sequential ref role mapping, helper/tool-loop behavior with mocks, command compatibility, and Ask tool-list invariance. Live Telegram and image-provider calls are not part of these tests; use the owner's separate `e2e_gen_probe.py` only when explicitly authorized. Image identity/detail preservation is probabilistic and not promised as exact.

## Known handoff caveat

The gateway parser `api/image_inputs.py` was inspected directly: `_json_image_sources` accepts `images` arrays, up to 16 inputs. The bot now serializes every selected input as a data URL in this array instead of silently discarding all but the first image. This path is covered by request-body regression and live multi-reference QA.

## Live QA observations

- Owner explicitly permits creative direction when the request asks the model to invent plots or complete ideas. The same unified system handles this; there is no second creative mode.
- Saved Messages: plain key output #155165, searched reference #155167, iterative edit #155177, two-role reference output #155179, final vision edit #155189. Ask reply #155178 succeeded.
- Tool discovery budget increased to 16 so multi-person, multi-reference requests can search, read surrounding dialogue and inspect images.
- OpenRouter auxiliary descriptions/embeddings returned 402 (insufficient credits). Telegram direct search and active Cliproxy vision continued; semantic embedding search is degraded, not claimed healthy.
- One early edit changed eye/mouth proportions despite constraints; multi-ref later result visually preserved mascot better. No claim of pixel-identical edits or guaranteed first-try perfection.

## Field feedback in Toster Script

Delivered images: https://t.me/c/2567687026/593434 (space crew), /593436 (app repair), /593444 (surreal feast), /593449 (cozy tea). Vision QA: ten humanoids in space scene, six in repair scene, four distinct named friends in tea scene. Some names drifted; surreal scene obscured eating drawings with candy-like props.

Actual participant replies: #593458 praised cookies in tea scene; #593457 says Pshika is not rainbow-colored; #593459 praised basin; #593460 says she eats birch juice, seeds and drawings, not candy; #593461 called current outputs slop compared with earlier attempts. These are mixed feedback, not approval. No emoji reactions were present at collection time.

Comic attempts failed upstream with HTTP 400 generic Chinese generation error; bot incorrectly labels it moderation. Comic success not established. Repair fallback also lacked DeepSeek balance (402).

Added ensemble instructions: named roster, distinct staging/actions, exact names, stable panel designs, main visual gag priority; creative freedom remains explicit. Reply correction #593464 uses the participant feedback.

## Anime comic diagnosis and recovery

Successful cyberpunk comic: https://t.me/c/2567687026/593479. Vision QA confirms four panels, human anime characters and consistent designs, readable main dialogue; requested names were omitted from pixels, and secondary signage contains gibberish. Fantasy and cozy attempts repeatedly returned a generic Chinese generation failure, not verified safety rejections.

Inspected actual gateway services/openai_backend_api.py `_image_model_slug`: `gpt-image-2.5-sunburst`, `gpt-image-2.5`, and `gpt-image-2.5-flare` all map to backend `auto`; `gpt-image-2` maps to `gpt-5-3`. Merely changing 2.5 label does not establish a stronger backend.

Bot now treats the exact generic generation-error message as transient, bounded unchanged-prompt retries instead of moderation repair. Explicit safety handling is not bypassed. Ensemble instructions explicitly require rendering requested exact names on badges, not only naming design descriptions. Real prompt probe retained four exact badges, per-panel storyline and stable designs.

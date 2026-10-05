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
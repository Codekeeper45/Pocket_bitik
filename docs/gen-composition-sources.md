# Gen: composition and reference accuracy

## Sources actually consulted

- Adobe, The basics of photography composition: https://www.adobe.com/creativecloud/photography/technique/composition.html
  Balance versus symmetry, negative space, leading lines, perspective, focus and depth, framing according to use. Rule of thirds is a guideline, not mandatory.
- Konart, Shading Your Drawings Like An Anime Movie, Clip Studio Tips: https://tips.clip-studio.com/en-us/articles/3054
  Character/background perspective and lighting must agree. Anime characters use clean linework and flat shadow shapes, while background treatment can be more painterly. This describes one anime production aesthetic, not every anime style.
- Animator Nicca, Simple Anime-style Coloring Techniques, Art Rocket: https://www.clipstudio.net/how-to-draw/archives/162911
  Cel shading uses simple shapes without mandatory blending; establish light source during sketch and match shadow shapes to it.

## Implementation

`_GEN_COMPOSITION_RULES` is included in `_gen_unified_system` for prompt planning. It guides the actual image prompt toward appropriate focal hierarchy, spacing, readable silhouettes, consistent perspective/light, style-specific rendering and explicit hand ownership. It preserves requested cast, style and edit invariants. Model selects appropriate principles, not an indiscriminate checklist.

`_GEN_ENSEMBLE_RULES` binds identity, reference evidence, visual attributes and position together. Sender is not necessarily image subject. No unsolicited badges or anonymous filler crowds.

These are inference-time instructions, not retraining and not a guarantee of zero anatomical/text artifacts. No ControlNet, layer compositing or automatic output QA has been claimed implemented.

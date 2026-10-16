# /gen art direction

The Gemini prompt path includes `gen_art_direction.ART_DIRECTION_RULES`, a compact craft adaptation of character-design, lighting-direction, and character-design-continuity. These complement the installed OpenAI imagegen prompting guidance.

User instructions override the heuristics. Open character prompts get readable silhouettes and restrained supporting details; explicit styles and complex scenes remain allowed. References lock identity only when used as character references. `-raw` continues to bypass prompt development.

Local skill sources: `/home/hermes/projects/image-skills-sources/`.
Upstreams: https://github.com/omer-metin/skills-for-antigravity and https://github.com/calesthio/generative-media-skills .

Verification performed: `python -m unittest tests.test_gen_unified -q` passed 32 tests. `python e2e_prompt_probe.py 'Нарисуй аниме девушку у окна с чашкой чая. Спокойное утро, без надписей.'` returned a real Gemini scene contract with one participant, anime style and no required text. This verifies prompting, not rendered image quality.

Deployment: Serverix 74e81a81, `/bot_new.py`, `/gen_art_direction.py`; read-back hashes matched local committed files. Empty active jobs checked before restart; subsequent bot.log confirmed Userbot started. Panel state can remain starting, so use runtime logs as additional evidence.

Generated-reference provenance is a separate improvement; do not conflate outgoing user uploads with generated outputs.

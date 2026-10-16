# Public visual references for /gen

`web_image_search` uses the existing Tavily image search credential. Images are downloaded, decoded, normalized, and included in the Gemini vision context. `choose_web_refs` assigns identity/style/background/template roles; selected references are remapped to actual image endpoint positions. Selected URLs are recorded in `gen_web_sources/sources.jsonl`. The result source line includes actual used web references, not every search hit.

Downloads pin the connection to a validated public DNS address, retain TLS hostname verification, validate each redirect, reject credentialed/local/private URLs, cap bytes and decoded pixels. Search results remain unverified, not canonical simply because they were returned. Official-source verification is not automatic.

`chat_get_avatar` retrieves only a participant of the current chat. Avatar contents must be inspected: an avatar can be a fictional drawing rather than a photo. Private handles are rejected in public search queries; the model is instructed to use public visual concepts only.

`gen_identity_guard.py` blocks plans that explicitly admit an absent source and propose a generic substitute. This is a conservative text-level guard, not proof of identity and not a complete semantic verification system.

## Verification

- `python -m unittest discover -s tests -p 'test_gen_*.py'` with test Telegram env: 157 tests passed.
- Real Tavily search returned downloaded, decoded Kirby reference images.
- Live /gen round two: Rustam used participant avatar plus vintage horror style web ref; Toster used manga screentone web ref. Logs confirmed search, selection, and actual reference input counts. Moral test used the participant's own published artwork plus a web paper-theatre style ref.
- Visual review: corrected photo edit preserves pose, shirt, glasses and applies bad makeup. Horror variation retains visual cues but loses shirt check pattern and has ambiguous mirror reflection. Toster manga keeps one sheet and coherent characters; first panel is somewhat busy and toaster slot geometry is imperfect.

User requested per-update local tests, deployment verification, then another live test. Public generations are separately authorized; do not send arbitrary new tests without that authorization.

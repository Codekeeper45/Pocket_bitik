# Live /gen evaluation, 2026-10-16

## Verified workflow

Normal /gen, no -raw. Production logs show prompt_by_ai=True. Actual tool invocations include chat_search, chat_ant_search, chat_read_context, chat_inspect_image, chat_get_avatar, web_image_search and choose_web_refs. Web search sometimes returns no safe image; model reformulated linocut search and retrieved valid images. Public source message entities verified as MessageEntityTextUrl, not plain labels.

## Six stress cases, two per original chat

- https://t.me/nastolkiggg/61824 : architectural makeup comic. Stable roles and labels across three panels; awkward measuring-tool contact and fingers; object markings drift. Multimodal reviewer: 6.7/10.
- https://t.me/nastolkiggg/61821 : avatar-mascot makeup duel. Two characters maintained; nameplates and original brushes disappear between panels. Reviewer: plot/composition 3.5/5, technical 3/5. This is explicitly a mascot interpretation, not a real portrait.
- https://t.me/TosterScriptChat/613522 : model selected NickupAI, Melony, Unholy Father from chat. Three readable panels, good labels; fingers and toaster scale inconsistent. Reviewer: composition/plot/readability 7/10, anatomy 6/10. Actual appearance not independently established.
- https://t.me/TosterScriptChat/613636 : stained-glass mascot story. Four panels, two consistent characters; coherent color-to-monochrome payoff. Jar disappears; toaster enters without setup. Reviewer 7/10.
- https://t.me/mafia_ot_svety_Chat/231631 : paper theatre. Three avatars/mascots, readable tableau. Names omitted; mirror perspective and cat pen grip flawed. Reviewer: composition 7, plot/readability/anatomy 6/10.
- https://t.me/mafia_ot_svety_Chat/231669 : museum comic. Three main characters; blocknote disappears and picture-frame geometry drifts; Moral cheek specks weak. Reviewer 6/10.

These scores are subjective multimodal review, not automated proof of canon or identity.

## Additional post-related results

- https://t.me/TosterScriptChat/613581 : update comic; readable title, but first panel has two spheres, inconsistent hand ownership and weak final payoff.
- https://t.me/TosterScriptChat/613593 : book fair; strong legible Cyrillic and living-book joke. Exceeded requested character maximum, used three inks instead of two, weak book/toaster hands.

## New chat

- https://t.me/c/3931608844/3968 : model independently found Qwen/Fable pair in author sketch #3847 and discussion #3859. Three-panel gift story. Minor hand/bench drift.
- https://t.me/c/3931608844/3977 : Gemma/Mistral bicycle paper diorama from textual headcanon #786. Pedal assembly and foot support imperfect.
- https://t.me/c/3931608844/3980 : headcanon drawing diptych. Clear joke and labels; perspective imperfect; planned hanging action replaced by already-hung picture.

## Bugs exposed and corrected

1. Panel counts wrongly triggered separate-page planner, bypassing tool-enabled pipeline and causing ValueError. Panels now stay on one image; explicit pages/-p retain multipage behavior.
2. Numeric participant IDs needed int conversion.
3. Web style refs could acquire unwanted subject roles through freeform REFS; explicit selected web roles now authoritative.
4. Profile-avatar source attribution initially referenced undefined roles variable; fixed and regression assertion added. Follow-up production verification required after release.
5. All Telegram links were treated as mandatory image sources; textual headcanon link #786 aborted before generation. Text messages now preserved as untrusted context, not rejected for lacking an image. Live retry produced #3977 and #3980.
6. Avatar acquisition now uses freshly resolved profile and download_profile_photo, not history[0]. Current photo ID persisted in reference metadata. Real helper probe returned current_main and valid image bytes.

## Remaining priorities

- Bound cumulative search latency and show queue position/real current stage. Heartbeat alone is not progress. Some searches took many minutes.
- Keep character-to-reference mapping distinct: prevent a participant face leaking into unrelated giants/background figures.
- Maintain panel state: number of props, ownership, object location, nameplates, mirror geometry.
- Verify required texts/colors/counts before delivery without hiding the image if QA is unavailable.
- Avoid selecting visually attractive but unrelated images as identity. Avatar is account-associated, not proof of a person's actual face.
- Separate retrieved text quotations from user-required image text during contract extraction: source captions repeatedly triggered quoted-text checks.
- OpenRouter embedding endpoint returned HTTP 402; fallback chat search worked, vector recall remains degraded.
- Owner artistic direction: beauty, warmth, tenderness, coherent visual art; creativity is not random absurdity or clutter. Do not cycle same museum/toaster/makeup gags with only material changes.

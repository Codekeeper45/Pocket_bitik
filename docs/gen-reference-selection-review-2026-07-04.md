# Code Review: `/gen` Reference Selection

Дата: 2026-07-04  
Файл: `bot_new.py`  
Scope: сбор контекста `/gen`, каталог фото, выбор `REFS`, refs из `/index`, отправка refs в image API.

## Summary

Ревью нашло 8 production-рискованных проблем: 4 HIGH и 4 MEDIUM. Главный симптом совпадает с наблюдением: модель может описывать соседнее фото, а выбирать другой ref, потому что номера и картинки были переданы раздельно, часть нужных image-documents выпадала, а индекс-поиск подтягивал лорно релевантные, но визуально слабые медиа.

## Findings

### HIGH: Номера `#1/#2/#3` могут съезжать относительно картинок

**Issue:** в vision-режиме был один общий текстовый список refs, а картинки шли отдельным хвостом без подписи перед каждой image-part.  
**Risk:** мультимодальная модель может связать номер с соседней картинкой.  
**Fix:** catalog теперь передаётся interleaved: text `REF #N ...` сразу перед соответствующим `image_url`.

### HIGH: Генератор получает thumbnails, а не качественные refs

**Issue:** `_merge_catalog_refs()` отправлял `thumb` 768px в image API.  
**Risk:** хуже узнаваемость лиц, персонажей, артов и мелких деталей.  
**Fix:** выбранные refs лениво конвертируются в JPEG max side 1536px через `_gen_ref_img()`; `thumb` остаётся только fallback.

### HIGH: Можно превысить лимит 16 референсов

**Issue:** лимит считал только refs, добавленные из каталога, и не учитывал уже приложенные пользователем изображения.  
**Risk:** image API может отклонить запрос или проигнорировать часть refs.  
**Fix:** `_merge_catalog_refs()` проверяет общий `len(out) >= GEN_CTX_REF_MAX`.

### HIGH: Каталог и ссылки игнорируют image-documents

**Issue:** часть путей принимала только `msg.photo`, но не PNG/WebP/JPEG, отправленные как документ.  
**Risk:** нужная картинка выпадает, модель выбирает ближайшее фото рядом.  
**Fix:** `_gen_fetch_link_refs()`, album loop, extra refs, `_gen_history_catalog()` и `_gen_index_candidates()` используют единый image predicate.

### MEDIUM: `/gen` ищет refs в индексе по контекстному описанию, не по чистой визуальности

**Issue:** `media_text` может быть лорным описанием изображения, а не тем, что видно визуально.  
**Risk:** поиск находит смысловой соседний объект, но плохой visual reference.  
**Fix:** `/index` refs теперь проходят retrieval pool -> fresh `_GEN_DESC_PROMPT` visual description -> LLM rerank.

### MEDIUM: Text/description режим видел слишком много слабых no-desc кандидатов

**Issue:** при `GEN_CTX_IMG_MAX_DESC=300` в список попадали `[описание недоступно]`.  
**Risk:** большой шумный список повышает шанс случайного выбора.  
**Fix:** если есть валидные описания, no-desc кандидаты без caption удаляются из text-mode catalog.

### MEDIUM: Vision-режим не фильтровал скриншоты/мемы

**Issue:** junk-filter работал только в text-mode.  
**Risk:** визуально яркие скриншоты/мемы могли попасть в refs.  
**Fix:** vision-mode тоже получает `_GEN_DESC_PROMPT` classification и отсекает `скриншот`/`мем`.

### MEDIUM: Фото-кандидаты слабо привязаны к соседнему тексту

**Issue:** catalog item имел только caption, без msg/date/author/nearby context.  
**Risk:** смысл соседней реплики мог быть перенесён на соседнее фото.  
**Fix:** history catalog добавляет `msg_id`, `date`, `author`, `nearby_text`, и это попадает в label каждого `REF #N`.

## Verification Plan

- `python -m py_compile bot_new.py`
- `python -m pyflakes bot_new.py`
- Smoke: interleaved vision parts preserve `REF #N -> image #N`.
- Smoke: `_merge_catalog_refs()` respects total 16 refs and uses high-quality `ref`.
- Smoke: image-documents pass link/history/index predicates.
- Smoke: no-desc and junk candidates are filtered as expected.


# Аудит gateway — 05.10.2026

## Выполнено
- Изучен контракт `api/accounts.py`: `POST /api/accounts/sync` принимает `AccountOperationRequest`; для всего пула корректный payload — `{"selection":{"mode":"all"}}`. Вариант `{"selection":"all"}` получил 422, после сверки схемы повторено штатно.
- Синхронизация завершена: `done=true`, `total=23`, `processed=23`, `synced=23`, ошибок/пропусков/удалений — 0. Список событий подтвердил success для каждого из 23 ID.
- Свежий API-снимок показывает 23/23 включены. 16 имеют доступную квоту, 7 подтверждённо исчерпали image quota; сбросы квоты доступны по ID в API (epoch `quota_reset_at`). Для лимитированных записей состояние показано как «лимит», квота 0; остальные имеют остаток 1–24.
- Важно: список содержит 23 записи, но повторяющиеся стабильные `user_id` означают 8 базовых идентичностей. Email и токены в этом отчёте не сохраняются.

## 502 / account rotation
- Код `services/image_failure.py`: `ImageFailure.switch_account` возвращает `outcome == "failure"`; outcome безусловно `text` при любом HTTP 400. Поэтому `upstream_text_reply` (400, `request`, не retryable) никогда не переключает аккаунт — даже когда это обычный текстовый отказ модели с отсутствием изображений.
- Явные `content_policy_violation` и `invalid_image_input` тоже 400/request и должны остаться без ротации; их смешивать с общей ошибкой нельзя. `no_image_generated` — 502/request, также не retryable/без переключения по текущей таблице. Уточнение: запись о событии edits 502 длительностью 58 секунд и её `image_attempts` не удалось достать: в найденных маршрутах нет очевидного API чтения call records, а локальный импорт repository остановился из-за отсутствующей переменной auth-key в shell окружении (секрет намеренно не выводился). Поэтому конкретный cause у того инцидента не доказан.
- Никакие изменения runtime/исходников не применялись: для безопасной правки классификации нужны настоящая запись и точное отделение safety/invalid input от generic upstream text. Также не подтверждено, что 58 секунд вызваны лимитом gateway, а не upstream ожиданием.

## Ограничения и проверки
- Проверены контракт и рабочий API с admin-авторизацией; `/health` не использовался.
- Первый sync запрос получил 422 (неверная форма), корректный запрос после чтения Pydantic-контракта успешно отработал.
- Конечный прогресс endpoint вернул `done=true`, 23/23; статус-категории в прогрессе: 16 «нормально», 7 «лимит», 0 ошибок.
- Запросы generation и edits для runtime-проверки не отправлялись: не удалось извлечь достоверную свежую запись для исправления, а production-генерации расходуют квоту. Следовательно, надежность не объявляется исправленной/проверенной.

## Окружение и сохранность
- Сервер: `server@127.0.0.1:2222`, проект `/home/server/projects/chatgpt2api`; GitHub repo/ветка не подтверждены.
- Перед действиями проверен `git status`. Существующие изменения (`.gitignore`, `pyproject.toml`, `services/oauth_login_service.py`, `services/openai_oauth.py`, `uv.lock` и новые файлы) оставлены без изменений; точечных исходных правок и перезапусков не было, поэтому backup не требовался.
- Отчёт сохранён отдельно: `/home/hermes/projects/Pocket_bitik/gateway_audit/REPORT.md`.

## Следующий безопасный шаг
Получить через штатную admin UI/API точную запись call/task для edits с полями `endpoint`, `outcome`, безопасно очищенным error-code и `image_attempts` (ID аккаунтов оставить, email/секреты удалить). Только после установления класса ошибки — отдельный тест-кейс на fake accounts для различения `content_policy_violation`/`invalid_image_input` и generic text failure; затем минимальная правка rotation и тесты без реальных затрат квоты.

## Отчёт о проделанной работе
1. Serverix — не использовался.
2. GitHub — репозиторий и ветка не подтверждены; удалённый каталог — `/home/server/projects/chatgpt2api`.
3. Изменения — создан только этот отчёт: `/home/hermes/projects/Pocket_bitik/gateway_audit/REPORT.md`; код/настройки gateway не менялись.
4. Тесты — admin API `GET /api/accounts` (200, 23 records), `POST /api/accounts/sync` (200 после коррекции схемы), `GET /api/accounts/operations/{id}` (done=true, synced=23, errors=0). Live generation/edit не выполнены.

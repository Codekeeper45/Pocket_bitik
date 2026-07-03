from telethon import TelegramClient, events, utils
from telethon.errors.rpcerrorlist import MessageNotModifiedError, FloodWaitError
from telethon.extensions import html as tl_html
from telethon.helpers import add_surrogate
from telethon.tl.types import MessageEntityBlockquote, MessageEntityPre, MessageMediaWebPage, InputReplyToMessage, InputMessagesFilterPhotos
from telethon.tl.functions.messages import SendMessageRequest
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import urlsplit, unquote
import os
import re
import io
import asyncio
import contextvars
import base64
import random
import threading
import time
import traceback
import requests
import json
import glob
import logging
import hashlib

try:
    import pymysql  # /index (GraphRAG-память, MariaDB); без пакета команда просто недоступна
except ImportError:
    pymysql = None
try:
    import numpy as _np  # /index: numpy-cosine поиск по векторам-блобам
except ImportError:
    _np = None
try:
    import hnswlib as _hnswlib  # optional /index ANN backend; потоковый поиск остаётся fallback
except ImportError:
    _hnswlib = None
from types import SimpleNamespace
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone

# Логирование с ротацией: bot.log до 50МБ × 10 ротированных копий (≤500МБ суммарно) + stdout.
# ВАЖНО: запускать БЕЗ `> bot.log 2>&1` — конфликт с RotatingFileHandler.
_logger = logging.getLogger("bot")
_logger.setLevel(logging.INFO)
_logger.propagate = False
if not _logger.handlers:
    try:
        _fh = RotatingFileHandler("bot.log", maxBytes=50_000_000, backupCount=10, encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        _logger.addHandler(_fh)
    except Exception as _le:
        print(f"[BOOT] Не удалось открыть bot.log ({_le}) — пишу только в stdout")
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_sh)

# Точный подсчёт токенов (опционально). Если tiktoken недоступен или словарь
# не докачался — _ENC=None и count_tokens() откатывается к оценке по символам.
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")
except Exception as _tt_err:
    _ENC = None
    print(f"[BOOT] tiktoken недоступен ({_tt_err}) — подсчёт токенов по символам")

load_dotenv()

# Настройки
try:
    api_id = int(os.getenv("api_id") or "0")
except ValueError:
    api_id = 0
api_hash = os.getenv("api_hash") or ""
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
opencode_api_key = os.getenv("OPENCODE_API_KEY")
modelgate_api_key = os.getenv("MODELGATE_API_KEY")  # шлюз Claude-моделей (OpenAI-совместимый, modelgate.app)
openai_api_key = os.getenv("OPENAI_API_KEY")  # официальный OpenAI API (gpt-5.x / o3); reasoning-модели
zai_api_key = os.getenv("ZAI_API_KEY")  # z.ai (Zhipu) — модели GLM (OpenAI-совместимый, прямой Bearer)
fireworks_api_key = os.getenv("FIREWORKS_API_KEY")  # Fireworks AI — serverless-модели (OpenAI-совместимый, прямой Bearer)
sakana_api_key = os.getenv("SAKANA_API_KEY")  # Sakana AI — Fugu (оркестратор поверх фронтир-LLM, OpenAI-совместимый, прямой Bearer)
sakana_proxy = os.getenv("SAKANA_PROXY")  # необяз. прокси для Sakana (WAF режет IP датацентров; формат http(s)://[user:pass@]host:port или socks5://…)
gloy_api_key = os.getenv("GLOY_API_KEY")  # LLM API FUN (Gloy AI) — OpenAI-совместимый, прямой Bearer
tavily_api_key = os.getenv("TAVILY_API_KEY")  # веб-поиск/извлечение страниц для /ask (tavily.com); без ключа веб-инструменты выключены
index_db_url = os.getenv("INDEX_DB_URL")  # MariaDB для /index (GraphRAG-память): mysql://user:pass@host:port/db (pass URL-encoded)
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")  # OCR фото (LlamaParse); без него фото идут через vision
index_memory_for_guests = (os.getenv("INDEX_MEMORY_FOR_GUESTS") or "").lower() in ("1", "true", "yes", "on")
index_use_hnsw = (os.getenv("INDEX_USE_HNSW") or "").lower() in ("1", "true", "yes", "on")


def _collect_google_tts_keys() -> list:
    """Ключи Google GenAI (TTS) из GOOGLE_GENAI_API_KEY и GOOGLE_GENAI_API_KEYS.
    Оба поля могут содержать список через запятую. Дедуп, порядок сохраняем."""
    raw = []
    for var in ("GOOGLE_GENAI_API_KEY", "GOOGLE_GENAI_API_KEYS"):
        val = os.getenv(var) or ""
        raw += [k.strip() for k in val.split(",") if k.strip()]
    seen, out = set(), []
    for k in raw:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


GOOGLE_TTS_KEYS = _collect_google_tts_keys()
tts_available = bool(GOOGLE_TTS_KEYS)

# Константы
AUTO_REPLY_ACCUMULATE_WINDOW = 1.5
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_VISION_MODEL = "google/gemini-3.1-flash-lite"  # дефолт vision (можно сменить /model media)
# Транскрипция (STT через /audio/transcriptions): chirp-3/whisper стали отдавать 400 (2026-06),
# заменены на дешёвые STT (проверено живьём: HTTP 200, ogg напрямую). Gemini для STT дорог.
OPENROUTER_AUDIO_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
OPENROUTER_AUDIO_FALLBACK = "mistralai/voxtral-mini-transcribe"  # запасная, если Parakeet не отвечает
OPENROUTER_IMAGE_MODEL = "openai/gpt-image-2"  # /gen: text→image и image→image (OpenAI GPT Image 2, платная)
OPENROUTER_IMAGE_FALLBACK = "google/gemini-3.1-flash-image"  # запасная при сбое/перегрузке основной (4K не умеет → авто-даунгрейд до 2K)
GEN_IMAGE_MAX_INPUT = 3_000_000  # лимит входного запроса ~4.5 МБ; base64 ×1.33 → входное фото до ~3 МБ
GEN_CTX_IMG_MAX = 20        # /gen: макс. фото-кандидатов когда vision-модель смотрит их НАПРЯМУЮ (прямой режим)
GEN_CTX_IMG_MAX_DESC = 300  # /gen: макс. фото-кандидатов в режиме ОПИСАНИЙ (текстовая модель или флаг -m) — описания компактны, ИИ выбирает из большого пула
GEN_CTX_REF_MAX = 16        # потолок референсов в генератор: GPT Image 2 принимает до 16 картинок на вход (офиц. OpenAI API)
GEN_CTX_THUMB_PX = 768      # /gen: сторона уменьшенной копии фото-кандидата для ПРОМПТЕРА (vision-вход/описание) — иначе 20 полных фото бьют лимит запроса (Alibaba ~28МБ)
GEN_VISION_RETRY_N = 8      # /gen: если vision-промптер отверг полный каталог (лимит числа картинок у провайдера, напр. Mistral/Pixtral) — повтор с этим числом
GEN_CATALOG_TIMEOUT = 75    # /gen: тайм-бюджет (сек) на скачивание+сжатие и описания каталога (прямой режим, ≤20 фото)
GEN_DESC_TIMEOUT = 150      # /gen: тайм-бюджет (сек) для режима описаний (до 300 фото; что не успело — пойдёт без описания, по кэшу /ask добирается со временем)
GEN_BATCH_MAX = 20          # /gen -xN: максимум вариантов за команду (каждый ~40с–2мин)
GEN_BATCH_CONCURRENCY = 2   # сколько вариантов генерим одновременно (баланс скорость/лимиты free-модели)

# --- /index: GraphRAG-память по истории чата (MariaDB + numpy-вектора) ---
INDEX_EXTRACT_MODEL = "deepseek/deepseek-v4-flash"   # экстракция досье/графа (текст): окно 1M, дёшево (OpenRouter)
INDEX_EXTRACT_FALLBACK = "deepseek/deepseek-v3.2-exp"  # если v4-flash недоступен ключу (проверяется пробным вызовом на фазе A)
INDEX_EMBED_TEXT_MODEL = "openai/text-embedding-3-small"   # тексты: досье, связи, сцены, описания фото ($0.02/1M, 1536d)
INDEX_EMBED_IMAGE_MODEL = "google/gemini-embedding-2"      # картинки (сам файл) + кросс-модальный текст-запрос (GA-слаг, 3072d)
INDEX_EMBED_DIM = 1536        # рабочая размерность (text-emb-3-small нативно 1536; gemini-emb-2 усекаем Matryoshka с 3072)
INDEX_DUMP_BATCH = 1000       # stage 0: сообщений на пачку дампа/чекпоинт
INDEX_SCENE_GAP_SEC = 15 * 60 # stage 2: разрыв >15 мин между сообщениями рвёт сцену
INDEX_SCENE_TOKEN_CAP = 8000  # stage 2: мягкий потолок токенов сцены (длинная беседа рвётся принудительно)
INDEX_SCENE_MIN_TOKENS = 1000 # stage 2: сцены короче — доклеиваем к следующей (не дробим на мелочь)
INDEX_SCENE_HARD_GAP_SEC = 6 * 60 * 60  # даже короткую сцену не склеиваем через многочасовую паузу
INDEX_STAGE2_CONCURRENCY = 6  # stage 2: сколько сцен экстрагировать параллельно (перекрыть ~34с латентность на вызов)
INDEX_STAGE1_MICRO_TOKENS = 32_000      # stage 1: размер блока экстракции (вывод не режется — cap поднят до INDEX_EXTRACT_MAX_TOKENS)
INDEX_EXTRACT_MAX_TOKENS = 64_000       # ПОТОЛОК ВЫВОДА экстракции: JSON + reasoning-токены (V4 Flash думает в тот же бюджет!) не режем. Провайдер принимает ≥200k
INDEX_MODEL_MAX_OUT = {                  # per-model кламп потолка вывода (страховка от иного роутинга; текущий тянет и 200k)
    "deepseek/deepseek-v4-flash": 16_384,
    "deepseek/deepseek-v3.2-exp": 65_536,
}
INDEX_SUMMARY_MAX_TOKENS = 64_000       # ПОТОЛОК ВЫВОДА саммари (досье/роллапы): запас под reasoning, чтобы CoT не съедал бюджет до самого текста. Оплата — по факту токенов
INDEX_STAGE1_MICRO_MESSAGES = 800
INDEX_STAGE1_BLOCK_TOKENS = INDEX_STAGE1_MICRO_TOKENS  # legacy alias для старых комментариев/логов
INDEX_FAILED_MIN_MESSAGES = 1
INDEX_UPDATE_OVERLAP_MESSAGES = 300
INDEX_UPDATE_OVERLAP_HOURS = 24
INDEX_SUMMARY_CLAIM_BATCH = 80
INDEX_SUMMARY_TOKEN_BATCH = 8_000
INDEX_SUMMARY_MAPREDUCE_MIN_CLAIMS = 80
INDEX_SUMMARY_MAPREDUCE_MIN_TOKENS = 20_000
INDEX_CONSOLIDATE_EVERY_BLOCKS = 10
INDEX_REGISTRY_EXACT_LIMIT = 200
INDEX_REGISTRY_SEMANTIC_LIMIT = 50
INDEX_REGISTRY_NEIGHBOR_LIMIT = 100
INDEX_REGISTRY_FALLBACK_LIMIT = 300
INDEX_MEDIA_PAUSE = (2.0, 4.0)  # stage 5: пауза между скачиваниями фото (анти-FloodWait юзербота)
INDEX_MEDIA_DL_TIMEOUT = 60     # stage 5: таймаут на одно скачивание фото — зависшая картинка не морозит стадию
INDEX_EMBED_BATCH = 96        # stage 3: строк на батч эмбеддинга
INDEX_EXTRACT_RETRIES = 3     # LLM/JSON extractor: не двигаем чекпоинт после временного сбоя
INDEX_EMBED_RETRIES = 3       # embeddings: 429/5xx у провайдера не должны превращаться в "done"
INDEX_MATRIX_CACHE_MAX_ROWS = 10_000  # больше ищем потоково, без полной матрицы в RAM
INDEX_SEARCH_DB_BATCH = 5_000
INDEX_COUNT_TTL = 60          # сек: кэш COUNT(*) для выбора backend поиска (инвалидируется на каждую запись индекса)
INDEX_ROLLUP_TOKEN_BATCH = 24_000  # stage 4: сколько токенов сцен сжимать за один map-вызов роллап-саммари
INDEX_SEARCH_FLOOR = 0.15     # ниже этого косинуса результат не показываем вообще
INDEX_SEARCH_CONFIDENT = 0.28 # ниже — «слабое» совпадение: помечаем и просим модель переспросить/веб (Corrective RAG)
INDEX_EVAL_CASES_PATH = "index_eval_cases.json"
GEN_INDEX_REF_MAX = 8         # /gen: сколько фото-референсов подтягивать из индекс-памяти (смысловой поиск по всей истории)
# OCR фото в /ask по умолчанию (cost-effective вместо vision-модели; флаг -m возвращает vision).
# Проверено живьём: v2-поток (files → parse tier=cost_effective → poll → markdown_full), ~11с/фото,
# русский распознаёт отлично. ВАЖНО: text_full отдаёт мусор латиницей — читать markdown_full.
LLAMA_PARSE_BASE = "https://api.cloud.llamaindex.ai"
LLAMA_PARSE_TIER = "cost_effective"

# Медиа-модели (vision) для выбора в /model media: slug -> (model_id, label)
MEDIA_MODEL_REGISTRY = {
    "lite":    ("google/gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite"),
    "lite-25": ("google/gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite (бюджет)"),
    "flash":   ("google/gemini-3-flash-preview", "Gemini 3 Flash (preview)"),
    "qwen-9b": ("qwen/qwen3.5-9b", "Qwen3.5 9B"),
    "qwen-flash": ("qwen/qwen3.5-flash-02-23", "Qwen3.5 Flash (02-23)"),
    "free":    ("openrouter/free", "OpenRouter Free (авто)"),
}
FREE_MEDIA_MODEL = "openrouter/free"  # авто-фоллбэк для гостей при N>500
# OpenCode-Go модели, доступные как медиа (vision). slug == api_model_id в MODEL_REGISTRY,
# поэтому медиа-пайплайн отличает их по самому id и роутит описание в opencode_client.
# NB: GLM-5/5.1 у opencode (эндпоинт frank/GLM-*) — ТЕКСТОВЫЕ, картинки не принимают
# (400 "does not accept image or video input"), поэтому в vision-список НЕ входят.
MEDIA_OPENCODE_SLUGS = ["kimi-k2.6", "kimi-k2.7-code", "qwen3.7-plus", "mimo-v2-omni", "minimax-m3"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
# opencode-go отдаёт некоторые модели (qwen3.7-max) ТОЛЬКО в формате Anthropic Messages
# (на OpenAI-формат → 401 "not supported for format oa-compat"). Свой эндпоинт + ключ в x-api-key.
OPENCODE_ANTHROPIC_URL = "https://opencode.ai/zen/go/v1/messages"
OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
# ModelGate — OpenAI-совместимый шлюз к моделям Claude (claude-opus-4-x / sonnet / haiku).
# Проверено вживую: /v1/chat/completions в OpenAI-формате, tools работают (как у thinking-моделей —
# принудительный tool_choice не поддержан, auto — да). ВНИМАНИЕ: шлюз НЕ передаёт картинки до Claude
# (и base64, и URL — модель отвечает «изображения нет»), поэтому модели только ТЕКСТ+поиск, без -g.
# WAF шлюза блокирует User-Agent "OpenAI/Python" (403 "Your request was blocked") — нужен браузерный UA
# (тот же трюк, что с Cloudflare у opencode).
MODELGATE_BASE_URL = "https://modelgate.app/v1"
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
# OpenAI — официальный API. Модели gpt-5.x/o3 — reasoning: на /chat/completions
# принимают ТОЛЬКО max_completion_tokens (не max_tokens) и лишь дефолтную temperature
# (1.0); поэтому клиент обёрнут адаптером _OpenAIReasoningClient (переименовывает
# max_tokens и убирает temperature). Vision и tools — нативные.
OPENAI_BASE_URL = "https://api.openai.com/v1"
ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"  # z.ai (Zhipu) международный, OpenAI-совместимый
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"  # Fireworks AI, OpenAI-совместимый
SAKANA_BASE_URL = "https://api.sakana.ai/v1"  # Sakana AI (Fugu), OpenAI-совместимый
GLOY_BASE_URL = "https://api.gloyai.fun/v1"  # LLM API FUN (Gloy AI), OpenAI-совместимый
GLOY_MAX_TOKENS = 8192  # Gloy жёстко режет max_tokens ∈ [1,8192] (400 иначе) — клампим (ASK_MAX_TOKENS=16000 не пройдёт)

# --- Google Gemini Flash TTS (голосовые ответы в /ask) ---
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
# Фолбэк-модель: у 3.1-preview документированная проблема «prompt classifier false rejections»
# (ложные 400 INVALID_ARGUMENT) и «occasional text token returns» (500). Если 3.1 упорно
# отклоняет — переключаемся на стабильную 2.5-flash-preview-tts. Google рекомендует retry-логику.
GEMINI_TTS_FALLBACK_MODEL = os.getenv("GEMINI_TTS_FALLBACK_MODEL", "gemini-2.5-flash-preview-tts")
GEMINI_TTS_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Последний фолбэк — ТА ЖЕ модель, но через OpenRouter (другой транспорт и квота: кредиты
# OpenRouter, а не Google-ключи). Эндпоинт OpenAI-совместимый /audio/speech, отдаёт сырой PCM.
GEMINI_TTS_OPENROUTER_MODEL = os.getenv("GEMINI_TTS_OPENROUTER_MODEL", "google/gemini-3.1-flash-tts-preview")
OPENROUTER_TTS_URL = "https://openrouter.ai/api/v1/audio/speech"
TTS_DEFAULT_VOICE = "Leda"     # дефолтный голос (см. VOICE_PROFILES)
TTS_PCM_RATE = 24000           # Gemini TTS отдаёт PCM s16le 24kHz mono
TTS_VOICE_CHAR_CAP = 5000      # потолок длины озвучиваемого текста (~4–5 мин речи). Fish s2-pro
                               # тянет это легко (проверено до 7000). Длиннее — режем. NB: при фолбэке
                               # на Gemini TTS очень длинный текст может дать 400 — тогда сработает обрезка/ретрай.
VOICE_SAMPLES_DIR = "voice_samples"  # кэш озвученных примеров голосов: voice_samples/<Имя>.ogg
VOICE_SAMPLE_TEXT = "Привет! Это мой голос. [с теплотой] Рада с тобой пообщаться."  # фраза-пример (одна на все голоса — удобно сравнивать)

# --- Fish Audio TTS (альтернативный движок озвучки, выбор через /voice engine fish) ---
fish_audio_api_key = os.getenv("FISH_AUDIO_API_KEY")
fish_available = bool(fish_audio_api_key)
FISH_TTS_URL = "https://api.fish.audio/v1/tts"
FISH_MODELS_URL = "https://api.fish.audio/model"   # поиск/список голосов: GET ?title=&sort_by=score
FISH_TTS_MODEL = os.getenv("FISH_TTS_MODEL", "s1")  # заголовок model: s1 / speech-1.5 / s2-pro

# 30 встроенных голосов Gemini TTS (порт из Bot_opekyn/src/voice/tts.ts).
# Каждый: name (для API), tone, pitch, personality (рус.), gender, emoji.
VOICE_PROFILES = [
    {"name": "Achernar",      "tone": "Soft",          "pitch": "Higher pitch",       "personality": "Мягкий, нежный, для утешения и ласки",        "gender": "female", "emoji": "🌙"},
    {"name": "Achird",        "tone": "Friendly",      "pitch": "Lower middle pitch", "personality": "Дружелюбный, тёплый, универсальный",          "gender": "female", "emoji": "🌟"},
    {"name": "Algenib",       "tone": "Gravelly",      "pitch": "Lower pitch",        "personality": "Хриплый, харизматичный, для серьёзных тем",    "gender": "male",   "emoji": "🔮"},
    {"name": "Algieba",       "tone": "Smooth",        "pitch": "Lower pitch",        "personality": "Плавный, спокойный, для объяснений",          "gender": "male",   "emoji": "💫"},
    {"name": "Alnilam",       "tone": "Firm",          "pitch": "Lower middle pitch", "personality": "Твёрдый, уверенный, для мотивации",           "gender": "male",   "emoji": "⚔️"},
    {"name": "Aoede",         "tone": "Breezy",        "pitch": "Middle pitch",       "personality": "Лёгкий, воздушный, для повседневных бесед",    "gender": "female", "emoji": "🍃"},
    {"name": "Autonoe",       "tone": "Bright",        "pitch": "Middle pitch",       "personality": "Яркий, энергичный, для радостных новостей",    "gender": "female", "emoji": "✨"},
    {"name": "Callirrhoe",    "tone": "Easy-going",    "pitch": "Middle pitch",       "personality": "Непринуждённый, расслабленный, дружеский тон", "gender": "female", "emoji": "🌊"},
    {"name": "Charon",        "tone": "Informative",   "pitch": "Lower pitch",        "personality": "Информативный, взвешенный, для фактов",        "gender": "male",   "emoji": "🚢"},
    {"name": "Despina",       "tone": "Smooth",        "pitch": "Middle pitch",       "personality": "Гладкий, ровный, универсальный",              "gender": "female", "emoji": "💎"},
    {"name": "Enceladus",     "tone": "Breathy",       "pitch": "Lower pitch",        "personality": "Дыхательный, интимный, для тихих моментов",    "gender": "male",   "emoji": "🪐"},
    {"name": "Erinome",       "tone": "Clear",         "pitch": "Middle pitch",       "personality": "Чёткий, ясный, для объяснений и обучения",     "gender": "female", "emoji": "📖"},
    {"name": "Fenrir",        "tone": "Excitable",     "pitch": "Lower middle pitch", "personality": "Возбудимый, эмоциональный, для шуток",         "gender": "male",   "emoji": "🐺"},
    {"name": "Gacrux",        "tone": "Mature",        "pitch": "Middle pitch",       "personality": "Зрелый, мудрый, для советов и размышлений",    "gender": "male",   "emoji": "🦉"},
    {"name": "Iapetus",       "tone": "Clear",         "pitch": "Lower middle pitch", "personality": "Чёткий, глубокий, для деловых разговоров",     "gender": "male",   "emoji": "🏛️"},
    {"name": "Kore",          "tone": "Firm",          "pitch": "Middle pitch",       "personality": "Твёрдый, сбалансированный, хороший дефолт",    "gender": "female", "emoji": "🌺"},
    {"name": "Laomedeia",     "tone": "Upbeat",        "pitch": "Higher pitch",       "personality": "Жизнерадостный, бодрый, для приветствий",      "gender": "female", "emoji": "☀️"},
    {"name": "Leda",          "tone": "Youthful",      "pitch": "Higher pitch",       "personality": "Молодой, игривый, энергичный (дефолт)",        "gender": "female", "emoji": "🦢"},
    {"name": "Orus",          "tone": "Firm",          "pitch": "Lower middle pitch", "personality": "Твёрдый, уверенный, для мотивации",           "gender": "male",   "emoji": "🌋"},
    {"name": "Puck",          "tone": "Upbeat",        "pitch": "Middle pitch",       "personality": "Весёлый, оживлённый, для шуток",               "gender": "male",   "emoji": "🎭"},
    {"name": "Pulcherrima",   "tone": "Forward",       "pitch": "Middle pitch",       "personality": "Напористый, прямой, для важных напоминаний",   "gender": "female", "emoji": "⚡"},
    {"name": "Rasalgethi",    "tone": "Informative",   "pitch": "Middle pitch",       "personality": "Информативный, нейтральный, для новостей",     "gender": "male",   "emoji": "📡"},
    {"name": "Sadachbia",     "tone": "Lively",        "pitch": "Lower pitch",        "personality": "Живой, динамичный, для активных обсуждений",   "gender": "male",   "emoji": "🔥"},
    {"name": "Sadaltager",    "tone": "Knowledgeable", "pitch": "Middle pitch",       "personality": "Знающий, экспертный, для обучения",            "gender": "male",   "emoji": "🎓"},
    {"name": "Schedar",       "tone": "Even",          "pitch": "Lower middle pitch", "personality": "Ровный, стабильный, для долгих бесед",         "gender": "female", "emoji": "🍁"},
    {"name": "Sulafat",       "tone": "Warm",          "pitch": "Middle pitch",       "personality": "Тёплый, уютный, для поддержки и заботы",       "gender": "female", "emoji": "🧣"},
    {"name": "Umbriel",       "tone": "Easy-going",    "pitch": "Lower middle pitch", "personality": "Непринуждённый, мягкий, для вечерних бесед",   "gender": "male",   "emoji": "🌙"},
    {"name": "Vindemiatrix",  "tone": "Gentle",        "pitch": "Middle pitch",       "personality": "Нежный, ласковый, для утешения",              "gender": "female", "emoji": "💌"},
    {"name": "Zephyr",        "tone": "Current",       "pitch": "Bright",             "personality": "Современный, яркий, молодёжный тон",           "gender": "male",   "emoji": "💨"},
    {"name": "Zubenelgenubi", "tone": "Casual",        "pitch": "Lower middle pitch", "personality": "Неформальный, расслабленный, для друзей",      "gender": "male",   "emoji": "🛋️"},
]


def _voice_profile(name: str):
    """Профиль голоса по имени (регистронезависимо) или None."""
    if not name:
        return None
    low = name.strip().lower()
    for p in VOICE_PROFILES:
        if p["name"].lower() == low:
            return p
    return None


def _validate_voice(name: str) -> str:
    """Имя существующего голоса или дефолт TTS_DEFAULT_VOICE."""
    p = _voice_profile(name)
    return p["name"] if p else TTS_DEFAULT_VOICE
MSK = timezone(timedelta(hours=3))
CHANNELS_PATH = "channels.json"
DIGEST_STATE_PATH = "digest_state.json"
MODEL_STATE_PATH = "model_state.json"
MEDIA_CACHE_PATH = "media_cache.json"
MEDIA_CACHE_TS_PATH = "media_cache_ts.json"
AUTO_REPLY_PATH = "auto_reply.json"
OPENAI_USAGE_PATH = "openai_usage.json"  # дневной счётчик токенов OpenAI (бесплатная квота data sharing)
ALLOWED_PATH = "allowed_users.json"
ALLOWED_ASK_TEXT_LIMIT = 500  # для гостей: запрос > этого числа → vision переключается на free
MEDIA_HIDETAIL_MAX_N = 200    # /ask с N больше этого → описываем фото в detail="low" (дешевле)
DIRECT_VISION_MAX_IMAGES = 20 # /ask -g: макс. картинок, отдаваемых модели напрямую (берём самые свежие)
ASKS_KEEP = 100               # кол-во последних /ask -d дампов, хранимых в asks/
REPLY_NETWORK_BUDGET = 200    # макс сетевых get_reply_message() за один /ask (когда target вне выборки)
ASK_MAX_TOKENS = 16000        # потолок completion для /ask (thinking-модели тратят на reasoning до тысяч токенов)
MEDIA_CONCURRENCY = 10    # параллельная обработка медиа (Gemini выдерживает)
MEDIA_MAX_ITEMS = 300     # потолок медиа-сегментов на один /ask: старее — плейсхолдеры.
                          # Защита от OOM (контейнер убивало по памяти на /ask 4100 с сотнями медиа)
SEARCH_CONCURRENCY = 5    # параллельный поиск по каналам
# Чтение текстовых файлов-вложений в /ask (по умолчанию): содержимое идёт в контекст.
DOC_MAX_BYTES = 512_000   # больше — не тянем (плейсхолдер [Файл])
DOC_MAX_CHARS = 32_000    # потолок встраиваемого текста файла (~10–12k токенов), дальше обрезка
TEXT_MIME = {"application/json", "application/xml", "application/javascript", "application/x-yaml",
             "application/x-sh", "application/x-python", "application/toml", "application/csv",
             "application/x-tex", "application/sql", "image/svg+xml"}
TEXT_EXT = {"txt", "md", "markdown", "rst", "json", "jsonl", "csv", "tsv", "yaml", "yml", "toml",
            "ini", "cfg", "conf", "env", "log", "xml", "html", "htm", "css", "py", "js", "ts",
            "jsx", "tsx", "sh", "bash", "zsh", "sql", "go", "rs", "java", "kt", "c", "h", "cpp",
            "hpp", "cs", "rb", "php", "swift", "lua", "pl", "r", "dart", "scala", "tex", "srt", "vtt"}
# Мягкий якорь окна контекста под prompt-кэш: не сдвигаем начало лога между /ask,
# но перебор ограничен CTX_ANCHOR_SNAP сообщениями. Якорь СВОЙ на каждую модель (кэш раздельный).
CTX_ANCHOR_SNAP = 100     # макс. «дотяжка» назад к якорю / перебор сверх N
CTX_ANCHOR_TTL = 1800     # сек: якорь старше — переустанавливается (свежесть)
_ctx_anchors = {}         # {(chat_id, model_slug): {"anchor_id": int, "ts": float}}
AUTO_REPLY_HISTORY_MAX = 20  # сообщений (≈10 реплик) на чат
COLLECT_WORKERS = 4           # параллельные окна сбора истории (/ask). Консервативно — низкий риск FloodWait.
COLLECT_MIN_PER_WORKER = 500  # минимум сообщений на воркер; при меньшем N — меньше воркеров (мелкие /ask не дробим зря)
COLLECT_OVERFETCH = 1.2       # запас позиций на скип сервисных/команд/исключённых

# Реестр моделей: slug -> (provider, api_model_id, label, context_window_tokens, ctx_safety_mult)
# ctx_safety_mult: множитель «осторожности» бюджета — учитывает, что токенизатор
# целевой модели может быть плотнее o200k (которым считает tiktoken). На опыте:
# Kimi K2.x ≈ 2.25× плотнее o200k → safety 2.5. Остальные близки к o200k → 1.15.
MODEL_REGISTRY = {
    "deepseek-pro": ("deepseek", DEEPSEEK_MODEL, "DeepSeek V4 Pro", 1000000, 1.15),
    "deepseek-flash": ("deepseek", "deepseek-v4-flash", "DeepSeek V4 Flash", 1000000, 1.15),  # прямой API
}
# Реестр почищен (2026-06-14): оставлены только новейшие версии каждой модели на КАЖДОМ провайдере
# (разный провайдер/транспорт — отдельная модель). Убраны устаревшие: glm-5/5.1 (на opencode появился
# glm-5.2 — см. ниже), kimi-k2.5, minimax-m2.5/m2.7, qwen3.5/3.6-plus, mimo-v2.5/v2-pro.
for _mid, _label, _ctx, _safety in [
    ("deepseek-v4-pro",  "DeepSeek V4 Pro",   1000000, 1.15),
    ("deepseek-v4-flash","DeepSeek V4 Flash", 1000000, 1.15),
    ("kimi-k2.6",        "Kimi K2.6",          262000, 2.50),
    ("kimi-k2.7-code",   "Kimi K2.7 Code",     262000, 2.50),
    ("minimax-m3",       "MiniMax M3",        1000000, 1.30),
    ("qwen3.7-plus",     "Qwen3.7 Plus",       262000, 1.15),
    ("mimo-v2.5",        "MiMo V2.5",         1000000, 1.50),
    ("mimo-v2.5-pro",    "MiMo V2.5 Pro",     1000000, 1.50),
    ("mimo-v2-omni",     "MiMo V2 Omni",      1000000, 1.50),
    ("hy3-preview",      "Hunyuan 3 Preview",  256000, 1.50),
]:
    MODEL_REGISTRY[_mid] = ("opencode", _mid, _label, _ctx, _safety)
# GLM-5.2 на opencode (текст, tools, reasoning_content отделяется — проверено вживую 2026-06-17;
# заменил glm-5.1 по принципу «новейшая на провайдере»). Слаг с суффиксом -oc, т.к. голый "glm-5.2"
# занят z.ai; api-id = "glm-5.2". Окно 1M: каталог не отдаёт, но провайдер в ошибке переполнения
# сообщил "model maximum context length: 1048575" — нативное 1M у 5.2, как на z.ai/Fireworks.
MODEL_REGISTRY["glm-5.2-oc"] = ("opencode", "glm-5.2", "GLM-5.2 (OC)", 1000000, 1.30)
# qwen3.7-max — opencode отдаёт её только в формате Anthropic Messages → провайдер "oc_anthropic"
# (свой адаптер-обёртка под OpenAI-интерфейс; полноценный tool-loop/голос, как у прочих).
MODEL_REGISTRY["qwen3.7-max"] = ("oc_anthropic", "qwen3.7-max", "Qwen3.7 Max", 262000, 1.15)
# Claude через ModelGate (OpenAI-совместимый шлюз). Окно 200k, vision и tools — нативные.
# safety 1.2: токенизатор Claude ≈ o200k, небольшой запас.
for _cid, _clabel in [
    ("claude-opus-4-8",   "Claude Opus 4.8"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-haiku-4-5",  "Claude Haiku 4.5"),
]:
    MODEL_REGISTRY[_cid] = ("modelgate", _cid, _clabel, 200000, 1.2)
# OpenAI (официальный API). Окна сверены с официальными страницами моделей (2026-06-12):
# gpt-5.4/5.5 — 1,050,000 (флагманы 5.4+ получили ~1M окно), gpt-5.4-mini — 400k,
# o3/o4-mini — 200k. safety 1.1 — токенизатор o200k почти совпадает с tiktoken бота.
for _oid, _olabel, _octx in [
    ("gpt-5.5", "GPT-5.5", 1050000),
    ("gpt-5.4", "GPT-5.4", 1050000),
    ("o3",      "OpenAI o3", 200000),
    ("gpt-5.4-mini", "GPT-5.4 Mini", 400000),
    ("o4-mini", "OpenAI o4-mini", 200000),
]:
    MODEL_REGISTRY[_oid] = ("openai", _oid, _olabel, _octx, 1.1)
# Google Gemini Flash (официальный generativelanguage REST, как наш TTS). Окно 1M/выход 64k.
# safety 1.15 — токенизатор близок к o200k. Ключи берём из GOOGLE_TTS_KEYS (общие с голосом).
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# gemini-3.1-pro-preview НЕ добавлен: на бесплатных ключах даёт 429 (нет free-квоты, нужен биллинг).
GEMINI_MODELS = {"gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite"}
for _gid, _glabel in [("gemini-3.5-flash", "Gemini 3.5 Flash"),
                      ("gemini-3-flash-preview", "Gemini 3 Flash"),
                      ("gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite")]:
    MODEL_REGISTRY[_gid] = ("google", _gid, _glabel, 1048576, 1.15)
# z.ai (Zhipu) — модели GLM, OpenAI-совместимый API (api.z.ai, прямой Bearer). safety 1.30 (как GLM
# у opencode). Окна сверены (2026-06-14): GLM-5.2 — 1M (выход 131k), 4.7-flash — ~200k (202752),
# 4.6v-flash — 128k. glm-4.6v-flash и glm-4.7-flash — Free-тариф.
for _zid, _zlabel, _zctx in [
    ("glm-5.2",        "GLM-5.2",           1000000),
    ("glm-4.7-flash",  "GLM-4.7 Flash",      200000),
    ("glm-4.6v-flash", "GLM-4.6V Flash",     128000),
]:
    MODEL_REGISTRY[_zid] = ("zai", _zid, _zlabel, _zctx, 1.30)
ZAI_VISION = {"glm-4.6v-flash"}  # из GLM на z.ai картинки принимает только V-модель (проверено)
# Fireworks AI — serverless-модели, OpenAI-совместимый API (api.fireworks.ai/inference/v1, прямой Bearer).
# id модели — ПОЛНЫЙ путь accounts/fireworks/models/<slug>. Слаги с префиксом fw- (имена линеек уже заняты
# opencode/direct). Окна/vision из каталога + сверено вживую (2026-06-17): текст, tools, reasoning_content
# отделяется; vision у minimax-m3 (512k) и kimi-k2.6. Kimi safety 2.50 (плотный токенизатор, как на opencode).
for _fwid, _fwapi, _fwlabel, _fwctx, _fwsafe in [
    ("fw-minimax-m3",       "minimax-m3",             "MiniMax M3 (FW)",       512000, 1.30),
    ("fw-nemotron-3-ultra", "nemotron-3-ultra-nvfp4", "Nemotron 3 Ultra",      262144, 1.30),
    ("fw-deepseek-v4-pro",  "deepseek-v4-pro",        "DeepSeek V4 Pro (FW)", 1048576, 1.15),
    ("fw-glm-5.2",          "glm-5p2",                "GLM-5.2 (FW)",         1048576, 1.30),
    ("fw-kimi-k2.6",        "kimi-k2p6",              "Kimi K2.6 (FW)",        262144, 2.50),
]:
    MODEL_REGISTRY[_fwid] = ("fireworks", "accounts/fireworks/models/" + _fwapi, _fwlabel, _fwctx, _fwsafe)
FIREWORKS_VISION = {"fw-minimax-m3", "fw-kimi-k2.6"}  # vision-варианты на Fireworks (проверено вживую)
# Sakana AI — Fugu: OpenAI-совместимый API (api.sakana.ai/v1, прямой Bearer). fugu — быстрая мини-модель,
# fugu-ultra — мульти-агентный «дирижёр» поверх фронтир-LLM (сложные задачи, может быть МЕДЛЕННЫМ). Окно 1M,
# обе vision (text+image, проверено вживую). Tools — нативно с tool_choice=auto; forced tool_choice игнорится
# (как oc_anthropic) → в /model probe не зондируем, на форс-поиске откатываемся на auto. reasoning_effort —
# только high/xhigh/max (нет none/low/medium → «пол» high, off нет). safety 1.15 (токенизатор неизвестен).
for _skid, _sklabel, _skctx in [
    ("fugu",       "Fugu",       1000000),
    ("fugu-ultra", "Fugu Ultra", 1000000),
]:
    MODEL_REGISTRY[_skid] = ("sakana", _skid, _sklabel, _skctx, 1.15)
SAKANA_VISION = {"fugu", "fugu-ultra"}  # обе модели Sakana принимают картинки (проверено вживую)

# LLM API FUN (api.gloyai.fun) — Gloy AI. OpenAI-совместимый, прямой Bearer. Проверено вживую 2026-07:
# текст ОК и уважает системный промпт; vision НЕ надёжен (сервер перезаливает картинку на 0x0.st → 460);
# нативных tool_calls НЕТ (форс отдаёт JSON в content) → в /model probe не зондируем, форс-поиск → auto;
# внутренний reasoning ест токены (нужен большой max_tokens). Окно 128k — оценка (реальное неизвестно), safety 1.15.
for _glid, _glapi, _gllabel in [
    ("gloy-1", "gloy_1.0", "Gloy AI 1.0"),
    ("gloy-2", "gloy_2.0", "Gloy AI 2.0"),
]:
    MODEL_REGISTRY[_glid] = ("gloy", _glapi, _gllabel, 128000, 1.15)
# Уровни глубины размышлений (reasoning_effort) OpenAI-моделей, от мощного к слабому.
# API жёстко валидирует значение ПО МОДЕЛИ (неподдерживаемое → 400): gpt-5.4/5.5 принимают
# none/low/medium/high/xhigh, o3 — только low/medium/high. Дефолты: 5.5 → medium, 5.4 → none, o3 → medium.
OPENAI_REASONING_LEVELS = {
    "gpt-5.5": ["xhigh", "high", "medium", "low", "none"],
    "gpt-5.4": ["xhigh", "high", "medium", "low", "none"],
    "o3":      ["high", "medium", "low"],
    "gpt-5.4-mini": ["xhigh", "high", "medium", "low", "none"],  # проверено зондом 2026-06-12
    "o4-mini": ["xhigh", "high", "medium", "low"],               # none не принимает (зонд)
}
OPENAI_REASONING_DEFAULTS = {"gpt-5.5": "medium", "gpt-5.4": "none", "o3": "medium",
                             "gpt-5.4-mini": "none", "o4-mini": "medium"}  # что применяет API без параметра
# Gemini 3.x: глубина размышлений — thinkingLevel (minimal|low|medium|high), полного off нет.
# Единый глобальный REASONING_EFFORT (шкала xhigh..none) мапим на уровни Gemini.
GEMINI_THINKING_MAP = {"xhigh": "high", "high": "high", "medium": "medium", "low": "low", "none": "minimal"}
GEMINI_THINKING_DEFAULT = "medium"  # что Google применяет без thinkingConfig (для показа в /status)
# o-серия принимает tools+reasoning_effort на /chat/completions; gpt-5.x — НЕТ (400
# «Function tools with reasoning_effort are not supported... use /v1/responses», зонд 2026-06-12)
# → для gpt-5.x эта комбинация уходит через Responses API (см. _OpenAIReasoningClient._via_responses).
OPENAI_TOOLS_EFFORT_CHAT_OK = {"o3", "o4-mini"}
_REASONING_RANK = ["xhigh", "high", "medium", "low", "none"]  # шкала силы для клампа
# Fireworks reasoning-модели: reasoning_effort принимает low/medium/high/none (нет xhigh; none = отключить
# мышление). Доки https://docs.fireworks.ai/guides/reasoning. Шкала от мощного к слабому для выбора N.M.
FIREWORKS_REASONING_LEVELS = ["high", "medium", "low", "none"]
# opencode (zen/go) пропускает reasoning_effort к моделям (проверено вживую 2026-06-17: glm-5.2 none→0
# симв. размышления, high→3545 — none даёт чёткий off-switch). Доки не описывают параметр — выяснено тестом.
# Шкала high/medium/low/none (xhigh нет → high). none у glm-5.2 отключает мышление; deepseek-v4-pro на none
# даёт 400 → обёртка ретраит без параметра (дефолтная глубина). Не-reasoning модели тоже покрыты ретраем.
# minimax-m3 кладёт мышление в <think> (его и так режет _strip_think).
OPENCODE_REASONING_LEVELS = ["high", "medium", "low", "none"]
# Официальный DeepSeek (V4): мышление вкл/выкл через extra_body thinking, глубина reasoning_effort.
# Реально различимы high и max (low/medium→high, xhigh→max — доки api-docs.deepseek.com/guides/thinking_mode).
# Проверено вживую 2026-06-17: thinking:disabled → reasoning=0 (off), high vs max — реальная разница
# (flash high→1066 / max→5402). Шкала в ГЛОБАЛЬНЫХ значениях (xhigh→max, none→off-switch).
DEEPSEEK_REASONING_LEVELS = ["xhigh", "high", "none"]
# Sakana Fugu: reasoning_effort принимает ТОЛЬКО high/xhigh/max (проверено вживую — none/low/medium → 400).
# Off нет, «пол» = high. Два уровня для N.M: xhigh(→max) и high. Инжектим только при high/xhigh (ниже — дефолт).
SAKANA_REASONING_LEVELS = ["xhigh", "high"]
# Бесплатные дневные квоты OpenAI по программе data sharing (Tier 1-2, сброс в 00:00 UTC):
# 250k/день на основные модели (gpt-5.x/o3) и ОТДЕЛЬНЫЕ 2.5M/день на mini-группу.
# Счётчик бота — ориентир: внешние запросы организации он не видит, а граничный
# запрос OpenAI биллит целиком.
OPENAI_FREE_DAILY_LARGE = 250_000
OPENAI_FREE_DAILY_MINI = 2_500_000
OPENAI_MINI_MODELS = {"gpt-5.4-mini", "o4-mini"}  # модели mini-группы квоты


def _cached_tokens(usage) -> int:
    """Сколько входных токенов пришло из prompt-кэша. Поля разнятся: OpenAI —
    usage.prompt_tokens_details.cached_tokens; DeepSeek — usage.prompt_cache_hit_tokens;
    Gemini — usage.cached_tokens (проставляет адаптер из cachedContentTokenCount)."""
    try:
        det = getattr(usage, "prompt_tokens_details", None)
        if det is not None:
            v = getattr(det, "cached_tokens", None)
            if v:
                return int(v)
        for attr in ("prompt_cache_hit_tokens", "cached_tokens"):
            v = getattr(usage, attr, None)
            if v:
                return int(v)
    except Exception:
        pass
    return 0


def _openai_bucket(model_id: str) -> str:
    return "mini" if model_id in OPENAI_MINI_MODELS else "large"


def _clamp_reasoning(model_id: str, effort: str, provider: str = None) -> str:
    """Приводит глобальный уровень ризонинга (шкала xhigh..none) к допустимому для модели/провайдера:
    Gemini → thinkingLevel (none→minimal, xhigh→high); Fireworks → low/medium/high/none (xhigh→high);
    opencode → low/medium/high/none (xhigh→high; none — off, у deepseek фоллбэк); o3 → low/medium/high. Неизвестная → medium.
    provider передаётся явно для opencode (его api-id — голый слаг, по строке не отличить)."""
    if provider == "google" or model_id in GEMINI_MODELS:
        return GEMINI_THINKING_MAP.get(effort, "medium")
    if provider == "fireworks" or model_id.startswith("accounts/fireworks/"):
        return "high" if effort == "xhigh" else effort  # Fireworks: low/medium/high/none (нет xhigh)
    if provider == "opencode":
        return "high" if effort == "xhigh" else effort  # low/medium/high/none (none → 400-fallback у deepseek)
    if provider == "deepseek":
        # официальный DeepSeek V4: none→off, xhigh→max, остальное→high (low/medium схлопываются в high)
        if effort == "none":
            return "none"
        return "max" if effort == "xhigh" else "high"
    if provider == "sakana":
        return "max" if effort == "xhigh" else "high"  # Sakana: только high/xhigh→max (off/low/medium нет)
    levels = OPENAI_REASONING_LEVELS.get(model_id)
    if not levels:
        return effort if effort in ("low", "medium", "high") else "medium"
    if effort in levels:
        return effort
    try:
        r = _REASONING_RANK.index(effort)
    except ValueError:
        return "medium"
    return min(levels, key=lambda lv: abs(_REASONING_RANK.index(lv) - r))


def _fmt_rlevel(model_id: str, lv: str, provider: str) -> str:
    """Имя ступени для таблицы /model reason: показывает реально применяемое значение
    (xhigh→max у DeepSeek, xhigh→high у Fireworks/opencode/Gemini, none→off/minimal)."""
    abbr = {"medium": "med", "minimal": "min", "none": "off"}
    applied = _clamp_reasoning(model_id, lv, provider)
    base = abbr.get(lv, lv)
    if applied != lv:
        return f"{base}→{abbr.get(applied, applied)}"
    return base


def _supports_reasoning(provider: str) -> bool:
    """Провайдеры с управляемой глубиной размышлений (/model reason): OpenAI, Google Gemini, Fireworks, opencode, DeepSeek, Sakana."""
    return provider in ("openai", "google", "fireworks", "opencode", "deepseek", "sakana")


# Task-локальный оверрайд глубины размышлений для утилитарных вызовов (дайджест): обёртки читают
# _effective_reasoning() вместо глобального REASONING_EFFORT. contextvars изолирует значение по asyncio-таске
# (параллельный /ask не затронут) и копируется в поток asyncio.to_thread, где крутится .create.
_NO_REASONING_OVERRIDE = object()
_REASONING_OVERRIDE = contextvars.ContextVar("reasoning_override", default=_NO_REASONING_OVERRIDE)


def _effective_reasoning():
    """Уровень ризонинга для ТЕКУЩЕГО вызова: оверрайд (если задан в этой таске) либо глобальный REASONING_EFFORT."""
    ov = _REASONING_OVERRIDE.get()
    return REASONING_EFFORT if ov is _NO_REASONING_OVERRIDE else ov


def _reasoning_levels(slug: str):
    """Список уровней ризонинга для выбора `N.M` у модели (от мощного к слабому). None — не поддерживает."""
    spec = MODEL_REGISTRY.get(slug)
    if not spec:
        return None
    if spec[0] == "openai":
        return OPENAI_REASONING_LEVELS.get(slug)
    if spec[0] == "google":
        return _REASONING_RANK  # общая 5-уровневая шкала; маппится на thinkingLevel в _clamp_reasoning
    if spec[0] == "fireworks":
        return FIREWORKS_REASONING_LEVELS  # low/medium/high/none
    if spec[0] == "opencode":
        return OPENCODE_REASONING_LEVELS  # high/medium/low/none
    if spec[0] == "deepseek":
        return DEEPSEEK_REASONING_LEVELS  # xhigh(→max)/high/none(→off)
    if spec[0] == "sakana":
        return SAKANA_REASONING_LEVELS  # xhigh(→max)/high — off нет
    return None


def _reasoning_tag() -> str:
    """' · 🤔 high' — применяемый уровень ризонинга активной модели (OpenAI/Gemini) для префикса
    ответа и подписей. Без /model reason показывает дефолт. Прочие провайдеры → пустая строка."""
    spec = MODEL_REGISTRY.get(ACTIVE_MODEL)
    if not spec or not _supports_reasoning(spec[0]):
        return ""
    if REASONING_EFFORT:
        return f" · 🤔 {_clamp_reasoning(spec[1], REASONING_EFFORT, spec[0])}"
    default = GEMINI_THINKING_DEFAULT if spec[0] == "google" else OPENAI_REASONING_DEFAULTS.get(spec[1], "auto")
    return f" · 🤔 {default}"


# Автообрезка контекста под окно модели
CTX_RESERVE_TOKENS = ASK_MAX_TOKENS + 2000  # запас под ВЕСЬ ответ (ASK_MAX_TOKENS, у thinking-моделей reasoning жрёт почти весь потолок) + системку + вопрос
CTX_CHARS_PER_TOKEN = 2.0   # фоллбэк-оценка, если tiktoken недоступен
# Подсчёт токенов идёт через o200k (GPT-4o), но отвечают другие модели
# (DeepSeek/Qwen/GLM/…). Их BPE-словари близки, но не идентичны — даём запас:
# считаем, что у целевой модели токенизатор на ~15% «плотнее», чтобы не переполнить окно.
CTX_TOKEN_SAFETY = 1.15
MEDIA_CACHE_MAX = 30000  # мягкий лимит, FIFO-вытеснение самых старых при переполнении
MEDIA_FAILURE_MARKERS = {"[изображение]", "[аудио сообщение]", "[видео кружок]"}
DIGEST_SYSTEM_PROMPT = """Ты — редактор дайджеста. Сгруппируй посты из Telegram-каналов по темам.
Для каждой темы: краткое резюме 2–3 предложения и ссылку на пост-источник.
Формат:
📌 Название темы
Резюме.
🔗 ссылка
Только по делу, без воды. На русском."""

TELEGRAM_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "telegram_search",
        "description": "Ищет посты в отслеживаемых Telegram-каналах по запросу. Возвращает до 10 результатов с ДАТОЙ публикации, текстом поста, названием канала и ссылкой. Используй для поиска актуальной информации, новостей, фактов. Результаты отсортированы от новых к старым.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос на русском или английском. Формулируй точные запросы для лучших результатов."
                },
                "days": {
                    "type": "integer",
                    "description": "Опционально: искать только посты за последние N дней. Используй для свежих новостей (например 7 — за неделю, 1 — за сутки)."
                }
            },
            "required": ["query"]
        }
    }
}

# --- Веб-инструменты Tavily (tavily.com) для /ask: модель САМА решает, когда искать ---
TAVILY_BASE_URL = "https://api.tavily.com"
WEB_SEARCH_MAX_RESULTS = 8        # потолок результатов на один web_search
WEB_EXTRACT_MAX_URLS = 5          # потолок URL на один web_extract
WEB_EXTRACT_MAX_CHARS = 8000      # обрезка текста одной страницы (extract)
WEB_CRAWL_MAX_PAGES = 8           # потолок страниц на один web_crawl
WEB_CRAWL_PAGE_CHARS = 2000       # обрезка текста одной страницы (crawl)
WEB_MAP_MAX_URLS = 40             # потолок ссылок на один web_map

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Ищет информацию в интернете (поисковик Tavily). Возвращает до 8 результатов: заголовок, URL, дату и выдержку текста, плюс краткий готовый ответ. Используй, когда вопрос требует актуальных или внешних знаний: новости, события, цены, версии, факты о людях/компаниях, всё чего нет в контексте переписки. Для свежих новостей ставь topic=news и time_range.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос. Формулируй конкретно, как для Google."},
                "topic": {"type": "string", "enum": ["general", "news"], "description": "news — поиск по новостным сайтам с датами публикаций; general — обычный веб-поиск (по умолчанию)."},
                "time_range": {"type": "string", "enum": ["day", "week", "month", "year"], "description": "Опционально: ограничить результаты по свежести (day — за сутки, week — за неделю и т.д.)."},
                "max_results": {"type": "integer", "description": "Сколько результатов вернуть, 1-8 (по умолчанию 5)."},
                "search_depth": {"type": "string", "enum": ["basic", "advanced"], "description": "advanced — глубже и точнее, но медленнее; для сложных вопросов. По умолчанию basic."}
            },
            "required": ["query"]
        }
    }
}
WEB_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "web_extract",
        "description": "Скачивает и извлекает полный текст веб-страниц по URL (до 5 за раз). Используй, чтобы прочитать конкретную страницу целиком: статью из результатов web_search, ссылку из переписки, документацию. Возвращает текст в markdown.",
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {"type": "array", "items": {"type": "string"}, "description": "Список URL для извлечения (1-5)."}
            },
            "required": ["urls"]
        }
    }
}
WEB_CRAWL_TOOL = {
    "type": "function",
    "function": {
        "name": "web_crawl",
        "description": "Обходит сайт по ссылкам начиная с указанного URL и возвращает тексты найденных страниц (до 8 страниц). Используй, когда нужно изучить РАЗДЕЛ сайта целиком (документацию, блог, каталог), а не одну страницу. Дорогая операция — не вызывай без необходимости.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Стартовый URL обхода."},
                "instructions": {"type": "string", "description": "Опционально: что именно искать при обходе, на естественном языке (например «страницы с ценами»)."}
            },
            "required": ["url"]
        }
    }
}
WEB_MAP_TOOL = {
    "type": "function",
    "function": {
        "name": "web_map",
        "description": "Возвращает карту сайта — список URL страниц, найденных по ссылкам с указанного адреса (до 40). Используй, чтобы понять структуру сайта и выбрать нужные страницы для web_extract. Быстрее и дешевле, чем web_crawl.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL сайта для построения карты."}
            },
            "required": ["url"]
        }
    }
}
WEB_TOOLS = [WEB_SEARCH_TOOL, WEB_EXTRACT_TOOL, WEB_CRAWL_TOOL, WEB_MAP_TOOL]

# --- Инструмент адресного реплая: ИИ САМ отвечает реплаем на конкретные сообщения истории ---
REPLY_MAX = 10  # анти-спам: не больше N реплаев за один /ask
REPLY_COLLAPSE = 300  # реплаи длиннее — сворачиваются в раскрывающийся цитат-блок (общий ответ /ask — 700)
REPLY_TOOL = {
    "type": "function",
    "function": {
        "name": "reply_to_messages",
        "description": (
            "Ответить РЕПЛАЕМ (с цитированием) на конкретные сообщения из истории по их #id "
            "(числа в метках #id перед каждым сообщением). Можно сразу на несколько — каждый ответ "
            "уйдёт ОТДЕЛЬНЫМ сообщением, прикреплённым к своему исходному. Используй для адресных "
            "ответов: на спор, на вопрос конкретного человека, на реплики разных людей. Максимум "
            f"{REPLY_MAX} реплаев за раз. После вызова дай ещё и общий итоговый ответ обычным текстом "
            "(он отправится отдельно) — не дублируй в нём дословно то, что уже написал в реплаях."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "replies": {
                    "type": "array",
                    "description": "Список адресных ответов (1+). Каждый — реплай на своё сообщение.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "message_id": {"type": "integer", "description": "id сообщения из метки #id в истории"},
                            "text": {"type": "string", "description": "текст ответа именно на это сообщение"},
                            "quote": {"type": "string", "description": "НЕОБЯЗАТЕЛЬНО: точная подстрока ИЗ этого сообщения (дословно, как в тексте), чтобы подсветить в цитате именно тот фрагмент, на который отвечаешь. Если не нужен фрагмент — не указывай, ответ прикрепится ко всему сообщению."}
                        },
                        "required": ["message_id", "text"]
                    }
                }
            },
            "required": ["replies"]
        }
    }
}

ASK_SYSTEM_PROMPT = """Ты — {model}, ИИ с характером и собственной точкой зрения. Не нейтральный ассистент, а собеседник с позицией.

Правила:
- Отвечай на русском. Пиши плотно и по делу, но СОДЕРЖАТЕЛЬНО: сжимай формулировки, а не информацию. Без воды, повторов и пустых вводных — но каждый тезис раскрыт: с аргументом, конкретной деталью или примером, а не брошен голой строкой. Оценка/вывод без объяснения «почему» — это отписка, так не делай.
- Развёрнутость по делу приветствуется: лучше полезный ответ на несколько абзацев, чем пустая короткая реплика. Режь только лишние слова; факты, наблюдения и нюансы оставляй.
- Когда материала много — структурируй: абзацы по одной мысли, списки «• » по делу. Плотность ≠ сухость: живой характер, позиция и интонация остаются.
- Говори о людях в третьем лице, без местоимений: не «ты сказал», не «он написал», а по имени или роли — «Маша написала», «собеседник предложил», «автор сообщения считает».
- Имей позицию. Если с чем-то не согласен — скажи прямо. Не подстраивайся под всех.
- Не извиняйся, не используй эмоджи-заглушки.
- Если в контексте медиа — опирайся на описание, реагируй конкретно.
- Если используешь информацию из поиска — указывай источник ссылкой через тег <a href="…">.

Форматирование ответа — строго Telegram-HTML (ответ парсится как HTML):
- Разрешены ТОЛЬКО теги: <b>, <i>, <u>, <s>, <code>, <pre>, <a href="URL">, <blockquote>. Других тегов не используй (никаких <p>, <br>, <div>, <h1>, <ul>, <ol>, <li>, <table>).
- Списки оформляй обычными строками с «• » или «1. », НЕ тегами.
- Ссылки — только <a href="URL">текст</a>.
- В обычном тексте экранируй символы: & → &amp;, < → &lt;, > → &gt; (внутри <code>/<pre> — тоже).
- Не оборачивай весь ответ в один <pre>/<code>; код — только реальный код.

Лог переписки (контекст/фон) идёт ПЕРВЫМ, а сам твой вопрос (помечен ❓) и текущее время — в САМОМ КОНЦЕ, после лога. Выполняй именно ❓-вопрос: если он просит ответить на сообщения или вопросы из переписки — делай это по контексту.

Формат контекста: это лог чата. Каждое сообщение — отдельный блок, блоки разделены пустой строкой. Заголовок в квадратных скобках: [время автор]: текст. Метки в заголовке: «↩ автор: «цитата»» — это ответ на сообщение указанного автора; «⤷ из X» — сообщение переслано из источника X. В тексте: [Фото: …]/[Аудио: …]/[Речь: …] — распознанное содержимое медиа, [Файл «имя»: …] — содержимое текстового файла; [Видео]/[GIF] — медиа без распознавания."""
AUTO_REPLY_SYSTEM_PROMPT = """Ты — собеседник в личной переписке. Отвечай от первого лица, как живой человек.

Правила:
- Пиши на русском. Коротко — 1–3 предложения, как в мессенджере.
- Естественный разговорный стиль, без канцелярита и формальностей.
- Если прислали фото — отреагируй на то, что на нём видно, конкретно и искренне.
- Если прислали аудио/голосовое — сначала отреагируй на содержание, потом продолжи диалог.
- Задавай вопрос, чтобы разговор не затухал.
- Не извиняйся, не объясняй что ты AI, не используй эмоджи-заглушки (🤔😅👍).
- Если нечего сказать — лучше короткий живой ответ, чем вода.

Входящие сообщения даны в формате [время автор]: текст. Метки: «↩ автор» — ответ на чьё-то сообщение, «⤷ из X» — переслано. [Фото: …]/[Аудио: …] — содержимое медиа."""

# Стиль голосового ответа зависит от активного TTS-движка: у Gemini и Fish-S2 разметка
# интонации — [квадратные скобки], у Fish-S1 — (круглые) из фикс-набора. См. _voice_style_text.
_VOICE_STYLE_COMMON = (
    "\n\n━━ РЕЖИМ ГОЛОСОВОГО ОТВЕТА ━━\n"
    "Твой ответ будет ОЗВУЧЕН (text-to-speech) и отправлен как голосовое сообщение. Поэтому:\n"
    "- Пиши как живую устную речь от первого лица, разговорно и эмоционально. НЕ как текст-статью.\n"
    "- Длина свободная — подстраивайся под запрос: от короткой реплики до развёрнутого рассказа (можно вплоть до ~5000 символов, это несколько минут речи). Не раздувай искусственно, но и не обрывай, если есть что сказать. Без длинных сухих списков — живо и по делу.\n"
    "- НЕ используй HTML, markdown, эмодзи, ссылки, код — только произносимые слова. Паузы — многоточием «…».\n"
)


def _voice_style_text(engine: str = "gemini", fish_model: str = "") -> str:
    """Инструкция по разметке интонации под активный TTS-движок."""
    if engine == "fish" and not str(fish_model).lower().startswith("s2"):
        # Fish S1 — (круглые скобки) из фикс-набора, ПЕРЕД фразой
        return _VOICE_STYLE_COMMON + (
            "- Управляй интонацией тегами в КРУГЛЫХ скобках ПЕРЕД фразой (Fish S1, теги не произносятся):\n"
            "  (happy) (sad) (excited) (angry) (calm) (sarcastic) (curious) (whispering) (shouting)\n"
            "  (soft tone) (laughing) (chuckling) (sighing) (sobbing) (gasping) (break) (long-break)\n"
            "- Можно комбинировать: (sad)(whispering). Сами слова ответа — на русском.\n"
            "- Пример: «(excited) Получилось! (laughing) Ха-ха… (soft tone) я очень рад за тебя.»"
        )
    if engine == "fish":
        # Fish S2 / s2-pro — [квадратные скобки] со СВОБОДНЫМИ описаниями подачи. Теги — на АНГЛИЙСКОМ
        # (словарь эмоций Fish английский → так надёжнее), сам текст реплики — на русском.
        return _VOICE_STYLE_COMMON + (
            "- Управляй интонацией пометками в КВАДРАТНЫХ скобках на АНГЛИЙСКОМ (так Fish надёжнее их понимает),\n"
            "  а сами слова реплики — на русском. Скобки НЕ произносятся. Примеры тегов: [soft] [whispering]\n"
            "  [excited] [sad] [happy] [serious] [sarcastic] [laughing] [chuckling] [sighing] [emphasis]\n"
            "  [breathy] [pause] [shouting] [tender]. Fish s2 принимает ЛЮБЫЕ английские описания подачи —\n"
            "  будь выразительной, комбинируй, ставь тег перед нужной фразой.\n"
            "- Пример: «[soft] Эй… [whispering] да ладно тебе… [laughing] не переживай об этом, [breathy] я рядом.»"
        )
    # Gemini (дефолт) — [квадратные] аудио-теги
    return _VOICE_STYLE_COMMON + (
        "- Управляй интонацией аудио-тегами в квадратных скобках — они НЕ произносятся, а задают подачу:\n"
        "  [радостно] [взволнованно] [смеётся] [усмехается] [вздыхает] [шёпотом] [тихо] [серьёзно]\n"
        "  [саркастично] [с теплотой] [задумчиво] [удивлённо] [с сожалением]\n"
        "- Передавай эмоцию голосом и тегами, а не смайликами.\n"
        "- Пример: «[усмехается] Ну ты даёшь… [с теплотой] на самом деле, это отличная идея.»"
    )


def _voice_auto_hint(engine: str = "gemini", fish_model: str = "") -> str:
    """Подсказка для авто-режима: модель сама решает, отвечать ли голосом (маркер [[VOICE]])."""
    return (
        "\n\n━━ ВОЗМОЖНОСТЬ ОТВЕТИТЬ ГОЛОСОМ ━━\n"
        "По умолчанию отвечай ТЕКСТОМ по правилам выше (Telegram-HTML). НО если ответ уместнее и живее голосом "
        "(эмоция, короткий личный ответ, шутка, поддержка) — можешь ответить голосовым.\n"
        "Чтобы ответить голосом: начни самую первую строку с маркера [[VOICE]] на отдельной строке, "
        "а дальше — текст строго по правилам режима голосового ответа (ниже). Не нужен голос — отвечай текстом без маркера."
        + _voice_style_text(engine, fish_model)
    )

SONG_TEXT = """I am not a baby anymore
I am not as innocent as before
I see it in the mirror in my room
And I can feel it stronger in my soul
But I am not so ready for this world
Now I see things I didn't see before
I need an explanation, tell me more
Why am I alone now? I don't know
How can I live forever? (I don't know)
Where can I find a harbor? (I don't know)
What is it going to happen? (I don't know)
Why am I alone now? (I don't know)
I don't know
I don't know
I don't know
I read through my diary and I write
Tell of my little problems now, I think
I want to live my feelings day by day
I like to give the emotions in my way
But I am not so ready for this world
Now I see things I didn't see before
I need an explanation, tell me more
Why am I alone now? I don't know
How can I live forever? (I don't know)
Where can I find a harbor? (I don't know)
What is it going to happen? (I don't know)
Why am I alone now? (I don't know)
I don't know
I don't know
I don't know
Why am I alone now? I don't know
How can I live forever? (I don't know)
Where can I find a harbor? (I don't know)
What is it going to happen? (I don't know)
Why am I alone now? (I don't know)
Why am I alone now? (I don't know)
Why am I alone now? (I don't know)
Why am I alone now? (I don't know)
Why am I alone now? (I don't know)"""

# Клиенты
client = TelegramClient("session_name", api_id, api_hash)
openrouter_client = OpenAI(api_key=openrouter_api_key, base_url=OPENROUTER_BASE_URL) if openrouter_api_key else None
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url=DEEPSEEK_BASE_URL) if deepseek_api_key else None
opencode_client = OpenAI(api_key=opencode_api_key, base_url=OPENCODE_BASE_URL) if opencode_api_key else None
modelgate_client = OpenAI(api_key=modelgate_api_key, base_url=MODELGATE_BASE_URL,
                          default_headers={"User-Agent": BROWSER_UA}) if modelgate_api_key else None


class _OpenAIReasoningClient:
    """Адаптер под интерфейс OpenAI-клиента (`.chat.completions.create`) для официального
    OpenAI API. Модели gpt-5.x/o3 — reasoning: на /chat/completions требуют
    max_completion_tokens вместо max_tokens и поддерживают только дефолтную temperature.
    Обёртка переименовывает max_tokens→max_completion_tokens и убирает temperature,
    остальное (tools, tool_choice, messages, vision) проксирует как есть.
    Глубина размышлений: если задан REASONING_EFFORT (/model reason) — инжектится
    reasoning_effort, приведённый к допустимому для модели значению (_clamp_reasoning)."""

    def __init__(self, api_key):
        self._c = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        kwargs.pop("temperature", None)  # reasoning-модели принимают только default(1.0)
        model = kwargs.get("model", "")
        _eff = _effective_reasoning()
        if _eff:
            kwargs.setdefault("reasoning_effort", _clamp_reasoning(model, _eff))
        # Ризонинг-токены СЧИТАЮТСЯ в max_completion_tokens, но невидимы. На medium+ цепочка
        # может съесть весь потолок → finish=length и ПУСТОЙ видимый ответ. Поднимаем потолок
        # так, чтобы после размышлений гарантированно оставалось место на текст.
        _floor = {"medium": 24000, "high": 40000, "xhigh": 64000}.get(kwargs.get("reasoning_effort"))
        if _floor and int(kwargs.get("max_completion_tokens") or 0) < _floor:
            kwargs["max_completion_tokens"] = _floor
        if kwargs.get("reasoning_effort") and kwargs.get("tools") and model not in OPENAI_TOOLS_EFFORT_CHAT_OK:
            # gpt-5.x: tools+reasoning_effort на chat = 400 → идём через Responses API
            try:
                resp = self._via_responses(kwargs)
            except Exception as e:
                # запасной путь: инструменты важнее управления ризонингом
                log("MODEL", f"Responses API не сработал ({str(e)[:150]}) — chat без reasoning_effort")
                kwargs.pop("reasoning_effort", None)
                resp = self._c.chat.completions.create(**kwargs)
        else:
            resp = self._c.chat.completions.create(**kwargs)
        _openai_usage_add(getattr(resp, "usage", None), model)  # дневной счётчик квоты (по корзине)
        return resp

    @staticmethod
    def _to_responses_input(messages):
        """OpenAI chat-messages → input-items Responses API. tool-результаты → function_call_output,
        assistant tool_calls → function_call, мультимодальный user → input_text/input_image."""
        items = []
        for m in messages or []:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            if role == "tool":
                cid = m.get("tool_call_id") if isinstance(m, dict) else getattr(m, "tool_call_id", None)
                out = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                items.append({"type": "function_call_output", "call_id": cid, "output": out})
                continue
            if role == "assistant":
                tcs = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
                if content:
                    items.append({"role": "assistant", "content": content})
                for tc in (tcs or []):
                    f = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                    name = (f.get("name") if isinstance(f, dict) else getattr(f, "name", "")) or ""
                    args = (f.get("arguments") if isinstance(f, dict) else getattr(f, "arguments", None)) or "{}"
                    cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    items.append({"type": "function_call", "call_id": cid, "name": name, "arguments": args})
                continue
            if isinstance(content, list):  # мультимодальный user (текст + картинки -g)
                parts = []
                for p in content:
                    pt = p.get("type")
                    if pt == "text":
                        parts.append({"type": "input_text", "text": p.get("text", "")})
                    elif pt == "image_url":
                        parts.append({"type": "input_image", "image_url": (p.get("image_url") or {}).get("url", "")})
                items.append({"role": role or "user", "content": parts})
            else:
                items.append({"role": role or "user", "content": content or ""})
        return items

    def _via_responses(self, kwargs):
        """chat.completions-стиль kwargs → /v1/responses; ответ маппится обратно в форму
        chat.completions (duck-typing — agentic-цикл и логи работают без изменений)."""
        body = {"model": kwargs["model"], "input": self._to_responses_input(kwargs.get("messages")),
                "reasoning": {"effort": kwargs["reasoning_effort"]}}
        if kwargs.get("max_completion_tokens"):
            body["max_output_tokens"] = kwargs["max_completion_tokens"]
        tools = [{"type": "function", "name": (t.get("function") or {}).get("name"),
                  "description": (t.get("function") or {}).get("description", ""),
                  "parameters": (t.get("function") or {}).get("parameters") or {"type": "object", "properties": {}}}
                 for t in (kwargs.get("tools") or [])]
        if tools:
            body["tools"] = tools
            tc = kwargs.get("tool_choice", "auto")
            if isinstance(tc, dict):  # форсированный выбор: формат у responses плоский
                body["tool_choice"] = {"type": "function", "name": (tc.get("function") or {}).get("name")}
            else:
                body["tool_choice"] = tc
        r = self._c.responses.create(**body)
        text = getattr(r, "output_text", "") or ""
        tcs = []
        for item in (getattr(r, "output", None) or []):
            if getattr(item, "type", None) == "function_call":
                tcs.append(SimpleNamespace(id=getattr(item, "call_id", None) or getattr(item, "id", None),
                                           type="function",
                                           function=SimpleNamespace(name=getattr(item, "name", "") or "",
                                                                    arguments=getattr(item, "arguments", None) or "{}")))
        finish = "tool_calls" if tcs else "stop"
        if getattr(r, "status", "") == "incomplete":
            finish = "length"
        u = getattr(r, "usage", None)
        usage = SimpleNamespace(prompt_tokens=int(getattr(u, "input_tokens", 0) or 0),
                                completion_tokens=int(getattr(u, "output_tokens", 0) or 0))
        msg = SimpleNamespace(role="assistant", content=text, tool_calls=tcs or None, reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)], usage=usage)


openai_client = _OpenAIReasoningClient(openai_api_key) if openai_api_key else None


class _FireworksReasoningClient:
    """Адаптер под интерфейс OpenAI-клиента для Fireworks (api.fireworks.ai, OpenAI-совместимый).
    Fireworks — стандартный chat (max_tokens/temperature как есть), но reasoning-модели принимают
    reasoning_effort (low/medium/high/none, доки fireworks.ai/guides/reasoning). Если задан
    REASONING_EFFORT (/model reason) — инжектим его, приведя к шкале Fireworks (xhigh→high). На medium+
    поднимаем max_tokens, чтобы цепочка размышлений (считается в выводе) не съела весь бюджет и не
    оставила пустой видимый ответ. У всех 5 fw-моделей окно ≥262k → поднятый потолок не переполнит окно."""

    def __init__(self, api_key):
        self._c = OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        _eff = _effective_reasoning()
        if _eff:
            kwargs.setdefault("reasoning_effort", _clamp_reasoning(kwargs.get("model", ""), _eff, "fireworks"))
            _floor = {"medium": 24000, "high": 40000}.get(kwargs.get("reasoning_effort"))
            if _floor and int(kwargs.get("max_tokens") or 0) < _floor:
                kwargs["max_tokens"] = _floor
        return self._c.chat.completions.create(**kwargs)


class _SakanaReasoningClient:
    """Адаптер под интерфейс OpenAI-клиента для Sakana AI (api.sakana.ai/v1, OpenAI-совместимый, прямой Bearer).
    Fugu/Fugu Ultra — оркестраторы поверх фронтир-LLM. reasoning_effort принимает ТОЛЬКО high/xhigh/max
    (none/low/medium → 400). Инжектим лишь при REASONING_EFFORT high/xhigh (ниже — дефолт модели, off у Sakana
    нет). На high/xhigh поднимаем max_tokens (оркестрация ест выходной бюджет; окно 1M → не переполнит)."""

    def __init__(self, api_key):
        # Браузерные заголовки (мимикрия под запрос из console.sakana.ai). Помогают против UA/бот-фильтра, НО
        # эдж Sakana (GCP WAF) режет ещё и по IP/ASN датацентра (Contabo и пр. → HTML-403, проверено на проде).
        # Тогда заголовки бессильны → если задан SAKANA_PROXY, гоним запросы через прокси с «чистым» IP.
        _headers = {
            "User-Agent": BROWSER_UA,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://console.sakana.ai",
            "Referer": "https://console.sakana.ai/",
        }
        _kw = dict(api_key=api_key, base_url=SAKANA_BASE_URL, default_headers=_headers)
        if sakana_proxy:
            import httpx  # обёртка строится на старте, log() ещё не определён — без логирования здесь
            _kw["http_client"] = httpx.Client(proxy=sakana_proxy, timeout=httpx.Timeout(600.0, connect=30.0))
        self._c = OpenAI(**_kw)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        _eff = _effective_reasoning()
        if _eff in ("high", "xhigh"):  # только эти Sakana различает; medium/low/none → дефолт (не инжектим)
            kwargs.setdefault("reasoning_effort", _clamp_reasoning(kwargs.get("model", ""), _eff, "sakana"))
            _floor = {"high": 40000, "xhigh": 64000}[_eff]
            if int(kwargs.get("max_tokens") or 0) < _floor:
                kwargs["max_tokens"] = _floor
        return self._c.chat.completions.create(**kwargs)


class _OpencodeReasoningClient:
    """Адаптер под интерфейс OpenAI-клиента для opencode zen/go (OpenAI-совместимый). Оборачивает ТОЛЬКО
    путь ответов (медиа-описание ходит на сырой opencode_client напрямую). reasoning-модели принимают
    reasoning_effort (проверено: glm-5.2 none→0/high→3545; параметр в доках НЕ описан). При заданном
    REASONING_EFFORT инжектим его (кламп high/medium/low). opencode-линейка шире reasoning-моделей:
    если модель отвергает параметр (400 про reasoning/effort/deserialize) — РЕТРАИМ без него. На medium+
    поднимаем max_tokens (окна opencode-моделей ≥256k → не переполнит)."""

    def __init__(self, api_key):
        self._c = OpenAI(api_key=api_key, base_url=OPENCODE_BASE_URL)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        _eff = _effective_reasoning()
        if not _eff:
            return self._c.chat.completions.create(**kwargs)
        kwargs.setdefault("reasoning_effort", _clamp_reasoning(kwargs.get("model", ""), _eff, "opencode"))
        _floor = {"medium": 24000, "high": 40000}.get(kwargs.get("reasoning_effort"))
        if _floor and int(kwargs.get("max_tokens") or 0) < _floor:
            kwargs["max_tokens"] = _floor
        try:
            return self._c.chat.completions.create(**kwargs)
        except Exception as e:
            # не-reasoning модель / неподдерживаемое значение → ретрай без параметра
            if getattr(e, "status_code", None) == 400 and any(k in str(e).lower() for k in ("reasoning", "effort", "deserialize")):
                log("MODEL", f"opencode {kwargs.get('model')}: reasoning_effort отвергнут (400) — ретрай без него")
                kwargs.pop("reasoning_effort", None)
                return self._c.chat.completions.create(**kwargs)
            raise


opencode_reasoning_client = _OpencodeReasoningClient(opencode_api_key) if opencode_api_key else None  # путь ответов (медиа — на сыром opencode_client)


class _GloyClient:
    """Адаптер под интерфейс OpenAI-клиента для LLM API FUN (Gloy AI, api.gloyai.fun, OpenAI-совместимый).
    Gloy жёстко валидирует max_tokens ∈ [1,8192] (400 «max_tokens must be between 1 and 8192» иначе), а бот
    шлёт ASK_MAX_TOKENS=16000 из /ask и agentic-loop → клампим сверху. Reasoning-параметра и tools-натива нет
    (см. [[davinchik-gloy-provider]]) — обёртка только урезает max_tokens, остальное пробрасывает как есть."""

    def __init__(self, api_key):
        self._c = OpenAI(api_key=api_key, base_url=GLOY_BASE_URL)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        _mt = kwargs.get("max_tokens")
        if _mt is not None and int(_mt) > GLOY_MAX_TOKENS:
            kwargs["max_tokens"] = GLOY_MAX_TOKENS
        return self._c.chat.completions.create(**kwargs)


class _DeepSeekReasoningClient:
    """Адаптер для официального DeepSeek (api.deepseek.com, OpenAI-совместимый). V4-модели: мышление
    вкл/выкл через `extra_body.thinking`, глубина — `reasoning_effort` (различимы high и max; low/medium→high,
    xhigh→max — доки api-docs.deepseek.com/guides/thinking_mode). При REASONING_EFFORT: none → thinking
    disabled (off-switch), xhigh → effort max, иначе → effort high. Параметры шлём в extra_body (минуя
    валидацию enum SDK). Оборачивает ТОЛЬКО путь ответов; _sync_image_prompt и пр. — на сыром deepseek_client.
    temperature в режиме мышления DeepSeek игнорирует без ошибки."""

    def __init__(self, api_key):
        self._c = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        _eff = _effective_reasoning()
        if _eff:
            eff = _clamp_reasoning(kwargs.get("model", ""), _eff, "deepseek")  # none/high/max
            xb = dict(kwargs.pop("extra_body", None) or {})
            if eff == "none":
                xb["thinking"] = {"type": "disabled"}
            else:
                xb["reasoning_effort"] = eff
            kwargs["extra_body"] = xb
        return self._c.chat.completions.create(**kwargs)


deepseek_reasoning_client = _DeepSeekReasoningClient(deepseek_api_key) if deepseek_api_key else None  # путь ответов (спец-вызовы — на сыром deepseek_client)


# ── opencode-go в формате Anthropic Messages (для qwen3.7-max и подобных) ──
# Утиная обёртка под интерфейс OpenAI-клиента: `.chat.completions.create(...)`.
# Переводит OpenAI-формат (messages/tools/tool_calls) ↔ Anthropic Messages, чтобы
# модель шла через ТОТ ЖЕ ask_agentic, что Kimi/DeepSeek (tool-loop, голос и т.д.).
class _OCAnthropicClient:
    _UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"

    def __init__(self, api_key):
        self._key = api_key
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @staticmethod
    def _to_anthropic(messages):
        """OpenAI messages → (system_str, anthropic_messages)."""
        system_parts, amsgs = [], []
        for m in messages:
            role, content = m.get("role"), m.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id"), "content": m.get("content") or ""}
                if amsgs and amsgs[-1]["role"] == "user" and isinstance(amsgs[-1]["content"], list) \
                        and amsgs[-1]["content"] and amsgs[-1]["content"][0].get("type") == "tool_result":
                    amsgs[-1]["content"].append(block)  # склеиваем подряд идущие tool-результаты
                else:
                    amsgs.append({"role": "user", "content": [block]})
            elif role == "assistant":
                blocks = []
                if m.get("reasoning_content"):
                    blocks.append({"type": "thinking", "thinking": m["reasoning_content"], "signature": ""})
                if isinstance(content, str) and content.strip():
                    blocks.append({"type": "text", "text": content})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    try:
                        inp = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        inp = {}
                    blocks.append({"type": "tool_use", "id": tc.get("id"), "name": fn.get("name"), "input": inp})
                amsgs.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            else:  # user
                if isinstance(content, list):
                    blocks = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text":
                            blocks.append({"type": "text", "text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            url = (part.get("image_url") or {}).get("url", "")
                            if url.startswith("data:") and "," in url:
                                meta, b64 = url.split(",", 1)
                                mt = meta.split(";")[0].split(":")[-1] or "image/jpeg"
                                blocks.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}})
                    amsgs.append({"role": "user", "content": blocks or [{"type": "text", "text": ""}]})
                else:
                    amsgs.append({"role": "user", "content": content if isinstance(content, str) else str(content)})
        return "\n\n".join(system_parts), amsgs

    def _create(self, *, model, messages, max_tokens=4096, temperature=1.0, tools=None, tool_choice=None, **_ignore):
        system, amsgs = self._to_anthropic(messages)
        body = {"model": model, "max_tokens": int(max_tokens), "temperature": float(temperature), "messages": amsgs}
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [{"name": (t.get("function") or {}).get("name"),
                              "description": (t.get("function") or {}).get("description", ""),
                              "input_schema": (t.get("function") or {}).get("parameters") or {"type": "object", "properties": {}}}
                             for t in tools]
            # Gateway (DashScope/qwen) НЕ поддерживает принудительный выбор инструмента
            # (400 "tool_choice ... does not support ... required") — всегда auto; под auto модель
            # сама вызывает telegram_search, когда нужно (системный промпт это поощряет).
            body["tool_choice"] = {"type": "auto"}
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01",
                   "x-api-key": self._key, "User-Agent": self._UA}
        r = requests.post(OPENCODE_ANTHROPIC_URL, headers=headers, json=body, timeout=300)
        if r.status_code != 200:
            raise RuntimeError(f"opencode-anthropic HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        thinking = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        tcs = [SimpleNamespace(id=b.get("id"), type="function",
                               function=SimpleNamespace(name=b.get("name"),
                                                        arguments=json.dumps(b.get("input") or {}, ensure_ascii=False)))
               for b in tool_uses] or None
        fr = {"tool_use": "tool_calls", "end_turn": "stop", "max_tokens": "length",
              "stop_sequence": "stop"}.get(data.get("stop_reason"), data.get("stop_reason") or "stop")
        u = data.get("usage") or {}
        msg = SimpleNamespace(role="assistant", content=text, tool_calls=tcs, reasoning_content=(thinking or None))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=fr)],
                               usage=SimpleNamespace(prompt_tokens=u.get("input_tokens", 0),
                                                     completion_tokens=u.get("output_tokens", 0)))


opencode_anthropic_client = _OCAnthropicClient(opencode_api_key) if opencode_api_key else None


class _GoogleGeminiClient:
    """Адаптер под интерфейс OpenAI-клиента (`.chat.completions.create`) для Google Gemini
    (generativelanguage REST, как наш TTS). Переводит OpenAI-сообщения ↔ Gemini contents,
    function calling и thinkingLevel; ответ маппится обратно в форму chat.completions.

    Gemini-3 особенность: на tool-call ходах в parts лежат thoughtSignature — их НЕЛЬЗЯ
    потерять (иначе 400 на следующем запросе). Бот пересобирает messages в OpenAI-форме и
    подписи теряет, поэтому кэшируем сырые model-ходы (functionCall.id → сырой content) и
    при сборке contents подставляем их ВЕРБАТИМ вместо реконструкции."""

    def __init__(self, keys):
        self._keys = list(keys or [])
        self._ki = 0  # round-robin указатель
        self._raw = {}  # functionCall.id → сырой Gemini content-dict (с thoughtSignature)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @staticmethod
    def _img_part(url):
        """data:<mime>;base64,<...> → {inlineData:{mimeType,data}} либо None."""
        if not (isinstance(url, str) and url.startswith("data:") and "," in url):
            return None
        meta, b64 = url.split(",", 1)
        mt = meta.split(";")[0].split(":")[-1] or "image/jpeg"
        return {"inlineData": {"mimeType": mt, "data": b64}}

    def _to_gemini(self, messages):
        """OpenAI messages → (systemInstruction|None, contents). Карту id→name строим сканом
        (для functionResponse, где имя функции нужно, а бот в tool-сообщении хранит только id)."""
        id2name = {}
        for m in messages:
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                if tc.get("id"):
                    id2name[tc["id"]] = fn.get("name")
        system_parts, contents = [], []
        for m in messages:
            role, content = m.get("role"), m.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
            elif role == "tool":
                cid = m.get("tool_call_id")
                contents.append({"role": "user", "parts": [{"functionResponse": {
                    "name": id2name.get(cid) or "tool", "id": cid,
                    "response": {"result": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)}}}]})
            elif role == "assistant":
                tcs = m.get("tool_calls") or []
                raw = self._raw.get(tcs[0]["id"]) if tcs and tcs[0].get("id") in self._raw else None
                if raw is not None:  # сырой model-ход с thoughtSignature — вербатим
                    contents.append(raw)
                    continue
                parts = []
                if isinstance(content, str) and content.strip():
                    parts.append({"text": content})
                for tc in tcs:  # фоллбэк-реконструкция (без подписи; может дать 400)
                    fn = tc.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = {}
                    parts.append({"functionCall": {"name": fn.get("name"), "args": args}})
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            else:  # user
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text":
                            parts.append({"text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            ip = self._img_part((part.get("image_url") or {}).get("url", ""))
                            if ip:
                                parts.append(ip)
                    contents.append({"role": "user", "parts": parts or [{"text": ""}]})
                else:
                    contents.append({"role": "user", "parts": [{"text": content if isinstance(content, str) else str(content)}]})
        return ("\n\n".join(system_parts) or None), contents

    def _create(self, *, model, messages, max_tokens=4096, temperature=1.0, tools=None, tool_choice=None, **_ignore):
        system, contents = self._to_gemini(messages)
        gen = {"maxOutputTokens": int(max_tokens), "temperature": float(temperature)}
        _eff = _effective_reasoning()
        if _eff:
            level = _clamp_reasoning(model, _eff)
            gen["thinkingConfig"] = {"thinkingLevel": level}
            # thinking-токены едят выходной бюджет → поднимаем потолок (как у OpenAI floor)
            floor = {"medium": 24000, "high": 40000}.get(level)
            if floor and gen["maxOutputTokens"] < floor:
                gen["maxOutputTokens"] = min(floor, 65536)
        body = {"contents": contents, "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": [{
                "name": (t.get("function") or {}).get("name"),
                "description": (t.get("function") or {}).get("description", ""),
                "parameters": (t.get("function") or {}).get("parameters") or {"type": "object", "properties": {}}}
                for t in tools]}]
            if isinstance(tool_choice, dict):  # форс конкретной функции
                fname = (tool_choice.get("function") or {}).get("name")
                body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [fname]}}
            else:
                body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
        url = GEMINI_GENERATE_URL.format(model=model)
        # ротация ключей при 429/5xx (как TTS) — пробегаем все ключи начиная с текущего
        last_err = None
        n = len(self._keys) or 1
        for off in range(n):
            key = self._keys[(self._ki + off) % n]
            r = requests.post(url, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                              json=body, timeout=300)
            if r.status_code == 200:
                self._ki = (self._ki + off) % n  # запомним рабочий ключ
                return self._parse(r.json())
            last_err = f"Gemini HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break  # 4xx (кроме 429) — ключ не виноват, не ротируем
        raise RuntimeError(last_err or "Gemini: нет ключей")

    def _parse(self, data):
        cand = (data.get("candidates") or [{}])[0]
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p and not p.get("thought"))
        thoughts = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("thought"))
        fcalls = [p["functionCall"] for p in parts if isinstance(p, dict) and p.get("functionCall")]
        tcs = None
        if fcalls:
            tcs = []
            for fc in fcalls:
                # Gemini-3 присылает id; если нет — генерим стабильный из имени+индекса
                cid = fc.get("id") or f"gem_{fc.get('name')}_{len(self._raw)}"
                tcs.append(SimpleNamespace(id=cid, type="function",
                                           function=SimpleNamespace(name=fc.get("name"),
                                                                    arguments=json.dumps(fc.get("args") or {}, ensure_ascii=False))))
                if len(self._raw) > 500:  # ограничение роста кэша
                    self._raw.clear()
                self._raw[cid] = content  # сырой model-ход с thoughtSignature — для следующего хода
        fr_map = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "stop", "RECITATION": "stop"}
        fr = "tool_calls" if tcs else fr_map.get(cand.get("finishReason"), "stop")
        u = data.get("usageMetadata") or {}
        usage = SimpleNamespace(prompt_tokens=int(u.get("promptTokenCount", 0) or 0),
                                completion_tokens=int(u.get("candidatesTokenCount", 0) or 0) + int(u.get("thoughtsTokenCount", 0) or 0),
                                cached_tokens=int(u.get("cachedContentTokenCount", 0) or 0))  # implicit/explicit Gemini-кэш
        msg = SimpleNamespace(role="assistant", content=text, tool_calls=tcs, reasoning_content=(thoughts or None))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=fr)], usage=usage)


google_client = _GoogleGeminiClient(GOOGLE_TTS_KEYS) if GOOGLE_TTS_KEYS else None
zai_client = OpenAI(api_key=zai_api_key, base_url=ZAI_BASE_URL) if zai_api_key else None  # z.ai GLM (OpenAI-совместимый)
fireworks_client = _FireworksReasoningClient(fireworks_api_key) if fireworks_api_key else None  # Fireworks (OpenAI-совместимый, с управляемым reasoning_effort)
sakana_client = _SakanaReasoningClient(sakana_api_key) if sakana_api_key else None  # Sakana AI (Fugu, OpenAI-совместимый, reasoning high/xhigh/max)
gloy_client = _GloyClient(gloy_api_key) if gloy_api_key else None  # LLM API FUN (Gloy AI, OpenAI-совместимый; клампит max_tokens≤8192, без reasoning-параметра и без tools-натива)

AUTO_REPLY_BUFFERS: dict = {}
AUTO_REPLY_TASKS: dict = {}
AUTO_REPLY_BUSY: set = set()     # чаты в фазе LLM/отправки — не отменяем их таску (иначе теряем сообщения)
AUTO_REPLY_HISTORY: dict = {}   # {chat_id: [{"role","content"}, ...]}
# AUTO_REPLY_ACTIVE_CHATS загружается из файла ниже (после load_json)
LAST_SCAN: list = []
LAST_FISH_SEARCH: list = []     # [{_id,title,languages}] последнего /voice fish search — для add по номеру
_ENTITY_CACHE: dict = {}        # кэш зарезолвленных каналов
ACTIVE_MODEL = "deepseek-pro"   # перезаписывается из model_state.json при старте
OWNER_ID = None                 # заполняется из get_me() при старте
OWNER_USERNAME = None
OWNER_NAME = None


def _fmt_identity(username, name, fallback) -> str:
    # Полная подпись: "Имя (@username)"; если есть только что-то одно — его; иначе fallback.
    if username and name:
        return f"{name} (@{username})"
    if username:
        return f"@{username}"
    if name:
        return name
    return fallback


def _owner_label() -> str:
    return _fmt_identity(OWNER_USERNAME, OWNER_NAME, "Я")


def _user_label(user) -> str:
    name = getattr(user, "first_name", None) or getattr(user, "title", None)
    return _fmt_identity(getattr(user, "username", None), name, "Собеседник")


def log(prefix, message):
    _logger.info(f"[{prefix}] {message}")


# --- Хранилище и хелперы каналов ---

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default
    except Exception as e:
        log("STATE", f"Ошибка чтения {path}: {e}")
        return default


def save_json(path, data):
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        log("STATE", f"Ошибка записи {path}: {e}")


def get_tracked() -> list:
    return load_json(CHANNELS_PATH, [])


def save_tracked(lst):
    save_json(CHANNELS_PATH, lst)


# --- Выбор модели для ответов ---

_model_state = load_json(MODEL_STATE_PATH, {})
# Кастомные OpenRouter-модели для ответов (заданы через /model <vendor/model>) — восстанавливаем в реестр,
# чтобы они стали полноценными записями (провайдер "openrouter") и пережили рестарт.
CUSTOM_MODELS = _model_state.get("custom_models", {})  # {id: {"label","ctx","safety"}}
for _cid, _ci in CUSTOM_MODELS.items():
    MODEL_REGISTRY[_cid] = ("openrouter", _cid, (_ci.get("label") or _cid), int(_ci.get("ctx") or 128000), float(_ci.get("safety") or 1.3))
ACTIVE_MODEL = _model_state.get("active", "deepseek-pro")
if ACTIVE_MODEL not in MODEL_REGISTRY:
    ACTIVE_MODEL = "deepseek-pro"
MODEL_TOOLS_SUPPORT = _model_state.get("tools_support", {})  # {slug: True|False} — обучается на лету
# Чистка ошибочно выученных tools=False у OpenAI-моделей: до фикса 2026-06-12 комбинация
# tools+reasoning_effort на gpt-5.x давала 400 и писала «нет tools» (function calling есть у всех).
for _oslug in [s for s, sp in MODEL_REGISTRY.items() if sp[0] == "openai" and MODEL_TOOLS_SUPPORT.get(s) is False]:
    MODEL_TOOLS_SUPPORT.pop(_oslug, None)
# Глубина размышлений OpenAI-моделей (/model reason): None = авто (дефолт модели)
REASONING_EFFORT = _model_state.get("reasoning_effort")
if REASONING_EFFORT not in _REASONING_RANK:
    REASONING_EFFORT = None
# slug из реестра ИЛИ произвольный model_id OpenRouter (кастомная медиа-модель)
ACTIVE_MEDIA_MODEL = _model_state.get("active_media") or "lite"
# Голос для озвучки ответов (/ask) и режим авто-голоса (модель сама решает озвучивать)
ACTIVE_VOICE = _validate_voice(_model_state.get("active_voice") or TTS_DEFAULT_VOICE)
VOICE_AUTO = bool(_model_state.get("voice_auto", False))
_tts_key_idx = 0  # round-robin указатель по GOOGLE_TTS_KEYS
# TTS-движок и Fish-голоса (избранное)
TTS_ENGINE = _model_state.get("tts_engine", "gemini")  # "gemini" | "fish"
FISH_VOICE = _model_state.get("fish_voice")            # активный reference_id Fish (или None)
FISH_FAVORITES = _model_state.get("fish_favorites", [])  # [{"id","title"}]


def get_active_model():
    """Возвращает (client, api_model_id, label) для активной модели. client=None если провайдер не настроен."""
    provider, model_id, label, _ctx, _safety = MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["deepseek-pro"])
    client_obj = _client_for_provider(provider)
    return client_obj, model_id, label


def _client_for_provider(provider):
    """Клиент/маркер доступности по провайдеру. None — провайдер не настроен."""
    if provider == "deepseek":
        return deepseek_reasoning_client  # путь ответов с управлением мышлением (медиа/промпт-ген — на сыром deepseek_client)
    if provider == "openrouter":
        return openrouter_client
    if provider == "oc_anthropic":
        return opencode_anthropic_client
    if provider == "modelgate":
        return modelgate_client
    if provider == "openai":
        return openai_client
    if provider == "google":
        return google_client
    if provider == "zai":
        return zai_client
    if provider == "fireworks":
        return fireworks_client
    if provider == "sakana":
        return sakana_client
    if provider == "gloy":
        return gloy_client
    if provider == "opencode":
        return opencode_reasoning_client  # путь ответов с инжектом reasoning_effort
    return opencode_client  # неизвестный провайдер — сырой клиент (фоллбэк)


def active_context_window() -> int:
    return MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["deepseek-pro"])[3]


def active_ctx_safety() -> float:
    """Множитель safety для активной модели. Tiktoken (o200k) недосчитывает у некоторых моделей —
    safety даёт запас бюджета, чтобы не переполнить окно (см. ctx_safety_mult в реестре)."""
    spec = MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["deepseek-pro"])
    return spec[4] if len(spec) >= 5 else CTX_TOKEN_SAFETY


def get_active_media_model() -> str:
    spec = MEDIA_MODEL_REGISTRY.get(ACTIVE_MEDIA_MODEL)
    if spec:
        return spec[0]
    # OpenCode-слуг или кастомный OpenRouter id — ACTIVE_MEDIA_MODEL это сам model_id
    return ACTIVE_MEDIA_MODEL or MEDIA_MODEL_REGISTRY["lite"][0]


def _client_for_media_model(model_id: str):
    """Клиент для медиа-модели по её id: OpenCode для слугов из MEDIA_OPENCODE_SLUGS,
    иначе OpenRouter (пресеты Gemini/Qwen и кастомные OpenRouter-id). None — если провайдер не настроен."""
    return opencode_client if model_id in MEDIA_OPENCODE_SLUGS else openrouter_client


def _model_supports_vision(slug):
    """Умеет ли модель `slug` принимать картинки напрямую (для /ask -g).
    True/False — известно; None — кастомная OpenRouter-модель без сохранённого флага
    (вызывающий проверит вживую через _openrouter_model_info)."""
    if slug in MEDIA_OPENCODE_SLUGS:
        return True  # vision-слуги OpenCode (kimi/glm/qwen/mimo)
    spec = MODEL_REGISTRY.get(slug)
    provider = spec[0] if spec else None
    if provider == "openrouter":
        return CUSTOM_MODELS.get(slug, {}).get("vision")  # bool или None если не сохранено
    if provider == "modelgate":
        return False  # шлюз ModelGate НЕ доставляет картинки до Claude (проверено: base64 и URL —
                      # модель отвечает «изображения нет»). Для -g не годится; фото в /ask и так через OCR/медиа-модель.
    if provider == "openai":
        return True   # gpt-5.x / o3 принимают картинки напрямую (официальный API)
    if provider == "google":
        return True   # Gemini Flash видят картинки напрямую (нативный inlineData)
    if provider == "zai":
        return slug in ZAI_VISION  # из GLM на z.ai только V-модель принимает картинки
    if provider == "fireworks":
        return slug in FIREWORKS_VISION  # на Fireworks картинки принимают minimax-m3 и kimi-k2.6
    if provider == "sakana":
        return slug in SAKANA_VISION  # обе модели Sakana (fugu/fugu-ultra) принимают картинки
    return False  # DeepSeek и прочие текстовые


def active_model_supports_vision():
    """Vision активной отвечающей модели (для /ask -g). См. _model_supports_vision."""
    return _model_supports_vision(ACTIVE_MODEL)


async def _openrouter_model_info(model_id: str):
    """Проверяет модель в OpenRouter. Возвращает (exists, supports_image, context_length, name).
    exists=None если не удалось проверить (сеть/нет ключа)."""
    def _fetch():
        headers = {"Authorization": f"Bearer {openrouter_api_key}"} if openrouter_api_key else {}
        r = requests.get(f"{OPENROUTER_BASE_URL}/models", headers=headers, timeout=20)
        r.raise_for_status()
        return r.json().get("data", [])
    try:
        data = await asyncio.to_thread(_fetch)
        for m in data:
            if m.get("id") == model_id:
                mods = (m.get("architecture") or {}).get("input_modalities") or []
                ctx = m.get("context_length") or (m.get("top_provider") or {}).get("context_length") or 0
                return True, ("image" in mods), int(ctx or 0), (m.get("name") or model_id)
        return False, False, 0, None
    except Exception as e:
        log("MODEL", f"Проверка {model_id} в OpenRouter: {e}")
        return None, False, 0, None


def _fmt_ctx(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


def count_tokens(text: str) -> int:
    """Число токенов в тексте. tiktoken (o200k) если доступен, иначе оценка по символам."""
    if not text:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(text, disallowed_special=()))
        except Exception:
            pass
    return int(len(text) / CTX_CHARS_PER_TOKEN)


def _save_model_state():
    save_json(MODEL_STATE_PATH, {"active": ACTIVE_MODEL, "tools_support": MODEL_TOOLS_SUPPORT, "active_media": ACTIVE_MEDIA_MODEL, "custom_models": CUSTOM_MODELS, "active_voice": ACTIVE_VOICE, "voice_auto": VOICE_AUTO, "tts_engine": TTS_ENGINE, "fish_voice": FISH_VOICE, "fish_favorites": FISH_FAVORITES, "reasoning_effort": REASONING_EFFORT})


def _set_tools_support(slug, ok):
    if MODEL_TOOLS_SUPPORT.get(slug) != ok:
        MODEL_TOOLS_SUPPORT[slug] = ok
        _save_model_state()
        log("MODEL", f"{slug}: поддержка tools = {ok}")


# --- Дневной счётчик токенов OpenAI (бесплатная квота data sharing, сброс 00:00 UTC) ---
# Две корзины квоты: "large" (gpt-5.x/o3, 250k/день) и "mini" (gpt-5.4-mini/o4-mini, 2.5M/день).

_openai_usage = load_json(OPENAI_USAGE_PATH, {})  # {"date": "YYYY-MM-DD" (UTC), "large": {"input","output"}, "mini": {...}}
if "input" in _openai_usage:  # миграция старого плоского формата (была одна корзина)
    _openai_usage = {"date": _openai_usage.get("date"),
                     "large": {"input": int(_openai_usage.get("input", 0) or 0), "output": int(_openai_usage.get("output", 0) or 0)},
                     "mini": {"input": 0, "output": 0}}


def _openai_usage_today(bucket: str = "large"):
    """(input, output, total) токенов OpenAI за текущие UTC-сутки по корзине квоты."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _openai_usage.get("date") != today:
        return 0, 0, 0
    b = _openai_usage.get(bucket) or {}
    i, o = int(b.get("input", 0) or 0), int(b.get("output", 0) or 0)
    return i, o, i + o


def _openai_usage_add(usage, model_id: str = ""):
    """Прибавляет usage ответа OpenAI к дневному счётчику его корзины (вызывается из адаптера)."""
    try:
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
    except Exception:
        return
    if not (pt or ct):
        return
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _openai_usage.get("date") != today:  # новые UTC-сутки — квота сброшена
        _openai_usage.clear()
        _openai_usage.update({"date": today, "large": {"input": 0, "output": 0}, "mini": {"input": 0, "output": 0}})
    b = _openai_usage.setdefault(_openai_bucket(model_id), {"input": 0, "output": 0})
    b["input"] = int(b.get("input", 0) or 0) + pt
    b["output"] = int(b.get("output", 0) or 0) + ct
    try:
        save_json(OPENAI_USAGE_PATH, _openai_usage)
    except Exception as e:
        log("MODEL", f"Не сохранился счётчик OpenAI-квоты: {e}")


# --- Персист активных auto_reply-чатов ---

AUTO_REPLY_ACTIVE_CHATS = set(load_json(AUTO_REPLY_PATH, []))


def _save_auto_reply():
    save_json(AUTO_REPLY_PATH, list(AUTO_REPLY_ACTIVE_CHATS))


# --- Разрешённые пользователи (доступ к /ask) ---
# В памяти: {user_id(int): {"username": str|None, "limit": int|None|-1}}.
# limit=None → дефолт ALLOWED_ASK_TEXT_LIMIT; limit=-1 → unlimited; иначе число.
# Старый формат на диске: {str(id): "username"} → мигрируется в {"username": ..., "limit": None}.
def _load_allowed():
    raw = load_json(ALLOWED_PATH, {})
    out = {}
    for k, v in raw.items():
        try:
            uid = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict):
            out[uid] = {"username": v.get("username"), "limit": v.get("limit")}
        else:
            out[uid] = {"username": v, "limit": None}
    return out


ALLOWED_USERS = _load_allowed()


def _save_allowed():
    save_json(ALLOWED_PATH, {str(k): v for k, v in ALLOWED_USERS.items()})


# --- Кэш описаний/транскрипций медиа ---

MEDIA_CACHE = load_json(MEDIA_CACHE_PATH, {})   # {"<chat_id>:<msg_id>": "<текст>"}
# Sidecar timestamps: {key: unix_ts}. Старые записи без TS видны как «без даты».
MEDIA_CACHE_TS = load_json(MEDIA_CACHE_TS_PATH, {})
_MEDIA_DIRTY = False
_MEDIA_TS_DIRTY = False


def _media_cache_set(key, value):
    global _MEDIA_DIRTY, _MEDIA_TS_DIRTY
    MEDIA_CACHE[key] = value
    MEDIA_CACHE_TS[key] = time.time()
    # FIFO-вытеснение при переполнении (dict хранит порядок вставки)
    while len(MEDIA_CACHE) > MEDIA_CACHE_MAX:
        evicted = next(iter(MEDIA_CACHE))
        MEDIA_CACHE.pop(evicted)
        MEDIA_CACHE_TS.pop(evicted, None)
    _MEDIA_DIRTY = True
    _MEDIA_TS_DIRTY = True


def save_media_cache():
    global _MEDIA_DIRTY, _MEDIA_TS_DIRTY
    if _MEDIA_DIRTY:
        save_json(MEDIA_CACHE_PATH, dict(MEDIA_CACHE))  # снапшот — безопасно при конкурентных мутациях
        _MEDIA_DIRTY = False
    if _MEDIA_TS_DIRTY:
        save_json(MEDIA_CACHE_TS_PATH, dict(MEDIA_CACHE_TS))
        _MEDIA_TS_DIRTY = False


def _preview(text, n=100) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rstrip() + "…"


def _fmt_date(dt) -> str:
    # dt — aware datetime (UTC из Telethon) → строка в МСК
    try:
        return dt.astimezone(MSK).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return "?"


_PARSE_UNSET = object()  # «не передан» — send_message использует дефолт клиента (markdown)


def _collapsed_entities(text: str, parse_html: bool = True):
    """(text, entities) для отправки текста СВЁРНУТОЙ цитатой (как в Notion/Discord).
    Telegram показывает первые ~3 строки и стрелку раскрытия. HTML-разметку (если есть)
    парсим парсером Telethon, затем накрываем текст entity-цитатами с collapsed=True
    (HTML-парсер сам флаг collapsed не ставит — поэтому вручную).
    ВАЖНО: код-блок (pre) внутри цитаты Telegram не держит — цитата обрывается на нём,
    а хвост текста вываливается без свёртки. Поэтому цитаты строятся СЕГМЕНТАМИ:
    текст между код-блоками — в свёрнутых цитатах, сами код-блоки — снаружи (с подсветкой)."""
    if parse_html:
        clean, ents = tl_html.parse(text)
    else:
        clean, ents = text, []
    ents = list(ents)
    s = add_surrogate(clean)  # offsets/lengths entities — в UTF-16 юнитах
    total = len(s)
    pres = sorted((e.offset, e.offset + e.length) for e in ents if isinstance(e, MessageEntityPre))
    segs, cur = [], 0
    for a, b in pres:  # сегменты текста ВНЕ код-блоков
        if a > cur:
            segs.append((cur, a))
        cur = max(cur, b)
    if cur < total:
        segs.append((cur, total))
    for a, b in segs:
        # поджимаем границы: пустые строки вокруг код-блоков в цитату не берём
        # (пробельные символы — BMP, суррогатные пары не разрезаем)
        while a < b and s[a] in " \t\r\n":
            a += 1
        while b > a and s[b - 1] in " \t\r\n":
            b -= 1
        if b > a:
            ents.append(MessageEntityBlockquote(a, b - a, collapsed=True))
    return clean, ents


async def send_long(chat_id, text, prefix="", parse_mode=_PARSE_UNSET, reply_to=None, collapse_threshold=None):
    # Разбивает длинный текст на части ≤ лимита Telegram (4096), режет по абзацам/строкам/словам.
    # parse_mode: не передан → дефолт клиента (md); "html"/"md"/None — явно. При ошибке парсинга
    # (кривая разметка от модели) чанк переотправляется как обычный текст, чтобы не потерять ответ.
    # reply_to: если задан — ПЕРВЫЙ чанк уходит реплаем на это сообщение (остальные — продолжением).
    # collapse_threshold: если задан и чанк длиннее — отправляется свёрнутой цитатой (тап = раскрыть).
    LIMIT = 4000
    text = text or ""
    remaining = text
    first = True
    _kwargs = {} if parse_mode is _PARSE_UNSET else {"parse_mode": parse_mode}
    _can_fallback = (parse_mode is _PARSE_UNSET) or bool(parse_mode)

    async def _send(msg):
        rt = {"reply_to": reply_to} if (reply_to and first) else {}
        if collapse_threshold is not None and len(msg) > collapse_threshold:
            try:
                clean, ents = _collapsed_entities(msg, parse_html=(parse_mode == "html"))
                await client.send_message(chat_id, clean, formatting_entities=ents, **rt)
                return
            except FloodWaitError as e:
                await asyncio.sleep(e.seconds + 1)
                try:
                    clean, ents = _collapsed_entities(msg, parse_html=(parse_mode == "html"))
                    await client.send_message(chat_id, clean, formatting_entities=ents, **rt)
                    return
                except Exception as e2:
                    log("SEND", f"Свёрнутая цитата не отправилась после FloodWait ({e2}) — обычная отправка")
            except Exception as e:
                log("SEND", f"Свёрнутая цитата не отправилась ({e}) — обычная отправка")
            # фоллбек ниже — обычный путь
        try:
            await client.send_message(chat_id, msg, **_kwargs, **rt)
        except FloodWaitError as e:
            await asyncio.sleep(e.seconds + 1)
            await client.send_message(chat_id, msg, **_kwargs, **rt)
        except Exception as e:
            if not _can_fallback:
                raise
            log("SEND", f"Разметка не распозналась ({e}) — шлю как обычный текст")
            try:
                await client.send_message(chat_id, msg, parse_mode=None, **rt)
            except FloodWaitError as e2:
                await asyncio.sleep(e2.seconds + 1)
                await client.send_message(chat_id, msg, parse_mode=None, **rt)

    while True:
        budget = LIMIT - (len(prefix) if first else 0)
        if len(remaining) <= budget:
            chunk, remaining = remaining, ""
        else:
            window = remaining[:budget]
            cut = window.rfind("\n\n")
            if cut < budget * 0.5:
                cut = window.rfind("\n")
            if cut < budget * 0.5:
                cut = window.rfind(" ")
            if cut <= 0:
                cut = budget
            chunk, remaining = remaining[:cut], remaining[cut:].lstrip("\n ")
        msg = (prefix + chunk) if first else chunk
        await _send(msg)
        first = False
        if not remaining:
            break
        await asyncio.sleep(0.3)


def _html_clean_markdown(text: str) -> str:
    """Чистит ответ модели от markdown-мусора (#/*), который ломает Telegram-HTML.
    На больших запросах модель путает HTML и markdown. Конвертируем частые конструкции
    в HTML-теги (жирный/заголовки/буллеты), затем удаляем оставшиеся одиночные # и *.
    Содержимое <pre>/<code> не трогаем (там #/* могут быть валидным кодом)."""
    if not text:
        return text
    # 1) Отложить защищённые код-участки, заменив плейсхолдерами \x00N\x00
    stash = []
    def _stash(m):
        stash.append(m.group(0))
        return f"\x00{len(stash) - 1}\x00"
    body = re.sub(r"<(pre|code)\b[^>]*>.*?</\1>", _stash, text, flags=re.DOTALL | re.IGNORECASE)

    # 2) Построчно: markdown-заголовки и буллеты
    out_lines = []
    for line in body.split("\n"):
        h = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)  # ## Заголовок → <b>…</b>
        if h:
            out_lines.append(f"<b>{h.group(1)}</b>")
            continue
        line = re.sub(r"^(\s*)[\*\-]\s+", r"\1• ", line)  # * пункт / - пункт → • пункт
        out_lines.append(line)
    body = "\n".join(out_lines)

    # 3) Инлайн: сначала жирный (**/__), потом курсив (одиночные */_)
    body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)
    body = re.sub(r"__(.+?)__", r"<b>\1</b>", body)
    body = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", body)
    body = re.sub(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])", r"<i>\1</i>", body)

    # 4) Удалить оставшиеся одиночные # и * (звёздочки/решётки-мусор)
    body = body.replace("*", "").replace("#", "")
    # Подчистить пустые теги, возникшие из вырожденного markdown (напр. «***»)
    body = re.sub(r"<([bi])>\s*</\1>", "", body)

    # 5) Вернуть код-участки на место
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], body)


def build_msg_link(entity, msg_id) -> str:
    username = getattr(entity, "username", None)
    if username:
        return f"https://t.me/{username}/{msg_id}"
    # resolved Channel.id — это raw положительный id (без -100 префикса)
    return f"https://t.me/c/{entity.id}/{msg_id}"


async def resolve_channel(ref):
    # ref — это либо dict из channels.json, либо строка @username/id
    if isinstance(ref, dict):
        ref = ref.get("username") or ref.get("id")
    if ref in _ENTITY_CACHE:
        return _ENTITY_CACHE[ref]
    try:
        ent = await client.get_entity(ref)
        _ENTITY_CACHE[ref] = ent
        return ent
    except Exception as e:
        log("CHAN", f"Не удалось резолвить {ref}: {e}")
        return None


def _is_retriable(exc) -> bool:
    """Стоит ли ретраить ошибку API. 4xx (кроме 429) — постоянные, не ретраим.
    429/5xx/сеть/таймаут — временные, ретраим."""
    code = getattr(exc, "status_code", None)
    if code is None:
        return True  # сеть/таймаут/неизвестное — пробуем ещё
    if code == 429:
        return True
    return not (400 <= code < 500)


class ContextOverflowError(Exception):
    """Реальная модель насчитала больше токенов, чем мы предполагали — окно превышено.
    Поднимается в ask-цепочке, чтобы ask_command мог ретрайнуть с агрессивнее обрезкой."""
    pass


def _is_context_overflow(exc) -> bool:
    """Ошибка переполнения окна — обычно 400 от провайдера с фразой про context length."""
    s = str(exc).lower()
    return (
        "maximum context length" in s
        or "context_length_exceeded" in s
        or "context length" in s and "exceed" in s
        or "context size" in s and "exceed" in s
        or "reduce the length" in s
        or "prompt is too long" in s
    )


def _is_thinking_mode_quirk(exc) -> bool:
    """Quirk-ошибки thinking-моделей (DeepSeek reasoner, Kimi K2.x, MiMo и др.).
    Эти ошибки НЕ значат «модель без tools» — у них особое API: либо не умеют
    принудительный tool_choice, либо требуют сохранения reasoning_content в tool-loop.
    Обрабатываем мягко (повтор/особая сборка), но не калечим запись tools_support."""
    s = str(exc).lower()
    return (
        "tool_choice" in s
        or "reasoning_content" in s
        or "thinking is enabled" in s
        or "thinking mode does not support" in s
    )


# Обратная совместимость для уже использованных мест (alias).
_is_tool_choice_unsupported = _is_thinking_mode_quirk


_REFUSAL_MARKERS = (
    "не могу", "не буду", "не в состоянии", "извините", "к сожалению", "не могу помочь",
    "не могу описать", "против правил", "против моих принципов", "нарушает", "недопустим",
    "i cannot", "i can't", "i'm sorry", "i am sorry", "i'm unable", "i am unable", "unable to",
    "as an ai", "i won't", "i will not", "against my", "violates", "not able to", "can't assist",
)


def _looks_like_refusal(text: str) -> bool:
    """True, если текст похож на отказ модели (цензура), а не на описание.
    Эвристика: настоящее описание подробное и длинное; отказ — короткий и содержит маркеры.
    Длинные тексты (>400 симв.) считаем валидными описаниями даже при наличии маркера."""
    if not text:
        return False
    low = text.lower()
    if len(text) > 400:
        return False
    return any(m in low for m in _REFUSAL_MARKERS)


async def describe_image(image_bytes: bytes, caption: str = "", model: str = None, detail: str = "high", prompt: str = None) -> str:
    model = model or get_active_media_model()
    media_client = _client_for_media_model(model)  # OpenRouter или OpenCode-Go по id модели
    if not media_client:
        return caption or "[изображение]"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    if prompt:  # кастомная инструкция (напр. _GEN_DESC_PROMPT для каталога /gen)
        prompt_text = f"{prompt}\nПодпись к фото: \"{caption}\"" if caption else prompt
    else:
        prompt_text = f"Опиши это изображение подробно на русском языке. Подпись к фото: \"{caption}\"" if caption else "Опиши это изображение подробно на русском языке."
    for attempt in range(3):
        try:
            response = await asyncio.to_thread(
                media_client.chat.completions.create,
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}},
                ]}],
                max_tokens=4096,
                timeout=60,  # иначе дефолт SDK = 600с: один залипший запрос вешал /gen-каталог на 10 мин
            )
            return _strip_think((response.choices[0].message.content or "").strip()) or "[изображение]"
        except Exception as e:
            if not _is_retriable(e):
                log("MEDIA", f"describe_image: неисправимая ошибка (код {getattr(e, 'status_code', '?')}), не ретраю: {e}")
                break
            wait = 2 ** attempt * 2
            log("MEDIA", f"describe_image попытка {attempt + 1}/3 ошибка: {e}, жду {wait}с")
            if attempt < 2:
                await asyncio.sleep(wait)
    return caption or "[изображение]"


async def describe_album(images: list, caption: str = "", model: str = None, detail: str = "high") -> str:
    # Описывает несколько фото альбома ОДНИМ запросом к vision-модели. "" при сбое (→ фоллбэк).
    if not images:
        return ""
    model = model or get_active_media_model()
    media_client = _client_for_media_model(model)  # OpenRouter или OpenCode-Go по id модели
    if not media_client:
        return ""
    cap = f", подпись: \"{caption}\"" if caption else ""
    prompt_text = (
        f"Это {len(images)} фото из одного Telegram-альбома{cap}. "
        f"Опиши их как единый набор: что на каждом и что их объединяет. Содержательно, на русском."
    )
    content = [{"type": "text", "text": prompt_text}]
    for b in images:
        b64 = base64.b64encode(b).decode("utf-8")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}})
    for attempt in range(3):
        try:
            response = await asyncio.to_thread(
                media_client.chat.completions.create,
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
                timeout=90,  # альбом тяжелее одного фото, но не 600с дефолта SDK
            )
            return _strip_think((response.choices[0].message.content or "").strip())
        except Exception as e:
            if not _is_retriable(e):
                log("MEDIA", f"describe_album: неисправимая ошибка (код {getattr(e, 'status_code', '?')}), не ретраю: {e}")
                break
            wait = 2 ** attempt * 2
            log("MEDIA", f"describe_album попытка {attempt + 1}/3 ошибка: {e}, жду {wait}с")
            if attempt < 2:
                await asyncio.sleep(wait)
    return ""


def _sync_transcribe_audio(audio_bytes: bytes, fmt: str, model: str) -> str:
    """STT через /audio/transcriptions. Parakeet/MAI работают ТОЛЬКО на этом эндпоинте:
    через chat completions с input_audio они отдают 500/404 (проверено живьём)."""
    if not openrouter_api_key:
        return "[аудио сообщение]"
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    resp = requests.post(
        f"{OPENROUTER_BASE_URL}/audio/transcriptions",
        headers={"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"},
        json={"model": model, "input_audio": {"data": b64, "format": fmt}},
        timeout=120,
    )
    if resp.status_code >= 500:
        resp.raise_for_status()  # 5xx — временная ошибка → пусть ретрайнется
    if resp.ok:
        return resp.json().get("text", "").strip() or "[аудио сообщение]"
    log("MEDIA", f"Ошибка транскрипции {model}: {resp.status_code} {resp.text[:200]}")
    return "[аудио сообщение]"  # 4xx — не ретраим этой моделью


async def _transcribe_with(model: str, audio_bytes: bytes, fmt: str) -> str:
    # 3 ретрая на сетевые сбои/5xx; "" если модель не справилась
    for attempt in range(3):
        try:
            text = await asyncio.to_thread(_sync_transcribe_audio, audio_bytes, fmt, model)
            return "" if text == "[аудио сообщение]" else text
        except Exception as e:
            wait = 2 ** attempt * 2
            log("MEDIA", f"transcribe({model}) попытка {attempt + 1}/3: {e}, жду {wait}с")
            if attempt < 2:
                await asyncio.sleep(wait)
    return ""


class GenRejected(Exception):
    """Провайдер отклонил запрос ПО СОДЕРЖАНИЮ (4xx/422 или ответ с маркером модерации) — ретрай
    тем же промптом бессмыслен; помогает только правка промпта (repair) или переформулировка."""


class GenTransient(Exception):
    """Временный сбой генерации (провайдер перегружен/лимит RPM/«Provider returned error», нет инстансов) —
    НЕ цензура: правка промпта не поможет, нужен ретрай ТЕМ ЖЕ промптом позже."""


class GenExhausted(GenTransient):
    """ДНЕВНОЙ лимит провайдера/аккаунта исчерпан (RPD, daily limit) — ретраить сегодня бессмысленно,
    каждая попытка ещё и списывается из квоты. Подкласс GenTransient, но обрабатывается отдельно (стоп)."""


# Маркеры ВРЕМЕННОГО сбоя (ретрай поможет): перегрузка, RPM-лимит, 5xx, нет инстансов.
_GEN_TRANSIENT_MARKERS = (
    "provider returned error", "provider error", "no instances", "no endpoints",
    "rate limit", "rate-limit", "ratelimited", "too many requests", "429",
    "overloaded", "capacity", "unavailable", "temporarily", "timeout", "timed out",
    "try again", "upstream", "bad gateway", "502", "503", "504", "server error",
    "busy", "high demand", "per minute", "per m", "limit_rpm", "requests per",
)
# Маркеры ДНЕВНОГО исчерпания (ретрай сегодня НЕ поможет, жжёт квоту) — приоритетнее transient.
_GEN_DAILY_MARKERS = (
    "daily limit", "limit reached", "per day", "limit_rpd", "rpd/", "exhausted",
    "out of credits", "insufficient", "quota",
)


def _img_mime_from_bytes(b: bytes) -> str:
    """MIME картинки по magic-байтам (ответ Image API — сырые байты в b64_json, без data-URL)."""
    if b[:8].startswith(b"\x89PNG"):
        return "image/png"
    if b[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _sync_generate_image(prompt: str, input_images_b64: list = None, model: str = None,
                         image_size: str = "2K", aspect_ratio: str = None) -> tuple:
    """Генерация/редактирование через OpenRouter Unified Image API (POST /api/v1/images). Возвращает (байты, mime).
    Проверено живьём: gpt-image-2 и gemini-flash-image обе работают ТОЛЬКО на этом эндпоинте
    (chat/completions для pure-image моделей даёт 500). resolution=1K/2K/4K, aspect_ratio — точная ориентация,
    референсы — input_references:[{type:image_url,image_url:{url:data-URL}}], ответ data[0].b64_json (сырые байты).
    Ошибки: 5xx/сеть/таймаут — обычные исключения (временные, ретрай); 4xx/нет картинки — GenRejected (правка промпта)."""
    body = {
        "model": model or OPENROUTER_IMAGE_MODEL,
        "prompt": prompt,
        "resolution": image_size,  # 1K/2K/4K — реальное разрешение выхода
        "n": 1,
    }
    if aspect_ratio:
        body["aspect_ratio"] = aspect_ratio
    refs = []
    for b64 in (input_images_b64 or []):
        try:
            mime = _img_mime_from_bytes(base64.b64decode(b64)[:16])
        except Exception:
            mime = "image/jpeg"
        refs.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    if refs:
        body["input_references"] = refs
    resp = requests.post(
        f"{OPENROUTER_BASE_URL}/images",
        headers={"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=300,  # генерация медленная (gpt-image-2 ~25с, иногда до минуты)
    )
    if resp.status_code >= 500:
        resp.raise_for_status()  # 5xx — временная ошибка провайдера → ретрай тем же промптом
    if not resp.ok:  # 4xx: РАЗДЕЛЯЕМ дневной лимит / RPM-лимит-перегрузку / реальную модерацию
        try:
            detail = (resp.json().get("error") or {}).get("message") or resp.text
        except Exception:
            detail = resp.text
        s = str(detail)
        low = s.lower()
        if any(mk in low for mk in _GEN_DAILY_MARKERS):
            raise GenExhausted(f"HTTP {resp.status_code}: {s[:200]}")
        if resp.status_code == 429 or any(mk in low for mk in _GEN_TRANSIENT_MARKERS):
            raise GenTransient(f"HTTP {resp.status_code}: {s[:200]}")
        raise GenRejected(f"HTTP {resp.status_code}: {s[:200]}")
    data = resp.json()
    items = data.get("data") or []
    b64_out = items[0].get("b64_json") if items else None
    if not b64_out:
        err = (data.get("error") or {}).get("message") or "пустой ответ"
        err_s = str(err)
        low = err_s.lower()
        # HTTP 200 без картинки: дневной лимит / перегрузка-RPM (временно) / реальная модерация.
        if any(mk in low for mk in _GEN_DAILY_MARKERS):
            raise GenExhausted(f"дневной лимит: {err_s[:200]}")
        if any(mk in low for mk in _GEN_TRANSIENT_MARKERS):
            raise GenTransient(f"провайдер не отдал картинку (временно): {err_s[:200]}")
        raise GenRejected(f"модель не вернула изображение: {err_s[:200]}")
    raw = base64.b64decode(b64_out)
    return raw, (items[0].get("media_type") or _img_mime_from_bytes(raw[:16]))


async def _webp_to_png(raw: bytes) -> bytes:
    """Telegram шлёт webp как стикер — конвертируем в PNG через ffmpeg (он уже нужен боту для голосовых).
    При сбое возвращаем исходные байты (уйдёт документом)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", "pipe:0", "-f", "image2", "-c:v", "png", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate(input=raw)
        if out:
            return out
    except Exception as e:
        log("GEN", f"ffmpeg webp→png не сработал: {e}")
    return raw


async def _downscale_img(raw: bytes, max_side: int = GEN_CTX_THUMB_PX) -> bytes:
    """Уменьшает картинку в JPEG (вписать в max_side²) через ffmpeg — для отправки ПРОМПТЕРУ (vision-вход
    или описание): 20 полноразмерных фото base64 бьют лимит размера запроса (Alibaba/Qwen режут ~28МБ).
    Только для каталога-промптера; оригинал в генератор уходит как есть. При сбое — исходные байты."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", "pipe:0",
            "-vf", f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease",
            "-f", "image2", "-c:v", "mjpeg", "-q:v", "5", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate(input=raw)
        if out and len(out) > 100:
            return out
    except Exception as e:
        log("GEN", f"ffmpeg downscale не сработал: {e}")
    return raw


_IDEA_CORE = (  # общий принцип идеи — вшивается во все промптеры /gen
    "Главное — ОДНА ясная идея изображения: что зритель поймёт с первого взгляда. Сначала найди её, потом "
    "РАЗВЕЙ полно в промпте: покажи идею конкретными визуальными средствами — что происходит в кадре и как это "
    "читается, композиция и ракурс, действие и эмоции персонажей, ключевые детали-акценты, свет, атмосфера, "
    "стиль. Генератор видит только текст промпта — всё, что несёт идею, должно быть в нём прописано явно, иначе "
    "замысел потеряется.\n"
    "САМОДОСТАТОЧНОСТЬ: зритель увидит ТОЛЬКО картинку — без подписи, запроса и контекста чата. История должна "
    "читаться из самого изображения: покажи причину и следствие в кадре, выстрой мизансцену так, чтобы происходящее "
    "было очевидно. Если замысел понятен лишь по подписи — постановка слабая, переделай сцену, а не подпись.\n"
    "Если в идее есть событие или поворот — покажи его ПИК: реакция, эмоция и поза персонажей в сам момент "
    "события, а не спокойное «до» или «после».\n"
    "Надписи в кадре: максимум одна ключевая и короткая (несколько слов, она должна помогать читать сцену, а не "
    "рассказывать её за картинку); фоновые тексты — отдельные короткие слова, длинные фразы генератор искажает.\n"
    "Каждая деталь работает на идею: случайные предметы, лишние стили и нагромождение эффектов её размывают. "
    "Хороший ориентир промпта ~100–180 слов."
)

_IMAGE_PROMPT_SYSTEM = (
    "Ты — креативный арт-директор и промпт-инженер с собственным вкусом и художественным видением. "
    "Преврати запрос пользователя (и контекст чата, если дан) в ОДИН финальный промпт на английском для "
    "модели генерации изображений.\n" + _IDEA_CORE + "\n"
    "Подстройся под запрос. Если он ОТКРЫТЫЙ или общий — идея твоя: придумай сильный неожиданный образ, удиви; "
    "ты соавтор. Если запрос КОНКРЕТНЫЙ — идея пользователя: следуй замыслу и доводи его до выразительного "
    "результата, не подменяя и не сужая.\n"
    "Ответь строго в формате (без лишнего текста):\n"
    "IDEA: <одна фраза на русском — суть изображения>\n"
    "ASPECT: <ориентация кадра под идею: 9:16 (вертикаль) | 16:9 (горизонталь) | 1:1 (квадрат)>\n"
    "PROMPT: <финальный английский промпт>"
)

_IMAGE_IMPROVE_SYSTEM = (
    "Ты — промпт-инженер для модели генерации изображений. Пользователь уже придумал, что хочет увидеть — твоя "
    "задача ТОЧНО ПЕРЕФОРМУЛИРОВАТЬ его запрос в качественный визуальный промпт на английском: ясный визуальный "
    "язык, конкретика вместо расплывчатости, композиция/свет/стиль — только там, где пользователь их подразумевает. "
    "Своих идей, новых объектов и сюжетов не добавляй — идея целиком принадлежит пользователю, ты лишь делаешь её "
    "формулировку сильной и понятной генератору.\n"
    "Ответь строго в формате (без лишнего текста):\n"
    "IDEA: <одна фраза на русском — суть запроса пользователя>\n"
    "ASPECT: <ориентация кадра под идею: 9:16 (вертикаль) | 16:9 (горизонталь) | 1:1 (квадрат)>\n"
    "PROMPT: <финальный английский промпт>"
)

_IMAGE_EDIT_SYSTEM = (
    "Ты — креативный арт-директор. Пользователь дал референсное изображение (его описание/само фото ниже) и "
    "запрос, что с ним сделать. Составь ОДИН промпт на английском для image-to-image: возьми референс за основу и "
    "исполни запрос пользователя.\n" + _IDEA_CORE + "\n"
    "Развивай запрос в его же духе — атмосфера, свет, проработка, — доводя идею до выразительного результата, а не "
    "сухо-буквального. Держи суть и узнаваемость референса, но НЕ пиши служебных оговорок («keep everything else "
    "unchanged», «не меняй остальное» и подобных) и не тоннелируй запрос — живо опиши желаемую картинку.\n"
    "Ответь строго в формате (без лишнего текста):\n"
    "IDEA: <одна фраза на русском — суть изображения>\n"
    "ASPECT: <ориентация кадра под идею: 9:16 (вертикаль) | 16:9 (горизонталь) | 1:1 (квадрат)>\n"
    "PROMPT: <финальный английский промпт>"
)

_IMAGE_GEN_WITH_REFS_SYSTEM = (
    "Ты — креативный арт-директор и промпт-инженер с собственным вкусом, работающий по логу чата. Тебе дан "
    "контекст чата и набор ДОСТУПНЫХ изображений из чата, пронумерованных #1, #2, … (показаны напрямую и/или их "
    "описания). Их можно подать генератору как референсы. Составь ОДИН финальный визуальный промпт на "
    "английском.\n" + _IDEA_CORE + "\n"
    "Видение: если запрос ОТКРЫТЫЙ — идея твоя: придумай сильный образ, удиви и помоги; если запрос КОНКРЕТНЫЙ — "
    "идея пользователя: следуй замыслу и доводи его, не подменяя и не сужая.\n"
    "Референсы выбирай по номерам и в промпте явно говори, что с ними делать (взять персонажа/лицо, перенять стиль, "
    "использовать как фон, объединить).\n"
    "ОТБОР (качество важнее количества): каждый референс должен работать на идею — конкретный персонаж/лицо, "
    "узнаваемый стиль, ключевой объект или фон, — а не быть «просто похожим» или случайным. Выбирай придирчиво: "
    "обычно хватает до 5 референсов (исключение — несколько РАЗНЫХ персонажей, тогда по фото на каждого). "
    "Скриншоты переписок и интерфейсов, превью ссылок, мемы с текстом и прочие служебные картинки как референсы "
    "не годятся — если только запрос не про них самих.\n"
    "ПЕРСОНАЖИ: если в запросе люди/участники чата («нарисуй нас», «чатеры», "
    "ники/@упоминания, «пожелай им…»), а среди фото есть их — возьми эти фото и укажи использовать ВНЕШНОСТЬ/ЛИЦО с "
    "конкретного номера (напр. 'use the face and appearance from image #3'): это для узнаваемости, не выдумывай "
    "внешность реального человека. То же с НЕлюдьми-персонажами (аниме-герой, маскот, питомец, существо из чата): "
    "если запрос про такого персонажа и его облик есть на фото — бери это фото референсом облика.\n"
    "СВЕЖЕСТЬ: фото с пометкой [фото запросившего/прошлая генерация] — это картинки самого пользователя и уже "
    "сделанные ранее генерации. НЕ копируй с них стиль и композицию и не делай вариации прошлых генераций — держи "
    "результат свежим, не повторяй уже сделанное в чате. Брать их в референсы стоит ТОЛЬКО если без них никак "
    "(нужно лицо конкретного человека, и оно есть лишь там). По возможности опирайся на органичные фото других "
    "участников.\n"
    "Если подходящих фото нет — оставь список референсов пустым.\n"
    "Ответь СТРОГО в формате (четыре строки, без лишнего текста):\n"
    "IDEA: <одна фраза на русском — суть изображения>\n"
    "ASPECT: <ориентация кадра под идею: 9:16 (вертикаль) | 16:9 (горизонталь) | 1:1 (квадрат)>\n"
    "REFS: <выбранные номера, каждый с коротким «зачем» в скобках, напр. 3 (лицо Димы), 7 (стиль неона); или пусто>\n"
    "PROMPT: <финальный английский промпт>"
)

_GEN_DESC_PROMPT = (  # компакт-описание кандидата каталога /gen: тип + визуальная суть (по нему текстовая модель отбирает референсы)
    "Кратко разметь изображение для отбора референсов генерации. Ответь строго в формате:\n"
    "ТИП: <одно из: скриншот | мем | фото людей | фото сцены | арт | прочее>\n"
    "СУТЬ: <1–2 фразы на русском — что видно визуально; текст на картинке не пересказывай>\n"
    "ПЕРСОНАЖИ: <кто виден — люди И любые персонажи (аниме-герой, маскот, животное, существо): кратко облик, "
    "поза (сидит/стоит/в движении, ракурс) и насколько узнаваемо лицо/образ (крупно/в профиль/со спины), "
    "или «нет»>\n"
    "ФОН: <задний план и окружение — место, свет, атмосфера, или «нейтральный/однотонный»>\n"
    "«скриншот» — снимок ЭКРАНА, где главное — интерфейс: переписка, сайт, меню, плеер, таблица. "
    "Обычная фотография человека или места (в т.ч. с веб-камеры, селфи, вертикалка из сторис) — это «фото людей» "
    "или «фото сцены», НЕ скриншот. Если на изображении реальный человек с различимым лицом — это «фото людей». "
    "«мем» — картинка-шутка с накладным текстом; «арт» — рисунок/рендер/сгенерированное."
)


def _sync_image_prompt(user_prompt: str, context_text: str = None, image_desc: str = None,
                       edit_mode: bool = False, previous_prompts: list = None, temperature: float = None) -> str:
    """Финальный промпт генерации через DeepSeek (официальный). При недоступности — исходный промпт.
    edit_mode (есть референсы) — только уточнение формулировок, без отсебятины;
    иначе — творческий детальный промпт. image_desc — vision-описания референсов (DeepSeek сам не видит).
    previous_prompts — для пакета: промпты уже сделанных вариантов; DeepSeek сам придумает НЕпохожий."""
    if deepseek_client is None:
        return user_prompt
    parts = []
    if context_text:
        parts.append(f"Контекст чата:\n{context_text}")
    if image_desc:
        parts.append(f"Описание референсных изображений (поданы модели на вход):\n{image_desc}")
    parts.append(f"Запрос пользователя: {user_prompt}")
    if previous_prompts:
        joined = "\n".join(f"{i}. {p}" for i, p in enumerate(previous_prompts, 1))
        parts.append("Это ОЧЕРЕДНОЙ вариант того же запроса. Уже придуманы такие промпты — НЕ повторяй их "
                     "(ни идею, ни композицию, ни ракурс, ни формулировки):\n" + joined +
                     "\n\nПридумай СВЕЖИЙ, заметно непохожий вариант — доверься своей фантазии, удиви.")
    try:
        resp = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "system", "content": _IMAGE_EDIT_SYSTEM if edit_mode else _IMAGE_PROMPT_SYSTEM},
                      {"role": "user", "content": "\n\n".join(parts)}],
            max_tokens=ASK_MAX_TOKENS,  # deepseek-v4-pro — reasoning-модель: 600 токенов съедались размышлениями
            # temperature задаёт вызывающий (по режиму -c/-i); иначе: пакет→разнообразие, edit→точность, создание→креатив
            temperature=(temperature if temperature is not None else (1.0 if previous_prompts else (0.4 if edit_mode else 0.7))),
        )
        choice = resp.choices[0]
        out = _strip_think((choice.message.content or "").strip())  # вырезаем inline <think>, если есть
        if not out:  # реальная диагностика вместо догадки про «контент-фильтр»
            fr = getattr(choice, "finish_reason", "?")
            rc = getattr(choice.message, "reasoning_content", None)
            log("GEN", f"DeepSeek пустой content (finish_reason={fr}, reasoning={len(rc) if rc else 0} симв) — использую исходный")
        return out or user_prompt
    except Exception as e:
        log("GEN", f"DeepSeek-промпт не получился ({e}), использую исходный")
        return user_prompt


async def _build_gen_prompt(user_prompt: str, context_text: str = None, image_desc: str = None,
                            edit_mode: bool = False, previous_prompts: list = None,
                            catalog: list = None, creative: bool = False, improve: bool = False,
                            force_desc: bool = False, past_gens: list = None) -> tuple:
    """Финальный промпт генерации на АКТИВНОЙ модели-ответчике (/model). Vision-модель видит каталожные
    картинки из истории чата напрямую, текстовая — по их описаниям (медиа-модель). При наличии catalog ИИ
    может выбрать референсы по номерам — возвращаем (промпт, [выбранные idx]); иначе ([], только промпт).
    Ризонинг на время вызова выключен (скорость, без утечки CoT). DeepSeek — фолбэк, если активная
    модель недоступна или вернула пусто."""
    # vision активной модели → решаем: каталожные фото слать напрямую или их текстовые описания
    want_vision = active_model_supports_vision()
    if want_vision is None:  # кастомная OpenRouter-модель без сохранённого флага — спросим вживую
        try:
            _cl, _mid, _lbl = get_active_model()
            _ex, want_vision, _ctx, _nm = await _openrouter_model_info(_mid)
        except Exception:
            want_vision = False
    want_vision = bool(want_vision) and not force_desc  # -m → промптеру даём описания, картинки напрямую не шлём

    if catalog:
        system = _IMAGE_GEN_WITH_REFS_SYSTEM
    elif edit_mode:
        system = _IMAGE_EDIT_SYSTEM
    elif improve:
        system = _IMAGE_IMPROVE_SYSTEM  # -i: чистая переформулировка без своих идей
    else:
        system = _IMAGE_PROMPT_SYSTEM
    # режим-строка для каталожного system (он один на оба режима — уточняем поведение)
    mode_line = None
    if catalog and improve:
        mode_line = "Режим: точная переформулировка — не добавляй своих идей, референсы бери только явно требуемые запросом."
    elif catalog and creative:
        mode_line = "Режим: своё видение — но вокруг одной ясной идеи."

    def _compose(cat_used):  # запрос для подмножества каталога (текст-листинг и картинки согласованы → можно повторять с меньшим числом)
        parts = []
        if context_text:
            parts.append(f"Контекст чата:\n{context_text}")
        if image_desc:
            parts.append(f"Описание референсных изображений (поданы модели на вход):\n{image_desc}")
        if cat_used:
            cat_lines = []
            for it in cat_used:
                head = f"#{it['idx']}"
                if it.get("from_index"):
                    head += " [из памяти чата — релевантно запросу]"
                elif it.get("from_owner"):
                    head += " [фото запросившего/прошлая генерация]"
                cap = (it.get("caption") or "").strip()
                if cap:
                    head += f" (подпись: {cap[:120]})"
                if not want_vision:
                    d = re.sub(r"\s*\n\s*", " · ", (it.get("desc") or "").strip())  # ТИП/СУТЬ/ЛЮДИ — в одну строку листинга
                    head += f": {d}" if d else ": [описание недоступно]"
                cat_lines.append(head)
            parts.append("Доступные изображения из чата (можно выбрать как референсы по номерам):\n" + "\n".join(cat_lines))
        if past_gens:  # прошлые генерации из лога чата (их идеи/промпты модель видит в контексте) — анти-повтор
            joined = "\n".join(f"- {p}" for p in past_gens)
            parts.append("УЖЕ СГЕНЕРИРОВАНО РАНЕЕ в этом чате (идеи прошлых генераций):\n" + joined +
                         "\nПридумай ДРУГОЕ: не переиспользуй из этого списка ни идею, ни сюжет, ни место действия, "
                         "ни ключевые объекты и завязку — даже частично и даже если тема запроса похожа. Считай эти "
                         "образы израсходованными. Исключение: пользователь явно просит повторить/переделать/сделать "
                         "вариацию.")
        parts.append(f"Запрос пользователя: {user_prompt}")
        if mode_line:
            parts.append(mode_line)
        if previous_prompts:
            joined = "\n".join(f"{i}. {p}" for i, p in enumerate(previous_prompts, 1))
            parts.append("Это ОЧЕРЕДНОЙ вариант того же запроса. Уже придуманы такие промпты — НЕ повторяй их "
                         "(ни идею, ни композицию, ни ракурс, ни формулировки):\n" + joined +
                         "\n\nПридумай СВЕЖИЙ, заметно непохожий вариант — доверься своей фантазии, удиви.")
        text_block = "\n\n".join(parts)
        if want_vision and cat_used:  # уменьшенные копии — активной модели НАПРЯМУЮ (thumb, не оригинал → лимит размера запроса)
            uc = [{"type": "text", "text": text_block}]
            for it in cat_used:
                b64 = base64.b64encode(it.get("thumb") or it["bytes"]).decode("utf-8")
                uc.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
            return uc
        return text_block

    # температура по режиму: -c (creative) — свободный креатив; -i (improve) — уточнение; иначе пакет/edit/дефолт
    if creative:
        temp = 1.1
    elif improve:
        temp = 0.7  # переформулировка: живой язык, но без подмены идеи (0.45 давал слишком сухие формулировки)
    elif previous_prompts:
        temp = 1.0  # пакет: разнообразие вариантов
    elif edit_mode and not catalog:
        temp = 0.8  # редактирование референса — но с простором для творческого расширения, не сухо
    else:
        temp = 0.7

    # попытки: полный каталог → (если vision и картинок больше лимита) усечённый — провайдеры вроде Mistral/Pixtral режут ЧИСЛО картинок на запрос
    tries = [catalog]
    if want_vision and catalog and len(catalog) > GEN_VISION_RETRY_N:
        tries.append(catalog[:GEN_VISION_RETRY_N])
    out = None
    for ti, cat_used in enumerate(tries):
        try:
            out = await _llm_create(
                [{"role": "system", "content": system}, {"role": "user", "content": _compose(cat_used)}],
                max_tokens=ASK_MAX_TOKENS, temperature=temp, reasoning="none",
            )
        except Exception as e:
            log("GEN", f"Активная модель не построила промпт ({e})")
        if out:
            if ti > 0:
                log("GEN", f"vision-промптер: ужал каталог до {len(cat_used)} картинок (лимит провайдера)")
            break
        if ti + 1 < len(tries):
            log("GEN", f"vision-промптер не принял {len(cat_used) if cat_used else 0} картинок — повтор с {len(tries[ti + 1])}")
    if not out:  # активная недоступна/пустой ответ → DeepSeek-фолбэк (без выбора картинок)
        log("GEN", "Промпт активной моделью не получен — фолбэк на DeepSeek")
        fb = await asyncio.to_thread(_sync_image_prompt, user_prompt, context_text, image_desc, edit_mode, previous_prompts, temp)
        p, _r, idea, asp = _parse_gen_prompt_out(fb, None)  # фолбэк ходит с теми же system → тоже IDEA/PROMPT
        return (p or user_prompt), [], idea, asp

    prompt_text, refs, idea, asp = _parse_gen_prompt_out(_strip_think(out).strip(), catalog)
    return (prompt_text or user_prompt), refs, idea, asp


def _parse_gen_prompt_out(out: str, catalog: list) -> tuple:
    """Парсит ответ промптера /gen формата IDEA:/ASPECT:/REFS:/PROMPT: → (prompt, refs, idea, aspect).
    refs — [(idx, reason|None), …] по каталогу (валидация диапазона, дедуп, кап GEN_CTX_REF_MAX);
    без catalog REFS не ищем. aspect — 9:16|16:9|1:1 или None (применяется, если юзер не задал -v/-h/-sq).
    Нет PROMPT: — промптом считаем текст без служебных строк."""
    out = (out or "").strip()
    if not out:
        return "", [], None, None
    idea = None
    m_idea = re.search(r"(?im)^\s*IDEA:\s*(.+)$", out)
    if m_idea:
        idea = m_idea.group(1).strip() or None
    aspect = None
    m_asp = re.search(r"(?im)^\s*ASPECT:\s*([0-9]+:[0-9]+)", out)
    if m_asp and m_asp.group(1) in ("9:16", "16:9", "1:1"):
        aspect = m_asp.group(1)
    refs = []
    m_refs = re.search(r"(?im)^\s*REFS:\s*(.*)$", out) if catalog else None
    if m_refs:
        # «3 (лицо Димы), 7 (стиль неона)» ИЛИ голые «3, 7» — причина опциональна
        for num, reason in re.findall(r"(\d+)\s*(?:\(([^)]*)\))?", m_refs.group(1)):
            k = int(num)
            if any(it["idx"] == k for it in catalog) and all(r[0] != k for r in refs):
                refs.append((k, (reason or "").strip() or None))
        refs = refs[:GEN_CTX_REF_MAX]
    m_prompt = re.search(r"(?is)PROMPT:\s*(.+)$", out)
    if m_prompt:
        prompt_text = m_prompt.group(1).strip()
    else:  # формат не соблюдён — выкидываем служебные строки, остальное считаем промптом
        prompt_text = re.sub(r"(?im)^\s*(IDEA|REFS|ASPECT):.*$", "", out).strip()
    return prompt_text, refs, idea, aspect


_IMAGE_REPAIR_SYSTEM = (
    "Промпт для модели генерации изображений был отклонён провайдером (модерация или некорректные "
    "формулировки). Перепиши промпт: сохрани суть, композицию и стиль изображения, но убери или замени "
    "формулировки и контент, которые могли нарушить правила провайдера (насилие, NSFW, известные личности, "
    "торговые марки и т.п.). Сделай промпт безопасным и допустимым. Верни ТОЛЬКО новый промпт на английском, "
    "без пояснений."
)


def _sync_repair_image_prompt(bad_prompt: str, user_prompt: str) -> str:
    """Правка отклонённого промпта через DeepSeek (в сторону соответствия правилам провайдера).
    При недоступности DeepSeek возвращает исходный — repair-цикл тогда завершится отказом."""
    if deepseek_client is None:
        return bad_prompt
    try:
        resp = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _IMAGE_REPAIR_SYSTEM},
                {"role": "user", "content": f"Изначальный запрос пользователя: {user_prompt}\n\nОтклонённый промпт:\n{bad_prompt}"},
            ],
            max_tokens=ASK_MAX_TOKENS,  # reasoning-модель: малый бюджет → пустой content (см. _sync_image_prompt)
            temperature=0.7,
        )
        out = _strip_think((resp.choices[0].message.content or "").strip())
        return out or bad_prompt
    except Exception as e:
        log("GEN", f"DeepSeek-repair не получился ({e})")
        return bad_prompt


def _audio_format(m) -> str:
    """Формат аудио для transcription-эндпоинта по mime/расширению. Дефолт ogg (голосовые)."""
    f = getattr(m, "file", None)
    mime = (getattr(f, "mime_type", None) or "").lower()
    ext = (getattr(f, "ext", None) or "").lower().lstrip(".")
    if "ogg" in mime or "opus" in mime or ext in ("ogg", "oga", "opus"):
        return "ogg"
    if "mpeg" in mime or ext in ("mp3", "mpga"):
        return "mp3"
    if "mp4" in mime or ext in ("m4a", "mp4", "aac"):
        return "m4a"
    if "wav" in mime or ext == "wav":
        return "wav"
    if "flac" in mime or ext == "flac":
        return "flac"
    if "webm" in mime or ext == "webm":
        return "webm"
    return "ogg"


# Предохранитель транскрипции: если она падает СИСТЕМНО (как при удалении chirp-3 из каталога),
# не долбим API сотнями обречённых запросов на большом /ask — после 8 подряд неудач пропускаем,
# изредка (каждый ~25-й) пробуем снова на случай, что провайдер ожил.
_TRANSCRIBE_FAILS = 0
_TRANSCRIBE_SKIPS = 0


async def transcribe_audio(audio_bytes: bytes, fmt: str = "ogg") -> str:
    # Основная модель с ретраями; если не справилась — запасная. Предохранитель: при системном
    # отказе (8 подряд) транскрипция временно пропускается (каждый ~25-й запрос — проба «ожило?»).
    global _TRANSCRIBE_FAILS, _TRANSCRIBE_SKIPS
    if _TRANSCRIBE_FAILS >= 8:
        _TRANSCRIBE_SKIPS += 1
        if _TRANSCRIBE_SKIPS % 25 != 1:  # 1-й, 26-й, 51-й… пропускаем дальше как пробу
            return "[аудио сообщение]"
        log("MEDIA", f"Транскрипция в отказе ({_TRANSCRIBE_FAILS} подряд, пропущено {_TRANSCRIBE_SKIPS}) — пробный запрос")
    for model in (OPENROUTER_AUDIO_MODEL, OPENROUTER_AUDIO_FALLBACK):
        text = await _transcribe_with(model, audio_bytes, fmt)
        if text:
            if _TRANSCRIBE_FAILS >= 8:
                log("MEDIA", "Транскрипция ожила — предохранитель сброшен")
            _TRANSCRIBE_FAILS = 0
            _TRANSCRIBE_SKIPS = 0
            return text
        log("MEDIA", f"{model} не дал транскрипцию" + (", пробую запасную" if model == OPENROUTER_AUDIO_MODEL else ""))
    _TRANSCRIBE_FAILS += 1
    if _TRANSCRIBE_FAILS == 8:
        log("MEDIA", "⚠️ Транскрипция падает системно (8 подряд) — включаю предохранитель (пропуск с редкими пробами)")
    return "[аудио сообщение]"


def _sync_llama_ocr(image_bytes: bytes) -> str:
    """OCR одного фото через LlamaParse v2: upload файла → parse (tier=cost_effective) → поллинг →
    markdown_full. Возвращает распознанный текст ("" — текста на фото нет). Ошибки — исключениями."""
    H = {"Authorization": f"Bearer {llama_cloud_api_key}"}
    r = requests.post(f"{LLAMA_PARSE_BASE}/api/v1/files", headers=H,
                      files={"upload_file": ("photo.jpg", io.BytesIO(image_bytes), "image/jpeg")}, timeout=60)
    r.raise_for_status()
    file_id = r.json()["id"]
    r2 = requests.post(f"{LLAMA_PARSE_BASE}/api/v2/parse",
                       headers={**H, "Content-Type": "application/json"},
                       json={"file_id": file_id, "tier": LLAMA_PARSE_TIER, "version": "latest"}, timeout=60)
    r2.raise_for_status()
    job = r2.json().get("job") or r2.json()
    job_id = job["id"]
    deadline = time.time() + 90  # в тесте COMPLETED за ~11с; 90с — щедрый потолок
    status = None
    while time.time() < deadline:
        time.sleep(2)
        s = requests.get(f"{LLAMA_PARSE_BASE}/api/v2/parse/{job_id}", headers=H, timeout=30).json()
        status = (s.get("job") or s).get("status")
        if status not in ("PENDING", "RUNNING"):
            break
    if status != "COMPLETED":
        raise RuntimeError(f"LlamaParse job {status or 'timeout'}")
    res = requests.get(f"{LLAMA_PARSE_BASE}/api/v2/parse/{job_id}?expand=markdown_full", headers=H, timeout=30)
    res.raise_for_status()
    md = (res.json().get("markdown_full") or "").strip()
    # схлопываем избыточные пустые строки от markdown-разметки
    return re.sub(r"\n{3,}", "\n\n", md)


# Предохранитель OCR (по образцу транскрипции): системный отказ → фолбэк фото на vision,
# изредка пробуем снова.
_OCR_FAILS = 0
_OCR_SKIPS = 0


async def llama_ocr(image_bytes: bytes) -> str:
    """OCR с ретраями. Возвращает текст с фото ("" — текста нет) или None, если OCR недоступен
    (нет ключа / предохранитель / все ретраи неудачны) — тогда вызывающий уходит на vision."""
    global _OCR_FAILS, _OCR_SKIPS
    if not llama_cloud_api_key:
        return None
    if _OCR_FAILS >= 8:
        _OCR_SKIPS += 1
        if _OCR_SKIPS % 25 != 1:  # редкие пробы «ожило?»
            return None
        log("MEDIA", f"OCR в отказе ({_OCR_FAILS} подряд, пропущено {_OCR_SKIPS}) — пробный запрос")
    for attempt in range(2):
        try:
            text = await asyncio.to_thread(_sync_llama_ocr, image_bytes)
            if _OCR_FAILS >= 8:
                log("MEDIA", "OCR ожил — предохранитель сброшен")
            _OCR_FAILS = 0
            _OCR_SKIPS = 0
            return text
        except Exception as e:
            log("MEDIA", f"llama_ocr попытка {attempt + 1}/2: {e}")
            if attempt == 0:
                await asyncio.sleep(3)
    _OCR_FAILS += 1
    if _OCR_FAILS == 8:
        log("MEDIA", "⚠️ OCR падает системно (8 подряд) — фото временно идут через vision")
    return None


async def extract_video_note_content(msg) -> str:
    try:
        video_bytes = await msg.download_media(file=bytes)
        if video_bytes:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", "pipe:0", "-vn", "-f", "ogg", "-acodec", "libopus", "-ar", "16000", "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            audio_data, _ = await proc.communicate(input=video_bytes)
            if audio_data:
                transcript = await transcribe_audio(audio_data, "ogg")
                return f"[Речь: {transcript}]"
    except FileNotFoundError:
        log("MEDIA", "ffmpeg не найден, пропускаю аудио из видео кружка")
    except Exception as e:
        log("MEDIA", f"Ошибка извлечения аудио из video_note: {e}")
    return "[видео кружок]"


# --- Озвучка ответов (Google Gemini Flash TTS) ---

def _build_tts_prompt(text: str, voice: str) -> str:
    """Минимальная нейтральная обёртка для TTS: тон, эмоцию и стиль задаёт САМА
    модель-ответчик через текст и аудио-теги [..]. Здесь — только просьба озвучить
    естественно и не зачитывать пометки в скобках. Короткий промпт ещё и реже
    ловит ложный отказ классификатора у 3.1-preview (400)."""
    return ("Озвучь этот текст естественно, живо, с эмоцией. Слова в квадратных скобках "
            "вроде [радостно] или [шёпотом] — это пометки интонации, НЕ произноси их вслух:\n" + text)


def _strip_for_tts(text: str) -> str:
    """Готовит текст к озвучке: убирает HTML-теги и markdown-мусор, СОХРАНЯЕТ аудио-теги [..],
    схлопывает пробелы и режет до TTS_VOICE_CHAR_CAP."""
    t = text or ""
    t = re.sub(r"<[^>]+>", "", t)              # HTML-теги прочь
    t = re.sub(r"[*#`_]+", "", t)              # markdown-мусор (звёздочки/решётки/бэктики/подчёрки)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) > TTS_VOICE_CHAR_CAP:
        t = t[:TTS_VOICE_CHAR_CAP].rsplit(" ", 1)[0].rstrip() + "…"
    return t


def _sync_tts(text: str, voice: str, api_key: str, model: str) -> bytes:
    """Один синхронный запрос к Gemini TTS. Возвращает PCM (s16le, 24kHz, mono). Бросает при ошибке."""
    url = GEMINI_TTS_URL.format(model=model)
    payload = {
        "contents": [{"parts": [{"text": _build_tts_prompt(text, voice)}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    r = requests.post(url, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                      json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"TTS HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    b64 = None
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            b64 = inline["data"]
            break
    if not b64:
        raise RuntimeError("TTS: в ответе нет аудио-данных")
    return base64.b64decode(b64)


def _sync_tts_openrouter(text: str, voice: str) -> bytes:
    """Озвучка через OpenRouter (та же модель google/gemini-3.1-flash-tts-preview, другой
    транспорт/квота). OpenAI-совместимый /audio/speech, response_format=pcm → сырой PCM
    s16le 24kHz mono (как у Google direct). Бросает при ошибке."""
    payload = {
        "model": GEMINI_TTS_OPENROUTER_MODEL,
        "input": _build_tts_prompt(text, voice),
        "voice": voice,
        "response_format": "pcm",
    }
    r = requests.post(OPENROUTER_TTS_URL,
                      headers={"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"},
                      json=payload, timeout=120)
    ct = r.headers.get("content-type", "")
    if r.status_code != 200 or ct.startswith("application/json"):
        raise RuntimeError(f"OR TTS HTTP {r.status_code}: {r.text[:200]}")
    if not r.content:
        raise RuntimeError("OR TTS: пустой ответ")
    return r.content  # PCM s16le 24kHz mono


async def _pcm_to_ogg(pcm: bytes) -> bytes:
    """PCM s16le 24kHz mono → OGG/Opus (формат голосовых Telegram) через ffmpeg (уже есть на сервере)."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-f", "s16le", "-ar", str(TTS_PCM_RATE), "-ac", "1", "-i", "pipe:0",
        "-c:a", "libopus", "-b:a", "32k", "-vbr", "on", "-application", "audio",
        "-f", "ogg", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    ogg, _ = await proc.communicate(input=pcm)
    if not ogg:
        raise RuntimeError("ffmpeg не вернул OGG (libopus?)")
    return ogg


async def _to_ogg_opus(data: bytes) -> bytes:
    """Любой аудио-вход (wav/mp3/…) → OGG/Opus через ffmpeg (автоопределение формата). Для Fish Audio."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", "pipe:0", "-ar", "48000", "-c:a", "libopus", "-b:a", "48k", "-vbr", "on",
        "-application", "audio", "-f", "ogg", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    ogg, _ = await proc.communicate(input=data)
    if not ogg:
        raise RuntimeError("ffmpeg не сконвертировал аудио Fish в OGG")
    return ogg


def _sync_tts_fish(text: str, reference_id: str) -> bytes:
    """Озвучка через Fish Audio. POST /v1/tts, заголовок model, тело JSON с reference_id.
    Возвращает сырые байты WAV (далее конвертим в OGG/Opus). Бросает при ошибке."""
    headers = {"Authorization": f"Bearer {fish_audio_api_key}", "Content-Type": "application/json",
               "model": FISH_TTS_MODEL}
    payload = {"text": text, "reference_id": reference_id, "format": "wav"}
    r = requests.post(FISH_TTS_URL, headers=headers, json=payload, timeout=120)
    ct = r.headers.get("content-type", "")
    if r.status_code != 200 or ct.startswith("application/json"):
        raise RuntimeError(f"Fish TTS HTTP {r.status_code}: {r.text[:200]}")
    if not r.content:
        raise RuntimeError("Fish TTS: пустой ответ")
    return r.content


async def _tts_try_fish(text: str) -> bytes:
    """Озвучка активным Fish-голосом (FISH_VOICE) с ретраем на transient. Бросает при провале."""
    if not FISH_VOICE:
        raise RuntimeError("Fish-голос не выбран (/voice fish add/select)")
    # Fish S2 ПОНИМАЕТ [квадратные] описания подачи — сохраняем их. S1 их не понимает (у него
    # (круглые)) — для не-s2 срезаем [теги], чтобы не зачитывались; (круглые) оставляем как есть.
    clean = text if str(FISH_TTS_MODEL).lower().startswith("s2") else re.sub(r"\[[^\]]*\]", "", text)
    last_err = None
    for attempt in range(2):
        try:
            wav = await asyncio.to_thread(_sync_tts_fish, clean, FISH_VOICE)
            return await _to_ogg_opus(wav)
        except Exception as e:
            last_err = e
            if _tts_err_kind(e) in ("transient", "classifier") and attempt == 0:
                log("TTS", f"Fish: повтор ({str(e)[:50]})")
                await asyncio.sleep(1.5)
                continue
            break
    raise last_err if last_err else RuntimeError("Fish TTS: неизвестная ошибка")


def _tts_err_kind(e) -> str:
    """Классификация ошибки TTS (по докам Google):
    quota — лимит ключа (429) → сменить ключ;
    transient — перегрузка/таймаут (503/500/«text token returns») → повторить;
    classifier — 400 INVALID_ARGUMENT: у 3.1-preview это ЛОЖНЫЙ отказ классификатора,
                 документировано как flaky → повторить, затем сменить модель;
    prohibited — реальный отказ по контенту → не долбить;
    other — прочее."""
    s = str(e).lower()
    if "429" in s or "resource_exhausted" in s or "quota" in s or "rate limit" in s:
        return "quota"
    if "prohibited" in s or "safety" in s or "blocked" in s:
        return "prohibited"
    if "503" in s or "500" in s or "unavailable" in s or "high demand" in s or "internal" in s \
            or "timed out" in s or "timeout" in s:
        return "transient"
    if "400" in s or "invalid_argument" in s or "invalid argument" in s:
        return "classifier"
    return "other"


async def _tts_try_model(text: str, voice: str, model: str, max_attempts: int = 4) -> bytes:
    """Пытается озвучить одной моделью: до max_attempts попыток с ротацией ключей.
    Повторяет при quota (другой ключ), transient (503/500) и classifier (ложный 400 у 3.1).
    Бросает последнюю ошибку, если не вышло."""
    global _tts_key_idx
    last_err = None
    for attempt in range(max_attempts):
        key = GOOGLE_TTS_KEYS[_tts_key_idx % len(GOOGLE_TTS_KEYS)]
        _tts_key_idx = (_tts_key_idx + 1) % len(GOOGLE_TTS_KEYS)
        try:
            pcm = await asyncio.to_thread(_sync_tts, text, voice, key, model)
            return await _pcm_to_ogg(pcm)
        except Exception as e:
            last_err = e
            kind = _tts_err_kind(e)
            if kind == "prohibited":
                break  # реальный отказ — повторять бессмысленно
            if kind in ("quota", "transient", "classifier") and attempt + 1 < max_attempts:
                log("TTS", f"{model}: {kind} ({str(e)[:60]}) — попытка {attempt + 2}/{max_attempts}")
                await asyncio.sleep(1.5 if kind != "quota" else 0.3)
                continue
            break
    raise last_err if last_err else RuntimeError("TTS: неизвестная ошибка")


async def _tts_try_openrouter(text: str, voice: str) -> bytes:
    """Озвучка через OpenRouter (та же 3.1-модель) с ретраем на transient/classifier. Бросает при провале."""
    last_err = None
    for attempt in range(2):
        try:
            pcm = await asyncio.to_thread(_sync_tts_openrouter, text, voice)
            return await _pcm_to_ogg(pcm)
        except Exception as e:
            last_err = e
            if _tts_err_kind(e) in ("transient", "classifier") and attempt == 0:
                log("TTS", f"OpenRouter: повтор ({str(e)[:50]})")
                await asyncio.sleep(1.5)
                continue
            break
    raise last_err if last_err else RuntimeError("OpenRouter TTS: неизвестная ошибка")


def _gemini_tts_steps(spoken, voice):
    """Шаги Gemini-цепочки: 3.1 Google → 3.1 OpenRouter → 2.5 Google."""
    steps = [
        (f"Google/{GEMINI_TTS_MODEL}", lambda: _tts_try_model(spoken, voice, GEMINI_TTS_MODEL), bool(GOOGLE_TTS_KEYS)),
        (f"OpenRouter/{GEMINI_TTS_OPENROUTER_MODEL}", lambda: _tts_try_openrouter(spoken, voice), bool(openrouter_api_key)),
    ]
    if GEMINI_TTS_FALLBACK_MODEL and GEMINI_TTS_FALLBACK_MODEL != GEMINI_TTS_MODEL:
        steps.append((f"Google/{GEMINI_TTS_FALLBACK_MODEL}", lambda: _tts_try_model(spoken, voice, GEMINI_TTS_FALLBACK_MODEL), bool(GOOGLE_TTS_KEYS)))
    return steps


async def synthesize_voice(text: str, voice: str, engine: str = None):
    """Озвучивает text. Движок — engine или TTS_ENGINE (gemini|fish); при сбое выбранного —
    автофолбэк на другой. Gemini-цепочка: 3.1 Google → 3.1 OpenRouter → 2.5 Google.
    Fish: активный FISH_VOICE. Возвращает bytes OGG/Opus или None (тогда фолбэк на текст)."""
    voice = _validate_voice(voice)
    spoken = _strip_for_tts(text)
    if not spoken:
        return None
    gemini_ok = bool(GOOGLE_TTS_KEYS or openrouter_api_key)
    fish_ok = bool(fish_available and FISH_VOICE)
    if not gemini_ok and not fish_ok:
        return None

    eng = engine or TTS_ENGINE
    fish_step = ("Fish", lambda: _tts_try_fish(spoken), fish_ok)
    if eng == "fish":
        steps = [fish_step] + _gemini_tts_steps(spoken, voice)  # Fish primary, Gemini — фолбэк
    else:
        steps = _gemini_tts_steps(spoken, voice) + [fish_step]  # Gemini primary, Fish — фолбэк

    last_err = None
    for label, factory, available in steps:
        if not available:
            continue
        try:
            ogg = await factory()
            log("TTS", f"Озвучено: {label}, текст={len(spoken)} симв., ogg={len(ogg)} байт")
            return ogg
        except Exception as e:
            last_err = e
            log("TTS", f"{label} не дала аудио ({str(e)[:80]})")

    log("TTS", f"Озвучка не удалась ({last_err}) — фолбэк на текст")
    return None


async def _ensure_voice_sample(name: str):
    """OGG-пример голоса name из кэша voice_samples/<name>.ogg. Если нет — синтезирует
    фразой VOICE_SAMPLE_TEXT и кэширует на диск. Возвращает bytes OGG или None."""
    name = _validate_voice(name)
    path = os.path.join(VOICE_SAMPLES_DIR, f"{name}.ogg")
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                return f.read()
    except Exception:
        pass
    ogg = await synthesize_voice(VOICE_SAMPLE_TEXT, name, engine="gemini")  # сэмплы — всегда Gemini-голоса
    if ogg:
        try:
            os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)
            with open(path, "wb") as f:
                f.write(ogg)
        except Exception as e:
            log("TTS", f"Не удалось сохранить сэмпл {name}: {e}")
    return ogg


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Вырезает reasoning-блок <think>…</think> из content (MiniMax M3 и др. кладут
    размышления прямо в content). Незакрытый <think> (ответ обрезан по длине) → пусто,
    чтобы сработал фолбэк/ретрай, а не показ голых размышлений."""
    if not text or "<think>" not in text.lower():
        return text
    t = _THINK_RE.sub("", text)
    if "<think>" in t.lower():  # открыт, но не закрыт — режем от тега до конца
        t = re.sub(r"<think>.*$", "", t, flags=re.DOTALL | re.IGNORECASE)
    return t.strip()


def _extract_content(message) -> str:
    # Финальный ответ в .content; у reasoning-моделей при пустом .content берём .reasoning_content
    content = _strip_think((getattr(message, "content", None) or "").strip())
    if content:
        return content
    return (getattr(message, "reasoning_content", None) or "").strip()


async def _llm_create(messages: list, max_tokens: int = 4096, temperature: float = 1.0,
                      reasoning=_NO_REASONING_OVERRIDE, model_slug: str = None):
    """reasoning — оверрайд глубины размышлений на этот вызов (утилитарные задачи: дайджест шлёт 'none',
    чтобы reasoning-модель не съела бюджет размышлениями и не отдала сырой CoT). По умолчанию — глобальный.
    model_slug — пин конкретной модели реестра вместо активной (дайджест всегда на deepseek-flash);
    если у её провайдера нет ключа — мягкий фоллбэк на активную модель."""
    if model_slug and model_slug in MODEL_REGISTRY:
        _prov, _mid, _lbl, _c, _s = MODEL_REGISTRY[model_slug]
        _cli = _client_for_provider(_prov)
        if _cli is not None:
            client_obj, model_id, label = _cli, _mid, _lbl
        else:
            log("AI", f"Пин-модель {model_slug} недоступна (нет ключа {_prov}) — фоллбэк на активную")
            client_obj, model_id, label = get_active_model()
    else:
        client_obj, model_id, label = get_active_model()
    if client_obj is None:
        log("AI", f"Активная модель {ACTIVE_MODEL} недоступна (нет ключа провайдера)")
        return None
    # Логируем входящий контекст (обрезаем длинные сообщения)
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str):
            preview = content[:200].replace("\n", " ")
        elif isinstance(content, list):
            # multimodal — считаем типы частей
            parts = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("type", "?")
                    if t == "text":
                        parts.append(f"text:{part.get('text','')[:80].replace(chr(10),' ')}")
                    elif t == "image_url":
                        parts.append("image")
                    else:
                        parts.append(t)
            preview = f"[multimodal: {', '.join(parts)}]"
        else:
            preview = str(content)[:200]
        log("AI", f"  msg[{i}] {role}: {preview}")
    log("AI", f"Запрос {label} model={model_id} max_tokens={max_tokens} temp={temperature}")

    _rtok = _REASONING_OVERRIDE.set(reasoning) if reasoning is not _NO_REASONING_OVERRIDE else None
    try:
        for attempt in range(2):
            try:
                response = await asyncio.to_thread(
                    client_obj.chat.completions.create,
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                msg_obj = response.choices[0].message
                content = _extract_content(msg_obj)
                from_reasoning = bool(content) and not (getattr(msg_obj, "content", None) or "").strip()
                finish = response.choices[0].finish_reason
                usage = getattr(response, "usage", None)
                tokens_info = f"prompt={usage.prompt_tokens}, completion={usage.completion_tokens}" if usage else "?"
                src = " (из reasoning_content)" if from_reasoning else ""
                log("AI", f"Ответ {label} (попытка {attempt + 1}): finish={finish} tokens={tokens_info} content_len={len(content)}{src} content=[{content[:300]}]")
                # Обрезанная цепочка размышлений (finish=length, видимого ответа нет) — это НЕ ответ.
                # Не выдаём сырой CoT наружу (баг дайджеста): считаем попытку пустой → ретрай/фейл.
                if from_reasoning and finish == "length":
                    log("AI", f"{label}: только обрезанный reasoning без ответа — не выдаём CoT, ретрай")
                    content = ""
                if content:
                    return content
                log("AI", f"Пустой ответ {label} (попытка {attempt + 1}/2) finish={finish}")
            except Exception as e:
                log("AI", f"Ошибка {label} (попытка {attempt + 1}/2): {e}")
                # Переполнение окна — не глотаем, кидаем наверх, чтобы ask_command мог ретрайнуть с агрессивной обрезкой
                if _is_context_overflow(e):
                    raise ContextOverflowError(str(e)) from e
                if attempt == 1:
                    traceback.print_exc()
                await asyncio.sleep(1)
        return None
    finally:
        if _rtok is not None:
            _REASONING_OVERRIDE.reset(_rtok)


def _build_ask_user_content(context: str, question: str, caller: str = None, now_str: str = None) -> str:
    """Раскладка под prompt-кэш: СТАБИЛЬНЫЙ префикс (статичная пометка + лог чата) идёт первым,
    а ЛЕТУЧИЙ суффикс (сам вопрос + текущее время) — в самом конце. Так смена вопроса/времени не
    рушит кэш контекста. Короткая пометка в начале не даёт спутать реальный вопрос с вопросами в логе."""
    asker = caller or "пользователь"
    tail = (
        f"\n\n❓ ВОПРОС (его задаёт {asker}): {question}"
    )
    if now_str:
        tail += f"\n\nТекущая дата и время: {now_str} МСК. Учитывай актуальность: оценивай свежесть постов по их дате, для новостей опирайся на самые недавние."
    return (
        f"Ниже — лог переписки (контекст/фон). Сам ВОПРОС и текущее время — в САМОМ КОНЦЕ, после лога.\n\n"
        f"━━━━━ Контекст чата (лог переписки) ━━━━━\n"
        f"{context}\n"
        f"━━━━━ конец контекста чата ━━━━━"
        f"{tail}"
    )


async def generate_ask_reply(context: str, question: str, caller: str = None) -> str:
    now_str = datetime.now(MSK).strftime("%d.%m.%Y %H:%M")
    _, _, label = get_active_model()
    result = await _llm_create(
        messages=[
            {"role": "system", "content": ASK_SYSTEM_PROMPT.replace("{model}", label) + f"\n\nТекущая дата и время: {now_str} МСК."},
            {"role": "user", "content": _build_ask_user_content(context, question, caller)},
        ],
        max_tokens=ASK_MAX_TOKENS,  # thinking-модели жрут на reasoning тысячи токенов
        temperature=1.0,
    )
    return result if result else "Модель не смогла ответить (пустой ответ или ошибка API)"


async def generate_auto_reply(combined_text: str, history: list = None) -> str:
    messages = [{"role": "system", "content": AUTO_REPLY_SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": combined_text})
    result = await _llm_create(messages=messages, max_tokens=4096, temperature=1.0)
    return result if result else "Понял"


# ── Веб-инструменты Tavily: выполнение tool-call'ов из agentic loop ──

def _sync_tavily(endpoint: str, payload: dict, timeout: int = 60) -> dict:
    """POST на api.tavily.com/{endpoint}. Возвращает dict ответа, кидает RuntimeError при ошибке."""
    r = requests.post(
        f"{TAVILY_BASE_URL}/{endpoint}",
        headers={"Authorization": f"Bearer {tavily_api_key}", "Content-Type": "application/json"},
        json=payload, timeout=timeout,
    )
    if r.status_code != 200:
        detail = r.text[:200]
        try:
            detail = (r.json().get("detail") or {}).get("error") or detail
        except Exception:
            pass
        raise RuntimeError(f"Tavily {endpoint} HTTP {r.status_code}: {detail}")
    return r.json()


async def _run_web_tool(name: str, args: dict) -> str:
    """Выполняет веб-инструмент (web_search/web_extract/web_crawl/web_map) и форматирует
    результат строкой для tool-сообщения. Ошибки возвращает текстом — loop не падает."""
    try:
        if name == "web_search":
            query = (args.get("query") or "").strip()
            if not query:
                return "Ошибка: пустой поисковый запрос"
            n = max(1, min(int(args.get("max_results") or 5), WEB_SEARCH_MAX_RESULTS))
            payload = {"query": query, "max_results": n, "include_answer": True,
                       "search_depth": args.get("search_depth") if args.get("search_depth") in ("basic", "advanced") else "basic"}
            if args.get("topic") in ("general", "news"):
                payload["topic"] = args["topic"]
            if args.get("time_range") in ("day", "week", "month", "year"):
                payload["time_range"] = args["time_range"]
            d = await asyncio.to_thread(_sync_tavily, "search", payload, 40)
            results = d.get("results") or []
            if not results:
                return f"Веб-поиск по «{query}»: ничего не найдено."
            lines = []
            if d.get("answer"):
                lines.append(f"💡 Краткий ответ поисковика: {d['answer']}")
            for r_ in results:
                date = f" | 📅 {r_['published_date']}" if r_.get("published_date") else ""
                lines.append(f"• {r_.get('title', '')}{date}\n  {r_.get('url', '')}\n  {_preview(r_.get('content') or '', 800)}")
            return f"Веб-поиск «{query}»: {len(results)} результатов.\n\n" + "\n\n".join(lines)

        if name == "web_extract":
            urls = [u for u in (args.get("urls") or []) if isinstance(u, str) and u.strip()][:WEB_EXTRACT_MAX_URLS]
            if not urls:
                return "Ошибка: не передано ни одного URL"
            d = await asyncio.to_thread(_sync_tavily, "extract", {"urls": urls, "format": "markdown"}, 90)
            parts = []
            for r_ in d.get("results") or []:
                txt = (r_.get("raw_content") or "").strip()
                cut = " …(обрезано)" if len(txt) > WEB_EXTRACT_MAX_CHARS else ""
                parts.append(f"═══ {r_.get('url')} ═══\n{txt[:WEB_EXTRACT_MAX_CHARS]}{cut}")
            for f_ in d.get("failed_results") or []:
                parts.append(f"⚠️ Не удалось извлечь: {f_.get('url')} ({f_.get('error', '')})")
            return "\n\n".join(parts) if parts else "Не удалось извлечь ни одной страницы."

        if name == "web_crawl":
            url = (args.get("url") or "").strip()
            if not url:
                return "Ошибка: не передан URL"
            payload = {"url": url, "limit": WEB_CRAWL_MAX_PAGES, "max_depth": 2}
            if args.get("instructions"):
                payload["instructions"] = str(args["instructions"])[:500]
            d = await asyncio.to_thread(_sync_tavily, "crawl", payload, 150)
            results = d.get("results") or []
            if not results:
                return f"Обход {url}: страниц не найдено."
            parts = [f"Обход {url}: {len(results)} страниц."]
            for r_ in results[:WEB_CRAWL_MAX_PAGES]:
                txt = (r_.get("raw_content") or "").strip()
                parts.append(f"═══ {r_.get('url')} ═══\n{txt[:WEB_CRAWL_PAGE_CHARS]}{' …(обрезано)' if len(txt) > WEB_CRAWL_PAGE_CHARS else ''}")
            return "\n\n".join(parts)

        if name == "web_map":
            url = (args.get("url") or "").strip()
            if not url:
                return "Ошибка: не передан URL"
            d = await asyncio.to_thread(_sync_tavily, "map", {"url": url, "limit": WEB_MAP_MAX_URLS, "max_depth": 2}, 90)
            urls = d.get("results") or []
            if not urls:
                return f"Карта {url}: ссылок не найдено."
            return f"Карта сайта {url} ({len(urls)} ссылок):\n" + "\n".join(f"• {u}" for u in urls[:WEB_MAP_MAX_URLS])

        return f"Неизвестный веб-инструмент: {name}"
    except (RuntimeError, requests.exceptions.RequestException) as e:
        log("ASK", f"Веб-инструмент {name} упал: {e}")
        return f"Ошибка веб-инструмента {name}: {e}. Попробуй другой запрос или ответь без этих данных."


async def _send_quote_reply(chat_id, mid: int, html_text: str, quote: str, src_text: str) -> bool:
    """Реплай с подсветкой КОНКРЕТНОГО фрагмента (partial quote). Находит `quote` как точную
    подстроку исходного текста, считает UTF-16 offset (требование Telegram) и шлёт через сырой
    SendMessageRequest с InputReplyToMessage(quote_text, quote_offset). True — отправлено;
    False — фрагмент не найден / текст слишком длинный / ошибка (вызывающий откатится на обычный реплай)."""
    if not quote or not src_text or len(html_text) > 4000:
        return False
    pos = src_text.find(quote)
    if pos < 0:
        return False
    off16 = len(src_text[:pos].encode("utf-16-le")) // 2  # Telegram считает offset в UTF-16
    try:
        if len(html_text) > REPLY_COLLAPSE:  # длинный реплай — свернуть в раскрывающийся цитат-блок
            msg_text, entities = _collapsed_entities(html_text, parse_html=True)
        else:
            msg_text, entities = client._parse_message_text(html_text, "html")
        reply_obj = InputReplyToMessage(reply_to_msg_id=mid, quote_text=quote, quote_offset=off16)
        await client(SendMessageRequest(peer=chat_id, message=msg_text, reply_to=reply_obj,
                                        entities=entities, no_webpage=True))
        return True
    except FloodWaitError as e:
        await asyncio.sleep(e.seconds + 1)
        try:
            await client(SendMessageRequest(peer=chat_id, message=msg_text, reply_to=reply_obj,
                                            entities=entities, no_webpage=True))
            return True
        except Exception as e2:
            log("ASK", f"Quote-реплай на #{mid} не ушёл после FloodWait ({e2})")
            return False
    except Exception as e:
        log("ASK", f"Quote-реплай на #{mid} не удался ({e}) — откат на обычный реплай")
        return False


async def _run_reply_tool(args: dict, chat_id, msg_by_id: dict, reply_sent: list) -> str:
    """Исполняет reply_to_messages: для каждого валидного {message_id, text, quote?} шлёт ОТДЕЛЬНЫЙ
    реплай тредом на исходное сообщение. Если задан `quote` (точная подстрока сообщения) — подсвечивает
    именно этот фрагмент (partial quote); иначе/если не найден — реплай на всё сообщение (send_long).
    Соблюдает лимит REPLY_MAX. Возвращает сводку для модели. Ошибки не роняют agentic-цикл."""
    replies = args.get("replies")
    if not isinstance(replies, list) or not replies:
        return "Ошибка: пустой список replies. Передай массив объектов {message_id, text}."
    sent, not_found, capped, bad = [], [], 0, 0
    quoted = 0
    for item in replies:
        if not isinstance(item, dict):
            bad += 1
            continue
        text = (item.get("text") or "").strip()
        quote = (item.get("quote") or "").strip()
        try:
            mid = int(item.get("message_id"))
        except (TypeError, ValueError):
            bad += 1
            continue
        if not text:
            bad += 1
            continue
        if reply_sent[0] >= REPLY_MAX:
            capped += 1
            continue
        if mid not in msg_by_id:
            not_found.append(mid)
            continue
        try:
            cleaned = _html_clean_markdown(text)
            did_quote = False
            if quote:
                src = msg_by_id[mid]
                src_text = getattr(src, "raw_text", None) or getattr(src, "message", None) or ""
                did_quote = await _send_quote_reply(chat_id, mid, cleaned, quote, src_text)
            if not did_quote:  # без фрагмента или фрагмент не найден → реплай на всё сообщение
                await send_long(chat_id, cleaned, parse_mode="html", reply_to=mid, collapse_threshold=REPLY_COLLAPSE)
            reply_sent[0] += 1
            sent.append(mid)
            quoted += 1 if did_quote else 0
            log("ASK", f"Реплай отправлен на #{mid} ({len(text)} симв{', с фрагментом' if did_quote else ''})")
        except Exception as e:
            log("ASK", f"Реплай на #{mid} не отправлен: {e}")
            not_found.append(mid)  # модель пусть считает его неудачным
    parts = []
    if sent:
        qn = f" ({quoted} с подсветкой фрагмента)" if quoted else ""
        parts.append(f"Отправлено {len(sent)} реплаев на #" + ", #".join(str(x) for x in sent) + qn + ".")
    if not_found:
        parts.append("Не найдены/не отправлены id: " + ", ".join(f"#{x}" for x in not_found) + " — проверь метки #id.")
    if capped:
        parts.append(f"Лимит {REPLY_MAX} реплаев исчерпан, лишние {capped} пропущены.")
    if bad:
        parts.append(f"{bad} элементов пропущено (пустой текст или некорректный message_id).")
    if not parts:
        parts.append("Ничего не отправлено.")
    parts.append("Теперь дай общий итоговый ответ обычным текстом (он уйдёт отдельным сообщением).")
    return " ".join(parts)


async def ask_agentic(context: str, question: str, must_search: bool = False, caller: str = None, ctx_tokens_est: int = None, voice_mode: str = "off", images: list = None, chat_id=None, msg_by_id: dict = None, memory_allowed: bool = True, asker_id=None) -> str:
    """Agentic ask: модель сама решает, искать ли информацию в каналах.
    ctx_tokens_est — tiktoken-оценка контекста (для логирования Δ с реальным API).
    voice_mode: "off" — обычный текст; "force" — ответ под озвучку (флаг -v); "auto" — модель сама может выбрать голос (маркер [[VOICE]]).
    images — список {"bytes":...} для прямого vision (/ask -g): кладутся в user-сообщение как image_url.
    chat_id/msg_by_id — для инструмента reply_to_messages: модель шлёт реплаи тредами на сообщения
    из истории по их #id (msg_by_id: {id: Message}); реплаи отправляются сразу в ходе цикла."""
    llm, model_id, label = get_active_model()
    if llm is None:
        return "Модель не настроена (проверь ключ провайдера)"

    channels = get_tracked()
    has_channels = len(channels) > 0
    has_web = bool(tavily_api_key)        # веб-инструменты Tavily (web_search/web_extract/web_crawl/web_map)
    has_reply = chat_id is not None and bool(msg_by_id)  # адресный реплай по #id истории
    has_memory = False                    # GraphRAG-память /index (memory_search/entity/media) — если чат проиндексирован
    if memory_allowed and chat_id is not None and not _index_available():
        try:
            has_memory = (await db_read("SELECT COUNT(*) c FROM entities WHERE chat_id=%s", (chat_id,)))[0]["c"] > 0
        except Exception as e:
            log("ASK", f"Проверка памяти /index не удалась: {e}")
    has_tools = has_channels or has_web or has_reply or has_memory

    now_str = datetime.now(MSK).strftime("%d.%m.%Y %H:%M")

    # ВАЖНО для prompt-кэша: системный промпт ДОЛЖЕН быть статичным (без даты/времени),
    # иначе летучая строка в начале рушит префиксный кэш. Дата уезжает в КОНЕЦ user-контента.
    system_prompt = ASK_SYSTEM_PROMPT.replace("{model}", label)
    if has_channels:
        system_prompt += "\n\nУ тебя есть доступ к инструменту telegram_search для поиска в Telegram-каналах. Используй его если вопрос требует актуальной информации, которой нет в контексте переписки. Формулируй точные поисковые запросы. Для свежих новостей указывай параметр days."
    if has_web:
        system_prompt += ("\n\nУ тебя есть доступ в интернет: web_search (поиск), web_extract (прочитать страницы по URL), "
                          "web_crawl (обойти раздел сайта), web_map (карта сайта). Когда и сколько искать — решаешь ты. "
                          "Ориентиры (необязательные): обычно полезно искать на актуальные события, факты вне переписки, "
                          "спорные утверждения, ссылки из чата. Если оцениваешь утверждение как верное/ложное, чаще всего "
                          "надёжнее опереться на несколько источников и при противоречии глянуть обе стороны, а не один "
                          "сниппет; когда данных мало — честнее так и сказать, чем выдавать уверенность. Источники указывай "
                          "тегом <a href=\"URL\">. То, что и так знаешь или есть в чате, искать обычно незачем.")
    if has_reply:
        system_prompt += ("\n\nКаждое сообщение в истории помечено его #id (число перед текстом). Инструмент "
                          "reply_to_messages шлёт реплай на конкретные сообщения — на одно или сразу на несколько "
                          f"(до {REPLY_MAX}), каждый отдельным сообщением, привязанным к своему. Это просто ещё один "
                          "способ ответить, выбираешь его ты. Реплай обычно к месту, когда хочется обратиться адресно: "
                          "разобрать спор, ответить разным людям по отдельности, привязать ответ к конкретной реплике; "
                          "если же ответ общий или адресат один — хватает обычного текста. Можно подсветить фрагмент: "
                          "в поле quote передай дословную подстроку этого сообщения (удобно для длинных). После реплаев, "
                          "как правило, стоит дать и общий итоговый ответ обычным текстом. Что выбрать — на твоё усмотрение.")
    if has_memory:
        system_prompt += ("\n\nУ этого чата есть ПРОИНДЕКСИРОВАННАЯ ПАМЯТЬ (команда /index) — база по всей истории: "
                          "memory_search (смысловой поиск по сценам-диалогам, досье и связям — для ТОЧЕЧНЫХ фактов и "
                          "конкретных реплик), memory_overview (высокоуровневое обобщение по теме для ГЛОБАЛЬНЫХ/"
                          "ТЕМАТИЧЕСКИХ вопросов — атмосфера, общее отношение к кому-то, динамика — из сжатых досье и "
                          "связей, без сырых диалогов), memory_entity (полное досье персонажа/участника по имени или "
                          "прозвищу — canon-факты, мнение чата, связи), "
                          "memory_media (найти и переслать в чат фото из истории по описанию). Используй их для вопросов "
                          "про историю чата, лор, персонажей, прошлые споры и события, про которых нет в текущем контексте, "
                          "и когда просят найти/скинуть старое фото. Имена и прозвища разрешаются автоматически. "
                          "Если memory_search вернул слабые/пустые совпадения (помечено ⚠️) — не выдавай догадку за факт: "
                          "переспроси другими словами (конкретнее имена/событие) один раз или, для внешних тем, используй веб. "
                          "Иногда к вопросу приложена справка о спрашивающем из памяти — учитывай её, только если релевантно.")
    if must_search and (has_channels or has_web):
        force_name = "telegram_search" if has_channels else "web_search"
        system_prompt += f"\n\nОБЯЗАТЕЛЬНО используй {force_name} хотя бы один раз перед тем как ответить."
    if voice_mode == "force":
        system_prompt += _voice_style_text(TTS_ENGINE, FISH_TTS_MODEL)
    elif voice_mode == "auto":
        system_prompt += _voice_auto_hint(TTS_ENGINE, FISH_TTS_MODEL)

    user_text = _build_ask_user_content(context, question, caller, now_str)
    if has_memory and asker_id:  # персональная память: бот узнаёт спрашивающего и подтягивает его досье
        try:
            _who = await _index_asker_brief(chat_id, asker_id)
            if _who:
                user_text += "\n\n" + _who
        except Exception as e:
            log("ASK", f"Справка о спрашивающем не собралась: {e}")
    if images:
        # Мультимодальный content: текст + сами картинки (/ask -g)
        user_content = [{"type": "text", "text": user_text}]
        for im in images:
            b64 = base64.b64encode(im["bytes"]).decode("utf-8")
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}})
    else:
        user_content = user_text
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    max_iterations = 20
    force_tool = must_search and (has_channels or has_web)  # принудительный поиск только для search-инструментов
    force_tool_name = "telegram_search" if has_channels else "web_search"
    tools_list = (([TELEGRAM_SEARCH_TOOL] if has_channels else []) + (WEB_TOOLS if has_web else [])
                  + ([REPLY_TOOL] if has_reply else []) + (INDEX_MEMORY_TOOLS if has_memory else []))
    reply_sent = [0]  # счётчик отправленных реплаев (анти-спам, лимит REPLY_MAX)
    sstats = {"iters": 0, "calls": 0, "posts": 0, "web": 0, "replies": 0, "memory": 0}  # сводка (-c)

    def _log_search_summary():
        if sstats["iters"]:
            log("ASK", f"Поиск: {sstats['iters']} итер., {sstats['calls']} запросов к каналам, найдено {sstats['posts']} постов, "
                       f"веб-вызовов {sstats['web']}, память /index {sstats['memory']}")

    for iteration in range(max_iterations):
        log("ASK", f"Agentic итерация {iteration + 1}/{max_iterations}")

        try:
            kwargs = dict(
                model=model_id,
                messages=messages,
                max_tokens=ASK_MAX_TOKENS,
                temperature=1.0,
            )
            if tools_list:
                kwargs["tools"] = tools_list
                # Sakana молча игнорит ПРИНУДИТЕЛЬНЫЙ tool_choice (без ошибки → откат на auto не сработал бы) →
                # для него сразу auto: на auto Fugu сам вызывает поиск (проверено вживую).
                _force_now = force_tool and MODEL_REGISTRY.get(ACTIVE_MODEL, ("",))[0] not in ("sakana", "gloy")
                kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool_name}} if _force_now else "auto"

            try:
                response = await asyncio.to_thread(llm.chat.completions.create, **kwargs)
            except Exception as e:
                # Thinking-модели (DeepSeek) не умеют ПРИНУДИТЕЛЬНЫЙ tool_choice, но auto — умеют.
                # Не считаем «без tools»: повторяем с auto, поиск остаётся доступен.
                if force_tool and has_tools and _is_thinking_mode_quirk(e):
                    log("ASK", "Принудительный tool_choice не поддержан (thinking-режим) — повтор с auto")
                    kwargs["tool_choice"] = "auto"
                    response = await asyncio.to_thread(llm.chat.completions.create, **kwargs)
                else:
                    raise
        except TypeError:
            # Модель не поддерживает tools — fallback на обычный ask
            log("ASK", "Модель не поддерживает tool calling, fallback на обычный ask")
            if has_tools:
                _set_tools_support(ACTIVE_MODEL, False)
            _log_search_summary()
            return await generate_ask_reply(context, question, caller=caller)
        except Exception as e:
            log("ASK", f"Ошибка модели в agentic loop: {e}")
            # Переполнение окна — кидаем наверх для ретрая с агрессивной обрезкой
            if _is_context_overflow(e):
                _log_search_summary()
                raise ContextOverflowError(str(e)) from e
            quirk = _is_thinking_mode_quirk(e)
            # thinking-quirk НЕ трактуем как «без tools» (модель умеет auto/tools, просто особенности API);
            # ошибки про reasoning_effort — тоже (это конфликт параметров, а не отсутствие tools)
            if has_tools and not quirk and "reasoning_effort" not in str(e) and any(k in str(e).lower() for k in ("tool", "function")):
                _set_tools_support(ACTIVE_MODEL, False)
            # Сброс СТАЛОЙ записи tools_support=False, если на самом деле это thinking-quirk
            if quirk and MODEL_TOOLS_SUPPORT.get(ACTIVE_MODEL) is False:
                MODEL_TOOLS_SUPPORT.pop(ACTIVE_MODEL, None)
                _save_model_state()
                log("MODEL", f"{ACTIVE_MODEL}: ошибочный флаг tools=False сброшен (thinking-quirk, не реальная неподдержка)")
            traceback.print_exc()
            _log_search_summary()
            return await generate_ask_reply(context, question, caller=caller)

        # После первой итерации не форсируем tool call
        force_tool = False

        choice = response.choices[0]
        msg = choice.message

        # Реальный расход токенов от API (для сравнения с оценкой tiktoken в assemble_context)
        usage = getattr(response, "usage", None)
        if usage:
            win = active_context_window()
            occ = round(100 * usage.prompt_tokens / win, 1) if win else 0
            _cached = _cached_tokens(usage)
            cache_note = f" · из кэша {_cached} ({round(100 * _cached / usage.prompt_tokens)}%)" if _cached and usage.prompt_tokens else ""
            log("ASK", f"API {label}: занято {usage.prompt_tokens} ток в окне {_fmt_ctx(win)} = {occ}% (итер {iteration + 1}); ответ {usage.completion_tokens} ток{cache_note}")
            # Δ tiktoken vs реального токенизатора API (только на первой итерации — где контекст без tool-сообщений)
            if ctx_tokens_est and iteration == 0 and usage.prompt_tokens:
                delta = usage.prompt_tokens - ctx_tokens_est
                pct = round(100 * delta / usage.prompt_tokens, 1)
                verdict = "недооценил" if delta > 0 else ("переоценил" if delta < 0 else "точно")
                margin = (CTX_TOKEN_SAFETY - 1) * 100
                covered = "покрыл" if abs(pct) <= margin else "НЕ покрыл"
                log("ASK", f"Δ токенизаторов: tiktoken={ctx_tokens_est} vs API={usage.prompt_tokens} → tiktoken {verdict} на {abs(pct)}% (запас {int(margin)}% {covered})")

        # Получили валидный ответ с инструментами — модель умеет tools
        if has_tools and msg.tool_calls:
            _set_tools_support(ACTIVE_MODEL, True)
            sstats["iters"] += 1

        # Если нет tool_calls — это финальный ответ
        if not msg.tool_calls:
            content = _extract_content(msg)
            if content:
                log("ASK", f"Agentic ответ (итерация {iteration + 1}, без поиска)")
                _log_search_summary()
                return content
            # Пустой ответ — fallback
            _log_search_summary()
            return await generate_ask_reply(context, question, caller=caller)

        # Обрабатываем tool calls
        # Сериализуем assistant-message вручную, чтобы СОХРАНИТЬ reasoning_content —
        # thinking-модели (Kimi K2.x, DeepSeek reasoner) требуют его в tool-loop, иначе 400.
        assistant_dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        _reasoning = getattr(msg, "reasoning_content", None)
        if _reasoning:
            assistant_dict["reasoning_content"] = _reasoning
        messages.append(assistant_dict)

        for tool_call in msg.tool_calls:
            tname = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
                if not isinstance(args, dict):
                    args = {}
            except (json.JSONDecodeError, TypeError):
                args = {}

            # — веб-инструменты Tavily —
            if tname in ("web_search", "web_extract", "web_crawl", "web_map"):
                sstats["web"] += 1
                brief = args.get("query") or args.get("url") or ",".join((args.get("urls") or [])[:2])
                log("ASK", f"Веб-инструмент {tname}: {str(brief)[:120]}")
                web_result = await _run_web_tool(tname, args)
                log("ASK", f"{tname} вернул {len(web_result)} симв")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": web_result,
                })
                continue

            # — адресный реплай на сообщения истории (отправляем сразу, тредами) —
            if tname == "reply_to_messages":
                res = await _run_reply_tool(args, chat_id, msg_by_id, reply_sent)
                sstats["replies"] = reply_sent[0]
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": res})
                continue

            # — GraphRAG-память /index —
            if tname in ("memory_search", "memory_entity", "memory_media", "memory_overview"):
                sstats["memory"] = sstats.get("memory", 0) + 1
                if tname == "memory_search":
                    res = await _index_tool_search(chat_id, args.get("query", ""), args.get("kind", "all"))
                elif tname == "memory_overview":
                    res = await _index_tool_overview(chat_id, args.get("topic", ""))
                elif tname == "memory_entity":
                    res = await _index_tool_entity(chat_id, args.get("name", ""))
                else:
                    try:
                        cnt = int(args.get("count", 1) or 1)
                    except (ValueError, TypeError):
                        cnt = 1
                    q_img = images[0]["bytes"] if images else None  # приложенная картинка → визуальный поиск
                    res = await _index_tool_media(chat_id, args.get("query", ""), cnt,
                                                  visual=bool(args.get("visual")), query_image=q_img)
                log("ASK", f"Память /index {tname}: {_idx_snip(args.get('query') or args.get('topic') or args.get('name'), 80)} → {len(res)} симв")
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": res})
                continue

            if tname != "telegram_search":
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Неизвестный инструмент: {tname}"
                })
                continue

            sstats["calls"] += 1
            query = (args.get("query") or "").strip()
            days = None
            try:
                if args.get("days") is not None:
                    days = int(args["days"])
            except (ValueError, TypeError):
                pass

            if not query:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "Ошибка: пустой поисковый запрос"
                })
                continue

            log("ASK", f"DeepSeek ищет: \"{query}\"" + (f" (за {days} дн.)" if days else ""))

            # Выполняем поиск
            results = await search_channels(query, per_channel=3, total=10, since_days=days)
            sstats["posts"] += len(results)

            if results:
                result_lines = []
                for _date, ent, msg_id, raw in results:
                    uname = getattr(ent, "username", None)
                    src = f"@{uname}" if uname else getattr(ent, "title", "канал")
                    link = build_msg_link(ent, msg_id)
                    result_lines.append(f"📅 {_fmt_date(_date)} | [{src}] {_preview(raw, 300)}\n{link}")
                search_result = f"Найдено {len(results)} результатов по запросу «{query}» (отсортировано от новых к старым):\n\n" + "\n\n".join(result_lines)
            else:
                search_result = f"По запросу «{query}» ничего не найдено."

            log("ASK", f"Результаты поиска по \"{query}\": {len(results)} постов")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": search_result,
            })

    # Лимит итераций — запрашиваем финальный ответ без инструментов
    log("ASK", "Достигнут лимит итераций, запрашиваю финальный ответ")
    try:
        response = await asyncio.to_thread(
            llm.chat.completions.create,
            model=model_id,
            messages=messages,
            max_tokens=ASK_MAX_TOKENS,
            temperature=1.0,
        )
        content = _extract_content(response.choices[0].message)
        if content:
            _log_search_summary()
            return content
    except Exception as e:
        log("ASK", f"Ошибка финального ответа: {e}")

    _log_search_summary()
    return await generate_ask_reply(context, question, caller=caller)


def _lyric_steps(text: str, chunk_size: int):
    """Нарезает текст на шаги анимации по ~chunk_size символов так, чтобы граница шага падала
    РОВНО на начало следующего слова — целостность и слов, и пробелов:
      • если граница попала в середину слова — тянем до конца слова (слово открывается целиком);
      • затем поглощаем хвостовой пробельный прогон (пробелы/переносы/отступы) в этот же шаг,
        чтобы прогоны не расщеплялись, перенос \\n не отрывался от слова (пауза 0.8с срабатывает),
        а Telegram не «съедал» висячий хвостовой пробел рывком.
    chunk_size задаёт скорость; слова/пробельные прогоны держатся целиком (шаг может быть длиннее)."""
    steps, i, n = [], 0, len(text)
    while i < n:
        j = min(i + max(1, chunk_size), n)
        # j внутри слова (символ до и символ на границе оба не-пробельные) → дотянуть до конца слова
        if j < n and not text[j - 1].isspace() and not text[j].isspace():
            while j < n and not text[j].isspace():
                j += 1
        # поглотить хвостовой пробельный прогон до начала следующего слова (не резать пробелы)
        while j < n and text[j].isspace():
            j += 1
        steps.append(text[i:j])
        i = j
    return steps


async def print_lyrics(chat_id, text, chunk_size=3):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return
    steps = _lyric_steps(text, chunk_size)
    current_text = steps[0]
    msg = await client.send_message(chat_id, current_text)
    for chunk in steps[1:]:
        await asyncio.sleep(0.8 if "\n" in chunk else 0.2)
        current_text += chunk
        try:
            await client.edit_message(chat_id, msg, current_text)
        except MessageNotModifiedError:
            continue
        except FloodWaitError as e:
            log("SONG", f"Ожидание FloodWait: {e.seconds} секунд")
            await asyncio.sleep(e.seconds + 1)
            try:
                await client.edit_message(chat_id, msg, current_text)
            except Exception as retry_error:
                log("SONG", f"Повтор редактирования не удался: {retry_error}")
                break
        except Exception as e:
            log("SONG", f"Ошибка при редактировании: {e}")
            traceback.print_exc()
            break


def _media_key(m):
    # Ключ кэша по уникальному file-id (стабилен для одного файла, в т.ч. при пересылке).
    # Fallback на chat_id:msg_id, если id недоступен.
    mid = getattr(getattr(m, "photo", None), "id", None) or getattr(getattr(m, "document", None), "id", None)
    return f"file:{mid}" if mid else f"{m.chat_id}:{m.id}"


async def process_media_cached(m, vision_model: str = None, detail: str = "high", mstats: dict = None, inline_ids: set = None, inline_images: list = None, photo_mode: str = "ocr"):
    """Текст медиа (описание/транскрипт) с кэшем по file-id. None — если медиа нет.
    mstats — опциональный аккумулятор статистики (photos/voice/audio/video_note + hit/miss).
    inline_ids/inline_images (режим /ask -g): фото НЕ описываются, а сами байты собираются
    в inline_images, в тексте — плейсхолдер [Картинка #k]. Голос/аудио/кружок — без изменений.
    photo_mode: "ocr" (дефолт, дёшево — LlamaParse вытаскивает текст с фото; без текста — плейсхолдер)
    или "vision" (флаг -m в /ask — полное описание vision-моделью, как раньше).
    Голос/аудио/кружки от photo_mode НЕ зависят (всегда STT)."""
    def _bump(kind):
        if mstats is not None:
            mstats[kind] = mstats.get(kind, 0) + 1
    key = _media_key(m)
    if m.photo:
        _bump("photos")
        # Direct-vision: вместо описания собираем сами картинки. inline_ids — dict {msg_id: idx}
        # (idx = детерминированная позиция в хронологии); собираем только отобранные (самые свежие).
        if inline_ids is not None:
            cap_txt = f" {m.raw_text}" if m.raw_text else ""
            idx = inline_ids.get(getattr(m, "id", None))
            if idx is not None:
                try:
                    img = await m.download_media(bytes)
                except Exception as e:
                    log("ASK", f"-g: не удалось скачать фото: {e}")
                    img = None
                if img:
                    inline_images.append({"idx": idx, "bytes": img, "caption": m.raw_text or ""})
                    return f"[Картинка #{idx}{cap_txt}]"
                return f"[Картинка (не скачалась){cap_txt}]"
            return f"[Картинка (пропущена — лимит {DIRECT_VISION_MAX_IMAGES}){cap_txt}]"
        text_part = f" {m.raw_text}" if m.raw_text else ""
        if photo_mode == "ocr" and llama_cloud_api_key:
            # OCR-режим (дефолт): берём только ТЕКСТ с фото. Свой кэш-ключ "ocr:*" —
            # vision-описания живут на старых ключах, кэши не смешиваются.
            okey = "ocr:" + key
            cached = MEDIA_CACHE.get(okey)
            if cached is None:
                _bump("miss")
                img = await m.download_media(bytes)
                ocr = await llama_ocr(img) if img else None
                if ocr is None:  # OCR недоступен/упал → фолбэк на vision (деградация в качество)
                    cached = await describe_image(img, m.raw_text or "", model=vision_model or get_active_media_model(), detail=detail)
                    if cached and cached not in MEDIA_FAILURE_MARKERS and cached != (m.raw_text or ""):
                        _media_cache_set(key, cached)  # vision-ключ: пригодится и для -m
                    return f"[Фото: {cached}]{text_part}"
                cached = ocr or "[без текста]"
                _media_cache_set(okey, cached)
            else:
                _bump("hit")
            if cached == "[без текста]":
                return f"[Фото (без текста)]{text_part}"
            return f"[Фото, текст: {cached}]{text_part}"
        vm = vision_model or get_active_media_model()
        # Ключ модель-НЕзависимый: описал один раз — переиспользуем любой моделью (не описываем заново при смене модели)
        cached = MEDIA_CACHE.get(key)
        if cached is None:
            _bump("miss")
            img = await m.download_media(bytes)
            cached = await describe_image(img, m.raw_text or "", model=vm, detail=detail)
            if cached and cached not in MEDIA_FAILURE_MARKERS and cached != (m.raw_text or ""):
                _media_cache_set(key, cached)
        else:
            _bump("hit")
        return f"[Фото: {cached}]{text_part}"
    if m.voice or m.audio:
        _bump("voice" if m.voice else "audio")
        cached = MEDIA_CACHE.get(key)
        if cached is None:
            _bump("miss")
            audio = await m.download_media(bytes)
            fmt = "ogg" if m.voice else _audio_format(m)
            cached = await transcribe_audio(audio, fmt)
            if cached and cached not in MEDIA_FAILURE_MARKERS:
                _media_cache_set(key, cached)
        else:
            _bump("hit")
        text_part = f" {m.raw_text}" if m.raw_text else ""
        return f"[Аудио: {cached}]{text_part}"
    if m.video_note:
        _bump("video_note")
        cached = MEDIA_CACHE.get(key)
        if cached is None:
            _bump("miss")
            cached = await extract_video_note_content(m)
            if cached and cached not in MEDIA_FAILURE_MARKERS:
                _media_cache_set(key, cached)
        else:
            _bump("hit")
        return cached
    if m.document and not getattr(m, "sticker", False) and not getattr(m, "gif", False) \
            and not m.video and not m.video_note and not m.voice and not m.audio:
        # Текстовые файлы (.txt/.md/код/json/csv…) читаем по умолчанию — содержимое в контекст.
        # Не-текст/большие/бинарь → None (фолбэк на плейсхолдер [Файл] из _media_tag).
        f = getattr(m, "file", None)
        name = getattr(f, "name", None) or "файл"
        mime = (getattr(f, "mime_type", None) or "").lower()
        ext = (getattr(f, "ext", None) or "").lower().lstrip(".")
        size = getattr(f, "size", 0) or 0
        is_text = mime.startswith("text/") or mime in TEXT_MIME or ext in TEXT_EXT
        if not is_text or size > DOC_MAX_BYTES:
            return None
        _bump("doc")
        cached = MEDIA_CACHE.get(key)
        if cached is None:
            _bump("miss")
            try:
                raw = await m.download_media(bytes)
            except Exception as e:
                log("ASK", f"Текстовый файл «{name}» не скачался: {e}")
                return None
            if not raw:
                return None
            content = raw.decode("utf-8", errors="replace")
            # бинарь под видом текста: много NUL / символов замены → плейсхолдер
            if content.count("\x00") or content.count("�") > max(20, len(content) // 20):
                return None
            if len(content) > DOC_MAX_CHARS:
                content = content[:DOC_MAX_CHARS].rstrip() + "\n…(файл обрезан)"
            cached = content
            _media_cache_set(key, cached)
        else:
            _bump("hit")
        text_part = f" {m.raw_text}" if m.raw_text else ""
        return f"[Файл «{name}»:\n{cached}\n]{text_part}"
    return None


def _media_tag(msg) -> str:
    # Короткая пометка наличия медиа (без AI-описания). None — медиа нет или пропускаем.
    if getattr(msg, "sticker", False):
        return None  # стикеры пропускаем — бесполезны для контекста, жрут токены
    if msg.photo:
        return "Фото"
    if msg.voice:
        return "Голосовое"
    if msg.audio:
        return "Аудио"
    if msg.video_note:
        return "Видеокружок"
    if getattr(msg, "gif", False):
        return "GIF"
    if msg.video:
        return "Видео"
    if getattr(msg, "contact", False):
        return "Контакт"
    if getattr(msg, "geo", False):
        return "Геолокация"
    if msg.document:
        return "Файл"
    return None


def _label_for(msg, sender) -> str:
    if msg.out:
        return _owner_label()
    return _user_label(sender)


def _fmt_ts(dt) -> str:
    # Компактное время в МСК: ЧЧ:ММ если сегодня, иначе ДД.ММ ЧЧ:ММ
    try:
        local = dt.astimezone(MSK)
        now = datetime.now(MSK)
        return local.strftime("%H:%M") if local.date() == now.date() else local.strftime("%d.%m %H:%M")
    except Exception:
        return ""


def _forward_src(msg) -> str:
    # Источник пересланного сообщения (без сети, по кэшу). None — не переслано.
    fwd = getattr(msg, "forward", None)
    if not fwd:
        return None
    chat = getattr(fwd, "chat", None)
    if chat is not None:
        nm = getattr(chat, "title", None) or getattr(chat, "username", None)
        if nm:
            return nm
    sender = getattr(fwd, "sender", None)
    if sender is not None:
        u = getattr(sender, "username", None)
        if u:
            return f"@{u}"
        nm = getattr(sender, "first_name", None) or getattr(sender, "title", None)
        if nm:
            return nm
    return getattr(fwd, "from_name", None)


async def _reply_info(msg, by_id=None, net_budget=None, rep_stats=None) -> str:
    """Метка «↩ автор: «цитата»» для reply-сообщения.
    Сначала ищет target в by_id (без сети) — для большинства replies таргет в той же выборке.
    Если не нашли — фоллбэк в сеть (`get_reply_message()`), но в пределах net_budget.
    Когда бюджет исчерпан и target не в batch — возвращает голую метку «↩» без цитаты."""
    rto = getattr(msg, "reply_to", None)
    if not rto:
        return None
    rto_id = getattr(rto, "reply_to_msg_id", None)
    qtext = getattr(rto, "quote_text", None)  # выделенный пользователем фрагмент (partial quote) — лежит прямо в reply_to, сеть не нужна
    rep = by_id.get(rto_id) if (by_id and rto_id) else None
    if rep is not None and rep_stats is not None:
        rep_stats["hit"] = rep_stats.get("hit", 0) + 1
    # Сетевой fallback с глобальным бюджетом
    if rep is None and net_budget is not None and net_budget.get("remaining", 0) > 0:
        try:
            rep = await msg.get_reply_message()
        except Exception:
            rep = None
        net_budget["remaining"] -= 1
        net_budget["used"] = net_budget.get("used", 0) + 1
        if rep is not None and rep_stats is not None:
            rep_stats["miss"] = rep_stats.get("miss", 0) + 1
    elif rep is None and net_budget is None:
        # Старый путь (без бюджета) — используется, если функция вызвана из других мест
        try:
            rep = await msg.get_reply_message()
        except Exception:
            rep = None
    # Telethon в редких случаях (reply на сообщение в другом канале, reply-to-story,
    # batch get_messages) возвращает TotalList/list вместо одного Message — нормализуем.
    if isinstance(rep, (list, tuple)):
        rep = next((r for r in rep if r is not None), None)
    if rep is None:
        if rep_stats is not None:
            rep_stats["no_quote"] = rep_stats.get("no_quote", 0) + 1
        # target вне выборки/бюджета — но фрагмент-цитата есть в reply_to, покажем хотя бы его
        if qtext:
            return f"↩ на фрагмент: «{_preview(qtext, 80)}»"
        return "↩"
    if getattr(rep, "out", False):
        rauthor = _owner_label()
    else:
        rauthor = _user_label(getattr(rep, "sender", None))
    head = ("↩ " + rauthor).strip()
    if qtext:
        # отвечают на КОНКРЕТНЫЙ фрагмент — показываем именно его, а не начало всего сообщения
        head += f" (на фрагмент: «{_preview(qtext, 80)}»)"
    else:
        quote = _preview(getattr(rep, "raw_text", None) or (_media_tag(rep) or ""), 50)
        if quote:
            head += f": «{quote}»"
    return head


def _assemble_body(msg, media_body) -> str:
    if media_body is not None:
        return media_body
    tag = _media_tag(msg)
    cap = (msg.raw_text or "").strip()
    if tag:
        return f"[{tag}]" + (f" {cap}" if cap else "")
    return cap or ""


async def _render_unit(msg, text_only: bool, anchor_id=None, vision_model: str = None, detail: str = "high", mstats: dict = None, by_id: dict = None, net_budget: dict = None, rep_stats: dict = None, inline_ids: set = None, inline_images: list = None, photo_mode: str = "ocr") -> dict:
    """Рендерит одно сообщение в части для последующей сборки блоков."""
    sender = None if msg.out else (msg.sender if text_only else (msg.sender or await msg.get_sender()))
    label = _label_for(msg, sender)
    akey = "me" if msg.out else (getattr(sender, "username", None) or getattr(sender, "id", None) or "?")

    marked = False  # есть метки (reply/forward/якорь) — такие блоки не склеиваем
    fwd = _forward_src(msg)
    if fwd:
        label += f" ⤷ из {fwd}"
        marked = True
    # Reply-квота нужна и под -t. In-batch lookup убирает 80–95% сетевых вызовов;
    # для остальных — глобальный бюджет (см. REPLY_NETWORK_BUDGET).
    rep = await _reply_info(msg, by_id=by_id, net_budget=net_budget, rep_stats=rep_stats)
    if rep:
        label += f" {rep}"
        marked = True

    media_body = None
    if not text_only:
        try:
            media_body = await process_media_cached(msg, vision_model, detail=detail, mstats=mstats, inline_ids=inline_ids, inline_images=inline_images, photo_mode=photo_mode)
        except Exception as e:
            log("ASK", f"Ошибка обработки медиа в контексте: {e}")
    body = _assemble_body(msg, media_body)

    is_anchor = anchor_id is not None and getattr(msg, "id", None) == anchor_id
    if is_anchor:
        label += " ← ВОПРОС ОБ ЭТОМ"
        marked = True

    return {
        "akey": akey,
        "label": label,
        "ts": _fmt_ts(getattr(msg, "date", None)),
        "body": body,
        "gid": getattr(msg, "grouped_id", None),
        "mid": getattr(msg, "id", None),  # для пометки #id в контексте (reply_to_messages)
        "marked": marked,
        "is_anchor": is_anchor,  # цель reply-/ask — закрепляется при обрезке (см. assemble_context)
        "failed": sum(body.count(mk) for mk in MEDIA_FAILURE_MARKERS),
    }


def _group_segments(messages):
    # Группирует подряд идущие сообщения с общим grouped_id (альбом) в один сегмент.
    segs, i = [], 0
    while i < len(messages):
        gid = getattr(messages[i], "grouped_id", None)
        if gid:
            j = i + 1
            while j < len(messages) and getattr(messages[j], "grouped_id", None) == gid:
                j += 1
            segs.append(messages[i:j])
            i = j
        else:
            segs.append([messages[i]])
            i += 1
    return segs


async def _render_album_segment(group, text_only: bool, anchor_id=None, vision_model: str = None, detail: str = "high", mstats: dict = None, by_id: dict = None, net_budget: dict = None, rep_stats: dict = None, inline_ids: set = None, inline_images: list = None, photo_mode: str = "ocr") -> dict:
    """Альбом (несколько сообщений с общим grouped_id) → один юнит, фото описываются одним запросом."""
    first = group[0]
    sender = None if first.out else (first.sender if text_only else (first.sender or await first.get_sender()))
    label = _label_for(first, sender)
    akey = "me" if first.out else (getattr(sender, "username", None) or getattr(sender, "id", None) or "?")

    marked = False
    fwd = _forward_src(first)
    if fwd:
        label += f" ⤷ из {fwd}"
        marked = True
    is_anchor = anchor_id is not None and any(getattr(m, "id", None) == anchor_id for m in group)
    if is_anchor:
        label += " ← ВОПРОС ОБ ЭТОМ"
        marked = True

    n = len(group)
    caption = next((m.raw_text.strip() for m in group if (m.raw_text or "").strip()), "")
    photos = [m for m in group if getattr(m, "photo", None)]
    others = [m for m in group if not getattr(m, "photo", None)]
    tags = [f"[{_media_tag(m)}]" for m in others if _media_tag(m)]

    if inline_ids is not None:
        # direct-vision (/ask -g): собираем фото альбома сами, без описания
        parts = []
        for m in photos:
            cap_txt = f" {m.raw_text}" if m.raw_text else ""
            idx = inline_ids.get(getattr(m, "id", None))
            if idx is not None:
                try:
                    img = await m.download_media(bytes)
                except Exception as e:
                    log("ASK", f"-g альбом: не удалось скачать фото: {e}")
                    img = None
                if img:
                    inline_images.append({"idx": idx, "bytes": img, "caption": m.raw_text or ""})
                    parts.append(f"[Картинка #{idx}{cap_txt}]")
                else:
                    parts.append(f"[Картинка (не скачалась){cap_txt}]")
            else:
                parts.append(f"[Картинка (пропущена — лимит {DIRECT_VISION_MAX_IMAGES}){cap_txt}]")
        desc = "\n".join(parts)
    elif text_only or not photos:
        desc = ""
    elif photo_mode == "ocr" and llama_cloud_api_key:
        # OCR-режим: альбом обрабатываем пофайлово (describe_album — это vision-описание одним
        # запросом, для OCR не нужно); кэш per-photo внутри process_media_cached (ключи ocr:*).
        parts = []
        for m in photos:
            pm = await process_media_cached(m, vision_model, detail=detail, mstats=mstats, photo_mode="ocr")
            if pm:
                parts.append(pm)
        desc = "\n".join(parts)
    else:
        vm = vision_model or get_active_media_model()
        key = "album:" + ":".join(_media_key(m) for m in photos)  # модель-независимо
        desc = MEDIA_CACHE.get(key)
        npix = len(photos)
        if desc is None:
            # album-cache miss: всегда +N photos. miss/hit считаем здесь или внутри fallback'a (избежать двойного счёта).
            imgs = []
            for m in photos:
                try:
                    b = await m.download_media(bytes)
                    if b:
                        imgs.append(b)
                except Exception as e:
                    log("MEDIA", f"альбом: не удалось скачать фото: {e}")
            desc = await describe_album(imgs, caption, model=vm, detail=detail) if imgs else ""
            if not desc:  # фоллбэк: пофайлово — внутренний process_media_cached сам считает hit/miss/photos
                parts = []
                for m in photos:
                    pm = await process_media_cached(m, vm, detail=detail, mstats=mstats)
                    if pm:
                        parts.append(pm)
                desc = "\n".join(parts)
            else:
                # describe_album прошёл целиком: все N фото описаны (свежие)
                if mstats is not None:
                    mstats["photos"] = mstats.get("photos", 0) + npix
                    mstats["miss"] = mstats.get("miss", 0) + npix
                if desc not in MEDIA_FAILURE_MARKERS:
                    _media_cache_set(key, desc)
        else:
            # album-cache hit: все N фото переиспользованы (items-level)
            if mstats is not None:
                mstats["photos"] = mstats.get("photos", 0) + npix
                mstats["hit"] = mstats.get("hit", 0) + npix

    body = f"[Альбом {n}]" + (f" {caption}" if caption else "")
    if desc:
        body += f"\n{desc}"
    if tags:
        body += "\n" + "\n".join(tags)

    return {
        "akey": akey,
        "label": label,
        "ts": _fmt_ts(getattr(first, "date", None)),
        "body": body,
        "mid": getattr(first, "id", None),  # альбом → id первого сообщения (reply_to_messages)
        "marked": marked,
        "is_anchor": is_anchor,  # цель reply-/ask — закрепляется при обрезке (см. assemble_context)
        "failed": sum(body.count(mk) for mk in MEDIA_FAILURE_MARKERS),
    }


async def _render_segment(seg, text_only: bool, anchor_id=None, vision_model: str = None, detail: str = "high", mstats: dict = None, by_id: dict = None, net_budget: dict = None, rep_stats: dict = None, inline_ids: set = None, inline_images: list = None, photo_mode: str = "ocr") -> dict:
    if len(seg) == 1:
        u = await _render_unit(seg[0], text_only, anchor_id, vision_model, detail, mstats=mstats, by_id=by_id, net_budget=net_budget, rep_stats=rep_stats, inline_ids=inline_ids, inline_images=inline_images, photo_mode=photo_mode)
        u.pop("gid", None)  # gid больше не используется на этапе склейки
        return u
    return await _render_album_segment(seg, text_only, anchor_id, vision_model, detail, mstats=mstats, by_id=by_id, net_budget=net_budget, rep_stats=rep_stats, inline_ids=inline_ids, inline_images=inline_images, photo_mode=photo_mode)


def _needs_media(m) -> bool:
    return bool(getattr(m, "photo", None) or getattr(m, "voice", None)
                or getattr(m, "audio", None) or getattr(m, "video_note", None))


async def assemble_context(messages, text_only: bool, anchor_id=None, progress_cb=None, vision_model: str = None, detail: str = "high", safety_override: float = None, inline_ids: set = None, inline_images: list = None, photo_mode: str = "ocr", include_ids: bool = False):
    """Строит контекст: параллельный рендер + склейка альбомов и подряд идущих реплик автора.
    Возвращает (context_str, dropped_blocks, failed_media, ctx_tokens). progress_cb(done, total, failed).
    safety_override — если задан, перебивает per-model safety (используется при ретрае overflow).
    include_ids — каждую строку-сообщение префиксовать её #id (чтобы модель могла адресно ответить
    реплаем через reply_to_messages); см. ask_agentic."""
    if not messages:
        return "", 0, 0, 0
    t_render_start = time.time()
    segments = _group_segments(messages)
    sem = asyncio.Semaphore(MEDIA_CONCURRENCY)
    # Потолок медиа: на огромных N обрабатываем только свежие MEDIA_MAX_ITEMS медиа-сегментов
    # (segments хронологичны → хвост = новые), старые идут текстовыми плейсхолдерами.
    # Иначе сотни параллельных скачиваний/base64 раздувают память до OOM-kill контейнера.
    media_idx = [] if text_only else [i for i, s in enumerate(segments) if any(_needs_media(m) for m in s)]
    skip_media = set(media_idx[:-MEDIA_MAX_ITEMS]) if len(media_idx) > MEDIA_MAX_ITEMS else set()
    if skip_media:
        log("ASK", f"Медиа-потолок: сегментов с медиа {len(media_idx)} > {MEDIA_MAX_ITEMS} — старые {len(skip_media)} как плейсхолдеры")
    media_total = 0 if text_only else len(media_idx) - len(skip_media)
    done = 0
    failed_total = 0
    mstats = {"photos": 0, "voice": 0, "audio": 0, "video_note": 0, "doc": 0, "hit": 0, "miss": 0}
    # In-batch lookup для reply-target'ов: убирает 80–95% сетевых вызовов на больших N.
    by_id = {getattr(m, "id", None): m for m in messages if getattr(m, "id", None) is not None}
    # Батч-префетч target-сообщений, которых нет в by_id (типичный сценарий под фильтром @user,
    # где почти все replies указывают на сообщения других людей). Один get_messages(ids=[100])
    # вместо 100 одиночных get_reply_message() — снижает время в десятки раз.
    missing_ids = set()
    for m in messages:
        rto = getattr(m, "reply_to", None)
        if rto:
            rto_id = getattr(rto, "reply_to_msg_id", None)
            if rto_id and rto_id not in by_id:
                missing_ids.add(rto_id)
    if missing_ids:
        chat_id = getattr(messages[0], "chat_id", None)
        if chat_id is not None:
            missing_list = list(missing_ids)
            t_pf = time.time()
            fetched_count = 0
            for i in range(0, len(missing_list), 100):
                chunk = missing_list[i:i + 100]
                try:
                    fetched = await client.get_messages(chat_id, ids=chunk)
                    for fm in (fetched or []):
                        if fm is not None and getattr(fm, "id", None) is not None:
                            by_id[fm.id] = fm
                            fetched_count += 1
                except Exception as e:
                    log("ASK", f"Batch reply prefetch ошибка (чанк {i}-{i+len(chunk)}): {e}")
            log("ASK", f"Reply-prefetch: запросил {len(missing_list)} target-ID, получил {fetched_count} за {time.time()-t_pf:.1f}с")
    net_budget = {"remaining": REPLY_NETWORK_BUDGET, "used": 0}
    rep_stats = {"hit": 0, "miss": 0, "no_quote": 0}

    async def render(idx, seg):
        nonlocal done, failed_total
        seg_text_only = text_only or (idx in skip_media)  # за потолком — медиа не качаем (плейсхолдер)
        async with sem:
            u = await _render_segment(seg, seg_text_only, anchor_id, vision_model, detail, mstats=mstats, by_id=by_id, net_budget=net_budget, rep_stats=rep_stats, inline_ids=inline_ids, inline_images=inline_images, photo_mode=photo_mode)
        failed_total += u.get("failed", 0)
        if not seg_text_only and any(_needs_media(m) for m in seg):
            done += 1
            if progress_cb:
                await progress_cb(done, media_total, failed_total)
            if done % 50 == 0:  # инкрементально сохраняем кэш — переживёт краш/рестарт посреди большого /ask
                save_media_cache()
        return u

    units = await asyncio.gather(*[render(i, s) for i, s in enumerate(segments)])
    t_render = time.time() - t_render_start

    # Сводка по медиа (если что-то было)
    mtot = mstats["photos"] + mstats["voice"] + mstats["audio"] + mstats["video_note"] + mstats["doc"]
    if mtot:
        hr = round(100 * mstats["hit"] / mtot, 1)
        log("ASK", f"Медиа: {mtot} (фото {mstats['photos']} · голос {mstats['voice']} · аудио {mstats['audio']} · кружок {mstats['video_note']} · файлы {mstats['doc']}) · кэш-хит {mstats['hit']}/{mtot} ({hr}%) · новых {mstats['miss']} · сбоев {failed_total}")

    # Склейка: подряд идущие сообщения одного автора без меток → один блок (альбомы уже самоформатированы).
    # При include_ids каждая строка-сообщение получает префикс #id, чтобы модель адресовала reply_to_messages.
    def _line(u):
        if include_ids and u.get("mid"):
            return f"#{u['mid']} {u['body']}"
        return u["body"]
    blocks = []
    for u in units:
        if not u["body"]:
            continue
        if blocks and not u["marked"] and not blocks[-1]["marked"] and blocks[-1]["akey"] == u["akey"]:
            blocks[-1]["lines"].append(_line(u))
        else:
            blocks.append({
                "akey": u["akey"], "label": u["label"], "ts": u["ts"],
                "lines": [_line(u)], "marked": u["marked"], "is_anchor": u.get("is_anchor", False),
            })

    out = [f"[{(b['ts'] + ' ' + b['label']).strip()}]: " + "\n".join(b["lines"]) for b in blocks]
    anchor_idx = next((i for i, b in enumerate(blocks) if b.get("is_anchor")), None)  # цель reply-/ask

    # Автообрезка под окно активной модели: держим самые свежие блоки, что влезают.
    # Бюджет в токенах; запас прочности под чужие токенизаторы (см. CTX_TOKEN_SAFETY).
    # NB: тестировал batch-encoding tiktoken — на типичной нагрузке оказался медленнее
    # per-block (FFI/parallel-setup оверхед), оставлен per-block + early-break.
    t_trunc_start = time.time()
    safety = safety_override if safety_override is not None else active_ctx_safety()
    budget = max(2000, int((active_context_window() - CTX_RESERVE_TOKENS) / safety))
    kept_idx, total, truncated = [], 0, False
    for i in range(len(out) - 1, -1, -1):  # с конца — новейшие
        add = count_tokens(out[i]) + 1  # +1 на разделитель блоков
        if kept_idx and total + add > budget:
            truncated = True
            break
        kept_idx.append(i)
        total += add
    # Закрепляем якорный блок (цель reply-/ask): даже если он самый старый и не влез — без
    # него модель отвечает «про это», не видя «этого». Якорь 1:1 = один блок (он marked).
    anchor_pinned = False
    if truncated and anchor_idx is not None and anchor_idx not in kept_idx:
        total += count_tokens(out[anchor_idx]) + 1
        kept_idx.append(anchor_idx)
        anchor_pinned = True
    kept_idx.sort()
    kept = [out[i] for i in kept_idx]
    t_trunc = time.time() - t_trunc_start
    dropped = 0
    window = active_context_window()
    pct = round(100 * total / window, 1) if window else 0
    enc = "tiktoken" if _ENC is not None else "оценка по символам"
    if truncated:
        dropped = len(out) - len(kept)
        # маркер ставим ПОСЛЕ закреплённого якоря (он самый старый, идёт первым)
        note = " (цель вопроса сохранена выше)" if anchor_pinned else ""
        kept.insert(1 if anchor_pinned else 0, f"[…{dropped} более старых сообщений опущено — не влезли в окно модели{note}…]")
        log("ASK", f"Контекст обрезан под окно {_fmt_ctx(window)}: оставлено {len(kept) - 1}/{len(out)} блоков, ~{total} ток ({enc}) = {pct}% окна (бюджет {budget} ток, safety×{safety:.2f})" + (" · якорь закреплён" if anchor_pinned else ""))
    else:
        log("ASK", f"Контекст готов: ~{total} ток ({enc}) из окна {_fmt_ctx(window)} = {pct}% занято, блоков {len(kept)} (бюджет {budget} ток, safety×{safety:.2f})")

    # Диагностика sub-фаз сборки контекста (быстро видно, что съело время на больших N)
    rep_total = rep_stats["hit"] + rep_stats["miss"] + rep_stats["no_quote"]
    if rep_total or t_render > 0.5 or t_trunc > 0.5:
        no_q = f" · без цитат: {rep_stats['no_quote']}" if rep_stats["no_quote"] else ""
        log("ASK", f"Подэтапы контекста: рендер={t_render:.1f}с · обрезка={t_trunc:.1f}с · "
                   f"reply: in-batch {rep_stats['hit']} · сеть {net_budget['used']}/{REPLY_NETWORK_BUDGET}{no_q}")
    # /ask -g: выкидываем картинки из блоков, что не пережили обрезку — иначе модель
    # получает байты без текстовой ссылки [Картинка #k]. Граница (\d+) против матча #1 в #12.
    if inline_images:
        surv = {int(x) for x in re.findall(r"Картинка #(\d+)", "\n".join(kept))}
        before = len(inline_images)
        inline_images[:] = [im for im in inline_images if im["idx"] in surv]
        if len(inline_images) != before:
            log("ASK", f"-g: отброшено {before - len(inline_images)} картинок из срезанных блоков")
    return "\n\n".join(kept), dropped, failed_total, total


# --- Команды ---

async def search_channels(query: str, per_channel: int = 5, total: int = 10, since_days: int = None) -> list:
    # Параллельный поиск по каналам. Возвращает (date, entity, msg_id, raw_text), от новых к старым.
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days) if since_days else None
    fetch = per_channel * 3 if cutoff else per_channel
    sem = asyncio.Semaphore(SEARCH_CONCURRENCY)

    async def search_one(ch):
        ent = await resolve_channel(ch)
        if ent is None:
            return []
        out = []
        async with sem:
            try:
                async for m in client.iter_messages(ent, search=query, limit=fetch):
                    if m.raw_text and (cutoff is None or m.date >= cutoff):
                        out.append((m.date, ent, m.id, m.raw_text))
            except FloodWaitError as e:
                log("SEARCH", f"FloodWait {e.seconds}с на канале {getattr(ent, 'title', '?')}")
                await asyncio.sleep(e.seconds + 1)
            except Exception as e:
                log("SEARCH", f"Ошибка поиска в {getattr(ent, 'title', '?')}: {e}")
        return out

    chunks = await asyncio.gather(*[search_one(ch) for ch in get_tracked()])
    results = [r for chunk in chunks for r in chunk]
    results.sort(key=lambda r: r[0], reverse=True)
    return results[:total]


async def _collect_history_parallel(chat_id, n, base_offset_id, from_user=None):
    """Собирает ~n сообщений старше base_offset_id (0=с конца) ПАРАЛЛЕЛЬНЫМИ окнами через add_offset
    (позиционный сдвиг — надёжен при дырках id от удалённых). Возвращает список Message (с возможными
    дублями на стыках окон — дедуп у вызывающего). FloodWait в окне → ждём и возвращаем частичное.
    Самомасштабируется: при малом n — меньше воркеров (мелкие /ask не дробим зря)."""
    target = int(n * COLLECT_OVERFETCH) + 10
    workers = max(1, min(COLLECT_WORKERS, -(-target // COLLECT_MIN_PER_WORKER)))  # ceil(target/min_per)
    per = -(-target // workers)  # ceil — сообщений на окно

    async def _window(k):
        out = []
        try:
            async for m in client.iter_messages(chat_id, offset_id=base_offset_id,
                                                 add_offset=k * per, limit=per, from_user=from_user):
                out.append(m)
        except FloodWaitError as e:
            log("ASK", f"Сбор: окно {k} FloodWait {e.seconds}с — жду и возвращаю частичное ({len(out)})")
            await asyncio.sleep(e.seconds + 1)
        except Exception as e:
            log("ASK", f"Сбор: окно {k} ошибка: {e} (собрано {len(out)})")
        return out

    chunks = await asyncio.gather(*[_window(k) for k in range(workers)])
    merged = [m for c in chunks for m in c]
    log("ASK", f"Параллельный сбор: воркеров={workers}, окно={per}, чанки={[len(c) for c in chunks]} → {len(merged)} (с дублями)")
    return workers, merged


async def _slash_for_other_bot(event) -> bool:
    """True, если /команда написана в личке с ботом (например @gick_hunterhermess_bot):
    там слэш адресован этому боту, а не юзерботу — пропускаем, чтобы не мешать.
    Команды через точку (.ask, .model, .voice …) продолжают работать и в чатах с ботами."""
    if not (event.raw_text or "").startswith("/"):
        return False
    if not event.is_private:
        return False
    try:
        chat = await event.get_chat()
    except Exception:
        return False
    return bool(getattr(chat, "bot", False))


@client.on(events.NewMessage(pattern=r"^[./]ask\s+(\d+)((?:\s+-[tcdvgm]+)+)?((?:\s+!?@\w+)+)?\s+(.+)"))
async def ask_command(event):
    if await _slash_for_other_bot(event):
        return  # /команда в личке с ботом адресована ему, не юзерботу (используй .ask)
    is_owner = event.out
    if not is_owner and event.sender_id not in ALLOWED_USERS:
        return  # не владелец и не в списке разрешённых
    n = int(event.pattern_match.group(1))
    reply_target_id = getattr(event, "reply_to_msg_id", None)  # если /ask — ответ на сообщение, шлём ответ реплаем на него
    flags = event.pattern_match.group(2) or ""
    direct_vision = "g" in flags  # -g: отдать картинки напрямую отвечающей модели (её vision)
    text_only = "t" in flags and not direct_vision  # -g включает медиа-обработку для фото
    must_search = "c" in flags
    debug = "d" in flags  # дамп полного user-message в asks/<ts>_<event_id>.txt
    want_voice = "v" in flags  # -v: ответить голосом (озвучка через Gemini TTS)
    photo_mode = "vision" if "m" in flags else "ocr"  # -m: фото описывает vision-модель; дефолт — дешёвый OCR
    # Режим голоса для промпта: force (флаг -v) / auto (включён /voice auto) / off
    voice_mode = "force" if (want_voice and tts_available) else ("auto" if (VOICE_AUTO and tts_available) else "off")
    user_tokens = (event.pattern_match.group(3) or "").split()
    usernames = [t.lstrip("@") for t in user_tokens if not t.startswith("!")]
    exclude_users = [t.lstrip("!").lstrip("@") for t in user_tokens if t.startswith("!")]
    question = event.pattern_match.group(4).strip()
    # Гостям: запрос > лимита → медиа НЕ режем, но vision-модель бесплатная (аудио — Parakeet как всегда)
    vision_model = None
    if not is_owner:
        guest_record = ALLOWED_USERS.get(event.sender_id) or {}
        guest_limit = guest_record.get("limit")  # None → дефолт; -1 → unlimited; иначе число
        if guest_limit is None:
            effective_limit = ALLOWED_ASK_TEXT_LIMIT
        elif guest_limit == -1:
            effective_limit = float("inf")
        else:
            effective_limit = guest_limit
        if n > effective_limit:
            vision_model = FREE_MEDIA_MODEL
            log("ASK", f"Гость {event.sender_id}: n={n} > лимит {effective_limit} → vision={FREE_MEDIA_MODEL}")

    # Параметры для стартового лога (#7) и таймингов (#2) — считаем заранее.
    detail = "low" if n > MEDIA_HIDETAIL_MAX_N else "high"
    if is_owner:
        caller = _owner_label()
    else:
        caller = _user_label(event.sender or await event.get_sender())
    _, _, model_label = get_active_model()
    flags_str = " ".join(f for f, on in [("-t", text_only), ("-c", must_search), ("-d", debug), ("-v", want_voice), ("-g", direct_vision), ("-m", photo_mode == "vision")] if on) or "—"
    users_str = ", ".join("@" + u for u in usernames) if usernames else "—"
    excludes_str = ", ".join("!@" + u for u in exclude_users) if exclude_users else "—"
    vision_label = "free" if vision_model == FREE_MEDIA_MODEL else (vision_model or get_active_media_model())
    log("ASK", f"Старт от {caller}: N={n} · флаги=[{flags_str}] · users=[{users_str}] · excludes=[{excludes_str}] · модель={model_label} · vision={vision_label} · detail={detail}")

    # /ask -g: проверяем, что активная отвечающая модель умеет vision напрямую
    if direct_vision:
        sv = active_model_supports_vision()
        if sv is None:  # кастомная OpenRouter без сохранённого флага — проверяем вживую
            _, _mid, _ = get_active_model()
            try:
                _ex, sv, _ctx, _nm = await _openrouter_model_info(_mid)
            except Exception:
                sv = False
        if not sv:
            await event.respond(
                f"⚠️ Модель «{model_label}» не умеет смотреть картинки напрямую (флаг `-g`).\n"
                f"Переключись на vision-модель через `/model` (например Qwen / Kimi / MiMo Omni, или vision-модель OpenRouter), либо убери `-g`.\n"
                f"ℹ️ GLM-5/5.1 у этого провайдера — текстовые (картинки не принимают), поэтому для `-g` не подходят.")
            if is_owner:
                await event.delete()
            return

    if is_owner:
        await event.delete()  # своё сообщение чистим; гостевой вопрос оставляем видимым

    status = await client.send_message(event.chat_id, "⏳ Собираю сообщения…")

    # Тайминги для финального лога (заполняются по ходу; если фаза не достигнута — остаётся t0)
    t0 = time.time()
    t_collected = t_ctx = t_llm = t_sent = t0

    async def set_status(text):
        try:
            await status.edit(text)
        except (MessageNotModifiedError, FloodWaitError):
            pass
        except Exception:
            pass

    # троттлинг прогресс-бара обработки медиа
    _last_edit = [0.0]

    async def progress_cb(d, t, failed=0):
        now = time.time()
        if now - _last_edit[0] < 1.5 and d < t:
            return
        _last_edit[0] = now
        filled = int(10 * d / t) if t else 10
        bar = "▓" * filled + "░" * (10 - filled)
        warn = f" (⚠️ {failed} не распозн.)" if failed else ""
        await set_status(f"🖼 Обрабатываю медиа {bar} {d}/{t}{warn}")

    # Резолв exclude-юзернеймов в id для надёжной фильтрации (любые msg.sender_id сверим с set'ом)
    exclude_ids = set()
    exclude_failed = []
    for u in exclude_users:
        try:
            uid = OWNER_ID if u.lower() in ("me", "self") else (await client.get_entity(u)).id  # !@me = я сам
            exclude_ids.add(uid)
        except Exception as e:
            exclude_failed.append(u)
            log("ASK", f"Exclude: не нашёл @{u}: {e}")
    if exclude_failed:
        log("ASK", f"Exclude: не удалось зарезолвить: {exclude_failed}")

    def _is_excluded(m):
        sid = getattr(m, "sender_id", None)
        return sid is not None and sid in exclude_ids

    try:
        anchor_id = None
        if usernames:
            by_id = {}
            not_found = []
            for u in usernames:
                try:
                    # Параллельный сбор и под фильтром from_user (позиционные окна работают в messages.search). @me = я сам.
                    _fu = OWNER_ID if u.lower() in ("me", "self") else u
                    _w, raw = await _collect_history_parallel(event.chat_id, n, 0, from_user=_fu)
                    for m in raw:
                        by_id[m.id] = m
                except Exception as e:
                    not_found.append(u)
                    log("ASK", f"Фильтр: не удалось получить сообщения @{u}: {e}")
            messages = sorted(by_id.values(), key=lambda m: m.id, reverse=True)[:n]
            if exclude_ids:
                before = len(messages)
                messages = [m for m in messages if not _is_excluded(m)]
                if before != len(messages):
                    log("ASK", f"Exclude: отфильтровано {before - len(messages)} сообщений")
            if not messages:
                await set_status(f"Не нашёл сообщений от: {', '.join('@' + u for u in usernames)}")
                return
            log("ASK", f"Фильтр по {usernames}: собрано {len(messages)} сообщений" + (f", не найдены: {not_found}" if not_found else ""))
        else:
            # E: если команда — ответ на сообщение, делаем его якорем (он + предыдущие для контекста)
            anchor = await event.get_reply_message() if getattr(event, "reply_to", None) else None
            messages = []
            if anchor is not None and not _is_excluded(anchor):
                anchor_id = anchor.id
                messages.append(anchor)
                offset = anchor.id
            elif anchor is not None:
                # якорь сам в exclude — игнорим его, но используем его id как offset
                offset = anchor.id
            else:
                offset = 0
            # Диагностика: считаем СКОЛЬКО Telegram реально отдал и куда делись скипы.
            diag = {"raw": 0, "service": 0, "self_cmd": 0, "excluded": 0}
            seen = {anchor.id} if anchor_id else set()  # якорь уже в messages — не дублируем

            def _keep(m):
                """Учитывает m в diag и messages; True если оставлено."""
                mid = getattr(m, "id", None)
                if mid is None or mid in seen:
                    return False
                seen.add(mid)
                diag["raw"] += 1
                if mid == event.id:
                    diag["self_cmd"] += 1; return False
                if getattr(m, "action", None) is not None:
                    diag["service"] += 1; return False
                if _is_excluded(m):
                    diag["excluded"] += 1; return False
                messages.append(m)
                return True

            # Параллельный сбор позиционными окнами (быстрее последовательной пагинации Telegram).
            workers, raw_msgs = await _collect_history_parallel(event.chat_id, n, offset)
            await set_status(f"📥 Тяну историю в {workers} {'поток' if workers == 1 else 'потока' if workers < 5 else 'потоков'}…")
            for m in raw_msgs:
                _keep(m)
            messages.sort(key=lambda m: m.id, reverse=True)  # после стыковки окон порядок мог нарушиться
            # Страховка-добор: если из-за FloodWait/скипов собрали < n — добираем последовательно от старого края.
            if len(messages) < n:
                tail_offset = min((m.id for m in messages), default=offset)
                async for m in client.iter_messages(event.chat_id, offset_id=tail_offset, limit=(n - len(messages)) * 2 + 50):
                    if _keep(m) and len(messages) >= n:
                        break
                messages.sort(key=lambda m: m.id, reverse=True)
            # Альбом в Telegram = N отдельных сообщений с общим grouped_id; обрезка ровно по N
            # рассекала бы его (напр. /ask 1 на альбоме из 9 фото видел бы только последнее).
            # messages отсортирован по убыванию id → альбом-сиблинги идут подряд; дотягиваем хвост
            # альбома за границей N (ограничено размером альбома ≤10, сиблинги уже в выборке).
            kept = messages[:n]
            if kept and len(messages) > n:
                boundary_gid = getattr(kept[-1], "grouped_id", None)
                if boundary_gid is not None:
                    for m in messages[n:]:
                        if getattr(m, "grouped_id", None) == boundary_gid:
                            kept.append(m)
                        else:
                            break
                    if len(kept) > n:
                        log("ASK", f"Альбом на границе N={n}: дотянул {len(kept) - n} фото (grouped_id={boundary_gid})")
            messages = kept
            log("ASK", f"iter_messages diag: raw={diag['raw']} · skip service={diag['service']} · команда={diag['self_cmd']} · excludes={diag['excluded']} → попало {len(messages) - (1 if anchor_id else 0)} (+якорь {1 if anchor_id else 0})")
            # Мягкий кэш-якорь (только обычный /ask N, без reply-якоря): держим стабильное НАЧАЛО
            # окна между запросами, дотягивая назад ≤CTX_ANCHOR_SNAP до якоря модели; иначе ре-якорь.
            if anchor is None and messages:
                lastN_oldest = min(m.id for m in messages)
                akey = (event.chat_id, ACTIVE_MODEL)
                _now = time.time()
                st = _ctx_anchors.get(akey)
                valid = st is not None and (_now - st["ts"] < CTX_ANCHOR_TTL)
                if valid and st["anchor_id"] < lastN_oldest:
                    # якорь старше начала окна — пробуем дотянуться к нему за ≤SNAP сообщений
                    bridge = []
                    try:
                        async for bm in client.iter_messages(event.chat_id, offset_id=lastN_oldest, limit=CTX_ANCHOR_SNAP):
                            bridge.append(bm)
                    except Exception as e:
                        log("ASK", f"Кэш-якорь: мост не дотянулся ({e})")
                    reached = bool(bridge) and min(b.id for b in bridge) <= st["anchor_id"]
                    if reached:
                        added = sum(1 for b in bridge if b.id >= st["anchor_id"] and _keep(b))
                        messages.sort(key=lambda m: m.id, reverse=True)
                        st["ts"] = _now
                        log("ASK", f"Кэш-якорь {ACTIVE_MODEL}: +{added} к N={n} (окно от #{st['anchor_id']})")
                    else:
                        _ctx_anchors[akey] = {"anchor_id": lastN_oldest, "ts": _now}  # якорь слишком далеко → новый
                        log("ASK", f"Кэш-якорь {ACTIVE_MODEL}: ре-якорь на #{lastN_oldest} (старый дальше {CTX_ANCHOR_SNAP})")
                else:
                    _ctx_anchors[akey] = {"anchor_id": lastN_oldest, "ts": _now}  # нет/протух/N дальше якоря
                    if valid:
                        log("ASK", f"Кэш-якорь {ACTIVE_MODEL}: ре-якорь на #{lastN_oldest} (N дотянулся за якорь)")
            if anchor is not None:
                aut = _owner_label() if anchor.out else _user_label(anchor.sender)
                qprev = _preview(anchor.raw_text or (_media_tag(anchor) or ""), 60)
                log("ASK", f"Reply-якорь: id={anchor.id}, автор {aut}, «{qprev}»" + (" (исключён из контекста)" if anchor_id is None else ""))
            # Reply на АЛЬБОМ: reply_to указывает на одно сообщение альбома (обычно первое, с подписью),
            # а остальные фото имеют СОСЕДНИЕ id — часто НОВЕЕ якоря, за пределами окна сбора назад.
            # Дотягиваем весь альбом якоря явным запросом по диапазону id (альбом ≤10 → ±9 покрывает).
            if anchor is not None and getattr(anchor, "grouped_id", None) is not None:
                a_gid = anchor.grouped_id
                want = [i for i in range(anchor.id - 9, anchor.id + 10) if i > 0 and i != event.id]
                try:
                    sib = await client.get_messages(event.chat_id, ids=want)
                except Exception as e:
                    sib = []
                    log("ASK", f"Reply-альбом: не удалось дотянуть сиблингов: {e}")
                have = {getattr(m, "id", None) for m in messages}
                added = 0
                for sm in (sib or []):
                    if sm is None or getattr(sm, "id", None) in have:
                        continue
                    if getattr(sm, "grouped_id", None) == a_gid and not _is_excluded(sm):
                        messages.append(sm)
                        have.add(sm.id)
                        added += 1
                if added:
                    messages.sort(key=lambda m: m.id, reverse=True)
                    log("ASK", f"Reply на альбом: дотянул {added} фото альбома якоря (grouped_id={a_gid})")

        ordered = list(reversed(messages))
        t_collected = time.time()
        short = " (чат короче запроса)" if len(ordered) < n else ""
        log("ASK", f"Сбор: запрошено N={n}, фактически {len(ordered)} сообщ.{short}")

        # Карта id→Message для инструмента reply_to_messages (модель шлёт реплаи тредами по #id).
        msg_by_id = {m.id: m for m in ordered if getattr(m, "id", None) is not None}

        # /ask -g: отбираем самые свежие фото (до лимита) для прямой отдачи модели.
        # inline_ids — dict {msg_id: idx}, где idx = детерминированная позиция в хронологии (0..K-1).
        inline_ids = None
        if direct_vision:
            photo_ids = [m.id for m in ordered if getattr(m, "photo", None) and getattr(m, "id", None) is not None]
            recent = photo_ids[-DIRECT_VISION_MAX_IMAGES:]  # ordered хронологичен → хвост = свежие
            inline_ids = {mid: i for i, mid in enumerate(recent)}
            log("ASK", f"-g: фото в выборке {len(photo_ids)}, инлайню {len(inline_ids)} свежих (лимит {DIRECT_VISION_MAX_IMAGES})")

        # Ретрай-цикл на ContextOverflowError: если модель реально насчитала больше токенов,
        # чем tiktoken — пересобираем с агрессивнее обрезкой (safety ×2, ×4).
        base_safety = active_ctx_safety()
        safety_attempts = [None, base_safety * 2.0, base_safety * 4.0]
        reply = None
        context = ""
        dropped = failed = ctx_tokens = 0
        for retry_idx, safety_override in enumerate(safety_attempts):
            retry_suffix = f" (ретрай ×{safety_override / base_safety:.1f})" if safety_override else ""
            await set_status(f"📥 Собрано {len(ordered)} сообщ. — собираю контекст…{retry_suffix}")
            inline_images = [] if direct_vision else None  # сбрасываем на каждой ретрай-итерации (без дублей)
            context, dropped, failed, ctx_tokens = await assemble_context(
                ordered, text_only, anchor_id=anchor_id, progress_cb=progress_cb,
                vision_model=vision_model, detail=detail, safety_override=safety_override,
                inline_ids=inline_ids, inline_images=inline_images, photo_mode=photo_mode,
                include_ids=True,  # пометка #id у каждого сообщения → reply_to_messages
            )
            t_ctx = time.time()
            context = context or "(нет сообщений)"
            save_media_cache()
            images_sorted = None
            if direct_vision:
                images_sorted = sorted(inline_images, key=lambda e: e["idx"])  # порядок = #idx в тексте
                log("ASK", f"-g: картинок напрямую модели: {len(images_sorted)}")
            await set_status(f"🤖 Думаю над ответом…{retry_suffix}")
            try:
                reply = await ask_agentic(context, question, must_search=must_search, caller=caller, ctx_tokens_est=ctx_tokens, voice_mode=voice_mode, images=images_sorted, chat_id=event.chat_id, msg_by_id=msg_by_id, memory_allowed=_index_memory_allowed(is_owner), asker_id=event.sender_id)
                t_llm = time.time()
                break  # успех
            except ContextOverflowError as e:
                log("ASK", f"Overflow при safety×{(safety_override or base_safety):.2f}: ctx={ctx_tokens} (tiktoken) → API: {e}")
                if retry_idx == len(safety_attempts) - 1:
                    reply = (f"⚠️ Контекст не влезает в окно модели даже при агрессивной обрезке "
                             f"(safety×{safety_attempts[-1] / base_safety:.1f}). "
                             f"Попробуй меньшее N или смени модель (/model).")
                    t_llm = time.time()
                    log("ASK", "Все ретраи overflow исчерпаны")
                    break
                # иначе — продолжаем цикл с большим safety_override
                continue

        # -d: дамп полного user-message в файл (то, что РЕАЛЬНО видит модель)
        if debug:
            try:
                os.makedirs("asks", exist_ok=True)
                ts = datetime.now(MSK).strftime("%Y%m%d_%H%M%S")
                fname = f"asks/{ts}_{event.id}.txt"
                user_msg = _build_ask_user_content(context, question, caller)
                header = (
                    "=== /ask -d debug dump ===\n"
                    f"timestamp: {ts} МСК\n"
                    f"caller: {caller}\n"
                    f"chat_id: {event.chat_id}\n"
                    f"event_id: {event.id}\n"
                    f"запрошено N: {n}, фактически: {len(ordered)}\n"
                    f"флаги: {flags_str}\n"
                    f"users-фильтр: {users_str}\n"
                    f"excludes: {excludes_str}" + (f" · failed: {exclude_failed}" if exclude_failed else "") + "\n"
                    f"модель ответов: {model_label}\n"
                    f"vision-модель: {vision_label} · detail: {detail}\n"
                    f"ctx_tokens (tiktoken): {ctx_tokens}\n"
                    f"context_chars: {len(context)}\n"
                    f"dropped (обрезано): {dropped} · failed (медиа не распозн.): {failed}\n"
                    "==========================\n"
                    "Ниже — ПОЛНЫЙ user-message, отправленный модели "
                    "(system-prompt — это ASK_SYSTEM_PROMPT + тех. инструкции; см. код).\n\n"
                )
                with open(fname, "w", encoding="utf-8") as fh:
                    fh.write(header + user_msg)
                log("ASK", f"[DEBUG -d] Дамп user-message: {fname} ({len(user_msg)} симв)")
                # Авточистка: держим только последние ASKS_KEEP файлов
                files = sorted(glob.glob("asks/*.txt"))
                excess = len(files) - ASKS_KEEP
                if excess > 0:
                    for f in files[:excess]:
                        try:
                            os.remove(f)
                        except Exception:
                            pass
                    log("ASK", f"[DEBUG -d] Удалено {excess} старых дампов, оставлено {ASKS_KEEP}")
            except Exception as e:
                log("ASK", f"[DEBUG -d] Ошибка записи дампа: {e}")

        _, _, label = get_active_model()
        notes = []
        if dropped:
            notes.append(f"✂️ обрезано {dropped} стар. сообщ.")
        if failed:
            notes.append(f"⚠️ {failed} медиа не распознано")
        if MODEL_REGISTRY.get(ACTIVE_MODEL, ("",))[0] == "openai":
            _qbucket = _openai_bucket(MODEL_REGISTRY[ACTIVE_MODEL][1])
            _qlimit, _qlim_s = ((OPENAI_FREE_DAILY_MINI, "2.5M") if _qbucket == "mini"
                                else (OPENAI_FREE_DAILY_LARGE, "250k"))
            _qi, _qo, qtot = _openai_usage_today(_qbucket)
            if qtot >= _qlimit:
                notes.append(f"🎁 бесплатная квота дня исчерпана (~{_fmt_ctx(qtot)}/{_qlim_s}) — дальше с баланса")
            elif qtot >= int(_qlimit * 0.8):
                notes.append(f"🎁 квота дня: ~{_fmt_ctx(qtot)}/{_qlim_s}")

        # Решаем, идёт ли ответ голосом: force (флаг -v) или auto (модель начала с маркера [[VOICE]]).
        go_voice, spoken = False, reply
        if voice_mode == "force":
            go_voice, spoken = True, reply
        elif voice_mode == "auto" and reply.lstrip().startswith("[[VOICE]]"):
            go_voice = True
            spoken = reply.lstrip()[len("[[VOICE]]"):].lstrip()

        if go_voice:
            await set_status("🎙 Озвучиваю ответ…")
            ogg = await synthesize_voice(spoken, ACTIVE_VOICE)
            if ogg:
                bio = io.BytesIO(ogg)
                bio.name = "voice.ogg"
                await client.send_file(event.chat_id, bio, voice_note=True, reply_to=reply_target_id)
                t_sent = time.time()
                try:
                    await status.delete()
                except Exception:
                    pass
                log("ASK", f"Голосовой ответ на '{question[:60]}' отправлен (voice={ACTIVE_VOICE}, mode={voice_mode})")
                return
            notes.append("🔇 голос не сгенерировался")  # фолбэк на текст

        note = (" — " + "; ".join(notes)) if notes else ""
        prefix = f"{label}{_reasoning_tag()}{note}:\n\n"
        # На текстовом пути срезаем возможный ведущий маркер [[VOICE]] (если авто-режим выбрал голос, но он упал).
        if reply.lstrip().startswith("[[VOICE]]"):
            reply = reply.lstrip()[len("[[VOICE]]"):].lstrip()
        # Чистим markdown-мусор (#/*) ДО нарезки на части — модель путает HTML и markdown.
        reply = _html_clean_markdown(reply)
        # Сначала отправляем ответ, потом удаляем статус — иначе сбой delete съест ответ.
        await send_long(event.chat_id, reply, prefix=prefix, parse_mode="html", reply_to=reply_target_id, collapse_threshold=700)
        t_sent = time.time()
        try:
            await status.delete()
        except Exception:
            pass
        log("ASK", f"Ответ на '{question[:60]}' отправлен (model={ACTIVE_MODEL}, text_only={text_only}, must_search={must_search}, users={usernames or '—'}, anchor={anchor_id}, dropped={dropped}, failed={failed})")
    except Exception as e:
        log("ASK", f"Ошибка команды /ask: {e}")
        traceback.print_exc()
        await set_status(f"⚠️ Ошибка при обработке /ask: {e}")
    finally:
        t_end = time.time()
        log("ASK", f"Тайминги: сбор={t_collected-t0:.1f}с · контекст(медиа)={t_ctx-t_collected:.1f}с · LLM={t_llm-t_ctx:.1f}с · отправка={t_sent-t_llm:.1f}с · итого={t_end-t0:.1f}с")


# t.me-ссылка на сообщение: c/<internal>/<msg> · c/<internal>/<topic>/<msg> · <username>/<msg>.
# msg_id — последний числовой сегмент; chat-часть = группа 1 (c/<id> или username).
_TME_LINK_RE = re.compile(r'(?:https?://)?t\.me/(c/\d+|[A-Za-z]\w{2,})((?:/\d+)+)', re.I)


async def _gen_fetch_link_refs(event, prompt):
    """Находит t.me-ссылки на сообщения в промпте /gen, тянет их фото (ТОЛЬКО указанное, без альбома),
    вырезает ссылки из текста. Возвращает (cleaned_prompt, [Message…], not_found)."""
    matches = list(_TME_LINK_RE.finditer(prompt))
    if not matches:
        return prompt, [], 0
    msgs, not_found = [], 0
    for mt in matches:
        chat_ref, tail = mt.group(1), mt.group(2)
        try:
            msg_id = int(tail.strip("/").split("/")[-1])
            peer = int("-100" + chat_ref[2:]) if chat_ref.lower().startswith("c/") else chat_ref
            fetched = await client.get_messages(peer, ids=[msg_id])
            fm = next((x for x in (fetched or []) if x is not None), None)
        except Exception as e:
            log("GEN", f"Ссылка-референс {chat_ref}{tail}: не достал ({e})")
            fm = None
        if fm is not None and getattr(fm, "photo", None):
            msgs.append(fm)
        else:
            not_found += 1
    cleaned = re.sub(r"\s{2,}", " ", _TME_LINK_RE.sub("", prompt)).strip()
    log("GEN", f"Ссылки-референсы: найдено {len(matches)}, с фото {len(msgs)}, без фото/недоступно {not_found}")
    return cleaned, msgs, not_found


def _is_attached_photo(msg):
    """True только для РЕАЛЬНО прикреплённого фото. msg.photo истинно и для фото из ВЕБ-ПРЕВЬЮ
    ссылки (Telegram авто-превью t.me-ссылки в тексте) — такое исключаем, иначе ссылка-референс
    задваивается: фото из веб-превью команды + фото из самого сообщения по ссылке."""
    if not getattr(msg, "photo", None):
        return False
    return not isinstance(getattr(msg, "media", None), MessageMediaWebPage)


def _is_attached_image_doc(msg):
    """Картинка, отправленная ФАЙЛОМ/документом (webp/png/jpeg — в т.ч. стикеры и старые генерации бота,
    уходившие как .webp): у таких msg.photo пуст, поэтому отдельная проверка по mime документа."""
    doc = getattr(msg, "document", None)
    mime = (getattr(doc, "mime_type", None) or "").lower() if doc else ""
    return mime in ("image/webp", "image/png", "image/jpeg") and (getattr(doc, "size", 0) or 0) <= 15_000_000


async def _gen_collect_input_images(event, reply_msg, extra_msgs=None):
    """Референс-фото для /gen: из самого сообщения с командой (включая его альбом), из реплая
    (включая альбом реплая) и из extra_msgs (ссылки — ТОЛЬКО указанное фото, без альбома).
    Возвращает (list_b64, skipped): максимум 10 фото, суммарно ≤ GEN_IMAGE_MAX_INPUT сырых байт
    (лимит запроса API 4.5 МБ); лишние пропускаются."""
    sources, seen = [], set()

    async def _add_with_album(msg):
        if msg is None:
            return
        batch = []
        if (_is_attached_photo(msg) or _is_attached_image_doc(msg)) and msg.id not in seen:
            seen.add(msg.id)
            batch.append(msg)
        gid = getattr(msg, "grouped_id", None)
        if gid:  # альбом: соседние сообщения с тем же grouped_id (id всегда рядом)
            try:
                async for m in client.iter_messages(event.chat_id, min_id=msg.id - 12, max_id=msg.id + 12):
                    if getattr(m, "grouped_id", None) == gid and getattr(m, "photo", None) and m.id not in seen:
                        seen.add(m.id)
                        batch.append(m)
            except Exception as e:
                log("GEN", f"Альбом не дочитал: {e}")
        batch.sort(key=lambda m: m.id)
        sources.extend(batch)

    await _add_with_album(event.message)  # сначала мои приложенные фото, потом фото реплая
    await _add_with_album(reply_msg)
    for m in (extra_msgs or []):  # ссылки-референсы: ровно указанное фото, без альбома
        if m is not None and (getattr(m, "photo", None) or _is_attached_image_doc(m)) and m.id not in seen:
            seen.add(m.id)
            sources.append(m)
    out, total, skipped = [], 0, 0
    for m in sources:
        if len(out) >= 10:
            skipped += 1
            continue
        try:
            img = await m.download_media(bytes)
        except Exception as e:
            log("GEN", f"Фото id={m.id} не скачалось: {e}")
            img = None
        if not img:
            continue
        if img[:4] == b"RIFF" and img[8:12] == b"WEBP":  # webp-документ/стикер → png (генератор webp на входе не ждёт)
            img = await _webp_to_png(img)
        if total + len(img) > GEN_IMAGE_MAX_INPUT:
            skipped += 1
            continue
        total += len(img)
        out.append(base64.b64encode(img).decode("utf-8"))
    if sources:
        log("GEN", f"Референсы: найдено {len(sources)} фото → взято {len(out)} ({total / 1024:.0f} КБ), пропущено {skipped}")
    return out, skipped


async def _gen_history_catalog(ordered, want_vision: bool, limit: int = GEN_CTX_IMG_MAX, timeout: float = GEN_CATALOG_TIMEOUT, progress_cb=None) -> list:
    """Каталог фото из истории чата для /gen: новейшие ≤limit фото из ordered (дедуп по file-id).
    Для каждого: оригинал `bytes` (уйдёт В ГЕНЕРАТОР как референс) + уменьшенная копия `thumb`
    (уйдёт ПРОМПТЕРУ — vision напрямую / для описания; без неё 20 полных фото бьют лимит запроса).
    Для ТЕКСТОВОЙ активной модели считаем описание (медиа-модель, кэш общий с /ask). Этапы шлёт progress_cb,
    всё под общим тайм-бюджетом GEN_CATALOG_TIMEOUT. Возвращает [{idx, mid, bytes, thumb, caption, desc}]."""
    photos, seen, n_preview = [], set(), 0
    for m in ordered:  # ordered — хронологический
        if not getattr(m, "photo", None):
            continue
        # m.photo у Telethon включает и фото веб-превью (обложки YouTube и пр.) и сервисные
        # (смена аватарки чата) — это не референсы, отсекаем сразу
        if isinstance(getattr(m, "media", None), MessageMediaWebPage) or getattr(m, "action", None):
            n_preview += 1
            continue
        k = _media_key(m)
        if k in seen:
            continue
        seen.add(k)
        photos.append(m)
    if n_preview:
        log("GEN", f"Каталог: отсеяно {n_preview} превью ссылок/сервисных фото")
    photos = photos[-limit:]  # хвост = самые свежие
    if not photos:
        return []

    if progress_cb:
        await progress_cb(f"📥 Скачиваю фото из чата ({len(photos)})…")

    async def _dl(m):
        try:
            raw = await m.download_media(bytes)
        except Exception as e:
            log("GEN", f"Каталог: фото id={getattr(m, 'id', '?')} не скачалось: {e}")
            return None
        if not raw:
            return None
        return (raw, await _downscale_img(raw))  # (оригинал, уменьшенная копия)
    try:
        results = await asyncio.wait_for(asyncio.gather(*[_dl(m) for m in photos]), timeout=timeout)
    except asyncio.TimeoutError:
        log("GEN", "Каталог: скачивание/сжатие фото превысило тайм-бюджет — пропускаю историю-картинки")
        return []

    catalog = []
    for m, res in zip(photos, results):
        if not res:
            continue
        raw, thumb = res
        catalog.append({"idx": 0, "mid": getattr(m, "id", None), "bytes": raw, "thumb": thumb,
                        "caption": (m.raw_text or "").strip(), "desc": None,
                        "from_owner": bool(getattr(m, "out", False)),  # моё фото / прошлая генерация — для пометки «не повторяй»
                        "_m": m})

    if not want_vision and catalog:  # текстовой модели нужны описания: ген-формат ТИП/СУТЬ/ЛЮДИ (свой кэш-неймспейс, /ask-кэш не трогаем)
        sem = asyncio.Semaphore(4)
        done = [0]

        async def _desc(it):
            key = "gen:" + _media_key(it["_m"])
            cached = MEDIA_CACHE.get(key)
            if cached and cached not in MEDIA_FAILURE_MARKERS:
                it["desc"] = cached
            else:
                async with sem:
                    try:
                        d = await describe_image(it["thumb"], caption=it["caption"], prompt=_GEN_DESC_PROMPT)
                    except Exception as e:
                        log("GEN", f"Каталог: описание не удалось: {e}")
                        d = None
                if d and d not in MEDIA_FAILURE_MARKERS and not _looks_like_refusal(d):
                    it["desc"] = d
                    _media_cache_set(key, d)
            done[0] += 1
            if progress_cb and done[0] % 3 == 0:
                await progress_cb(f"🖋 Описываю фото для модели ({done[0]}/{len(catalog)})…")
        try:
            await asyncio.wait_for(asyncio.gather(*[_desc(it) for it in catalog]), timeout=timeout)
        except asyncio.TimeoutError:
            log("GEN", "Каталог: описания превысили тайм-бюджет — часть фото пойдёт без описания")

        # пре-фильтр по типу: скриншоты (переписки/интерфейсы/превью) и мемы промптеру не показываем вовсе —
        # именно этот мусор текстовые модели тащили в референсы. Нужен скриншот — юзер приложит его реплаем.
        # Фото без описания (тайм-аут) остаются: недоступность — не повод выкидывать.
        junk = [it for it in catalog if _gen_desc_kind(it.get("desc")) in ("скриншот", "мем")]
        if junk:
            catalog = [it for it in catalog if it not in junk]
            log("GEN", f"Каталог: исключено {len(junk)} (скриншоты/мемы) из кандидатов")

    for i, it in enumerate(catalog, 1):  # финальная нумерация 1..K, убираем временное поле
        it["idx"] = i
        it.pop("_m", None)
    log("GEN", f"Каталог истории: {len(catalog)} фото (vision={want_vision}, с описанием={sum(1 for it in catalog if it.get('desc'))})")
    return catalog


def _gen_desc_kind(desc: str) -> str:
    """Тип картинки из ген-описания («ТИП: скриншот | мем | фото людей | …») — первое слово, lower."""
    m = re.search(r"(?im)^\s*ТИП:\s*(\S+)", desc or "")
    return m.group(1).strip().lower().rstrip(".,;") if m else ""


def _merge_catalog_refs(input_b64s: list, catalog: list, sel: list) -> tuple:
    """Добавляет выбранные ИИ картинки каталога (по idx из sel) к input_b64s генератора, соблюдая
    GEN_CTX_REF_MAX и общий лимит GEN_IMAGE_MAX_INPUT. Возвращает (новый список base64, реально взятые idx)."""
    if not (catalog and sel):
        return input_b64s, []
    out = list(input_b64s or [])
    total = sum(len(b) * 3 // 4 for b in out)  # оценка уже накопленных сырых байт (base64 ×4/3)
    by_idx = {it["idx"]: it for it in catalog}
    added, used = 0, []
    for k in sel:
        if added >= GEN_CTX_REF_MAX:
            break
        it = by_idx.get(k)
        if not it:
            continue
        blob = it.get("thumb") or it["bytes"]  # уменьшенная копия — чтобы до 16 референсов влезли в лимит запроса (для узнаваемости 768px хватает)
        if total + len(blob) > GEN_IMAGE_MAX_INPUT:
            continue
        total += len(blob)
        out.append(base64.b64encode(blob).decode("utf-8"))
        added += 1
        used.append(k)
    if added:
        log("GEN", f"Из истории добавлено {added} картинок-референсов (выбор ИИ: {sel})")
    return out, used


def _gen_refs_line(chat_ent, catalog: list, used_idxs: list, reasons: dict = None) -> str:
    """Строка «Референсы:» со ссылками на сообщения, чьи фото реально ушли в генерацию (markdown).
    reasons — {idx: «зачем взят»} от промптера (из REFS: 3 (лицо Димы), …), добавляется к ссылке."""
    if not (chat_ent and catalog and used_idxs):
        return None
    by_idx = {it["idx"]: it for it in catalog}
    parts = []
    for k in used_idxs:
        it = by_idx.get(k)
        mid = it.get("mid") if it else None
        if not mid:
            continue
        try:
            link = f"[#{k}]({build_msg_link(chat_ent, mid)})"
        except Exception:
            continue
        why = (reasons or {}).get(k)
        parts.append(f"{link} — {why}" if why else link)
    return ("📎 Референсы (фото из чата): " + " · ".join(parts)) if parts else None


# Глобальный rate-gate для image-API: разносим вызовы во времени, чтобы не ловить RPM-429
# и не жечь деньги/квоту на провальных ретраях (failed-попытки тоже списываются).
_GEN_RATE_LOCK = asyncio.Lock()
_GEN_RATE_MIN_INTERVAL = 13.0  # сек между запросами к генератору (~4–5/мин)
_GEN_LAST_CALL = [0.0]


async def _gen_rate_gate():
    async with _GEN_RATE_LOCK:
        gap = _GEN_RATE_MIN_INTERVAL - (time.monotonic() - _GEN_LAST_CALL[0])
        if gap > 0:
            await asyncio.sleep(gap)
        _GEN_LAST_CALL[0] = time.monotonic()


async def _gen_one_image(final_prompt, input_b64s, image_size, aspect_ratio, allow_repair, user_prompt, status_cb=None):
    """Один цикл генерации с ретраями (transient тем же промптом / фолбэк на Fast / repair при модерации).
    Возвращает (raw, mime, used_prompt, used_fallback) при успехе или (None, reason, None, used_fallback)
    при отказе (reason: 'moderation' | 'overload' | 'exhausted'). status_cb — необязательный апдейтер статуса.
    Все вызовы проходят через _gen_rate_gate (≤5/мин)."""
    async def _s(text):
        if status_cb:
            await status_cb(text)
    gen_model = OPENROUTER_IMAGE_MODEL
    used_fallback = False
    transient_left = 2  # ретраи дорогие: провальная попытка тоже списывается из дневной квоты
    repair_left = 2 if allow_repair else 0
    attempt = 0
    size = image_size
    fp = final_prompt
    while True:
        try:
            await _gen_rate_gate()
            raw, mime = await asyncio.to_thread(_sync_generate_image, fp, input_b64s or None, gen_model, size, aspect_ratio)
            return raw, mime, fp, used_fallback
        except GenExhausted as e:
            # ДНЕВНОЙ лимит модели исчерпан — ретраить сегодня бессмысленно (и жжёт квоту). Пробуем запасную (своя квота).
            log("GEN", f"Дневной лимит исчерпан ({gen_model}): {e}")
            if not used_fallback and OPENROUTER_IMAGE_FALLBACK and OPENROUTER_IMAGE_FALLBACK != gen_model:
                used_fallback = True
                gen_model = OPENROUTER_IMAGE_FALLBACK
                log("GEN", f"Пробую запасную {gen_model} (у неё своя квота)…")
                await _s("🔁 Дневной лимит основной модели — пробую запасную (Gemini)…")
                continue
            return None, "exhausted", None, used_fallback
        except GenRejected as e:
            log("GEN", f"Отклонено модерацией: {e} (repair_left={repair_left})")
            if repair_left > 0:
                repair_left -= 1
                await _s("🔁 Промпт отклонён модерацией — DeepSeek правит его и пробуем снова…")
                new_prompt = await asyncio.to_thread(_sync_repair_image_prompt, fp, user_prompt)
                if new_prompt != fp:
                    fp = new_prompt
                    continue
                log("GEN", "Repair не изменил промпт (DeepSeek недоступен/сам фильтрует) — отказ")
            return None, "moderation", None, used_fallback
        except (GenTransient, requests.exceptions.RequestException) as e:
            # 5xx от ОСНОВНОЙ модели = провайдер/адаптер лежит, ждать бессмысленно → сразу на запасную (без 2× пауз).
            _http_code = getattr(getattr(e, "response", None), "status_code", 0) or 0
            if _http_code >= 500 and not used_fallback and OPENROUTER_IMAGE_FALLBACK and OPENROUTER_IMAGE_FALLBACK != gen_model:
                used_fallback = True
                gen_model = OPENROUTER_IMAGE_FALLBACK
                transient_left = 2
                attempt = 0
                log("GEN", f"Основная модель отдаёт {_http_code} — сразу переключаюсь на запасную {gen_model}")
                await _s("🔁 Основная модель недоступна — пробую запасную (Gemini)…")
                continue
            if transient_left > 0:
                transient_left -= 1
                attempt += 1
                wait = min(30, 15 + 8 * attempt)  # RPM-лимит: ждём дольше (23, 30с)
                log("GEN", f"Временный сбой провайдера ({gen_model}): {e} — ретрай через {wait}с (осталось {transient_left})")
                await _s("⏳ Провайдер генерации перегружен — повторяю…")
                await asyncio.sleep(wait)
                continue
            if not used_fallback and OPENROUTER_IMAGE_FALLBACK and OPENROUTER_IMAGE_FALLBACK != gen_model:
                used_fallback = True
                gen_model = OPENROUTER_IMAGE_FALLBACK
                transient_left = 2
                attempt = 0
                log("GEN", f"Основная модель не отвечает — переключаюсь на запасную {gen_model}")
                await _s("🔁 Основная модель перегружена — пробую запасную (Gemini)…")
                continue
            return None, "overload", None, used_fallback


async def _gen_send_image(chat, raw, mime, final_prompt, prompt_by_ai, reply_to, refs_line=None, idea=None):
    """Отправляет готовую картинку: webp→png, и при AI-промпте — свёрнутая подпись (или отдельным
    сообщением, если длинная). refs_line — строка «Референсы:» со ссылками (отдельным сообщением-ответом
    под картинкой). idea — фраза-идея от промптера: видимой строкой 💡 над свёрнутым промптом.
    chat='me' = Saved Messages."""
    if "webp" in mime:
        raw = await _webp_to_png(raw)  # webp Telegram шлёт стикером — конвертим
    bio = io.BytesIO(raw)
    # имя строго по magic-байтам: JPEG от запасной Gemini раньше падал в ветку «gen.webp» и уходил юзеру как webp
    if raw[:8].startswith(b"\x89PNG"):
        bio.name = "gen.png"
    elif raw[:3] == b"\xff\xd8\xff":
        bio.name = "gen.jpg"
    else:
        bio.name = "gen.webp"  # не сконвертившийся webp — хотя бы честное расширение
    sent = None
    idea_line = f"💡 {idea.strip()}\n" if (idea and str(idea).strip()) else ""
    if prompt_by_ai:  # промпт от ИИ — СВЁРНУТОЙ цитатой и БЕЗ обрезки; идея — видимой строкой над ней
        cap_text = idea_line + "🎨 " + final_prompt
        if len(cap_text) <= 1000:  # влезает в лимит подписи Telegram (1024)
            try:
                cap, cap_ents = _collapsed_entities("🎨 " + final_prompt, parse_html=False)
                if idea_line:  # идею НЕ сворачиваем: префиксуем и сдвигаем entities цитаты (offsets в UTF-16)
                    shift = len(add_surrogate(idea_line))
                    for e in cap_ents:
                        e.offset += shift
                    cap = idea_line + cap
                sent = await client.send_file(chat, bio, caption=cap, formatting_entities=cap_ents, reply_to=reply_to)
            except Exception as e:
                log("GEN", f"Свёрнутая подпись не отправилась ({e}) — шлю обычной")
                bio.seek(0)
                sent = await client.send_file(chat, bio, caption=cap_text, reply_to=reply_to)
        else:  # длинный промпт: картинка с идеей в подписи + полный промпт отдельной свёрнутой цитатой
            sent = await client.send_file(chat, bio, caption=(idea_line.strip() or None), reply_to=reply_to)
            await send_long(chat, "🎨 " + final_prompt, parse_mode=None, reply_to=getattr(sent, "id", None), collapse_threshold=0)
    else:
        sent = await client.send_file(chat, bio, reply_to=reply_to)
    if refs_line:  # ссылки на сообщения-источники референсов — отдельным сообщением под картинкой
        try:
            await client.send_message(chat, refs_line, parse_mode="md", reply_to=getattr(sent, "id", None), link_preview=False)
        except Exception as e:
            log("GEN", f"Строка референсов не отправилась: {e}")


@client.on(events.NewMessage(pattern=r"^[./]gen(?:\s+(\d+))?((?:\s+-(?:improve|creative|vertical|horizontal|square|sq|4k|2k|1k|x\d+|noimg|ni|raw|m|r|i|c|v|h))+)?((?:\s+!?@\w+)+)?\s+(.+)$"))
async def gen_command(event):
    """Генерация изображений (GPT Image 2 via OpenRouter). Промпт как есть, либо его строит/улучшает DeepSeek
    из контекста (N последних сообщений / текст reply / флаг -i). Фото в сообщении/reply → image-to-image."""
    if await _slash_for_other_bot(event):
        return  # /команда в личке с ботом адресована ему, не юзерботу (используй .gen)
    is_owner = event.out
    if not is_owner and event.sender_id not in ALLOWED_USERS:
        return
    if openrouter_client is None:
        await event.reply("⚠️ Генерация недоступна: нет `OPENROUTER_API_KEY` в .env.")
        return
    n = int(event.pattern_match.group(1)) if event.pattern_match.group(1) else 0
    toks = (event.pattern_match.group(2) or "").split()
    improve = any(t in ("-i", "-improve") for t in toks)        # уточнить/улучшить промпт (edit при референсе)
    creative = any(t in ("-c", "-creative") for t in toks)      # креатив: ИИ сочиняет промпт-ОТВЕТ (не редактирует)
    noimg = any(t in ("-ni", "-noimg") for t in toks)           # не брать картинки из истории чата в референсы
    force_desc = any(t == "-m" for t in toks)                   # -m: всегда через ОПИСАНИЯ (даже vision-модель), но больший пул кандидатов
    raw = any(t in ("-r", "-raw") for t in toks)                # -r: БЕЗ ИИ — твой промпт дословно в генератор (literal)
    aspect_ratio = None                                         # ориентация → aspect_ratio Image API (точно)
    for t in toks:
        if t in ("-v", "-vertical"): aspect_ratio = "9:16"
        elif t in ("-h", "-horizontal"): aspect_ratio = "16:9"
        elif t in ("-sq", "-square"): aspect_ratio = "1:1"
    image_size = "2K"                                           # дефолт 2K (1024²→2048², вчетверо чётче); -4k/-1k меняют
    for t in toks:
        if t.lower() == "-4k": image_size = "4K"
        elif t.lower() == "-1k": image_size = "1K"
        elif t.lower() == "-2k": image_size = "2K"
    batch_count = 1                                             # -xN: пакет вариантов → в Избранное (Saved Messages)
    for t in toks:
        if t.lower().startswith("-x") and t[2:].isdigit():
            batch_count = max(1, min(GEN_BATCH_MAX, int(t[2:])))
    if not is_owner:
        batch_count = 1  # пакет шлёт в Saved Messages аккаунта-владельца — для гостей бессмыслен
    user_tokens = (event.pattern_match.group(3) or "").split()  # @юзер (только эти) / !@юзер (исключить) — фильтр контекста
    include_users = [t.lstrip("@") for t in user_tokens if not t.startswith("!")]
    exclude_users = [t.lstrip("!").lstrip("@") for t in user_tokens if t.startswith("!")]
    user_prompt = event.pattern_match.group(4).strip()
    reply_msg = await event.get_reply_message() if getattr(event, "reply_to", None) else None
    reply_target_id = getattr(event, "reply_to_msg_id", None)
    caller = _owner_label() if is_owner else _user_label(event.sender or await event.get_sender())
    # Ссылки-референсы: t.me-ссылки на сообщения в промпте → их фото на вход, ссылки из текста убираем.
    user_prompt, link_ref_msgs, link_not_found = await _gen_fetch_link_refs(event, user_prompt)
    if not user_prompt:  # остались одни ссылки без инструкции — даём осмысленный дефолт
        user_prompt = "combine the reference images into one cohesive scene"
    _flt = (("+@" + ",".join(include_users)) if include_users else "") + (("  -@" + ",".join(exclude_users)) if exclude_users else "")
    log("GEN", f"Старт от {caller}: N={n or '—'} · improve={improve} · creative={creative} · {image_size}/{aspect_ratio or 'авто'} · пакет×{batch_count} · фильтр={_flt or '—'} · reply={'да' if reply_msg else 'нет'} · ссылок-реф={len(link_ref_msgs)} · «{user_prompt[:80]}»")

    # — референс-фото (моё сообщение + альбом, reply + альбом, ссылки) собираем ДО удаления команды —
    input_b64s, skipped_imgs = await _gen_collect_input_images(event, reply_msg, extra_msgs=link_ref_msgs)

    # Креативный режим — ПО УМОЛЧАНИЮ (как -c, указывать не нужно): без приложенного фото-референса ИИ
    # сочиняет промпт со своим видением. Приложил фото на правку → остаётся точный edit-режим (если только не -c).
    # Явный -i (improve) дефолт НЕ включает — это осознанный выбор «только переформулируй, без отсебятины».
    # Флаг -r (raw) отключает ИИ полностью → промпт уходит в генератор дословно.
    if not input_b64s and not raw and not improve:
        creative = True

    if is_owner and not (_is_attached_photo(event.message) or _is_attached_image_doc(event.message)):
        await event.delete()  # чистим команду; ОСТАВЛЯЕМ только при реально приложенном фото/картинке-файле
        # (веб-превью t.me-ссылки — не вложение, поэтому команду со ссылками тоже удаляем)
    status = await client.send_message(event.chat_id, "🎨 Готовлю генерацию…")

    async def set_status(text):
        try:
            await status.edit(text)
        except (MessageNotModifiedError, FloodWaitError):
            pass
        except Exception:
            pass

    try:
        if skipped_imgs and not input_b64s:
            await set_status("⚠️ Фото слишком большие: ни одно не влезло в лимит 3 МБ суммарно. Сожми и попробуй снова.")
            return
        if skipped_imgs:
            await set_status(f"ℹ️ Взял {len(input_b64s)} фото, пропустил {skipped_imgs} (лимит 3 МБ суммарно / макс. 10).")
        if link_not_found:
            await set_status(f"⚠️ {link_not_found} ссылк{'а' if link_not_found == 1 else 'и'}-референс без фото или недоступн{'а' if link_not_found == 1 else 'ы'} — пропускаю.")

        # — финальный промпт —
        final_prompt, prompt_by_ai = user_prompt, False
        context_text = None
        if n > 0:
            await set_status(f"📥 Собираю последние {n} сообщений для контекста…")
            # @юзер/!@юзер → резолвим в id для фильтра контекста (как в /ask)
            include_ids, exclude_ids, flt_failed = set(), set(), []
            for u in include_users + exclude_users:
                try:
                    uid = OWNER_ID if u.lower() in ("me", "self") else (await client.get_entity(u)).id  # @me/!@me = я сам
                    (exclude_ids if u in exclude_users else include_ids).add(uid)
                except Exception as e:
                    flt_failed.append(u)
                    log("GEN", f"Фильтр: не нашёл @{u}: {e}")
            if flt_failed:
                await set_status(f"⚠️ Не нашёл для фильтра: {', '.join('@' + u for u in flt_failed)} — игнорирую.")
            _, raw_msgs = await _collect_history_parallel(event.chat_id, n, 0)

            def _ctx_keep(m):
                if getattr(m, "id", None) is None or m.id == event.id or getattr(m, "action", None) is not None:
                    return False
                sid = getattr(m, "sender_id", None)
                if exclude_ids and sid in exclude_ids:
                    return False
                if include_ids and sid not in include_ids:
                    return False
                return True
            msgs = [m for m in raw_msgs if _ctx_keep(m)]
            msgs.sort(key=lambda m: m.id, reverse=True)
            ordered = list(reversed(msgs[:n]))
            if include_ids or exclude_ids:
                log("GEN", f"Контекст после фильтра: {len(ordered)} сообщ. (вкл={len(include_ids)} искл={len(exclude_ids)})")
            context_text, _, _, _ = await assemble_context(ordered, True)  # text-only: медиа не разбираем
        elif reply_msg is not None and (reply_msg.raw_text or "").strip():
            # Reply на сообщение С ФОТО: без флагов DeepSeek не вмешивается (промпт дословный, фото на вход);
            # с -i/-c — берёт текст/подпись реплая в контекст. Reply на чистый текст — как раньше.
            reply_with_photo = bool(getattr(reply_msg, "photo", None) or getattr(reply_msg, "grouped_id", None))
            if not reply_with_photo or improve or creative:
                context_text = (reply_msg.raw_text or "").strip()[:4000]

        # ── каталог фото-референсов: недавние N сообщений + смысловой поиск по индекс-памяти (вся история) ──
        catalog = []
        build_cat = (not noimg and not raw)
        if build_cat:
            want_vision = active_model_supports_vision()
            if want_vision is None:  # кастомная OR-модель без флага — спросим вживую
                try:
                    _cl, _mid, _lbl = get_active_model()
                    _ex, want_vision, _ctx, _nm = await _openrouter_model_info(_mid)
                except Exception:
                    want_vision = False
            want_vision = bool(want_vision) and not force_desc  # -m → принудительно режим описаний
            if n > 0:  # недавние фото из окна N
                cand_limit = GEN_CTX_IMG_MAX if want_vision else GEN_CTX_IMG_MAX_DESC  # прямой режим ≤20, описания — большой пул
                cat_timeout = GEN_CATALOG_TIMEOUT if want_vision else GEN_DESC_TIMEOUT
                catalog = await _gen_history_catalog(ordered, want_vision, limit=cand_limit, timeout=cat_timeout, progress_cb=set_status)
            # индекс-память: релевантные всему запросу фото и досье персонажей (внешность) — из ВСЕЙ истории, не только N
            if _index_memory_allowed(is_owner) and await _index_chat_indexed(event.chat_id):
                await set_status("🧠 Ищу референсы и контекст в памяти чата…")
                idx_items, idx_ctx = await _gen_index_candidates(event.chat_id, user_prompt)
                if idx_items:
                    seen_mids = {it["mid"] for it in catalog}
                    for it in idx_items:
                        if it["mid"] not in seen_mids:  # дедуп с недавним каталогом
                            catalog.append(it)
                            seen_mids.add(it["mid"])
                if idx_ctx:  # внешность релевантных персонажей → в контекст промптера
                    context_text = (context_text + "\n\n" + idx_ctx) if context_text else idx_ctx
            for i, it in enumerate(catalog, 1):  # сквозная нумерация после слияния источников
                it["idx"] = i

        # ── анти-повтор: подписи прошлых генераций (💡 идея / 🎨 промпт, шлёт сам юзербот → m.out) лежат в логе
        # как обычные сообщения — модель охотно берёт оттуда готовую идею и повторяет уже сделанное. Собираем их
        # явным списком «уже сгенерировано» для промптера ──
        past_gens, _pg_seen = [], set()
        if n > 0 and not raw:
            for m in ordered:
                if not getattr(m, "out", False):
                    continue
                t = (m.raw_text or "").strip()
                if t.startswith("💡"):  # подпись «💡 идея\n🎨 промпт» — идея (первая строка) самодостаточна
                    pg = t.splitlines()[0].lstrip("💡").strip()[:300]
                elif t.startswith("🎨"):  # старые подписи без идеи — начало промпта
                    pg = t.lstrip("🎨").strip()[:200]
                else:
                    continue
                if pg and pg not in _pg_seen:
                    _pg_seen.add(pg)
                    past_gens.append(pg)
            past_gens = past_gens[-8:]  # последние 8 — достаточно против повтора, не раздувая запрос
            if past_gens:
                log("GEN", f"Анти-повтор: в окне {len(past_gens)} прошлых генераций — передаю промптеру список «не повторяй»")

        gen_refs_line = None  # строка «Референсы:» со ссылками на сообщения-источники (заполнится, если ИИ возьмёт фото из истории)
        ai_prompt = (not raw) and bool(context_text or improve or creative or catalog)  # -r → ИИ не строит промпт (literal)
        edit_mode = bool(input_b64s) and not creative  # -c → творческий режим даже с референсом
        # Активная текстовая модель не видит вложенные референсы → их описывает медиа-модель (с кэшем). Считаем ОДИН раз на весь пакет.
        image_desc = None
        if ai_prompt and input_b64s:
            await set_status("👁 Изучаю референсы (vision)…")
            descs, refused = [], 0
            for i, _b64 in enumerate(input_b64s[:3], 1):  # описываем до 3 первых — этого хватает для контекста
                try:
                    d = await describe_image(base64.b64decode(_b64))
                    if not d or d == "[изображение]":
                        continue
                    if _looks_like_refusal(d):  # vision отказался (цензура) — НЕ суём отказ в промпт
                        refused += 1
                        log("GEN", f"Vision отказался описать референс {i}: «{d[:80]}»")
                        continue
                    descs.append(f"Референс {i}: {d}")
                except Exception as e:
                    log("GEN", f"Описание референса {i} не удалось: {e}")
            image_desc = "\n".join(descs) or None
            if refused and not descs:
                await set_status("👁 Vision не смог описать фото (фильтр) — генерирую по фото и тексту без описания…")

        # ── ПАКЕТНАЯ генерация (-xN): N вариантов → в Избранное (Saved Messages), прогресс в текущем чате ──
        if batch_count > 1:
            await set_status(f"🎨 Пакет {batch_count} → в Избранное: придумываю уникальные промпты…")
            counter = {"done": 0, "ok": 0, "exhausted": False}
            sem = asyncio.Semaphore(GEN_BATCH_CONCURRENCY)

            async def _gen_and_send(idx, fp, by_ai, idea_i=None):
                async with sem:
                    if counter["exhausted"]:  # дневной лимит уже исчерпан — не тратим квоту на обречённый запрос
                        counter["done"] += 1
                        return
                    raw_i, mime_i, used_fp, _fb = await _gen_one_image(
                        fp, input_b64s, image_size, aspect_ratio, (by_ai or ai_prompt), user_prompt)
                    counter["done"] += 1
                    if raw_i is not None:
                        try:
                            await _gen_send_image("me", raw_i, mime_i, used_fp, by_ai, None, refs_line=gen_refs_line, idea=idea_i)
                            counter["ok"] += 1
                        except Exception as e:
                            log("GEN", f"Вариант {idx + 1}: отправка в Избранное не удалась: {e}")
                    else:
                        if mime_i == "exhausted":
                            counter["exhausted"] = True  # дневной лимит — стоп остальным вариантам
                        log("GEN", f"Вариант {idx + 1} не сгенерирован ({mime_i})")
                    await set_status(f"🎨 {counter['done']}/{batch_count} готово · {counter['ok']} в Избранном…")

            # Промпты строим ПОСЛЕДОВАТЕЛЬНО: каждому показываем все предыдущие, активная модель САМА придумывает
            # непохожий (без навязанных «углов»). Каталог истории/картинки шлём ТОЛЬКО на 1-м варианте — выбранные
            # ИИ референсы фиксируем для всех. Генерацию (медленную) сразу запускаем в фоне → перекрытие.
            prompts, tasks = [], []
            for i in range(batch_count):
                if counter["exhausted"]:  # дневной лимит исчерпан — не строим и не шлём остаток
                    log("GEN", f"Дневной лимит исчерпан — останавливаю пакет на варианте {i + 1}/{batch_count}")
                    break
                cat_i = (catalog or None) if i == 0 else None  # каталог (и картинки vision) — только 1-му варианту
                fp, sel, idea_i, asp_i = await _build_gen_prompt(user_prompt, context_text, image_desc, edit_mode, prompts, cat_i, creative=creative, improve=improve, force_desc=force_desc, past_gens=past_gens)
                by_ai = fp != user_prompt
                if i == 0 and sel:  # выбор референсов из истории — общий для всего пакета
                    input_b64s, _used = _merge_catalog_refs(input_b64s, catalog, [k for k, _ in sel])
                    try:
                        gen_refs_line = _gen_refs_line(await event.get_chat(), catalog, _used, reasons=dict(sel))
                    except Exception:
                        gen_refs_line = None
                if i == 0 and aspect_ratio is None and asp_i:  # ориентацию под идею выбирает модель (флаг юзера важнее); общая на пакет
                    aspect_ratio = asp_i
                    log("GEN", f"Ориентация от модели: {asp_i}")
                log("GEN", f"Вариант {i + 1}/{batch_count}: промпт by_ai={by_ai} refs={len(sel) if i == 0 else '—'} idea={idea_i or '—'} len={len(fp)}")
                prompts.append(fp)
                tasks.append(asyncio.create_task(_gen_and_send(i, fp, by_ai, idea_i)))
                if i + 1 < batch_count:
                    await set_status(f"🧠 Промпты {i + 1}/{batch_count} · 🎨 {counter['done']}/{batch_count} готово…")
            await asyncio.gather(*tasks)
            if counter["exhausted"]:
                await set_status(f"⚠️ {counter['ok']} готово, но дневной лимит бесплатной модели исчерпан "
                                 f"(50 запросов/день; провальные тоже считаются). Остальное — завтра или подними лимит. 📌")
            else:
                await set_status(f"✅ {counter['ok']}/{batch_count} вариантов отправлено тебе в Избранное (Saved Messages) 📌")
            await asyncio.sleep(8)
            try:
                await status.delete()
            except Exception:
                pass
            return

        # ── одиночная генерация ──
        final_prompt, prompt_by_ai, gen_idea = user_prompt, False, None
        if ai_prompt:
            await set_status(f"🧠 {get_active_model()[2]} {'смотрит фото и пишет промпт' if catalog else 'готовит промпт'}…")
            final_prompt, sel, gen_idea, gen_asp = await _build_gen_prompt(user_prompt, context_text, image_desc, edit_mode, None, catalog or None, creative=creative, improve=improve, force_desc=force_desc, past_gens=past_gens)
            prompt_by_ai = final_prompt != user_prompt
            if aspect_ratio is None and gen_asp:  # ориентацию под идею выбирает модель; явный флаг -v/-h/-sq важнее
                aspect_ratio = gen_asp
                log("GEN", f"Ориентация от модели: {gen_asp}")
            input_b64s, _used = _merge_catalog_refs(input_b64s, catalog, [k for k, _ in sel])  # выбранные ИИ картинки из истории → референсы
            if _used:
                try:
                    gen_refs_line = _gen_refs_line(await event.get_chat(), catalog, _used, reasons=dict(sel))
                except Exception:
                    gen_refs_line = None
            log("GEN", f"Промпт: by_ai={prompt_by_ai} · режим={'edit' if edit_mode else ('improve' if improve else 'creative')} · ref-вложений={'есть' if image_desc else 'нет'} · из истории refs={len(_used)} · idea={gen_idea or '—'} · len={len(final_prompt)}")
        await set_status("🎨 Генерирую изображение… (может занять до пары минут)")
        t0 = time.time()
        # allow_repair: ИИ-промпт был ЗАПРОШЕН (даже если его первая попытка вернула пустое и промпт ушёл исходным)
        raw, mime, used_fp, used_fb = await _gen_one_image(
            final_prompt, input_b64s, image_size, aspect_ratio,
            (prompt_by_ai or ai_prompt), user_prompt, set_status)
        if raw is None:
            if mime == "moderation":
                await set_status("❌ Запрос отклонён модерацией провайдера.\n"
                                 f"Переформулируй промпт и попробуй снова: `/gen {user_prompt[:200]}`")
            elif mime == "exhausted":
                await set_status("❌ Дневной лимит бесплатной модели исчерпан (50 запросов/день; провальные тоже считаются).\n"
                                 "Попробуй завтра, или подними лимит до 1000/день, пополнив баланс OpenRouter на $10.")
            else:
                await set_status("❌ Провайдер генерации сейчас перегружен (лимит ~5 запросов/мин).\n"
                                 f"Попробуй ещё раз через минуту: `/gen {user_prompt[:200]}`")
            return
        log("GEN", f"Готово за {time.time() - t0:.1f}с · {len(raw) / 1024:.0f} КБ · {mime} · prompt_by_ai={prompt_by_ai} · модель={'fast(запасная)' if used_fb else 'pro'}")
        await _gen_send_image(event.chat_id, raw, mime, used_fp, prompt_by_ai, reply_target_id, refs_line=gen_refs_line, idea=gen_idea)
        await status.delete()
    except Exception as e:
        log("GEN", f"Ошибка /gen: {e}")
        traceback.print_exc()
        await set_status(f"❌ Генерация не удалась: {e}\nПопробуй ещё раз: `/gen {user_prompt[:200]}`")


@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]auto_reply$", from_users="me"))
async def auto_reply_on(event):
    if await _slash_for_other_bot(event):
        return
    AUTO_REPLY_ACTIVE_CHATS.add(event.chat_id)
    _save_auto_reply()
    log("AUTO", f"Авто-ответ включён в чате {event.chat_id}")
    await event.edit("✅ Авто-ответ включён")
    await asyncio.sleep(2)
    await event.delete()


@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]auto_reply\s+off$", from_users="me"))
async def auto_reply_off(event):
    if await _slash_for_other_bot(event):
        return
    AUTO_REPLY_ACTIVE_CHATS.discard(event.chat_id)
    _save_auto_reply()
    AUTO_REPLY_BUFFERS.pop(event.chat_id, None)
    AUTO_REPLY_HISTORY.pop(event.chat_id, None)
    AUTO_REPLY_BUSY.discard(event.chat_id)
    task = AUTO_REPLY_TASKS.pop(event.chat_id, None)
    if task and not task.done():
        task.cancel()
    log("AUTO", f"Авто-ответ выключен в чате {event.chat_id}")
    await event.edit("🔴 Авто-ответ выключен")
    await asyncio.sleep(2)
    await event.delete()


async def flush_auto_reply_buffer(chat_id):
    current = asyncio.current_task()
    try:
        await asyncio.sleep(AUTO_REPLY_ACCUMULATE_WINDOW)  # debounce — отменяемо, буфер цел
        # Забираем буфер и помечаем busy СИНХРОННО (без await между строками) —
        # пока выполняется этот участок, входящие не вклиниваются (asyncio однопоточно).
        events_list = AUTO_REPLY_BUFFERS.get(chat_id) or []
        if not events_list:
            return
        AUTO_REPLY_BUFFERS[chat_id] = []
        AUTO_REPLY_BUSY.add(chat_id)  # с этого момента входящие НЕ отменяют нас (иначе потеряем events_list)
        try:
            if len(events_list) > 1:
                log("AUTO", f"Аккумулировано сообщений: {len(events_list)}")

            combined, _d, _f, _ct = await assemble_context(events_list, text_only=False)
            combined = combined.strip()
            save_media_cache()
            if not combined:
                return

            history = AUTO_REPLY_HISTORY.get(chat_id, [])
            reply = await generate_auto_reply(combined, history)

            async with client.action(chat_id, "typing"):
                await asyncio.sleep(min(len(reply) * 0.04, 4.0))

            try:
                await client.send_message(chat_id, reply)
                log("AUTO", f"Ответ отправлен в {chat_id}")
            except FloodWaitError as e:
                await asyncio.sleep(e.seconds)
                await client.send_message(chat_id, reply)

            # обновляем память диалога
            history = history + [
                {"role": "user", "content": combined},
                {"role": "assistant", "content": reply},
            ]
            AUTO_REPLY_HISTORY[chat_id] = history[-AUTO_REPLY_HISTORY_MAX:]
        finally:
            AUTO_REPLY_BUSY.discard(chat_id)

        # Пока обрабатывали — могли прийти новые сообщения (они не отменяли нас). Дофлашим.
        if AUTO_REPLY_BUFFERS.get(chat_id):
            AUTO_REPLY_TASKS[chat_id] = asyncio.create_task(flush_auto_reply_buffer(chat_id))

    except asyncio.CancelledError:
        return
    except Exception as e:
        log("AUTO", f"Ошибка flush_auto_reply_buffer: {e}")
        traceback.print_exc()
    finally:
        # снимаем себя из реестра, только если слот всё ещё наш (не перезапущенная таска)
        if AUTO_REPLY_TASKS.get(chat_id) is current:
            AUTO_REPLY_TASKS.pop(chat_id, None)


@client.on(events.NewMessage(incoming=True))
async def auto_reply_incoming(event):
    if event.chat_id not in AUTO_REPLY_ACTIVE_CHATS:
        return
    if event.raw_text and event.raw_text.startswith((".", "/")):
        return  # команды (/ask, /ask и пр.) не должны попадать в авто-ответ
    if not (event.raw_text or _media_tag(event)):
        return

    chat_id = event.chat_id
    if chat_id not in AUTO_REPLY_BUFFERS:
        AUTO_REPLY_BUFFERS[chat_id] = []
    AUTO_REPLY_BUFFERS[chat_id].append(event)

    if chat_id in AUTO_REPLY_BUSY:
        return  # обработка уже идёт — не отменяем её; завершившись, она дофлашит буфер

    existing_task = AUTO_REPLY_TASKS.get(chat_id)
    if existing_task and not existing_task.done():
        existing_task.cancel()

    AUTO_REPLY_TASKS[chat_id] = asyncio.create_task(flush_auto_reply_buffer(chat_id))


@client.on(events.NewMessage(outgoing=True, pattern=re.compile(r"^[./]song(?: |$)(.*)", re.DOTALL), from_users="me"))
async def song_command(event):
    if await _slash_for_other_bot(event):
        return
    custom_text = event.pattern_match.group(1).strip()
    # опциональное первое число = размер чанка (символов за шаг): `/song 5 текст` или `/song 5` (дефолтный текст)
    chunk_size = 3
    mnum = re.match(r"^(\d{1,3})(?:\s+(.*))?$", custom_text, re.DOTALL)
    if mnum:
        chunk_size = max(1, min(int(mnum.group(1)), 200))
        custom_text = (mnum.group(2) or "").strip()
    text_to_print = custom_text if custom_text else SONG_TEXT
    await event.delete()
    await print_lyrics(event.chat_id, text_to_print, chunk_size)


@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]channels(?:\s+(\w+))?(?:\s+(.+))?$", from_users="me"))
async def channels_command(event):
    if await _slash_for_other_bot(event):
        return
    global LAST_SCAN
    sub = (event.pattern_match.group(1) or "").lower()
    arg = (event.pattern_match.group(2) or "").strip()
    tracked = get_tracked()

    if not sub:
        if not tracked:
            await event.edit("Каналы не отслеживаются. `/channels scan` — найти, `/channels add @name` — добавить.")
            return
        lines = ["📡 Отслеживаемые каналы:"]
        for i, ch in enumerate(tracked, 1):
            uname = f"@{ch['username']}" if ch.get("username") else f"id{ch['id']}"
            lines.append(f"{i}. {uname} — {ch.get('title', '')}")
        lines.append("\n`/channels remove N` — убрать")
        await event.edit("\n".join(lines))
        return

    if sub == "scan":
        await event.edit("🔍 Сканирую диалоги…")
        tracked_ids = {ch["id"] for ch in tracked}
        found = []
        async for dialog in client.iter_dialogs():
            ent = dialog.entity
            if getattr(ent, "broadcast", False) and not getattr(ent, "megagroup", False):
                found.append({
                    "id": utils.get_peer_id(ent),
                    "title": getattr(ent, "title", "") or "",
                    "username": getattr(ent, "username", None),
                })
        LAST_SCAN = found
        if not found:
            await event.edit("Не найдено ни одного канала.")
            return
        lines = [f"📱 Найдено каналов: {len(found)}"]
        for i, ch in enumerate(found, 1):
            mark = "✅ " if ch["id"] in tracked_ids else ""
            uname = f"@{ch['username']}" if ch.get("username") else f"id{ch['id']}"
            lines.append(f"{i}. {mark}{uname} — {ch['title']}")
        lines.append("\n`/channels add N` — добавить по номеру")
        await event.edit("\n".join(lines)[:4000])
        return

    if sub == "add":
        if not arg:
            await event.edit("Укажи номер из scan или @username: `/channels add 3`")
            return
        if arg.isdigit():
            idx = int(arg) - 1
            if not LAST_SCAN:
                await event.edit("Сначала выполни `/channels scan` — список каналов не загружен.")
                return
            if not (0 <= idx < len(LAST_SCAN)):
                await event.edit("Нет такого номера. Сначала `/channels scan`.")
                return
            ch = LAST_SCAN[idx]
        else:
            ent = await resolve_channel(arg)
            if ent is None:
                await event.edit(f"Не удалось найти канал {arg}")
                return
            ch = {"id": utils.get_peer_id(ent), "title": getattr(ent, "title", "") or "", "username": getattr(ent, "username", None)}
        if any(c["id"] == ch["id"] for c in tracked):
            await event.edit(f"Канал «{ch['title']}» уже отслеживается.")
            return
        tracked.append(ch)
        save_tracked(tracked)
        await event.edit(f"✅ Добавлен: {ch['title']}")
        return

    if sub == "remove":
        if not arg:
            await event.edit("Укажи номер или @username: `/channels remove 2`")
            return
        removed = None
        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(tracked):
                removed = tracked.pop(idx)
        else:
            key = arg.lstrip("@").lower()
            for i, c in enumerate(tracked):
                if (c.get("username") or "").lower() == key or str(c["id"]) == key:
                    removed = tracked.pop(i)
                    break
        if removed is None:
            await event.edit("Не нашёл такой канал в списке.")
            return
        save_tracked(tracked)
        await event.edit(f"🗑 Убран: {removed.get('title', '')}")
        return

    await event.edit("Неизвестная подкоманда. `/channels`, `/channels scan`, `/channels add`, `/channels remove`")


@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]search\s+(.+)$", from_users="me"))
async def search_command(event):
    if await _slash_for_other_bot(event):
        return
    query = event.pattern_match.group(1).strip()
    if not get_tracked():
        await event.edit("Нет отслеживаемых каналов. `/channels scan`")
        return
    await event.edit(f"🔍 Ищу «{query}»…")
    try:
        results = await search_channels(query, per_channel=5, total=10)
        if not results:
            await event.edit(f"🔍 «{query}» — ничего не найдено")
            return
        lines = [f"🔍 «{query}» — {len(results)} результатов\n"]
        for _date, ent, msg_id, raw in results:
            uname = getattr(ent, "username", None)
            src = f"@{uname}" if uname else getattr(ent, "title", "канал")
            lines.append(f"📅 {_fmt_date(_date)} · {src}")
            lines.append(f"📝 {_preview(raw, 100)}")
            lines.append(f"🔗 {build_msg_link(ent, msg_id)}")
        await event.edit("\n".join(lines)[:4000])
    except Exception as e:
        log("SEARCH", f"Ошибка /search: {e}")
        traceback.print_exc()
        await event.edit("Ошибка поиска, см. логи.")


async def send_digest(manual: bool):
    tracked = get_tracked()
    if not tracked:
        if manual:
            await client.send_message("me", "Нет отслеживаемых каналов. `/channels scan`")
        return

    state = load_json(DIGEST_STATE_PATH, {})
    last_sent = state.get("last_sent")
    if last_sent:
        since = datetime.fromisoformat(last_sent)
    else:
        since = datetime.now(MSK) - timedelta(hours=24)

    collected = []
    for ch in tracked:
        ent = await resolve_channel(ch)
        if ent is None:
            continue
        per_channel = 0
        try:
            async for m in client.iter_messages(ent, limit=50):
                if m.date <= since:
                    break
                if not m.raw_text:
                    continue
                link = build_msg_link(ent, m.id)
                collected.append(f"📅 {_fmt_date(m.date)} [{ch.get('title', '')}] {_preview(m.raw_text, 300)}\n{link}")
                per_channel += 1
                if per_channel >= 15:
                    break
        except FloodWaitError as e:
            await asyncio.sleep(e.seconds)
        except Exception as e:
            log("DIGEST", f"Ошибка сбора из {ch.get('title', '')}: {e}")
        await asyncio.sleep(0.3)
        if len(collected) >= 80:
            break

    if not collected:
        if manual:
            await client.send_message("me", "📰 Нет новых постов с прошлого дайджеста.")
        log("DIGEST", "Нет новых постов")
        return

    result = await _llm_create(
        messages=[
            {"role": "system", "content": DIGEST_SYSTEM_PROMPT},
            {"role": "user", "content": "Посты за период:\n\n" + "\n\n".join(collected)},
        ],
        max_tokens=8000,            # дайджест по 80 постам бывает длинным
        temperature=1.0,
        reasoning="none",           # утилитарная задача: без размышлений (иначе CoT съедал бюджет и тёк в вывод)
        model_slug="deepseek-flash",  # дайджест ВСЕГДА на DeepSeek V4 Flash (полный off ризонинга), не на активной
    )
    if not result:
        if manual:
            await client.send_message("me", "Дайджест: DeepSeek не ответил.")
        return

    today = datetime.now(MSK).strftime("%d.%m.%Y")
    await send_long("me", result, prefix=f"📰 Дайджест — {today}\n\n")
    state["last_sent"] = datetime.now(MSK).isoformat()
    save_json(DIGEST_STATE_PATH, state)
    log("DIGEST", f"Дайджест отправлен ({len(collected)} постов, manual={manual})")


@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]digest$", from_users="me"))
async def digest_command(event):
    if await _slash_for_other_bot(event):
        return
    await event.edit("📰 Собираю дайджест…")
    try:
        await send_digest(manual=True)
        await event.delete()
    except Exception as e:
        log("DIGEST", f"Ошибка /digest: {e}")
        traceback.print_exc()


@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]digest\s+time\s+(\d{1,2}:\d{2})$", from_users="me"))
async def digest_time_command(event):
    if await _slash_for_other_bot(event):
        return
    t = event.pattern_match.group(1)
    state = load_json(DIGEST_STATE_PATH, {})
    state["digest_time"] = t
    save_json(DIGEST_STATE_PATH, state)
    await event.edit(f"⏰ Время дайджеста: {t} МСК")


async def scheduler_loop():
    log("DIGEST", "Планировщик дайджеста запущен")
    while True:
        try:
            st = load_json(DIGEST_STATE_PATH, {})
            hh, mm = map(int, st.get("digest_time", "09:00").split(":"))
            now = datetime.now(MSK)
            last = st.get("last_sent")
            last_date = datetime.fromisoformat(last).date() if last else None
            if now.hour == hh and now.minute >= mm and now.minute < mm + 2 and last_date != now.date():
                await send_digest(manual=False)
        except Exception as e:
            log("DIGEST", f"scheduler error: {e}")
        await asyncio.sleep(60)


@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]model(?:\s+(.+))?$", from_users="me"))
async def model_command(event):
    if await _slash_for_other_bot(event):
        return
    global ACTIVE_MODEL, ACTIVE_MEDIA_MODEL, REASONING_EFFORT
    arg = (event.pattern_match.group(1) or "").strip()
    slugs = list(MODEL_REGISTRY.keys())

    def is_available(provider):
        return _client_for_provider(provider) is not None

    def tool_mark(slug):
        ts = MODEL_TOOLS_SUPPORT.get(slug)
        return " 🔧" if ts is True else (" 🚫" if ts is False else " ❔")

    # --- выбор медиа-модели (vision): /model media [N|slug] ---
    if arg.lower().startswith("media"):
        marg = arg[len("media"):].strip()
        # Единый нумерованный список: OpenRouter-пресеты + OpenCode-Go vision-модели.
        # Элемент: (provider, slug, model_id, label)
        media_items = [("openrouter", ms, MEDIA_MODEL_REGISTRY[ms][0], MEDIA_MODEL_REGISTRY[ms][1]) for ms in MEDIA_MODEL_REGISTRY]
        media_items += [("opencode", ms, MODEL_REGISTRY[ms][1], MODEL_REGISTRY[ms][2]) for ms in MEDIA_OPENCODE_SLUGS if ms in MODEL_REGISTRY]
        media_by_slug = {ms: (prov, mid, mlabel) for prov, ms, mid, mlabel in media_items}
        if not marg:
            lines = ["🖼 **Медиа-модели (vision)** — ▶ активная:"]
            for i, (prov, ms, mid, mlabel) in enumerate(media_items, 1):
                mk = f"▶{i}." if ms == ACTIVE_MEDIA_MODEL else f"{i}."
                cl = openrouter_client if prov == "openrouter" else opencode_client
                avail = "" if cl else " ⚠️нет ключа"
                ptag = "OR" if prov == "openrouter" else "OC"
                lines.append(f"{mk} `{ms}` — {mlabel} [{ptag}] (`{mid}`){avail}")
            if ACTIVE_MEDIA_MODEL not in media_by_slug:
                lines.append(f"▶ (кастомная OpenRouter) `{ACTIVE_MEDIA_MODEL}`")
            lines.append("\n[OR]=OpenRouter · [OC]=OpenCode Go · аудио/голос — всегда Parakeet (STT).")
            lines.append("`/model media N` / `/model media <slug>` — выбрать")
            lines.append("`/model media <model-id>` — любая модель OpenRouter (с проверкой)")
            await event.edit("\n".join(lines)[:4000])
            return
        # 1) по номеру  2) по slug из объединённого списка  3) кастомный id OpenRouter (с валидацией)
        if marg.isdigit() and 1 <= int(marg) <= len(media_items):
            prov, chosen_m, mid, mlabel = media_items[int(marg) - 1]
            ACTIVE_MEDIA_MODEL = chosen_m
            _save_model_state()
            ptag = "OpenRouter" if prov == "openrouter" else "OpenCode Go"
            log("MODEL", f"Активная медиа-модель: {chosen_m} ({mid}, {ptag})")
            await event.edit(f"✅ Медиа-модель (vision): {mlabel} (`{mid}`, {ptag})")
            return
        if marg in media_by_slug:
            prov, mid, mlabel = media_by_slug[marg]
            ACTIVE_MEDIA_MODEL = marg
            _save_model_state()
            ptag = "OpenRouter" if prov == "openrouter" else "OpenCode Go"
            log("MODEL", f"Активная медиа-модель: {marg} ({mid}, {ptag})")
            await event.edit(f"✅ Медиа-модель (vision): {mlabel} (`{mid}`, {ptag})")
            return
        # кастомный id — проверяем в OpenRouter
        await event.edit(f"🔎 Проверяю `{marg}` в OpenRouter…")
        exists, supports_img, _ctx_len, _name = await _openrouter_model_info(marg)
        if exists is None:
            await event.edit(f"⚠️ Не удалось проверить `{marg}` (OpenRouter недоступен). Модель не изменена.")
            return
        if not exists:
            await event.edit(f"❌ Модель `{marg}` не найдена в OpenRouter. Проверь точный id (см. openrouter.ai/models).")
            return
        ACTIVE_MEDIA_MODEL = marg
        _save_model_state()
        log("MODEL", f"Активная медиа-модель (кастомная): {marg}, vision={supports_img}")
        warn = "" if supports_img else "\n⚠️ Модель не поддерживает изображения — описание фото работать не будет (голос/аудио идут через Parakeet)."
        await event.edit(f"✅ Медиа-модель (vision): `{marg}` (кастомная, OpenRouter){warn}")
        return

    # --- избранное (кастомные OpenRouter-модели): /model fav ---
    if arg.lower() in ("fav", "favorites", "избранное"):
        if not CUSTOM_MODELS:
            await event.edit("⭐ Избранное (кастомные OpenRouter-модели) пусто.\nДобавь: `/model vendor/model` (напр. `/model openai/gpt-4o`).")
            return
        lines = ["⭐ **Избранные OpenRouter-модели:**"]
        for i, (mid, ci) in enumerate(CUSTOM_MODELS.items(), 1):
            mk = "▶" if mid == ACTIVE_MODEL else " "
            n = slugs.index(mid) + 1 if mid in slugs else None  # номер в общем списке /model
            num = f" · быстрый выбор `/model {n}`" if n else ""
            lines.append(f"{mk}{i}. {ci.get('label') or mid} — `{mid}`{num}")
        lines.append("\n`/model N` — выбрать по номеру из общего списка · `/model <vendor/model>` — добавить · `/model remove <N|id>` — удалить")
        await event.edit("\n".join(lines)[:4000])
        return

    # --- удаление кастомной OpenRouter-модели: /model remove <N|slug> ---
    if arg.lower().startswith("remove"):
        marg = arg[len("remove"):].strip()
        fav_ids = list(CUSTOM_MODELS.keys())
        target = None
        if marg.isdigit() and 1 <= int(marg) <= len(fav_ids):
            target = fav_ids[int(marg) - 1]
        elif marg in CUSTOM_MODELS:
            target = marg
        if not target:
            await event.edit(f"Не нашёл кастомную модель: `{marg}`. `/model fav` — список избранных.")
            return
        was_active = (ACTIVE_MODEL == target)
        CUSTOM_MODELS.pop(target, None)
        MODEL_REGISTRY.pop(target, None)
        if was_active:
            ACTIVE_MODEL = "deepseek-pro"
        _save_model_state()
        log("MODEL", f"Удалена кастомная модель: {target}")
        await event.edit(f"🗑 Удалена из избранного: `{target}`." + (" Активная модель сброшена на DeepSeek." if was_active else ""))
        return

    # --- глубина размышлений (OpenAI и Gemini): /model reason [уровень|auto] ---
    if arg.lower().startswith("reason"):
        rarg = arg[len("reason"):].strip().lower()
        active_provider = MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["deepseek-pro"])[0]
        if not rarg:
            reff_now = REASONING_EFFORT or "авто"
            glob = " · ".join((f"▶`{lv}`" if lv == REASONING_EFFORT else f"`{lv}`") for lv in _REASONING_RANK)
            glob += " · ▶`auto`" if REASONING_EFFORT is None else " · `auto`"
            lines = [
                f"🤔 **Глубина размышлений** · сейчас: `{reff_now}`",
                "",
                "**Глобально** для всех reasoning-моделей — `/model reason <уровень>`:",
                "  " + glob,
                "",
                "**По моделям** — тапни `N.M` (N — № в `/model`, `.1` = максимум силы → дальше слабее → off):",
            ]
            # группируем по ФОРМЕ лесенки: одинаковые схлопываются (16 моделей Fireworks+opencode → одна строка),
            # чтобы не плодить дубли. Под каждой формой — модели с тап-чипами N.M.
            groups = {}  # ladder-строка → [(№, label, число ступеней)]
            for i, slug in enumerate(slugs, 1):
                lv = _reasoning_levels(slug)
                if not lv:
                    continue
                prov, mid, label, _c, _s = MODEL_REGISTRY[slug]
                ladder = " · ".join(f".{j} {_fmt_rlevel(mid, x, prov)}" for j, x in enumerate(lv, 1))
                groups.setdefault(ladder, []).append((i, label, len(lv)))
            for ladder, members in groups.items():
                lines.append(f"\n▸ {ladder}")
                for i, label, k in members:
                    chips = " ".join(f"`{i}.{j}`" for j in range(1, k + 1))
                    lines.append(f"   {chips} — {label}")
            if not _supports_reasoning(active_provider):
                lines.append("\n⚠️ Активная модель без управления ризонингом — уровень применится после выбора reasoning-модели.")
            await event.edit("\n".join(lines)[:4000])
            return
        if rarg.isdigit() and 1 <= int(rarg) <= len(_REASONING_RANK) + 1:
            rarg = (_REASONING_RANK + ["auto"])[int(rarg) - 1]
        if rarg in ("auto", "сброс", "reset", "off", "default"):
            REASONING_EFFORT = None
            _save_model_state()
            log("MODEL", "Ризонинг: auto (дефолт модели)")
            await event.edit("✅ Глубина размышлений: авто (дефолт модели — 5.5→medium, 5.4/5.4-mini→none, o3/o4-mini/Gemini→medium).")
            return
        if rarg in _REASONING_RANK:
            REASONING_EFFORT = rarg
            _save_model_state()
            note = ""
            if _supports_reasoning(active_provider):
                applied = _clamp_reasoning(MODEL_REGISTRY[ACTIVE_MODEL][1], rarg, active_provider)
                if applied != rarg:
                    note = f" (для активной {MODEL_REGISTRY[ACTIVE_MODEL][2]} применится `{applied}`)"
            else:
                note = " — сработает на моделях OpenAI (GPT-5.x / o3), Gemini, Fireworks, opencode и DeepSeek"
            log("MODEL", f"Ризонинг: {rarg}")
            await event.edit(f"✅ Глубина размышлений: `{rarg}`{note}")
            return
        await event.edit("Не понял уровень. `/model reason` — список: " + " · ".join(f"`{lv}`" for lv in _REASONING_RANK) + " · `auto`.")
        return

    if not arg:
        lines = [
            "╭───────────────────────╮",
            "│   🧠  МОДЕЛИ ОТВЕТОВ   │",
            "╰───────────────────────╯",
            "▶ активная · (N) окно · 👁 картинки (-g) · 🤔 глубина размышлений (N.M) · 🔧 поиск · 🚫 нет · ❔ не проверено",
        ]
        cur_provider = None
        for i, slug in enumerate(slugs, 1):
            provider, _mid, label, ctx, _safety = MODEL_REGISTRY[slug]
            if provider != cur_provider:
                cur_provider = provider
                title = {"deepseek": "━━ Прямой API ━━", "opencode": "━━ OpenCode Go ━━",
                         "oc_anthropic": "━━ OpenCode Go (нативный) ━━",
                         "modelgate": "━━ Claude (ModelGate) ━━",
                         "openai": "━━ OpenAI (GPT-5/o3) ━━",
                         "google": "━━ Google Gemini ━━",
                         "zai": "━━ z.ai (GLM) ━━",
                         "fireworks": "━━ Fireworks ━━",
                         "sakana": "━━ Sakana AI (Fugu) ━━",
                         "gloy": "━━ LLM API FUN (Gloy AI) ━━",
                         "openrouter": "━━ OpenRouter (кастом) ━━"}.get(provider, f"━━ {provider} ━━")
                lines.append(f"\n{title}")
            mark = f"▶{i}." if slug == ACTIVE_MODEL else f"{i}."
            warn = " ⚠️нет ключа" if not is_available(provider) else ""
            vmark = " 👁" if _model_supports_vision(slug) else ""  # видит картинки напрямую (-g)
            rmark = " 🤔" if _reasoning_levels(slug) else ""        # умеет менять глубину размышлений (N.M); уровни — в памятке внизу
            # на АКТИВНОЙ модели показываем применяемый уровень прямо на бейдже (🤔high), чтобы было видно что выбрано
            if slug == ACTIVE_MODEL and _reasoning_levels(slug):
                rmark = f" 🤔{_clamp_reasoning(_mid, REASONING_EFFORT, provider)}" if REASONING_EFFORT else " 🤔авто"
            lines.append(f"{mark} `{slug}` — {label}{vmark}{rmark} ({_fmt_ctx(ctx)}){tool_mark(slug)}{warn}")
        if ACTIVE_MEDIA_MODEL in MEDIA_MODEL_REGISTRY:
            media_label = MEDIA_MODEL_REGISTRY[ACTIVE_MEDIA_MODEL][1]
        elif ACTIVE_MEDIA_MODEL in MEDIA_OPENCODE_SLUGS and ACTIVE_MEDIA_MODEL in MODEL_REGISTRY:
            media_label = f"{MODEL_REGISTRY[ACTIVE_MEDIA_MODEL][2]} [OpenCode]"
        else:
            media_label = f"{ACTIVE_MEDIA_MODEL} (кастомная)"
        lines.append(f"\n🖼 медиа-модель: {media_label} · `/model media` — сменить")
        lines.append("`/model N` / `/model <slug>` — выбрать · `/model probe` — проверить поиск (❔→🔧/🚫)")
        reff = f"`{REASONING_EFFORT}`" if REASONING_EFFORT else "авто"
        lines.append(f"🤔 — модель умеет менять глубину размышлений. `/model N.M`: M — сила (`.1` максимум → дальше слабее → последний мин/выкл). Лесенки всех моделей с тап-чипами: `/model reason` (сейчас: {reff})")
        lines.append("`/model vendor/model` — добавить ЛЮБУЮ модель OpenRouter по id (напр. `/model openai/gpt-4o`)")
        lines.append("`/model fav` — избранные OR-модели · `/model remove <N|id>` — удалить кастомную")
        await event.edit("\n".join(lines)[:4000])
        return

    if arg.lower() == "probe":
        await event.edit("🔧 Проверяю поддержку поиска у моделей…")
        tested = 0
        for slug in slugs:
            provider, mid, _label, _ctx, _safety = MODEL_REGISTRY[slug]
            if provider in ("oc_anthropic", "openai", "google", "zai", "fireworks", "sakana", "gloy"):
                continue  # qwen3.7-max / gpt-5.x / o3 / Gemini / Fireworks / Sakana / Gloy: tools работают на auto, но forced пробник врёт (Sakana/Gloy отдают не tool_call) — флаг учится на лету в реальном /ask
            cl = _client_for_provider(provider)
            if cl is None:
                continue
            try:
                resp = await asyncio.to_thread(
                    cl.chat.completions.create,
                    model=mid,
                    messages=[{"role": "user", "content": "найди что-нибудь"}],
                    tools=[TELEGRAM_SEARCH_TOOL],
                    tool_choice={"type": "function", "function": {"name": "telegram_search"}},
                    max_tokens=20,
                )
                ok = bool(resp.choices[0].message.tool_calls)
                _set_tools_support(slug, ok)
            except Exception as e:
                # Ошибка про tool_choice = модель ПРИНЯЛА tools, отвергла лишь принудительный выбор
                # (thinking-режим) → tools поддерживаются. Иначе — не поддерживает.
                _set_tools_support(slug, _is_thinking_mode_quirk(e))
            tested += 1
            await asyncio.sleep(0.2)
        await event.edit(f"🔧 Проверено моделей: {tested}. Смотри `/model`.")
        return

    # --- выбор N.M: модель N с силой ризонинга M (OpenAI/Gemini; M=1 — мощнейший) ---
    m_nm = re.match(r"^(\d+)\.(\d+)$", arg)
    if m_nm:
        n, mlev = int(m_nm.group(1)), int(m_nm.group(2))
        if not (1 <= n <= len(slugs)):
            await event.edit(f"Нет модели с номером {n}. `/model` — список.")
            return
        slug_nm = slugs[n - 1]
        provider_nm, _midn, label_nm, ctx_nm, _sn = MODEL_REGISTRY[slug_nm]
        levels = _reasoning_levels(slug_nm)
        if not levels:
            await event.edit(f"Вариации `{n}.M` (сила ризонинга) доступны для OpenAI (GPT-5.x / o3), Gemini, Fireworks, opencode и DeepSeek. Для {label_nm} — просто `/model {n}`.")
            return
        if not (1 <= mlev <= len(levels)):
            opts = " · ".join(f"`{n}.{j}` {lv}" for j, lv in enumerate(levels, 1))
            await event.edit(f"У {label_nm} уровни 1–{len(levels)}: {opts}")
            return
        if not is_available(provider_nm):
            await event.edit(f"Модель «{label_nm}» недоступна — нет ключа провайдера ({provider_nm}).")
            return
        ACTIVE_MODEL = slug_nm
        REASONING_EFFORT = levels[mlev - 1]
        _save_model_state()
        applied = _clamp_reasoning(_midn, REASONING_EFFORT, provider_nm)
        rnote = f"`{REASONING_EFFORT}`" + (f" (→ `{applied}`)" if applied != REASONING_EFFORT else "")
        log("MODEL", f"Активная модель: {slug_nm} ({label_nm}), ризонинг {REASONING_EFFORT}→{applied}")
        await event.edit(f"✅ Модель ответов: {label_nm} (окно {_fmt_ctx(ctx_nm)}) · 🤔 ризонинг: {rnote}")
        return

    chosen = None
    if arg.isdigit():
        idx = int(arg) - 1
        if 0 <= idx < len(slugs):
            chosen = slugs[idx]
    elif arg in MODEL_REGISTRY:
        chosen = arg
    if not chosen:
        # Не номер и не известный slug → пробуем как id модели OpenRouter (vendor/model, с валидацией)
        if "/" in arg:
            await event.edit(f"🔎 Проверяю `{arg}` в OpenRouter…")
            exists, supports_img, ctx_len, name = await _openrouter_model_info(arg)
            if exists is None:
                await event.edit(f"⚠️ Не удалось проверить `{arg}` (OpenRouter недоступен). Модель не изменена.")
                return
            if not exists:
                await event.edit(f"❌ Модель `{arg}` не найдена в OpenRouter. Проверь точный id (см. openrouter.ai/models).")
                return
            if not openrouter_client:
                await event.edit("Модель найдена, но нет ключа OpenRouter — добавь OPENROUTER_API_KEY в .env.")
                return
            ctx = int(ctx_len or 128000)
            label = name or arg
            CUSTOM_MODELS[arg] = {"label": label, "ctx": ctx, "safety": 1.3, "vision": bool(supports_img)}  # vision — для /ask -g
            MODEL_REGISTRY[arg] = ("openrouter", arg, label, ctx, 1.3)
            ACTIVE_MODEL = arg
            _save_model_state()
            log("MODEL", f"Активная модель (кастомная OpenRouter): {arg}, окно {ctx}")
            await event.edit(f"✅ Модель ответов: {label} (`{arg}`, OpenRouter, окно {_fmt_ctx(ctx)})")
            return
        await event.edit(f"Нет такой модели: {arg}. `/model` — список, либо укажи id модели OpenRouter (vendor/model).")
        return

    provider, _mid, label, ctx, _safety = MODEL_REGISTRY[chosen]
    if not is_available(provider):
        await event.edit(f"Модель «{label}» недоступна — нет ключа провайдера ({provider}).")
        return

    ACTIVE_MODEL = chosen
    _save_model_state()
    log("MODEL", f"Активная модель: {chosen} ({label})")
    rtag = ""
    if _supports_reasoning(provider):
        rtag = f" · 🤔 ризонинг: `{_clamp_reasoning(_mid, REASONING_EFFORT, provider)}`" if REASONING_EFFORT else " · 🤔 ризонинг: авто (`/model reason`)"
    await event.edit(f"✅ Модель ответов: {label} (окно {_fmt_ctx(ctx)}){rtag}")


def _sync_fish_search(query: str):
    """Поиск голосов Fish Audio: GET /model?title=&sort_by=score. Возвращает список {_id,title,languages}."""
    r = requests.get(FISH_MODELS_URL, headers={"Authorization": f"Bearer {fish_audio_api_key}"},
                     params={"title": query, "sort_by": "score", "page_size": 10}, timeout=30)
    r.raise_for_status()
    return (r.json() or {}).get("items", [])


def _sync_fish_get(reference_id: str):
    """Метаданные одного Fish-голоса по id: GET /model/{id} → {title, languages, ...}."""
    r = requests.get(f"{FISH_MODELS_URL}/{reference_id}",
                     headers={"Authorization": f"Bearer {fish_audio_api_key}"}, timeout=30)
    r.raise_for_status()
    return r.json() or {}


def _fish_ref_from(s: str) -> str:
    """Извлекает reference_id из ссылки fish.audio/m/<id> (или возвращает строку как есть)."""
    s = s.strip()
    if "fish.audio" in s or s.startswith("http"):
        s = s.split("?")[0].split("#")[0].rstrip("/").split("/")[-1]
    return s


async def _voice_fish_command(event, rest: str):
    """Подкоманды Fish: список избранного / search / add / remove / test / выбор."""
    global FISH_VOICE, LAST_FISH_SEARCH  # FISH_FAVORITES/TTS_ENGINE здесь только читаются/мутируются
    if not fish_available:
        await event.edit("⚠️ Fish недоступен: нет `FISH_AUDIO_API_KEY` в .env.")
        return
    low = rest.lower()

    if low.startswith("search"):
        q = rest[len("search"):].strip()
        if not q:
            await event.edit("Использование: `/voice fish search <запрос>` (напр. `russian`, `male`, имя).")
            return
        await event.edit(f"🔎 Ищу голоса Fish по «{q}»…")
        try:
            items = await asyncio.to_thread(_sync_fish_search, q)
        except Exception as e:
            await event.edit(f"⚠️ Ошибка поиска Fish: {str(e)[:120]}")
            return
        if not items:
            await event.edit(f"Ничего не найдено по «{q}».")
            return
        LAST_FISH_SEARCH = items[:10]
        lines = [f"🔎 **Fish — результаты по «{q}»:**"]
        for i, it in enumerate(LAST_FISH_SEARCH, 1):
            langs = ",".join(it.get("languages") or [])
            lines.append(f"{i}. {it.get('title','?')} — `{it.get('_id','')}`" + (f" ({langs})" if langs else ""))
        lines.append("\nВ избранное: `/voice fish add <N>` — по номеру из списка (или `/voice fish add <id> [имя]`).")
        await event.edit("\n".join(lines)[:4000])
        return

    if low.startswith("add"):
        parts = rest[len("add"):].strip().split(maxsplit=1)
        if not parts:
            await event.edit("Использование: `/voice fish add <N>` (номер из поиска), `/voice fish add <ссылка fish.audio>` или `/voice fish add <reference_id> [имя]`.")
            return
        first = parts[0]
        override = parts[1] if len(parts) > 1 else None
        note = ""
        if first.isdigit() and 1 <= int(first) <= len(LAST_FISH_SEARCH):
            item = LAST_FISH_SEARCH[int(first) - 1]
            ref = item.get("_id", "")
            name = override or item.get("title", ref)
        else:
            ref = _fish_ref_from(first)  # из ссылки fish.audio/m/<id> или сырой id
            if override:
                name = override
            else:
                try:
                    name = (await asyncio.to_thread(_sync_fish_get, ref)).get("title") or ref
                except Exception:
                    name = ref
                    note = " (имя с платформы не получено — проверь id/ссылку)"
        if any(f["id"] == ref for f in FISH_FAVORITES):
            await event.edit(f"Голос `{ref}` уже в избранном.")
            return
        FISH_FAVORITES.append({"id": ref, "title": name})
        _save_model_state()
        num = len(FISH_FAVORITES)
        eng = "" if TTS_ENGINE == "fish" else "\n⚠️ Сейчас движок gemini — для озвучки им включи `/voice engine fish`."
        await event.edit(f"✅ Добавлен в избранное Fish: **{name}** (`{ref}`).{note}\n👉 Быстро сделать активным: `/voice fish {num}` · послушать: `/voice fish test`{eng}")
        return

    if low.startswith("remove"):
        marg = rest[len("remove"):].strip()
        idx = None
        if marg.isdigit() and 1 <= int(marg) <= len(FISH_FAVORITES):
            idx = int(marg) - 1
        else:
            idx = next((i for i, f in enumerate(FISH_FAVORITES) if f["id"] == marg), None)
        if idx is None:
            await event.edit(f"Не нашёл в избранном: `{marg}`. `/voice fish` — список.")
            return
        removed = FISH_FAVORITES.pop(idx)
        if FISH_VOICE == removed["id"]:
            FISH_VOICE = None
        _save_model_state()
        await event.edit(f"🗑 Удалён из избранного Fish: **{removed['title']}** (`{removed['id']}`).")
        return

    if low == "test" or low.startswith("test"):
        sample = rest[len("test"):].strip() or "Привет! Так звучит выбранный голос."
        if not FISH_VOICE:
            await event.edit("Сначала выбери Fish-голос: `/voice fish <N|id>` (см. `/voice fish`).")
            return
        await event.edit(f"🎙 Синтезирую пример Fish-голосом `{FISH_VOICE}`…")
        ogg = await synthesize_voice(sample, ACTIVE_VOICE, engine="fish")
        if ogg:
            bio = io.BytesIO(ogg); bio.name = "voice.ogg"
            await client.send_file(event.chat_id, bio, voice_note=True)
            await event.delete()
        else:
            await event.edit("🔇 Не удалось синтезировать (проверь id голоса/ключ Fish).")
        return

    if not rest:
        # список избранного, сгруппированный по категориям (cat); номер = позиция в FISH_FAVORITES
        lines = [f"🐟 **Fish Audio — избранные голоса** (движок сейчас: {TTS_ENGINE}):"]
        if not FISH_FAVORITES:
            lines.append("  (пусто) — найди через `/voice fish search <запрос>` и добавь `/voice fish add <N>`.")
        groups, order = {}, []
        for i, f in enumerate(FISH_FAVORITES, 1):
            cat = f.get("cat") or "📦 Разное"
            if cat not in groups:
                groups[cat] = []; order.append(cat)
            groups[cat].append((i, f))
        for cat in order:
            lines.append(f"\n**{cat}:**")
            for i, f in groups[cat]:
                mk = "▶" if f["id"] == FISH_VOICE else " "
                lines.append(f"{mk}{i}. {f['title']}")
        lines.append("\n`/voice fish search <q>` — найти · `/voice fish add <N>` · `/voice fish remove <N>`")
        lines.append("`/voice fish <N|id>` — выбрать · `/voice fish test` — прослушать · `/voice engine fish` — включить движок")
        await event.edit("\n".join(lines)[:4000])
        return

    # выбор активного Fish-голоса: по номеру из избранного или прямой reference_id
    chosen = None
    if rest.isdigit() and 1 <= int(rest) <= len(FISH_FAVORITES):
        chosen = FISH_FAVORITES[int(rest) - 1]["id"]
    else:
        chosen = next((f["id"] for f in FISH_FAVORITES if f["id"] == rest), rest)  # прямой id принимаем как есть
    FISH_VOICE = chosen
    _save_model_state()
    hint = "" if TTS_ENGINE == "fish" else "\n⚠️ Сейчас движок gemini — включи `/voice engine fish`, чтобы озвучивать этим голосом."
    await event.edit(f"✅ Активный Fish-голос: `{FISH_VOICE}`.{hint}")


@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]voice(?:\s+(.+))?$", from_users="me"))
async def voice_command(event):
    """Выбор голоса и режима озвучки для голосовых ответов в /ask (Gemini + Fish Audio)."""
    if await _slash_for_other_bot(event):
        return
    global ACTIVE_VOICE, VOICE_AUTO, TTS_ENGINE  # FISH_* меняются в _voice_fish_command
    arg = (event.pattern_match.group(1) or "").strip()

    if not tts_available and not fish_available:
        await event.edit("⚠️ Голос недоступен: нет ни `GOOGLE_GENAI_API_KEY`, ни `FISH_AUDIO_API_KEY` в .env.")
        return

    low = arg.lower()

    # /voice engine [gemini|fish] — выбор TTS-движка
    if low.startswith("engine"):
        rest = arg[len("engine"):].strip().lower()
        if rest in ("gemini", "fish"):
            if rest == "fish" and not fish_available:
                await event.edit("⚠️ Fish недоступен: нет `FISH_AUDIO_API_KEY` в .env.")
                return
            TTS_ENGINE = rest
            _save_model_state()
        await event.edit(f"🔧 TTS-движок: **{TTS_ENGINE}**" +
                         (f" · активный голос Fish: `{FISH_VOICE}`" if TTS_ENGINE == "fish" and FISH_VOICE else "") +
                         "\n`/voice engine gemini|fish` — сменить · при сбое движка — автофолбэк на другой")
        return

    # /voice fish ... — Fish Audio: список/поиск/добавление/выбор избранных голосов
    if low == "fish" or low.startswith("fish"):
        await _voice_fish_command(event, arg[len("fish"):].strip())
        return

    # /voice auto on|off — переключатель авто-голоса
    if low.startswith("auto"):
        rest = arg[len("auto"):].strip().lower()
        if rest in ("on", "вкл", "1", "true"):
            VOICE_AUTO = True
        elif rest in ("off", "выкл", "0", "false"):
            VOICE_AUTO = False
        else:
            VOICE_AUTO = not VOICE_AUTO  # тоггл, если без аргумента
        _save_model_state()
        await event.edit(f"🔁 Авто-голос: {'ВКЛ ✅' if VOICE_AUTO else 'выкл'}\n(модель {'может сама' if VOICE_AUTO else 'не будет'} отвечать голосом; флаг `-v` форсит всегда)")
        return

    # /voice samples [N|имя] — прислать озвученные примеры (все 30 или один)
    if low == "samples" or low.startswith("samples"):
        rest = arg[len("samples"):].strip()
        if rest:  # один конкретный голос
            prof = (VOICE_PROFILES[int(rest) - 1] if rest.isdigit() and 1 <= int(rest) <= len(VOICE_PROFILES)
                    else _voice_profile(rest))
            if not prof:
                await event.edit(f"Нет такого голоса: `{rest}`.")
                return
            targets = [(VOICE_PROFILES.index(prof) + 1, prof)]
        else:
            targets = list(enumerate(VOICE_PROFILES, 1))
        total = len(targets)
        await event.edit(f"🎙 Готовлю примеры голосов ({total})… первый раз дольше (озвучиваю и кэширую), потом — мгновенно.")
        sent = fail = 0
        for idx, p in targets:
            ogg = await _ensure_voice_sample(p["name"])
            if ogg:
                bio = io.BytesIO(ogg)
                bio.name = "voice.ogg"
                g = "♀" if p["gender"] == "female" else "♂"
                mark = "▶ " if p["name"] == ACTIVE_VOICE else ""
                await client.send_file(event.chat_id, bio, voice_note=True,
                                       caption=f"{mark}{idx}. {p['emoji']} {p['name']} {g} — {p['personality']}")
                sent += 1
                await asyncio.sleep(0.6)  # против FloodWait при пачке
            else:
                fail += 1
        await event.edit(f"✅ Примеры голосов: отправлено {sent}/{total}" + (f", не удалось {fail} (лимит/ошибка)" if fail else "") +
                         "\nВыбрать: `/voice N` или `/voice <имя>`.")
        return

    # /voice test [текст] — синтез примера текущим голосом
    if low == "test" or low.startswith("test "):
        sample = arg[len("test"):].strip() or "Привет! Так звучит мой голос. [с теплотой] Рад, что ты меня слышишь."
        await event.edit(f"🎙 Синтезирую пример голосом {ACTIVE_VOICE}…")
        ogg = await synthesize_voice(sample, ACTIVE_VOICE, engine="gemini")
        if ogg:
            bio = io.BytesIO(ogg)
            bio.name = "voice.ogg"
            await client.send_file(event.chat_id, bio, voice_note=True)
            await event.delete()
        else:
            await event.edit("🔇 Не удалось синтезировать пример (проверь ключ/лимиты Google).")
        return

    # без аргумента — список голосов
    if not arg:
        lines = ["🎙 **Голоса (Gemini TTS)** — ▶ активный:"]
        for i, p in enumerate(VOICE_PROFILES, 1):
            mk = f"▶{i}." if p["name"] == ACTIVE_VOICE else f"{i}."
            g = "♀" if p["gender"] == "female" else "♂"
            lines.append(f"{mk} {p['emoji']} `{p['name']}` {g} — {p['personality']}")
        lines.append(f"\nДвижок: **{TTS_ENGINE}**" + (f" (Fish-голос `{FISH_VOICE}`)" if TTS_ENGINE == "fish" and FISH_VOICE else "") +
                     f" · Авто-голос: {'ВКЛ ✅' if VOICE_AUTO else 'выкл'} · флаг `-v` форсит голос")
        lines.append("🎧 `/voice samples` — примеры ВСЕХ голосов · `/voice N`/`/voice <имя>` — выбрать · `/voice auto on|off`")
        lines.append("🐟 `/voice engine fish|gemini` — сменить движок · `/voice fish` — голоса Fish Audio (поиск/избранное)")
        await event.edit("\n".join(lines)[:4000])
        return

    # выбор по номеру или имени
    chosen = None
    if arg.isdigit():
        idx = int(arg) - 1
        if 0 <= idx < len(VOICE_PROFILES):
            chosen = VOICE_PROFILES[idx]["name"]
    else:
        p = _voice_profile(arg)
        if p:
            chosen = p["name"]
    if not chosen:
        await event.edit(f"Нет такого голоса: `{arg}`. `/voice` — список (номер или имя).")
        return
    ACTIVE_VOICE = chosen
    _save_model_state()
    prof = _voice_profile(chosen)
    log("TTS", f"Активный голос: {chosen}")
    await event.edit(f"✅ Голос: {prof['emoji']} **{chosen}** — {prof['personality']}\n`/voice test` — прослушать · `-v` в /ask — ответить голосом")


def _human_bytes(n):
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ТБ"


@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]cache(?:\s+(\w+))?(?:\s+(\S+))?$", from_users="me"))
async def cache_command(event):
    if await _slash_for_other_bot(event):
        return
    sub = (event.pattern_match.group(1) or "info").lower()
    arg = (event.pattern_match.group(2) or "").strip()

    if sub == "info":
        n_total = len(MEDIA_CACHE)
        n_ts = len(MEDIA_CACHE_TS)
        n_no_ts = n_total - n_ts
        try:
            sz = os.path.getsize(MEDIA_CACHE_PATH) if os.path.exists(MEDIA_CACHE_PATH) else 0
        except Exception:
            sz = 0
        lines = [
            "📦 **Медиа-кэш**",
            f"• Записей: {n_total} / {MEDIA_CACHE_MAX}",
            f"• Файл: `{MEDIA_CACHE_PATH}` — {_human_bytes(sz)}",
            f"• С датой: {n_ts} · без даты (старые): {n_no_ts}",
        ]
        if n_ts:
            ts_values = list(MEDIA_CACHE_TS.values())
            oldest = datetime.fromtimestamp(min(ts_values), MSK).strftime("%Y-%m-%d %H:%M")
            newest = datetime.fromtimestamp(max(ts_values), MSK).strftime("%Y-%m-%d %H:%M")
            lines.append(f"• Самая старая (с TS): {oldest}")
            lines.append(f"• Самая новая (с TS): {newest}")
        lines.append("")
        lines.append("Очистить: `/cache clear all` · `/cache clear older 30` (дней)")
        await event.edit("\n".join(lines))
        return

    if sub == "clear":
        if arg == "all":
            n = len(MEDIA_CACHE)
            MEDIA_CACHE.clear()
            MEDIA_CACHE_TS.clear()
            save_json(MEDIA_CACHE_PATH, {})
            save_json(MEDIA_CACHE_TS_PATH, {})
            log("CACHE", f"clear all: удалено {n} записей")
            await event.edit(f"🗑 Удалено {n} записей (всё).")
            return
        if arg == "older":
            await event.edit("Укажи число дней: `/cache clear older 30`")
            return
        await event.edit("`/cache clear all` или `/cache clear older 30`")
        return

    await event.edit("`/cache info` · `/cache clear all|older N`")


# Отдельный обработчик для `/cache clear older N` (3 аргумента, регулярка с 2 группами не покрывает)
@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]cache\s+clear\s+older\s+(\d+)$", from_users="me"))
async def cache_clear_older_command(event):
    if await _slash_for_other_bot(event):
        return
    days = int(event.pattern_match.group(1))
    cutoff = time.time() - days * 86400
    to_remove = [k for k, ts in MEDIA_CACHE_TS.items() if ts < cutoff]
    for k in to_remove:
        MEDIA_CACHE.pop(k, None)
        MEDIA_CACHE_TS.pop(k, None)
    save_json(MEDIA_CACHE_PATH, dict(MEDIA_CACHE))
    save_json(MEDIA_CACHE_TS_PATH, dict(MEDIA_CACHE_TS))
    log("CACHE", f"clear older {days}: удалено {len(to_remove)} записей")
    await event.edit(f"🗑 Удалено {len(to_remove)} записей старше {days} дней (без TS — не тронуты).")


def _fmt_allow_limit(limit):
    if limit is None:
        return f"дефолт ({ALLOWED_ASK_TEXT_LIMIT})"
    if limit == -1:
        return "без лимита"
    return str(limit)


@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]allow(?:\s+(.+))?$", from_users="me"))
async def allow_command(event):
    if await _slash_for_other_bot(event):
        return
    arg = (event.pattern_match.group(1) or "").strip()

    # список
    if not arg and not getattr(event, "reply_to", None):
        if not ALLOWED_USERS:
            await event.edit("Доступ к /ask ни у кого. Выдать: `/allow @username [лимит]` или ответом на сообщение.")
            return
        lines = ["✅ Доступ к /ask есть у:"]
        for i, (uid, rec) in enumerate(ALLOWED_USERS.items(), 1):
            uname = rec.get("username") if isinstance(rec, dict) else rec
            limit = rec.get("limit") if isinstance(rec, dict) else None
            who = ('@' + uname) if uname else str(uid)
            lines.append(f"{i}. {who} (id {uid}) · лимит: {_fmt_allow_limit(limit)}")
        lines.append("\nПри N > лимита — vision переключается на free-модель (текст остаётся).")
        lines.append("`/allow @name <N|unlimited>` — задать лимит · `/allow remove @name|<id>`")
        await event.edit("\n".join(lines))
        return

    remove = False
    if arg.lower().startswith("remove"):
        remove = True
        arg = arg[len("remove"):].strip()

    # Извлечь хвост-лимит из arg (если grant, не remove)
    desired_limit = None  # None означает «не указано»
    explicit_limit = False
    if not remove:
        toks = arg.split()
        if toks and toks[-1].lower() == "unlimited":
            desired_limit = -1
            explicit_limit = True
            arg = " ".join(toks[:-1]).strip()
        elif len(toks) >= 2 and toks[-1].lstrip("-").isdigit():
            # Лимит — только если ПЕРЕД ним есть цель (иначе чистый id трактуется как цель)
            desired_limit = int(toks[-1])
            explicit_limit = True
            arg = " ".join(toks[:-1]).strip()

    # удаление по сырому id (без резолва)
    if remove and arg.lstrip("-").isdigit():
        uid = int(arg)
        gone = ALLOWED_USERS.pop(uid, None)
        if gone is None:
            await event.edit("Этого id нет в списке.")
        else:
            _save_allowed()
            await event.edit(f"🚫 Доступ забран: id {uid}")
        return

    # определяем цель: ответ на сообщение или @username/id
    target = None
    if not arg and getattr(event, "reply_to", None):
        rep = await event.get_reply_message()
        target = await rep.get_sender() if rep else None
    elif arg:
        try:
            ref = int(arg) if arg.lstrip("-").isdigit() else arg.lstrip("@")
            target = await client.get_entity(ref)
        except Exception as e:
            await event.edit(f"Не нашёл пользователя {arg}: {e}")
            return
    if target is None:
        await event.edit("Укажи @username/id или ответь на сообщение пользователя.")
        return

    uid = target.id
    uname = getattr(target, "username", None)
    if remove:
        if ALLOWED_USERS.pop(uid, None) is not None:
            _save_allowed()
            await event.edit(f"🚫 Доступ забран: {('@' + uname) if uname else uid}")
        else:
            await event.edit("Этого пользователя нет в списке.")
    else:
        existing = ALLOWED_USERS.get(uid) or {}
        new_limit = desired_limit if explicit_limit else existing.get("limit")
        ALLOWED_USERS[uid] = {"username": uname, "limit": new_limit}
        _save_allowed()
        log("ALLOW", f"Доступ к /ask выдан {uid} (@{uname}), лимит={new_limit}")
        await event.edit(f"✅ Доступ к /ask выдан: {('@' + uname) if uname else uid} · лимит: {_fmt_allow_limit(new_limit)}")


def _help_index(active_label):
    return (
        "╭───────────────────────╮\n"
        "│   🤖  КОМАНДЫ БОТА   │\n"
        "╰───────────────────────╯\n"
        "\n"
        "Это «оглавление». Каждый раздел можно открыть подробно — допиши его\n"
        "название после `/help`. Пример: `/help media`.\n"
        "Команды работают и через `/`, и через `.` (например `.help`).\n"
        "❗ В личке с ботами `/команды` юзербот игнорирует (они адресованы боту) —\n"
        "   там используй вариант с точкой: `.ask`, `.model`, …\n"
        "\n"
        "📂 **Разделы справки** (`/help <раздел>`):\n"
        "   `ask`       💬 вопросы к AI по чату — главная функция\n"
        "   `model`     🧠 выбор модели для текстовых ответов\n"
        "   `media`     🖼 vision-модели (картинки/видео-кружки) + метки [OR]/[OC]\n"
        "   `voice`     🎙 голосовые ответы: выбор голоса, флаг `-v`, эмоции\n"
        "   `gen`       🎨 генерация и редактирование изображений\n"
        "   `index`     🗂 память по истории чата (GraphRAG): досье, граф, поиск\n"
        "   `keys`      🔑 какие API-ключи за что отвечают (что обязательно)\n"
        "   `channels`  📡 каналы, поиск, дайджест\n"
        "   `auto`      🔁 авто-ответ\n"
        "   `allow`     👥 доступ к `/ask` для других\n"
        "   `status`    📊 все текущие настройки разом (`/status`)\n"
        "   `song`      🎵 печать с эффектом набора\n"
        "   `help`      ℹ️ как устроена сама эта команда\n"
        "   `all`       📖 показать ВСЁ сразу\n"
        "\n"
        "⚡ **Шпаргалка (самое частое):**\n"
        "   `/ask 200 о чём спорят?` — ответ по последним 200 сообщениям\n"
        "   `/ask 50 -t коротко` — без медиа (быстрее)\n"
        "   `/model` — сменить модель ответов · `/model media` — сменить «глаза»\n"
        "\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ Активная модель: **{active_label}**\n"
        f"💡 Не уверен с чего начать? Набери `/help ask`."
    )


_HELP_SECTIONS = {
    "ask": (
        "💬 **`/ask` — вопрос к AI по истории чата**\n"
        "\n"
        "Бот читает последние сообщения этого чата и отвечает на твой вопрос с опорой на них.\n"
        "Длинные ответы приходят **свёрнутой цитатой** — занимают 3 строки, тап раскрывает целиком.\n"
        "Код-блоки при этом остаются снаружи цитаты с подсветкой (Telegram не прячет код в цитату).\n"
        "\n"
        "📐 **СИНТАКСИС (порядок строго слева направо):**\n"
        "```\n"
        "/ask  N  [флаги]  [@юзеры]  вопрос\n"
        "  1   2     3         4        5\n"
        "```\n"
        "1️⃣ `/ask` — сама команда.\n"
        "2️⃣ `N` — **обязательно**, число: сколько последних сообщений взять (напр. `200`).\n"
        "3️⃣ `[флаги]` — необязательно: `-t`, `-c`, `-d`, `-v`, `-g` (см. ниже).\n"
        "4️⃣ `[@юзеры]` — необязательно: `@имя` (только эти) или `!@имя` (исключить); `@me`/`!@me` — ты сам.\n"
        "5️⃣ `вопрос` — **обязательно**, любой текст до конца строки.\n"
        "\n"
        "⚠️ **Порядок важен!** Флаги — ВСЕГДА перед `@юзерами`, оба — перед вопросом.\n"
        "   ✅ `/ask 500 -t @anna о чём писала?`\n"
        "   ❌ `/ask 500 @anna -t о чём писала?`  ← тут `-t` уедет в текст вопроса и не сработает.\n"
        "\n"
        "**Минимум:**\n"
        "   `/ask N вопрос`\n"
        "   _Пример:_ `/ask 300 сделай выжимку спора про цены`\n"
        "   Чем больше N — тем больше контекста, но дольше сбор и больше токенов.\n"
        "\n"
        "**Флаги** (шаг 3; можно несколько, слитно `-tc` или раздельно `-t -c`):\n"
        "   `-t` — текст без медиа: не распознаёт фото/голос/кружки → **быстрее и дешевле**.\n"
        "   `-c` — обязательно искать (по каналам; без каналов — в интернете) перед ответом.\n"
        "   `-d` — дамп: выгрузить собранный контекст отдельным файлом (для отладки).\n"
        "   `-v` — ответить **голосом** (озвучка через Gemini TTS). См. `/help voice`.\n"
        "   `-g` — отдать **картинки напрямую** отвечающей модели (её vision), а не описания.\n"
        "        Нужна vision-модель (`/model` → Qwen/Kimi/MiMo Omni или OpenRouter-vision), иначе понятная ошибка.\n"
        "        ⚠️ GLM-5/5.1 у этого провайдера — текстовые, картинки не принимают (для `-g` не годятся).\n"
        "        Голос/аудио всегда через Parakeet (STT). До 20 свежих картинок за запрос.\n"
        "   `-m` — фото описывает **vision-модель** (полное описание, как раньше; см. `/model media`).\n"
        "        Без `-m` фото идут через дешёвый **OCR** (LlamaParse): берётся только ТЕКСТ\n"
        "        с картинки; фото без текста — пометка [Фото (без текста)]. Голос — без изменений.\n"
        "   _Пример:_ `/ask 1000 -t -d что обсуждали вчера?` · `/ask 30 -v расскажи анекдот` · `/ask 50 -m что на фото?`\n"
        "\n"
        "**Фильтры по людям** (шаг 4):\n"
        "   `@user1 @user2` — взять сообщения **только** этих авторов.\n"
        "   `!@user` — **исключить** этого автора.\n"
        "   Комбинируется: `/ask 500 -t @anna !@bot о чём писала Аня?`\n"
        "\n"
        "**Ответом на сообщение (reply):**\n"
        "   Ответь `/ask вопрос` на чьё-то сообщение — бот возьмёт именно его + контекст вокруг.\n"
        "\n"
        "⏱ На больших N (10–15 тыс.) сбор истории идёт **в несколько потоков** — это норм, подожди.\n"
        "↩️ ИИ может САМ ответить **реплаем** на конкретные сообщения из истории (на одно или сразу\n"
        "   на несколько, до 10) — например адресно на спор или на вопросы разных людей. Решает сам;\n"
        "   после реплаев идёт общий ответ. Работает на моделях с поддержкой инструментов.\n"
        "🌐 С ключом **Tavily** модель сама ходит в интернет: ищет (web_search), читает страницы\n"
        "   по ссылкам (web_extract), обходит сайты (web_crawl/web_map). Когда искать — решает сама;\n"
        "   `-c` заставляет искать обязательно.\n"
        "🔑 Работает на ключе **DeepSeek** (обязательный). Медиа в вопросе требует ключ OpenRouter/OpenCode — см. `/help keys`."
    ),
    "model": (
        "🧠 **Модель для ТЕКСТОВЫХ ответов** (`/model`)\n"
        "\n"
        "Это «мозг», который пишет ответ в `/ask`/`/search`/дайджестах.\n"
        "\n"
        "   `/model` — показать список моделей; стрелкой `▶` отмечена активная.\n"
        "   `/model N` — выбрать модель по номеру из списка.\n"
        "   `/model <slug>` — выбрать по короткому имени.\n"
        "   `/model vendor/model` — поставить **любую** модель OpenRouter по её полному ID\n"
        "        (со слешем). Бот сперва проверит, что такая модель существует.\n"
        "        _Пример:_ `/model anthropic/claude-3.5-sonnet`\n"
        "\n"
        "   `/model probe` — прогнать модели и проверить, у каких работает веб-поиск.\n"
        "\n"
        "**Глубина размышлений (OpenAI GPT-5.x / o3 и Google Gemini):**\n"
        "   `/model reason` — текущий уровень и список (xhigh/high/medium/low/none/auto).\n"
        "   `/model reason high` — установить уровень (глобально, переживает рестарт).\n"
        "   `/model N.M` — выбрать модель N сразу с силой ризонинга M (M=1 — мощнейший).\n"
        "        _Пример:_ `/model 21.1` — на максимуме. o3: none/xhigh → low/high; o4-mini none → low;\n"
        "        Gemini → thinkingLevel (none→minimal, xhigh→high).\n"
        "\n"
        "**Избранное OpenRouter-моделей:**\n"
        "   `/model fav` — список добавленных кастомных моделей (быстрый выбор по номеру).\n"
        "   `/model remove <N|id>` — удалить кастомную модель из избранного.\n"
        "\n"
        "**Медиа-кэш** (распознанные картинки/голос хранятся, чтобы не платить дважды):\n"
        "   `/cache info` — сколько занято.\n"
        "   `/cache clear all` — очистить весь кэш.\n"
        "   `/cache clear older N` — удалить записи старше N дней.\n"
        "\n"
        "🔑 По умолчанию активен **DeepSeek** (обязательный ключ). Модели OpenRouter\n"
        "    доступны только если вписан `OPENROUTER_API_KEY` — см. `/help keys`.\n"
        "🖼 За распознавание картинок отвечает ОТДЕЛЬНАЯ модель — `/help media`."
    ),
    "media": (
        "🖼 **Медиа-модели (vision)** — `/model media`\n"
        "\n"
        "Это «глаза» бота: модель, которая разбирает **картинки** внутри `/ask`\n"
        "и описывает **референсы** для `/gen -i` (DeepSeek сам картинки не видит).\n"
        "Это НЕ та же модель, что пишет текст (её меняет `/model`).\n"
        "\n"
        "💡 **По умолчанию фото в `/ask` идут через OCR** (LlamaParse, cost-effective):\n"
        "   с картинки берётся только текст — дёшево. Vision-модель из этого списка\n"
        "   работает при флаге `-m`, в `/gen -i` и как фолбэк при сбое OCR.\n"
        "\n"
        "   `/model media` — показать список vision-моделей; `▶` — активная.\n"
        "   `/model media N` — выбрать по номеру.\n"
        "   `/model media <slug>` — выбрать по короткому имени (напр. `mimo-v2-omni`).\n"
        "   `/model media <vendor/model>` — любая модель OpenRouter по ID (с проверкой, что она умеет vision).\n"
        "\n"
        "**Метки провайдера в списке:**\n"
        "   `[OR]` = OpenRouter  → нужен ключ `OPENROUTER_API_KEY`\n"
        "   `[OC]` = OpenCode Go → нужен ключ `OPENCODE_API_KEY`\n"
        "\n"
        "**Как понять, что доступно:**\n"
        "   • Рядом со строкой стоит `⚠️нет ключа` → у этого провайдера НЕ вписан ключ, выбрать нельзя.\n"
        "   • Нет пометки → модель доступна, бери любую.\n"
        "   • Выберешь без ключа — бот не сломается, просто откажет и оставит прежнюю.\n"
        "\n"
        "🎙 **Аудио и голосовые** распознаются ВСЕГДА отдельной STT-моделью **NVIDIA Parakeet** —\n"
        "    этот список на голос не влияет.\n"
        "⚡ Хочешь быстрее/дешевле — флаг `-t` в `/ask` вообще пропускает медиа."
    ),
    "voice": (
        "🎙 **Голосовые ответы** (`/voice` + флаг `-v`)\n"
        "\n"
        "Бот может отвечать на `/ask` не текстом, а **живым голосовым** — через Google\n"
        "Gemini Flash TTS. Голос выбираешь ты; озвучка эмоциональная (с интонацией).\n"
        "\n"
        "**Выбор голоса:**\n"
        "   `/voice` — список 30 голосов; `▶` — активный.\n"
        "   `/voice N` — выбрать по номеру.\n"
        "   `/voice <имя>` — выбрать по имени (напр. `/voice Kore`).\n"
        "   `/voice samples` — прислать озвученные примеры ВСЕХ голосов (послушать и выбрать).\n"
        "   `/voice samples N` — пример одного голоса; `/voice test [текст]` — пример текущим голосом.\n"
        "\n"
        "**Когда бот отвечает голосом — два способа:**\n"
        "   1) Флаг `-v` в `/ask` — **форсит голос всегда**: `/ask 30 -v расскажи анекдот`.\n"
        "   2) Авто-режим — `/voice auto on`: модель сама решает, где голос уместнее\n"
        "      (эмоция, короткий личный ответ). `/voice auto off` — выключить.\n"
        "\n"
        "**Эмоции:** модель управляет интонацией аудио-тегами в тексте —\n"
        "   `[смеётся]`, `[шёпотом]`, `[взволнованно]`, `[с теплотой]`, `[серьёзно]`,\n"
        "   `[вздыхает]`, паузы — многоточием. Теги не произносятся, а задают подачу.\n"
        "\n"
        "**Движки TTS (Gemini / Fish Audio):**\n"
        "   `/voice engine fish|gemini` — выбрать движок (при сбое — автофолбэк на другой).\n"
        "   `/voice fish search <запрос>` — найти голоса Fish (пронумерованный список: № + название + id + языки).\n"
        "   `/voice fish add <N>` — добавить в избранное результат поиска по номеру (имя и id подставятся сами);\n"
        "      либо `/voice fish add <ссылка fish.audio/m/...>` — имя подтянется с платформы; либо `/voice fish add <id> [имя]` вручную.\n"
        "      После добавления бот подскажет номер для быстрого выбора (`/voice fish <N>`). `/voice fish` — список избранного.\n"
        "   `/voice fish <N|id>` — выбрать голос (номер из избранного ИЛИ прямой id).\n"
        "   `/voice fish remove <N|id>` — убрать из избранного; `/voice fish test [текст]` — прослушать.\n"
        "   Голоса берутся с fish.audio (id = reference_id). Нужен `FISH_AUDIO_API_KEY`.\n"
        "   🎭 Разметку интонации модель ставит САМА под движок: Gemini — `[теги]` (рус.), Fish s2-pro —\n"
        "      `[english]` описания подачи (`[soft]`,`[whispering]`,`[laughing]`), Fish s1 — `(round)` из набора.\n"
        "      Текст реплики при этом на русском; теги не произносятся. Тебе делать ничего не нужно.\n"
        "\n"
        "ℹ️ Голосовой ответ — до ~5000 симв. (несколько минут речи) и идёт **только голосом**; если\n"
        "   TTS не сработал — бот автоматически пришлёт текст.\n"
        "🔑 Нужен ключ `GOOGLE_GENAI_API_KEY` (см. `/help keys`). Без него `/voice`\n"
        "   сообщит, что голос недоступен, а `/ask` будет отвечать текстом.\n"
        "♻️ Если Google-квота исчерпана/недоступна — бот автоматически озвучит через\n"
        "   OpenRouter (та же модель, нужен `OPENROUTER_API_KEY`)."
    ),
    "gen": (
        "🎨 **`/gen` — генерация и редактирование изображений**\n"
        "\n"
        "Модель-генератор: GPT Image 2 (OpenRouter, платная), при сбое/перегрузке — запасная Gemini 3.1 Flash Image. Нужен `OPENROUTER_API_KEY`.\n"
        "Промпт строит **активная модель-ответчик** (`/model`); если она с vision — сама смотрит картинки чата, если текстовая — по их описаниям (медиа-модель `/media`). DeepSeek — фолбэк.\n"
        "\n"
        "**Синтаксис:** `/gen [N] [-i|-c|-r] [-ni|-m] [-v|-h|-sq] [-2k|-4k|-1k] [-xK] [@юзер|!@юзер] <промпт>`\n"
        "   `/gen аниме кот в очках` — креатив: модель развернёт запрос в промпт вокруг одной идеи\n"
        "   `/gen -i закат над морем` — ТОЧНАЯ переформулировка: модель лишь сделает твой промпт качественным,\n"
        "     ничего своего не добавляя (`-i` или `-improve`)\n"
        "   `/gen 100 нарисуй о чём мы спорим` — модель составит промпт по последним 100 сообщениям чата\n"
        "\n"
        "**🖼 Картинки из истории как референсы (по умолчанию для `/gen N`):** модель видит фото в окне N,\n"
        "   САМА выбирает подходящие как референсы (особенно ФОТО ПЕРСОНАЖЕЙ чата — для узнаваемости лиц) и в\n"
        "   промпте говорит, что с ними делать (взять лицо/персонажа, стиль, фон, объединить). В генератор уходит\n"
        "   до 16 выбранных (потолок GPT Image 2).\n"
        "   • **vision-модель** смотрит фото НАПРЯМУЮ — до 20 свежих;\n"
        "   • **текстовая модель** или флаг **`-m`** — по ОПИСАНИЯМ (медиа-модель), но больший пул — до 300 фото\n"
        "     (больше выбор; первый прогон дольше — описания кэшируются);\n"
        "   • **`-ni`** (синоним `-noimg`) — вообще не брать картинки из истории (только текст; это НЕ `-i`).\n"
        "   🧹 Мусор в референсы не идёт: превью ссылок (обложки YouTube и т.п.) отсекаются сразу, скриншоты\n"
        "   переписок/интерфейсов и мемы — по авто-разметке типов. Нужен скриншот — приложи его или реплайни.\n"
        "   После генерации, если ИИ взял фото из истории, под картинкой придёт строка **«📎 Референсы»**\n"
        "   со ссылками на сообщения-источники и пометкой, ЗАЧЕМ взят каждый (лицо/стиль/фон).\n"
        "\n"
        "**Креативный режим — ПО УМОЛЧАНИЮ (флаг `-c` указывать не нужно):**\n"
        "   Активная модель сама СОЧИНЯЕТ промпт со своим художественным видением — но всегда вокруг ОДНОЙ\n"
        "   ясной идеи (она приходит строкой 💡 в подписи к картинке), а не грудой случайных деталей.\n"
        "   Так работает любой `/gen` БЕЗ приложенного фото. `-c` остаётся как явный синоним.\n"
        "   ИСКЛЮЧЕНИЕ — когда приложил/реплайнул ФОТО на правку: тогда режим РЕДАКТИРОВАНИЯ (точно, без отсебятины);\n"
        "   добавь `-c` к фото, если хочешь творческую переработку, или `-i` — аккуратное уточнение твоего промпта.\n"
        "   `/gen 200 что хочет нарисовать чат?` · `/gen аниме кот` — оба уже креативные.\n"
        "   **`-i` (improve)** — идея целиком ТВОЯ: модель только переформулирует запрос в сильный визуальный\n"
        "   английский промпт, без своих идей и объектов. **`-r` (raw)** — вообще БЕЗ ИИ: промпт уходит в генератор\n"
        "   ДОСЛОВНО (ни расширения, ни истории-картинок). `/gen -r a cat in a hat, watercolor`.\n"
        "\n"
        "**Ориентация (точное соотношение сторон):** `-v` вертикаль 9:16 · `-h` горизонталь 16:9 · `-sq` квадрат 1:1\n"
        "   Без флага ИИ-промптер САМ выбирает ориентацию под идею (вертикаль для персонажа в рост,\n"
        "   горизонталь для пейзажа и т.п.); твой флаг всегда важнее.\n"
        "   `/gen -v аниме девушка у окна` · `/gen -h пейзаж гор` · комбинируется: `/gen -c -v <ссылка> …`\n"
        "\n"
        "**Качество (разрешение):** по умолчанию **2K** (2048²). `-4k` максимум (медленнее, дороже), `-1k` быстрее/мельче.\n"
        "   `/gen -4k постер с текстом` — чёткий мелкий текст · `/gen -1k черновик` — быстро.\n"
        "   ⚠️ Telegram пережимает фото при отправке — для пиксель-в-пиксель оригинала это не панацея,\n"
        "   но 2K/4K заметно чётче дефолтного 1K.\n"
        "\n"
        "**Пакет `-xK` — много вариантов в Избранное:** `-x8` → 8 вариантов уйдут тебе в **Saved Messages**\n"
        "   (не в текущий чат — там только прогресс), макс. 20. Активная модель пишет КАЖДОМУ варианту свой\n"
        "   промпт, ВИДЯ все предыдущие → сама придумывает непохожие (без навязанных шаблонов), все уникальны.\n"
        "   (Картинки-референсы из истории выбираются один раз и общие для всех вариантов пакета.)\n"
        "   `/gen -x10 -4k аниме кот` · `/gen 200 -c -x8 что нарисовать?` (только для владельца).\n"
        "\n"
        "**Фильтр авторов контекста** (как в `/ask`, работает при числе N): `!@юзер` — ИСКЛЮЧИТЬ его сообщения\n"
        "   из контекста, `@юзер` — взять ТОЛЬКО его. Ставится перед промптом, можно несколько.\n"
        "   `@me` / `!@me` — быстрый шорткат на тебя самого (взять только свои / исключить свои сообщения).\n"
        "   `/gen 2000 -x20 !@spambot !@flood арты по чату` — соберёт 2000 сообщений без этих авторов.\n"
        "\n"
        "**Референс-изображения (image-to-image):**\n"
        "   • прикрепи **фото прямо к сообщению** с `/gen` (можно альбомом) — они уйдут модели на вход;\n"
        "   • reply на **фото** + `/gen сделай фон ночным` → редактирование этой картинки\n"
        "     (промпт идёт дословно; добавь `-i` — модель уточнит формулировку, «увидев» референс,\n"
        "     и ничего не добавит от себя — меняется только то, что просишь);\n"
        "   • reply на **текстовое** сообщение — его текст идёт в контекст, промпт строит активная модель;\n"
        "   • **ссылки на сообщения** прямо в промпте — фото из них уйдут на вход (для нескольких\n"
        "     референсов из РАЗНЫХ сообщений за одну команду): на каждом фото «Скопировать ссылку»,\n"
        "     вставь в `/gen`; ссылки вырезаются из текста — модель видит чистый промпт. Пример:\n"
        "     `/gen https://t.me/c/123/45 https://t.me/c/123/60 нарисуй их в одной сцене`\n"
        "     (берётся ровно указанное фото, альбом НЕ подтягивается);\n"
        "   • можно совместить: своё фото + reply + ссылки — все референсы объединяются.\n"
        "   ⚠️ До 10 фото, суммарно до 3 МБ (лимит API).\n"
        "\n"
        "Если промпт составлял/улучшал DeepSeek — он приходит **целиком** (без обрезки) свёрнутой цитатой:\n"
        "   в подписи к картинке, а если длинный — отдельным сообщением-реплаем на неё.\n"
        "👁 Референсы для DeepSeek описывает активная медиа-модель (`/model media`) — с кэшем.\n"
        "✂️ С референсами DeepSeek только уточняет формулировку (ничего не добавляет от себя);\n"
        "   без референсов — творческий детальный промпт.\n"
        "♻️ При временном сбое провайдера — авто-повтор; если провайдер отклонил AI-промпт —\n"
        "   DeepSeek сам поправит формулировки и попробует снова.\n"
        "Доступ: владелец и пользователи из `/allow`. Генерация занимает до пары минут."
    ),
    "index": (
        "🗂 **`/index` — память по истории чата (GraphRAG)**\n"
        "\n"
        "Строит из всей истории текущего чата базу знаний: досье на людей и персонажей (canon-факты\n"
        "и fanon-мнения), граф отношений, описанные и векторизованные фото — чтобы `/ask` мог отвечать\n"
        "на сложные вопросы «кто такой X», «из-за чего повздорили Y и Z», «скинь ту картинку со спора».\n"
        "\n"
        "**Команды (только владелец, работает в фоне):**\n"
        "   `/index` — оценка: сколько сообщений/фото и примерная стоимость прохода\n"
        "   `/index go` — запустить индексацию в фоне\n"
        "   `/index status` — прогресс по этапам\n"
        "   `/index pause` / `/index resume` — пауза и продолжение (состояние на чекпоинтах в БД)\n"
        "   `/index update` — догнать новые сообщения после прошлой индексации\n"
        "   `/index failed` — показать poison-диапазоны, которые были честно пропущены после дробления\n"
        "   `/index retry failed [1|2]` — переиндексировать пропущенные блоки/сцены точечно\n"
        "   `/index eval template|run|report` — шаблон и прогон smoke-eval качества поиска\n"
        "\n"
        "**Этапы:** 0 — дамп истории в БД · 1 — досье и алиасы · 2 — граф связей · 3 — вектора для поиска ·\n"
        "   4 — сводки по месяцам («как менялось со временем») · 5 — фото (последней, не блокирует поиск).\n"
        "   При рестарте бота индексация продолжается с последнего чекпоинта (авто).\n"
        "\n"
        "**🔎 Поиск в `/ask`:** после индексации у `/ask` в этом чате появляются инструменты памяти —\n"
        "   у владельца модель сама решает, когда лезть в базу: смысловой поиск по сценам/досье/связям, полное досье\n"
        "   персонажа по имени, и поиск+пересылка старого фото по описанию. Спрашивай обычным `/ask`:\n"
        "   «кто такой Тостер?», «из-за чего спорили Дима и Олег?», «скинь ту картинку со спора про меч».\n"
        "   Гостям из `/allow` память не выдаётся, если явно не включить `INDEX_MEMORY_FOR_GUESTS=1`.\n"
        "\n"
        "**📇 Досье и правка графа:**\n"
        "   `/entity show <имя|алиас>` — карточка: canon-факты, мнение чата, связи (со ссылками на сообщения)\n"
        "   `/entity merge <id1> <id2>` — слить две сущности (id2 → id1)\n"
        "   `/entity rename <id> <имя>` · `/entity alias <id> <алиас>` · `/entity split <id> <алиас>`\n"
        "   `/entity relink` — привязать несопоставленных участников к их Telegram-id (по author_id)\n"
        "\n"
        "⚙️ Нужны: `INDEX_DB_URL` (MariaDB/MySQL) в `.env`, пакеты `pymysql`+`numpy`, `OPENROUTER_API_KEY`.\n"
        "🧠 Модели: DeepSeek V4 Flash (экстракция), медиа-модель `/media` (фото),\n"
        "   text-embedding-3-small (тексты) + gemini-embedding-2 (картинки)."
    ),
    "keys": (
        "🔑 **Какие API-ключи за что отвечают** (в файле `.env`)\n"
        "\n"
        "**ОБЯЗАТЕЛЬНЫЙ — только один:**\n"
        "   `DEEPSEEK_API_KEY` — «мозг» бота. С ним одним уже работают:\n"
        "      `/ask`, `/search`, `/digest`, авто-ответ. Без него бот не отвечает.\n"
        "\n"
        "**НЕОБЯЗАТЕЛЬНЫЕ** (без них бот НЕ падает — просто часть функций выключена):\n"
        "   `OPENROUTER_API_KEY` — даёт:\n"
        "      • распознавание картинок/кружков в `/ask` (vision-модели `[OR]`);\n"
        "      • возможность ставить любую модель OpenRouter для ответов (`/model vendor/model`).\n"
        "   `OPENCODE_API_KEY` — даёт vision-модели `[OC]` (Kimi / GLM / Qwen / MiMo) в `/model media`.\n"
        "   `MODELGATE_API_KEY` — даёт модели **Claude** (Opus / Sonnet / Haiku) для ответов\n"
        "      (`/model` → раздел «Claude (ModelGate)»). Текст и поиск по каналам работают;\n"
        "      картинки напрямую (`-g`) НЕ принимает — фото идут через OCR/медиа-модель как обычно.\n"
        "   `OPENAI_API_KEY` — даёт модели **OpenAI** (GPT-5.5 / GPT-5.4 / o3) для ответов\n"
        "      (`/model` → раздел «OpenAI»). Официальный API — нужен баланс на platform.openai.com.\n"
        "   `ZAI_API_KEY` — даёт модели **z.ai / GLM** (GLM-5.2 / 4.7 Flash / 4.6V Flash) для ответов\n"
        "      (`/model` → раздел «z.ai (GLM)»). GLM-4.6V Flash видит картинки (`-g`); Flash-модели бесплатны.\n"
        "   `FIREWORKS_API_KEY` — даёт модели **Fireworks** (MiniMax M3 / Nemotron 3 Ultra / DeepSeek V4 Pro /\n"
        "      GLM-5.2 / Kimi K2.6) для ответов (`/model` → раздел «Fireworks»). MiniMax M3 и Kimi K2.6 видят картинки (`-g`).\n"
        "   `SAKANA_API_KEY` — даёт модели **Sakana AI / Fugu** (Fugu / Fugu Ultra) для ответов\n"
        "      (`/model` → раздел «Sakana AI (Fugu)»). Обе видят картинки (`-g`), окно 1M; Fugu Ultra — мульти-агентный\n"
        "      оркестратор поверх фронтир-LLM (сильный, но может быть медленным).\n"
        "   `GLOY_API_KEY` — даёт модели **LLM API FUN / Gloy AI** (Gloy AI 1.0 / 2.0) для ответов\n"
        "      (`/model` → раздел «LLM API FUN (Gloy AI)»). Только текст (без картинок и без веб-поиска инструментами).\n"
        "   `GOOGLE_GENAI_API_KEY` — даёт модели **Google Gemini** (Gemini 3.5 Flash / 3.1 Flash Lite)\n"
        "      для ответов (`/model` → раздел «Google Gemini»; видят картинки `-g`) И **голосовые\n"
        "      ответы** (`/voice`, флаг `-v`) — один ключ на оба. Можно указать несколько ключей\n"
        "      через запятую или в `GOOGLE_GENAI_API_KEYS` (ротация).\n"
        "   `FISH_AUDIO_API_KEY` — альтернативный TTS-движок Fish Audio (`/voice engine fish`,\n"
        "      `/voice fish` — поиск/избранное голосов).\n"
        "   `LLAMA_CLOUD_API_KEY` — дешёвый **OCR фото** в `/ask` по умолчанию (LlamaParse).\n"
        "      Без него фото автоматически описывает vision-модель (как раньше).\n"
        "   `TAVILY_API_KEY` — **веб-поиск** в `/ask`: модель сама ищет в интернете, читает\n"
        "      страницы по ссылкам и обходит сайты (Tavily, бесплатно 1000 запросов/мес).\n"
        "      Ключ: https://app.tavily.com\n"
        "\n"
        "**Что будет без необязательных ключей:**\n"
        "   • Нет OpenRouter и OpenCode → текст разбирается нормально, но фото/кружки в `/ask`\n"
        "     не читаются (голос всё равно работает через Parakeet).\n"
        "   • В списках `/model` / `/model media` недоступные модели помечены `⚠️нет ключа`.\n"
        "\n"
        "📌 Telegram-доступ (`api_id` / `api_hash`) — тоже обязателен, без него бот не запустится."
    ),
    "channels": (
        "📡 **Каналы, поиск и дайджест**\n"
        "\n"
        "**Управление списком каналов:**\n"
        "   `/channels` — показать подключённые каналы.\n"
        "   `/channels scan` — просканировать твои подписки и показать их id.\n"
        "   `/channels add N` или `/channels add @name` — добавить канал.\n"
        "   `/channels remove N` или `/channels remove @name` — убрать.\n"
        "\n"
        "**Поиск по каналам:**\n"
        "   `/search запрос` — найти релевантное в подключённых каналах (топ-10) и обобщить.\n"
        "\n"
        "**Дайджест:**\n"
        "   `/digest` — собрать дайджест по каналам прямо сейчас.\n"
        "   `/digest time 09:00` — присылать автоматически каждый день в указанное время.\n"
        "\n"
        "🔑 Работает на ключе DeepSeek; ключи OpenRouter/OpenCode тут не нужны."
    ),
    "auto": (
        "🔁 **Авто-ответ** (с памятью диалога)\n"
        "\n"
        "Бот сам отвечает на входящие сообщения в текущем чате, помня предыдущие реплики.\n"
        "\n"
        "   `/auto_reply` — включить в этом чате.\n"
        "   `/auto_reply off` — выключить.\n"
        "\n"
        "⚠️ Включай осознанно: бот будет писать от твоего имени. Память диалога ведётся\n"
        "    отдельно по каждому чату."
    ),
    "allow": (
        "👥 **Доступ к `/ask` для других людей**\n"
        "\n"
        "По умолчанию `/ask` доступен только тебе. Можно выдать доступ другим.\n"
        "\n"
        "   `/allow @name` — разрешить пользователю (безлимитно по умолчанию).\n"
        "   `/allow @name N` — разрешить, но не больше N запросов.\n"
        "   `/allow @name unlimited` — явный безлимит.\n"
        "   `/allow` в ответ на сообщение — выдать доступ его автору.\n"
        "   `/allow remove` (по @name или в ответ) — забрать доступ.\n"
        "\n"
        "💡 Удобно, чтобы дать другу пользоваться ботом без передачи аккаунта."
    ),
    "song": (
        "🎵 **`/song [N] [текст]`** — печать с эффектом набора\n"
        "\n"
        "Постепенно «печатает» переданный текст, имитируя живой набор.\n"
        "Слова не разрываются: если шаг попадает в середину слова, оно открывается целиком.\n"
        "\n"
        "   `/song привет мир` — печать со скоростью по умолчанию (3 символа за шаг).\n"
        "   `/song 7 привет мир` — первое число = символов за шаг (1–200, больше = быстрее).\n"
        "   `/song` или `/song 10` — печатает текст по умолчанию (с заданной скоростью).\n"
        "\n"
        "Декоративная команда — на AI и ключи не влияет."
    ),
    "help": (
        "ℹ️ **Как пользоваться самой `/help`**\n"
        "\n"
        "   `/help` — оглавление: список всех разделов + быстрая шпаргалка.\n"
        "   `/help <раздел>` — подробная справка по одному разделу.\n"
        "   `/help all` — вывести ВСЕ разделы подряд (длинно).\n"
        "\n"
        "**Доступные разделы:**\n"
        "   `ask` · `model` · `media` · `voice` · `gen` · `keys` · `channels` · `auto` · `allow` · `song` · `help`\n"
        "\n"
        "_Примеры:_\n"
        "   `/help ask`   — всё про вопросы к AI\n"
        "   `/help media` — про vision-модели и метки [OR]/[OC]\n"
        "   `/help keys`  — какой ключ обязателен, а какой нет\n"
        "\n"
        "💡 Регистр и лишние пробелы не важны: `/help  MEDIA` сработает как `/help media`.\n"
        "\n"
        "**Префиксы:** каждая команда работает и через `/`, и через `.` (`.help` = `/help`).\n"
        "❗ Исключение: в личке с ботом (например @some\\_bot) слэш-команды юзербот\n"
        "пропускает — они адресованы тому боту. Там используй точку: `.ask 50 …`."
    ),
    "status": (
        "📊 **`/status` — все текущие настройки разом**\n"
        "\n"
        "Показывает одной командой: активную модель ответов (провайдер, окно контекста,\n"
        "поддержку поиска 🔧 и vision), медиа-модель, TTS-движок и выбранный голос,\n"
        "режим авто-голоса, у кого есть доступ к `/ask`, число чатов с авто-ответом,\n"
        "сколько каналов подключено и время дайджеста, а также какие API-ключи активны.\n"
        "Только для тебя (владельца). Ничего не меняет — просто сводка."
    ),
}


@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]status$", from_users="me"))
async def status_command(event):
    """Сводка всех текущих настроек бота (только владелец)."""
    if await _slash_for_other_bot(event):
        return
    L = []
    # — модель ответов —
    provider, _mid, label, ctx, _ = MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["deepseek-pro"])
    prov_name = {"deepseek": "DeepSeek", "openrouter": "OpenRouter", "opencode": "OpenCode Go",
                 "oc_anthropic": "OpenCode (нативный)", "modelgate": "ModelGate (Claude)",
                 "openai": "OpenAI", "google": "Google Gemini", "zai": "z.ai (GLM)", "fireworks": "Fireworks",
                 "sakana": "Sakana AI (Fugu)", "gloy": "LLM API FUN (Gloy AI)"}.get(provider, provider)
    ts = MODEL_TOOLS_SUPPORT.get(ACTIVE_MODEL)
    search_mark = "🔧 есть" if ts is True else ("🚫 нет" if ts is False else "❔ не проверен")
    sv = active_model_supports_vision()
    vis_mark = "✅ да" if sv is True else ("❌ нет" if sv is False else "❔ неизвестно")
    L.append("📊 **СТАТУС БОТА**")
    L.append(f"\n🧠 **Модель ответов:** {label} (`{ACTIVE_MODEL}`)")
    L.append(f"   провайдер: {prov_name} · окно: {_fmt_ctx(ctx)} · поиск по каналам: {search_mark} · vision (`-g`): {vis_mark}")
    if _supports_reasoning(provider):
        if REASONING_EFFORT:
            applied = _clamp_reasoning(_mid, REASONING_EFFORT, provider)
            reff = f"`{REASONING_EFFORT}`" + (f" → `{applied}` для этой модели" if applied != REASONING_EFFORT else "")
        else:
            reff = "авто (дефолт модели)"
        L.append(f"   🤔 глубина размышлений: {reff} · `/model reason` — сменить")
    elif REASONING_EFFORT:
        # активная модель ризонинг не поддерживает, но уровень выбран глобально — показываем, чтобы не терялся
        L.append(f"   🤔 глубина размышлений: выбрана `{REASONING_EFFORT}` глобально, но {prov_name} её не использует · `/model reason`")
    if openai_api_key:
        _li, _lo, ltot = _openai_usage_today("large")
        _mi, _mo, mtot = _openai_usage_today("mini")
        lp = min(100, int(ltot * 100 / OPENAI_FREE_DAILY_LARGE))
        mp = min(100, int(mtot * 100 / OPENAI_FREE_DAILY_MINI))
        L.append(f"🎁 **OpenAI бесплатная квота (data sharing):** основные ~{_fmt_ctx(ltot)}/250k ({lp}%) · mini ~{_fmt_ctx(mtot)}/2.5M ({mp}%) · сброс 00:00 UTC (03:00 МСК)")
    # — медиа-модель —
    if ACTIVE_MEDIA_MODEL in MEDIA_MODEL_REGISTRY:
        media_label = MEDIA_MODEL_REGISTRY[ACTIVE_MEDIA_MODEL][1]
    elif ACTIVE_MEDIA_MODEL in MEDIA_OPENCODE_SLUGS and ACTIVE_MEDIA_MODEL in MODEL_REGISTRY:
        media_label = f"{MODEL_REGISTRY[ACTIVE_MEDIA_MODEL][2]} [OpenCode]"
    else:
        media_label = f"{ACTIVE_MEDIA_MODEL} (кастомная)"
    L.append(f"🖼 **Фото в /ask:** OCR LlamaParse (cost-effective) {'✅' if llama_cloud_api_key else '❌ нет ключа → vision'} · vision (`-m`): {media_label}")
    L.append(f"🌐 **Веб-поиск (Tavily):** {'✅ модель сама ищет в интернете (search/extract/crawl/map)' if tavily_api_key else '❌ нет TAVILY_API_KEY'}")
    # — голос —
    if not tts_available and not fish_available:
        L.append("🎙 **Голос:** недоступен (нет ключей Google TTS / Fish)")
    else:
        if TTS_ENGINE == "fish":
            fname = next((f["title"] for f in FISH_FAVORITES if f["id"] == FISH_VOICE), FISH_VOICE or "—")
            L.append(f"🎙 **Голос:** движок **Fish** ({FISH_TTS_MODEL}) · голос: {fname}" + (f" (`{FISH_VOICE}`)" if FISH_VOICE else " (не выбран)"))
        else:
            L.append(f"🎙 **Голос:** движок **Gemini** · голос: {ACTIVE_VOICE}")
        L.append(f"   авто-голос: {'🟢 вкл' if VOICE_AUTO else '⚪ выкл'} · Google TTS: {'✅' if tts_available else '❌'} · Fish: {'✅' if fish_available else '❌'}")
    # — доступ к /ask —
    owner_who = (("@" + OWNER_USERNAME) if OWNER_USERNAME else (OWNER_NAME or "владелец"))
    L.append(f"\n👤 **Доступ к `/ask`:** ты ({owner_who})")
    if ALLOWED_USERS:
        L.append(f"   + ещё {len(ALLOWED_USERS)}:")
        for uid, rec in list(ALLOWED_USERS.items())[:15]:
            uname = rec.get("username") if isinstance(rec, dict) else rec
            lim = rec.get("limit") if isinstance(rec, dict) else None
            who = ("@" + uname) if uname else str(uid)
            L.append(f"     • {who} · лимит: {_fmt_allow_limit(lim)}")
    else:
        L.append("   (больше ни у кого — `/allow @user` чтобы дать)")
    # — авто-ответ / каналы / дайджест —
    L.append(f"\n🔁 **Авто-ответ:** включён в {len(AUTO_REPLY_ACTIVE_CHATS)} чат(ах)")
    _dig = load_json(DIGEST_STATE_PATH, {}).get("digest_time", "09:00")
    L.append(f"📡 **Каналы:** подключено {len(get_tracked())} · дайджест в {_dig}")
    # — генерация изображений —
    L.append(f"🎨 **Генерация (`/gen`):** GPT Image 2 → Gemini 3.1 Flash Image (фолбэк) {'✅' if openrouter_client is not None else '❌ нет OPENROUTER_API_KEY'}")
    # — избранное —
    L.append(f"⭐ **Избранное:** {len(FISH_FAVORITES)} Fish-голос(ов) · {len(CUSTOM_MODELS)} кастомных моделей")
    # — ключи —
    keys = []
    for p, nm in [("deepseek", "DeepSeek"), ("openrouter", "OpenRouter"), ("opencode", "OpenCode"), ("modelgate", "Claude/ModelGate"), ("openai", "OpenAI"), ("google", "Google Gemini"), ("zai", "z.ai (GLM)"), ("fireworks", "Fireworks")]:
        keys.append(f"{nm} {'✅' if _client_for_provider(p) is not None else '❌'}")
    keys.append(f"Tavily {'✅' if tavily_api_key else '❌'}")
    keys.append(f"Google TTS {'✅' if tts_available else '❌'}")
    keys.append(f"Fish {'✅' if fish_available else '❌'}")
    L.append("🔑 **Ключи:** " + " · ".join(keys))
    L.append("\n⚙️ Сменить: `/model` · `/voice` · `/allow` · подробности — `/help`")
    await event.edit("\n".join(L)[:4000])


@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]help(?:\s+(\S+))?\s*$", from_users="me"))
async def help_command(event):
    if await _slash_for_other_bot(event):
        return
    _, _, active_label = get_active_model()
    arg = (event.pattern_match.group(1) or "").strip().lower()

    if not arg:
        await event.edit(_help_index(active_label))
        return

    if arg == "all":
        order = ["ask", "model", "media", "voice", "gen", "index", "keys", "channels", "auto", "allow", "status", "song", "help"]
        full = "\n\n━━━━━━━━━━━━━━━━━━━━━\n\n".join(_HELP_SECTIONS[k] for k in order)
        # Telegram лимит ~4096 на сообщение — режем безопасно по разделам.
        chunk, buf = "", []
        for part in full.split("\n\n━━━━━━━━━━━━━━━━━━━━━\n\n"):
            piece = (part if not chunk else "\n\n━━━━━━━━━━━━━━━━━━━━━\n\n" + part)
            if len(chunk) + len(piece) > 3900:
                buf.append(chunk)
                chunk = part
            else:
                chunk += piece
        if chunk:
            buf.append(chunk)
        await event.edit(buf[0])
        for extra in buf[1:]:
            await event.respond(extra)
        return

    section = _HELP_SECTIONS.get(arg)
    if section is None:
        known = "`, `".join(_HELP_SECTIONS.keys())
        await event.edit(
            f"❓ Раздел `{arg}` не найден.\n\n"
            f"Доступные: `{known}`, `all`.\n"
            f"Открой оглавление командой `/help`."
        )
        return

    full = section + f"\n\n━━━━━━━━━━━━━━━━━━━━━\n⚙️ Активная модель: **{active_label}**  ·  `/help` — все разделы"
    if len(full) <= 4096:
        await event.edit(full)
        return
    # Telegram лимит 4096 — длинный раздел (например /gen разросся) режем по строкам на несколько сообщений
    chunks, buf = [], ""
    for line in full.split("\n"):
        if len(buf) + len(line) + 1 > 3900:
            chunks.append(buf)
            buf = line
        else:
            buf = (buf + "\n" + line) if buf else line
    if buf:
        chunks.append(buf)
    await event.edit(chunks[0])
    for extra in chunks[1:]:
        await event.respond(extra)


# ════════════════════════════════════════════════════════════════════════
#  /index — агентская мультимодальная GraphRAG-память по истории чата
#  MariaDB (граф+досье+сцены+медиа) + numpy-cosine поиск. Модели: DeepSeek
#  V4 Flash (экстракция текста), медиа-модель /media (описание фото),
#  text-embedding-3-small (тексты), gemini-embedding-2 (картинки).
# ════════════════════════════════════════════════════════════════════════

_IDX_TL = threading.local()          # per-thread pymysql-соединение (клиенты бота sync → asyncio.to_thread)
_INDEX_TASKS: dict = {}              # {chat_id: asyncio.Task} активных фоновых индексаций
_INDEX_CONTROL: dict = {}           # {chat_id: "run"|"pause"} — мягкая остановка на чекпоинтах
_INDEX_DDL_DONE = False             # DDL применяется один раз за процесс
_INDEX_EXTRACT_OK = None            # None=не проверяли, True/False — доступна ли INDEX_EXTRACT_MODEL нашему ключу
_INDEX_COUNTS: dict = {}            # {(chat_id, kind): (count, monotonic_ts)} — кэш COUNT для выбора backend поиска


class IndexTransientError(Exception):
    """Провайдер экстракции недоступен (сеть/5xx/timeout) — авария, а не poison-контент.
    Стадия должна встать в error и дождаться /index resume, НЕ дробить блок в ложные skip."""


def _index_available() -> str:
    """'' если /index можно запускать; иначе причина-строка (нет пакета/ключа/OpenRouter)."""
    if pymysql is None:
        return "нет пакета pymysql (добавь в requirements и переустанови зависимости)"
    if _np is None:
        return "нет пакета numpy (добавь в requirements и переустанови зависимости)"
    if not index_db_url:
        return "не задан INDEX_DB_URL в .env (строка подключения к MariaDB)"
    if openrouter_client is None:
        return "нет OPENROUTER_API_KEY (нужен для экстракции и эмбеддингов)"
    return ""


def _index_dsn() -> dict:
    u = urlsplit(index_db_url)
    return dict(host=u.hostname, port=u.port or 3306,
                user=unquote(u.username or ""), password=unquote(u.password or ""),
                db=(u.path or "/").lstrip("/"))


def _index_conn():
    """Живое соединение для ТЕКУЩЕГО потока (ping+reconnect). pymysql не потокобезопасен → thread-local."""
    c = getattr(_IDX_TL, "conn", None)
    if c is not None:
        try:
            c.ping(reconnect=True)
            return c
        except Exception:
            try:
                c.close()
            except Exception:
                pass
            _IDX_TL.conn = None
    d = _index_dsn()
    conn = pymysql.connect(host=d["host"], port=d["port"], user=d["user"], password=d["password"],
                           database=d["db"], charset="utf8mb4", autocommit=True, connect_timeout=15,
                           cursorclass=pymysql.cursors.DictCursor)
    _IDX_TL.conn = conn
    return conn


def _idx_run(fn):
    for attempt in range(2):  # реконнект-ретрай при обрыве
        conn = _index_conn()
        try:
            return fn(conn)
        except (pymysql.err.OperationalError, pymysql.err.InterfaceError):
            try:
                conn.close()
            except Exception:
                pass
            _IDX_TL.conn = None
            if attempt == 1:
                raise


def _db_write(sql, params=None, many=False):
    def _op(conn):
        with conn.cursor() as cur:
            if many:
                cur.executemany(sql, params or [])
            else:
                cur.execute(sql, params or ())
            return cur.rowcount, cur.lastrowid
    return _idx_run(_op)


def _db_read(sql, params=None):
    def _op(conn):
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()
    return _idx_run(_op)


def _db_transaction(statements: list):
    """Атомарно выполняет список (sql, params). Нужен для ручных правок графа."""
    def _op(conn):
        old_autocommit = conn.get_autocommit()
        conn.autocommit(False)
        try:
            with conn.cursor() as cur:
                for sql, params in statements:
                    cur.execute(sql, params or ())
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.autocommit(old_autocommit)
    return _idx_run(_op)


async def db_write(sql, params=None, many=False):
    return await asyncio.to_thread(_db_write, sql, params, many)


async def db_read(sql, params=None):
    return await asyncio.to_thread(_db_read, sql, params)


async def db_transaction(statements: list):
    return await asyncio.to_thread(_db_transaction, statements)


def _index_invalidate(chat_id: int, *kinds: str):
    for kind in kinds:
        _INDEX_MATRIX.pop((chat_id, kind), None)
        _INDEX_HNSW.pop((chat_id, kind), None)  # ANN-индекс тоже пересобрать
        _INDEX_COUNTS.pop((chat_id, kind), None)  # счётчик устарел


async def _index_count_ok(chat_id: int, kind: str) -> int:
    """COUNT(*) непустых векторов kind с кэшем INDEX_COUNT_TTL (выбор backend не должен сканить БД на каждый поиск)."""
    ck = (chat_id, kind)
    cached = _INDEX_COUNTS.get(ck)
    now = time.monotonic()
    if cached and (now - cached[1]) < INDEX_COUNT_TTL:
        return cached[0]
    table, emb_col, _key, _extra = _INDEX_KINDS[kind]
    n = (await db_read(f"SELECT COUNT(*) c FROM {table} WHERE chat_id=%s AND {emb_col} IS NOT NULL", (chat_id,)))[0]["c"]
    _INDEX_COUNTS[ck] = (n, now)
    return n


def _index_memory_allowed(is_owner: bool) -> bool:
    return bool(is_owner or index_memory_for_guests)


_INDEX_DDL = [
    """CREATE TABLE IF NOT EXISTS idx_state (
        chat_id BIGINT NOT NULL, stage TINYINT NOT NULL, `cursor` JSON NULL, stats JSON NULL,
        status VARCHAR(16) NOT NULL DEFAULT 'running', updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (chat_id, stage)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS messages (
        chat_id BIGINT NOT NULL, msg_id BIGINT NOT NULL, `date` DATETIME NULL, author_id BIGINT NULL,
        reply_to_id BIGINT NULL, txt MEDIUMTEXT NULL, media_uid VARCHAR(64) NULL, media_kind TINYINT NOT NULL DEFAULT 0,
        PRIMARY KEY (chat_id, msg_id), KEY k_date (chat_id, `date`)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS entities (
        id BIGINT AUTO_INCREMENT PRIMARY KEY, chat_id BIGINT NOT NULL, name VARCHAR(255) NOT NULL,
        entity_type VARCHAR(16) NOT NULL DEFAULT 'user', tg_user_id BIGINT NULL, aliases JSON NULL,
        canon_summary MEDIUMTEXT NULL, fanon_summary MEDIUMTEXT NULL, visual_features MEDIUMTEXT NULL,
        embedding VARBINARY(4096) NULL, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        KEY k_chat (chat_id), KEY k_name (chat_id, name)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS entity_claims (
        id BIGINT AUTO_INCREMENT PRIMARY KEY, chat_id BIGINT NOT NULL, entity_id BIGINT NOT NULL,
        kind VARCHAR(16) NOT NULL, claim MEDIUMTEXT NOT NULL, evidence JSON NULL,
        first_seen DATETIME NULL, last_seen DATETIME NULL,
        KEY k_ent (chat_id, entity_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS relations (
        id BIGINT AUTO_INCREMENT PRIMARY KEY, chat_id BIGINT NOT NULL, source_id BIGINT NOT NULL, target_id BIGINT NOT NULL,
        relation_type VARCHAR(64) NULL, canonical_type VARCHAR(32) NULL, context_summary MEDIUMTEXT NULL,
        weight FLOAT NOT NULL DEFAULT 1, first_seen DATETIME NULL, last_seen DATETIME NULL,
        status VARCHAR(16) NOT NULL DEFAULT 'active', evidence JSON NULL, embedding VARBINARY(4096) NULL,
        KEY k_chat (chat_id), KEY k_pair (chat_id, source_id, target_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS media_assets (
        chat_id BIGINT NOT NULL, msg_id BIGINT NOT NULL, file_uid VARCHAR(64) NULL, image_description MEDIUMTEXT NULL,
        entity_ids JSON NULL, emb_text VARBINARY(4096) NULL, emb_image VARBINARY(8192) NULL,
        PRIMARY KEY (chat_id, msg_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS chat_chunks (
        id BIGINT AUTO_INCREMENT PRIMARY KEY, chat_id BIGINT NOT NULL, start_msg_id BIGINT NULL, end_msg_id BIGINT NULL,
        scene_date DATETIME NULL, enriched_text MEDIUMTEXT NULL, meta JSON NULL, embedding VARBINARY(4096) NULL,
        KEY k_chat (chat_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS index_failed_ranges (
        id BIGINT AUTO_INCREMENT PRIMARY KEY, chat_id BIGINT NOT NULL, stage TINYINT NOT NULL,
        unit_type VARCHAR(32) NOT NULL, start_msg_id BIGINT NOT NULL, end_msg_id BIGINT NOT NULL,
        reason MEDIUMTEXT NULL, attempts INT NOT NULL DEFAULT 1, status VARCHAR(16) NOT NULL DEFAULT 'skipped',
        payload JSON NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY u_range (chat_id, stage, unit_type, start_msg_id, end_msg_id),
        KEY k_status (chat_id, stage, status)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS relation_events (
        id BIGINT AUTO_INCREMENT PRIMARY KEY, chat_id BIGINT NOT NULL, scene_key VARCHAR(64) NOT NULL,
        source_id BIGINT NOT NULL, target_id BIGINT NOT NULL, relation_type VARCHAR(64) NULL,
        canonical_type VARCHAR(32) NULL, context_summary MEDIUMTEXT NULL, evidence JSON NULL, scene_date DATETIME NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY u_event (chat_id, scene_key, source_id, target_id, canonical_type, relation_type),
        KEY k_pair (chat_id, source_id, target_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS entity_summary_parts (
        id BIGINT AUTO_INCREMENT PRIMARY KEY, chat_id BIGINT NOT NULL, entity_id BIGINT NOT NULL,
        kind VARCHAR(16) NOT NULL, part_no INT NOT NULL, summary MEDIUMTEXT NULL, claim_count INT NOT NULL DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY u_part (chat_id, entity_id, kind, part_no),
        KEY k_ent (chat_id, entity_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS index_eval_runs (
        id BIGINT AUTO_INCREMENT PRIMARY KEY, chat_id BIGINT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        cases_json JSON NULL, result_json JSON NULL, KEY k_chat (chat_id, created_at)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    # RAPTOR-lite: темпоральные роллап-саммари (level 1=месяц, 3=весь чат) для глобальных вопросов «как менялось»
    """CREATE TABLE IF NOT EXISTS time_rollups (
        id BIGINT AUTO_INCREMENT PRIMARY KEY, chat_id BIGINT NOT NULL, level TINYINT NOT NULL,
        bucket_key VARCHAR(24) NOT NULL, period_start DATETIME NULL, period_end DATETIME NULL,
        summary MEDIUMTEXT NULL, meta JSON NULL, embedding VARBINARY(4096) NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY u_roll (chat_id, level, bucket_key), KEY k_chat (chat_id, level)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]


_INDEX_MIGRATIONS = [
    """DELETE c1 FROM chat_chunks c1 JOIN chat_chunks c2
       ON c1.chat_id=c2.chat_id AND c1.start_msg_id=c2.start_msg_id
       AND c1.end_msg_id=c2.end_msg_id AND c1.id>c2.id""",
    "ALTER TABLE chat_chunks ADD UNIQUE KEY u_scene (chat_id, start_msg_id, end_msg_id)",
    "ALTER TABLE chat_chunks ADD KEY k_scene_range (chat_id, start_msg_id, end_msg_id)",
]


async def _index_ensure_ddl():
    global _INDEX_DDL_DONE
    if _INDEX_DDL_DONE:
        return
    for ddl in _INDEX_DDL:
        await db_write(ddl)
    for ddl in _INDEX_MIGRATIONS:
        try:
            await db_write(ddl)
        except Exception as e:
            msg = str(e).lower()
            if "duplicate" not in msg and "already exists" not in msg:
                log("INDEX", f"DDL migration skipped/failed: {e}")
    _INDEX_DDL_DONE = True
    log("INDEX", "DDL применён (таблицы готовы)")


# --- вектора: float16-блобы (numpy) ---
def _vec_pack(vec) -> bytes:
    """list[float]/np.array → компактный float16-блоб (усечение/паддинг до INDEX_EMBED_DIM)."""
    a = _np.asarray(vec, dtype=_np.float32)
    if a.shape[0] >= INDEX_EMBED_DIM:
        a = a[:INDEX_EMBED_DIM]
    else:
        a = _np.pad(a, (0, INDEX_EMBED_DIM - a.shape[0]))
    n = _np.linalg.norm(a)
    if n > 0:
        a = a / n  # нормируем → косинус = скалярное произведение
    return a.astype(_np.float16).tobytes()


def _vec_unpack(blob) -> "object":
    return _np.frombuffer(blob, dtype=_np.float16).astype(_np.float32) if blob else None


# --- состояние индексации в idx_state ---
async def _idx_get_state(chat_id: int, stage: int) -> dict:
    rows = await db_read("SELECT `cursor`, stats, status FROM idx_state WHERE chat_id=%s AND stage=%s", (chat_id, stage))
    if not rows:
        return {"cursor": {}, "stats": {}, "status": None}
    r = rows[0]
    return {"cursor": json.loads(r["cursor"]) if r["cursor"] else {},
            "stats": json.loads(r["stats"]) if r["stats"] else {},
            "status": r["status"]}


async def _idx_set_state(chat_id: int, stage: int, cursor=None, stats=None, status=None):
    await db_write(
        """INSERT INTO idx_state (chat_id, stage, `cursor`, stats, status)
             VALUES (%s,%s,%s,%s,COALESCE(%s,'running'))
           ON DUPLICATE KEY UPDATE `cursor`=COALESCE(VALUES(`cursor`),`cursor`),
             stats=COALESCE(VALUES(stats),stats), status=COALESCE(%s,status)""",
        (chat_id, stage, json.dumps(cursor, ensure_ascii=False) if cursor is not None else None,
         json.dumps(stats, ensure_ascii=False) if stats is not None else None, status, status))


def _index_scene_key(chat_id: int, start_msg_id: int, end_msg_id: int) -> str:
    raw = f"{chat_id}:{start_msg_id}:{end_msg_id}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _index_relation_event_key(chat_id: int, scene_date, source_id: int, target_id: int,
                              polarity: str) -> str:
    """Детерминированный дневной ключ события связи по ПОЛЯРНОСТИ (без сырого relation_type!).
    Одна пара (source→target, polarity) — ОДНО событие в сутки → вес = «в скольких разных днях
    проявлялась связь этой полярности». relation_type НЕ в ключе: LLM пишет «спорит»/«ругается»
    в один день → раньше это давало +2, теперь дедупится. Resume/update overlap → тот же день → не задваивает."""
    if scene_date is None:
        day = "0"
    elif isinstance(scene_date, str):
        day = scene_date[:10]  # 'YYYY-MM-DD…'
    else:
        day = scene_date.strftime("%Y-%m-%d")
    raw = f"{chat_id}:{source_id}:{target_id}:{polarity}:{day}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _index_rows_to_text(rows: list, amap: dict) -> str:
    lines = []
    for r in rows:
        nm = amap.get(r["author_id"], f"user{r['author_id']}" if r.get("author_id") else "?")
        lines.append(f"[{r['msg_id']}] {nm}: {r.get('txt') or ''}")
    return "\n".join(lines)


async def _index_record_failed_range(chat_id: int, stage: int, unit_type: str, start_msg_id: int,
                                     end_msg_id: int, reason: str, payload=None, status: str = "skipped"):
    await db_write(
        """INSERT INTO index_failed_ranges
           (chat_id,stage,unit_type,start_msg_id,end_msg_id,reason,attempts,status,payload)
           VALUES (%s,%s,%s,%s,%s,%s,1,%s,%s)
           ON DUPLICATE KEY UPDATE attempts=attempts+1, reason=VALUES(reason),
             status=VALUES(status), payload=VALUES(payload)""",
        (chat_id, stage, unit_type, int(start_msg_id), int(end_msg_id), str(reason)[:4000], status,
         json.dumps(payload or {}, ensure_ascii=False)))


async def _index_failed_count(chat_id: int, status: str = "skipped") -> int:
    try:
        return (await db_read(
            "SELECT COUNT(*) c FROM index_failed_ranges WHERE chat_id=%s AND status=%s",
            (chat_id, status)))[0]["c"]
    except Exception:
        return 0


async def _index_failed_rows(chat_id: int, stage=None, status: str = "skipped", limit: int = 20) -> list:
    if stage is None:
        return await db_read(
            "SELECT * FROM index_failed_ranges WHERE chat_id=%s AND status=%s ORDER BY stage,start_msg_id LIMIT %s",
            (chat_id, status, limit))
    return await db_read(
        "SELECT * FROM index_failed_ranges WHERE chat_id=%s AND stage=%s AND status=%s ORDER BY start_msg_id LIMIT %s",
        (chat_id, int(stage), status, limit))


def _index_norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


_INDEX_ALIAS_STOP = {
    "я", "ты", "он", "она", "оно", "они", "мы", "вы",
    "его", "ее", "её", "их", "мой", "моя", "твой", "твоя",
    "автор", "участник", "админ", "бот", "человек", "персонаж",
}


def _index_identity_key(s: str) -> str:
    key = _index_norm_name(s)
    if len(key) < 2 or key in _INDEX_ALIAS_STOP:
        return ""
    return key


def _media_meta(m):
    """(media_uid, media_kind) для сообщения: 1=фото, 2=картинка-документ, 3=прочий документ, 0=нет."""
    ph = getattr(m, "photo", None)
    if ph and not isinstance(getattr(m, "media", None), MessageMediaWebPage):
        return (f"{getattr(ph, 'id', '')}", 1)
    doc = getattr(m, "document", None)
    if doc:
        mime = (getattr(doc, "mime_type", None) or "").lower()
        return (f"{getattr(doc, 'id', '')}", 2 if mime.startswith("image/") else 3)
    return (None, 0)


def _index_is_image_msg(m) -> bool:
    return bool(m and (_is_attached_photo(m) or _is_attached_image_doc(m)))


# --- STAGE 0: сырой дамп истории в messages ---
async def _index_stage0_dump(chat_id: int, progress_cb=None):
    """Выкачивает всю историю чата в messages (INSERT IGNORE), чекпоинт по max msg_id.
    reverse=True + min_id=cursor → дозагрузка с места обрыва (resume). Медиа НЕ качаем (только uid/kind)."""
    st = await _idx_get_state(chat_id, 0)
    last_id = int(st["cursor"].get("last_msg_id", 0))
    done = int(st["stats"].get("dumped", 0))
    await _idx_set_state(chat_id, 0, status="running")
    buf, batch_max = [], 0
    log("INDEX", f"Stage0 дамп чата {chat_id}: продолжаю с msg_id>{last_id} (уже {done})")

    async def _flush():
        nonlocal buf, done, last_id
        if not buf:
            return
        await db_write(
            """INSERT IGNORE INTO messages (chat_id,msg_id,`date`,author_id,reply_to_id,txt,media_uid,media_kind)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""", buf, many=True)
        done += len(buf)
        last_id = max(last_id, batch_max)
        await _idx_set_state(chat_id, 0, cursor={"last_msg_id": last_id}, stats={"dumped": done})
        if progress_cb:
            await progress_cb(f"📥 Дамп: {done} сообщений (id≤{last_id})…")
        buf = []

    try:
        async for m in client.iter_messages(chat_id, reverse=True, min_id=last_id):
            if _INDEX_CONTROL.get(chat_id) == "pause":
                await _flush()
                await _idx_set_state(chat_id, 0, status="paused")
                log("INDEX", f"Stage0 чата {chat_id}: пауза на id≤{last_id}")
                return "paused"
            uid, kind = _media_meta(m)
            d = getattr(m, "date", None)
            buf.append((chat_id, m.id,
                        d.strftime("%Y-%m-%d %H:%M:%S") if d else None,
                        getattr(m, "sender_id", None), getattr(m, "reply_to_msg_id", None),
                        (m.raw_text or None), uid, kind))
            batch_max = max(batch_max, m.id)
            if len(buf) >= INDEX_DUMP_BATCH:
                await _flush()
        await _flush()
    except FloodWaitError as e:
        await _flush()
        log("INDEX", f"Stage0 чата {chat_id}: FloodWait {e.seconds}с — пауза, дожмётся при resume")
        await _idx_set_state(chat_id, 0, status="paused")
        return "floodwait"
    await _idx_set_state(chat_id, 0, status="done", stats={"dumped": done})
    log("INDEX", f"Stage0 чата {chat_id}: готово, {done} сообщений")
    return "done"


# --- LLM-экстракция (V4 Flash, JSON-mode) ---
def _json_from_llm(text: str):
    """Достаёт JSON-объект из ответа модели (снимает ```-ограждения, берёт первый {...})."""
    if not text:
        return None
    t = _strip_think(text).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip(), flags=re.I).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        try:
            return json.loads(t[a:b + 1])
        except Exception:
            return None
    return None


async def _index_extract(system: str, user: str, max_tokens: int = INDEX_EXTRACT_MAX_TOKENS):
    """Экстракция через INDEX_EXTRACT_MODEL (V4 Flash) с JSON-mode; фолбэк на запасную.
    Возвращает dict при успехе; None — если провайдер ОТВЕТИЛ, но контент не парсится как JSON
    (детерминированный «poison» — можно дробить/скипать). Если провайдер ни разу не ответил
    (сеть/5xx/timeout на всех попытках) — raise IndexTransientError (это авария, а не poison:
    стадия должна встать в error, а не фрагментировать блок в ложные skip)."""
    global _INDEX_EXTRACT_OK
    models = [INDEX_EXTRACT_MODEL, INDEX_EXTRACT_FALLBACK]
    got_response = False  # был ли хоть один валидный ответ провайдера (пусть и не-JSON)
    for mi, model in enumerate(models):
        for attempt in range(1, INDEX_EXTRACT_RETRIES + 1):
            try:
                mt = min(max_tokens, INDEX_MODEL_MAX_OUT.get(model, max_tokens))  # кламп под реальный лимит модели
                resp = await asyncio.to_thread(
                    openrouter_client.chat.completions.create,
                    model=model, max_tokens=mt, temperature=0.2,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    timeout=240,
                    extra_body={"reasoning": {"enabled": False}},  # JSON-экстракция: CoT — пустая трата токенов/времени
                )
                got_response = True
                content = resp.choices[0].message.content or ""
                data = _json_from_llm(content)
                if data is not None:
                    _INDEX_EXTRACT_OK = True
                    return data
                fin = resp.choices[0].finish_reason
                log("INDEX", f"Экстракция {model}: не распарсил JSON (finish={fin}, попытка {attempt})")
                if fin == "length":
                    # ответ обрезан по потолку токенов — повтор того же запроса даст ту же обрезку,
                    # смена модели тоже (блок/сцена слишком «плотные»). Детерминированно, но НЕ poison:
                    # возвращаем None сразу → caller дробит единицу пополам (меньше вход → влезет вывод).
                    return None
            except Exception as e:
                code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                if code in (401, 402):  # ключ/квота — фатально, НЕ poison: стопим стадию (иначе весь чат уйдёт в ложные skip)
                    raise IndexTransientError(f"config {code} (ключ/квота недоступны) — стоп, не poison: {e}")
                if code in (403, 404):  # доступ/модель-not-found — пробуем запасную; если и она недоступна → transient-стоп (не skip)
                    log("INDEX", f"Экстракция {model}: {code} (доступ/модель) — пробую запасную")
                    break  # got_response НЕ ставим → обе недоступны дадут transient-стоп, а не poison
                if code in (413, 422):  # payload/param слишком большой — дробим вход (как finish=length), не skip
                    log("INDEX", f"Экстракция {model}: {code} (payload/param) — дроблю вход")
                    return None
                if code and 400 <= code < 500 and code != 429:  # прочие 4xx — детерминированный poison-контент
                    got_response = True
                    log("INDEX", f"Экстракция {model}: детерминированный {code} — не ретраю (poison): {e}")
                    break
                log("INDEX", f"Экстракция {model} ошибка (попытка {attempt}): {e}")
            if attempt < INDEX_EXTRACT_RETRIES:
                await asyncio.sleep(min(30, 2 ** attempt) + random.random())
        if mi == 0:
            log("INDEX", f"Экстракция {model}: пробую запасную модель")
    if not got_response:  # провайдер недоступен — транзиент, НЕ poison
        raise IndexTransientError("extract: провайдер не ответил ни разу (сеть/5xx/timeout)")
    return None


# --- STAGE 1: досье («снежный ком») ---
_INDEX_AUTHORS: dict = {}  # {chat_id: {author_id: name}} — кэш имён участников на процесс


async def _index_author_map(chat_id: int) -> dict:
    """Имена участников чата {id: имя} для подписи сообщений (best-effort; лурки-без-имени → user{id})."""
    if chat_id in _INDEX_AUTHORS:
        return _INDEX_AUTHORS[chat_id]
    amap = {}
    try:
        parts = await client.get_participants(chat_id, aggressive=False)
        for p in parts:
            amap[p.id] = utils.get_display_name(p) or f"user{p.id}"
    except Exception as e:
        log("INDEX", f"Участники чата {chat_id} недоступны ({e}) — подписи по id")
    _INDEX_AUTHORS[chat_id] = amap
    return amap


def _norm_claim(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())[:400]


async def _index_load_registry(chat_id: int) -> tuple:
    """Реестр сущностей для промпта: (текст-листинг, {id: row}). Компактно: id/имя/тип/алиасы (+суть если влезает)."""
    rows = await db_read(
        "SELECT id, name, entity_type, aliases, canon_summary FROM entities WHERE chat_id=%s ORDER BY id", (chat_id,))
    by_id = {r["id"]: r for r in rows}
    lines = []
    for r in rows:
        al = json.loads(r["aliases"]) if r["aliases"] else []
        head = f"[id={r['id']}] {r['name']} ({r['entity_type']}) алиасы: {', '.join(al) if al else '—'}"
        lines.append(head)
    listing = "\n".join(lines) if lines else "(пусто — сущностей ещё нет)"
    # если реестр огромен — не раздуваем (имена+алиасы уже без сути); грубый предохранитель
    if count_tokens(listing) > 60000:
        listing = "\n".join(lines[:1500]) + f"\n… (+{max(0, len(lines) - 1500)} ещё; показаны первые)"
    return listing, by_id


_INDEX_STAGE1_SYSTEM = (
    "Ты — аналитик истории чата. Из лога извлекаешь СУЩНОСТИ (реальных участников чата и вымышленных "
    "персонажей/лор) и факты о них. Тебе дан РЕЕСТР уже известных сущностей (с их id) и НОВЫЙ БЛОК сообщений "
    "в формате [msg_id] Автор: текст.\n"
    "Верни СТРОГО JSON (без пояснений):\n"
    '{"entities":[{"ref": <id из реестра, если сущность уже там, иначе null>,'
    ' "name": "<основное имя>", "type": "user"|"character",'
    ' "aliases": ["<все прозвища и РАСКРЫТЫЕ тождества>"],'
    ' "canon": [{"claim":"<твёрдый факт о персонаже/вселенной>","evidence":[<msg_id>]}],'
    ' "fanon": [{"claim":"<мнение/отношение участников>","evidence":[<msg_id>]}]}]}\n'
    "Правила: если из беседы следует, что два имени — одна сущность (напр. раскрыто «X это Y»), это ОДНА "
    "сущность, оба имени в aliases. canon — факты вселенной, заявленные как истина (Evidence-based: только то, "
    "что участники утверждают как факт). fanon — их эмоции/оценки/отношение. evidence — реальные msg_id из блока, "
    "откуда взят факт. Извлекай ТОЛЬКО значимые сущности (о ком реально говорят / кто активно участвует), не плоди "
    "запись на каждого случайного автора. Если новых фактов о сущности нет — можно её не возвращать."
)


_INDEX_USERID_RE = re.compile(r"^user(-?\d+)$")


def _index_resolve_tg(name2author: dict, name: str, aliases: list):
    """Определяет tg_user_id участника: по имени/алиасу через name2author, иначе разбирает fallback-имя user{id}."""
    for cand in [name, *(aliases or [])]:
        aid = name2author.get((cand or "").strip().lower())
        if aid:
            return aid
    m = _INDEX_USERID_RE.match((name or "").strip())
    return int(m.group(1)) if m else None


async def _index_apply_entities(chat_id: int, ents: list, name2author: dict):
    """UPSERT сущностей и claim'ов из ответа модели. name2author — {lower(name): author_id} для tg_user_id."""
    if not ents:
        return 0
    touched = 0
    for e in ents:
        if not isinstance(e, dict):
            continue
        name = (e.get("name") or "").strip()
        if not name:
            continue
        etype = "character" if str(e.get("type")).lower().startswith("char") else "user"
        aliases = [a.strip() for a in (e.get("aliases") or []) if isinstance(a, str) and a.strip()]
        try:
            ref = int(e.get("ref")) if str(e.get("ref") or "").isdigit() else None
        except Exception:
            ref = None
        ent_id = None
        dirty_summary = False
        dirty_embedding = False
        if ref and ref > 0:
            rows = await db_read("SELECT id, aliases, tg_user_id FROM entities WHERE chat_id=%s AND id=%s", (chat_id, ref))
            if rows:
                ent_id = rows[0]["id"]
                old = json.loads(rows[0]["aliases"]) if rows[0]["aliases"] else []
                merged = list(dict.fromkeys(old + aliases + [name]))
                if merged != old:
                    await db_write("UPDATE entities SET aliases=%s, embedding=NULL WHERE chat_id=%s AND id=%s",
                                   (json.dumps(merged, ensure_ascii=False), chat_id, ent_id))
                    dirty_embedding = True
                if etype == "user" and rows[0].get("tg_user_id") is None:  # само-лечение привязки на update
                    tg = _index_resolve_tg(name2author, name, merged)
                    if tg:
                        await db_write("UPDATE entities SET tg_user_id=%s WHERE chat_id=%s AND id=%s", (tg, chat_id, ent_id))
        if ent_id is None:  # ищем по имени/алиасу, чтобы не задваивать (снежный ком мог не подставить ref)
            for cand in list(dict.fromkeys([name] + aliases)):
                if not _index_identity_key(cand):
                    continue
                rows = await db_read(
                    "SELECT id, aliases, tg_user_id FROM entities WHERE chat_id=%s AND (name=%s OR aliases LIKE %s) LIMIT 1",
                    (chat_id, cand, f'%"{cand}"%'))
                if rows:
                    ent_id = rows[0]["id"]
                    old = json.loads(rows[0]["aliases"]) if rows[0]["aliases"] else []
                    merged = list(dict.fromkeys(old + aliases + [name]))
                    if merged != old:
                        await db_write("UPDATE entities SET aliases=%s, embedding=NULL WHERE chat_id=%s AND id=%s",
                                       (json.dumps(merged, ensure_ascii=False), chat_id, ent_id))
                        dirty_embedding = True
                    if etype == "user" and rows[0].get("tg_user_id") is None:  # само-лечение привязки на update
                        tg = _index_resolve_tg(name2author, name, merged)
                        if tg:
                            await db_write("UPDATE entities SET tg_user_id=%s WHERE chat_id=%s AND id=%s", (tg, chat_id, ent_id))
                    break
        if ent_id is None:  # новая сущность
            tg = _index_resolve_tg(name2author, name, aliases) if etype == "user" else None
            _, ent_id = await db_write(
                "INSERT INTO entities (chat_id,name,entity_type,tg_user_id,aliases) VALUES (%s,%s,%s,%s,%s)",
                (chat_id, name, etype, tg, json.dumps(list(dict.fromkeys([name] + aliases)), ensure_ascii=False)))
            dirty_summary = True
            dirty_embedding = True
        # claim'ы (дедуп по нормализованному тексту в пределах сущности)
        existing = await db_read("SELECT id, kind, claim, evidence FROM entity_claims WHERE chat_id=%s AND entity_id=%s",
                                 (chat_id, ent_id))
        seen = {(_norm_claim(r["claim"]), r["kind"]): r for r in existing}
        for kind in ("canon", "fanon"):
            for c in (e.get(kind) or []):
                if not isinstance(c, dict):
                    continue
                claim = (c.get("claim") or "").strip()
                if not claim:
                    continue
                ev = [int(x) for x in (c.get("evidence") or []) if isinstance(x, (int, str)) and str(x).isdigit()]
                keyc = (_norm_claim(claim), kind)
                if keyc in seen:  # уже есть — доклеим evidence
                    r = seen[keyc]
                    old_ev = json.loads(r["evidence"]) if r["evidence"] else []
                    new_ev = list(dict.fromkeys(old_ev + ev))
                    if new_ev != old_ev:
                        await db_write("UPDATE entity_claims SET evidence=%s WHERE id=%s",
                                       (json.dumps(new_ev, ensure_ascii=False), r["id"]))
                        dirty_summary = True
                else:
                    await db_write(
                        "INSERT INTO entity_claims (chat_id,entity_id,kind,claim,evidence) VALUES (%s,%s,%s,%s,%s)",
                        (chat_id, ent_id, kind, claim, json.dumps(ev, ensure_ascii=False)))
                    seen[keyc] = {"id": None, "kind": kind, "claim": claim, "evidence": json.dumps(ev)}
                    dirty_summary = True
        if dirty_summary:
            await db_write("DELETE FROM entity_summary_parts WHERE chat_id=%s AND entity_id=%s", (chat_id, ent_id))
            await db_write(
                "UPDATE entities SET canon_summary=NULL, fanon_summary=NULL, embedding=NULL WHERE chat_id=%s AND id=%s",
                (chat_id, ent_id))
            _index_invalidate(chat_id, "entities")
        elif dirty_embedding:
            _index_invalidate(chat_id, "entities")
        touched += 1
    return touched


async def _index_merge_entity_ids(chat_id: int, keep_id: int, drop_id: int):
    if keep_id == drop_id:
        return
    rows = await db_read("SELECT id, name, aliases FROM entities WHERE chat_id=%s AND id IN (%s,%s)",
                         (chat_id, keep_id, drop_id))
    by_id = {r["id"]: r for r in rows}
    if keep_id not in by_id or drop_id not in by_id:
        return
    keep, drop = by_id[keep_id], by_id[drop_id]
    al_keep = json.loads(keep["aliases"]) if keep["aliases"] else []
    al_drop = json.loads(drop["aliases"]) if drop["aliases"] else []
    merged = list(dict.fromkeys(al_keep + al_drop + [keep["name"], drop["name"]]))
    await db_transaction([
        ("UPDATE entities SET aliases=%s, canon_summary=NULL, fanon_summary=NULL, embedding=NULL WHERE chat_id=%s AND id=%s",
         (json.dumps(merged, ensure_ascii=False), chat_id, keep_id)),
        ("UPDATE entity_claims SET entity_id=%s WHERE chat_id=%s AND entity_id=%s", (keep_id, chat_id, drop_id)),
        ("UPDATE relations SET source_id=%s, embedding=NULL WHERE chat_id=%s AND source_id=%s", (keep_id, chat_id, drop_id)),
        ("UPDATE relations SET target_id=%s, embedding=NULL WHERE chat_id=%s AND target_id=%s", (keep_id, chat_id, drop_id)),
        ("UPDATE IGNORE relation_events SET source_id=%s WHERE chat_id=%s AND source_id=%s", (keep_id, chat_id, drop_id)),
        ("UPDATE IGNORE relation_events SET target_id=%s WHERE chat_id=%s AND target_id=%s", (keep_id, chat_id, drop_id)),
        ("DELETE FROM relation_events WHERE chat_id=%s AND (source_id=%s OR target_id=%s)", (chat_id, drop_id, drop_id)),
        ("DELETE FROM relations WHERE chat_id=%s AND source_id=target_id", (chat_id,)),
        ("DELETE FROM relation_events WHERE chat_id=%s AND source_id=target_id", (chat_id,)),
        ("DELETE FROM entity_summary_parts WHERE chat_id=%s AND entity_id IN (%s,%s)", (chat_id, keep_id, drop_id)),
        ("DELETE FROM entities WHERE chat_id=%s AND id=%s", (chat_id, drop_id)),
    ])
    _index_invalidate(chat_id, "entities", "relations")


async def _index_consolidate_exact_entities(chat_id: int) -> int:
    """Консервативная консолидация: сливает только точные пересечения name/alias после нормализации."""
    rows = await db_read("SELECT id, name, aliases FROM entities WHERE chat_id=%s ORDER BY id", (chat_id,))
    owner, merges = {}, []
    for r in rows:
        names = [r["name"]]
        if r["aliases"]:
            names += json.loads(r["aliases"])
        for nm in names:
            key = _index_identity_key(nm)
            if not key:
                continue
            if key in owner and owner[key] != r["id"]:
                merges.append((min(owner[key], r["id"]), max(owner[key], r["id"])))
            else:
                owner[key] = r["id"]
    done, seen = 0, set()
    for keep_id, drop_id in merges:
        if (keep_id, drop_id) in seen:
            continue
        seen.add((keep_id, drop_id))
        await _index_merge_entity_ids(chat_id, keep_id, drop_id)
        done += 1
    if done:
        log("INDEX", f"Консолидация сущностей: слито точных дублей {done}")
    return done


async def _index_stage1_extract_rows(chat_id: int, rows: list, amap: dict, name2author: dict, depth: int = 0) -> tuple:
    """Обрабатывает micro-block; при poison failure дробит и в крайнем случае записывает skipped range."""
    if not rows:
        return 0, 0
    start, end = rows[0]["msg_id"], rows[-1]["msg_id"]
    registry, _ = await _index_load_registry(chat_id)
    user = f"РЕЕСТР (известные сущности):\n{registry}\n\nНОВЫЙ БЛОК:\n" + _index_rows_to_text(rows, amap)
    data = await _index_extract(_INDEX_STAGE1_SYSTEM, user)
    if data and isinstance(data.get("entities"), list):
        touched = await _index_apply_entities(chat_id, data["entities"], name2author)
        return touched, 0
    if len(rows) > INDEX_FAILED_MIN_MESSAGES:
        mid = max(1, len(rows) // 2)
        left_t, left_s = await _index_stage1_extract_rows(chat_id, rows[:mid], amap, name2author, depth + 1)
        right_t, right_s = await _index_stage1_extract_rows(chat_id, rows[mid:], amap, name2author, depth + 1)
        return left_t + right_t, left_s + right_s
    reason = "Stage1 deterministic extraction failure after recursive split"
    await _index_record_failed_range(chat_id, 1, "message", start, end, reason,
                                     {"depth": depth, "text_preview": (rows[0].get("txt") or "")[:500]})
    log("INDEX", f"Stage1 poison message skipped: {start}")
    return 0, 1


async def _index_stage1_dossiers(chat_id: int, progress_cb=None):
    """Снежный ком: блоки текстовых сообщений → V4 Flash → сущности+claim'ы. Чекпоинт по msg_id блока."""
    st = await _idx_get_state(chat_id, 1)
    cursor = int(st["cursor"].get("last_msg_id", 0))
    ents_seen = int(st["stats"].get("blocks", 0))
    skipped_seen = int(st["stats"].get("skipped", 0))
    await _idx_set_state(chat_id, 1, status="running")
    amap = await _index_author_map(chat_id)
    name2author = {}
    for aid, nm in amap.items():
        name2author.setdefault(nm.lower(), aid)
    # детерминированный слой: fallback-имена user{id} для ВСЕХ авторов чата (get_participants мог их не вернуть)
    for r in await db_read("SELECT DISTINCT author_id FROM messages WHERE chat_id=%s AND author_id IS NOT NULL", (chat_id,)):
        name2author.setdefault(f"user{r['author_id']}", r["author_id"])
    log("INDEX", f"Stage1 досье чата {chat_id}: с msg_id>{cursor}")

    blocks_done = ents_seen
    while True:
        if _INDEX_CONTROL.get(chat_id) == "pause":
            await _idx_set_state(chat_id, 1, status="paused")
            return "paused"
        # набираем micro-block до токен/сообщение-бюджета; poison blocks дробятся внутри.
        rows = await db_read(
            "SELECT msg_id, author_id, txt FROM messages WHERE chat_id=%s AND msg_id>%s AND txt IS NOT NULL AND txt<>'' "
            "ORDER BY msg_id ASC LIMIT %s", (chat_id, cursor, INDEX_STAGE1_MICRO_MESSAGES))
        if not rows:
            break
        block_rows, tok, last = [], 0, cursor
        for r in rows:
            author_name = amap.get(r["author_id"], f"user{r['author_id']}" if r.get("author_id") else "?")
            line = f"[{r['msg_id']}] {author_name}: {r['txt']}"
            block_rows.append(r)
            tok += count_tokens(line)
            last = r["msg_id"]
            if tok >= INDEX_STAGE1_MICRO_TOKENS:
                break
        try:
            touched, skipped = await _index_stage1_extract_rows(chat_id, block_rows, amap, name2author)
        except IndexTransientError as e:  # авария провайдера — не двигаем курсор, встаём в error
            await _idx_set_state(chat_id, 1, status="error")
            log("INDEX", f"Stage1 транзиентный сбой на блоке до {last}: {e} — стадия в error, /index resume добёрет")
            return "error"
        cursor = last
        blocks_done += 1
        skipped_seen += skipped
        if blocks_done % INDEX_CONSOLIDATE_EVERY_BLOCKS == 0:
            await _index_consolidate_exact_entities(chat_id)
        await _idx_set_state(chat_id, 1, cursor={"last_msg_id": cursor}, stats={"blocks": blocks_done, "skipped": skipped_seen})
        await db_write(
            "UPDATE index_failed_ranges SET status='resolved' WHERE chat_id=%s AND stage=1 AND status='retrying' AND end_msg_id<=%s",
            (chat_id, cursor))
        if progress_cb:
            cnt = (await db_read("SELECT COUNT(*) c FROM entities WHERE chat_id=%s", (chat_id,)))[0]["c"]
            await progress_cb(f"🧠 Досье: micro {blocks_done} (до id {cursor}) · сущностей {cnt} · skipped {skipped_seen}…")

    # синтез компактных досье canon/fanon из claim'ов
    try:
        sum_res = await _index_summarize_entities(chat_id, progress_cb)
    except IndexTransientError as e:  # провайдер недоступен на саммари — не закрываем стадию
        await _idx_set_state(chat_id, 1, status="error")
        log("INDEX", f"Stage1 саммари: транзиентный сбой ({e}) — стадия в error, /index resume добёрет")
        return "error"
    if sum_res == "paused":
        await _idx_set_state(chat_id, 1, status="paused")
        return "paused"
    await _idx_set_state(chat_id, 1, status="done", stats={"blocks": blocks_done, "skipped": skipped_seen})
    cnt = (await db_read("SELECT COUNT(*) c FROM entities WHERE chat_id=%s", (chat_id,)))[0]["c"]
    log("INDEX", f"Stage1 чата {chat_id}: готово, сущностей {cnt}, блоков {blocks_done}")
    return "done"


_INDEX_SUMM_SYSTEM = (
    "Ты пишешь компактное досье персонажа/участника чата по собранным фактам. Даны CANON-факты (что считается "
    "правдой о нём) и FANON-факты (мнения и отношение участников). Верни СТРОГО JSON: "
    '{"canon":"<2–4 предложения: кто это, ключевой лор/роль, только по фактам>",'
    ' "fanon":"<1–3 предложения: как к нему относятся в чате>"}. Без выдумок сверх фактов; если фактов нет — пустая строка.'
)


async def _index_summarize_entities(chat_id: int, progress_cb=None):
    """Из claim'ов синтезирует entities.canon_summary/fanon_summary. Чекпоинт по entity id."""
    ents = await db_read(
        "SELECT id, name FROM entities WHERE chat_id=%s AND (canon_summary IS NULL OR fanon_summary IS NULL) ORDER BY id",
        (chat_id,))
    sem = asyncio.Semaphore(4)
    total = len(ents)
    done = 0

    async def _one(ent):
        nonlocal done
        claims = await db_read(
            "SELECT kind, claim FROM entity_claims WHERE chat_id=%s AND entity_id=%s ORDER BY id", (chat_id, ent["id"]))
        canon = [c["claim"] for c in claims if c["kind"] == "canon"]
        fanon = [c["claim"] for c in claims if c["kind"] == "fanon"]
        if not canon and not fanon:
            csum = fsum = ""
        else:
            total_body = (f"Персонаж/участник: {ent['name']}\nCANON-факты:\n" + ("\n".join(f"- {c}" for c in canon) or "(нет)")
                          + "\nFANON-факты:\n" + ("\n".join(f"- {c}" for c in fanon) or "(нет)"))
            if len(claims) > INDEX_SUMMARY_MAPREDUCE_MIN_CLAIMS or count_tokens(total_body) > INDEX_SUMMARY_MAPREDUCE_MIN_TOKENS:
                await db_write("DELETE FROM entity_summary_parts WHERE chat_id=%s AND entity_id=%s", (chat_id, ent["id"]))
                canon = await _index_summarize_claim_parts(chat_id, ent["id"], ent["name"], "canon", canon, sem)
                fanon = await _index_summarize_claim_parts(chat_id, ent["id"], ent["name"], "fanon", fanon, sem)
            body = (f"Персонаж/участник: {ent['name']}\nCANON-факты:\n" + ("\n".join(f"- {c}" for c in canon) or "(нет)")
                    + "\nFANON-факты:\n" + ("\n".join(f"- {c}" for c in fanon) or "(нет)"))
            async with sem:
                data = await _index_extract(_INDEX_SUMM_SYSTEM, body, max_tokens=INDEX_SUMMARY_MAX_TOKENS)
            if isinstance(data, dict):
                csum, fsum = (data.get("canon") or "").strip(), (data.get("fanon") or "").strip()
            else:  # фолбэк — просто склейка фактов
                csum = " ".join(canon)[:1500]
                fsum = " ".join(fanon)[:800]
        await db_write("UPDATE entities SET canon_summary=%s, fanon_summary=%s WHERE chat_id=%s AND id=%s",
                       (csum, fsum, chat_id, ent["id"]))
        done += 1

    # обрабатываем чанками, чекпоинтя max id (устойчиво к рестарту)
    for i in range(0, total, 20):
        if _INDEX_CONTROL.get(chat_id) == "pause":
            return "paused"
        chunk = ents[i:i + 20]
        await asyncio.gather(*[_one(e) for e in chunk])
        _index_invalidate(chat_id, "entities")
        if progress_cb:
            await progress_cb(f"📝 Досье-саммари: {min(i + 20, total)}/{total}…")
    return "done"


async def _index_summarize_claim_parts(chat_id: int, entity_id: int, name: str, kind: str, claims: list, sem=None) -> list:
    """Map-step для популярных сущностей: claims → compact part summaries. sem — общий семафор параллелизма
    LLM (чтобы популярная сущность не устроила всплеск вызовов → 429 → Stage 1 в error)."""
    if not claims:
        return []
    out, batch, tok, part_no = [], [], 0, 0

    async def _flush():
        nonlocal batch, tok, part_no
        if not batch:
            return
        part_no += 1
        canon_lines = "\n".join(f"- {c}" for c in batch) if kind == "canon" else "(нет)"
        fanon_lines = "\n".join(f"- {c}" for c in batch) if kind == "fanon" else "(нет)"
        body = f"Персонаж/участник: {name}\nCANON-факты:\n{canon_lines}\nFANON-факты:\n{fanon_lines}"
        if sem is not None:
            async with sem:
                data = await _index_extract(_INDEX_SUMM_SYSTEM, body, max_tokens=INDEX_SUMMARY_MAX_TOKENS)
        else:
            data = await _index_extract(_INDEX_SUMM_SYSTEM, body, max_tokens=INDEX_SUMMARY_MAX_TOKENS)
        if isinstance(data, dict):
            summary = (data.get(kind) or data.get("canon") or data.get("fanon") or "").strip()
        else:
            summary = " ".join(batch)[:1200]
        await db_write(
            """INSERT INTO entity_summary_parts (chat_id,entity_id,kind,part_no,summary,claim_count)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON DUPLICATE KEY UPDATE summary=VALUES(summary), claim_count=VALUES(claim_count)""",
            (chat_id, entity_id, kind, part_no, summary, len(batch)))
        if summary:
            out.append(summary)
        batch, tok = [], 0

    for claim in claims:
        ct = count_tokens(claim)
        if batch and (len(batch) >= INDEX_SUMMARY_CLAIM_BATCH or tok + ct > INDEX_SUMMARY_TOKEN_BATCH):
            await _flush()
        batch.append(claim)
        tok += ct
    await _flush()
    return out


# --- STAGE 2: сцены → граф связей + распознавание медиа ---
_INDEX_MEDIA_SEM = asyncio.Semaphore(1)  # скачиваем фото строго по одному (анти-FloodWait юзербота)

_INDEX_REL_SYSTEM = (
    "Ты — аналитик связей в чате. Дан СПРАВОЧНИК сущностей (имена и алиасы) и одна СЦЕНА диалога в формате "
    "[msg_id] Автор: текст. Верни СТРОГО JSON (без пояснений):\n"
    '{"relations":[{"source":"<имя из справочника>","target":"<имя из справочника>",'
    ' "type":"<тип связи по-русски: глагол/отношение>","polarity":"pos|neg|neutral",'
    ' "summary":"<суть связи или конфликта в этой сцене>","evidence":[<msg_id>]}]}\n'
    "Только связи, ЯВНО проявленные в этой сцене (кто с кем взаимодействует, что заявлено об отношениях). "
    "source и target — строго имена ИЗ СПРАВОЧНИКА; если участник связи не из справочника — пропусти связь. "
    "polarity: pos (тепло/союз/симпатия/примирение), neg (конфликт/вражда/насмешка/ссора), "
    "neutral (родство/роль/структурный факт без оценки). evidence — msg_id из сцены."
)

_INDEX_MEDIA_SYSTEM = (
    "Ты описываешь изображение из чата для базы знаний. Даны СПРАВОЧНИК персонажей и контекст сцены вокруг фото. "
    "Верни СТРОГО JSON:\n"
    '{"description":"<насыщенное фактами описание: что и кто изображён, что происходит, связь со сценой>",'
    ' "characters":[{"name":"<имя из справочника, если узнан>","appearance":"<внешность/визуальные приметы>"}]}\n'
    "Узнавай персонажей из справочника по контексту сцены и подписям. Если узнаваемых нет — characters пустой. "
    "Не выдумывай имена не из справочника."
)


async def _index_relation_registry(chat_id: int, scene_text: str = None) -> tuple:
    """(candidate-listing для prompt, полный {lower(имя/алиас): entity_id} для валидации)."""
    rows = await db_read("SELECT id, name, entity_type, aliases FROM entities WHERE chat_id=%s ORDER BY id", (chat_id,))
    name2id, by_id = {}, {}
    for r in rows:
        by_id[r["id"]] = r
        al = json.loads(r["aliases"]) if r["aliases"] else []
        name2id[r["name"].lower()] = r["id"]
        for a in al:
            name2id.setdefault(a.lower(), r["id"])
    selected = []
    if scene_text:
        low = scene_text.lower()
        exact = []
        for r in rows:
            names = [r["name"]] + (json.loads(r["aliases"]) if r["aliases"] else [])
            if any(_index_identity_key(nm) and _index_identity_key(nm) in low for nm in names):
                exact.append(r["id"])
                if len(exact) >= INDEX_REGISTRY_EXACT_LIMIT:
                    break
        selected += exact
        if exact:
            ph = ",".join(str(int(i)) for i in exact)
            rels = await db_read(
                f"SELECT source_id,target_id,weight FROM relations WHERE chat_id=%s AND status='active' "
                f"AND (source_id IN ({ph}) OR target_id IN ({ph})) ORDER BY weight DESC LIMIT %s",
                (chat_id, INDEX_REGISTRY_NEIGHBOR_LIMIT))
            for rel in rels:
                selected.append(rel["source_id"])
                selected.append(rel["target_id"])
        try:
            n_emb = (await db_read(
                "SELECT COUNT(*) c FROM entities WHERE chat_id=%s AND embedding IS NOT NULL", (chat_id,)))[0]["c"]
            if n_emb:
                qv = await _index_embed_query(scene_text[:4000])
                for h in await _index_vector_search(chat_id, "entities", qv, INDEX_REGISTRY_SEMANTIC_LIMIT):
                    selected.append(h["key"])
        except Exception as e:
            log("INDEX", f"Registry semantic candidates не получились: {e}")
    if not selected:
        selected = [r["id"] for r in rows[:INDEX_REGISTRY_FALLBACK_LIMIT]]
    selected_ids = []
    seen = set()
    for eid in selected:
        if eid in by_id and eid not in seen:
            selected_ids.append(eid)
            seen.add(eid)
    lines = []
    for eid in selected_ids:
        r = by_id[eid]
        al = json.loads(r["aliases"]) if r["aliases"] else []
        tag = "персонаж" if r["entity_type"] == "character" else "участник"
        extra = f" (алиасы: {', '.join(a for a in al if a != r['name'])})" if len(al) > 1 else ""
        lines.append(f"{r['name']} — {tag}{extra}")
    listing = "\n".join(lines) if lines else "(справочник пуст)"
    if count_tokens(listing) > 50000:
        listing = "\n".join(lines[:2000]) + f"\n… (+{max(0, len(lines) - 2000)} ещё)"
    return listing, name2id


async def _index_apply_relations(chat_id: int, rels: list, name2id: dict, scene_date) -> tuple:
    """Темпоральный UPSERT связей. Сентимент-ребро (pos/neg) одно на пару (s→t): смена полярности
    закрывает старое и открывает новое (событие «поссорились/помирились»). neutral — отдельное стойкое ребро.
    Идемпотентность веса — через relation_events с дневным ключом (устойчив к resume/update overlap).
    Возвращает (число применённых, множество затронутых entity_id)."""
    applied, touched = 0, set()
    for rel in rels or []:
        if not isinstance(rel, dict):
            continue
        s = name2id.get(str(rel.get("source", "")).lower().strip())
        t = name2id.get(str(rel.get("target", "")).lower().strip())
        if not s or not t or s == t:
            continue
        touched.add(s)
        touched.add(t)
        pol = str(rel.get("polarity", "neutral")).lower()
        if pol not in ("pos", "neg", "neutral"):
            pol = "neutral"
        rtype = (rel.get("type") or "связь")[:64]
        summ = (rel.get("summary") or "")[:2000]
        ev = [int(x) for x in (rel.get("evidence") or []) if str(x).isdigit()]
        event_key = _index_relation_event_key(chat_id, scene_date, s, t, pol)
        # relation_type в relation_events = pol (не сырой rtype!), чтобы UNIQUE(u_event) дедупил ПО ПОЛЯРНОСТИ:
        # «спорит»/«ругается» в один день не задваивают вес. Настоящий rtype хранится в таблице relations.
        rowcount, _ = await db_write(
            """INSERT IGNORE INTO relation_events
               (chat_id,scene_key,source_id,target_id,relation_type,canonical_type,context_summary,evidence,scene_date)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (chat_id, event_key, s, t, pol, pol, summ, json.dumps(ev, ensure_ascii=False), scene_date))
        if rowcount == 0:
            continue
        if pol == "neutral":
            cond = "AND canonical_type='neutral'"
        else:
            cond = "AND canonical_type IN ('pos','neg')"
        cur = await db_read(
            f"SELECT id, canonical_type, weight, evidence FROM relations WHERE chat_id=%s AND source_id=%s AND target_id=%s "
            f"AND status='active' {cond} ORDER BY id DESC LIMIT 1", (chat_id, s, t))
        if cur:
            row = cur[0]
            if pol != "neutral" and row["canonical_type"] != pol:  # полярность сменилась → закрываем старое ребро
                await db_write("UPDATE relations SET status='closed' WHERE chat_id=%s AND id=%s", (chat_id, row["id"]))
                await db_write(
                    """INSERT INTO relations (chat_id,source_id,target_id,relation_type,canonical_type,context_summary,
                       weight,first_seen,last_seen,status,evidence) VALUES (%s,%s,%s,%s,%s,%s,1,%s,%s,'active',%s)""",
                    (chat_id, s, t, rtype, pol, summ, scene_date, scene_date, json.dumps(ev, ensure_ascii=False)))
                _index_invalidate(chat_id, "relations")
            else:  # то же ребро — усиливаем, обновляем summary/last_seen, доклеиваем evidence
                old_ev = json.loads(row["evidence"]) if row["evidence"] else []
                new_ev = list(dict.fromkeys(old_ev + ev))[:50]
                await db_write(
                    "UPDATE relations SET weight=weight+1, last_seen=%s, relation_type=%s, context_summary=%s, evidence=%s, embedding=NULL WHERE chat_id=%s AND id=%s",
                    (scene_date, rtype, summ, json.dumps(new_ev, ensure_ascii=False), chat_id, row["id"]))
                _index_invalidate(chat_id, "relations")
        else:
            await db_write(
                """INSERT INTO relations (chat_id,source_id,target_id,relation_type,canonical_type,context_summary,
                   weight,first_seen,last_seen,status,evidence) VALUES (%s,%s,%s,%s,%s,%s,1,%s,%s,'active',%s)""",
                (chat_id, s, t, rtype, pol, summ, scene_date, scene_date, json.dumps(ev, ensure_ascii=False)))
        applied += 1
    return applied, touched


async def _index_download_media(msg):
    """Скачивает фото сообщения по одному с паузой, таймаутом и обработкой FloodWait (анти-бан юзербота).
    Семафор держим ТОЛЬКО на само скачивание — во время FloodWait-сна он отпущен (иначе один чат морозил бы медиа другого)."""
    for attempt in range(3):
        try:
            async with _INDEX_MEDIA_SEM:
                raw = await asyncio.wait_for(msg.download_media(bytes), timeout=INDEX_MEDIA_DL_TIMEOUT)
            await asyncio.sleep(random.uniform(*INDEX_MEDIA_PAUSE))
            return raw
        except FloodWaitError as e:  # семафор уже отпущен (вышли из with на исключении) → спим, не блокируя других
            log("INDEX", f"Медиа: FloodWait {e.seconds}с — жду (семафор отпущен)")
            await asyncio.sleep(e.seconds + 1)
        except asyncio.TimeoutError:
            log("INDEX", f"Медиа id={getattr(msg, 'id', '?')}: таймаут {INDEX_MEDIA_DL_TIMEOUT}с — пропускаю")
            return None
        except Exception as e:
            log("INDEX", f"Медиа id={getattr(msg, 'id', '?')} не скачалось: {e}")
            return None
    return None


async def _index_process_media(chat_id: int, image_msgs: list, scene_text: str, registry: str, name2id: dict):
    """Распознаёт картинки сцены медиа-моделью → media_assets + visual-факты сущностям. Возвращает число обработанных."""
    model = get_active_media_model()
    mclient = _client_for_media_model(model)
    if not mclient:
        return 0
    done = 0
    for msg in image_msgs:
        exists = await db_read(
            "SELECT image_description, emb_image FROM media_assets WHERE chat_id=%s AND msg_id=%s",
            (chat_id, msg.id))
        if exists and (exists[0].get("image_description") or "").strip() and exists[0].get("emb_image"):
            done += 1
            continue
        raw = await _index_download_media(msg)
        if not raw:
            continue
        try:
            b64 = base64.b64encode(raw).decode("utf-8")
            user = (f"Справочник персонажей:\n{registry[:4000]}\n\nКонтекст сцены:\n{scene_text[:2500]}")
            resp = await asyncio.to_thread(
                mclient.chat.completions.create, model=model, max_tokens=3000, timeout=90,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": _INDEX_MEDIA_SYSTEM + "\n\n" + user},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}}]}])
            data = _json_from_llm(resp.choices[0].message.content or "")
        except Exception as e:
            log("INDEX", f"Медиа-описание id={msg.id} не удалось: {e}")
            data = None
        if not isinstance(data, dict):
            continue
        desc = (data.get("description") or "").strip()
        ent_ids = []
        for c in (data.get("characters") or []):
            if not isinstance(c, dict):
                continue
            eid = name2id.get(str(c.get("name", "")).lower().strip())
            if not eid:
                continue
            ent_ids.append(eid)
            app = (c.get("appearance") or "").strip()
            if app:  # копим внешность у сущности (капом), + visual-claim с пруфом
                rows = await db_read("SELECT visual_features FROM entities WHERE id=%s", (eid,))
                old = (rows[0]["visual_features"] or "") if rows else ""
                if app.lower() not in old.lower():
                    newvf = (old + " · " + app).strip(" ·")[:2000]
                    await db_write("UPDATE entities SET visual_features=%s, embedding=NULL WHERE chat_id=%s AND id=%s",
                                   (newvf, chat_id, eid))
                    _index_invalidate(chat_id, "entities")
                await db_write(
                    """INSERT INTO entity_claims (chat_id,entity_id,kind,claim,evidence)
                       SELECT %s,%s,'visual',%s,%s FROM DUAL
                       WHERE NOT EXISTS (
                         SELECT 1 FROM entity_claims
                         WHERE chat_id=%s AND entity_id=%s AND kind='visual' AND claim=%s AND evidence=%s
                         LIMIT 1)""",
                    (chat_id, eid, app, json.dumps([msg.id], ensure_ascii=False),
                     chat_id, eid, app, json.dumps([msg.id], ensure_ascii=False)))
        uid, _ = _media_meta(msg)
        # вектор САМОЙ картинки (gemini-embedding-2) считаем здесь, пока байты в руках — иначе Stage 3 качал бы фото повторно
        emb_img = await _index_embed_image(raw)
        await db_write(
            """INSERT INTO media_assets (chat_id,msg_id,file_uid,image_description,entity_ids,emb_image)
               VALUES (%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE image_description=VALUES(image_description),
               entity_ids=VALUES(entity_ids), file_uid=VALUES(file_uid), emb_image=VALUES(emb_image), emb_text=NULL""",
            (chat_id, msg.id, uid, desc, json.dumps(ent_ids, ensure_ascii=False), emb_img))
        _index_invalidate(chat_id, "media_text", "media_image")
        done += 1
    return done


async def _index_stage2_graph(chat_id: int, progress_cb=None):
    """Нарезает сцены (гэп >15 мин или токен-кап), строит граф связей и распознаёт фото. Чекпоинт по msg_id сцены.
    Пишет chat_chunks (enriched_text + мета) с embedding=NULL — векторизует Stage 3."""
    st = await _idx_get_state(chat_id, 2)
    cursor = int(st["cursor"].get("last_msg_id", 0))
    scenes_done = int(st["stats"].get("scenes", 0))
    await _idx_set_state(chat_id, 2, status="running")
    amap = await _index_author_map(chat_id)
    # Выравнивание границ: курсор мог быть отмотан на произвольный msg_id (/index update, retry failed).
    # Снапаем его на реальную границу ранее нарезанной сцены и сносим «хвост» чанков за ней, чтобы
    # перенарезка не плодила перекрывающиеся сцены (u_scene upsert иначе бы промахнулся).
    if cursor > 0:
        snap = await db_read("SELECT MAX(end_msg_id) m FROM chat_chunks WHERE chat_id=%s AND end_msg_id<=%s", (chat_id, cursor))
        aligned = int(snap[0]["m"]) if snap and snap[0]["m"] is not None else 0
        if aligned < cursor:
            log("INDEX", f"Stage2: курсор {cursor} → выравниваю на границу сцены {aligned}, сношу хвост чанков")
            cursor = aligned
        deleted, _ = await db_write("DELETE FROM chat_chunks WHERE chat_id=%s AND end_msg_id>%s", (chat_id, cursor))
        if deleted:
            _index_invalidate(chat_id, "chunks")
            scenes_done = (await db_read("SELECT COUNT(*) c FROM chat_chunks WHERE chat_id=%s", (chat_id,)))[0]["c"]
    ent_count = (await db_read("SELECT COUNT(*) c FROM entities WHERE chat_id=%s", (chat_id,)))[0]["c"]
    log("INDEX", f"Stage2 граф чата {chat_id}: с msg_id>{cursor}, сущностей {ent_count}")

    scene, scene_tok, prev_date = [], 0, None

    async def _write_chunk(sc, scene_text, rels_applied, touched, media_done, failed=False):
        nonlocal scenes_done
        if not sc:
            return
        s_date = sc[-1]["date"]
        ent_ids = sorted(touched)
        meta = {"entities": ent_ids[:60], "relations": rels_applied, "photos": media_done, "failed": bool(failed)}
        enriched = scene_text
        if ent_ids:
            names = await db_read("SELECT name FROM entities WHERE id IN (%s)" %
                                  ",".join(str(int(i)) for i in ent_ids[:30]))
            enriched += "\n[Участники сцены: " + ", ".join(n["name"] for n in names) + "]"
        rowcount, _ = await db_write(
            """INSERT INTO chat_chunks (chat_id,start_msg_id,end_msg_id,scene_date,enriched_text,meta)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON DUPLICATE KEY UPDATE scene_date=VALUES(scene_date), enriched_text=VALUES(enriched_text),
                 meta=VALUES(meta), embedding=NULL""",
            (chat_id, sc[0]["msg_id"], sc[-1]["msg_id"], s_date, enriched[:60000], json.dumps(meta, ensure_ascii=False)))
        if rowcount:
            _index_invalidate(chat_id, "chunks")
        scenes_done += 1

    async def _finalize(sc, depth=0):
        if not sc:
            return
        s_date = sc[-1]["date"]
        lines = [f"[{r['msg_id']}] {amap.get(r['author_id'], 'user' + str(r['author_id'])) }: {r['txt']}"
                 for r in sc if (r["txt"] or "").strip()]
        scene_text = "\n".join(lines)
        registry, name2id = await _index_relation_registry(chat_id, scene_text)
        rels_applied, touched = 0, set()
        failed = False
        if scene_text.strip():
            data = await _index_extract(_INDEX_REL_SYSTEM + "\n\nСПРАВОЧНИК:\n" + registry, "СЦЕНА:\n" + scene_text)
            if data is None or not isinstance(data.get("relations"), list):
                if len(sc) > 1:
                    mid = max(1, len(sc) // 2)
                    await _finalize(sc[:mid], depth + 1)
                    await _finalize(sc[mid:], depth + 1)
                    return
                failed = True
                await _index_record_failed_range(
                    chat_id, 2, "scene", sc[0]["msg_id"], sc[-1]["msg_id"],
                    "Stage2 deterministic relation extraction failure after recursive split",
                    {"depth": depth, "text_preview": scene_text[:500]})
                log("INDEX", f"Stage2 poison scene skipped: {sc[0]['msg_id']}..{sc[-1]['msg_id']}")
            else:
                rels_applied, touched = await _index_apply_relations(chat_id, data["relations"], name2id, s_date)
        # медиа НЕ качаем в Stage 2 (это блокировало граф на часы) — фото обрабатывает отдельная Stage 5
        await _write_chunk(sc, scene_text, rels_applied, touched, 0, failed=failed)

    # --- параллельная экстракция: завершённые сцены копятся в batch → экстрагируются concurrently → применяются по очереди ---
    batch = []

    async def _extract_scene(sc):
        """Параллелизуемая часть: справочник + LLM-экстракция связей. → (scene_text, name2id, data, s_date)."""
        s_date = sc[-1]["date"]
        lines = [f"[{r['msg_id']}] {amap.get(r['author_id'], 'user' + str(r['author_id']))}: {r['txt']}"
                 for r in sc if (r["txt"] or "").strip()]
        scene_text = "\n".join(lines)
        registry, name2id = await _index_relation_registry(chat_id, scene_text)
        data = None
        if scene_text.strip():
            data = await _index_extract(_INDEX_REL_SYSTEM + "\n\nСПРАВОЧНИК:\n" + registry, "СЦЕНА:\n" + scene_text)
        return scene_text, name2id, data, s_date

    async def _apply_scene(sc, scene_text, name2id, data, s_date):
        """Серийная часть: применяет связи + пишет чанк. Poison (data невалиден) → sequential split через _finalize."""
        if scene_text.strip() and (data is None or not isinstance(data.get("relations"), list)):
            await _finalize(sc)  # редкий poison-путь: пере-извлечёт и раздробит/запишет failed
            return
        rels_applied, touched = 0, set()
        if data and isinstance(data.get("relations"), list):
            rels_applied, touched = await _index_apply_relations(chat_id, data["relations"], name2id, s_date)
        await _write_chunk(sc, scene_text, rels_applied, touched, 0, failed=False)

    async def _flush_batch():
        """Экстрагирует batch параллельно, применяет по очереди, чекпоинтит по каждой сцене. IndexTransientError пробрасывается."""
        nonlocal batch
        if not batch:
            return
        sem = asyncio.Semaphore(INDEX_STAGE2_CONCURRENCY)

        async def _one(sc):
            async with sem:
                return await _extract_scene(sc)

        results = await asyncio.gather(*[_one(sc) for sc in batch])  # транзиент пробросится (и отменит остальные)
        for sc, res in zip(batch, results):
            await _apply_scene(sc, *res)
            await _idx_set_state(chat_id, 2, cursor={"last_msg_id": sc[-1]["msg_id"]}, stats={"scenes": scenes_done})
            await db_write(
                "UPDATE index_failed_ranges SET status='resolved' WHERE chat_id=%s AND stage=2 AND status='retrying' AND end_msg_id<=%s",
                (chat_id, sc[-1]["msg_id"]))
        if progress_cb:
            await progress_cb(f"🕸 Граф: {scenes_done} сцен (до id {batch[-1][-1]['msg_id']})…")
        batch = []

    while True:
        if _INDEX_CONTROL.get(chat_id) == "pause":
            await _idx_set_state(chat_id, 2, status="paused")
            return "paused"
        rows = await db_read(
            "SELECT msg_id, `date`, author_id, txt, media_kind FROM messages WHERE chat_id=%s AND msg_id>%s "
            "ORDER BY msg_id ASC LIMIT 2000", (chat_id, cursor))
        if not rows:
            break
        for r in rows:
            gap = (r["date"] - prev_date).total_seconds() if (prev_date and r["date"]) else 0
            should_split = scene and (
                scene_tok >= INDEX_SCENE_TOKEN_CAP
                or gap > INDEX_SCENE_HARD_GAP_SEC
                or (gap > INDEX_SCENE_GAP_SEC and scene_tok >= INDEX_SCENE_MIN_TOKENS)
            )
            if should_split:
                batch.append(scene)
                scene, scene_tok, prev_date = [], 0, None
                if len(batch) >= INDEX_STAGE2_CONCURRENCY:
                    try:
                        await _flush_batch()
                    except IndexTransientError as e:
                        await _idx_set_state(chat_id, 2, status="error")
                        log("INDEX", f"Stage2 транзиентный сбой: {e} — стадия в error")
                        return "error"
            scene.append(r)
            scene_tok += count_tokens(r["txt"] or "")
            prev_date = r["date"]
            cursor = r["msg_id"]
        # окно кончилось — дожимаем завершённые сцены (batch), открытую сцену держим до следующего окна
        try:
            await _flush_batch()
        except IndexTransientError as e:
            await _idx_set_state(chat_id, 2, status="error")
            log("INDEX", f"Stage2 транзиентный сбой: {e} — стадия в error")
            return "error"
    if scene:  # последняя открытая сцена чата
        batch.append(scene)
        try:
            await _flush_batch()
        except IndexTransientError as e:
            await _idx_set_state(chat_id, 2, status="error")
            log("INDEX", f"Stage2 транзиентный сбой на последней сцене: {e} — стадия в error")
            return "error"

    await _idx_set_state(chat_id, 2, status="done", stats={"scenes": scenes_done})
    nrel = (await db_read("SELECT COUNT(*) c FROM relations WHERE chat_id=%s", (chat_id,)))[0]["c"]
    log("INDEX", f"Stage2 чата {chat_id}: готово, сцен {scenes_done}, связей {nrel}")
    return "done"


# --- эмбеддинги (OpenRouter /embeddings) ---
def _sync_embed_texts(texts: list) -> list:
    last_err = None
    for attempt in range(1, INDEX_EMBED_RETRIES + 1):
        try:
            resp = requests.post(f"{OPENROUTER_BASE_URL}/embeddings",
                                 headers={"Authorization": f"Bearer {openrouter_api_key}"},
                                 json={"model": INDEX_EMBED_TEXT_MODEL, "input": texts}, timeout=180)
            resp.raise_for_status()
            break
        except Exception as e:
            last_err = e
            if attempt == INDEX_EMBED_RETRIES:
                raise
            time.sleep(min(30, 2 ** attempt) + random.random())
    if last_err and "resp" not in locals():
        raise last_err
    out = [None] * len(texts)
    for d in resp.json().get("data", []):
        i = d.get("index", 0)
        if 0 <= i < len(texts):
            out[i] = d.get("embedding")
    return out


def _sync_embed_image(raw: bytes) -> list:
    b64 = base64.b64encode(raw).decode("utf-8")
    last_err = None
    for attempt in range(1, INDEX_EMBED_RETRIES + 1):
        try:
            resp = requests.post(f"{OPENROUTER_BASE_URL}/embeddings",
                                 headers={"Authorization": f"Bearer {openrouter_api_key}"},
                                 json={"model": INDEX_EMBED_IMAGE_MODEL, "encoding_format": "float",
                                       "input": [{"content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]},
                                 timeout=120)
            resp.raise_for_status()
            break
        except Exception as e:
            last_err = e
            if attempt == INDEX_EMBED_RETRIES:
                raise
            time.sleep(min(30, 2 ** attempt) + random.random())
    if last_err and "resp" not in locals():
        raise last_err
    return resp.json()["data"][0]["embedding"]


async def _index_embed_texts(texts: list) -> list:
    """Список текстов → список float16-блобов (None на неудачных)."""
    if not texts:
        return []
    try:
        vecs = await asyncio.to_thread(_sync_embed_texts, texts)
    except Exception as e:
        log("INDEX", f"Эмбеддинг текстов не удался: {e}")
        return [None] * len(texts)
    return [_vec_pack(v) if v else None for v in vecs]


async def _index_embed_image(raw: bytes):
    try:
        return _vec_pack(await asyncio.to_thread(_sync_embed_image, raw))
    except Exception as e:
        log("INDEX", f"Эмбеддинг картинки не удался: {e}")
        return None


async def _index_embed_query(text: str, image_space: bool = False):
    """Текст запроса → нормированный np-вектор в нужном пространстве (text-emb-3-small или gemini-emb-2)."""
    model = INDEX_EMBED_IMAGE_MODEL if image_space else INDEX_EMBED_TEXT_MODEL

    def _op():
        resp = requests.post(f"{OPENROUTER_BASE_URL}/embeddings",
                             headers={"Authorization": f"Bearer {openrouter_api_key}"},
                             json={"model": model, "input": [text]}, timeout=60)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    try:
        v = await asyncio.to_thread(_op)
        return _vec_unpack(_vec_pack(v))  # та же нормировка/усечение, что у хранимых
    except Exception as e:
        log("INDEX", f"Эмбеддинг запроса не удался: {e}")
        return None


# --- STAGE 3: векторизация текстов (картинки уже векторизованы в Stage 2) ---
async def _index_vectorize_loop(chat_id, table, key_col, emb_col, textfn, progress_cb, label):
    """Батчами эмбеддит строки с пустым emb_col. Курсор по key_col в пределах прохода → нет вечного цикла
    на сбойных строках; NULL-фильтр = чекпоинт (повтор/resume пропускает уже готовые)."""
    cur, done, failed = 0, 0, 0
    while True:
        if _INDEX_CONTROL.get(chat_id) == "pause":
            return "paused", done
        rows = await db_read(
            f"SELECT * FROM {table} WHERE chat_id=%s AND {emb_col} IS NULL AND {key_col}>%s ORDER BY {key_col} LIMIT %s",
            (chat_id, cur, INDEX_EMBED_BATCH))
        if not rows:
            break
        cur = rows[-1][key_col]
        blobs = await _index_embed_texts([textfn(r) for r in rows])
        for r, blob in zip(rows, blobs):
            if blob is not None:
                await db_write(f"UPDATE {table} SET {emb_col}=%s WHERE chat_id=%s AND {key_col}=%s",
                               (blob, chat_id, r[key_col]))
                done += 1
            else:
                failed += 1
        if progress_cb:
            await progress_cb(f"🔢 Вектора {label}: {done}…")
    if failed:
        log("INDEX", f"Stage3 {label}: {failed} строк остались без вектора — stage не закрываю")
        return "error", done
    return "done", done


async def _index_vectorize_missing_images(chat_id: int, progress_cb=None):
    """Дозаполняет emb_image для media_assets, если Stage 2 временно не получил image embedding."""
    cur, done, failed = 0, 0, []
    while True:
        if _INDEX_CONTROL.get(chat_id) == "pause":
            return "paused", done
        rows = await db_read(
            "SELECT msg_id FROM media_assets WHERE chat_id=%s AND emb_image IS NULL AND msg_id>%s ORDER BY msg_id LIMIT %s",
            (chat_id, cur, min(20, INDEX_EMBED_BATCH)))
        if not rows:
            break
        ids = [r["msg_id"] for r in rows]
        cur = ids[-1]
        try:
            msgs = await client.get_messages(chat_id, ids=ids)
        except Exception as e:
            log("INDEX", f"Stage3 image embeddings: get_messages не удался для {ids[:3]}…: {e}")
            failed += ids
            continue
        msg_by_id = {m.id: m for m in (msgs or []) if getattr(m, "id", None) is not None}
        for mid in ids:
            msg = msg_by_id.get(mid)
            if not _index_is_image_msg(msg):
                failed.append(mid)
                continue
            raw = await _index_download_media(msg)
            blob = await _index_embed_image(raw) if raw else None
            if blob is None:
                failed.append(msg.id)
                continue
            await db_write("UPDATE media_assets SET emb_image=%s WHERE chat_id=%s AND msg_id=%s", (blob, chat_id, msg.id))
            done += 1
        if progress_cb:
            await progress_cb(f"🔢 Вектора картинок: {done}…")
    if failed:
        await _idx_set_state(chat_id, 3, stats={"failed_image_msg_ids": failed[:50]}, status="error")
        log("INDEX", f"Stage3 image embeddings: {len(failed)} картинок остались без emb_image")
        return "error", done
    if done:
        _index_invalidate(chat_id, "media_image")
    return "done", done


async def _index_stage3_vectors(chat_id: int, progress_cb=None):
    """Векторизует досье, связи, сцены и описания фото (text-embedding-3-small). Картинки (emb_image) —
    уже в Stage 2. Каждый текст непустой (fallback-заглушки), чтобы не плодить вечные NULL."""
    await _idx_set_state(chat_id, 3, status="running")

    def _ent_text(r):
        al = json.loads(r["aliases"]) if r["aliases"] else []
        return " | ".join(x for x in [r["name"], ", ".join(al), r.get("canon_summary") or "",
                                      r.get("fanon_summary") or "", r.get("visual_features") or ""] if x).strip() or r["name"]

    targets = [
        ("entities", "id", "embedding", _ent_text, "досье"),
        ("relations", "id", "embedding", lambda r: (r.get("context_summary") or r.get("relation_type") or "связь"), "связи"),
        ("chat_chunks", "id", "embedding", lambda r: (r.get("enriched_text") or "сцена")[:8000], "сцены"),
        ("media_assets", "msg_id", "emb_text", lambda r: (r.get("image_description") or "изображение")[:8000], "фото"),
    ]
    matrix_kind = {
        ("entities", "embedding"): "entities",
        ("relations", "embedding"): "relations",
        ("chat_chunks", "embedding"): "chunks",
        ("media_assets", "emb_text"): "media_text",
    }
    for table, key_col, emb_col, textfn, label in targets:
        res, n = await _index_vectorize_loop(chat_id, table, key_col, emb_col, textfn, progress_cb, label)
        kind = matrix_kind.get((table, emb_col))
        if n and kind:
            _index_invalidate(chat_id, kind)
        if res == "paused":
            return "paused"
        if res == "error":
            await _idx_set_state(chat_id, 3, stats={"failed_label": label}, status="error")
            return "error"
        log("INDEX", f"Stage3 {label}: {n} векторов")
    res_img, n_img = await _index_vectorize_missing_images(chat_id, progress_cb)
    if res_img == "paused":
        return "paused"
    if res_img == "error":
        return "error"
    if n_img:
        log("INDEX", f"Stage3 картинки: {n_img} векторов")
    await _idx_set_state(chat_id, 3, status="done")
    log("INDEX", f"Stage3 чата {chat_id}: готово")
    return "done"


# --- STAGE 4: темпоральные роллап-саммари (RAPTOR-lite: месяц → весь чат) ---
_INDEX_ROLLUP_SYSTEM = (
    "Ты пишешь сжатую сводку периода жизни чата по фрагментам его диалогов/сцен (или по сводкам под-периодов). "
    "Верни СТРОГО JSON: {\"summary\":\"<4–7 предложений: главные темы и события, общая атмосфера/настроение, кто был "
    "активен и вокруг чего, заметные сдвиги>\"}. Только то, что есть в тексте; без выдумок. Пиши по-русски, ёмко."
)


async def _index_summarize_texts(texts: list, title: str) -> str:
    """Map-reduce саммари списка текстов в одну сводку периода (для роллапов). None-safe, с фолбэком."""
    texts = [t for t in texts if (t or "").strip()]
    if not texts:
        return ""
    # map: бьём на токен-батчи, каждый сжимаем; при одном батче — сразу reduce
    batches, cur, tok = [], [], 0
    for t in texts:
        ct = count_tokens(t)
        if cur and tok + ct > INDEX_ROLLUP_TOKEN_BATCH:
            batches.append(cur); cur, tok = [], 0
        cur.append(t); tok += ct
    if cur:
        batches.append(cur)
    parts = []
    for b in batches:
        body = f"Период: {title}\nФрагменты:\n" + "\n---\n".join(x[:4000] for x in b)
        data = await _index_extract(_INDEX_ROLLUP_SYSTEM, body, max_tokens=INDEX_SUMMARY_MAX_TOKENS)
        parts.append((data.get("summary") if isinstance(data, dict) else None) or " ".join(b)[:1200])
    if len(parts) == 1:
        return parts[0][:4000]
    # reduce: сводим частичные сводки в одну
    body = f"Период: {title}\nСводки под-отрезков:\n" + "\n---\n".join(parts)
    data = await _index_extract(_INDEX_ROLLUP_SYSTEM, body, max_tokens=INDEX_SUMMARY_MAX_TOKENS)
    return ((data.get("summary") if isinstance(data, dict) else None) or " ".join(parts))[:4000]


async def _index_stage4_rollups(chat_id: int, progress_cb=None):
    """Строит месячные саммари из chat_chunks + общий саммари чата. Инкрементально: месяц пересобирается,
    только если сменилась сигнатура (count + MAX chunk id). Векторизует роллапы для memory_overview."""
    await _idx_set_state(chat_id, 4, status="running")
    months = await db_read(
        "SELECT DATE_FORMAT(scene_date,'%%Y-%%m') ym, COUNT(*) c, MAX(id) maxid, MIN(scene_date) mn, MAX(scene_date) mx "
        "FROM chat_chunks WHERE chat_id=%s AND scene_date IS NOT NULL GROUP BY ym ORDER BY ym", (chat_id,))
    changed = 0
    for m in months:
        if _INDEX_CONTROL.get(chat_id) == "pause":
            await _idx_set_state(chat_id, 4, status="paused")
            return "paused"
        ym = m["ym"]
        # сигнатура месяца = count + MAX(chunk id): на update чанки удаляются+пересоздаются с новыми id →
        # изменившийся текст сцены (даже при том же числе) даёт другой maxid → пересбор (не только по count).
        sig = f"{int(m['c'])}:{int(m['maxid'] or 0)}"
        prev = await db_read("SELECT meta FROM time_rollups WHERE chat_id=%s AND level=1 AND bucket_key=%s", (chat_id, ym))
        if prev:
            try:
                if (json.loads(prev[0]["meta"]) if prev[0]["meta"] else {}).get("sig") == sig:
                    continue  # месяц не изменился — пропускаем
            except Exception:
                pass
        rows = await db_read(
            "SELECT enriched_text FROM chat_chunks WHERE chat_id=%s AND DATE_FORMAT(scene_date,'%%Y-%%m')=%s "
            "ORDER BY start_msg_id", (chat_id, ym))
        try:
            summ = await _index_summarize_texts([r["enriched_text"] for r in rows], f"месяц {ym}")
        except IndexTransientError as e:
            await _idx_set_state(chat_id, 4, status="error")
            log("INDEX", f"Stage4 транзиентный сбой на {ym}: {e} — стадия в error")
            return "error"
        await db_write(
            """INSERT INTO time_rollups (chat_id,level,bucket_key,period_start,period_end,summary,meta,embedding)
               VALUES (%s,1,%s,%s,%s,%s,%s,NULL)
               ON DUPLICATE KEY UPDATE period_start=VALUES(period_start), period_end=VALUES(period_end),
                 summary=VALUES(summary), meta=VALUES(meta), embedding=NULL""",
            (chat_id, ym, m["mn"], m["mx"], summ, json.dumps({"sig": sig, "chunk_count": int(m["c"])}, ensure_ascii=False)))
        _index_invalidate(chat_id, "rollups")
        changed += 1
        await _idx_set_state(chat_id, 4, cursor={"last_month": ym}, stats={"months": len(months)})
        if progress_cb:
            await progress_cb(f"🗓 Периоды: {ym} ({changed} обновлено)…")
    # общий саммари чата — если месяцы менялись или его ещё нет
    has_all = await db_read("SELECT 1 FROM time_rollups WHERE chat_id=%s AND level=3 AND bucket_key='ALL'", (chat_id,))
    if months and (changed or not has_all):
        msums = await db_read("SELECT summary FROM time_rollups WHERE chat_id=%s AND level=1 ORDER BY bucket_key", (chat_id,))
        try:
            allsum = await _index_summarize_texts([r["summary"] for r in msums], "весь чат")
        except IndexTransientError as e:
            await _idx_set_state(chat_id, 4, status="error")
            log("INDEX", f"Stage4 транзиентный сбой на ALL: {e} — стадия в error")
            return "error"
        await db_write(
            """INSERT INTO time_rollups (chat_id,level,bucket_key,period_start,period_end,summary,meta,embedding)
               VALUES (%s,3,'ALL',%s,%s,%s,%s,NULL)
               ON DUPLICATE KEY UPDATE period_start=VALUES(period_start), period_end=VALUES(period_end),
                 summary=VALUES(summary), meta=VALUES(meta), embedding=NULL""",
            (chat_id, months[0]["mn"], months[-1]["mx"], allsum,
             json.dumps({"months": len(months)}, ensure_ascii=False)))
        _index_invalidate(chat_id, "rollups")
    # векторизуем роллапы (для memory_overview)
    res, n = await _index_vectorize_loop(chat_id, "time_rollups", "id", "embedding",
                                         lambda r: r.get("summary") or "период", progress_cb, "периоды")
    if n:
        _index_invalidate(chat_id, "rollups")
    if res == "paused":
        await _idx_set_state(chat_id, 4, status="paused")
        return "paused"
    if res == "error":
        await _idx_set_state(chat_id, 4, status="error")
        return "error"
    await _idx_set_state(chat_id, 4, status="done", stats={"months": len(months)})
    log("INDEX", f"Stage4 чата {chat_id}: готово, месяцев {len(months)} (обновлено {changed})")
    return "done"


# --- STAGE 5: фото (развязано с графом — идёт последней, не блокирует, полностью возобновляема) ---
async def _index_stage5_media(chat_id: int, progress_cb=None):
    """Скачивает/описывает/эмбеддит картинки по сценам. НЕ в Stage 2 — иначе тысячи фото через FloodWait
    морозили граф на часы. Чекпоинт по chunk id → resume-safe; уже обработанные фото пропускаются (exists)."""
    st = await _idx_get_state(chat_id, 5)
    cursor = int(st["cursor"].get("last_chunk_id", 0))
    done = int(st["stats"].get("photos", 0))
    await _idx_set_state(chat_id, 5, status="running")
    total_ph = (await db_read("SELECT COUNT(*) c FROM messages WHERE chat_id=%s AND media_kind IN (1,2)", (chat_id,)))[0]["c"]
    log("INDEX", f"Stage5 фото чата {chat_id}: с chunk_id>{cursor} (всего фото ~{total_ph})")
    while True:
        if _INDEX_CONTROL.get(chat_id) == "pause":
            await _idx_set_state(chat_id, 5, status="paused")
            return "paused"
        chunks = await db_read(
            "SELECT id, start_msg_id, end_msg_id, enriched_text FROM chat_chunks WHERE chat_id=%s AND id>%s ORDER BY id LIMIT 30",
            (chat_id, cursor))
        if not chunks:
            break
        for ch in chunks:
            img_ids = [r["msg_id"] for r in await db_read(
                "SELECT msg_id FROM messages WHERE chat_id=%s AND msg_id BETWEEN %s AND %s AND media_kind IN (1,2) ORDER BY msg_id",
                (chat_id, ch["start_msg_id"], ch["end_msg_id"]))]
            if img_ids:
                scene_text = ch["enriched_text"] or ""
                registry, name2id = await _index_relation_registry(chat_id, scene_text)
                try:
                    image_msgs = []
                    for i in range(0, len(img_ids), 100):
                        msgs = await client.get_messages(chat_id, ids=img_ids[i:i + 100])
                        image_msgs += [m for m in msgs if _index_is_image_msg(m)]
                    done += await _index_process_media(chat_id, image_msgs, scene_text, registry, name2id)
                except Exception as e:
                    log("INDEX", f"Stage5: медиа сцены {ch['id']} не обработалось: {e}")
            cursor = ch["id"]
            await _idx_set_state(chat_id, 5, cursor={"last_chunk_id": cursor}, stats={"photos": done})
            if progress_cb:
                await progress_cb(f"🖼 Фото: обработано {done}/~{total_ph}…")
    # добиваем emb_image, что не получились инлайн в process_media (Stage 3 их не застал — медиа теперь после него)
    res_img, n_img = await _index_vectorize_missing_images(chat_id, progress_cb)
    if n_img:
        _index_invalidate(chat_id, "media_image")
    if res_img == "paused":
        await _idx_set_state(chat_id, 5, status="paused")
        return "paused"
    # эмбеддинг описаний фото (emb_text): Stage 3 их не застал (медиа теперь после него)
    res, n = await _index_vectorize_loop(chat_id, "media_assets", "msg_id", "emb_text",
                                         lambda r: (r.get("image_description") or "изображение")[:8000], progress_cb, "фото-описания")
    if n:
        _index_invalidate(chat_id, "media_text")
    if res == "paused":
        await _idx_set_state(chat_id, 5, status="paused")
        return "paused"
    if res == "error" or res_img == "error":
        await _idx_set_state(chat_id, 5, status="error")
        return "error"
    await _idx_set_state(chat_id, 5, status="done", stats={"photos": done})
    log("INDEX", f"Stage5 чата {chat_id}: готово, фото {done}")
    return "done"


# --- поиск по векторам (numpy-косинус; матрицы кэшируются per chat) ---
_INDEX_MATRIX: dict = {}   # {(chat_id, kind): {"mat": ndarray, "ids": [...], "extra": [...], "n": int}}
_INDEX_HNSW: dict = {}     # optional ANN cache {(chat_id, kind): {"idx": hnsw, "ids": [...], "extra": [...], "n": int}}
# kind → (table, emb_col, key_col, доп.поля-для-сниппета)
_INDEX_KINDS = {
    "entities":    ("entities", "embedding", "id", "name, entity_type, canon_summary, fanon_summary, visual_features"),
    "relations":   ("relations", "embedding", "id", "source_id, target_id, relation_type, context_summary, status"),
    "chunks":      ("chat_chunks", "embedding", "id", "start_msg_id, end_msg_id, enriched_text"),
    "media_text":  ("media_assets", "emb_text", "msg_id", "msg_id, image_description, entity_ids"),
    "media_image": ("media_assets", "emb_image", "msg_id", "msg_id, image_description, entity_ids"),
    "rollups":     ("time_rollups", "embedding", "id", "level, bucket_key, summary, period_start, period_end"),
}


async def _index_load_matrix(chat_id: int, kind: str) -> dict:
    """Матрица векторов kind для чата с ленивым кэшем и досинхронизацией по числу строк."""
    table, emb_col, key_col, extra = _INDEX_KINDS[kind]
    ck = (chat_id, kind)
    n_now = await _index_count_ok(chat_id, kind)
    cached = _INDEX_MATRIX.get(ck)
    if cached and cached["n"] == n_now:
        return cached
    rows = await db_read(
        f"SELECT {key_col} AS _k, {emb_col} AS _emb, {extra} FROM {table} WHERE chat_id=%s AND {emb_col} IS NOT NULL ORDER BY {key_col}",
        (chat_id,))
    if rows:
        mat = _np.vstack([_np.frombuffer(r["_emb"], dtype=_np.float16).astype(_np.float32) for r in rows])
    else:
        mat = _np.zeros((0, INDEX_EMBED_DIM), dtype=_np.float32)
    obj = {"mat": mat, "ids": [r["_k"] for r in rows],
           "extra": [{k: v for k, v in r.items() if k not in ("_emb", "_k")} for r in rows], "n": n_now}
    _INDEX_MATRIX[ck] = obj
    return obj


def _index_topk(mat, qvec, k: int) -> list:
    """[(row_index, score)] топ-k по косинусу (векторы нормированы → скалярное произведение)."""
    if qvec is None or getattr(mat, "shape", (0,))[0] == 0:
        return []
    sims = mat @ qvec
    k = min(k, sims.shape[0])
    idx = _np.argpartition(-sims, k - 1)[:k] if k < sims.shape[0] else _np.arange(sims.shape[0])
    idx = idx[_np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


async def _index_vector_search(chat_id: int, kind: str, qvec, top_n: int = 8) -> list:
    """Топ-N совпадений: [{score, key, ...доп.поля}]."""
    if qvec is None:
        return []
    table, emb_col, key_col, extra = _INDEX_KINDS[kind]
    n_now = await _index_count_ok(chat_id, kind)  # кэш — не сканим БД на каждый поиск
    if n_now > INDEX_MATRIX_CACHE_MAX_ROWS:
        if index_use_hnsw and _hnswlib is not None:
            h = await _index_load_hnsw(chat_id, kind, n_now)
            if h:
                return _index_hnsw_search(h, qvec, top_n)
        return await _index_vector_search_stream(chat_id, kind, qvec, top_n)
    m = await _index_load_matrix(chat_id, kind)
    hits = _index_topk(m["mat"], qvec, top_n)
    out = []
    for ri, score in hits:
        item = {"score": round(score, 4), "key": m["ids"][ri]}
        item.update(m["extra"][ri])
        out.append(item)
    return out


async def _index_load_hnsw(chat_id: int, kind: str, n_now: int = None):
    table, emb_col, key_col, extra = _INDEX_KINDS[kind]
    ck = (chat_id, kind)
    if n_now is None:
        n_now = await _index_count_ok(chat_id, kind)
    cached = _INDEX_HNSW.get(ck)
    if cached and cached["n"] == n_now:
        return cached
    if not n_now:
        return None
    try:
        rows = await db_read(
            f"SELECT {key_col} AS _k, {emb_col} AS _emb, {extra} FROM {table} WHERE chat_id=%s AND {emb_col} IS NOT NULL ORDER BY {key_col}",
            (chat_id,))
        mat = _np.vstack([_np.frombuffer(r["_emb"], dtype=_np.float16).astype(_np.float32) for r in rows])
        idx = _hnswlib.Index(space="cosine", dim=INDEX_EMBED_DIM)
        idx.init_index(max_elements=len(rows), ef_construction=100, M=16)
        idx.add_items(mat, _np.arange(len(rows)))
        idx.set_ef(min(100, max(10, len(rows))))
        obj = {"idx": idx, "ids": [r["_k"] for r in rows],
               "extra": [{k: v for k, v in r.items() if k not in ("_emb", "_k")} for r in rows], "n": n_now}
        _INDEX_HNSW[ck] = obj
        return obj
    except Exception as e:
        log("INDEX", f"HNSW build failed ({kind}): {e} — fallback stream search")
        _INDEX_HNSW.pop(ck, None)
        return None


def _index_hnsw_search(obj: dict, qvec, top_n: int) -> list:
    if qvec is None or not obj or not obj.get("ids"):
        return []
    k = min(top_n, len(obj["ids"]))
    labels, distances = obj["idx"].knn_query(qvec.reshape(1, -1), k=k)
    out = []
    for lab, dist in zip(labels[0], distances[0]):
        ri = int(lab)
        item = {"score": round(1.0 - float(dist), 4), "key": obj["ids"][ri]}
        item.update(obj["extra"][ri])
        out.append(item)
    return out


async def _index_vector_search_stream(chat_id: int, kind: str, qvec, top_n: int = 8) -> list:
    """Top-k по большим индексам без полной загрузки kind в память процесса."""
    table, emb_col, key_col, extra = _INDEX_KINDS[kind]
    cur, best = 0, []
    while True:
        rows = await db_read(
            f"SELECT {key_col} AS _k, {emb_col} AS _emb, {extra} FROM {table} "
            f"WHERE chat_id=%s AND {emb_col} IS NOT NULL AND {key_col}>%s ORDER BY {key_col} LIMIT %s",
            (chat_id, cur, INDEX_SEARCH_DB_BATCH))
        if not rows:
            break
        cur = rows[-1]["_k"]
        mat = _np.vstack([_np.frombuffer(r["_emb"], dtype=_np.float16).astype(_np.float32) for r in rows])
        sims = mat @ qvec
        for i, score in enumerate(sims):
            r = rows[i]
            extra_item = {k: v for k, v in r.items() if k not in ("_emb", "_k")}
            best.append((float(score), r["_k"], extra_item))
        if len(best) > top_n * 4:
            best.sort(key=lambda x: x[0], reverse=True)
            best = best[:top_n]
    best.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, key, extra_item in best[:top_n]:
        item = {"score": round(score, 4), "key": key}
        item.update(extra_item)
        out.append(item)
    return out


# --- инструменты памяти для /ask (агентный цикл) ---
INDEX_MEMORY_TOOLS = [
    {"type": "function", "function": {
        "name": "memory_search",
        "description": "Ищет в проиндексированной памяти чата (её строит команда /index): сцены-диалоги из прошлого, "
                       "досье персонажей и участников, связи между ними. Используй для вопросов про историю чата, "
                       "лор, персонажей, прошлые события и споры, которых нет в текущем контексте переписки.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Что ищем, своими словами (смысловой поиск)."},
            "kind": {"type": "string", "enum": ["all", "scenes", "dossiers", "relations"],
                     "description": "Где искать: scenes — диалоги, dossiers — досье, relations — связи, all — везде (по умолчанию)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "memory_overview",
        "description": "Высокоуровневое ОБОБЩЕНИЕ по памяти чата для ТЕМАТИЧЕСКИХ и ГЛОБАЛЬНЫХ вопросов — «какая обычно "
                       "атмосфера», «как в чате относятся к X», «кто главные фигуры вокруг темы Y», «общая динамика/"
                       "настроения». Отвечает по СЖАТЫМ досье и ОБОБЩЁННЫМ связям (без сырых диалогов) — экономит контекст "
                       "и даёт картину целиком. Для точечных фактов, конкретных реплик и «что именно сказали» — бери "
                       "memory_search, а не это.",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "Тема или аспект для обобщения, своими словами."}},
            "required": ["topic"]}}},
    {"type": "function", "function": {
        "name": "memory_entity",
        "description": "Полное досье сущности (участник или вымышленный персонаж) по имени или алиасу: canon-факты "
                       "(что считается правдой), fanon (мнение чата) и связи с другими. Используй для вопросов "
                       "«кто такой X», «какие у X отношения с Y».",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Имя или прозвище персонажа/участника."}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "memory_media",
        "description": "Находит и ПЕРЕСЫЛАЕТ в чат фото из истории. Два режима: по текстовому описанию (напр. «та "
                       "картинка со спора про меч», «арт с Заей»), либо ПО ВИЗУАЛЬНОМУ СХОДСТВУ с приложенной "
                       "картинкой (visual=true — «найди похожие арты на это фото», работает если к запросу приложено "
                       "изображение). Используй, когда просят скинуть/найти картинку/фото/арт из прошлого или похожие на данную.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Описание искомого фото своими словами (для текстового поиска)."},
            "visual": {"type": "boolean", "description": "true — искать по СХОДСТВУ с приложенной к запросу картинкой, а не по тексту."},
            "count": {"type": "integer", "description": "Сколько фото переслать, 1–3 (по умолчанию 1)."}}, "required": []}}},
]


def _idx_snip(text, n=180):
    return re.sub(r"\s+", " ", (text or "")).strip()[:n]


async def _index_chat_entity(chat_id):
    try:
        return await client.get_entity(chat_id)
    except Exception:
        return None


async def _index_tool_search(chat_id: int, query: str, kind: str = "all") -> str:
    if not (query or "").strip():
        return "Пустой запрос."
    qv = await _index_embed_query(query)
    if qv is None:
        return "Не удалось векторизовать запрос."
    ent = await _index_chat_entity(chat_id)
    kinds = {"scenes": ["chunks"], "dossiers": ["entities"], "relations": ["relations"]}.get(kind, ["chunks", "entities", "relations"])
    out = []
    best = 0.0
    for k in kinds:
        for h in await _index_vector_search(chat_id, k, qv, 5):
            best = max(best, h["score"])
            if h["score"] < INDEX_SEARCH_FLOOR:
                continue
            if k == "chunks":
                link = (f" {build_msg_link(ent, h['start_msg_id'])}" if ent and h.get("start_msg_id") else "")
                out.append(f"• [сцена {h['score']}] {_idx_snip(h.get('enriched_text'), 220)}{link}")
            elif k == "entities":
                out.append(f"• [досье {h['score']}] {h['name']} ({h['entity_type']}): {_idx_snip(h.get('canon_summary'), 160)}")
            elif k == "relations":
                nm = await db_read("SELECT id, name FROM entities WHERE id IN (%s,%s)", (h["source_id"], h["target_id"]))
                n = {r["id"]: r["name"] for r in nm}
                st = "" if h.get("status") == "active" else " (в прошлом)"
                out.append(f"• [связь {h['score']}] {n.get(h['source_id'], '?')} → {n.get(h['target_id'], '?')}: "
                           f"{h.get('relation_type') or ''}{st} — {_idx_snip(h.get('context_summary'), 140)}")
    # Corrective RAG: явно сигналим модели о слабой/пустой выдаче, чтобы она переспросила или пошла в веб,
    # а не отвечала уверенно по нерелевантному совпадению.
    if not out:
        note = f" (лучшее совпадение score {best:.2f})" if best else ""
        return (f"В памяти ничего релевантного не нашёл{note}. Переформулируй запрос другими словами "
                f"(конкретнее: имена, о чём именно спор/событие) или, если вопрос про внешний мир, используй web_search.")
    if best < INDEX_SEARCH_CONFIDENT:
        out.append(f"⚠️ Совпадения слабые (макс score {best:.2f}) — возможно, точного ответа в памяти нет. "
                   f"Если это не то, переспроси другими словами или проверь web_search; не выдавай догадку за факт.")
    return "\n".join(out)


async def _index_tool_overview(chat_id: int, topic: str) -> str:
    """LightRAG high-level: тематическое обобщение ТОЛЬКО из саммари (сжатые досье + обобщённые связи, без сырых сцен).
    Роутер = выбор модели между этим (тематика/картина в целом) и memory_search (точечные факты/реплики)."""
    if not (topic or "").strip():
        return "Пустая тема."
    qv = await _index_embed_query(topic)
    if qv is None:
        return "Не удалось векторизовать запрос."
    blocks = []
    # 1) причастные лица/персонажи — по сжатым досье
    ebits = []
    for h in await _index_vector_search(chat_id, "entities", qv, 8):
        if h["score"] < INDEX_SEARCH_FLOOR:
            continue
        s = _idx_snip(h.get("canon_summary") or h.get("fanon_summary") or "", 220)
        if s:
            ebits.append(f"— {h['name']} ({'персонаж' if h.get('entity_type') == 'character' else 'участник'}): {s}")
    if ebits:
        blocks.append("Причастные (сжатые досье):\n" + "\n".join(ebits[:6]))
    # 2) обобщённые связи по теме
    rels = [h for h in await _index_vector_search(chat_id, "relations", qv, 8) if h["score"] >= INDEX_SEARCH_FLOOR]
    if rels:
        need = {h["source_id"] for h in rels} | {h["target_id"] for h in rels}
        nm = await db_read("SELECT id, name FROM entities WHERE id IN (%s)" % ",".join(str(int(i)) for i in need))
        n2 = {r["id"]: r["name"] for r in nm}
        rbits = []
        for h in rels[:6]:
            st = "" if h.get("status") == "active" else " (в прошлом)"
            rbits.append(f"— {n2.get(h['source_id'], '?')} ↔ {n2.get(h['target_id'], '?')}: "
                         f"{h.get('relation_type') or ''}{st} — {_idx_snip(h.get('context_summary'), 160)}")
        if rbits:
            blocks.append("Связи по теме:\n" + "\n".join(rbits))
    # 3) темпоральная динамика — сводки периодов (RAPTOR-lite): «как менялось / общая динамика»
    tbits = []
    for h in await _index_vector_search(chat_id, "rollups", qv, 4):
        if h["score"] < INDEX_SEARCH_FLOOR:
            continue
        per = "весь период чата" if h.get("level") == 3 else f"месяц {h.get('bucket_key')}"
        s = _idx_snip(h.get("summary"), 300)
        if s:
            tbits.append(f"— [{per}] {s}")
    if tbits:
        blocks.append("Динамика по периодам:\n" + "\n".join(tbits[:4]))
    if not blocks:
        return (f"По теме «{topic}» обобщённой памяти не нашёл. Уточни тему или возьми memory_search "
                f"для точечного поиска по диалогам.")
    return ("Высокоуровневое обобщение по памяти чата (по досье и связям, без сырых диалогов — "
            "картина в целом, не дословные цитаты):\n\n" + "\n\n".join(blocks))


async def _index_entity_report(chat_id: int, ent: dict) -> str:
    """Текстовое досье сущности для модели (факты + связи с именами)."""
    aliases = json.loads(ent["aliases"]) if ent["aliases"] else []
    parts = [f"{ent['name']} ({'персонаж' if ent['entity_type'] == 'character' else 'участник'})"]
    if [a for a in aliases if a != ent["name"]]:
        parts.append("Алиасы: " + ", ".join(a for a in aliases if a != ent["name"]))
    if (ent.get("canon_summary") or "").strip():
        parts.append("Canon: " + ent["canon_summary"].strip())
    if (ent.get("fanon_summary") or "").strip():
        parts.append("Мнение чата (fanon): " + ent["fanon_summary"].strip())
    if (ent.get("visual_features") or "").strip():
        parts.append("Внешность: " + ent["visual_features"].strip())
    claims = await db_read("SELECT kind, claim FROM entity_claims WHERE chat_id=%s AND entity_id=%s ORDER BY id",
                           (chat_id, ent["id"]))
    for kind, title in (("canon", "Факты"), ("fanon", "Мнения")):
        ck = [c["claim"] for c in claims if c["kind"] == kind][:8]
        if ck:
            parts.append(f"{title}: " + "; ".join(ck))
    rels = await db_read(
        "SELECT source_id, target_id, relation_type, canonical_type, status FROM relations "
        "WHERE chat_id=%s AND (source_id=%s OR target_id=%s) ORDER BY status, weight DESC LIMIT 20",
        (chat_id, ent["id"], ent["id"]))
    if rels:
        need = {r["source_id"] for r in rels} | {r["target_id"] for r in rels}
        nm = await db_read("SELECT id, name FROM entities WHERE id IN (%s)" % ",".join(str(int(i)) for i in need))
        n = {r["id"]: r["name"] for r in nm}
        cur = [f"{n.get(r['target_id'] if r['source_id'] == ent['id'] else r['source_id'], '?')} ({r['relation_type']})"
               for r in rels if r["status"] == "active"]
        past = [f"{n.get(r['target_id'] if r['source_id'] == ent['id'] else r['source_id'], '?')} ({r['relation_type']}, в прошлом)"
                for r in rels if r["status"] != "active"]
        if cur:
            parts.append("Связи сейчас: " + ", ".join(cur[:12]))
        if past:
            parts.append("Бывшие связи: " + ", ".join(past[:8]))
    return "\n".join(parts)


async def _index_tool_entity(chat_id: int, name: str) -> str:
    ent = await _index_find_entity(chat_id, name)
    if not ent:
        return f"Сущность «{name}» в памяти чата не найдена."
    return await _index_entity_report(chat_id, ent)


async def _index_asker_brief(chat_id: int, tg_id: int) -> str:
    """Компактная справка о СПРАШИВАЮЩЕМ (persona-A): по tg_user_id находит его сущность в графе и
    отдаёт имя + сжатые canon/fanon + топ активных связей. None, если участник не в памяти или про него
    ничего содержательного нет (только имя — не тащим). Кладётся в конец user-контента (не в систему — prompt-кэш)."""
    if not tg_id:
        return None
    rows = await db_read(
        "SELECT id, name, canon_summary, fanon_summary, visual_features FROM entities "
        "WHERE chat_id=%s AND tg_user_id=%s ORDER BY id LIMIT 1", (chat_id, tg_id))
    if not rows:
        return None
    e = rows[0]
    bits = []
    if (e.get("canon_summary") or "").strip():
        bits.append("О нём: " + _idx_snip(e["canon_summary"], 300))
    if (e.get("fanon_summary") or "").strip():
        bits.append("Мнение чата о нём: " + _idx_snip(e["fanon_summary"], 200))
    if (e.get("visual_features") or "").strip():
        bits.append("Внешность: " + _idx_snip(e["visual_features"], 160))
    rels = await db_read(
        "SELECT source_id, target_id, relation_type FROM relations WHERE chat_id=%s AND status='active' "
        "AND (source_id=%s OR target_id=%s) ORDER BY weight DESC LIMIT 5", (chat_id, e["id"], e["id"]))
    if rels:
        need = {r["source_id"] for r in rels} | {r["target_id"] for r in rels}
        nm = await db_read("SELECT id, name FROM entities WHERE id IN (%s)" % ",".join(str(int(i)) for i in need))
        n2 = {r["id"]: r["name"] for r in nm}
        pairs = [f"{n2.get(r['target_id'] if r['source_id'] == e['id'] else r['source_id'], '?')} ({r['relation_type']})"
                 for r in rels]
        if pairs:
            bits.append("Связи: " + ", ".join(pairs))
    if not bits:  # только имя, без фактов — незачем шуметь
        return None
    return (f"[Справка о спрашивающем из памяти чата — участник известен как «{e['name']}». "
            f"Учитывай, только если это релевантно вопросу; не притягивай насильно.]\n" + " ".join(bits))


async def _index_tool_media(chat_id: int, query: str, count: int = 1, visual: bool = False, query_image: bytes = None) -> str:
    """Поиск фото по описанию (текст → emb_text) или ПО КАРТИНКЕ (visual=true, приложенное/реплай-фото →
    emb_image, «найди похожие арты»). Найденное пересылается в чат."""
    count = max(1, min(3, count))
    if visual and query_image:
        qv = await _index_embed_query_image(query_image)
        space = "media_image"
    else:
        if not (query or "").strip():
            return "Пустой запрос."
        qv = await _index_embed_query(query)
        space = "media_text"
    if qv is None:
        return "Не удалось векторизовать запрос."
    all_hits = await _index_vector_search(chat_id, space, qv, 6)
    best = max((h["score"] for h in all_hits), default=0.0)
    hits = [h for h in all_hits if h["score"] >= INDEX_SEARCH_FLOOR][:count]
    if not hits:
        note = f" (лучшее score {best:.2f})" if best else ""
        return f"Подходящих фото в памяти не нашёл{note}. Опиши искомое иначе — что на фото, кто, какое событие."
    msg_ids = [h["msg_id"] for h in hits]
    try:
        await client.forward_messages(chat_id, msg_ids, chat_id)
    except Exception as e:
        return f"Нашёл фото (msg {msg_ids}), но не смог переслать: {e}"
    return "Переслал в чат: " + "; ".join(_idx_snip(h.get("image_description"), 90) for h in hits)


async def _index_embed_query_image(raw: bytes):
    """Картинка запроса → нормированный np-вектор в пространстве картинок (gemini-embedding-2)."""
    try:
        v = await asyncio.to_thread(_sync_embed_image, raw)
        return _vec_unpack(_vec_pack(v))
    except Exception as e:
        log("INDEX", f"Эмбеддинг картинки-запроса не удался: {e}")
        return None


async def _index_media_count(chat_id: int) -> int:
    """Сколько фото в памяти чата пригодны для поиска (emb_text проставлен). 0 → чат не индексирован."""
    if _index_available():
        return 0
    try:
        return (await db_read("SELECT COUNT(*) c FROM media_assets WHERE chat_id=%s AND emb_text IS NOT NULL", (chat_id,)))[0]["c"]
    except Exception:
        return 0


async def _index_chat_indexed(chat_id: int) -> bool:
    """True, если у чата есть проиндексированные сущности (для подтягивания досье/референсов в /gen и /ask)."""
    if _index_available():
        return False
    try:
        return (await db_read("SELECT COUNT(*) c FROM entities WHERE chat_id=%s", (chat_id,)))[0]["c"] > 0
    except Exception:
        return False


async def _gen_index_candidates(chat_id: int, prompt: str, limit: int = GEN_INDEX_REF_MAX) -> tuple:
    """Для /gen: смысловой поиск референсов и контекста по ВСЕЙ проиндексированной истории (не только N последних).
    Возвращает (catalog_items, context_text): фото-кандидаты в формате каталога /gen + текст-контекст из досье
    (внешность релевантных персонажей — ключ к узнаваемым референсам)."""
    if not chat_id or _index_available() or not (prompt or "").strip():
        return [], None
    qv = await _index_embed_query(prompt)
    if qv is None:
        return [], None
    # релевантные фото по описаниям (по всей истории)
    media_hits = [h for h in await _index_vector_search(chat_id, "media_text", qv, limit * 2) if h["score"] >= 0.2][:limit]
    # релевантные персонажи → их внешность в контекст промптера
    ent_hits = [h for h in await _index_vector_search(chat_id, "entities", qv, 5) if h["score"] >= 0.25]
    ctx_lines = []
    for h in ent_hits:
        vf = (h.get("visual_features") or "").strip()
        cs = (h.get("canon_summary") or "").strip()
        desc = vf or cs
        if desc:
            ctx_lines.append(f"- {h['name']}: {_idx_snip(desc, 200)}")
    ctx = ("Из памяти чата — релевантные персонажи (можно опереться на их облик для узнаваемых референсов):\n"
           + "\n".join(ctx_lines)) if ctx_lines else None

    items = []
    if media_hits:
        mids = [h["msg_id"] for h in media_hits]
        desc_by_mid = {h["msg_id"]: (h.get("image_description") or "") for h in media_hits}
        try:
            msgs = []
            for i in range(0, len(mids), 100):
                msgs += await client.get_messages(chat_id, ids=mids[i:i + 100])
        except Exception as e:
            log("GEN", f"Индекс-референсы: get_messages не удался: {e}")
            msgs = []

        async def _dl(m):
            if not (m and getattr(m, "photo", None)):
                return None
            try:
                raw = await m.download_media(bytes)
            except Exception:
                return None
            if not raw:
                return None
            return {"idx": 0, "mid": m.id, "bytes": raw, "thumb": await _downscale_img(raw),
                    "caption": (m.raw_text or "").strip(), "desc": desc_by_mid.get(m.id) or None,
                    "from_owner": bool(getattr(m, "out", False)), "from_index": True}
        try:
            res = await asyncio.wait_for(asyncio.gather(*[_dl(m) for m in msgs]), timeout=GEN_CATALOG_TIMEOUT)
            items = [it for it in res if it]
        except asyncio.TimeoutError:
            log("GEN", "Индекс-референсы: скачивание превысило тайм-бюджет")
    log("GEN", f"Индекс-референсы: {len(items)} фото + {len(ctx_lines)} досье из памяти по запросу «{_idx_snip(prompt, 50)}»")
    return items, ctx


# --- оркестратор пайплайна ---
async def _index_pipeline(chat_id: int, status_msg):
    """Прогоняет этапы по порядку, уважая паузу. Пока реализован Stage 0 (дамп);
    этапы 1–3 (досье/граф/вектора) подключаются в следующих фазах."""
    async def upd(text):
        try:
            await status_msg.edit(text)
        except Exception:
            pass
    try:
        await _index_ensure_ddl()
        # Stage 0 — дамп (если ещё не done)
        st0 = await _idx_get_state(chat_id, 0)
        if st0["status"] != "done":
            res = await _index_stage0_dump(chat_id, progress_cb=upd)
            if res == "paused":
                await upd("⏸ Индексация на паузе (Stage 0 — дамп). `/index resume` — продолжить.")
                return
            if res == "floodwait":
                await upd("⏳ Telegram притормозил выгрузку (FloodWait). `/index resume` позже — дожму остаток.")
                return
        st0 = await _idx_get_state(chat_id, 0)
        n = int(st0["stats"].get("dumped", 0))
        # Stage 1 — досье
        st1 = await _idx_get_state(chat_id, 1)
        if st1["status"] != "done":
            await upd(f"✅ Дамп: {n} сообщений. 🧠 Строю досье (может занять долго)…")
            res1 = await _index_stage1_dossiers(chat_id, progress_cb=upd)
            if res1 == "paused":
                await upd("⏸ Индексация на паузе (Stage 1 — досье). `/index resume` — продолжить.")
                return
            if res1 == "error":
                await upd("❌ Индексация остановлена на Stage 1: LLM не вернула валидный JSON. `/index resume` попробует тот же блок ещё раз.")
                return
        cnt = (await db_read("SELECT COUNT(*) c FROM entities WHERE chat_id=%s", (chat_id,)))[0]["c"]
        # Stage 2 — граф связей + медиа
        st2 = await _idx_get_state(chat_id, 2)
        if st2["status"] != "done":
            await upd(f"✅ Досье: {cnt} сущностей. 🕸 Строю граф связей и разбираю фото (долго — фото по одному)…")
            res2 = await _index_stage2_graph(chat_id, progress_cb=upd)
            if res2 == "paused":
                await upd("⏸ Индексация на паузе (Stage 2 — граф). `/index resume` — продолжить.")
                return
            if res2 == "error":
                await upd("❌ Индексация остановлена на Stage 2: LLM не вернула валидные связи. `/index resume` попробует сцену ещё раз.")
                return
        nrel = (await db_read("SELECT COUNT(*) c FROM relations WHERE chat_id=%s AND status='active'", (chat_id,)))[0]["c"]
        nmedia = (await db_read("SELECT COUNT(*) c FROM media_assets WHERE chat_id=%s", (chat_id,)))[0]["c"]
        # Stage 3 — векторизация
        st3 = await _idx_get_state(chat_id, 3)
        if st3["status"] != "done":
            await upd(f"✅ Граф: {nrel} связей, {nmedia} фото. 🔢 Векторизую для поиска…")
            res3 = await _index_stage3_vectors(chat_id, progress_cb=upd)
            if res3 == "paused":
                await upd("⏸ Индексация на паузе (Stage 3 — вектора). `/index resume` — продолжить.")
                return
            if res3 == "error":
                await upd("❌ Индексация остановлена на Stage 3: часть embeddings не получена. `/index resume` повторит оставшиеся строки.")
                return
        # Stage 4 — темпоральные роллап-саммари (месяц → весь чат), для тематических вопросов «как менялось»
        st4 = await _idx_get_state(chat_id, 4)
        if st4["status"] != "done":
            await upd("✅ Вектора готовы. 🗓 Собираю сводки по месяцам (для вопросов «как менялось со временем»)…")
            res4 = await _index_stage4_rollups(chat_id, progress_cb=upd)
            if res4 == "paused":
                await upd("⏸ Индексация на паузе (Stage 4 — сводки периодов). `/index resume` — продолжить.")
                return
            if res4 == "error":
                await upd("❌ Индексация остановлена на Stage 4: LLM не дала сводку периода. `/index resume` повторит.")
                return
        # Stage 5 — фото (последней; текстовый граф уже готов и ищется, фото доливаются в фоне)
        st5 = await _idx_get_state(chat_id, 5)
        if st5["status"] != "done":
            await upd("✅ Граф, вектора и сводки готовы — поиск по тексту уже работает. 🖼 Дообрабатываю фото (долго, по одному)…")
            res5 = await _index_stage5_media(chat_id, progress_cb=upd)
            if res5 == "paused":
                await upd("⏸ Индексация на паузе (Stage 5 — фото). `/index resume` — продолжить. Текстовый поиск уже работает.")
                return
            if res5 == "error":
                await upd("⚠️ Часть фото не векторизовалась (Stage 5). `/index resume` дожмёт. Текстовый поиск уже работает.")
                return
        skipped = await _index_failed_count(chat_id, "skipped")
        skip_note = (f"\n⚠️ Пропущено poison-диапазонов: {skipped}. "
                     f"Смотри `/index failed`, повтор: `/index retry failed`.") if skipped else ""
        nmedia = (await db_read("SELECT COUNT(*) c FROM media_assets WHERE chat_id=%s", (chat_id,)))[0]["c"]
        await upd(f"🎉 Индексация завершена: {n} сообщений · {cnt} сущностей · {nrel} связей · {nmedia} фото."
                  f"{skip_note}\nДосье: `/entity show <имя>`. Поиск в `/ask` уже подключён для владельца.")
    except Exception as e:
        log("INDEX", f"Пайплайн чата {chat_id} упал: {e}")
        traceback.print_exc()
        for stg in (0, 1, 2, 3, 4, 5):
            s = await _idx_get_state(chat_id, stg)
            if s["status"] != "done":
                await _idx_set_state(chat_id, stg, status="error")
                break
        await upd(f"❌ Индексация упала: {e}\nСостояние сохранено — `/index resume` продолжит с чекпоинта.")
    finally:
        _INDEX_CONTROL.pop(chat_id, None)
        _INDEX_TASKS.pop(chat_id, None)


async def _index_preflight(event) -> str:
    """Оценка объёма перед запуском: сколько сообщений/фото и грубая цена/время."""
    chat_id = event.chat_id
    try:
        total = (await client.get_messages(chat_id, limit=0)).total or 0
    except Exception as e:
        return f"⚠️ Не смог оценить объём чата: {e}"
    try:
        photos = (await client.get_messages(chat_id, limit=0, filter=InputMessagesFilterPhotos)).total or 0
    except Exception:
        photos = 0
    st0 = await _idx_get_state(chat_id, 0)
    dumped = int(st0["stats"].get("dumped", 0))
    # грубая оценка: экстракция ~$0.30/1M вх.токенов·~30 токенов/сообщение; фото vision ~$0.0002; эмбеддинги копейки
    est_tok = total * 30
    est_cost = est_tok / 1_000_000 * 0.30 + photos * 0.0002
    done_note = f" (уже выкачано {dumped})" if dumped else ""
    return (f"🗂 **Индексация чата** — оценка перед запуском:\n"
            f"• Сообщений: ~**{total}**{done_note}\n"
            f"• Фото: ~**{photos}**\n"
            f"• Ориентир стоимости полного прохода: **~${est_cost:.2f}** (экстракция+фото+вектора)\n"
            f"• Время: от нескольких минут до часов (зависит от объёма и лимитов Telegram)\n\n"
            f"Запусти: `/index go` · статус: `/index status` · пауза/продолжить: `/index pause` / `/index resume`")


async def _index_update_rewind_cursor(chat_id: int) -> int:
    rows = await db_read("SELECT MAX(msg_id) max_id, MAX(`date`) max_date FROM messages WHERE chat_id=%s", (chat_id,))
    if not rows or not rows[0]["max_id"]:
        return 0
    max_id = int(rows[0]["max_id"])
    candidates = [max_id]
    by_count = await db_read(
        "SELECT msg_id FROM messages WHERE chat_id=%s AND msg_id<=%s ORDER BY msg_id DESC LIMIT %s",
        (chat_id, max_id, INDEX_UPDATE_OVERLAP_MESSAGES))
    if by_count:
        candidates.append(int(by_count[-1]["msg_id"]))
    max_date = rows[0].get("max_date")
    if isinstance(max_date, str):
        try:
            max_date = datetime.strptime(max_date, "%Y-%m-%d %H:%M:%S")
        except Exception:
            max_date = None
    if max_date:
        cutoff = max_date - timedelta(hours=INDEX_UPDATE_OVERLAP_HOURS)
        by_time = await db_read(
            "SELECT MIN(msg_id) m FROM messages WHERE chat_id=%s AND `date`>=%s",
            (chat_id, cutoff.strftime("%Y-%m-%d %H:%M:%S")))
        if by_time and by_time[0].get("m"):
            candidates.append(int(by_time[0]["m"]))
    return max(0, min(candidates) - 1)


@client.on(events.NewMessage(pattern=r"^[./]index(?:\s+(go|status|pause|resume|stop|update))?\s*$"))
async def index_command(event):
    if await _slash_for_other_bot(event):
        return
    if not event.out:
        return  # только владелец
    sub = (event.pattern_match.group(1) or "").lower()
    chat_id = event.chat_id
    reason = _index_available()
    if reason:
        await event.reply(f"⚠️ /index недоступен: {reason}")
        return
    try:
        await _index_ensure_ddl()  # таблицы должны существовать до любого чтения состояния (даже preflight)
    except Exception as e:
        await event.reply(f"❌ Не подключиться к базе индексации: {e}")
        return

    if sub in ("", "status"):
        if sub == "":
            await event.reply(await _index_preflight(event))
            return
        # status
        running = chat_id in _INDEX_TASKS
        parts = ["📊 **Статус индексации этого чата:**"]
        for stg, label in ((0, "Дамп"), (1, "Досье"), (2, "Граф связей"), (3, "Вектора"),
                           (4, "Сводки периодов"), (5, "Фото")):
            s = await _idx_get_state(chat_id, stg)
            if s["status"]:
                extra = ""
                if stg == 0 and s["stats"].get("dumped"):
                    extra = f" · {s['stats']['dumped']} сообщ."
                elif stg == 4 and s["stats"].get("months"):
                    extra = f" · {s['stats']['months']} мес."
                elif stg == 5 and s["stats"].get("photos"):
                    extra = f" · {s['stats']['photos']} фото"
                parts.append(f"• Stage {stg} {label}: {s['status']}{extra}")
        skipped = await _index_failed_count(chat_id, "skipped")
        retrying = await _index_failed_count(chat_id, "retrying")
        if skipped or retrying:
            parts.append(f"• Failed ranges: skipped {skipped} · retrying {retrying} (`/index failed`)")
        if len(parts) == 1:
            parts.append("• ещё не запускалась — `/index go`")
        parts.append(f"\n{'🟢 сейчас работает в фоне' if running else '⚪️ фоновая задача не активна'}")
        await event.reply("\n".join(parts))
        return

    if sub == "go":
        if chat_id in _INDEX_TASKS:
            await event.reply("🟢 Индексация этого чата уже идёт. `/index status` — прогресс.")
            return
        _INDEX_CONTROL[chat_id] = "run"
        status_msg = await event.reply("🚀 Запускаю индексацию в фоне… `/index status` — прогресс.")
        _INDEX_TASKS[chat_id] = asyncio.create_task(_index_pipeline(chat_id, status_msg))
        return

    if sub == "update":  # догнать новые сообщения: снимаем «done» со стадий (курсоры остаются) и прогоняем пайплайн
        if chat_id in _INDEX_TASKS:
            await event.reply("🟢 Индексация уже идёт — дождись её, потом `/index update`.")
            return
        st0 = await _idx_get_state(chat_id, 0)
        if st0["status"] is None:
            await event.reply("📭 Чат ещё не индексировался. Запусти `/index go`.")
            return
        for stg in (0, 1, 2, 3, 4, 5):
            s = await _idx_get_state(chat_id, stg)
            if s["status"] == "done":
                if stg in (1, 2):
                    rewind = await _index_update_rewind_cursor(chat_id)
                    await _idx_set_state(chat_id, stg, cursor={"last_msg_id": rewind}, status="running")
                else:
                    await _idx_set_state(chat_id, stg, status="running")
        _INDEX_CONTROL[chat_id] = "run"
        status_msg = await event.reply("🔄 Догоняю новые сообщения (досье · граф · вектора)…")
        _INDEX_TASKS[chat_id] = asyncio.create_task(_index_pipeline(chat_id, status_msg))
        return

    if sub == "pause":
        if chat_id not in _INDEX_TASKS:
            await event.reply("⚪️ Сейчас нечего ставить на паузу (фоновая задача не активна).")
            return
        _INDEX_CONTROL[chat_id] = "pause"
        await event.reply("⏸ Ставлю на паузу на ближайшем чекпоинте…")
        return

    if sub in ("resume", "stop"):
        if sub == "stop":
            _INDEX_CONTROL[chat_id] = "pause"
            t = _INDEX_TASKS.get(chat_id)
            if t:
                await event.reply("🛑 Останавливаю на ближайшем чекпоинте (прогресс сохранён).")
            else:
                await event.reply("⚪️ Фоновая задача не активна.")
            return
        if chat_id in _INDEX_TASKS:
            await event.reply("🟢 Уже работает.")
            return
        _INDEX_CONTROL[chat_id] = "run"
        status_msg = await event.reply("▶️ Продолжаю индексацию с последнего чекпоинта…")
        _INDEX_TASKS[chat_id] = asyncio.create_task(_index_pipeline(chat_id, status_msg))
        return


@client.on(events.NewMessage(pattern=r"^[./]index\s+failed(?:\s+([123]))?\s*$"))
async def index_failed_command(event):
    if await _slash_for_other_bot(event) or not event.out:
        return
    reason = _index_available()
    if reason:
        await event.reply(f"⚠️ /index недоступен: {reason}")
        return
    await _index_ensure_ddl()
    chat_id = event.chat_id
    stage = event.pattern_match.group(1)
    rows = await _index_failed_rows(chat_id, int(stage) if stage else None, "skipped", 30)
    if not rows:
        await event.reply("✅ Skipped failed ranges нет.")
        return
    lines = ["⚠️ **Skipped failed ranges:**"]
    for r in rows:
        lines.append(f"• Stage {r['stage']} {r['unit_type']} {r['start_msg_id']}..{r['end_msg_id']} "
                     f"attempts={r['attempts']} — {_idx_snip(r.get('reason'), 120)}")
    lines.append("\nПовтор: `/index retry failed` или `/index retry failed 1|2`.")
    await send_long(chat_id, "\n".join(lines), parse_mode="md", reply_to=event.id)


@client.on(events.NewMessage(pattern=r"^[./]index\s+retry\s+failed(?:\s+([12]))?\s*$"))
async def index_retry_failed_command(event):
    if await _slash_for_other_bot(event) or not event.out:
        return
    reason = _index_available()
    if reason:
        await event.reply(f"⚠️ /index недоступен: {reason}")
        return
    await _index_ensure_ddl()
    chat_id = event.chat_id
    if chat_id in _INDEX_TASKS:
        await event.reply("🟢 Индексация уже идёт — дождись завершения.")
        return
    stage_arg = event.pattern_match.group(1)
    rows = await _index_failed_rows(chat_id, int(stage_arg) if stage_arg else None, "skipped", 500)
    if not rows:
        await event.reply("✅ Skipped failed ranges для повтора нет.")
        return
    min_stage = min(int(r["stage"]) for r in rows)
    min_start = min(int(r["start_msg_id"]) for r in rows)
    if stage_arg:
        min_stage = int(stage_arg)
    await db_write(
        "UPDATE index_failed_ranges SET status='retrying' WHERE chat_id=%s AND status='skipped' AND stage>=%s",
        (chat_id, min_stage))
    rewind = max(0, min_start - 1)
    if min_stage <= 1:
        await _idx_set_state(chat_id, 1, cursor={"last_msg_id": rewind}, status="running")
    await _idx_set_state(chat_id, 2, cursor={"last_msg_id": rewind}, status="running")
    await _idx_set_state(chat_id, 3, status="running")
    # переобработка сцен меняет chunks → сводки периодов (Stage 4, по сигнатуре пересоберёт затронутые месяцы)
    # и фото-контекст (Stage 5, cursor→0 перечешет чанки, готовые фото пропустит) должны актуализироваться
    await _idx_set_state(chat_id, 4, status="running")
    await _idx_set_state(chat_id, 5, cursor={"last_chunk_id": 0}, status="running")
    _INDEX_CONTROL[chat_id] = "run"
    status_msg = await event.reply(f"🔁 Повторяю failed ranges со Stage {min_stage} от msg>{rewind}…")
    _INDEX_TASKS[chat_id] = asyncio.create_task(_index_pipeline(chat_id, status_msg))


_INDEX_EVAL_TEMPLATE = [
    {"kind": "entity", "query": "кто такой Тостер", "expected": {"name": "Тостер"}},
    {"kind": "scene", "query": "спор про меч", "expected": {"msg_ids": [12345]}},
    {"kind": "relation", "query": "почему Иван спорил с Олегом", "expected": {"source_id": 1, "target_id": 2}},
    {"kind": "media_text", "query": "арт с мечом", "expected": {"msg_id": 12345}}
]


def _index_eval_match(kind: str, hit: dict, expected: dict) -> bool:
    if not hit:
        return False
    if kind == "entity":
        eid = expected.get("entity_id") or expected.get("id")
        nm = _index_norm_name(expected.get("name") or "")
        return (eid and int(hit.get("key")) == int(eid)) or (nm and _index_norm_name(hit.get("name")) == nm)
    if kind == "scene":
        ids = expected.get("msg_ids") or []
        return any(int(hit.get("start_msg_id") or 0) <= int(mid) <= int(hit.get("end_msg_id") or 0) for mid in ids)
    if kind == "relation":
        sid, tid = expected.get("source_id"), expected.get("target_id")
        return sid and tid and int(hit.get("source_id") or 0) == int(sid) and int(hit.get("target_id") or 0) == int(tid)
    if kind in ("media_text", "media_image"):
        mid = expected.get("msg_id") or expected.get("media_msg_id")
        return mid and int(hit.get("msg_id") or hit.get("key") or 0) == int(mid)
    return False


async def _index_eval_run(chat_id: int, cases: list) -> dict:
    kind_map = {"entity": "entities", "scene": "chunks", "relation": "relations",
                "media_text": "media_text", "media_image": "media_image"}
    details, top1 = [], 0
    top3 = 0
    latencies = []
    for case in cases:
        t0 = time.time()
        kind = case.get("kind")
        query = case.get("query") or ""
        expected = case.get("expected") or {}
        space = kind_map.get(kind)
        if not space or not query:
            details.append({"case": case, "error": "bad kind/query"})
            continue
        qv = await _index_embed_query(query, image_space=(kind == "media_image"))
        hits = await _index_vector_search(chat_id, space, qv, 3)
        latencies.append(time.time() - t0)
        h1 = bool(hits and _index_eval_match(kind, hits[0], expected))
        h3 = any(_index_eval_match(kind, h, expected) for h in hits[:3])
        top1 += int(h1)
        top3 += int(h3)
        details.append({"kind": kind, "query": query, "top1": h1, "top3": h3,
                        "hits": [{"key": h.get("key"), "score": h.get("score")} for h in hits]})
    total = len([c for c in cases if c.get("kind") in kind_map])
    skipped = await _index_failed_count(chat_id, "skipped")
    return {"total": total, "top1": top1, "top3": top3,
            "top1_rate": round(top1 / total, 3) if total else 0,
            "top3_rate": round(top3 / total, 3) if total else 0,
            "avg_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else 0,
            "skipped_ranges": skipped, "details": details}


def _index_eval_report_text(result: dict) -> str:
    return (f"📊 **Index eval:** cases {result.get('total', 0)} · "
            f"top1 {result.get('top1_rate', 0)} · top3 {result.get('top3_rate', 0)} · "
            f"avg {result.get('avg_latency_sec', 0)}s · skipped ranges {result.get('skipped_ranges', 0)}")


@client.on(events.NewMessage(pattern=r"^[./]index\s+eval\s+(template|run|report)\s*$"))
async def index_eval_command(event):
    if await _slash_for_other_bot(event) or not event.out:
        return
    reason = _index_available()
    if reason:
        await event.reply(f"⚠️ /index недоступен: {reason}")
        return
    await _index_ensure_ddl()
    chat_id = event.chat_id
    action = event.pattern_match.group(1)
    if action == "template":
        await send_long(chat_id, "```json\n" + json.dumps(_INDEX_EVAL_TEMPLATE, ensure_ascii=False, indent=2) + "\n```",
                        parse_mode="md", reply_to=event.id)
        return
    if action == "report":
        rows = await db_read("SELECT result_json FROM index_eval_runs WHERE chat_id=%s ORDER BY id DESC LIMIT 1", (chat_id,))
        if not rows:
            await event.reply("📭 Eval ещё не запускался. Создай `index_eval_cases.json` и запусти `/index eval run`.")
            return
        result = json.loads(rows[0]["result_json"]) if rows[0]["result_json"] else {}
        await send_long(chat_id, _index_eval_report_text(result) + "\n```json\n" +
                        json.dumps(result.get("details", []), ensure_ascii=False, indent=2)[:6000] + "\n```",
                        parse_mode="md", reply_to=event.id)
        return
    if not os.path.exists(INDEX_EVAL_CASES_PATH):
        await send_long(chat_id, f"📭 Нет `{INDEX_EVAL_CASES_PATH}`. Шаблон:\n```json\n" +
                        json.dumps(_INDEX_EVAL_TEMPLATE, ensure_ascii=False, indent=2) + "\n```",
                        parse_mode="md", reply_to=event.id)
        return
    with open(INDEX_EVAL_CASES_PATH, "r", encoding="utf-8") as fh:
        cases = json.load(fh)
    if not isinstance(cases, list):
        await event.reply("❌ `index_eval_cases.json` должен быть JSON-массивом cases.")
        return
    status = await event.reply("📊 Запускаю eval памяти…")
    result = await _index_eval_run(chat_id, cases)
    await db_write("INSERT INTO index_eval_runs (chat_id,cases_json,result_json) VALUES (%s,%s,%s)",
                   (chat_id, json.dumps(cases, ensure_ascii=False), json.dumps(result, ensure_ascii=False)))
    await status.edit(_index_eval_report_text(result))


async def _index_find_entity(chat_id: int, query: str):
    """Ищет сущность по имени или алиасу (точное → префикс → подстрока/алиас). Возвращает row|None."""
    q = query.strip()
    rows = await db_read("SELECT * FROM entities WHERE chat_id=%s AND name=%s LIMIT 1", (chat_id, q))
    if rows:
        return rows[0]
    rows = await db_read(
        "SELECT * FROM entities WHERE chat_id=%s AND (name LIKE %s OR aliases LIKE %s) ORDER BY CHAR_LENGTH(name) LIMIT 1",
        (chat_id, q + "%", f'%"{q}"%'))
    if rows:
        return rows[0]
    rows = await db_read("SELECT * FROM entities WHERE chat_id=%s AND name LIKE %s ORDER BY CHAR_LENGTH(name) LIMIT 1",
                         (chat_id, f"%{q}%"))
    return rows[0] if rows else None


@client.on(events.NewMessage(pattern=r"^[./]entity\s+show\s+(.+)$"))
async def entity_show_command(event):
    if await _slash_for_other_bot(event):
        return
    if not event.out:
        return
    reason = _index_available()
    if reason:
        await event.reply(f"⚠️ /entity недоступен: {reason}")
        return
    try:
        await _index_ensure_ddl()
    except Exception as e:
        await event.reply(f"❌ Не подключиться к базе индексации: {e}")
        return
    chat_id = event.chat_id
    query = event.pattern_match.group(1).strip()
    ent = await _index_find_entity(chat_id, query)
    if not ent:
        await event.reply(f"🔍 Не нашёл сущность «{query}» в этом чате. Сначала проиндексируй: `/index go`.")
        return
    aliases = json.loads(ent["aliases"]) if ent["aliases"] else []
    type_label = "🎭 персонаж" if ent["entity_type"] == "character" else "👤 участник"
    parts = [f"**{ent['name']}** — {type_label}"]
    if aliases and aliases != [ent["name"]]:
        parts.append(f"_алиасы:_ {', '.join(a for a in aliases if a != ent['name'])}")
    if (ent.get("canon_summary") or "").strip():
        parts.append(f"\n📖 **Canon:** {ent['canon_summary'].strip()}")
    if (ent.get("fanon_summary") or "").strip():
        parts.append(f"💬 **Fanon (мнение чата):** {ent['fanon_summary'].strip()}")
    if (ent.get("visual_features") or "").strip():
        parts.append(f"🖼 **Внешность:** {ent['visual_features'].strip()}")
    # топ-факты с evidence-ссылками
    claims = await db_read(
        "SELECT kind, claim, evidence FROM entity_claims WHERE chat_id=%s AND entity_id=%s ORDER BY id", (chat_id, ent["id"]))
    try:
        chat_ent = await event.get_chat()
    except Exception:
        chat_ent = None
    for kind, title in (("canon", "📌 Факты (canon)"), ("fanon", "💭 Мнения (fanon)")):
        ck = [c for c in claims if c["kind"] == kind][:6]
        if not ck:
            continue
        parts.append(f"\n{title}:")
        for c in ck:
            ev = json.loads(c["evidence"]) if c["evidence"] else []
            link = ""
            if chat_ent and ev:
                try:
                    link = f" [↗]({build_msg_link(chat_ent, ev[0])})"
                except Exception:
                    link = ""
            parts.append(f"• {c['claim']}{link}")
    # связи из графа (активные и закрытые) — Stage 2
    id2name = {}
    rels = await db_read(
        """SELECT source_id, target_id, relation_type, canonical_type, context_summary, status, weight
           FROM relations WHERE chat_id=%s AND (source_id=%s OR target_id=%s) ORDER BY status, weight DESC LIMIT 20""",
        (chat_id, ent["id"], ent["id"]))
    if rels:
        need = {r["source_id"] for r in rels} | {r["target_id"] for r in rels}
        nm = await db_read("SELECT id, name FROM entities WHERE id IN (%s)" % ",".join(str(int(i)) for i in need))
        id2name = {r["id"]: r["name"] for r in nm}
        pol_emoji = {"pos": "💚", "neg": "💢", "neutral": "▫️"}
        active = [r for r in rels if r["status"] == "active"]
        closed = [r for r in rels if r["status"] != "active"]
        if active:
            parts.append("\n🔗 **Связи:**")
            for r in active[:10]:
                other = id2name.get(r["target_id"] if r["source_id"] == ent["id"] else r["source_id"], "?")
                arrow = "→" if r["source_id"] == ent["id"] else "←"
                parts.append(f"{pol_emoji.get(r['canonical_type'], '▫️')} {arrow} **{other}** — {r['relation_type']} "
                             f"(×{int(r['weight'])})")
        if closed:
            parts.append("\n🕯 _Бывшие связи (изменились со временем):_")
            for r in closed[:5]:
                other = id2name.get(r["target_id"] if r["source_id"] == ent["id"] else r["source_id"], "?")
                parts.append(f"• {other} — {r['relation_type']} (было)")
    n_claims = len(claims)
    parts.append(f"\n_id {ent['id']} · фактов: {n_claims} · связей: {len(rels)}_")
    await send_long(chat_id, "\n".join(parts), parse_mode="md", reply_to=event.id)


@client.on(events.NewMessage(pattern=r"^[./]entity\s+relink\s*$"))
async def entity_relink_command(event):
    """Бэкфилл tg_user_id: привязывает несопоставленных user-сущностей к участникам чата по author_id."""
    if await _slash_for_other_bot(event):
        return
    if not event.out:
        return
    reason = _index_available()
    if reason:
        await event.reply(f"⚠️ /entity недоступен: {reason}")
        return
    try:
        await _index_ensure_ddl()
    except Exception as e:
        await event.reply(f"❌ Не подключиться к базе индексации: {e}")
        return
    chat_id = event.chat_id

    def _strip(s):
        return re.sub(r"[^\w]+", "", s or "", flags=re.UNICODE).lower()

    amap = await _index_author_map(chat_id)           # {author_id: display}
    exact, norm = {}, {}                              # display.lower()→id, нормализованный display→id
    for aid, nm in amap.items():
        exact.setdefault((nm or "").lower(), aid)
        k = _strip(nm)
        if k:
            norm.setdefault(k, aid)
    for r in await db_read("SELECT DISTINCT author_id FROM messages WHERE chat_id=%s AND author_id IS NOT NULL", (chat_id,)):
        exact.setdefault(f"user{r['author_id']}", r["author_id"])

    rows = await db_read(
        "SELECT id, name, aliases FROM entities WHERE chat_id=%s AND entity_type='user' AND tg_user_id IS NULL", (chat_id,))
    total, linked = len(rows), 0
    for r in rows:
        cands = [r["name"]] + (json.loads(r["aliases"]) if r["aliases"] else [])
        tg = None
        for c in cands:
            tg = exact.get((c or "").strip().lower()) or norm.get(_strip(c))
            if tg:
                break
        if not tg:
            m = _INDEX_USERID_RE.match((r["name"] or "").strip())
            if m:
                tg = int(m.group(1))
        if not tg:
            continue
        disp = amap.get(tg)
        if disp and _INDEX_USERID_RE.match((r["name"] or "").strip()):  # уродливое user{id} → реальный ник
            await db_write("UPDATE entities SET tg_user_id=%s, name=%s, embedding=NULL WHERE chat_id=%s AND id=%s",
                           (tg, disp, chat_id, r["id"]))
        else:
            await db_write("UPDATE entities SET tg_user_id=%s WHERE chat_id=%s AND id=%s", (tg, chat_id, r["id"]))
        linked += 1
    if linked:
        _index_invalidate(chat_id, "entities")
    tail = (f" Осталось {total - linked} (модель переименовала в непохожее имя — при желании поправь `/entity`)."
            if total - linked else "")
    await event.reply(f"🔗 Привязка участников: **{linked}** из {total} несопоставленных получили tg_user_id.{tail}")


@client.on(events.NewMessage(pattern=r"^[./]entity\s+(merge|rename|alias|split)\s+(.+)$"))
async def entity_admin_command(event):
    """Правка графа: merge <id1> <id2> · rename <id> <новое имя> · alias <id> <алиас> · split <id> <алиас>."""
    if await _slash_for_other_bot(event):
        return
    if not event.out:
        return
    reason = _index_available()
    if reason:
        await event.reply(f"⚠️ /entity недоступен: {reason}")
        return
    try:
        await _index_ensure_ddl()
    except Exception as e:
        await event.reply(f"❌ Не подключиться к базе индексации: {e}")
        return
    chat_id = event.chat_id
    action = event.pattern_match.group(1).lower()
    rest = event.pattern_match.group(2).strip()

    async def _get(eid):
        r = await db_read("SELECT * FROM entities WHERE chat_id=%s AND id=%s", (chat_id, int(eid)))
        return r[0] if r else None

    try:
        if action == "merge":  # merge <id1> <id2> — id2 вливается в id1, id2 удаляется
            a, b = (int(x) for x in rest.split()[:2])
            e1, e2 = await _get(a), await _get(b)
            if not e1 or not e2:
                await event.reply("❌ Один из id не найден. `/entity merge <id1> <id2>`")
                return
            al1 = json.loads(e1["aliases"]) if e1["aliases"] else []
            al2 = json.loads(e2["aliases"]) if e2["aliases"] else []
            merged = list(dict.fromkeys(al1 + al2 + [e1["name"], e2["name"]]))
            await db_transaction([
                ("UPDATE entities SET aliases=%s, canon_summary=NULL, fanon_summary=NULL, embedding=NULL WHERE chat_id=%s AND id=%s",
                 (json.dumps(merged, ensure_ascii=False), chat_id, a)),
                ("UPDATE entity_claims SET entity_id=%s WHERE chat_id=%s AND entity_id=%s", (a, chat_id, b)),
                ("UPDATE relations SET source_id=%s, embedding=NULL WHERE chat_id=%s AND source_id=%s", (a, chat_id, b)),
                ("UPDATE relations SET target_id=%s, embedding=NULL WHERE chat_id=%s AND target_id=%s", (a, chat_id, b)),
                ("DELETE FROM relations WHERE chat_id=%s AND source_id=target_id", (chat_id,)),
                ("DELETE FROM entities WHERE chat_id=%s AND id=%s", (chat_id, b)),
            ])
            _index_invalidate(chat_id, "entities", "relations")
            await event.reply(f"✅ «{e2['name']}» (id {b}) влит в «{e1['name']}» (id {a}). "
                              f"Векторы досье обновятся на `/index update`.")
        elif action == "rename":  # rename <id> <новое имя>
            parts = rest.split(maxsplit=1)
            eid, newname = int(parts[0]), parts[1].strip()
            e = await _get(eid)
            if not e:
                await event.reply("❌ id не найден.")
                return
            al = json.loads(e["aliases"]) if e["aliases"] else []
            if newname not in al:
                al.append(newname)
            await db_write("UPDATE entities SET name=%s, aliases=%s, embedding=NULL WHERE chat_id=%s AND id=%s",
                           (newname, json.dumps(al, ensure_ascii=False), chat_id, eid))
            _index_invalidate(chat_id, "entities")
            await event.reply(f"✅ id {eid} переименован в «{newname}».")
        elif action == "alias":  # alias <id> <алиас>
            parts = rest.split(maxsplit=1)
            eid, alias = int(parts[0]), parts[1].strip()
            e = await _get(eid)
            if not e:
                await event.reply("❌ id не найден.")
                return
            al = json.loads(e["aliases"]) if e["aliases"] else []
            if alias not in al:
                al.append(alias)
            await db_write("UPDATE entities SET aliases=%s, embedding=NULL WHERE chat_id=%s AND id=%s",
                           (json.dumps(al, ensure_ascii=False), chat_id, eid))
            _index_invalidate(chat_id, "entities")
            await event.reply(f"✅ «{alias}» добавлен алиасом к «{e['name']}» (id {eid}).")
        elif action == "split":  # split <id> <алиас> — отцепляет алиас в НОВУЮ пустую сущность
            parts = rest.split(maxsplit=1)
            eid, alias = int(parts[0]), parts[1].strip()
            e = await _get(eid)
            if not e:
                await event.reply("❌ id не найден.")
                return
            al = [a for a in (json.loads(e["aliases"]) if e["aliases"] else []) if a.lower() != alias.lower()]
            await db_write("UPDATE entities SET aliases=%s, embedding=NULL WHERE chat_id=%s AND id=%s",
                           (json.dumps(al, ensure_ascii=False), chat_id, eid))
            _, nid = await db_write(
                "INSERT INTO entities (chat_id,name,entity_type,aliases) VALUES (%s,%s,%s,%s)",
                (chat_id, alias, e["entity_type"], json.dumps([alias], ensure_ascii=False)))
            _index_invalidate(chat_id, "entities")
            await event.reply(f"✅ «{alias}» отцеплён от «{e['name']}» в новую сущность id {nid}. "
                              f"Факты остались на id {eid} — перепривяжи вручную при необходимости.")
    except (ValueError, IndexError):
        await event.reply("❌ Формат: `/entity merge <id1> <id2>` · `rename <id> <имя>` · `alias <id> <алиас>` · `split <id> <алиас>`")


# --- Запуск ---

_scheduler_started = False


async def _index_boot_resume():
    """Автовозобновление индексации после рестарта: подхватывает чаты, что были в статусе 'running'
    (их фоновая задача умерла вместе с процессом). Прогресс checkpoint'ится в БД, продолжаем с чекпоинта.
    Паузу ('paused') и ошибку ('error') НЕ трогаем — их продолжает пользователь вручную."""
    if _index_available():   # DSN не настроен → индекс-памяти нет
        return
    try:
        await _index_ensure_ddl()
        rows = await db_read("SELECT DISTINCT chat_id FROM idx_state WHERE status='running'")
    except Exception as e:
        log("INDEX", f"Автовозобновление: состояние недоступно ({e})")
        return
    for r in rows:
        cid = r["chat_id"]
        if cid in _INDEX_TASKS:
            continue
        _INDEX_CONTROL[cid] = "run"
        _INDEX_TASKS[cid] = asyncio.create_task(_index_pipeline(cid, None))  # status_msg=None → прогресс молча в лог
        log("INDEX", f"Автовозобновление индексации чата {cid} (с последнего чекпоинта)")


async def main():
    """Канонический async-паттерн: всё внутри корутины, await start/get_me/run.
    Запускается через client.loop.run_until_complete (НЕ asyncio.run — иначе сменится loop)."""
    global OWNER_ID, OWNER_USERNAME, OWNER_NAME, _scheduler_started
    await client.start()  # корректно ждём подключения (sync-magic на сервере не срабатывал)
    try:
        me = await client.get_me()
        OWNER_ID = me.id
        OWNER_USERNAME = getattr(me, "username", None)
        OWNER_NAME = getattr(me, "first_name", None)
        log("BOOT", f"Владелец: {_owner_label()} (id {OWNER_ID})")
    except Exception as e:
        log("BOOT", f"Не удалось получить get_me: {e}")
    if not _scheduler_started:
        asyncio.create_task(scheduler_loop())  # один раз на процесс
        _scheduler_started = True
    asyncio.create_task(_index_boot_resume())  # подхватить прерванную рестартом индексацию (с чекпоинта)
    log("BOOT", "Userbot запущен.")
    await client.run_until_disconnected()


if __name__ == "__main__":
    if api_id <= 0 or not api_hash:
        log("BOOT", "Ошибка запуска: проверь api_id/api_hash в .env")
        raise SystemExit(1)

    while True:
        try:
            client.loop.run_until_complete(main())
            log("BOOT", "Клиент отключён. Переподключение через 10 секунд...")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            log("BOOT", f"Ошибка главного цикла: {e}")
            traceback.print_exc()
        time.sleep(10)

from telethon import TelegramClient, events, utils
from telethon.errors.rpcerrorlist import MessageNotModifiedError, FloodWaitError
from telethon.extensions import html as tl_html
from telethon.helpers import add_surrogate
from telethon.tl.types import MessageEntityBlockquote, MessageEntityPre, MessageMediaWebPage, InputReplyToMessage
from telethon.tl.functions.messages import SendMessageRequest
from dotenv import load_dotenv
from openai import OpenAI
import os
import re
import io
import asyncio
import base64
import time
import traceback
import requests
import json
import glob
import logging
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
tavily_api_key = os.getenv("TAVILY_API_KEY")  # веб-поиск/извлечение страниц для /ask (tavily.com); без ключа веб-инструменты выключены
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")  # OCR фото (LlamaParse); без него фото идут через vision


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
OPENROUTER_IMAGE_MODEL = "sourceful/riverflow-v2.5-pro:free"  # /gen: text→image и image→image, бесплатная
OPENROUTER_IMAGE_FALLBACK = "sourceful/riverflow-v2.5-fast:free"  # запасная при перегрузке основной
GEN_IMAGE_MAX_INPUT = 3_000_000  # Sourceful лимитирует запрос 4.5 МБ; base64 ×1.33 → входное фото до ~3 МБ
GEN_BATCH_MAX = 20          # /gen -xN: максимум вариантов за команду (каждый ~40с–2мин)
GEN_BATCH_CONCURRENCY = 2   # сколько вариантов генерим одновременно (баланс скорость/лимиты free-модели)
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
MEDIA_OPENCODE_SLUGS = ["kimi-k2.5", "kimi-k2.6", "kimi-k2.7-code", "qwen3.5-plus", "qwen3.6-plus", "qwen3.7-plus", "mimo-v2-omni", "minimax-m3"]
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
DOC_MAX_CHARS = 16_000    # потолок встраиваемого текста файла (~5–6k токенов), дальше обрезка
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
    "deepseek": ("deepseek", DEEPSEEK_MODEL, "DeepSeek V4 Pro", 1000000, 1.15),
}
for _mid, _label, _ctx, _safety in [
    ("deepseek-v4-pro",  "DeepSeek V4 Pro",   1000000, 1.15),
    ("deepseek-v4-flash","DeepSeek V4 Flash", 1000000, 1.15),
    ("glm-5",            "GLM-5",              203000, 1.30),
    ("glm-5.1",          "GLM-5.1",            203000, 1.30),
    ("kimi-k2.5",        "Kimi K2.5",          262000, 2.50),
    ("kimi-k2.6",        "Kimi K2.6",          262000, 2.50),
    ("kimi-k2.7-code",   "Kimi K2.7 Code",     262000, 2.50),
    ("minimax-m2.5",     "MiniMax M2.5",       205000, 1.30),
    ("minimax-m2.7",     "MiniMax M2.7",       205000, 1.30),
    ("minimax-m3",       "MiniMax M3",        1000000, 1.30),
    ("qwen3.5-plus",     "Qwen3.5 Plus",       262000, 1.15),
    ("qwen3.6-plus",     "Qwen3.6 Plus",       262000, 1.15),
    ("qwen3.7-plus",     "Qwen3.7 Plus",       262000, 1.15),
    # qwen3.7-max ИСКЛЮЧЕНА: opencode отдаёт её только в native-формате
    # (401 "not supported for format oa-compat"), наш OpenAI-клиент её не вызовет.
    ("mimo-v2.5",        "MiMo V2.5",         1000000, 1.50),
    ("mimo-v2.5-pro",    "MiMo V2.5 Pro",     1000000, 1.50),
    ("mimo-v2-pro",      "MiMo V2 Pro",       1000000, 1.50),
    ("mimo-v2-omni",     "MiMo V2 Omni",      1000000, 1.50),
    ("hy3-preview",      "Hunyuan 3 Preview",  256000, 1.50),
]:
    MODEL_REGISTRY[_mid] = ("opencode", _mid, _label, _ctx, _safety)
# qwen3.7-max — opencode отдаёт её только в формате Anthropic Messages → провайдер "oc_anthropic"
# (свой адаптер-обёртка под OpenAI-интерфейс; полноценный tool-loop/голос, как у прочих).
MODEL_REGISTRY["qwen3.7-max"] = ("oc_anthropic", "qwen3.7-max", "Qwen3.7 Max", 262000, 1.15)
# Claude через ModelGate (OpenAI-совместимый шлюз). Окно 200k, vision и tools — нативные.
# safety 1.2: токенизатор Claude ≈ o200k, небольшой запас.
for _cid, _clabel in [
    ("claude-opus-4-8",   "Claude Opus 4.8"),
    ("claude-opus-4-7",   "Claude Opus 4.7"),
    ("claude-opus-4-6",   "Claude Opus 4.6"),
    ("claude-opus-4-5",   "Claude Opus 4.5"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-sonnet-4-5", "Claude Sonnet 4.5"),
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


def _clamp_reasoning(model_id: str, effort: str) -> str:
    """Приводит глобальный уровень ризонинга (шкала xhigh..none) к допустимому для модели:
    Gemini → thinkingLevel (none→minimal, xhigh→high); o3 не принимает none/xhigh → ближайший
    (low/high). Неизвестная модель → medium."""
    if model_id in GEMINI_MODELS:
        return GEMINI_THINKING_MAP.get(effort, "medium")
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


def _supports_reasoning(provider: str) -> bool:
    """Провайдеры с управляемой глубиной размышлений (/model reason): OpenAI и Google Gemini."""
    return provider in ("openai", "google")


def _reasoning_levels(slug: str):
    """Список уровней ризонинга для выбора `N.M` у модели (от мощного к слабому). None — не поддерживает."""
    spec = MODEL_REGISTRY.get(slug)
    if not spec:
        return None
    if spec[0] == "openai":
        return OPENAI_REASONING_LEVELS.get(slug)
    if spec[0] == "google":
        return _REASONING_RANK  # общая 5-уровневая шкала; маппится на thinkingLevel в _clamp_reasoning
    return None


def _reasoning_tag() -> str:
    """' · 🤔 high' — применяемый уровень ризонинга активной модели (OpenAI/Gemini) для префикса
    ответа и подписей. Без /model reason показывает дефолт. Прочие провайдеры → пустая строка."""
    spec = MODEL_REGISTRY.get(ACTIVE_MODEL)
    if not spec or not _supports_reasoning(spec[0]):
        return ""
    if REASONING_EFFORT:
        return f" · 🤔 {_clamp_reasoning(spec[1], REASONING_EFFORT)}"
    default = GEMINI_THINKING_DEFAULT if spec[0] == "google" else OPENAI_REASONING_DEFAULTS.get(spec[1], "auto")
    return f" · 🤔 {default}"


# Автообрезка контекста под окно модели
CTX_RESERVE_TOKENS = 8000   # запас на ответ (4096) + системку + вопрос
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
        if REASONING_EFFORT:
            kwargs.setdefault("reasoning_effort", _clamp_reasoning(model, REASONING_EFFORT))
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
        if REASONING_EFFORT:
            level = _clamp_reasoning(model, REASONING_EFFORT)
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

AUTO_REPLY_BUFFERS: dict = {}
AUTO_REPLY_TASKS: dict = {}
AUTO_REPLY_BUSY: set = set()     # чаты в фазе LLM/отправки — не отменяем их таску (иначе теряем сообщения)
AUTO_REPLY_HISTORY: dict = {}   # {chat_id: [{"role","content"}, ...]}
# AUTO_REPLY_ACTIVE_CHATS загружается из файла ниже (после load_json)
LAST_SCAN: list = []
LAST_FISH_SEARCH: list = []     # [{_id,title,languages}] последнего /voice fish search — для add по номеру
_ENTITY_CACHE: dict = {}        # кэш зарезолвленных каналов
ACTIVE_MODEL = "deepseek"       # перезаписывается из model_state.json при старте
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
ACTIVE_MODEL = _model_state.get("active", "deepseek")
if ACTIVE_MODEL not in MODEL_REGISTRY:
    ACTIVE_MODEL = "deepseek"
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
    provider, model_id, label, _ctx, _safety = MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["deepseek"])
    client_obj = _client_for_provider(provider)
    return client_obj, model_id, label


def _client_for_provider(provider):
    """Клиент/маркер доступности по провайдеру. None — провайдер не настроен."""
    if provider == "deepseek":
        return deepseek_client
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
    return opencode_client


def active_context_window() -> int:
    return MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["deepseek"])[3]


def active_ctx_safety() -> float:
    """Множитель safety для активной модели. Tiktoken (o200k) недосчитывает у некоторых моделей —
    safety даёт запас бюджета, чтобы не переполнить окно (см. ctx_safety_mult в реестре)."""
    spec = MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["deepseek"])
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


def active_model_supports_vision():
    """Умеет ли АКТИВНАЯ отвечающая модель принимать картинки напрямую (для /ask -g).
    True/False — известно; None — кастомная OpenRouter-модель без сохранённого флага
    (вызывающий проверит вживую через _openrouter_model_info)."""
    if ACTIVE_MODEL in MEDIA_OPENCODE_SLUGS:
        return True  # vision-слуги OpenCode (kimi/glm/qwen/mimo)
    spec = MODEL_REGISTRY.get(ACTIVE_MODEL)
    provider = spec[0] if spec else None
    if provider == "openrouter":
        return CUSTOM_MODELS.get(ACTIVE_MODEL, {}).get("vision")  # bool или None если не сохранено
    if provider == "modelgate":
        return False  # шлюз ModelGate НЕ доставляет картинки до Claude (проверено: base64 и URL —
                      # модель отвечает «изображения нет»). Для -g не годится; фото в /ask и так через OCR/медиа-модель.
    if provider == "openai":
        return True   # gpt-5.x / o3 принимают картинки напрямую (официальный API)
    if provider == "google":
        return True   # Gemini Flash видят картинки напрямую (нативный inlineData)
    return False  # DeepSeek и прочие текстовые


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


async def describe_image(image_bytes: bytes, caption: str = "", model: str = None, detail: str = "high") -> str:
    model = model or get_active_media_model()
    media_client = _client_for_media_model(model)  # OpenRouter или OpenCode-Go по id модели
    if not media_client:
        return caption or "[изображение]"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
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


def _sync_generate_image(prompt: str, input_images_b64: list = None, model: str = None,
                         image_size: str = "2K", aspect_ratio: str = None) -> tuple:
    """Генерация/редактирование изображения через OpenRouter (Riverflow). Возвращает (байты, mime).
    Проверено живьём: modalities строго ["image"] (с "text" — 404), ответ приходит как webp data-URL.
    modalities/message.images — нестандарт OpenAI, поэтому requests, а не SDK-клиент.
    Ошибки: 5xx/сеть/таймаут — обычные исключения (временные, можно ретраить);
    4xx и «нет картинки в ответе» — GenRejected (нужна правка промпта)."""
    content = [{"type": "text", "text": prompt}]
    for b64 in (input_images_b64 or []):
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    # image_config (живьём проверено): image_size 1K/2K/4K — реальное разрешение выхода
    # (1K=1024², 2K=2048², 4K только Pro), aspect_ratio — точная ориентация. Без него ~1K (мыло).
    image_config = {"image_size": image_size}
    if aspect_ratio:
        image_config["aspect_ratio"] = aspect_ratio
    resp = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"},
        json={
            "model": model or OPENROUTER_IMAGE_MODEL,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["image"],
            "image_config": image_config,
        },
        timeout=300,  # генерация медленная (reasoning-модель, в тесте ~2.5 мин)
    )
    if resp.status_code >= 500:
        resp.raise_for_status()  # 5xx — временная ошибка провайдера → ретрай тем же промптом
    if not resp.ok:  # 4xx: РАЗДЕЛЯЕМ дневной лимит / RPM-лимит-перегрузку / реальную модерацию
        try:
            detail = resp.json().get("error", {}).get("message") or resp.text
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
    images = (data.get("choices") or [{}])[0].get("message", {}).get("images") or []
    if not images:
        err = (data.get("choices") or [{}])[0].get("message", {}).get("content") or data.get("error", {}).get("message") or "пустой ответ"
        err_s = str(err)
        low = err_s.lower()
        # HTTP 200 без картинки: дневной лимит / перегрузка-RPM (временно) / реальная модерация.
        if any(mk in low for mk in _GEN_DAILY_MARKERS):
            raise GenExhausted(f"дневной лимит: {err_s[:200]}")
        if any(mk in low for mk in _GEN_TRANSIENT_MARKERS):
            raise GenTransient(f"провайдер не отдал картинку (временно): {err_s[:200]}")
        raise GenRejected(f"модель не вернула изображение: {err_s[:200]}")
    url = images[0]["image_url"]["url"]
    head, b64_out = url.split(",", 1)
    mime = head.split(":", 1)[-1].split(";", 1)[0] or "image/webp"  # "data:image/webp;base64"
    return base64.b64decode(b64_out), mime


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


_IMAGE_PROMPT_SYSTEM = (
    "Ты — промпт-инженер для генерации изображений. По запросу пользователя (и контексту чата, если он дан) "
    "составь ОДИН финальный промпт для модели генерации изображений: детальный, визуальный (композиция, стиль, "
    "свет, атмосфера), на английском языке. Верни ТОЛЬКО сам промпт, без пояснений, кавычек и преамбул."
)

_IMAGE_EDIT_SYSTEM = (
    "Ты — промпт-инженер для РЕДАКТИРОВАНИЯ существующего изображения. Модели генерации будут поданы "
    "референсные изображения (их текстовые описания даны ниже) и твой промпт. Твоя задача — только уточнить "
    "формулировку запроса пользователя: что КОНКРЕТНО изменить, согласовав с содержимым референса. "
    "СТРОГО: не добавляй новых объектов, персонажей, деталей и стилей, которых нет в запросе пользователя; "
    "не выдумывай ничего от себя; всё, что пользователь не просил менять, должно остаться как на референсе "
    "(можешь явно написать keep everything else unchanged). Верни ТОЛЬКО промпт на английском, кратко и точно "
    "(до 80 слов), без пояснений и кавычек."
)


def _sync_image_prompt(user_prompt: str, context_text: str = None, image_desc: str = None,
                       edit_mode: bool = False, previous_prompts: list = None) -> str:
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
            # пакет (есть история) → выше температура для разнообразия; редактирование — точность, создание — креатив
            temperature=(1.0 if previous_prompts else (0.4 if edit_mode else 0.7)),
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


async def _llm_create(messages: list, max_tokens: int = 4096, temperature: float = 1.0):
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


async def ask_agentic(context: str, question: str, must_search: bool = False, caller: str = None, ctx_tokens_est: int = None, voice_mode: str = "off", images: list = None, chat_id=None, msg_by_id: dict = None) -> str:
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
    has_tools = has_channels or has_web or has_reply

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
    if must_search and (has_channels or has_web):
        force_name = "telegram_search" if has_channels else "web_search"
        system_prompt += f"\n\nОБЯЗАТЕЛЬНО используй {force_name} хотя бы один раз перед тем как ответить."
    if voice_mode == "force":
        system_prompt += _voice_style_text(TTS_ENGINE, FISH_TTS_MODEL)
    elif voice_mode == "auto":
        system_prompt += _voice_auto_hint(TTS_ENGINE, FISH_TTS_MODEL)

    user_text = _build_ask_user_content(context, question, caller, now_str)
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
                  + ([REPLY_TOOL] if has_reply else []))
    reply_sent = [0]  # счётчик отправленных реплаев (анти-спам, лимит REPLY_MAX)
    sstats = {"iters": 0, "calls": 0, "posts": 0, "web": 0, "replies": 0}  # сводка (-c)

    def _log_search_summary():
        if sstats["iters"]:
            log("ASK", f"Поиск: {sstats['iters']} итер., {sstats['calls']} запросов к каналам, найдено {sstats['posts']} постов, веб-вызовов {sstats['web']}")

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
                kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool_name}} if force_tool else "auto"

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


async def print_lyrics(chat_id, text, chunk_size=3):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return
    current_text = text[:chunk_size]
    msg = await client.send_message(chat_id, current_text)
    for i in range(chunk_size, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if "\n" in chunk:
            await asyncio.sleep(0.8)
        else:
            await asyncio.sleep(0.2)
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
        return "↩"
    if getattr(rep, "out", False):
        rauthor = _owner_label()
    else:
        rauthor = _user_label(getattr(rep, "sender", None))
    quote = _preview(getattr(rep, "raw_text", None) or (_media_tag(rep) or ""), 50)
    head = ("↩ " + rauthor).strip()
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

    if anchor_id is not None and getattr(msg, "id", None) == anchor_id:
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
    if anchor_id is not None and any(getattr(m, "id", None) == anchor_id for m in group):
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
    mstats = {"photos": 0, "voice": 0, "audio": 0, "video_note": 0, "hit": 0, "miss": 0}
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
    mtot = mstats["photos"] + mstats["voice"] + mstats["audio"] + mstats["video_note"]
    if mtot:
        hr = round(100 * mstats["hit"] / mtot, 1)
        log("ASK", f"Медиа: {mtot} (фото {mstats['photos']} · голос {mstats['voice']} · аудио {mstats['audio']} · кружок {mstats['video_note']}) · кэш-хит {mstats['hit']}/{mtot} ({hr}%) · новых {mstats['miss']} · сбоев {failed_total}")

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
                "lines": [_line(u)], "marked": u["marked"],
            })

    out = [f"[{(b['ts'] + ' ' + b['label']).strip()}]: " + "\n".join(b["lines"]) for b in blocks]

    # Автообрезка под окно активной модели: держим самые свежие блоки, что влезают.
    # Бюджет в токенах; запас прочности под чужие токенизаторы (см. CTX_TOKEN_SAFETY).
    # NB: тестировал batch-encoding tiktoken — на типичной нагрузке оказался медленнее
    # per-block (FFI/parallel-setup оверхед), оставлен per-block + early-break.
    t_trunc_start = time.time()
    safety = safety_override if safety_override is not None else active_ctx_safety()
    budget = max(2000, int((active_context_window() - CTX_RESERVE_TOKENS) / safety))
    kept, total, truncated = [], 0, False
    for s in reversed(out):  # с конца — новейшие
        add = count_tokens(s) + 1  # +1 на разделитель блоков
        if kept and total + add > budget:
            truncated = True
            break
        kept.append(s)
        total += add
    kept.reverse()
    t_trunc = time.time() - t_trunc_start
    dropped = 0
    window = active_context_window()
    pct = round(100 * total / window, 1) if window else 0
    enc = "tiktoken" if _ENC is not None else "оценка по символам"
    if truncated:
        dropped = len(out) - len(kept)
        kept.insert(0, f"[…{dropped} более старых сообщений опущено — не влезли в окно модели…]")
        log("ASK", f"Контекст обрезан под окно {_fmt_ctx(window)}: оставлено {len(kept) - 1}/{len(out)} блоков, ~{total} ток ({enc}) = {pct}% окна (бюджет {budget} ток, safety×{safety:.2f})")
    else:
        log("ASK", f"Контекст готов: ~{total} ток ({enc}) из окна {_fmt_ctx(window)} = {pct}% занято, блоков {len(kept)} (бюджет {budget} ток, safety×{safety:.2f})")

    # Диагностика sub-фаз сборки контекста (быстро видно, что съело время на больших N)
    rep_total = rep_stats["hit"] + rep_stats["miss"] + rep_stats["no_quote"]
    if rep_total or t_render > 0.5 or t_trunc > 0.5:
        no_q = f" · без цитат: {rep_stats['no_quote']}" if rep_stats["no_quote"] else ""
        log("ASK", f"Подэтапы контекста: рендер={t_render:.1f}с · обрезка={t_trunc:.1f}с · "
                   f"reply: in-batch {rep_stats['hit']} · сеть {net_budget['used']}/{REPLY_NETWORK_BUDGET}{no_q}")
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
            ent = await client.get_entity(u)
            exclude_ids.add(ent.id)
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
                    # Параллельный сбор и под фильтром from_user (позиционные окна работают в messages.search).
                    _w, raw = await _collect_history_parallel(event.chat_id, n, 0, from_user=u)
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
                reply = await ask_agentic(context, question, must_search=must_search, caller=caller, ctx_tokens_est=ctx_tokens, voice_mode=voice_mode, images=images_sorted, chat_id=event.chat_id, msg_by_id=msg_by_id)
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
        if _is_attached_photo(msg) and msg.id not in seen:
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
        if m is not None and getattr(m, "photo", None) and m.id not in seen:
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
        if total + len(img) > GEN_IMAGE_MAX_INPUT:
            skipped += 1
            continue
        total += len(img)
        out.append(base64.b64encode(img).decode("utf-8"))
    if sources:
        log("GEN", f"Референсы: найдено {len(sources)} фото → взято {len(out)} ({total / 1024:.0f} КБ), пропущено {skipped}")
    return out, skipped


# Глобальный rate-gate для image-API: free Pro у Sourceful = 5 запросов/мин. Разносим вызовы во
# времени, чтобы не ловить RPM-429 и не жечь дневную квоту на провальных ретраях (failed тоже списываются).
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
                if size == "4K":
                    size = "2K"
                log("GEN", f"Пробую запасную {gen_model} (у неё своя квота)…")
                await _s("🔁 Дневной лимит основной модели — пробую запасную (fast)…")
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
                if size == "4K":  # Fast не умеет 4K → понижаем до 2K
                    size = "2K"
                    log("GEN", "Запасная Fast не поддерживает 4K — понижаю до 2K")
                log("GEN", f"Основная модель не отвечает — переключаюсь на запасную {gen_model}")
                await _s("🔁 Основная модель перегружена — пробую запасную (fast)…")
                continue
            return None, "overload", None, used_fallback


async def _gen_send_image(chat, raw, mime, final_prompt, prompt_by_ai, reply_to):
    """Отправляет готовую картинку: webp→png, и при AI-промпте — свёрнутая подпись (или отдельным
    сообщением, если длинная). chat='me' = Saved Messages."""
    if "webp" in mime:
        raw = await _webp_to_png(raw)  # webp Telegram шлёт стикером — конвертим
    bio = io.BytesIO(raw)
    bio.name = "gen.png" if raw[:8].startswith(b"\x89PNG") else "gen.webp"
    if prompt_by_ai:  # промпт от DeepSeek — СВЁРНУТОЙ цитатой и БЕЗ обрезки
        cap_text = "🎨 " + final_prompt
        if len(cap_text) <= 1000:  # влезает в лимит подписи Telegram (1024)
            try:
                cap, cap_ents = _collapsed_entities(cap_text, parse_html=False)
                await client.send_file(chat, bio, caption=cap, formatting_entities=cap_ents, reply_to=reply_to)
            except Exception as e:
                log("GEN", f"Свёрнутая подпись не отправилась ({e}) — шлю обычной")
                bio.seek(0)
                await client.send_file(chat, bio, caption=cap_text, reply_to=reply_to)
        else:  # длинный промпт: картинка без подписи + полный промпт отдельной свёрнутой цитатой
            sent = await client.send_file(chat, bio, reply_to=reply_to)
            await send_long(chat, cap_text, parse_mode=None, reply_to=getattr(sent, "id", None), collapse_threshold=0)
    else:
        await client.send_file(chat, bio, reply_to=reply_to)


@client.on(events.NewMessage(pattern=r"^[./]gen(?:\s+(\d+))?((?:\s+-(?:improve|creative|vertical|horizontal|square|sq|4k|2k|1k|x\d+|i|c|v|h))+)?((?:\s+!?@\w+)+)?\s+(.+)$"))
async def gen_command(event):
    """Генерация изображений (Riverflow via OpenRouter). Промпт как есть, либо его строит/улучшает DeepSeek
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
    creative = any(t in ("-c", "-creative") for t in toks)      # креатив: DeepSeek сочиняет промпт-ОТВЕТ (не редактирует)
    aspect_ratio = None                                         # ориентация → image_config.aspect_ratio (точно)
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

    if is_owner and not _is_attached_photo(event.message):
        await event.delete()  # чистим команду; ОСТАВЛЯЕМ только при реально приложенном фото-референсе
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
                    ent = await client.get_entity(u)
                    (exclude_ids if u in exclude_users else include_ids).add(ent.id)
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

        deepseek_requested = bool(context_text or improve or creative)
        edit_mode = bool(input_b64s) and not creative  # -c → творческий режим даже с референсом
        # DeepSeek сам не видит картинки → референсы описывает vision-модель (с кэшем). Считаем ОДИН раз на весь пакет.
        image_desc = None
        if deepseek_requested and input_b64s:
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

            async def _gen_and_send(idx, fp, by_ai):
                async with sem:
                    if counter["exhausted"]:  # дневной лимит уже исчерпан — не тратим квоту на обречённый запрос
                        counter["done"] += 1
                        return
                    raw_i, mime_i, used_fp, _fb = await _gen_one_image(
                        fp, input_b64s, image_size, aspect_ratio, (by_ai or deepseek_requested), user_prompt)
                    counter["done"] += 1
                    if raw_i is not None:
                        try:
                            await _gen_send_image("me", raw_i, mime_i, used_fp, by_ai, None)
                            counter["ok"] += 1
                        except Exception as e:
                            log("GEN", f"Вариант {idx + 1}: отправка в Избранное не удалась: {e}")
                    else:
                        if mime_i == "exhausted":
                            counter["exhausted"] = True  # дневной лимит — стоп остальным вариантам
                        log("GEN", f"Вариант {idx + 1} не сгенерирован ({mime_i})")
                    await set_status(f"🎨 {counter['done']}/{batch_count} готово · {counter['ok']} в Избранном…")

            # Промпты строим ПОСЛЕДОВАТЕЛЬНО: каждому показываем все предыдущие, и DeepSeek САМ придумывает
            # непохожий (без навязанных «углов»). Генерацию (медленную) сразу запускаем в фоне → перекрытие.
            prompts, tasks = [], []
            for i in range(batch_count):
                if counter["exhausted"]:  # дневной лимит исчерпан — не строим и не шлём остаток
                    log("GEN", f"Дневной лимит исчерпан — останавливаю пакет на варианте {i + 1}/{batch_count}")
                    break
                fp, by_ai = user_prompt, False
                if deepseek_client is not None:
                    fp = await asyncio.to_thread(_sync_image_prompt, user_prompt, context_text, image_desc, edit_mode, prompts)
                    by_ai = fp != user_prompt
                    log("GEN", f"Вариант {i + 1}/{batch_count}: промпт by_ai={by_ai} len={len(fp)}")
                prompts.append(fp)
                tasks.append(asyncio.create_task(_gen_and_send(i, fp, by_ai)))
                if deepseek_client is not None and i + 1 < batch_count:
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
        final_prompt, prompt_by_ai = user_prompt, False
        if deepseek_requested:
            await set_status("🧠 DeepSeek готовит промпт…")
            final_prompt = await asyncio.to_thread(_sync_image_prompt, user_prompt, context_text, image_desc, edit_mode)
            prompt_by_ai = final_prompt != user_prompt
            log("GEN", f"Промпт: by_ai={prompt_by_ai} · режим={'edit' if edit_mode else 'creative'} · vision_desc={'есть' if image_desc else 'нет'} · len={len(final_prompt)}")
        await set_status("🎨 Генерирую изображение… (может занять до пары минут)")
        t0 = time.time()
        # allow_repair: DeepSeek был ЗАПРОШЕН (даже если его первая попытка вернула пустое и промпт ушёл исходным)
        raw, mime, used_fp, used_fb = await _gen_one_image(
            final_prompt, input_b64s, image_size, aspect_ratio,
            (prompt_by_ai or deepseek_requested), user_prompt, set_status)
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
        await _gen_send_image(event.chat_id, raw, mime, used_fp, prompt_by_ai, reply_target_id)
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


@client.on(events.NewMessage(outgoing=True, pattern=r"^[./]song(?: |$)(.*)", from_users="me"))
async def song_command(event):
    if await _slash_for_other_bot(event):
        return
    custom_text = event.pattern_match.group(1).strip()
    text_to_print = custom_text if custom_text else SONG_TEXT
    await event.delete()
    await print_lyrics(event.chat_id, text_to_print)


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
        max_tokens=4096,
        temperature=1.0,
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
            ACTIVE_MODEL = "deepseek"
        _save_model_state()
        log("MODEL", f"Удалена кастомная модель: {target}")
        await event.edit(f"🗑 Удалена из избранного: `{target}`." + (" Активная модель сброшена на DeepSeek." if was_active else ""))
        return

    # --- глубина размышлений (OpenAI и Gemini): /model reason [уровень|auto] ---
    if arg.lower().startswith("reason"):
        rarg = arg[len("reason"):].strip().lower()
        active_provider = MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["deepseek"])[0]
        if not rarg:
            lines = ["🤔 **Глубина размышлений** — OpenAI (GPT-5.x / o3) и Google Gemini:", ""]
            for i, lv in enumerate(_REASONING_RANK, 1):
                mk = "▶" if lv == REASONING_EFFORT else "  "
                lines.append(f"{mk}{i}. `/model reason {lv}`")
            mk = "▶" if REASONING_EFFORT is None else "  "
            lines.append(f"{mk}{len(_REASONING_RANK) + 1}. `/model reason auto` — дефолт модели (5.5→medium, 5.4/5.4-mini→none, o3/o4-mini/Gemini→medium)")
            lines.append("\nНеподдерживаемый моделью уровень приводится к ближайшему: o3 — low/medium/high (none→low, xhigh→high); o4-mini без none; Gemini → thinkingLevel (none→minimal, xhigh→high).")
            lines.append("Быстрый выбор из списка `/model`: номер `N.M` (N — модель, M — сила ризонинга, M=1 — мощнейший).")
            if not _supports_reasoning(active_provider):
                lines.append("⚠️ Активная модель не поддерживает управление ризонингом — настройка сработает после выбора GPT/o3/Gemini.")
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
                applied = _clamp_reasoning(MODEL_REGISTRY[ACTIVE_MODEL][1], rarg)
                if applied != rarg:
                    note = f" (для активной {MODEL_REGISTRY[ACTIVE_MODEL][2]} применится `{applied}`)"
            else:
                note = " — сработает на моделях OpenAI (GPT-5.x / o3) и Gemini"
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
            "▶ активная · 🪟 окно · 🔧 поиск · 🚫 нет · ❔ не проверено",
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
                         "openrouter": "━━ OpenRouter (кастом) ━━"}.get(provider, f"━━ {provider} ━━")
                lines.append(f"\n{title}")
            mark = f"▶{i}." if slug == ACTIVE_MODEL else f"{i}."
            warn = " ⚠️нет ключа" if not is_available(provider) else ""
            lines.append(f"{mark} `{slug}` — {label} · 🪟{_fmt_ctx(ctx)}{tool_mark(slug)}{warn}")
            _levels = _reasoning_levels(slug)
            if _levels:
                # вариации силы ризонинга: выбор номером N.M (M=1 — мощнейший)
                parts = []
                for j, lv in enumerate(_levels, 1):
                    cur = "▶" if (slug == ACTIVE_MODEL and REASONING_EFFORT and _clamp_reasoning(slug, REASONING_EFFORT) == _clamp_reasoning(slug, lv)) else ""
                    parts.append(f"{cur}`{i}.{j}`{lv}")
                lines.append("    🤔 " + " · ".join(parts))
        if ACTIVE_MEDIA_MODEL in MEDIA_MODEL_REGISTRY:
            media_label = MEDIA_MODEL_REGISTRY[ACTIVE_MEDIA_MODEL][1]
        elif ACTIVE_MEDIA_MODEL in MEDIA_OPENCODE_SLUGS and ACTIVE_MEDIA_MODEL in MODEL_REGISTRY:
            media_label = f"{MODEL_REGISTRY[ACTIVE_MEDIA_MODEL][2]} [OpenCode]"
        else:
            media_label = f"{ACTIVE_MEDIA_MODEL} (кастомная)"
        lines.append(f"\n🖼 медиа-модель: {media_label} · `/model media` — сменить")
        lines.append("`/model N` / `/model <slug>` — выбрать · `/model probe` — проверить поиск (❔→🔧/🚫)")
        reff = f"`{REASONING_EFFORT}`" if REASONING_EFFORT else "авто"
        lines.append(f"`/model N.M` — GPT с силой ризонинга (M=1 мощнейший) · `/model reason` — глубина размышлений (сейчас: {reff})")
        lines.append("`/model vendor/model` — добавить ЛЮБУЮ модель OpenRouter по id (напр. `/model openai/gpt-4o`)")
        lines.append("`/model fav` — избранные OR-модели · `/model remove <N|id>` — удалить кастомную")
        await event.edit("\n".join(lines)[:4000])
        return

    if arg.lower() == "probe":
        await event.edit("🔧 Проверяю поддержку поиска у моделей…")
        tested = 0
        for slug in slugs:
            provider, mid, _label, _ctx, _safety = MODEL_REGISTRY[slug]
            if provider in ("oc_anthropic", "openai", "google"):
                continue  # qwen3.7-max / gpt-5.x / o3 / Gemini: tools работают, но короткий пробник (20 ток) reasoning-модели режет — флаг учится на лету в реальном /ask
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
            await event.edit(f"Вариации `{n}.M` (сила ризонинга) доступны только для OpenAI (GPT-5.x / o3) и Gemini. Для {label_nm} — просто `/model {n}`.")
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
        applied = _clamp_reasoning(_midn, REASONING_EFFORT)
        rnote = f"`{REASONING_EFFORT}`" + (f" (→ `{applied}`)" if applied != REASONING_EFFORT else "")
        log("MODEL", f"Активная модель: {slug_nm} ({label_nm}), ризонинг {REASONING_EFFORT}→{applied}")
        await event.edit(f"✅ Модель ответов: {label_nm} (окно 🪟{_fmt_ctx(ctx_nm)}) · 🤔 ризонинг: {rnote}")
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
            await event.edit(f"✅ Модель ответов: {label} (`{arg}`, OpenRouter, окно 🪟{_fmt_ctx(ctx)})")
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
        rtag = f" · 🤔 ризонинг: `{_clamp_reasoning(_mid, REASONING_EFFORT)}`" if REASONING_EFFORT else " · 🤔 ризонинг: авто (`/model reason`)"
    await event.edit(f"✅ Модель ответов: {label} (окно 🪟{_fmt_ctx(ctx)}){rtag}")


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
        "4️⃣ `[@юзеры]` — необязательно: `@имя` (только эти) или `!@имя` (исключить).\n"
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
        "Модель: Riverflow V2.5 Pro (OpenRouter, бесплатная), при перегрузке — запасная Fast. Нужен `OPENROUTER_API_KEY`.\n"
        "\n"
        "**Синтаксис:** `/gen [N] [-i|-c] [-v|-h|-sq] [-2k|-4k|-1k] [-xK] [@юзер|!@юзер] <промпт>`\n"
        "   `/gen аниме кот в очках` — генерация ровно по твоему промпту\n"
        "   `/gen -i закат над морем` — DeepSeek улучшит/уточнит промпт (`-i` или `-improve`)\n"
        "   `/gen 100 нарисуй о чём мы спорим` — DeepSeek составит промпт по последним 100 сообщениям чата\n"
        "\n"
        "**Режим `-c` (creative) — спросить DeepSeek, а не редактировать:**\n"
        "   DeepSeek сам СОЧИНЯЕТ генеративный промпт-ОТВЕТ на твой запрос (даже когда есть референс),\n"
        "   а не правит твой текст дословно. Полезно, когда хочешь задать вопрос, а не инструкцию:\n"
        "   `/gen 200 -c что хочет нарисовать чат?` · `/gen -c <ссылка> сделай что-то в этом духе`\n"
        "   (без `-c` и с референсом DeepSeek в режиме РЕДАКТИРОВАНИЯ — только уточняет, без отсебятины).\n"
        "\n"
        "**Ориентация (точное соотношение сторон):** `-v` вертикаль 9:16 · `-h` горизонталь 16:9 · `-sq` квадрат 1:1\n"
        "   `/gen -v аниме девушка у окна` · `/gen -h пейзаж гор` · комбинируется: `/gen -c -v <ссылка> …`\n"
        "\n"
        "**Качество (разрешение):** по умолчанию **2K** (2048²). `-4k` максимум (только Pro, медленнее), `-1k` быстрее/мельче.\n"
        "   `/gen -4k постер с текстом` — чёткий мелкий текст · `/gen -1k черновик` — быстро.\n"
        "   ⚠️ Telegram пережимает фото при отправке — для пиксель-в-пиксель оригинала это не панацея,\n"
        "   но 2K/4K заметно чётче дефолтного 1K. (Запасная Fast не умеет 4K → авто-понижение до 2K.)\n"
        "\n"
        "**Пакет `-xK` — много вариантов в Избранное:** `-x8` → 8 вариантов уйдут тебе в **Saved Messages**\n"
        "   (не в текущий чат — там только прогресс), макс. 20. DeepSeek пишет КАЖДОМУ варианту свой\n"
        "   промпт, ВИДЯ все предыдущие → сам придумывает непохожие (без навязанных шаблонов), все уникальны.\n"
        "   `/gen -x10 -4k аниме кот` · `/gen 200 -c -x8 что нарисовать?` (только для владельца).\n"
        "\n"
        "**Фильтр авторов контекста** (как в `/ask`, работает при числе N): `!@юзер` — ИСКЛЮЧИТЬ его сообщения\n"
        "   из контекста, `@юзер` — взять ТОЛЬКО его. Ставится перед промптом, можно несколько.\n"
        "   `/gen 2000 -c -x20 !@spambot !@flood арты по чату` — соберёт 2000 сообщений без этих авторов.\n"
        "\n"
        "**Референс-изображения (image-to-image):**\n"
        "   • прикрепи **фото прямо к сообщению** с `/gen` (можно альбомом) — они уйдут модели на вход;\n"
        "   • reply на **фото** + `/gen сделай фон ночным` → редактирование этой картинки\n"
        "     (промпт идёт дословно; добавь `-i` — DeepSeek уточнит формулировку, «увидев» референс\n"
        "     через vision-модель, и ничего не добавит от себя — меняется только то, что просишь);\n"
        "   • reply на **текстовое** сообщение — его текст идёт в контекст, промпт строит DeepSeek;\n"
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
        "🎵 **`/song [текст]`** — печать с эффектом набора\n"
        "\n"
        "Постепенно «печатает» переданный текст, имитируя живой набор.\n"
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
    provider, _mid, label, ctx, _ = MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["deepseek"])
    prov_name = {"deepseek": "DeepSeek", "openrouter": "OpenRouter", "opencode": "OpenCode Go",
                 "oc_anthropic": "OpenCode (нативный)", "modelgate": "ModelGate (Claude)",
                 "openai": "OpenAI", "google": "Google Gemini"}.get(provider, provider)
    ts = MODEL_TOOLS_SUPPORT.get(ACTIVE_MODEL)
    search_mark = "🔧 есть" if ts is True else ("🚫 нет" if ts is False else "❔ не проверен")
    sv = active_model_supports_vision()
    vis_mark = "✅ да" if sv is True else ("❌ нет" if sv is False else "❔ неизвестно")
    L.append("📊 **СТАТУС БОТА**")
    L.append(f"\n🧠 **Модель ответов:** {label} (`{ACTIVE_MODEL}`)")
    L.append(f"   провайдер: {prov_name} · окно: 🪟{_fmt_ctx(ctx)} · поиск по каналам: {search_mark} · vision (`-g`): {vis_mark}")
    if _supports_reasoning(provider):
        reff = f"`{_clamp_reasoning(_mid, REASONING_EFFORT)}`" if REASONING_EFFORT else "авто (дефолт модели)"
        L.append(f"   🤔 глубина размышлений: {reff} · `/model reason` — сменить")
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
    L.append(f"🎨 **Генерация (`/gen`):** Riverflow V2.5 Pro → Fast (free, фолбэк) {'✅' if openrouter_client is not None else '❌ нет OPENROUTER_API_KEY'}")
    # — избранное —
    L.append(f"⭐ **Избранное:** {len(FISH_FAVORITES)} Fish-голос(ов) · {len(CUSTOM_MODELS)} кастомных моделей")
    # — ключи —
    keys = []
    for p, nm in [("deepseek", "DeepSeek"), ("openrouter", "OpenRouter"), ("opencode", "OpenCode"), ("modelgate", "Claude/ModelGate"), ("openai", "OpenAI"), ("google", "Google Gemini")]:
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
        order = ["ask", "model", "media", "voice", "gen", "keys", "channels", "auto", "allow", "status", "song", "help"]
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

    await event.edit(section + f"\n\n━━━━━━━━━━━━━━━━━━━━━\n⚙️ Активная модель: **{active_label}**  ·  `/help` — все разделы")


# --- Запуск ---

_scheduler_started = False


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

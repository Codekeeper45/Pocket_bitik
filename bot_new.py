import telethon as tg
from telethon import TelegramClient, events, utils
from telethon.errors.rpcerrorlist import MessageNotModifiedError, FloodWaitError
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
import subprocess
import json
import glob
import logging
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
oylan_api_key = os.getenv("OYLAN_API_KEY")  # ISSAI Oylan (provider "oylan", не OpenAI-совместим)


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
OPENROUTER_VISION_MODEL = "google/gemini-3.1-flash-lite"  # дефолт vision (можно сменить .model media)
OPENROUTER_AUDIO_MODEL = "google/chirp-3"
OPENROUTER_AUDIO_FALLBACK = "openai/whisper-large-v3-turbo"  # запасная транскрипция, если Chirp-3 не отвечает

# Медиа-модели (vision) для выбора в .model media: slug -> (model_id, label)
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
MEDIA_OPENCODE_SLUGS = ["kimi-k2.5", "kimi-k2.6", "glm-5", "glm-5.1", "qwen3.5-plus", "qwen3.6-plus", "mimo-v2-omni"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
# Oylan (ISSAI) — stateful assistant API (НЕ OpenAI-совместим): создать ассистента → interaction → ответ.
# Авторизация: заголовок "Authorization: Api-Key <ключ>". Спека: oylan.nu.edu.kz/api/v1/swagger.json
OYLAN_BASE_URL = (os.getenv("OYLAN_BASE_URL", "https://oylan.nu.edu.kz/api/v1")).rstrip("/")
OYLAN_MODEL = os.getenv("OYLAN_MODEL", "Oylan")  # реальное имя модели в API (GET /assistant/models/)

# --- Google Gemini Flash TTS (голосовые ответы в .ask) ---
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
TTS_VOICE_CHAR_CAP = 1500      # потолок длины озвучиваемого текста (~1–1.5 мин речи). Длиннее
                               # режем. Очень длинные аудио у preview-модели деградируют/могут дать 400.
VOICE_SAMPLES_DIR = "voice_samples"  # кэш озвученных примеров голосов: voice_samples/<Имя>.ogg
VOICE_SAMPLE_TEXT = "Привет! Это мой голос. [с теплотой] Рада с тобой пообщаться."  # фраза-пример (одна на все голоса — удобно сравнивать)

# --- Fish Audio TTS (альтернативный движок озвучки, выбор через .voice engine fish) ---
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
ALLOWED_PATH = "allowed_users.json"
ALLOWED_ASK_TEXT_LIMIT = 500  # для гостей: запрос > этого числа → vision переключается на free
MEDIA_HIDETAIL_MAX_N = 200    # .ask с N больше этого → описываем фото в detail="low" (дешевле)
DIRECT_VISION_MAX_IMAGES = 10 # .ask -g: макс. картинок, отдаваемых модели напрямую (берём самые свежие)
ASKS_KEEP = 100               # кол-во последних .ask -d дампов, хранимых в asks/
REPLY_NETWORK_BUDGET = 200    # макс сетевых get_reply_message() за один .ask (когда target вне выборки)
ASK_MAX_TOKENS = 16000        # потолок completion для .ask (thinking-модели тратят на reasoning до тысяч токенов)
MEDIA_CONCURRENCY = 10    # параллельная обработка медиа (Gemini выдерживает)
SEARCH_CONCURRENCY = 5    # параллельный поиск по каналам
AUTO_REPLY_HISTORY_MAX = 20  # сообщений (≈10 реплик) на чат
COLLECT_WORKERS = 4           # параллельные окна сбора истории (.ask). Консервативно — низкий риск FloodWait.
COLLECT_MIN_PER_WORKER = 500  # минимум сообщений на воркер; при меньшем N — меньше воркеров (мелкие .ask не дробим зря)
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
    ("minimax-m2.5",     "MiniMax M2.5",       205000, 1.30),
    ("minimax-m2.7",     "MiniMax M2.7",       205000, 1.30),
    ("qwen3.5-plus",     "Qwen3.5 Plus",       262000, 1.15),
    ("qwen3.6-plus",     "Qwen3.6 Plus",       262000, 1.15),
    ("qwen3.7-max",      "Qwen3.7 Max",        262000, 1.15),
    ("mimo-v2.5",        "MiMo V2.5",         1000000, 1.50),
    ("mimo-v2.5-pro",    "MiMo V2.5 Pro",     1000000, 1.50),
    ("mimo-v2-pro",      "MiMo V2 Pro",       1000000, 1.50),
    ("mimo-v2-omni",     "MiMo V2 Omni",      1000000, 1.50),
    ("hy3-preview",      "Hunyuan 3 Preview",  256000, 1.50),
]:
    MODEL_REGISTRY[_mid] = ("opencode", _mid, _label, _ctx, _safety)
# Oylan (ISSAI) — провайдер "oylan" (свой адаптер, не OpenAI). Окно ~32k, safety 1.3.
MODEL_REGISTRY["oylan"] = ("oylan", OYLAN_MODEL, "Oylan 3", 32000, 1.3)
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

ASK_SYSTEM_PROMPT = """Ты — {model}, ИИ с характером и собственной точкой зрения. Не нейтральный ассистент, а собеседник с позицией.

Правила:
- Отвечай на русском. Не ограничивай себя по длине — давай полный, развёрнутый ответ, столько сколько требует вопрос. Не комкай и не сокращай.
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

Твой вопрос помечен значком ❓ и продублирован ДО и ПОСЛЕ контекста чата. Между метками — лог переписки (контекст). Выполняй именно ❓-вопрос: если он просит ответить на сообщения или вопросы из переписки — делай это по контексту.

Формат контекста: это лог чата. Каждое сообщение — отдельный блок, блоки разделены пустой строкой. Заголовок в квадратных скобках: [время автор]: текст. Метки в заголовке: «↩ автор: «цитата»» — это ответ на сообщение указанного автора; «⤷ из X» — сообщение переслано из источника X. В тексте: [Фото: …]/[Аудио: …]/[Речь: …] — распознанное содержимое медиа; [Видео]/[Файл]/[GIF] — медиа без распознавания."""
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

# Стиль голосового ответа: как писать текст, который будет ОЗВУЧЕН (TTS).
# Это инструкция модели по эмоциям/аудио-тегам для живой подачи.
VOICE_STYLE_PROMPT = """

━━ РЕЖИМ ГОЛОСОВОГО ОТВЕТА ━━
Твой ответ будет ОЗВУЧЕН (text-to-speech) и отправлен как голосовое сообщение. Поэтому:
- Пиши как живую устную речь от первого лица, разговорно и эмоционально. НЕ как текст-статью.
- Коротко-средне: до ~1500 символов (голосовое примерно на минуту-полторы). Не растягивай в простыню и не перечисляй длинными списками — говори живо и по делу.
- НЕ используй HTML, markdown, эмодзи, ссылки, код — только произносимые слова.
- Управляй интонацией аудио-тегами в квадратных скобках — они НЕ произносятся, а задают подачу:
  [радостно] [взволнованно] [смеётся] [усмехается] [вздыхает] [шёпотом] [тихо] [серьёзно]
  [саркастично] [с теплотой] [задумчиво] [удивлённо] [с сожалением]
- Паузы — многоточием «…». Передавай эмоцию голосом и тегами, а не смайликами.
- Пример: «[усмехается] Ну ты даёшь… [с теплотой] на самом деле, это отличная идея.»"""

# Подсказка для авто-режима: модель сама решает, отвечать ли голосом.
VOICE_AUTO_HINT = """

━━ ВОЗМОЖНОСТЬ ОТВЕТИТЬ ГОЛОСОМ ━━
По умолчанию отвечай ТЕКСТОМ по правилам выше (Telegram-HTML). НО если ответ будет уместнее и живее голосом (эмоциональная реакция, короткий личный ответ, шутка, поддержка) — ты можешь ответить голосовым.
Чтобы ответить голосом: начни самую первую строку ответа с маркера [[VOICE]] на отдельной строке, а дальше дай текст строго по правилам режима голосового ответа (ниже). Если голос не нужен — просто отвечай текстом без маркера.
""" + VOICE_STYLE_PROMPT

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
# Oylan не OpenAI-совместим (свой requests-адаптер) — клиента нет, держим строку-маркер доступности.
oylan_client = "oylan" if oylan_api_key else None

AUTO_REPLY_BUFFERS: dict = {}
AUTO_REPLY_TASKS: dict = {}
AUTO_REPLY_BUSY: set = set()     # чаты в фазе LLM/отправки — не отменяем их таску (иначе теряем сообщения)
AUTO_REPLY_HISTORY: dict = {}   # {chat_id: [{"role","content"}, ...]}
# AUTO_REPLY_ACTIVE_CHATS загружается из файла ниже (после load_json)
LAST_SCAN: list = []
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
# Кастомные OpenRouter-модели для ответов (заданы через .model <vendor/model>) — восстанавливаем в реестр,
# чтобы они стали полноценными записями (провайдер "openrouter") и пережили рестарт.
CUSTOM_MODELS = _model_state.get("custom_models", {})  # {id: {"label","ctx","safety"}}
for _cid, _ci in CUSTOM_MODELS.items():
    MODEL_REGISTRY[_cid] = ("openrouter", _cid, (_ci.get("label") or _cid), int(_ci.get("ctx") or 128000), float(_ci.get("safety") or 1.3))
ACTIVE_MODEL = _model_state.get("active", "deepseek")
if ACTIVE_MODEL not in MODEL_REGISTRY:
    ACTIVE_MODEL = "deepseek"
MODEL_TOOLS_SUPPORT = _model_state.get("tools_support", {})  # {slug: True|False} — обучается на лету
# slug из реестра ИЛИ произвольный model_id OpenRouter (кастомная медиа-модель)
ACTIVE_MEDIA_MODEL = _model_state.get("active_media") or "lite"
# Голос для озвучки ответов (.ask) и режим авто-голоса (модель сама решает озвучивать)
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
    if provider == "oylan":
        return oylan_client
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
    """Умеет ли АКТИВНАЯ отвечающая модель принимать картинки напрямую (для .ask -g).
    True/False — известно; None — кастомная OpenRouter-модель без сохранённого флага
    (вызывающий проверит вживую через _openrouter_model_info)."""
    if ACTIVE_MODEL in MEDIA_OPENCODE_SLUGS:
        return True  # vision-слуги OpenCode (kimi/glm/qwen/mimo)
    spec = MODEL_REGISTRY.get(ACTIVE_MODEL)
    provider = spec[0] if spec else None
    if provider == "openrouter":
        return CUSTOM_MODELS.get(ACTIVE_MODEL, {}).get("vision")  # bool или None если не сохранено
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
    save_json(MODEL_STATE_PATH, {"active": ACTIVE_MODEL, "tools_support": MODEL_TOOLS_SUPPORT, "active_media": ACTIVE_MEDIA_MODEL, "custom_models": CUSTOM_MODELS, "active_voice": ACTIVE_VOICE, "voice_auto": VOICE_AUTO, "tts_engine": TTS_ENGINE, "fish_voice": FISH_VOICE, "fish_favorites": FISH_FAVORITES})


def _set_tools_support(slug, ok):
    if MODEL_TOOLS_SUPPORT.get(slug) != ok:
        MODEL_TOOLS_SUPPORT[slug] = ok
        _save_model_state()
        log("MODEL", f"{slug}: поддержка tools = {ok}")


# --- Персист активных auto_reply-чатов ---

AUTO_REPLY_ACTIVE_CHATS = set(load_json(AUTO_REPLY_PATH, []))


def _save_auto_reply():
    save_json(AUTO_REPLY_PATH, list(AUTO_REPLY_ACTIVE_CHATS))


# --- Разрешённые пользователи (доступ к .ask) ---
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


async def send_long(chat_id, text, prefix="", parse_mode=_PARSE_UNSET):
    # Разбивает длинный текст на части ≤ лимита Telegram (4096), режет по абзацам/строкам/словам.
    # parse_mode: не передан → дефолт клиента (md); "html"/"md"/None — явно. При ошибке парсинга
    # (кривая разметка от модели) чанк переотправляется как обычный текст, чтобы не потерять ответ.
    LIMIT = 4000
    text = text or ""
    remaining = text
    first = True
    _kwargs = {} if parse_mode is _PARSE_UNSET else {"parse_mode": parse_mode}
    _can_fallback = (parse_mode is _PARSE_UNSET) or bool(parse_mode)

    async def _send(msg):
        try:
            await client.send_message(chat_id, msg, **_kwargs)
        except FloodWaitError as e:
            await asyncio.sleep(e.seconds + 1)
            await client.send_message(chat_id, msg, **_kwargs)
        except Exception as e:
            if not _can_fallback:
                raise
            log("SEND", f"Разметка не распозналась ({e}) — шлю как обычный текст")
            try:
                await client.send_message(chat_id, msg, parse_mode=None)
            except FloodWaitError as e2:
                await asyncio.sleep(e2.seconds + 1)
                await client.send_message(chat_id, msg, parse_mode=None)

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
            return (response.choices[0].message.content or "").strip() or "[изображение]"
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
            return (response.choices[0].message.content or "").strip()
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
    if not openrouter_api_key:
        return "[аудио сообщение]"
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    resp = requests.post(
        f"{OPENROUTER_BASE_URL}/audio/transcriptions",
        headers={"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"},
        json={"model": model, "input_audio": {"data": b64, "format": fmt}},
        timeout=60,
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


async def transcribe_audio(audio_bytes: bytes, fmt: str = "ogg") -> str:
    # Chirp-3 с ретраями; если не справился — запасная Whisper.
    for model in (OPENROUTER_AUDIO_MODEL, OPENROUTER_AUDIO_FALLBACK):
        text = await _transcribe_with(model, audio_bytes, fmt)
        if text:
            return text
        log("MEDIA", f"{model} не дал транскрипцию" + (", пробую запасную" if model == OPENROUTER_AUDIO_MODEL else ""))
    return "[аудио сообщение]"


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
        "-c:a", "libopus", "-b:a", "24k", "-vbr", "on", "-application", "voip",
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
        "ffmpeg", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "24k", "-vbr", "on",
        "-application", "voip", "-f", "ogg", "pipe:1",
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
        raise RuntimeError("Fish-голос не выбран (.voice fish add/select)")
    clean = re.sub(r"\[[^\]]*\]", "", text)  # Fish не понимает [аудио-теги] — убираем
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


def _extract_content(message) -> str:
    # Финальный ответ в .content; у reasoning-моделей при пустом .content берём .reasoning_content
    content = (getattr(message, "content", None) or "").strip()
    if content:
        return content
    return (getattr(message, "reasoning_content", None) or "").strip()


def _sync_oylan_answer(system_prompt: str, user_text: str, websearch: bool, max_tokens: int) -> str:
    """Oylan (ISSAI): создать ассистента → стримовый interaction → собрать текст → удалить ассистента.
    Важно: НЕстримовый путь у Oylan возвращает 500, поэтому используем stream=true (SSE):
    события chunk накапливаем, финальный текст берём из события complete. Авторизация Api-Key."""
    h = {"Authorization": f"Api-Key {oylan_api_key}"}
    # Имя ассистента должно быть уникальным (Oylan: 400 «already exists»). Ассистент временный, удаляется.
    payload = {"name": f"davinchik-{time.time_ns()}", "model": OYLAN_MODEL, "temperature": 1.0,
               "max_tokens": int(max_tokens), "system_instructions": system_prompt,
               "websearch_enabled": bool(websearch)}
    r = requests.post(f"{OYLAN_BASE_URL}/assistant/", headers=h, json=payload, timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Oylan create HTTP {r.status_code}: {r.text[:200]}")
    aid = (r.json() or {}).get("id")
    if not aid:
        raise RuntimeError("Oylan: ассистент без id")
    try:
        rr = requests.post(f"{OYLAN_BASE_URL}/assistant/{aid}/interactions/", headers=h,
                           data={"content": user_text, "stream": "true"}, timeout=180, stream=True)
        if rr.status_code != 200:
            raise RuntimeError(f"Oylan interaction HTTP {rr.status_code}: {rr.text[:200]}")
        chunks, final = [], None
        for raw in rr.iter_lines():  # bytes (UTF-8)
            if not raw:
                continue
            s = raw.decode("utf-8", "replace")
            if s.startswith("data:"):
                s = s[5:].strip()
            try:
                obj = json.loads(s)
            except Exception:
                continue
            t = obj.get("type")
            if t == "chunk":
                chunks.append(obj.get("content", ""))
            elif t == "complete":
                mm = obj.get("model_message") or {}
                final = (mm.get("content") if isinstance(mm, dict) else None) or final
            elif t == "error":
                raise RuntimeError(f"Oylan stream error: {str(obj)[:160]}")
        reply = (final or "".join(chunks)).strip()
        if not reply:
            raise RuntimeError("Oylan: пустой ответ потока")
        return reply
    finally:
        try:
            requests.delete(f"{OYLAN_BASE_URL}/assistant/{aid}/", headers=h, timeout=30)
        except Exception:
            pass


async def _oylan_answer(system_prompt: str, user_text: str, websearch: bool = False, max_tokens: int = 4096) -> str:
    return await asyncio.to_thread(_sync_oylan_answer, system_prompt, user_text, websearch, max_tokens)


def _active_provider() -> str:
    return MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["deepseek"])[0]


async def _llm_create(messages: list, max_tokens: int = 4096, temperature: float = 1.0):
    client_obj, model_id, label = get_active_model()
    if client_obj is None:
        log("AI", f"Активная модель {ACTIVE_MODEL} недоступна (нет ключа провайдера)")
        return None
    # Oylan — свой адаптер (не OpenAI): склеиваем system + последний user-текст.
    if _active_provider() == "oylan":
        sys_p = next((m["content"] for m in messages if m.get("role") == "system" and isinstance(m.get("content"), str)), "")
        user_parts = [m["content"] for m in messages if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
        try:
            return await _oylan_answer(sys_p, "\n\n".join(user_parts), websearch=False, max_tokens=max_tokens)
        except Exception as e:
            log("AI", f"Oylan ошибка: {e}")
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


def _build_ask_user_content(context: str, question: str, caller: str = None) -> str:
    """Вопрос дублируется ДО и ПОСЛЕ лога, лог обёрнут разделителями и явно помечен как фон.
    Так модель не путает реальный вопрос с чужими вопросами внутри переписки."""
    asker = caller or "пользователь"
    return (
        f"❓ ВОПРОС (его задаёт {asker}): {question}\n\n"
        f"━━━━━ Контекст чата (лог переписки) ━━━━━\n"
        f"{context}\n"
        f"━━━━━ конец контекста чата ━━━━━\n\n"
        f"❓ Повторяю вопрос (от {asker}): {question}"
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


async def ask_agentic(context: str, question: str, must_search: bool = False, caller: str = None, ctx_tokens_est: int = None, voice_mode: str = "off", images: list = None) -> str:
    """Agentic ask: модель сама решает, искать ли информацию в каналах.
    ctx_tokens_est — tiktoken-оценка контекста (для логирования Δ с реальным API).
    voice_mode: "off" — обычный текст; "force" — ответ под озвучку (флаг -v); "auto" — модель сама может выбрать голос (маркер [[VOICE]]).
    images — список {"bytes":...} для прямого vision (.ask -g): кладутся в user-сообщение как image_url."""
    llm, model_id, label = get_active_model()
    if llm is None:
        return "Модель не настроена (проверь ключ провайдера)"

    channels = get_tracked()
    has_channels = len(channels) > 0

    now_str = datetime.now(MSK).strftime("%d.%m.%Y %H:%M")

    # Oylan (ISSAI) — свой assistant-API, не OpenAI tool-loop. Поиск — через websearch_enabled (флаг -c).
    if _active_provider() == "oylan":
        sys_p = ASK_SYSTEM_PROMPT.replace("{model}", label) + f"\n\nТекущая дата и время: {now_str} МСК."
        if voice_mode == "force":
            sys_p += VOICE_STYLE_PROMPT
        elif voice_mode == "auto":
            sys_p += VOICE_AUTO_HINT
        try:
            reply = await _oylan_answer(sys_p, _build_ask_user_content(context, question, caller),
                                        websearch=must_search, max_tokens=ASK_MAX_TOKENS)
            return reply or "Oylan вернул пустой ответ."
        except Exception as e:
            log("ASK", f"Oylan ошибка: {e}")
            return f"⚠️ Oylan не ответил: {str(e)[:150]}"

    system_prompt = ASK_SYSTEM_PROMPT.replace("{model}", label) + f"\n\nТекущая дата и время: {now_str} МСК. Учитывай актуальность: оценивай свежесть постов по их дате, для вопросов о новостях опирайся на самые недавние."
    if has_channels:
        system_prompt += "\n\nУ тебя есть доступ к инструменту telegram_search для поиска в Telegram-каналах. Используй его если вопрос требует актуальной информации, которой нет в контексте переписки. Формулируй точные поисковые запросы. Для свежих новостей указывай параметр days."
    if must_search and has_channels:
        system_prompt += "\n\nОБЯЗАТЕЛЬНО используй telegram_search хотя бы один раз перед тем как ответить."
    if voice_mode == "force":
        system_prompt += VOICE_STYLE_PROMPT
    elif voice_mode == "auto":
        system_prompt += VOICE_AUTO_HINT

    user_text = _build_ask_user_content(context, question, caller)
    if images:
        # Мультимодальный content: текст + сами картинки (.ask -g)
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

    max_iterations = 10
    force_tool = must_search and has_channels
    sstats = {"iters": 0, "calls": 0, "posts": 0}  # сводка поиска (-c)

    def _log_search_summary():
        if sstats["iters"]:
            log("ASK", f"Поиск: {sstats['iters']} итер., {sstats['calls']} запросов к каналам, найдено {sstats['posts']} постов суммарно")

    for iteration in range(max_iterations):
        log("ASK", f"Agentic итерация {iteration + 1}/{max_iterations}")

        try:
            kwargs = dict(
                model=model_id,
                messages=messages,
                max_tokens=ASK_MAX_TOKENS,
                temperature=1.0,
            )
            if has_channels:
                kwargs["tools"] = [TELEGRAM_SEARCH_TOOL]
                kwargs["tool_choice"] = {"type": "function", "function": {"name": "telegram_search"}} if force_tool else "auto"

            try:
                response = await asyncio.to_thread(llm.chat.completions.create, **kwargs)
            except Exception as e:
                # Thinking-модели (DeepSeek) не умеют ПРИНУДИТЕЛЬНЫЙ tool_choice, но auto — умеют.
                # Не считаем «без tools»: повторяем с auto, поиск остаётся доступен.
                if force_tool and has_channels and _is_thinking_mode_quirk(e):
                    log("ASK", "Принудительный tool_choice не поддержан (thinking-режим) — повтор с auto")
                    kwargs["tool_choice"] = "auto"
                    response = await asyncio.to_thread(llm.chat.completions.create, **kwargs)
                else:
                    raise
        except TypeError:
            # Модель не поддерживает tools — fallback на обычный ask
            log("ASK", "Модель не поддерживает tool calling, fallback на обычный ask")
            if has_channels:
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
            # thinking-quirk НЕ трактуем как «без tools» (модель умеет auto/tools, просто особенности API)
            if has_channels and not quirk and any(k in str(e).lower() for k in ("tool", "function")):
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
            log("ASK", f"API {label}: занято {usage.prompt_tokens} ток в окне {_fmt_ctx(win)} = {occ}% (итер {iteration + 1}); ответ {usage.completion_tokens} ток")
            # Δ tiktoken vs реального токенизатора API (только на первой итерации — где контекст без tool-сообщений)
            if ctx_tokens_est and iteration == 0 and usage.prompt_tokens:
                delta = usage.prompt_tokens - ctx_tokens_est
                pct = round(100 * delta / usage.prompt_tokens, 1)
                verdict = "недооценил" if delta > 0 else ("переоценил" if delta < 0 else "точно")
                margin = (CTX_TOKEN_SAFETY - 1) * 100
                covered = "покрыл" if abs(pct) <= margin else "НЕ покрыл"
                log("ASK", f"Δ токенизаторов: tiktoken={ctx_tokens_est} vs API={usage.prompt_tokens} → tiktoken {verdict} на {abs(pct)}% (запас {int(margin)}% {covered})")

        # Получили валидный ответ с инструментами — модель умеет tools
        if has_channels and msg.tool_calls:
            _set_tools_support(ACTIVE_MODEL, True)
            sstats["iters"] += 1
            sstats["calls"] += len(msg.tool_calls)

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
            if tool_call.function.name != "telegram_search":
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Неизвестный инструмент: {tool_call.function.name}"
                })
                continue

            query = ""
            days = None
            try:
                args = json.loads(tool_call.function.arguments)
                query = (args.get("query") or "").strip()
                days = args.get("days")
                if days is not None:
                    days = int(days)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
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


async def process_media_cached(m, vision_model: str = None, detail: str = "high", mstats: dict = None, inline_ids: set = None, inline_images: list = None):
    """Текст медиа (описание/транскрипт) с кэшем по file-id. None — если медиа нет.
    mstats — опциональный аккумулятор статистики (photos/voice/audio/video_note + hit/miss).
    inline_ids/inline_images (режим .ask -g): фото НЕ описываются, а сами байты собираются
    в inline_images, в тексте — плейсхолдер [Картинка #k]. Голос/аудио/кружок — без изменений."""
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
        text_part = f" {m.raw_text}" if m.raw_text else ""
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
    if rep is None:
        if rep_stats is not None:
            rep_stats["no_quote"] = rep_stats.get("no_quote", 0) + 1
        return "↩"
    if rep.out:
        rauthor = _owner_label()
    else:
        rauthor = _user_label(rep.sender)
    quote = _preview(rep.raw_text or (_media_tag(rep) or ""), 50)
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


async def _render_unit(msg, text_only: bool, anchor_id=None, vision_model: str = None, detail: str = "high", mstats: dict = None, by_id: dict = None, net_budget: dict = None, rep_stats: dict = None, inline_ids: set = None, inline_images: list = None) -> dict:
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
            media_body = await process_media_cached(msg, vision_model, detail=detail, mstats=mstats, inline_ids=inline_ids, inline_images=inline_images)
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


async def _render_album_segment(group, text_only: bool, anchor_id=None, vision_model: str = None, detail: str = "high", mstats: dict = None, by_id: dict = None, net_budget: dict = None, rep_stats: dict = None, inline_ids: set = None, inline_images: list = None) -> dict:
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
        # direct-vision (.ask -g): собираем фото альбома сами, без описания
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
        "marked": marked,
        "failed": sum(body.count(mk) for mk in MEDIA_FAILURE_MARKERS),
    }


async def _render_segment(seg, text_only: bool, anchor_id=None, vision_model: str = None, detail: str = "high", mstats: dict = None, by_id: dict = None, net_budget: dict = None, rep_stats: dict = None, inline_ids: set = None, inline_images: list = None) -> dict:
    if len(seg) == 1:
        u = await _render_unit(seg[0], text_only, anchor_id, vision_model, detail, mstats=mstats, by_id=by_id, net_budget=net_budget, rep_stats=rep_stats, inline_ids=inline_ids, inline_images=inline_images)
        u.pop("gid", None)  # gid больше не используется на этапе склейки
        return u
    return await _render_album_segment(seg, text_only, anchor_id, vision_model, detail, mstats=mstats, by_id=by_id, net_budget=net_budget, rep_stats=rep_stats, inline_ids=inline_ids, inline_images=inline_images)


def _needs_media(m) -> bool:
    return bool(getattr(m, "photo", None) or getattr(m, "voice", None)
                or getattr(m, "audio", None) or getattr(m, "video_note", None))


async def assemble_context(messages, text_only: bool, anchor_id=None, progress_cb=None, vision_model: str = None, detail: str = "high", safety_override: float = None, inline_ids: set = None, inline_images: list = None):
    """Строит контекст: параллельный рендер + склейка альбомов и подряд идущих реплик автора.
    Возвращает (context_str, dropped_blocks, failed_media, ctx_tokens). progress_cb(done, total, failed).
    safety_override — если задан, перебивает per-model safety (используется при ретрае overflow)."""
    if not messages:
        return "", 0, 0, 0
    t_render_start = time.time()
    segments = _group_segments(messages)
    sem = asyncio.Semaphore(MEDIA_CONCURRENCY)
    media_total = 0 if text_only else sum(1 for s in segments if any(_needs_media(m) for m in s))
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

    async def render(seg):
        nonlocal done, failed_total
        async with sem:
            u = await _render_segment(seg, text_only, anchor_id, vision_model, detail, mstats=mstats, by_id=by_id, net_budget=net_budget, rep_stats=rep_stats, inline_ids=inline_ids, inline_images=inline_images)
        failed_total += u.get("failed", 0)
        if not text_only and any(_needs_media(m) for m in seg):
            done += 1
            if progress_cb:
                await progress_cb(done, media_total, failed_total)
            if done % 50 == 0:  # инкрементально сохраняем кэш — переживёт краш/рестарт посреди большого .ask
                save_media_cache()
        return u

    units = await asyncio.gather(*[render(s) for s in segments])
    t_render = time.time() - t_render_start

    # Сводка по медиа (если что-то было)
    mtot = mstats["photos"] + mstats["voice"] + mstats["audio"] + mstats["video_note"]
    if mtot:
        hr = round(100 * mstats["hit"] / mtot, 1)
        log("ASK", f"Медиа: {mtot} (фото {mstats['photos']} · голос {mstats['voice']} · аудио {mstats['audio']} · кружок {mstats['video_note']}) · кэш-хит {mstats['hit']}/{mtot} ({hr}%) · новых {mstats['miss']} · сбоев {failed_total}")

    # Склейка: подряд идущие сообщения одного автора без меток → один блок (альбомы уже самоформатированы).
    blocks = []
    for u in units:
        if not u["body"]:
            continue
        if blocks and not u["marked"] and not blocks[-1]["marked"] and blocks[-1]["akey"] == u["akey"]:
            blocks[-1]["lines"].append(u["body"])
        else:
            blocks.append({
                "akey": u["akey"], "label": u["label"], "ts": u["ts"],
                "lines": [u["body"]], "marked": u["marked"],
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
    Самомасштабируется: при малом n — меньше воркеров (мелкие .ask не дробим зря)."""
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


@client.on(events.NewMessage(pattern=r"^\.ask\s+(\d+)((?:\s+-[tcdvg]+)+)?((?:\s+!?@\w+)+)?\s+(.+)"))
async def ask_command(event):
    is_owner = event.out
    if not is_owner and event.sender_id not in ALLOWED_USERS:
        return  # не владелец и не в списке разрешённых
    n = int(event.pattern_match.group(1))
    flags = event.pattern_match.group(2) or ""
    direct_vision = "g" in flags  # -g: отдать картинки напрямую отвечающей модели (её vision)
    text_only = "t" in flags and not direct_vision  # -g включает медиа-обработку для фото
    must_search = "c" in flags
    debug = "d" in flags  # дамп полного user-message в asks/<ts>_<event_id>.txt
    want_voice = "v" in flags  # -v: ответить голосом (озвучка через Gemini TTS)
    # Режим голоса для промпта: force (флаг -v) / auto (включён .voice auto) / off
    voice_mode = "force" if (want_voice and tts_available) else ("auto" if (VOICE_AUTO and tts_available) else "off")
    user_tokens = (event.pattern_match.group(3) or "").split()
    usernames = [t.lstrip("@") for t in user_tokens if not t.startswith("!")]
    exclude_users = [t.lstrip("!").lstrip("@") for t in user_tokens if t.startswith("!")]
    question = event.pattern_match.group(4).strip()
    # Гостям: запрос > лимита → медиа НЕ режем, но vision-модель бесплатная (аудио — chirp как всегда)
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
    flags_str = " ".join(f for f, on in [("-t", text_only), ("-c", must_search), ("-d", debug), ("-v", want_voice), ("-g", direct_vision)] if on) or "—"
    users_str = ", ".join("@" + u for u in usernames) if usernames else "—"
    excludes_str = ", ".join("!@" + u for u in exclude_users) if exclude_users else "—"
    vision_label = "free" if vision_model == FREE_MEDIA_MODEL else (vision_model or get_active_media_model())
    log("ASK", f"Старт от {caller}: N={n} · флаги=[{flags_str}] · users=[{users_str}] · excludes=[{excludes_str}] · модель={model_label} · vision={vision_label} · detail={detail}")

    # .ask -g: проверяем, что активная отвечающая модель умеет vision напрямую
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
                f"Переключись на vision-модель через `.model` (например GLM-5 / Qwen / Kimi, или vision-модель OpenRouter), либо убери `-g`.")
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
            messages = messages[:n]
            log("ASK", f"iter_messages diag: raw={diag['raw']} · skip service={diag['service']} · команда={diag['self_cmd']} · excludes={diag['excluded']} → попало {len(messages) - (1 if anchor_id else 0)} (+якорь {1 if anchor_id else 0})")
            if anchor is not None:
                aut = _owner_label() if anchor.out else _user_label(anchor.sender)
                qprev = _preview(anchor.raw_text or (_media_tag(anchor) or ""), 60)
                log("ASK", f"Reply-якорь: id={anchor.id}, автор {aut}, «{qprev}»" + (" (исключён из контекста)" if anchor_id is None else ""))

        ordered = list(reversed(messages))
        t_collected = time.time()
        short = " (чат короче запроса)" if len(ordered) < n else ""
        log("ASK", f"Сбор: запрошено N={n}, фактически {len(ordered)} сообщ.{short}")

        # .ask -g: отбираем самые свежие фото (до лимита) для прямой отдачи модели.
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
                inline_ids=inline_ids, inline_images=inline_images,
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
                reply = await ask_agentic(context, question, must_search=must_search, caller=caller, ctx_tokens_est=ctx_tokens, voice_mode=voice_mode, images=images_sorted)
                t_llm = time.time()
                break  # успех
            except ContextOverflowError as e:
                log("ASK", f"Overflow при safety×{(safety_override or base_safety):.2f}: ctx={ctx_tokens} (tiktoken) → API: {e}")
                if retry_idx == len(safety_attempts) - 1:
                    reply = (f"⚠️ Контекст не влезает в окно модели даже при агрессивной обрезке "
                             f"(safety×{safety_attempts[-1] / base_safety:.1f}). "
                             f"Попробуй меньшее N или смени модель (.model).")
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
                    "=== .ask -d debug dump ===\n"
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
                await client.send_file(event.chat_id, bio, voice_note=True)
                t_sent = time.time()
                try:
                    await status.delete()
                except Exception:
                    pass
                log("ASK", f"Голосовой ответ на '{question[:60]}' отправлен (voice={ACTIVE_VOICE}, mode={voice_mode})")
                return
            notes.append("🔇 голос не сгенерировался")  # фолбэк на текст

        note = (" — " + "; ".join(notes)) if notes else ""
        prefix = f"{label}{note}:\n\n"
        # На текстовом пути срезаем возможный ведущий маркер [[VOICE]] (если авто-режим выбрал голос, но он упал).
        if reply.lstrip().startswith("[[VOICE]]"):
            reply = reply.lstrip()[len("[[VOICE]]"):].lstrip()
        # Чистим markdown-мусор (#/*) ДО нарезки на части — модель путает HTML и markdown.
        reply = _html_clean_markdown(reply)
        # Сначала отправляем ответ, потом удаляем статус — иначе сбой delete съест ответ.
        await send_long(event.chat_id, reply, prefix=prefix, parse_mode="html")
        t_sent = time.time()
        try:
            await status.delete()
        except Exception:
            pass
        log("ASK", f"Ответ на '{question[:60]}' отправлен (model={ACTIVE_MODEL}, text_only={text_only}, must_search={must_search}, users={usernames or '—'}, anchor={anchor_id}, dropped={dropped}, failed={failed})")
    except Exception as e:
        log("ASK", f"Ошибка команды .ask: {e}")
        traceback.print_exc()
        await set_status(f"⚠️ Ошибка при обработке .ask: {e}")
    finally:
        t_end = time.time()
        log("ASK", f"Тайминги: сбор={t_collected-t0:.1f}с · контекст(медиа)={t_ctx-t_collected:.1f}с · LLM={t_llm-t_ctx:.1f}с · отправка={t_sent-t_llm:.1f}с · итого={t_end-t0:.1f}с")


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.auto_reply$", from_users="me"))
async def auto_reply_on(event):
    AUTO_REPLY_ACTIVE_CHATS.add(event.chat_id)
    _save_auto_reply()
    log("AUTO", f"Авто-ответ включён в чате {event.chat_id}")
    await event.edit("✅ Авто-ответ включён")
    await asyncio.sleep(2)
    await event.delete()


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.auto_reply\s+off$", from_users="me"))
async def auto_reply_off(event):
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
    if event.raw_text and event.raw_text.startswith("."):
        return  # команды (.ask и пр.) не должны попадать в авто-ответ
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


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.song(?: |$)(.*)", from_users="me"))
async def song_command(event):
    custom_text = event.pattern_match.group(1).strip()
    text_to_print = custom_text if custom_text else SONG_TEXT
    await event.delete()
    await print_lyrics(event.chat_id, text_to_print)


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.channels(?:\s+(\w+))?(?:\s+(.+))?$", from_users="me"))
async def channels_command(event):
    global LAST_SCAN
    sub = (event.pattern_match.group(1) or "").lower()
    arg = (event.pattern_match.group(2) or "").strip()
    tracked = get_tracked()

    if not sub:
        if not tracked:
            await event.edit("Каналы не отслеживаются. `.channels scan` — найти, `.channels add @name` — добавить.")
            return
        lines = ["📡 Отслеживаемые каналы:"]
        for i, ch in enumerate(tracked, 1):
            uname = f"@{ch['username']}" if ch.get("username") else f"id{ch['id']}"
            lines.append(f"{i}. {uname} — {ch.get('title', '')}")
        lines.append("\n`.channels remove N` — убрать")
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
        lines.append("\n`.channels add N` — добавить по номеру")
        await event.edit("\n".join(lines)[:4000])
        return

    if sub == "add":
        if not arg:
            await event.edit("Укажи номер из scan или @username: `.channels add 3`")
            return
        if arg.isdigit():
            idx = int(arg) - 1
            if not LAST_SCAN:
                await event.edit("Сначала выполни `.channels scan` — список каналов не загружен.")
                return
            if not (0 <= idx < len(LAST_SCAN)):
                await event.edit("Нет такого номера. Сначала `.channels scan`.")
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
            await event.edit("Укажи номер или @username: `.channels remove 2`")
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

    await event.edit("Неизвестная подкоманда. `.channels`, `.channels scan`, `.channels add`, `.channels remove`")


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.search\s+(.+)$", from_users="me"))
async def search_command(event):
    query = event.pattern_match.group(1).strip()
    if not get_tracked():
        await event.edit("Нет отслеживаемых каналов. `.channels scan`")
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
        log("SEARCH", f"Ошибка .search: {e}")
        traceback.print_exc()
        await event.edit("Ошибка поиска, см. логи.")


async def send_digest(manual: bool):
    tracked = get_tracked()
    if not tracked:
        if manual:
            await client.send_message("me", "Нет отслеживаемых каналов. `.channels scan`")
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


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.digest$", from_users="me"))
async def digest_command(event):
    await event.edit("📰 Собираю дайджест…")
    try:
        await send_digest(manual=True)
        await event.delete()
    except Exception as e:
        log("DIGEST", f"Ошибка .digest: {e}")
        traceback.print_exc()


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.digest\s+time\s+(\d{1,2}:\d{2})$", from_users="me"))
async def digest_time_command(event):
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


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.model(?:\s+(.+))?$", from_users="me"))
async def model_command(event):
    global ACTIVE_MODEL, ACTIVE_MEDIA_MODEL
    arg = (event.pattern_match.group(1) or "").strip()
    slugs = list(MODEL_REGISTRY.keys())

    def is_available(provider):
        return _client_for_provider(provider) is not None

    def tool_mark(slug):
        ts = MODEL_TOOLS_SUPPORT.get(slug)
        return " 🔧" if ts is True else (" 🚫" if ts is False else " ❔")

    # --- выбор медиа-модели (vision): .model media [N|slug] ---
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
            lines.append("\n[OR]=OpenRouter · [OC]=OpenCode Go · аудио/голос — всегда Chirp-3.")
            lines.append("`.model media N` / `.model media <slug>` — выбрать")
            lines.append("`.model media <model-id>` — любая модель OpenRouter (с проверкой)")
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
        warn = "" if supports_img else "\n⚠️ Модель не поддерживает изображения — описание фото работать не будет (голос/аудио идут через Chirp)."
        await event.edit(f"✅ Медиа-модель (vision): `{marg}` (кастомная, OpenRouter){warn}")
        return

    # --- избранное (кастомные OpenRouter-модели): .model fav ---
    if arg.lower() in ("fav", "favorites", "избранное"):
        if not CUSTOM_MODELS:
            await event.edit("⭐ Избранное (кастомные OpenRouter-модели) пусто.\nДобавь: `.model vendor/model` (напр. `.model openai/gpt-4o`).")
            return
        lines = ["⭐ **Избранные OpenRouter-модели:**"]
        for i, (mid, ci) in enumerate(CUSTOM_MODELS.items(), 1):
            mk = "▶" if mid == ACTIVE_MODEL else " "
            lines.append(f"{mk}{i}. {ci.get('label') or mid} — `{mid}`")
        lines.append("\n`.model <vendor/model>` — выбрать/добавить · `.model remove <N|id>` — удалить")
        await event.edit("\n".join(lines)[:4000])
        return

    # --- удаление кастомной OpenRouter-модели: .model remove <N|slug> ---
    if arg.lower().startswith("remove"):
        marg = arg[len("remove"):].strip()
        fav_ids = list(CUSTOM_MODELS.keys())
        target = None
        if marg.isdigit() and 1 <= int(marg) <= len(fav_ids):
            target = fav_ids[int(marg) - 1]
        elif marg in CUSTOM_MODELS:
            target = marg
        if not target:
            await event.edit(f"Не нашёл кастомную модель: `{marg}`. `.model fav` — список избранных.")
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
                         "openrouter": "━━ OpenRouter (кастом) ━━", "oylan": "━━ Oylan (ISSAI) ━━"}.get(provider, f"━━ {provider} ━━")
                lines.append(f"\n{title}")
            mark = f"▶{i}." if slug == ACTIVE_MODEL else f"{i}."
            warn = " ⚠️нет ключа" if not is_available(provider) else ""
            lines.append(f"{mark} `{slug}` — {label} · 🪟{_fmt_ctx(ctx)}{tool_mark(slug)}{warn}")
        if ACTIVE_MEDIA_MODEL in MEDIA_MODEL_REGISTRY:
            media_label = MEDIA_MODEL_REGISTRY[ACTIVE_MEDIA_MODEL][1]
        elif ACTIVE_MEDIA_MODEL in MEDIA_OPENCODE_SLUGS and ACTIVE_MEDIA_MODEL in MODEL_REGISTRY:
            media_label = f"{MODEL_REGISTRY[ACTIVE_MEDIA_MODEL][2]} [OpenCode]"
        else:
            media_label = f"{ACTIVE_MEDIA_MODEL} (кастомная)"
        lines.append(f"\n🖼 медиа-модель: {media_label} · `.model media` — сменить")
        lines.append("`.model N` / `.model <slug>` — выбрать · `.model probe` — проверить поиск (❔→🔧/🚫)")
        lines.append("`.model vendor/model` — добавить ЛЮБУЮ модель OpenRouter по id (напр. `.model openai/gpt-4o`)")
        lines.append("`.model fav` — избранные OR-модели · `.model remove <N|id>` — удалить кастомную")
        await event.edit("\n".join(lines)[:4000])
        return

    if arg.lower() == "probe":
        await event.edit("🔧 Проверяю поддержку поиска у моделей…")
        tested = 0
        for slug in slugs:
            provider, mid, _label, _ctx, _safety = MODEL_REGISTRY[slug]
            if provider == "oylan":
                continue  # Oylan не OpenAI tool-calling — пропускаем проверку поиска
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
        await event.edit(f"🔧 Проверено моделей: {tested}. Смотри `.model`.")
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
            CUSTOM_MODELS[arg] = {"label": label, "ctx": ctx, "safety": 1.3, "vision": bool(supports_img)}  # vision — для .ask -g
            MODEL_REGISTRY[arg] = ("openrouter", arg, label, ctx, 1.3)
            ACTIVE_MODEL = arg
            _save_model_state()
            log("MODEL", f"Активная модель (кастомная OpenRouter): {arg}, окно {ctx}")
            await event.edit(f"✅ Модель ответов: {label} (`{arg}`, OpenRouter, окно 🪟{_fmt_ctx(ctx)})")
            return
        await event.edit(f"Нет такой модели: {arg}. `.model` — список, либо укажи id модели OpenRouter (vendor/model).")
        return

    provider, _mid, label, ctx, _safety = MODEL_REGISTRY[chosen]
    if not is_available(provider):
        await event.edit(f"Модель «{label}» недоступна — нет ключа провайдера ({provider}).")
        return

    ACTIVE_MODEL = chosen
    _save_model_state()
    log("MODEL", f"Активная модель: {chosen} ({label})")
    await event.edit(f"✅ Модель ответов: {label} (окно 🪟{_fmt_ctx(ctx)})")


def _sync_fish_search(query: str):
    """Поиск голосов Fish Audio: GET /model?title=&sort_by=score. Возвращает список {_id,title,languages}."""
    r = requests.get(FISH_MODELS_URL, headers={"Authorization": f"Bearer {fish_audio_api_key}"},
                     params={"title": query, "sort_by": "score", "page_size": 10}, timeout=30)
    r.raise_for_status()
    return (r.json() or {}).get("items", [])


async def _voice_fish_command(event, rest: str):
    """Подкоманды Fish: список избранного / search / add / remove / test / выбор."""
    global FISH_VOICE, FISH_FAVORITES, TTS_ENGINE
    if not fish_available:
        await event.edit("⚠️ Fish недоступен: нет `FISH_AUDIO_API_KEY` в .env.")
        return
    low = rest.lower()

    if low.startswith("search"):
        q = rest[len("search"):].strip()
        if not q:
            await event.edit("Использование: `.voice fish search <запрос>` (напр. `russian`, `male`, имя).")
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
        lines = [f"🔎 **Fish — результаты по «{q}»:**"]
        for it in items[:10]:
            langs = ",".join(it.get("languages") or [])
            lines.append(f"• {it.get('title','?')} — `{it.get('_id','')}`" + (f" ({langs})" if langs else ""))
        lines.append("\nДобавить: `.voice fish add <id> [имя]` → потом `.voice fish` для выбора.")
        await event.edit("\n".join(lines)[:4000])
        return

    if low.startswith("add"):
        parts = rest[len("add"):].strip().split(maxsplit=1)
        if not parts:
            await event.edit("Использование: `.voice fish add <reference_id> [имя]`.")
            return
        ref = parts[0]
        name = parts[1] if len(parts) > 1 else ref
        if any(f["id"] == ref for f in FISH_FAVORITES):
            await event.edit(f"Голос `{ref}` уже в избранном.")
            return
        FISH_FAVORITES.append({"id": ref, "title": name})
        _save_model_state()
        await event.edit(f"✅ Добавлен в избранное Fish: **{name}** (`{ref}`). Всего: {len(FISH_FAVORITES)}.\n`.voice fish` — список и выбор.")
        return

    if low.startswith("remove"):
        marg = rest[len("remove"):].strip()
        idx = None
        if marg.isdigit() and 1 <= int(marg) <= len(FISH_FAVORITES):
            idx = int(marg) - 1
        else:
            idx = next((i for i, f in enumerate(FISH_FAVORITES) if f["id"] == marg), None)
        if idx is None:
            await event.edit(f"Не нашёл в избранном: `{marg}`. `.voice fish` — список.")
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
            await event.edit("Сначала выбери Fish-голос: `.voice fish <N|id>` (см. `.voice fish`).")
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
        # список избранного
        lines = [f"🐟 **Fish Audio — избранные голоса** (движок сейчас: {TTS_ENGINE}):"]
        if not FISH_FAVORITES:
            lines.append("  (пусто) — найди через `.voice fish search <запрос>` и добавь `.voice fish add <id> [имя]`.")
        for i, f in enumerate(FISH_FAVORITES, 1):
            mk = "▶" if f["id"] == FISH_VOICE else " "
            lines.append(f"{mk}{i}. {f['title']} — `{f['id']}`")
        lines.append("\n`.voice fish search <q>` — найти · `.voice fish add <id> [имя]` · `.voice fish remove <N>`")
        lines.append("`.voice fish <N|id>` — выбрать · `.voice fish test` — прослушать · `.voice engine fish` — включить движок")
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
    hint = "" if TTS_ENGINE == "fish" else "\n⚠️ Сейчас движок gemini — включи `.voice engine fish`, чтобы озвучивать этим голосом."
    await event.edit(f"✅ Активный Fish-голос: `{FISH_VOICE}`.{hint}")


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.voice(?:\s+(.+))?$", from_users="me"))
async def voice_command(event):
    """Выбор голоса и режима озвучки для голосовых ответов в .ask (Gemini + Fish Audio)."""
    global ACTIVE_VOICE, VOICE_AUTO, TTS_ENGINE, FISH_VOICE, FISH_FAVORITES
    arg = (event.pattern_match.group(1) or "").strip()

    if not tts_available and not fish_available:
        await event.edit("⚠️ Голос недоступен: нет ни `GOOGLE_GENAI_API_KEY`, ни `FISH_AUDIO_API_KEY` в .env.")
        return

    low = arg.lower()

    # .voice engine [gemini|fish] — выбор TTS-движка
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
                         "\n`.voice engine gemini|fish` — сменить · при сбое движка — автофолбэк на другой")
        return

    # .voice fish ... — Fish Audio: список/поиск/добавление/выбор избранных голосов
    if low == "fish" or low.startswith("fish"):
        await _voice_fish_command(event, arg[len("fish"):].strip())
        return

    # .voice auto on|off — переключатель авто-голоса
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

    # .voice samples [N|имя] — прислать озвученные примеры (все 30 или один)
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
                         "\nВыбрать: `.voice N` или `.voice <имя>`.")
        return

    # .voice test [текст] — синтез примера текущим голосом
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
        lines.append("🎧 `.voice samples` — примеры ВСЕХ голосов · `.voice N`/`.voice <имя>` — выбрать · `.voice auto on|off`")
        lines.append("🐟 `.voice engine fish|gemini` — сменить движок · `.voice fish` — голоса Fish Audio (поиск/избранное)")
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
        await event.edit(f"Нет такого голоса: `{arg}`. `.voice` — список (номер или имя).")
        return
    ACTIVE_VOICE = chosen
    _save_model_state()
    prof = _voice_profile(chosen)
    log("TTS", f"Активный голос: {chosen}")
    await event.edit(f"✅ Голос: {prof['emoji']} **{chosen}** — {prof['personality']}\n`.voice test` — прослушать · `-v` в .ask — ответить голосом")


def _human_bytes(n):
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ТБ"


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.cache(?:\s+(\w+))?(?:\s+(\S+))?$", from_users="me"))
async def cache_command(event):
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
        lines.append("Очистить: `.cache clear all` · `.cache clear older 30` (дней)")
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
            await event.edit("Укажи число дней: `.cache clear older 30`")
            return
        await event.edit("`.cache clear all` или `.cache clear older 30`")
        return

    await event.edit("`.cache info` · `.cache clear all|older N`")


# Отдельный обработчик для `.cache clear older N` (3 аргумента, регулярка с 2 группами не покрывает)
@client.on(events.NewMessage(outgoing=True, pattern=r"^\.cache\s+clear\s+older\s+(\d+)$", from_users="me"))
async def cache_clear_older_command(event):
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


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.allow(?:\s+(.+))?$", from_users="me"))
async def allow_command(event):
    arg = (event.pattern_match.group(1) or "").strip()

    # список
    if not arg and not getattr(event, "reply_to", None):
        if not ALLOWED_USERS:
            await event.edit("Доступ к .ask ни у кого. Выдать: `.allow @username [лимит]` или ответом на сообщение.")
            return
        lines = ["✅ Доступ к .ask есть у:"]
        for i, (uid, rec) in enumerate(ALLOWED_USERS.items(), 1):
            uname = rec.get("username") if isinstance(rec, dict) else rec
            limit = rec.get("limit") if isinstance(rec, dict) else None
            who = ('@' + uname) if uname else str(uid)
            lines.append(f"{i}. {who} (id {uid}) · лимит: {_fmt_allow_limit(limit)}")
        lines.append(f"\nПри N > лимита — vision переключается на free-модель (текст остаётся).")
        lines.append("`.allow @name <N|unlimited>` — задать лимит · `.allow remove @name|<id>`")
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
        log("ALLOW", f"Доступ к .ask выдан {uid} (@{uname}), лимит={new_limit}")
        await event.edit(f"✅ Доступ к .ask выдан: {('@' + uname) if uname else uid} · лимит: {_fmt_allow_limit(new_limit)}")


def _help_index(active_label):
    return (
        "╭───────────────────────╮\n"
        "│   🤖  КОМАНДЫ БОТА   │\n"
        "╰───────────────────────╯\n"
        "\n"
        "Это «оглавление». Каждый раздел можно открыть подробно — допиши его\n"
        "название после `.help`. Пример: `.help media`.\n"
        "\n"
        "📂 **Разделы справки** (`.help <раздел>`):\n"
        "   `ask`       💬 вопросы к AI по чату — главная функция\n"
        "   `model`     🧠 выбор модели для текстовых ответов\n"
        "   `media`     🖼 vision-модели (картинки/видео-кружки) + метки [OR]/[OC]\n"
        "   `voice`     🎙 голосовые ответы: выбор голоса, флаг `-v`, эмоции\n"
        "   `keys`      🔑 какие API-ключи за что отвечают (что обязательно)\n"
        "   `channels`  📡 каналы, поиск, дайджест\n"
        "   `auto`      🔁 авто-ответ\n"
        "   `allow`     👥 доступ к `.ask` для других\n"
        "   `song`      🎵 печать с эффектом набора\n"
        "   `help`      ℹ️ как устроена сама эта команда\n"
        "   `all`       📖 показать ВСЁ сразу\n"
        "\n"
        "⚡ **Шпаргалка (самое частое):**\n"
        "   `.ask 200 о чём спорят?` — ответ по последним 200 сообщениям\n"
        "   `.ask 50 -t коротко` — без медиа (быстрее)\n"
        "   `.model` — сменить модель ответов · `.model media` — сменить «глаза»\n"
        "\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ Активная модель: **{active_label}**\n"
        f"💡 Не уверен с чего начать? Набери `.help ask`."
    )


_HELP_SECTIONS = {
    "ask": (
        "💬 **`.ask` — вопрос к AI по истории чата**\n"
        "\n"
        "Бот читает последние сообщения этого чата и отвечает на твой вопрос с опорой на них.\n"
        "\n"
        "📐 **СИНТАКСИС (порядок строго слева направо):**\n"
        "```\n"
        ".ask  N  [флаги]  [@юзеры]  вопрос\n"
        "  1   2     3         4        5\n"
        "```\n"
        "1️⃣ `.ask` — сама команда.\n"
        "2️⃣ `N` — **обязательно**, число: сколько последних сообщений взять (напр. `200`).\n"
        "3️⃣ `[флаги]` — необязательно: `-t`, `-c`, `-d`, `-v`, `-g` (см. ниже).\n"
        "4️⃣ `[@юзеры]` — необязательно: `@имя` (только эти) или `!@имя` (исключить).\n"
        "5️⃣ `вопрос` — **обязательно**, любой текст до конца строки.\n"
        "\n"
        "⚠️ **Порядок важен!** Флаги — ВСЕГДА перед `@юзерами`, оба — перед вопросом.\n"
        "   ✅ `.ask 500 -t @anna о чём писала?`\n"
        "   ❌ `.ask 500 @anna -t о чём писала?`  ← тут `-t` уедет в текст вопроса и не сработает.\n"
        "\n"
        "**Минимум:**\n"
        "   `.ask N вопрос`\n"
        "   _Пример:_ `.ask 300 сделай выжимку спора про цены`\n"
        "   Чем больше N — тем больше контекста, но дольше сбор и больше токенов.\n"
        "\n"
        "**Флаги** (шаг 3; можно несколько, слитно `-tc` или раздельно `-t -c`):\n"
        "   `-t` — текст без медиа: не распознаёт фото/голос/кружки → **быстрее и дешевле**.\n"
        "   `-c` — обязательно искать по подключённым каналам перед ответом.\n"
        "   `-d` — дамп: выгрузить собранный контекст отдельным файлом (для отладки).\n"
        "   `-v` — ответить **голосом** (озвучка через Gemini TTS). См. `.help voice`.\n"
        "   `-g` — отдать **картинки напрямую** отвечающей модели (её vision), а не описания.\n"
        "        Нужна vision-модель (`.model` → GLM-5/Qwen/Kimi или OpenRouter-vision), иначе понятная ошибка.\n"
        "        Голос/аудио всегда через Chirp-3. До 10 свежих картинок за запрос.\n"
        "   _Пример:_ `.ask 1000 -t -d что обсуждали вчера?` · `.ask 30 -v расскажи анекдот`\n"
        "\n"
        "**Фильтры по людям** (шаг 4):\n"
        "   `@user1 @user2` — взять сообщения **только** этих авторов.\n"
        "   `!@user` — **исключить** этого автора.\n"
        "   Комбинируется: `.ask 500 -t @anna !@bot о чём писала Аня?`\n"
        "\n"
        "**Ответом на сообщение (reply):**\n"
        "   Ответь `.ask вопрос` на чьё-то сообщение — бот возьмёт именно его + контекст вокруг.\n"
        "\n"
        "⏱ На больших N (10–15 тыс.) сбор истории идёт **в несколько потоков** — это норм, подожди.\n"
        "🔑 Работает на ключе **DeepSeek** (обязательный). Медиа в вопросе требует ключ OpenRouter/OpenCode — см. `.help keys`."
    ),
    "model": (
        "🧠 **Модель для ТЕКСТОВЫХ ответов** (`.model`)\n"
        "\n"
        "Это «мозг», который пишет ответ в `.ask`/`.search`/дайджестах.\n"
        "\n"
        "   `.model` — показать список моделей; стрелкой `▶` отмечена активная.\n"
        "   `.model N` — выбрать модель по номеру из списка.\n"
        "   `.model <slug>` — выбрать по короткому имени.\n"
        "   `.model vendor/model` — поставить **любую** модель OpenRouter по её полному ID\n"
        "        (со слешем). Бот сперва проверит, что такая модель существует.\n"
        "        _Пример:_ `.model anthropic/claude-3.5-sonnet`\n"
        "\n"
        "   `.model probe` — прогнать модели и проверить, у каких работает веб-поиск.\n"
        "\n"
        "**Избранное OpenRouter-моделей:**\n"
        "   `.model fav` — список добавленных кастомных моделей (быстрый выбор по номеру).\n"
        "   `.model remove <N|id>` — удалить кастомную модель из избранного.\n"
        "\n"
        "**Oylan (ISSAI):** провайдер `oylan` (модель Oylan). Это свой assistant-API\n"
        "   (не OpenAI): без tool-поиска через `-c` → используется встроенный websearch.\n"
        "   Нужен `OYLAN_API_KEY`. Выбор: `.model oylan`.\n"
        "\n"
        "**Медиа-кэш** (распознанные картинки/голос хранятся, чтобы не платить дважды):\n"
        "   `.cache info` — сколько занято.\n"
        "   `.cache clear all` — очистить весь кэш.\n"
        "   `.cache clear older N` — удалить записи старше N дней.\n"
        "\n"
        "🔑 По умолчанию активен **DeepSeek** (обязательный ключ). Модели OpenRouter\n"
        "    доступны только если вписан `OPENROUTER_API_KEY` — см. `.help keys`.\n"
        "🖼 За распознавание картинок отвечает ОТДЕЛЬНАЯ модель — `.help media`."
    ),
    "media": (
        "🖼 **Медиа-модели (vision)** — `.model media`\n"
        "\n"
        "Это «глаза» бота: модель, которая разбирает **картинки и видео-кружки** внутри `.ask`.\n"
        "Это НЕ та же модель, что пишет текст (её меняет `.model`).\n"
        "\n"
        "   `.model media` — показать список vision-моделей; `▶` — активная.\n"
        "   `.model media N` — выбрать по номеру.\n"
        "   `.model media <slug>` — выбрать по короткому имени (напр. `glm-5`).\n"
        "   `.model media <vendor/model>` — любая модель OpenRouter по ID (с проверкой, что она умеет vision).\n"
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
        "🎙 **Аудио и голосовые** распознаются ВСЕГДА отдельным движком **Chirp-3** —\n"
        "    этот список на голос не влияет.\n"
        "⚡ Хочешь быстрее/дешевле — флаг `-t` в `.ask` вообще пропускает медиа."
    ),
    "voice": (
        "🎙 **Голосовые ответы** (`.voice` + флаг `-v`)\n"
        "\n"
        "Бот может отвечать на `.ask` не текстом, а **живым голосовым** — через Google\n"
        "Gemini Flash TTS. Голос выбираешь ты; озвучка эмоциональная (с интонацией).\n"
        "\n"
        "**Выбор голоса:**\n"
        "   `.voice` — список 30 голосов; `▶` — активный.\n"
        "   `.voice N` — выбрать по номеру.\n"
        "   `.voice <имя>` — выбрать по имени (напр. `.voice Kore`).\n"
        "   `.voice samples` — прислать озвученные примеры ВСЕХ голосов (послушать и выбрать).\n"
        "   `.voice samples N` — пример одного голоса; `.voice test [текст]` — пример текущим голосом.\n"
        "\n"
        "**Когда бот отвечает голосом — два способа:**\n"
        "   1) Флаг `-v` в `.ask` — **форсит голос всегда**: `.ask 30 -v расскажи анекдот`.\n"
        "   2) Авто-режим — `.voice auto on`: модель сама решает, где голос уместнее\n"
        "      (эмоция, короткий личный ответ). `.voice auto off` — выключить.\n"
        "\n"
        "**Эмоции:** модель управляет интонацией аудио-тегами в тексте —\n"
        "   `[смеётся]`, `[шёпотом]`, `[взволнованно]`, `[с теплотой]`, `[серьёзно]`,\n"
        "   `[вздыхает]`, паузы — многоточием. Теги не произносятся, а задают подачу.\n"
        "\n"
        "**Движки TTS (Gemini / Fish Audio):**\n"
        "   `.voice engine fish|gemini` — выбрать движок (при сбое — автофолбэк на другой).\n"
        "   `.voice fish search <запрос>` — найти голоса Fish (покажет название + id + языки).\n"
        "   `.voice fish add <id> [имя]` — добавить голос в избранное; `.voice fish` — список избранного.\n"
        "   `.voice fish <N|id>` — выбрать голос (номер из избранного ИЛИ прямой id).\n"
        "   `.voice fish remove <N|id>` — убрать из избранного; `.voice fish test [текст]` — прослушать.\n"
        "   Голоса берутся с fish.audio (id = reference_id). Нужен `FISH_AUDIO_API_KEY`.\n"
        "\n"
        "ℹ️ Голосовой ответ — до ~1500 симв. (примерно минута-полторы речи) и идёт **только голосом**; если\n"
        "   TTS не сработал — бот автоматически пришлёт текст.\n"
        "🔑 Нужен ключ `GOOGLE_GENAI_API_KEY` (см. `.help keys`). Без него `.voice`\n"
        "   сообщит, что голос недоступен, а `.ask` будет отвечать текстом.\n"
        "♻️ Если Google-квота исчерпана/недоступна — бот автоматически озвучит через\n"
        "   OpenRouter (та же модель, нужен `OPENROUTER_API_KEY`)."
    ),
    "keys": (
        "🔑 **Какие API-ключи за что отвечают** (в файле `.env`)\n"
        "\n"
        "**ОБЯЗАТЕЛЬНЫЙ — только один:**\n"
        "   `DEEPSEEK_API_KEY` — «мозг» бота. С ним одним уже работают:\n"
        "      `.ask`, `.search`, `.digest`, авто-ответ. Без него бот не отвечает.\n"
        "\n"
        "**НЕОБЯЗАТЕЛЬНЫЕ** (без них бот НЕ падает — просто часть функций выключена):\n"
        "   `OPENROUTER_API_KEY` — даёт:\n"
        "      • распознавание картинок/кружков в `.ask` (vision-модели `[OR]`);\n"
        "      • возможность ставить любую модель OpenRouter для ответов (`.model vendor/model`).\n"
        "   `OPENCODE_API_KEY` — даёт vision-модели `[OC]` (Kimi / GLM / Qwen / MiMo) в `.model media`.\n"
        "   `GOOGLE_GENAI_API_KEY` — даёт **голосовые ответы** (`.voice`, флаг `-v`). Можно\n"
        "      указать несколько ключей через запятую или в `GOOGLE_GENAI_API_KEYS` (ротация).\n"
        "   `FISH_AUDIO_API_KEY` — альтернативный TTS-движок Fish Audio (`.voice engine fish`,\n"
        "      `.voice fish` — поиск/избранное голосов).\n"
        "   `OYLAN_API_KEY` — провайдер ответов **Oylan (ISSAI)**, модель Oylan (`.model oylan`).\n"
        "\n"
        "**Что будет без необязательных ключей:**\n"
        "   • Нет OpenRouter и OpenCode → текст разбирается нормально, но фото/кружки в `.ask`\n"
        "     не читаются (голос всё равно работает через Chirp-3).\n"
        "   • В списках `.model` / `.model media` недоступные модели помечены `⚠️нет ключа`.\n"
        "\n"
        "📌 Telegram-доступ (`api_id` / `api_hash`) — тоже обязателен, без него бот не запустится."
    ),
    "channels": (
        "📡 **Каналы, поиск и дайджест**\n"
        "\n"
        "**Управление списком каналов:**\n"
        "   `.channels` — показать подключённые каналы.\n"
        "   `.channels scan` — просканировать твои подписки и показать их id.\n"
        "   `.channels add N` или `.channels add @name` — добавить канал.\n"
        "   `.channels remove N` или `.channels remove @name` — убрать.\n"
        "\n"
        "**Поиск по каналам:**\n"
        "   `.search запрос` — найти релевантное в подключённых каналах (топ-10) и обобщить.\n"
        "\n"
        "**Дайджест:**\n"
        "   `.digest` — собрать дайджест по каналам прямо сейчас.\n"
        "   `.digest time 09:00` — присылать автоматически каждый день в указанное время.\n"
        "\n"
        "🔑 Работает на ключе DeepSeek; ключи OpenRouter/OpenCode тут не нужны."
    ),
    "auto": (
        "🔁 **Авто-ответ** (с памятью диалога)\n"
        "\n"
        "Бот сам отвечает на входящие сообщения в текущем чате, помня предыдущие реплики.\n"
        "\n"
        "   `.auto_reply` — включить в этом чате.\n"
        "   `.auto_reply off` — выключить.\n"
        "\n"
        "⚠️ Включай осознанно: бот будет писать от твоего имени. Память диалога ведётся\n"
        "    отдельно по каждому чату."
    ),
    "allow": (
        "👥 **Доступ к `.ask` для других людей**\n"
        "\n"
        "По умолчанию `.ask` доступен только тебе. Можно выдать доступ другим.\n"
        "\n"
        "   `.allow @name` — разрешить пользователю (безлимитно по умолчанию).\n"
        "   `.allow @name N` — разрешить, но не больше N запросов.\n"
        "   `.allow @name unlimited` — явный безлимит.\n"
        "   `.allow` в ответ на сообщение — выдать доступ его автору.\n"
        "   `.allow remove` (по @name или в ответ) — забрать доступ.\n"
        "\n"
        "💡 Удобно, чтобы дать другу пользоваться ботом без передачи аккаунта."
    ),
    "song": (
        "🎵 **`.song [текст]`** — печать с эффектом набора\n"
        "\n"
        "Постепенно «печатает» переданный текст, имитируя живой набор.\n"
        "Декоративная команда — на AI и ключи не влияет."
    ),
    "help": (
        "ℹ️ **Как пользоваться самой `.help`**\n"
        "\n"
        "   `.help` — оглавление: список всех разделов + быстрая шпаргалка.\n"
        "   `.help <раздел>` — подробная справка по одному разделу.\n"
        "   `.help all` — вывести ВСЕ разделы подряд (длинно).\n"
        "\n"
        "**Доступные разделы:**\n"
        "   `ask` · `model` · `media` · `voice` · `keys` · `channels` · `auto` · `allow` · `song` · `help`\n"
        "\n"
        "_Примеры:_\n"
        "   `.help ask`   — всё про вопросы к AI\n"
        "   `.help media` — про vision-модели и метки [OR]/[OC]\n"
        "   `.help keys`  — какой ключ обязателен, а какой нет\n"
        "\n"
        "💡 Регистр и лишние пробелы не важны: `.help  MEDIA` сработает как `.help media`."
    ),
}


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.help(?:\s+(\S+))?\s*$", from_users="me"))
async def help_command(event):
    _, _, active_label = get_active_model()
    arg = (event.pattern_match.group(1) or "").strip().lower()

    if not arg:
        await event.edit(_help_index(active_label))
        return

    if arg == "all":
        order = ["ask", "model", "media", "voice", "keys", "channels", "auto", "allow", "song", "help"]
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
            f"Открой оглавление командой `.help`."
        )
        return

    await event.edit(section + f"\n\n━━━━━━━━━━━━━━━━━━━━━\n⚙️ Активная модель: **{active_label}**  ·  `.help` — все разделы")


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

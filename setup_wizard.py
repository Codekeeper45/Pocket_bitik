#!/usr/bin/env python3
"""Interactive automated setup for Pocket Bitik.
Automatically configures environment, provider presets, validation, connection testing, and run scripts.
Credentials never appear in command arguments or shell history.
"""
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent

PRESETS = {
    '1': {
        'title': 'OpenRouter (сотни моделей: DeepSeek, Claude, Llama)',
        'base_url': 'https://openrouter.ai/api/v1',
        'default_model': 'deepseek/deepseek-chat',
        'vision': False,
        'dummy_key': False,
        'key_prompt': 'OpenRouter API key (sk-or-v1-...): ',
    },
    '2': {
        'title': 'DeepSeek Official (официальный deepseek-chat)',
        'base_url': 'https://api.deepseek.com/v1',
        'default_model': 'deepseek-chat',
        'vision': False,
        'dummy_key': False,
        'key_prompt': 'DeepSeek API key (sk-...): ',
    },
    '3': {
        'title': 'Локальная Ollama (бесплатно на своем ПК, без ключей)',
        'base_url': 'http://localhost:11434/v1',
        'default_model': 'llama3.2',
        'vision': False,
        'dummy_key': True,
        'key_prompt': 'Ключ не требуется (нажмите Enter): ',
    },
    '4': {
        'title': 'Groq (сверхбыстрый Llama 3.3)',
        'base_url': 'https://api.groq.com/openai/v1',
        'default_model': 'llama-3.3-70b-versatile',
        'vision': False,
        'dummy_key': False,
        'key_prompt': 'Groq API key (gsk_...): ',
    },
    '5': {
        'title': 'OpenAI Official (ChatGPT: gpt-4o-mini)',
        'base_url': 'https://api.openai.com/v1',
        'default_model': 'gpt-4o-mini',
        'vision': True,
        'dummy_key': False,
        'key_prompt': 'OpenAI API key (sk-...): ',
    },
    '6': {
        'title': 'Свой сервер / Кастомный URL (vLLM, LM Studio, etc.)',
        'base_url': '',
        'default_model': '',
        'vision': False,
        'dummy_key': False,
        'key_prompt': 'API key (для локального сервера можно любое слово): ',
    },
}

def quote(value):
    if any(c in value for c in ('\n', '\r', '\x00')) or '${' in value:
        raise ValueError('Multiline values and dotenv interpolation are not supported')
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'

def write_env(values, path, overwrite=False):
    path = Path(path)
    content = '# Private configuration. Never commit.\n' + ''.join(k + '=' + quote(v) + '\n' for k, v in values.items())
    if path.is_symlink():
        raise ValueError('Refusing symlink')
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
    flags |= os.O_TRUNC if overwrite else os.O_EXCL
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, 'w') as stream:
        if hasattr(os, 'fchmod'):
            os.fchmod(stream.fileno(), 0o600)
        stream.write(content)

def validate(values):
    if not values.get('api_id', '').isdigit() or int(values['api_id']) <= 0:
        raise ValueError('Invalid Telegram api_id')
    if not re.fullmatch(r'[a-fA-F0-9]{32}', values.get('api_hash', '')):
        raise ValueError('api_hash must contain 32 hexadecimal characters')
    u = urllib.parse.urlsplit(values.get('OPENAI_COMPAT_BASE_URL', ''))
    if u.scheme not in ('http', 'https') or not u.hostname or u.username or u.password or u.query or u.fragment:
        raise ValueError('Invalid API base URL')
    if u.scheme == 'http' and u.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Use HTTPS for remote providers')
    if not values.get('OPENAI_COMPAT_MODEL') or not values.get('OPENAI_COMPAT_API_KEY'):
        raise ValueError('Model and key are required (local servers may use a dummy key)')
    for v in values.values():
        quote(v)

def ask(label, default=''):
    prompt = f"{label} [{default}]: " if default else f"{label}: "
    val = input(prompt).strip()
    return val if val else default

def check_api_connection(base_url, api_key):
    """Тестовый пинг к /models для быстрой проверки доступности и ключа."""
    url = base_url.rstrip('/') + '/models'
    req = urllib.request.Request(
        url,
        headers={'Authorization': f'Bearer {api_key}', 'User-Agent': 'PocketBitikSetup/1.0'}
    )
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode('utf-8', errors='ignore'))
                models = [m.get('id') for m in body.get('data', []) if isinstance(m, dict) and m.get('id')]
                return True, models
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return False, str(e)
    return False, 'Unknown status'

def create_run_scripts(python_bin):
    """Создает исполняемый скрипт run.sh для удобного запуска одной командой."""
    run_sh = ROOT / 'run.sh'
    sh_content = f"""#!/usr/bin/env bash
# Автоматический запуск Pocket Bitik
cd "$(dirname "$0")"
exec "{python_bin}" bot_new.py "$@"
"""
    run_sh.write_text(sh_content)
    try:
        run_sh.chmod(0o755)
    except Exception:
        pass

    if os.name == 'nt' or sys.platform == 'win32':
        run_bat = ROOT / 'run.bat'
        bat_content = f"""@echo off
rem Запуск Pocket Bitik для Windows
cd /d "%~dp0"
"{python_bin}" bot_new.py %*
"""
        run_bat.write_text(bat_content)

def main():
    if sys.version_info < (3, 11):
        raise SystemExit('Требуется Python 3.11 или новее')
    os.chdir(ROOT)

    print('=' * 60)
    print(' Мастер автоматической установки и настройки Pocket Bitik')
    print(' Все секреты вводятся скрыто; .env и сессия защищены (0600)')
    print('=' * 60)

    # 1. Автоматическое виртуальное окружение
    if sys.prefix == sys.base_prefix and '--in-venv' not in sys.argv:
        ans = ask('Создать изолированное окружение .venv и установить зависимости? [Y/n]', 'Y')
        if ans.lower() in ('y', 'yes', 'д', 'да'):
            print('-> Создаем виртуальное окружение .venv...')
            subprocess.run([sys.executable, '-m', 'venv', str(ROOT / '.venv')], check=True)
            py = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

            print('-> Устанавливаем зависимости из requirements.txt...')
            use_uv = subprocess.run(['which', 'uv'], capture_output=True).returncode == 0
            if use_uv:
                subprocess.run(['uv', 'pip', 'install', '--python', str(py), '-r', str(ROOT / 'requirements.txt')], check=True)
            else:
                subprocess.run([str(py), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements.txt')], check=True)

            print('-> Зависимости установлены, перезапускаем мастер внутри окружения...\n')
            raise SystemExit(subprocess.call([str(py), str(Path(__file__)), '--in-venv']))

    p = ROOT / '.env'
    if p.exists() and ask('Файл .env уже существует. Перезаписать? Введите OVERWRITE для замены', 'N') != 'OVERWRITE':
        raise SystemExit('Текущая конфигурация сохранена без изменений.')

    # 2. Telegram учетные данные
    print('\n[1/3] Настройка Telegram API (с https://my.telegram.org)')
    raw_api = ask('Введите api_id (или скопируйте всю строку с my.telegram.org)')

    # Умный парсинг если скопирован блок целиком
    m_id = re.search(r'(?:api_id[:\s=]+)?(\d{5,10})', raw_api)
    m_hash = re.search(r'(?:api_hash[:\s=]+)?([a-fA-F0-9]{32})', raw_api)

    api_id = m_id.group(1) if m_id else raw_api
    if m_hash:
        api_hash = m_hash.group(1)
        print('-> api_hash распознан автоматически из скопированного текста.')
    else:
        api_hash = getpass.getpass('Telegram api_hash: ').strip()

    # 3. Выбор провайдера нейросетей
    print('\n[2/3] Выбор провайдера нейросетей (любой OpenAI-compatible):')
    for num, pinfo in PRESETS.items():
        print(f"  {num}) {pinfo['title']}")

    choice = ask('Выберите номер провайдера [1-6]', '1')
    preset = PRESETS.get(choice, PRESETS['1'])

    if choice == '6':
        base_url = ask('URL сервера включая /v1 (например, http://localhost:8000/v1)')
        model_id = ask('ID модели на сервере')
        key_input = getpass.getpass('API-ключ (для локального сервера можно любое слово): ').strip() or 'dummy-key'
        has_vision = ask('Поддерживает ли эта модель изображения (Vision)? [y/N]', 'N').lower() in ('y', 'yes', 'д', 'да')
    else:
        base_url = preset['base_url']
        print(f"-> Base URL: {base_url}")
        model_id = ask('ID модели', preset['default_model'])
        if preset['dummy_key']:
            key_input = 'ollama-local'
            print('-> Локальный провайдер: API-ключ не требуется.')
        else:
            key_input = getpass.getpass(preset['key_prompt']).strip()
        has_vision = preset['vision']

    # Быстрый тест связи с провайдером
    print('-> Проверяем подключение к провайдеру...')
    ok, details = check_api_connection(base_url, key_input)
    if ok:
        print('-> [УСПЕХ] Сервер ответил корректно!')
        if isinstance(details, list) and details:
            if model_id in details:
                print(f"-> Модель '{model_id}' подтверждена в списке моделей провайдера.")
            else:
                print(f"-> Доступно моделей: {len(details)}")
    else:
        print(f"-> [ПРЕДУПРЕЖДЕНИЕ] Тест связи не прошел: {details}")
        print("   (Конфигурация будет сохранена, но проверьте ключ и URL)")

    values = {
        'api_id': api_id,
        'api_hash': api_hash,
        'TELEGRAM_SESSION': str(ROOT / 'pocket_session'),
        'OPENAI_COMPAT_ENABLED': 'true',
        'OPENAI_COMPAT_BASE_URL': base_url,
        'OPENAI_COMPAT_MODEL': model_id,
        'OPENAI_COMPAT_API_KEY': key_input,
        'OPENAI_COMPAT_VISION': 'true' if has_vision else 'false'
    }

    validate(values)
    write_env(values, p, overwrite=p.exists())
    print('\n-> Конфигурация сохранена в .env (права доступа 0600).')

    # 4. Вход в Telegram прямо из мастера
    print('\n[3/3] Авторизация в Telegram:')
    ans_login = ask('Выполнить вход в Telegram прямо сейчас? [Y/n]', 'Y')
    if ans_login.lower() in ('y', 'yes', 'д', 'да'):
        try:
            from telethon.sync import TelegramClient
            client = TelegramClient(values['TELEGRAM_SESSION'], int(values['api_id']), values['api_hash'])
            client.start(
                phone=lambda: input('Номер телефона (+...): '),
                code_callback=lambda: getpass.getpass('Код подтверждения из Telegram: '),
                password=lambda: getpass.getpass('Пароль 2FA (если включен): ')
            )
            client.disconnect()
            session_file = Path(values['TELEGRAM_SESSION'] + '.session')
            if session_file.exists():
                session_file.chmod(0o600)
            print('-> [УСПЕХ] Сессия Telegram успешно создана и защищена (права 0600).')
        except Exception as e:
            print(f"-> Ошибка при авторизации: {e}")
            print('-> Вы сможете войти позже при первом запуске бота.')

    # 5. Генерация скриптов запуска
    create_run_scripts(sys.executable)
    print('\n' + '=' * 60)
    print(' Установка и настройка завершены!')
    print(' Запуск бота одной командой:')
    print('   ./run.sh')
    print('=' * 60)
    print('Примечание: текстовый чат, память и контекст работают через')
    print('выбранного провайдера. Генерация картинок и аудио требует отдельных')
    print('совместимых сервисов генерации медиа.')

if __name__ == '__main__':
    main()

"""Credential-free safety defaults for unittest discovery.

This package initializer runs before discovered ``tests.test_*`` modules import
``bot_new``. Keep all Telethon session artifacts outside the repository and
never borrow credentials from the developer's environment.
"""
import os
import tempfile

# Explicit inert values are required by bot_new's import-time configuration.
# Overwrite inherited values so tests cannot accidentally use real credentials.
os.environ.update({
    "api_id": "1",
    "api_hash": "test-only-not-a-real-telegram-hash",
    "CEREBRAS_API_KEY": "test-only-not-a-real-key",
    "TELEGRAM_SESSION": os.path.join(
        tempfile.mkdtemp(prefix="pocket-bitik-tests-"), "telegram-test-session"
    ),
})

# bot_new loads the repository .env with override=True. Its secrets must not
# become usable by imported SDK clients during tests. These dummy values take
# precedence over .env; unset optional provider keys even if present there.
for _key in (
    "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "OPENCODE_API_KEY",
    "GROK_API_KEY", "XAI_API_KEY", "CLIPROXY_API_KEY",
    "CHATGPT2API_AUTH_KEY", "TAVILY_API_KEY", "LLAMA_CLOUD_API_KEY",
):
    os.environ[_key] = ""

# bot_new explicitly loads its adjacent .env with override=True and manually
# copies dotenv_values into os.environ, bypassing PYTHON_DOTENV_DISABLED. Replace
# dotenv readers before bot_new imports their names, so secrets are never read.
import dotenv

def _no_dotenv_load(*_args, **_kwargs):
    return False

def _empty_dotenv_values(*_args, **_kwargs):
    return {}

dotenv.load_dotenv = _no_dotenv_load
dotenv.dotenv_values = _empty_dotenv_values

# Import-time SDK clients should not be allowed to reach networks. Tests that
# exercise network-facing behavior must mock the relevant client explicitly.
from .gen_test_bootstrap import install_network_guards
install_network_guards()

__all__ = ["install_network_guards"]

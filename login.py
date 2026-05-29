#!/usr/bin/env python3
"""Одноразовый вход в Telegram для создания свежего session_name.session.
Запусти интерактивно: python3 login.py — введёшь телефон и код из Telegram."""
import os
from telethon import TelegramClient
from dotenv import load_dotenv

load_dotenv()
api_id = int(os.getenv("api_id") or "0")
api_hash = os.getenv("api_hash") or ""

with TelegramClient("session_name", api_id, api_hash) as client:
    me = client.get_me()
    print(f"\n✅ Вход выполнен: {me.first_name} @{me.username or '—'} (id {me.id})")
    print("Сессия сохранена в session_name.session. Можно запускать бота.")

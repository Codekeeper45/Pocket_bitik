#!/usr/bin/env python3
"""Resolve AI-related channel usernames and create channels.json"""
import asyncio
import json
import os
from telethon import TelegramClient
from dotenv import load_dotenv

load_dotenv()

api_id = int(os.getenv("api_id") or "0")
api_hash = os.getenv("api_hash") or ""

# Публичные каналы по @username — впишите свои.
# Пример (замените на нужные вам):
AI_CHANNELS = [
    "@durov",
    "@telegram",
]

# Закрытые каналы по id (узнать id можно через .channels scan в боте) — впишите свои.
# Пример (замените на нужные вам):
PRIVATE_CHANNELS = [
    {"id": -1001234567890, "title": "Пример закрытого канала"},
]


async def main():
    client = TelegramClient("session_name", api_id, api_hash)
    await client.start()

    resolved = []

    for username in AI_CHANNELS:
        name = username.lstrip("@")
        try:
            entity = await client.get_entity(username)
            channel_id = entity.id
            title = getattr(entity, "title", name)
            real_username = getattr(entity, "username", None)
            resolved.append({
                "id": channel_id,
                "title": title,
                "username": real_username,
            })
            print(f"✅ {name} → {title} (id={channel_id})")
        except Exception as e:
            print(f"❌ {name} → {e}")

    # Add private channels
    for ch in PRIVATE_CHANNELS:
        try:
            entity = await client.get_entity(ch["id"])
            title = getattr(entity, "title", ch["title"])
            real_username = getattr(entity, "username", None)
            resolved.append({
                "id": ch["id"],
                "title": title,
                "username": real_username,
            })
            print(f"✅ {ch['title']} (id={ch['id']})")
        except Exception as e:
            print(f"❌ {ch['title']} → {e}")

    # Save
    with open("channels.json", "w", encoding="utf-8") as f:
        json.dump(resolved, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено {len(resolved)} каналов в channels.json")
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
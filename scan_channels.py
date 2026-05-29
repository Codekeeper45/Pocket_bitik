#!/usr/bin/env python3
"""Scan all channels and add AI-related ones to channels.json"""
import asyncio
import json
import os
from telethon import TelegramClient
from dotenv import load_dotenv

load_dotenv()

api_id = int(os.getenv("api_id") or "0")
api_hash = os.getenv("api_hash") or ""

CHANNELS_PATH = "channels.json"

AI_KEYWORDS = [
    "ai", "нейро", "нейросет", "gpt", "llm", "искусственн", "ии",
    "автоматизац", "промпт", "prompt", "machine learning", "deep learning",
    "нейрон", "робот", "bot", "боты", "n8n", "no code", "nocode",
    "lowcode", "low code", "smart", "умный", "digital", "диджитал",
    "синтез", "генерат", "openai", "chatgpt", "claude", "gemini",
    "midjourney", "stable diffusion", "dall-e", "dalle",
]


async def main():
    client = TelegramClient("session_name", api_id, api_hash)
    await client.start()

    # Load existing channels
    try:
        with open(CHANNELS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    existing_ids = {ch["id"] for ch in existing}
    print(f"Уже в списке: {len(existing)} каналов")

    # Scan all dialogs
    all_channels = []
    async for dialog in client.iter_dialogs():
        ent = dialog.entity
        if getattr(ent, "broadcast", False) and not getattr(ent, "megagroup", False):
            all_channels.append({
                "id": ent.id,
                "title": getattr(ent, "title", "") or "",
                "username": getattr(ent, "username", None),
            })

    print(f"Всего каналов в подписках: {len(all_channels)}")

    # Find AI-related channels not yet tracked
    new_channels = []
    for ch in all_channels:
        if ch["id"] in existing_ids:
            continue
        text = f"{ch['title']} {ch.get('username') or ''}".lower()
        if any(kw in text for kw in AI_KEYWORDS):
            new_channels.append(ch)

    print(f"\nНовые AI-каналы для добавления: {len(new_channels)}")
    for i, ch in enumerate(new_channels, 1):
        uname = f"@{ch['username']}" if ch.get("username") else f"id{ch['id']}"
        print(f"  {i}. {uname} — {ch['title']}")

    # Merge
    merged = existing + new_channels
    with open(CHANNELS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\nИтого в channels.json: {len(merged)} каналов")
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
"""Recover a final answer without repeating delivered tool output."""
import asyncio

FINAL_INSTRUCTION = ('Дай отдельный содержательный итоговый ответ на исходный запрос. '
                     'Адресные реплаи уже доставлены, не копируй их дословно. '
                     'Не вызывай инструменты: сейчас нужен только финальный ответ.')
FINAL_FAILURE = 'Не удалось получить финальный ответ модели после отправленных реплаев. Попробуй повторить запрос.'

async def recover_final(llm, model_id, messages, extract, max_tokens=16000):
    # Completed tool responses remain in context, but no tools are offered again.
    request = list(messages) + [{'role': 'user', 'content': FINAL_INSTRUCTION}]
    try:
        response = await asyncio.to_thread(llm.chat.completions.create,
            model=model_id, messages=request, max_tokens=max_tokens, temperature=0.7)
        if getattr(response, 'choices', None):
            msg = response.choices[0].message
            if not getattr(msg, 'tool_calls', None):
                text = extract(msg) if getattr(msg, 'content', None) else ''
                if text and text.strip():
                    return text
    except Exception:
        pass
    return FINAL_FAILURE

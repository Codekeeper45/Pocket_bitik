"""Classify explicit moderation signals separately from empty transport responses."""


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def response_status(response):
    """Return 'moderation', 'ok', or 'empty'; never infer moderation from emptiness."""
    prompt_feedback = _get(response, "promptFeedback") or _get(response, "prompt_feedback")
    if _get(prompt_feedback, "blockReason") or _get(prompt_feedback, "block_reason"):
        return "moderation"
    choices = _get(response, "choices") or []
    for choice in choices:
        message = _get(choice, "message")
        refusal = _get(message, "refusal")
        finish = _get(choice, "finish_reason")
        if refusal or (isinstance(finish, str) and finish.lower() in {"content_filter", "safety"}):
            return "moderation"
    return "ok" if choices else "empty"


def refusal_text(message):
    """Return the provider's actual refusal text without replacing it."""
    return _get(message, "refusal")


MODERATION_MESSAGE = "Запрос отклонён моделью по правилам безопасности."
EMPTY_RESPONSE_MESSAGE = "Модель вернула пустой ответ после повторного запроса; попробуй позже."

__all__ = ["response_status", "refusal_text", "MODERATION_MESSAGE", "EMPTY_RESPONSE_MESSAGE"]

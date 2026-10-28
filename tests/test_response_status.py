from types import SimpleNamespace as NS

from response_status import response_status, refusal_text


def test_explicit_moderation_signals():
    assert response_status({"promptFeedback": {"blockReason": "SAFETY"}}) == "moderation"
    assert response_status({"prompt_feedback": {"block_reason": "blocked"}}) == "moderation"
    assert response_status(NS(choices=[NS(message=NS(refusal="No"), finish_reason="stop")])) == "moderation"
    assert response_status(NS(choices=[NS(message=NS(refusal=None), finish_reason="content_filter")])) == "moderation"
    assert response_status(NS(choices=[NS(message=NS(refusal=None), finish_reason="SAFETY")])) == "moderation"


def test_empty_is_not_moderation_and_refusal_text_is_preserved():
    assert response_status(NS(choices=[])) == "empty"
    assert response_status(NS(choices=[NS(message=NS(content="answer", refusal=None), finish_reason="stop")])) == "ok"
    assert refusal_text(NS(content="", refusal="Provider refusal")) == "Provider refusal"
    assert refusal_text(NS(content="ordinary response", refusal=None)) is None

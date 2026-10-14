from types import SimpleNamespace
from gen_provider import transport_response_error


def test_ngrok_bandwidth_is_not_authentication_or_moderation():
    response = SimpleNamespace(status_code=403, text='<!DOCTYPE html><noscript>ERR_NGROK_725</noscript>', headers={'Content-Type': 'text/html'})
    error = transport_response_error(response)
    assert error.kind == 'transport_quota'
    assert not error.retryable
    assert 'ERR_NGROK_725' in str(error)


def test_html_bad_gateway_is_retryable_transport():
    response = SimpleNamespace(status_code=502, text='<html>Bad gateway</html>', headers={'Content-Type': 'text/html'})
    error = transport_response_error(response)
    assert error.kind == 'transport'
    assert error.retryable


def test_real_json_authentication_is_not_transport_html():
    response = SimpleNamespace(status_code=403, text='{"error":{"code":"invalid_api_key"}}', headers={'Content-Type': 'application/json'})
    assert transport_response_error(response) is None

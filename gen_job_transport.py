"""Short authenticated submit/poll requests for long-running image API jobs.

Avoids HTTP edge response timeouts without replaying a generation on poll errors.
"""
import time
import uuid
import requests


def post_image_job(base_url, endpoint, body, headers, *, timeout=660, session=requests, sleep=time.sleep):
    deadline = time.monotonic() + timeout
    request_headers = dict(headers)
    request_headers['Idempotency-Key'] = uuid.uuid4().hex
    job_url = base_url.rstrip('/') + '/image-jobs'
    payload = {'endpoint': endpoint, 'body': body}
    response = None
    # Safe to re-submit only with the same idempotency key.
    for attempt in range(3):
        try:
            response = session.post(job_url, headers=request_headers, json=payload, timeout=(15, 30))
            if response.status_code not in (502, 503, 504):
                break
        except requests.RequestException:
            if attempt == 2:
                from gen_contracts import ProviderError
                raise ProviderError('timeout', 'Image job submission outcome unknown; not replaying generation', 504, False)
        if attempt < 2:
            sleep(2)
    if response.status_code in (502, 503, 504, 520, 522, 524):
        from gen_contracts import ProviderError
        raise ProviderError('transport', 'Image job submission unavailable; outcome unknown, not replaying generation', response.status_code, False)
    if response.status_code != 202:
        return response
    job_id = response.json()['id']
    if not isinstance(job_id, str) or not job_id.isalnum():
        raise ValueError('invalid image job identifier')
    while time.monotonic() < deadline:
        sleep(3)
        try:
            response = session.get(job_url + '/' + job_id, headers=headers, timeout=(15, 30))
        except requests.RequestException:
            continue  # polling never replays a generation
        if response.status_code == 202:
            continue
        if response.status_code >= 500 and 'text/html' in response.headers.get('Content-Type', '').lower():
            continue
        return response
    from gen_contracts import ProviderError
    raise ProviderError('timeout', 'Image job polling deadline reached; generation is not re-submitted', 504, False)

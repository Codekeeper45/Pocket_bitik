import unittest
from unittest.mock import Mock
import requests
from gen_job_transport import post_image_job


def response(status, data=None, html=False):
    r = Mock(status_code=status, headers={'Content-Type': 'text/html' if html else 'application/json'})
    r.json.return_value = data
    return r


class JobTransportTests(unittest.TestCase):
    def test_poll_network_errors_do_not_resubmit(self):
        s = Mock()
        s.post.return_value = response(202, {'id': 'abc123'})
        final = response(200, {'data': [{'b64_json': 'test'}]})
        s.get.side_effect = [requests.Timeout(), response(530, html=True), response(202), final]
        self.assertIs(post_image_job('https://test/v1','/images/edits',{}, {},session=s,sleep=lambda _:None), final)
        self.assertEqual(s.post.call_count, 1)
        self.assertEqual(s.get.call_count, 4)

    def test_submit_retry_same_idempotency_key(self):
        s = Mock()
        s.post.side_effect = [requests.Timeout(), response(202, {'id':'abc123'})]
        s.get.return_value = response(200)
        post_image_job('https://test/v1', '/images/generations', {}, {},session=s,sleep=lambda _:None)
        self.assertEqual(s.post.call_count, 2)
        self.assertEqual(s.post.call_args_list[0].kwargs['headers']['Idempotency-Key'],s.post.call_args_list[1].kwargs['headers']['Idempotency-Key'])

    def test_upstream_error_is_preserved(self):
        s = Mock()
        s.post.return_value = response(202, {'id':'abc123'})
        s.get.return_value = response(400, {'error': {'code':'invalid_type'}})
        self.assertEqual(post_image_job('https://test/v1','/images/edits',{}, {},session=s,sleep=lambda _:None).status_code,400)

    def test_auth_rejection_not_retried(self):
        s=Mock()
        s.post.return_value=response(401)
        self.assertEqual(post_image_job('https://test/v1','/images/edits',{}, {},session=s,sleep=lambda _:None).status_code,401)
        s.get.assert_not_called()
        self.assertEqual(s.post.call_count,1)


if __name__ == '__main__': unittest.main()

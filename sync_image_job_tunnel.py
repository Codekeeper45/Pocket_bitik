"""Refresh Serverix image-job URL after a quick-tunnel restart, without restart.
Executed periodically by a user timer; the bot reads this small file per request.
"""
import json
import re
import subprocess
import sys
import requests

def main():
    log = subprocess.run(
        ['journalctl', '--user', '-u', 'pocket-image-jobs-tunnel.service', '-n', '1000', '--no-pager'],
        capture_output=True, text=True, check=True
    ).stdout
    urls = re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com', log)
    if not urls:
        # Tunnel might need restart to emit URL
        subprocess.run(['systemctl', '--user', 'restart', 'pocket-image-jobs-tunnel.service'], check=True)
        import time; time.sleep(4)
        log = subprocess.run(
            ['journalctl', '--user', '-u', 'pocket-image-jobs-tunnel.service', '-n', '100', '--no-pager'],
            capture_output=True, text=True, check=True
        ).stdout
        urls = re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com', log)
        if not urls:
            raise RuntimeError('image job tunnel URL missing')

    base = urls[-1]
    requests.get(base + '/health', timeout=15).raise_for_status()

    sys.path.insert(0, '/home/hermes/integrations/serverix')
    from serverix_ctl import ServerixClient
    c = ServerixClient()
    expected = json.dumps({'base_url': base + '/v1'})

    try:
        current = c.get_file_content('74e81a81', '/image_job_transport.json')
    except Exception as e:
        print(f"Warning: could not read current config from Serverix ({e})")
        current = ''

    if current != expected:
        try:
            c.write_file_content('74e81a81', '/image_job_transport.json', expected)
            print('image job endpoint refreshed:', base)
        except Exception as e:
            print(f"Notice: Serverix wings daemon temporarily unreachable: {e}")

if __name__ == '__main__':
    main()

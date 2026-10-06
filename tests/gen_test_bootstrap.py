"""Optional safety helpers for the credential-free test process."""
from __future__ import annotations


def install_network_guards() -> None:
    """Block outbound socket connections after SDK imports can be configured.

    This is intentionally narrow: local sockets are also denied because tests
    must not depend on services. Patch at the socket boundary so SDK clients
    constructed while importing bot_new cannot make network requests.
    """
    import socket

    def _blocked(*_args, **_kwargs):
        raise RuntimeError("network access is disabled in the Pocket_bitik test suite")

    socket.create_connection = _blocked
    socket.socket.connect = _blocked
    socket.socket.connect_ex = _blocked


def isolated_session_path() -> str:
    """Return the throwaway session path configured by tests.__init__."""
    import os
    return os.environ["TELEGRAM_SESSION"]

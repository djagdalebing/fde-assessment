"""Run an ASGI app on a real socket, in a background thread.

Needed because ``httpx.ASGITransport`` drains the whole response before it
returns - which is fine for correctness tests but makes any latency measurement
meaningless. Anything asserting on TTFT or incremental delivery has to go over
a real socket.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time

import uvicorn


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def live_server(app, timeout: float = 10.0):
    """Yield the base URL of ``app`` served by uvicorn on an ephemeral port."""
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout
    while not server.started:
        if time.monotonic() > deadline:
            server.should_exit = True
            raise TimeoutError("uvicorn did not start in time")
        time.sleep(0.01)

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)

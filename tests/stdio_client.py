"""A minimal, dependency-free JSON-RPC/stdio client.

Deliberately *not* built on the SDK client: the point of the Task 1 tests is to
assert what physically appears on the child's stdout, so the harness has to read
the raw bytes rather than trust an abstraction that would hide framing bugs.
"""

from __future__ import annotations

import json
import subprocess
import threading
from typing import Any

PROTOCOL_VERSION = "2026-07-28"


class StdioServerProcess:
    """Spawn an MCP server, speak JSON-RPC to it, capture stdout and stderr apart."""

    def __init__(self, args: list[str], cwd: str | None = None, env: dict | None = None) -> None:
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=cwd,
            env=env,
        )
        self._next_id = 0
        self.stdout_lines: list[str] = []
        self.stderr_text: list[str] = []
        # Drain stderr on a thread so a chatty server can never fill the pipe
        # buffer and deadlock the test.
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self.stderr_text.append(line)

    def _read_line(self, timeout: float = 10.0) -> str:
        assert self._proc.stdout is not None
        result: list[str] = []

        def reader() -> None:
            line = self._proc.stdout.readline()  # type: ignore[union-attr]
            result.append(line)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout)
        if not result:
            raise TimeoutError(f"server produced no stdout line within {timeout}s")
        line = result[0]
        if line == "":
            raise EOFError(f"server closed stdout; stderr was:\n{''.join(self.stderr_text)}")
        self.stdout_lines.append(line)
        return line

    def send(self, payload: dict[str, Any]) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        self.send(payload)
        while True:
            message = json.loads(self._read_line(timeout))
            # Skip server-initiated notifications; match on the id we sent.
            if message.get("id") == request_id:
                return message

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self.send(payload)

    def initialize(self) -> dict[str, Any]:
        response = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "fde-test-harness", "version": "1.0.0"},
            },
        )
        self.notify("notifications/initialized")
        return response

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
            self._proc.wait(timeout=5)

    def __enter__(self) -> "StdioServerProcess":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

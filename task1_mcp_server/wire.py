"""Ownership of the stdio wire. Standard library only, by design.

This module must be importable - and callable - *before* the MCP SDK or any
other dependency is imported, because the one gap the rest of the isolation
story cannot close is a library printing a banner at import time. So it pulls
in nothing but the standard library and lives on its own.

``python -m task1_mcp_server`` claims the wire here first, then imports the
server. The modules themselves stay import-safe, so a test can import
``task1_mcp_server.server`` without the process's stdout moving under it.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from collections.abc import Iterator
from typing import Any

if sys.platform != "win32":  # pragma: no branch
    import fcntl


class _UnownedBuffer(io.RawIOBase):
    """A binary layer that forwards writes but never closes what it wraps."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def write(self, data) -> int:
        return self._inner.write(data)

    def flush(self) -> None:
        self._inner.flush()

    def writable(self) -> bool:
        return True

    def close(self) -> None:
        # Deliberately does not close ``self._inner``.
        with contextlib.suppress(Exception):
            self._inner.flush()

    def fileno(self) -> int:
        return self._inner.fileno()


class StdoutToStderr:
    """A ``sys.stdout`` replacement that forwards everything to stderr.

    Defensive, not cosmetic: one ``print()`` reachable from a tool handler is
    enough to wedge a client's JSON parser mid-session, and the failure looks
    like a protocol bug rather than a logging bug.
    """

    def __init__(self, stderr) -> None:
        self._stderr = stderr

    def write(self, data: str) -> int:
        if data.strip():
            self._stderr.write(f"[stdout-diverted] {data}")
        else:
            self._stderr.write(data)
        return len(data)

    def flush(self) -> None:
        self._stderr.flush()

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def fileno(self) -> int:
        return self._stderr.fileno()

    @property
    def buffer(self):
        """The binary layer, wrapped so a caller cannot close stderr's.

        Handing out ``sys.stderr.buffer`` directly looked harmless and was not:
        the standard force-UTF-8 idiom,
        ``io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")``, closes the
        underlying buffer when it is collected - taking stderr with it and
        aborting the interpreter with "lost sys.stderr". The SDK ships an
        ``_UnownedTextWrapper`` for exactly this hazard; this is the same idea.
        """
        return _UnownedBuffer(self._stderr.buffer)

    @property
    def name(self) -> str:
        return "<stdout diverted to stderr>"

    def close(self) -> None:
        # Never close stderr on stdout's behalf.
        self.flush()

    def reconfigure(self, **kwargs) -> None:
        # Accepted and ignored: encoding is not ours to change, and raising
        # would break an import that is merely being tidy.
        return None

    def detach(self):
        raise io.UnsupportedOperation("stdout is diverted and cannot be detached")

    def __getattr__(self, name: str):
        """Everything else a file object is expected to answer.

        A library probing ``sys.stdout.closed``, ``.mode``, ``.errors``,
        ``.line_buffering`` or ``.isatty()`` must not get an AttributeError out
        of an import - the precise scenario this guard exists to survive.
        Forwarding to the stream we actually write to answers all of them, and
        answers them truthfully. The hand-written stubs this replaces had to be
        kept in step with the file protocol by hand and invented values where
        the real stream had a real one.

        Only the attributes with genuinely different semantics are defined
        explicitly above: ``write`` tags its output, ``buffer`` refuses
        ownership, ``close`` refuses to close stderr, ``detach`` refuses
        outright, and ``name`` says what this object is.
        """
        return getattr(self._stderr, name)

    def __iter__(self):
        return iter(())


def _dup_above_std(fd: int) -> int:
    """Duplicate ``fd`` onto a descriptor that cannot land in 0/1/2.

    ``os.dup`` returns the lowest free descriptor, which - if stdin happened to
    be closed - can be 0, 1 or 2. The duplicate would then be clobbered by the
    very redirection this function exists to set up.
    """
    if sys.platform == "win32":  # pragma: no cover
        duplicate = os.dup(fd)
        if duplicate <= 2:
            os.close(duplicate)
            raise OSError(f"duplicate of fd {fd} landed in the standard range")
        return duplicate
    try:
        return fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 3)
    except OSError as exc:
        # A closed fd 0 or 1 gave a raw traceback and exit 1. The Windows branch
        # above guards exactly this hazard; POSIX had no handling at all.
        raise OSError(
            f"file descriptor {fd} is not open - an MCP stdio server needs both "
            f"stdin and stdout connected to its client"
        ) from exc


@contextlib.contextmanager
def claim_stdio_wire() -> Iterator[Any]:
    """Take exclusive ownership of the JSON-RPC wire, then seal fd 0 and fd 1.

    The order matters, and getting it wrong is the classic stdio bug:

    1. ``dup`` the real stdout into a private descriptor - that duplicate is
       now the only handle on the wire, and nothing else in the process knows
       about it.
    2. Point fd 1 at fd 2. Every subsequent write to "stdout" by any code path
       - ``print``, a C extension, a forked child that inherits fd 1 - is now
       physically incapable of reaching the client.
    3. Point fd 0 at the null device for the same reason in reverse: a child
       process must not be able to steal bytes out of the client's requests.
    4. Swap ``sys.stdout`` for the diverting guard, so Python-level writes are
       additionally tagged on stderr instead of silently vanishing.

    The transport is handed the private duplicates explicitly. Everything is
    restored on the way out.
    """
    wire_out_fd = _dup_above_std(1)
    wire_in_fd = _dup_above_std(0)
    saved_fd1 = _dup_above_std(1)
    saved_fd0 = _dup_above_std(0)
    saved_stdout = sys.stdout
    null_fd = os.open(os.devnull, os.O_RDONLY)

    reader = writer = None
    try:
        os.dup2(2, 1)
        os.dup2(null_fd, 0)
        sys.stdout = StdoutToStderr(sys.stderr)  # type: ignore[assignment]
        writer = os.fdopen(wire_out_fd, "w", encoding="utf-8", buffering=1, closefd=True)
        reader = os.fdopen(wire_in_fd, "r", encoding="utf-8", errors="replace", closefd=True)
        yield reader, writer
    finally:
        sys.stdout = saved_stdout
        for handle, raw_fd in ((writer, wire_out_fd), (reader, wire_in_fd)):
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.flush()
                    handle.close()
            else:  # pragma: no cover - only if fdopen itself failed
                with contextlib.suppress(OSError):
                    os.close(raw_fd)
        with contextlib.suppress(OSError):
            os.dup2(saved_fd1, 1)
            os.dup2(saved_fd0, 0)
        for fd in (saved_fd1, saved_fd0, null_fd):
            with contextlib.suppress(OSError):
                os.close(fd)

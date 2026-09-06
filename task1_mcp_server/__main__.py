"""Entry point: ``python -m task1_mcp_server``.

The wire is claimed *before* the server module - and therefore before the MCP
SDK and everything it pulls in - is imported. A dependency that prints a banner
at import time is the most common real cause of a corrupted stdio stream, and
it is the one case a guard installed inside ``main()`` arrives too late to stop.
"""

from __future__ import annotations

import asyncio
import builtins
import os
import signal
import sys

from task1_mcp_server.wire import claim_stdio_wire


#: ``BaseExceptionGroup`` is a builtin from 3.11. Looked up rather than named
#: directly so this module still imports on 3.10, where the empty tuple makes
#: the ``except`` clause simply never match.
_EXCEPTION_GROUP = getattr(builtins, "BaseExceptionGroup", ())


def _flatten(group: BaseException) -> list[BaseException]:
    """Every leaf exception in a (possibly nested) exception group."""
    if _EXCEPTION_GROUP and isinstance(group, _EXCEPTION_GROUP):
        leaves: list[BaseException] = []
        for exc in group.exceptions:
            leaves.extend(_flatten(exc))
        return leaves
    return [group]


def _install_interrupt_handler() -> None:
    """Make Ctrl-C stop the process instead of being ignored.

    The SDK reads stdin on a worker thread parked in ``readline``. A
    KeyboardInterrupt goes to the MAIN thread, so the loop waits on a thread
    blocked in a read that never returns and the process sits there - alive
    after 10s once the reader has settled. (It exits promptly if the signal
    lands during startup, which is why the hang looks intermittent.)

    Closing the read descriptor from the handler was tried first and is not the
    answer: on this platform it does not wake a thread already blocked in the
    read, and installing a handler for SIGTERM as well broke the one signal
    that *did* terminate cleanly. So SIGINT exits directly, and SIGTERM is left
    on its default disposition, which already works.

    In-flight responses are lost on Ctrl-C. That is the right trade for an
    interactive interrupt - a protocol client shuts the server down by closing
    stdin, which still runs the full drain.
    """
    def _stop(signum, _frame):
        sys.stderr.write("interrupted\n")
        sys.stderr.flush()
        os._exit(130)

    try:
        signal.signal(signal.SIGINT, _stop)
    except (ValueError, OSError):  # pragma: no cover - not the main thread
        pass


def run() -> int:
    """Claim the wire, then serve. Returns a process exit code."""
    # Installed before the wire is claimed, so a Ctrl-C during startup is
    # handled too rather than surfacing as a traceback.
    _install_interrupt_handler()
    try:
        with claim_stdio_wire() as (wire_in, wire_out):
            # Imported inside the claim on purpose. Do not hoist.
            from task1_mcp_server.server import configure_logging, serve

            configure_logging()
            try:
                asyncio.run(serve(wire_in, wire_out))
            except (KeyboardInterrupt, EOFError):
                pass
    except _EXCEPTION_GROUP as group:
        # anyio wraps transport failures in a TaskGroup's exception group, and
        # a BaseExceptionGroup is NOT an OSError - so a client closing the pipe
        # mid-session ("python -m task1_mcp_server < bulk.jsonl | head -2")
        # escaped as a 46-line traceback carrying absolute paths, and exited 1.
        # The comment below claimed this class of crash was closed. It was not,
        # for the wrapped case.
        flat = _flatten(group)
        if flat and all(isinstance(exc, OSError) for exc in flat):
            sys.stderr.write(f"{flat[0]}\n")
            return 2
        raise
    except OSError as exc:
        # A closed or unusable fd 0/1 is a configuration problem, not a crash.
        # The wire helper raises a clear OSError; it used to propagate out of
        # module scope and print a chained traceback anyway.
        sys.stderr.write(f"{exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

"""Incremental Server-Sent Events parsing.

The same boundary problem as the redactor, one layer down: the HTTP byte
chunks an upstream sends do not align with SSE event boundaries. A single
``data:`` line can arrive in three TCP reads, and two events can arrive in one.
Anything that assumes ``chunk == event`` will drop or corrupt deltas under load,
and will usually pass a test suite that feeds it neatly-aligned fixtures.

``SSEDecoder`` therefore keeps a byte-level residue and only surfaces complete
events. It also decodes UTF-8 incrementally, because a multi-byte character can
be split across reads too - a real failure mode the moment a model emits an
emoji or a non-Latin script.
"""

from __future__ import annotations

import codecs
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class SSEEvent:
    event: str | None
    data: str
    #: ``id:`` and ``retry:`` are part of the EventSource contract - the client
    #: uses them to resume a dropped stream from where it left off, and to
    #: control its reconnect delay. Parsing them and then dropping them, which
    #: is what happened before, silently breaks resumption for every consumer.
    id: str | None = None
    retry: str | None = None

    @property
    def is_done(self) -> bool:
        return self.data.strip() == "[DONE]"


class SSEDecoder:
    """Feed bytes, get whole events out."""

    def __init__(self, max_event_bytes: int = 1024 * 1024) -> None:
        self._text_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""
        self._pending_cr = ""
        self._at_start = True
        self._max_event_bytes = max_event_bytes

    def feed(self, data: bytes) -> Iterator[SSEEvent]:
        decoded = self._pending_cr + self._text_decoder.decode(data)
        self._pending_cr = ""
        if self._at_start and decoded:
            # A leading BOM is stripped per the EventSource spec. Left in, it
            # became part of the first field name ("\ufeffdata"), so the entire
            # first event was silently discarded.
            decoded = decoded.lstrip("\ufeff")
            if decoded:
                self._at_start = False
        # Normalise only the newly arrived text. Re-normalising the whole
        # buffer on every chunk was quadratic: 1MB took 0.038s, 4MB took 0.779s.
        # A trailing lone "\r" is held back rather than converted, because its
        # "\n" may still be in the next chunk and converting early split one
        # multi-line event into two.
        if decoded.endswith("\r"):
            self._pending_cr = "\r"
            decoded = decoded[:-1]
        self._buffer += decoded.replace("\r\n", "\n").replace("\r", "\n")

        # A stream that never sends a blank line would otherwise buffer without
        # bound - 4MB of unterminated event became a 4MB string.
        if len(self._buffer) > self._max_event_bytes:
            raise ValueError(f"SSE event exceeded {self._max_event_bytes} bytes without a terminator")

        while "\n\n" in self._buffer:
            raw, self._buffer = self._buffer.split("\n\n", 1)
            event = self._parse_block(raw)
            if event is not None:
                yield event

    def flush(self) -> Iterator[SSEEvent]:
        """End of body: a final event may be missing its blank-line terminator."""
        tail = self._pending_cr + self._text_decoder.decode(b"", final=True)
        self._pending_cr = ""
        self._buffer += tail.replace("\r\n", "\n").replace("\r", "\n")
        remainder, self._buffer = self._buffer, ""
        if remainder.strip():
            event = self._parse_block(remainder)
            if event is not None:
                yield event

    @staticmethod
    def _parse_block(raw: str) -> SSEEvent | None:
        event_name: str | None = None
        event_id: str | None = None
        retry: str | None = None
        data_lines: list[str] = []
        for line in raw.split("\n"):
            if not line or line.startswith(":"):  # blank or comment/keep-alive
                continue
            field, _, value = line.partition(":")
            # "field: value" - exactly one leading space is stripped per spec.
            value = value.removeprefix(" ")
            if field == "data":
                data_lines.append(value)
            elif field == "event":
                event_name = value
            elif field == "id":
                event_id = value
            elif field == "retry":
                retry = value
        if not data_lines:
            return None
        return SSEEvent(event=event_name, data="\n".join(data_lines), id=event_id, retry=retry)


def encode_sse(
    data: str, event: str | None = None, event_id: str | None = None, retry: str | None = None
) -> bytes:
    """Re-encode an event, preserving the fields a client needs to resume."""
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    if event is not None:
        lines.append(f"event: {event}")
    if retry is not None:
        lines.append(f"retry: {retry}")
    lines.append(f"data: {data}")
    return ("\n".join(lines) + "\n\n").encode()

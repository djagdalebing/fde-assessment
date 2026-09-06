"""Mock upstream LLM provider emitting OpenAI-style SSE.

Its job in the test suite is to be *hostile* about framing: PII is deliberately
split across deltas character by character, and the SSE events are written into
the socket in byte groupings that do not align with event boundaries. A
guardrail that only works on tidy input fails here.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="mock-llm-provider")

DEFAULT_SCRIPT = (
    "Sure. I found the account. ",
    "The contact email is ad",
    "a.lovelace",
    "@examp",
    "le.com",
    " and the card on file is 4111 ",
    "1111 1111 ",
    "1111",
    ". The SSN on record is 123",
    "-45-",
    "6789",
    ". Support line: 555-123-4567. ",
    "Nothing else to report.",
)

#: Named scripts the tests select with ``{"scenario": "..."}``.
SCENARIOS: dict[str, tuple[str, ...]] = {
    "default": DEFAULT_SCRIPT,
    "clean": ("Your order ", "shipped on ", "Tuesday and ", "arrives Friday."),
    "pii_at_end": ("All set. Reach me at ", "grace", "@example", ".com"),
    "single_char": tuple("Email: ada@example.com done."),
    "one_shot": ("Email ada@example.com and card 4111111111111111 done.",),
    "unicode": ("Réservation confirmée 🎉 ", "contact: ada@example.com", " — merci!"),
    "adjacent": ("a@b.co ", "c@d.co ", "e@f.co"),
    "false_positives": (
        "Order 1234567890123456 is not a card. ",
        "Version 1.2.3. Ratio 123-45-6789 is an SSN though.",
    ),
}


def chunk_payload(content: str, model: str, created: int, finish_reason=None) -> dict:
    return {
        "id": "chatcmpl-mock-0001",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content} if content else {}, "finish_reason": finish_reason}],
    }


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    model = body.get("model", "mock-model-v1")
    scenario = body.get("scenario", "default")
    delay = float(body.get("chunk_delay", 0.0))
    pieces = SCENARIOS.get(scenario, DEFAULT_SCRIPT)

    if not body.get("stream"):
        text = "".join(pieces)
        return JSONResponse(
            {
                "id": "chatcmpl-mock-0001",
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            }
        )

    created = int(time.time())

    async def generate() -> AsyncIterator[bytes]:
        yield b'data: ' + json.dumps(chunk_payload("", model, created)).encode() + b"\n\n"
        for piece in pieces:
            if delay:
                await asyncio.sleep(delay)
            yield b"data: " + json.dumps(chunk_payload(piece, model, created)).encode() + b"\n\n"
        yield b"data: " + json.dumps(chunk_payload("", model, created, finish_reason="stop")).encode() + b"\n\n"
        yield b"data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}

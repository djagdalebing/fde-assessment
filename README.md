# FDE Assessment — MCP & LLM Gateway Tasks

Four runnable Python services: a strict MCP server, an MCP security gateway, a
streaming PII guardrail, and a rate-limiting model router with failover.

**564 tests pass** (138 / 109 / 206 / 111).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run.sh test          # full suite
./run.sh demo2         # MCP gateway: a viewer is refused an admin_ tool
./run.sh demo3         # streaming guardrail: TTFT and redacted output
./run.sh demo4         # router: 3000ms failover, then rate-limit refusal
```

`./run.sh` with no argument lists every service and port.

## Scope

Everything here earns its place against the assessment text or one of that
task's three stated evaluation criteria.

| Implemented | Criterion it serves |
|---|---|
| fd-level stdout claim, `sys.stdout` guard, unowned `buffer` | Task 1 — *STDIO Isolation* |
| synthesised `-32700`/`-32600`, EOF drain, one `data.errors` shape, `outputSchema` | Task 1 — *Protocol Compliance* |
| strict Pydantic, `CUST-XXXXX`, positive `amount`, `reason` floor/ceiling and non-blank check, magnitude and sub-cent guards | Task 1 — *Validation* |
| envelope validation, `NaN`/`Infinity` rejection, re-serialised forwarding, downstream response validation | Task 2 — *Parsing JSON-RPC wire format* |
| pooled client, service-token swap, header sanitisation, upstream error suppression, body and response caps | Task 2 — *Proxy middleware construction and HTTP forwarding* |
| `^admin_` rule table, refusal before forwarding, near-miss name normalisation, `-32001` vs `-32003`, JSON-RPC framing on HTTP errors | Task 2 — *method-level authorization and clean error handling* |
| incremental SSE decode, hold-back buffer, per-stream redactor state, residue before `finish_reason` | Task 3 — *async stream chunking and buffer state management* |
| single compiled alternation, first-char reject, bounded rescue pass, Luhn/SSA validators, separate keyword-SSN engine | Task 3 — *performant regex matching over partial text streams* |
| O(1) memory in response length, large-delta offload, early release of safe bytes | Task 3 — *memory efficiency and low-latency proxying* |
| `asyncio.timeout` wall-clock deadline, shielded reserve/settle, orphan drain, dedicated limiter executor | Task 4 — *async concurrency and timeout race conditions* |
| sliding window, `BEGIN IMMEDIATE`, real eviction, fail-closed estimator, charge clamp, per-API-key budget | Task 4 — *rate-limiter state eviction and token tracking* |
| failover taxonomy, `Retry-After` propagation, constant messages, `details` allow-list, success-path field allow-list | Task 4 — *graceful fallback and standardized error sanitization* |

## Task 1 — MCP server with strict validation and stdio isolation

`task1_mcp_server/` · `./run.sh task1` · 138 tests

Two tools — `get_customer_record` and `trigger_refund` — on the official SDK's
low-level `Server` over stdio. The low-level API rather than `MCPServer` because
the task is graded on JSON-RPC error mapping, and that needs direct control over
whether a failure becomes an error frame or a result.

**Two failure kinds, two wire shapes.** The distinction a client needs in order
to behave sensibly:

| Failure | Wire shape | Why |
|---|---|---|
| Bad `customer_id`, non-positive `amount`, short `reason`, unknown tool | JSON-RPC error, `-32602` | The call never happened. Retrying unchanged is pointless; the agent must fix the request. |
| Customer not found, refund over balance | Result with `isError: true` | The call happened. The model should see the text and reason about it. |
| Malformed JSON, bad envelope, unusable `id` | `-32700` / `-32600` | The frame was never a request. |
| Anything unexpected in a handler | `-32603`, message `"Internal server error"` | The client gets a code; the operator gets the detail in the log. |

Validation is Pydantic with `extra="forbid"`, `strict=True` and `frozen=True`,
so a wrong *type* is a rejection rather than a coercion: `"100.00"` is not a
float and `amount: true` is not a number. Every `-32602` raised by the schema
layer carries a per-field `data.errors` breakdown, so an agent can see which
argument was wrong instead of guessing. The SDK's own rejections are normalised
into that same shape, so `error["data"]["errors"][0]["code"]` reads the same way
whichever layer refused.

**Money is quantised once.** A refund converts to integer cents a single time,
and both the receipt and the ledger movement derive from that one number, so the
amount a client is told it was charged is exactly the amount that left the
balance. A refund that rounds to zero cents is refused rather than accepted as a
no-op, and amounts too large to express in cents are rejected before conversion.
A rejected refund is proven to have no side effect.

**stdout isolation.** `wire.claim_stdio_wire()` `dup`s the real stdout into a
private descriptor, then points fd 1 at fd 2 and fd 0 at `/dev/null`. Every
subsequent write to "stdout" by any code path is then physically incapable of
reaching the client, and the transport is handed the private duplicate
explicitly. `sys.stdout` is additionally swapped for a guard that tags and
forwards to stderr, and logging goes to stderr only.

The test fires six attacks from inside a tool handler — `print`,
`sys.stdout.write`, `os.write(1, ...)`, `sys.__stdout__`, a subprocess
inheriting fd 1, and a raw `libc write(2)` through ctypes — and asserts the
stream still parses and every byte surfaced on stderr.

> Note: mcp 2.x's own `stdio_server()` performs a similar fd diversion when it
> claims fd 1 itself. It does not here, because we hand it our own descriptors —
> so this is our mechanism, not a restatement of the SDK's.

**`-32700` and `-32600` are synthesised.** The SDK discards a malformed line
before the run loop sees it, so a client would otherwise wait on a response that
never comes. `LineRecordingInput` keeps each raw line in a FIFO and
`protocol_error_pump` pairs it back to the failure to build the right frame and
recover the `id` where one is recoverable.

**Responses are drained before exit.** When stdin closes, `DrainTracker` waits
for every dispatched request id to be answered before the process ends. It
counts ids rather than holding a set, so a client that reuses an id still gets
every receipt for work that was actually done.

---

## Task 2 — MCP security gateway

`task2_mcp_gateway/` · `./run.sh task2-gateway` (+ `task2-downstream`) · 109 tests

A JSON-RPC reverse proxy that authenticates a Bearer token, resolves a role, and
refuses `admin_*` tool calls from non-admins with `-32001 Unauthorized Tool
Call`. `tools/list` is forwarded transparently, as specified.

The load-bearing property is stronger than "returns an error": **a denied call is
never forwarded**. The mock downstream does no authorization of its own and
records every request it receives, so tests assert its call log is *empty*. A
gateway that filtered the response instead of the request would pass a weaker
test and fail this one.

Decisions worth flagging:

- **The gateway forwards a re-serialised copy of its own parsed object**, never
  the client's raw bytes. This structurally eliminates parser differentials —
  the class of bug where the gateway and the downstream disagree about what a
  payload means.
- **Policy is a table, not an `if`.** The `^admin_` rule is one `ToolRule`, so a
  second rule is a row rather than a branch. The gate is also checked against a
  normalised tool name, which can only ever refuse more.
- **The caller's token is not forwarded upstream.** The gateway authenticates as
  itself and asserts identity in `X-MCP-User`/`X-MCP-Role`. Nothing is copied
  from the client request, so a viewer sending `X-MCP-Role: admin` is not
  believed.
- **Auth failures are uniform.** Unknown, malformed and missing tokens produce a
  byte-identical response, and token comparison is constant-time. Authentication
  answers `-32003`, distinct from the `-32001` the brief assigns to the
  authorization decision, so a client can tell "my token is invalid" from "my
  role is insufficient" by code alone.
- **Upstream errors are sanitised.** A non-2xx downstream body is logged and
  replaced — those bodies carry internal hostnames, DSNs and stack traces — and
  that check runs before the transport branch, so an error framed as SSE is
  sanitised like any other.
- **The downstream's reply is validated like the request.** A mismatched `id`, a
  body carrying both `result` and `error`, or one that is not a Response object
  is refused rather than relayed.
- **Both directions are capped.** The inbound cap protects the gateway from its
  clients as the bytes arrive; the response cap protects it from the server it
  proxies to.

---

## Task 3 — Streaming PII guardrail

`task3_streaming_guardrail/` · `./run.sh task3-gateway` (+ `task3-provider`) · 206 tests

An LLM gateway that redacts the three classes the brief names — emails, SSNs and
credit cards — from an SSE stream in flight. It does not redact anything else: a
phone number in the demo output passes through untouched, deliberately.

**The problem.** A model streams `"Contact ada"`, then `"@exampl"`, then
`"e.com now"`. No single chunk looks like an email; the concatenation does. A
redactor that buffers the whole response to be safe destroys the point of
streaming.

**The approach.** Keep a bounded, *raw* tail buffer and each chunk ask one
question: could the entire remainder of the buffer still sit inside a match?
`regex`'s `fullmatch(..., partial=True)` answers it. Everything before that point
is final — redact and emit. Keep the rest.

The invariant the tests assert: **every chunking of a stream produces the same
output as redacting the text whole** — per-character chunks, eleven fixed chunk
sizes, 200 random splits of a fixture, and 2,000 fuzzed contexts around each
pattern. Memory is O(hold-back), not O(response) — a 100k-token response holds
the same bytes as a 10-token one, and that is measured rather than asserted.

**Validators decide, not the pattern.** A candidate run of digits is accepted as
a card only if it is 13–19 digits, its prefix and length agree with a real
scheme (Visa, Amex, Mastercard, Discover, Diners, JCB, UnionPay), *and* it
passes Luhn — cheapest test first. That is what lets the pattern stay broad
enough to catch a PAN written in groups of four, five or six, across spaces,
dashes, dots, non-breaking spaces or newlines in a markdown list, without
redacting ordinary numbers.

Two details that are not decoration:

- **A validator veto rescans for a shorter match at the same position.**
  `"4111 1111 1111 1111 12500"` matches greedily as five groups and fails Luhn;
  the rescan is what finds the valid card inside it.
- **A window of already-emitted text is carried as left context.** Every pattern
  is anchored by a lookbehind; emitting text throws that context away, and the
  window is what keeps the streaming result identical to the whole-text one.
- **Large deltas move off the event loop.** Redaction above 4 KB runs in a
  worker thread, so one tenant's oversized delta does not add latency to a
  co-tenant's request.

Latency is measured against uvicorn on a real port — `httpx.ASGITransport`
drains a whole response and cannot tell a streaming gateway from a buffering
one. The test asserts the first content delta arrives in under half the total
stream time against a provider that trickles; `./run.sh demo3` reports **47 ms
TTFT on a 658 ms stream**, and throughput on prose is **914k chars/s** through
one redactor.

---

## Task 4 — Rate limiting and model fallback

`task4_model_router/` · `./run.sh task4-router` (+ mock providers) · 111 tests

**Sliding window, not a fixed bucket.** A calendar-minute bucket lets a tenant
spend 50k at 11:59:59 and 50k again at 12:00:00 — the exact burst the limit
exists to prevent. A test constructs that burst directly. The budget is keyed on
the API key, as the brief specifies.

**Reserve, then reconcile.** The token cost of a completion is not known until
it finishes, but the limit has to be enforced before the call. Each request
reserves prompt tokens plus the requested completion ceiling and settles to the
provider's reported usage when the response lands. A request that fails before a
provider generated releases its hold; one where a provider generated and the
response was unusable is charged, because those tokens are spent either way. The
estimator fails closed — deeply nested or non-string payloads are measured from
their serialised form rather than skipped — and a settled charge is clamped
relative to the tenant's own limit.

**The race is closed in SQL.** The check and the insert share one
`BEGIN IMMEDIATE` transaction, so N concurrent requests cannot all read the same
"current usage" and all proceed. 2,000 simultaneous requests against a 50,000
budget are admitted up to the limit and no further, and the ledger matches real
spend exactly.

**State is on disk.** SQLite in WAL mode, with a covering index on
`(tenant_key, created_ms, tokens)` so the hot query is an index range scan.
An in-memory counter would hand a crash-looping gateway a fresh 50k per restart.
Expired rows are evicted inside each reservation, bounded per call so clearing a
backlog cannot hold the write lock. Limiter I/O runs on its own sized executor,
so queue wait never lands inside a provider's deadline.

**The deadline is wall-clock.** `asyncio.timeout` wraps each attempt rather than
relying on httpx's *per-phase* timeouts, which a trickling provider resets
indefinitely. On expiry the in-flight request is cancelled and its connection
released before the secondary is tried. `./run.sh demo4` fails over in 3 s
against a primary that hangs for 10 s. Failover covers a provider's credential
and capacity errors as well as timeouts, so a rotated key moves traffic to the
secondary instead of taking the route down; a genuine caller error is returned
as the caller's own, not retried.

**Errors are sanitised by construction.** Messages are constants, never built
from upstream data, and `details` is an allow-list — so a new field cannot leak
by being added upstream. Provider error bodies, endpoints and exception text go
to the log under a request id, never to the client. Every provider rate-limited
surfaces as a `429` carrying the provider's own `Retry-After` when it sent one.

Two concurrency details that took a while to get right:

- **The reservation is shielded and its orphan drained.** The `to_thread` hop
  can be cancelled after SQLite has committed the reservation but before the
  router receives it; the orphaned hold is found and released rather than left
  to expire.
- **A settled reservation is never released.** A cancellation arriving after
  `settle()` begins must not also run `release()`, or a client that hangs up at
  the right moment gets a completion for free.

---

## Verification

Each task was reviewed by an independent grader against its own three criteria,
told to run the code and reproduce before reporting, across several rounds.

Every fix carries a regression test that fails against the code before it, and
every one was mutation-tested: the source is broken on each load-bearing path to
confirm the new test fails against it, and the mutation is asserted to have
applied before the result is believed. That is what distinguishes a test that
holds a property from one that merely passes.

The properties the suite holds, rather than the cases it happens to cover:

- A denied tool call never reaches the downstream — asserted on the downstream's
  own call log, not on the gateway's response.
- Every chunking of a stream redacts to the same bytes as the whole text.
- Memory held by the guardrail is flat in response length.
- Concurrent requests are admitted up to the budget and no further, with the
  ledger matching real spend exactly.
- A rejected refund moves no money, and the receipt equals the ledger movement.

## Layout

```
task1_mcp_server/       wire.py  __main__.py  schemas.py  store.py  server.py
task2_mcp_gateway/      auth.py  policy.py  gateway.py  errors.py  mock_downstream.py
task3_streaming_guardrail/  redactor.py  sse.py  gateway.py  mock_llm.py
task4_model_router/     limiter.py  tokens.py  providers.py  router.py  app.py
tests/                  one file per task, plus stdio and live-server harnesses
```

## Notes

- `CUST-XXXXX` is read as the literal prefix plus **five ASCII digits**; the
  pattern is one constant in `schemas.py` if that reading is wrong.
- The guardrail streams on every path. The task requires that the full response
  is never accumulated before forwarding, so `stream: false` is answered with
  SSE — a deliberate reading of the task over OpenAI compatibility.
- Tenant API keys, bearer tokens and provider credentials are literals for
  demonstration. In a real deployment they resolve against an identity provider;
  `TokenVerifier` and the `API_KEYS` map are the seams where that swap happens.
- `demo4` lowers the limit to 300 tokens/minute so the refusal is reachable in
  ten requests. The brief's figure, 50,000, is the default everywhere else.

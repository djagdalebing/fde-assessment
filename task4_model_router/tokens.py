"""Token estimation for the rate limiter.

The limiter needs a number *before* the call, so this is necessarily an
estimate. It is deliberately conservative: a low estimate lets a tenant exceed
the limit, a high one costs them a little headroom, so the error is biased
toward over-counting. The reservation is reconciled against the provider's
reported usage as soon as the response arrives, so the drift is short-lived.

``tiktoken`` is used when its encoding is already available locally. It is
never fetched on demand - a rate limiter that makes a network call on the hot
path, and fails when that call does, is worse than a good approximation.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any

logger = logging.getLogger("model-router.tokens")

#: Empirical average for English text under BPE encodings. Under-counts on
#: code and on non-Latin scripts, which is why it is only the fallback.
_CHARS_PER_TOKEN = 3.6
#: Per-message framing overhead (role, separators) in chat formats.
_MESSAGE_OVERHEAD = 4

#: What to reserve for the completion when the caller does not say.
#:
#: The reservation has to be an upper bound on what the provider can bill, or
#: the limit is enforced against the estimate instead of against spend: a
#: request that simply omits ``max_tokens`` lets the model run to its own
#: limit. It is settled down to the provider's reported usage the moment the
#: response lands, so a generous ceiling costs headroom, never throughput.
DEFAULT_COMPLETION_CEILING = int(os.environ.get("ROUTER_DEFAULT_MAX_TOKENS", "4096"))

#: How deep the estimator recurses before measuring the remainder in bulk.
#: Past this it does NOT stop counting - see ``_walk_text``.
_MAX_WALK_DEPTH = 20

#: Keys that carry configuration rather than prompt text, skipped so a long
#: model name is not billed as prompt.
#:
#: Two rules, both learned the hard way. It is applied at the TOP LEVEL ONLY -
#: applied at every depth, a message-level ``metadata`` was free. And keys that
#: can hold arbitrary caller content are NOT in it: ``metadata``, ``user``,
#: ``stop`` and ``logit_bias`` are all forwarded to the provider verbatim, and
#: 500 KB parked in ``metadata`` estimated at 16 tokens while arriving upstream
#: in full. The invariant is the one stated in ``_walk_text``: anything the
#: gateway forwards, it must count.
_NON_PROMPT_KEYS = frozenset({
    "model", "stream", "stream_options", "temperature", "top_p", "top_k", "n", "best_of",
    "max_tokens", "max_completion_tokens", "max_output_tokens", "presence_penalty",
    "frequency_penalty", "seed", "logprobs", "top_logprobs",
    "scenario", "chunk_delay", "store", "service_tier", "parallel_tool_calls",
})

@functools.lru_cache(maxsize=4)
def _encoding(name: str = "cl100k_base"):
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception as exc:  # network unavailable, package missing, cache cold
        logger.info("tiktoken unavailable (%s); using heuristic token estimation", type(exc).__name__)
        return None


def count_text_tokens(text: str) -> int:
    if not text:
        return 0
    encoding = _encoding()
    if encoding is not None:
        return len(encoding.encode(text, disallowed_special=()))
    # Round up: never under-report.
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def _walk_text(value: Any, depth: int = 0) -> int:
    """Count every scrap of text anywhere in a request body.

    A field allow-list was the wrong shape. ``_call_provider`` forwards
    ``{**body}`` - the whole thing - but the estimator only walked
    ``messages``/``prompt``/``system``/``input``/``tools``, so 405 KB parked in
    a ``context_documents`` key was estimated at **21 tokens** and billed at
    90,235. Anything the gateway forwards, it must count.

    Two rules make that literally true rather than nearly true:

    * **The depth guard fails closed.** It used to ``return 0``, so wrapping a
      200 KB string in 25 nested dicts estimated it at 16 tokens - and because
      the check happens before any provider call, ten concurrent requests all
      passed at 16 and sent 1,038,488 tokens against a 50,000/minute budget, a
      20.8x breach. Past the depth limit the remainder is measured from its
      serialised form instead: cheaper than recursing, and never zero.
    * **Numbers are counted.** They were skipped as "not text", so 300,000
      floats - 2.9 MB forwarded verbatim to the provider - estimated at 16.
      A provider bills for the tokens in the body it receives, whatever their
      JSON type.
    """
    if depth > _MAX_WALK_DEPTH:
        # Fail closed. ``default=str`` so an unserialisable leaf cannot raise
        # out of the estimator and take the request down with it.
        try:
            return count_text_tokens(json.dumps(value, default=str))
        except (TypeError, ValueError, RecursionError):  # pragma: no cover
            return count_text_tokens(str(value))
    if isinstance(value, str):
        return count_text_tokens(value)
    if value is None:
        return 0
    if isinstance(value, bool):
        # "true"/"false" as sent on the wire.
        return 1
    if isinstance(value, (int, float)):
        return count_text_tokens(repr(value))
    if isinstance(value, dict):
        total = 0
        for key, item in value.items():
            # The KEY is forwarded as surely as the value. Skipping it let 3.9MB
            # of text sit in a key and estimate at 8 tokens, against 487,507 for
            # the same text one position to the right.
            if isinstance(key, str):
                total += count_text_tokens(key)
            # Depth 0 only: these names mean "configuration" at the top of a
            # request body and nothing in particular anywhere else.
            if depth == 0 and isinstance(key, str) and key in _NON_PROMPT_KEYS:
                continue
            total += _walk_text(item, depth + 1)
        return total
    if isinstance(value, (list, tuple)):
        return sum(_walk_text(item, depth + 1) for item in value)
    return count_text_tokens(str(value))


def _serialised_floor(body: dict[str, Any]) -> int:
    """Tokens for the body as it actually goes on the wire.

    A floor under the structural walk, and the reason it exists is that the
    walk has now been evaded three separate times - text parked in an
    unwalked field, text hidden by nesting past the depth guard, text in a
    dict key, and text under a skip-listed name. Each was fixed as an
    instance. This closes the class: ``_call_provider`` forwards ``{**body}``,
    so whatever it serialises is what the provider bills for, and no shape of
    input can cost less than that.

    The cost is precision on configuration fields - a long ``model`` name now
    counts, where the skip-list existed to exempt it. That is the right trade:
    over-counting a config field wastes a little of a tenant's own budget,
    while under-counting one hands out 300x the limit.
    """
    try:
        return count_text_tokens(json.dumps(body, default=str))
    except (TypeError, ValueError, RecursionError):  # pragma: no cover
        return count_text_tokens(str(body))


def estimate_request_tokens(body: dict[str, Any], default_max_tokens: int | None = None) -> int:
    """An upper bound on what this request can cost, not a guess at what it will.

    Two parts, and both have to be ceilings or the limit is not a limit:

    * **Prompt** - every string anywhere in the body, because everything in the
      body is forwarded to the provider and billed.
    * **Completion** - what the caller asked for, times ``n``, defaulting to
      ``DEFAULT_COMPLETION_CEILING`` when they did not say.
    """
    prompt = max(_walk_text(body), _serialised_floor(body))
    messages = body.get("messages")
    if isinstance(messages, list):
        prompt += _MESSAGE_OVERHEAD * len(messages)

    requested_max = None
    for field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        candidate = body.get(field)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            requested_max = max(requested_max or 0, candidate)
    if requested_max is None:
        requested_max = default_max_tokens if default_max_tokens is not None else DEFAULT_COMPLETION_CEILING
    multiplier = 1
    for field in ("n", "best_of"):
        candidate = body.get(field)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 1:
            multiplier = max(multiplier, candidate)

    return max(1, prompt + requested_max * multiplier)


#: Anything above this is a broken or hostile provider, not a real completion.
#: Beyond 2**63 the sqlite driver raises OverflowError, which turned a
#: completion the tenant had already paid for into a 500 charged at zero.
MAX_REPORTED_TOKENS = 10_000_000


def actual_tokens_from_response(payload: dict[str, Any], fallback: int) -> int:
    """Read the provider's own accounting; fall back to the reservation.

    Every value is range-checked. The figure goes straight into the spend
    ledger, so an unvalidated one is either a crash or free tokens.
    """

    def sane(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if value < 0:
            return None
        if value > MAX_REPORTED_TOKENS:
            logger.warning("provider reported an implausible %d tokens; clamping", value)
            return MAX_REPORTED_TOKENS
        return value

    usage = payload.get("usage")
    if isinstance(usage, dict):
        total = sane(usage.get("total_tokens"))
        if total is not None:
            return total
        prompt = sane(usage.get("prompt_tokens"))
        completion = sane(usage.get("completion_tokens"))
        if prompt is not None and completion is not None:
            return min(prompt + completion, MAX_REPORTED_TOKENS)
    return fallback

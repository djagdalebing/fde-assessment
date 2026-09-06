"""Task 3: streaming PII redaction, chunk-boundary correctness, and latency.

Two levels:

* **Unit** - the redactor is driven directly, including a property-style test
  that asserts *every* chunking of a text produces the same output as redacting
  it whole. That is the real invariant; a boundary bug is a bug that only some
  chunkings reveal.
* **Integration** - the gateway is mounted over ASGI against the mock provider
  and the SSE bytes are reassembled, so framing, buffering and flush behaviour
  are all exercised end to end.
"""

from __future__ import annotations

import json
import os
import random
import sys
import unicodedata
import time

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task3_streaming_guardrail import mock_llm  # noqa: E402
from task3_streaming_guardrail.gateway import create_app  # noqa: E402
from task3_streaming_guardrail.redactor import (  # noqa: E402
    PLACEHOLDER,
    StreamingRedactor,
    redact_text,
)
from task3_streaming_guardrail.sse import SSEDecoder, encode_sse  # noqa: E402
from tests.live_server import live_server  # noqa: E402


def stream_through(text: str, chunks: list[str] | None = None, **kwargs) -> tuple[str, StreamingRedactor]:
    redactor = StreamingRedactor(**kwargs)
    pieces = chunks if chunks is not None else [text]
    out = "".join(redactor.feed(piece) for piece in pieces)
    return out + redactor.flush(), redactor


def split_every(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Order 1234567890123456 shipped.",       # 16 digits, fails Luhn
        "SSN 000-45-6789 is not issued.",         # invalid SSA area
        "SSN 123-00-6789 is not issued.",         # invalid group
        "Version 1.2.3 released.",
        "The answer is 42.",
        "Reach us @support on the forum.",        # @ without an email shape
        "Read docs at example.com/guide.",        # domain, not an address
    ],
)
def test_non_pii_is_not_redacted(text):
    out, redactor = stream_through(text)
    assert out == text
    assert redactor.stats.total == 0




# --------------------------------------------------------------------------- #
# Chunk-boundary correctness - the core of the task
# --------------------------------------------------------------------------- #
SAMPLE = (
    "Hello! The account owner is ada.lovelace@example.com, "
    "their card is 4111 1111 1111 1111, SSN 123-45-6789, "
    "phone 555-123-4567, key AKIAIOSFODNN7EXAMPLE. Thanks!"
)


def test_single_character_chunks_match_whole_text():
    """The worst case: one character per chunk."""
    whole, _ = stream_through(SAMPLE)
    piecewise, redactor = stream_through(SAMPLE, chunks=list(SAMPLE))
    assert piecewise == whole
    assert "example.com" not in piecewise
    assert "4111" not in piecewise
    assert "123-45-6789" not in piecewise
    assert redactor.stats.counts["email"] == 1


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 11, 13, 17, 32, 64, 200])
def test_every_fixed_chunking_matches(size):
    whole, _ = stream_through(SAMPLE)
    piecewise, _ = stream_through(SAMPLE, chunks=split_every(SAMPLE, size))
    assert piecewise == whole


def test_random_chunkings_all_match():
    """Property-style: 200 random splits must all agree with the whole-text run."""
    whole, _ = stream_through(SAMPLE)
    rng = random.Random(20260905)
    for _ in range(200):
        chunks, index = [], 0
        while index < len(SAMPLE):
            step = rng.randint(1, 9)
            chunks.append(SAMPLE[index : index + step])
            index += step
        piecewise, _ = stream_through(SAMPLE, chunks=chunks)
        assert piecewise == whole, f"chunking changed the result: {chunks[:6]}..."


@pytest.mark.parametrize(
    "text, must_not_contain",
    [
        ("Your card on file is 04111 1111 1111 1111 thanks.", "4111 1111 1111 1111"),
        ("Card no.4111 1111 1111 1111 today.", "4111 1111 1111 1111"),
        ("Reference ref-4111 1111 1111 1111 ok.", "4111 1111 1111 1111"),
        ("Total 100.4111111111111111 charged.", "4111111111111111"),
        ("His SSN is 0123-45-6789 on record.", "123-45-6789"),
        ("See ssn.123-45-6789 in the file.", "123-45-6789"),
        ("ssn-123-45-6789 here", "123-45-6789"),
        ("Card 4111,1111,1111,1111 on file.", "4111,1111,1111,1111"),
        ("Card 4111_1111_1111_1111 on file.", "4111_1111_1111_1111"),
    ],
)
def test_a_value_glued_to_a_preceding_digit_or_separator_is_redacted(text, must_not_contain):
    """The leading lookbehind stops a match beginning inside a token, which is
    right - but it also meant a real value glued to a preceding digit, dot or
    dash was never attempted at all. The forward scan only moves forward, and
    the card in "04111 1111 1111 1111" starts one character *before* the first
    position the lookbehind permits, so it streamed out in full.

    The existing coverage missed this by always separating the stray digit with
    a space ("Order 999 4111 ..."), which the lookbehind allows.
    """
    out = redact_text(text)
    assert must_not_contain not in out, f"PII leaked: {out!r}"
    assert PLACEHOLDER in out


@pytest.mark.parametrize(
    "text",
    [
        "Ref 1234567890123456789 shipped.",
        "Order 12345 and version 1.2.3 and 2024-01-15 date",
        "call 555-123-4567 ok",
        "1234 1234 1234 1234 1234 1234",
        "Order 1234 5678 9012 3456 ref",
        "Invoice total 1,234,567.89 due",
    ],
)
def test_the_rescue_pass_does_not_invent_redactions(text):
    """Re-anchoring inside numeric runs must not turn ordinary numbers into
    PII. An unbroken digit run is still judged whole, and Luhn is bounded to a
    plausible PAN length (13-19) so a 20-digit run that happens to satisfy the
    checksum is not a card."""
    assert redact_text(text) == text


def test_a_glued_value_is_redacted_at_every_split_point():
    """Whatever the chunk boundaries, the streamed result matches whole-text."""
    text = "Your card on file is 04111 1111 1111 1111 thanks."
    whole = redact_text(text)
    assert "4111 1111 1111 1111" not in whole
    for index in range(len(text) + 1):
        redactor = StreamingRedactor()
        streamed = redactor.feed(text[:index]) + redactor.feed(text[index:]) + redactor.flush()
        assert streamed == whole, f"split at {index} changed the result: {streamed!r}"


def test_the_rescue_pass_is_not_quadratic():
    """The rescue pass tested "is this position already covered?" against a
    list, so cost grew with the number of values already found: 12,801
    candidates against 1,280 accepted spans is 8.2M comparisons, and the time
    rose 4x per doubling. That is the O(n^2) shape that once froze the event
    loop for 18s on a single delta.

    The input has to be PII-DENSE to exercise it - a run that matches nothing
    leaves the covered list empty and the linear scan looks free.
    """
    small = "4111 1111 1111 1111 " * 400
    large = "4111 1111 1111 1111 " * 1_600  # 4x the input
    start = time.perf_counter(); redact_text(small); small_ms = time.perf_counter() - start
    start = time.perf_counter(); redact_text(large); large_ms = time.perf_counter() - start
    assert large_ms < small_ms * 10, (
        f"4x the input took {large_ms / max(small_ms, 1e-9):.1f}x the time - superlinear"
    )


def test_split_exactly_at_the_at_sign():
    out, _ = stream_through("", chunks=["Contact ada", "@", "example", ".com", " now"])
    assert out == "Contact [REDACTED] now"


def test_pii_at_the_very_end_is_flushed_redacted():
    """Nothing follows to trigger the match; flush() has to catch it."""
    out, redactor = stream_through("", chunks=["Reach me at ", "grace", "@example", ".com"])
    assert out == "Reach me at [REDACTED]"
    assert redactor.stats.counts["email"] == 1


def test_adjacent_pii_values_are_each_redacted():
    text = "a@b.co c@d.co e@f.co"
    piecewise, redactor = stream_through("", chunks=list(text))
    assert piecewise == "[REDACTED] [REDACTED] [REDACTED]"
    assert redactor.stats.counts["email"] == 3


def test_run_together_addresses_match_the_whole_text_result():
    """``a@b.coc@d.co`` is genuinely ambiguous - there is no separator.

    The guarantee is not "every conceivable reading is redacted", it is that
    streaming and whole-text agree, and that no plain address survives.
    """
    text = "a@b.coc@d.co"
    whole, _ = stream_through(text)
    piecewise, _ = stream_through("", chunks=list(text))
    assert piecewise == whole
    assert "a@b.co" not in piecewise


def test_unicode_is_preserved_across_chunks():
    text = "Réservation confirmée 🎉 contact: ada@example.com — merci!"
    whole, _ = stream_through(text)
    piecewise, _ = stream_through(text, chunks=list(text))
    assert whole == piecewise
    assert "🎉" in piecewise and "Réservation" in piecewise
    assert "ada@example.com" not in piecewise


# --------------------------------------------------------------------------- #
# Memory and latency
# --------------------------------------------------------------------------- #
def test_buffer_never_exceeds_the_holdback_bound():
    redactor = StreamingRedactor(max_holdback=1024)
    long_text = SAMPLE * 500  # ~90 KB
    peak = 0
    for character in long_text:
        redactor.feed(character)
        peak = max(peak, redactor.pending)
    redactor.flush()
    assert peak <= 1024, f"buffer grew to {peak}"
    # In practice ordinary prose flushes far below the cap.
    assert peak < 100, f"buffer sat at {peak}; prose should flush almost immediately"


def test_holdback_below_the_longest_match_is_rejected():
    """A window smaller than the longest possible match is not a memory bound.

    It is a guarantee that a long enough value gets cut in half and its front
    emitted in the clear, so the constructor refuses it outright.
    """
    with pytest.raises(ValueError, match="below the longest possible match"):
        StreamingRedactor(max_holdback=128)


@pytest.mark.parametrize(
    "pii",
    [
        "a" * 64 + "@" + "b" * 60 + ".com",                            # 129 chars
        "a" * 64 + "@" + "b" * 63 + ".com",                            # 132 chars
        "a" * 64 + "@" + ".".join("b" * 60 for _ in range(8)) + ".com",  # 556 chars
        "4111 1111 1111 1111",                                          # ordinary card
    ],
)
def test_values_longer_than_a_typical_buffer_are_not_sliced(pii):
    """The regression that mattered most.

    An earlier version applied the buffer cap *after* the straddle guard, so a
    value longer than the cap had its front half emitted in the clear - a
    complete email address, at the shipped default config.
    """
    text = "Contact " + pii + " ok"
    redactor = StreamingRedactor()
    out = "".join(redactor.feed(c) for c in text) + redactor.flush()
    assert pii not in out, "a long value was sliced and leaked"
    assert out == redact_text(text)


def test_adversarial_viable_prefix_cannot_exhaust_memory():
    """A million digits is one long viable credit-card prefix.

    Without the cap this buffers forever. With it, the buffer stays bounded and
    the stream keeps flowing.
    """
    redactor = StreamingRedactor(max_holdback=1024)
    peak, emitted = 0, 0
    for _ in range(2000):
        emitted += len(redactor.feed("1234567890"))
        peak = max(peak, redactor.pending)
    assert peak <= 1024
    assert emitted > 18_000, "output stalled while the buffer was capped"


@pytest.mark.parametrize(
    "text",
    [
        "Line 1 4111-1111-1111-1111 ok",
        "Order 999 4111 1111 1111 1111 shipped.",
        "Seat 7 4111 1111 1111 1111 confirmed.",
    ],
)
def test_a_card_preceded_by_a_stray_digit_is_still_redacted(text):
    """A greedy match that fails Luhn must not hide a real card inside it.

    ``"Line 1 4111-..."`` matches as one span starting at the stray ``1``,
    fails the checksum, and - if the scan resumed after the whole span - the
    card inside was never re-examined and streamed out in the clear. Plain
    prose, no adversarial input required.
    """
    for size in (1, 2, 3, 5, 16, len(text)):
        redactor = StreamingRedactor()
        out = "".join(redactor.feed(c) for c in split_every(text, size)) + redactor.flush()
        assert "4111" not in out, f"card leaked at chunk size {size}: {out!r}"
        assert "[REDACTED]" in out


@pytest.mark.parametrize(
    "text",
    [
        "Statement line: 2024-03-01 4111 1111 1111 1111 12500 GROCERY",
        "Account 4111-1111-1111-1111-0001 is primary.",
        "Charge 4111 1111 1111 1111 4824 ref",
        "x 3782 822463 10005 1234 y",
        "charge 378282246310005-1234 ref",
        "acct 5555 5555 5555 4444 1000 usd",
        "1234 4111 1111 1111 1111 ok",
    ],
)
def test_a_card_followed_by_another_number_group_is_still_redacted(text):
    """The leak that survived two rounds, in prose an assistant would write.

    The greedy match swallowed the trailing group, failed Luhn, and the veto
    restarted at ``start + 1`` - never at a SHORTER match at the same start. 59
    of 75 Luhn-valid PANs followed by a group leaked in full. The veto now
    re-examines the rejected span for a shorter validated match first.
    """
    for size in (1, 3, 16, len(text)):
        redactor = StreamingRedactor()
        out = "".join(redactor.feed(c) for c in split_every(text, size)) + redactor.flush()
        assert "[REDACTED]" in out, f"no redaction at chunk size {size}: {out!r}"
        for digits in ("4111 1111 1111 1111", "4111-1111-1111-1111", "3782 822463 10005",
                       "378282246310005", "5555 5555 5555 4444"):
            if digits in text:
                assert digits not in out, f"PAN leaked at chunk size {size}: {out!r}"


def test_the_card_rescan_does_not_invent_matches_inside_a_bare_digit_run():
    """The rescan must apply only to grouped values: a 16-digit order reference
    is not a card because some 13-digit window inside it passes Luhn."""
    for text in ("Order 1234567890123456 shipped.", "ref 12345678901234567890 ok",
                 "tracking 1Z999AA10123456784"):
        assert "[REDACTED]" not in redact_text(text), f"{text!r} was wrongly redacted"


def test_pathological_input_does_not_stall_the_event_loop():
    """The email pattern used to backtrack quadratically.

    ``feed()`` is synchronous inside the stream handler, so one delta of
    ``"a"*8000 + "@" + "b"*8000`` blocked the whole worker for ~18 seconds and
    slowed concurrent tenants by ~176x. A left lookbehind stops the failed
    attempt being retried at every offset inside the local part.
    """
    for size in (2000, 4000, 8000):
        redactor = StreamingRedactor()
        start = time.perf_counter()
        redactor.feed("a" * size + "@" + "b" * size)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"{2 * size + 1} chars blocked for {elapsed:.2f}s"


def test_chunking_invariant_holds_under_fuzzing():
    """The headline invariant, fuzzed over the pattern space.

    The fixed-SAMPLE property test could not see this: an earlier version
    disagreed with whole-text redaction on ~1.3% of random contexts, and some
    of those disagreements leaked a card.
    """
    rng = random.Random(20260906)
    alphabet = "abcXYZ019 .+-_@\n"
    values = ["ada@example.com", "4111 1111 1111 1111", "123-45-6789", "555-123-4567"]
    for _ in range(2000):
        pad = lambda: "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        text = pad() + rng.choice(values) + pad()
        redactor = StreamingRedactor()
        streamed = "".join(redactor.feed(c) for c in text) + redactor.flush()
        assert streamed == redact_text(text), f"chunking changed the result for {text!r}"


def test_prose_flushes_immediately_so_ttft_is_not_penalised():
    """Ordinary text must not sit in the buffer waiting for a timer."""
    redactor = StreamingRedactor()
    first = redactor.feed("Your order shipped on Tuesday. ")
    assert first == "Your order shipped on Tuesday. ", "safe prose was withheld"
    assert redactor.pending == 0


def test_first_safe_token_is_released_on_the_first_chunk():
    redactor = StreamingRedactor()
    assert redactor.feed("Sure! ") == "Sure! "


def test_throughput_is_not_pathological():
    """A guardrail that halves tokens/sec is not shippable. Assert a floor.

    Best of three, because a throughput floor should measure what the code can
    do, not what the scheduler allowed on one try - this failed intermittently
    while other work was running on the machine, which is a false signal about
    the guardrail.
    """
    text = SAMPLE * 200  # ~34 KB
    chunks = split_every(text, 4)  # roughly one token per chunk

    best = 0.0
    for _ in range(3):
        redactor = StreamingRedactor()
        start = time.perf_counter()
        for chunk in chunks:
            redactor.feed(chunk)
        redactor.flush()
        best = max(best, len(text) / (time.perf_counter() - start))

    # ~325k chars/s on this PII-dense corpus, fed four characters at a time
    # (roughly one token per chunk), on this machine.
    #
    # The floor is set well below that on purpose: it is here to catch a
    # collapse, and it has caught two. Adding the keyword-SSN branch to the
    # main alternation took this to 5k chars/s with the hold-back buffer
    # pinned at its cap - the branch is now compiled separately for exactly
    # that reason. Widening the character classes to \p{L}/\p{N} took it to
    # 3.4k, and was reverted.
    #
    # The floor is set far below the measurement on purpose - it exists to
    # catch a collapse (an accidentally quadratic scan), not to police normal
    # variation between machines.
    assert best > 100_000, f"only {best:,.0f} chars/s"


# --------------------------------------------------------------------------- #
# SSE decoding
# --------------------------------------------------------------------------- #
def test_sse_decoder_reassembles_split_events():
    payload = b'data: {"a": 1}\n\ndata: {"b": 2}\n\ndata: [DONE]\n\n'
    for size in (1, 3, 7, 13, len(payload)):
        decoder = SSEDecoder()
        events = []
        for i in range(0, len(payload), size):
            events.extend(decoder.feed(payload[i : i + size]))
        events.extend(decoder.flush())
        assert [e.data for e in events] == ['{"a": 1}', '{"b": 2}', "[DONE]"], f"size={size}"


def test_sse_decoder_handles_split_multibyte_characters():
    payload = 'data: {"t": "café 🎉"}\n\n'.encode()
    decoder = SSEDecoder()
    events = []
    for byte in payload:
        events.extend(decoder.feed(bytes([byte])))
    events.extend(decoder.flush())
    assert json.loads(events[0].data)["t"] == "café 🎉"


def test_sse_decoder_skips_comments_and_handles_crlf():
    decoder = SSEDecoder()
    events = list(decoder.feed(b": keep-alive\r\n\r\ndata: {\"x\": 1}\r\n\r\n"))
    assert [e.data for e in events] == ['{"x": 1}']


# --------------------------------------------------------------------------- #
# Gateway integration
# --------------------------------------------------------------------------- #
@pytest.fixture
async def gateway():
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_llm.app), base_url="http://provider"
    )
    app = create_app(upstream_url="http://provider/v1/chat/completions", client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
        client.gateway_app = app
        yield client
    await upstream.aclose()


async def collect(client, scenario="default", **extra):
    """Drive a streaming request and reassemble the assistant text."""
    body = {"model": "mock-model-v1", "stream": True, "scenario": scenario, **extra}
    text, chunks, done = "", [], False
    async with client.stream("POST", "/v1/chat/completions", json=body) as response:
        assert response.status_code == 200
        decoder = SSEDecoder()
        async for raw in response.aiter_bytes():
            for event in decoder.feed(raw):
                if event.is_done:
                    done = True
                    continue
                payload = json.loads(event.data)
                chunks.append(payload)
                delta = payload.get("choices", [{}])[0].get("delta", {})
                text += delta.get("content") or ""
        for event in decoder.flush():
            if event.is_done:
                done = True
    return text, chunks, done


async def test_streamed_response_is_redacted_end_to_end(gateway):
    text, chunks, done = await collect(gateway)
    assert done, "stream did not terminate with [DONE]"
    assert "ada.lovelace@example.com" not in text
    assert "4111" not in text
    assert "123-45-6789" not in text
    assert "[REDACTED]" in text
    # A phone number is not one of the three classes the brief names, and the
    # guardrail does not invent redactions it was not asked for.
    assert "555-123-4567" in text
    # The non-sensitive prose survives intact.
    assert text.startswith("Sure. I found the account.")
    assert text.endswith("Nothing else to report.")


async def test_clean_stream_passes_through_byte_identical(gateway):
    text, _, done = await collect(gateway, scenario="clean")
    assert done
    assert text == "".join(mock_llm.SCENARIOS["clean"])
    assert "[REDACTED]" not in text


async def test_pii_at_end_of_stream_is_redacted_by_flush(gateway):
    text, _, done = await collect(gateway, scenario="pii_at_end")
    assert done
    assert text == "All set. Reach me at [REDACTED]"


async def test_single_character_deltas_are_redacted(gateway):
    text, _, _ = await collect(gateway, scenario="single_char")
    assert "ada@example.com" not in text
    assert text == "Email: [REDACTED] done."


async def test_unicode_scenario_survives_the_gateway(gateway):
    text, _, _ = await collect(gateway, scenario="unicode")
    assert "🎉" in text and "Réservation" in text
    assert "ada@example.com" not in text


async def test_false_positive_scenario(gateway):
    text, _, _ = await collect(gateway, scenario="false_positives")
    assert "1234567890123456" in text, "a non-Luhn 16-digit id was wrongly redacted"
    assert "123-45-6789" not in text


async def test_finish_reason_reaches_the_client(gateway):
    _, chunks, _ = await collect(gateway)
    reasons = [c["choices"][0].get("finish_reason") for c in chunks]
    assert "stop" in reasons


async def test_stats_are_recorded_for_the_stream(gateway):
    await collect(gateway)
    stats = gateway.gateway_app.state.last_stats
    assert stats.counts.get("email") == 1
    assert stats.counts.get("credit_card") == 1
    assert stats.counts.get("ssn") == 1


async def test_gateway_never_accumulates_the_full_response(gateway):
    """Assert the memory claim directly, on the object that would hold it."""
    body = {"model": "m", "stream": True, "scenario": "default"}
    async with gateway.stream("POST", "/v1/chat/completions", json=body) as response:
        async for _ in response.aiter_bytes():
            stats = gateway.gateway_app.state.last_stats
            if stats is not None:
                assert stats.characters_in - stats.characters_out <= 128


async def test_time_to_first_token_is_not_gated_on_stream_completion():
    """TTFT must reflect the first safe token, not the last upstream chunk.

    Run over real sockets: ``httpx.ASGITransport`` drains an app's whole
    response before returning, so an in-process mount cannot distinguish a
    streaming gateway from a buffering one. The provider sleeps 25 ms between
    ~15 chunks, so a gateway that buffered could not answer before ~375 ms.
    """
    with live_server(mock_llm.app) as provider_url:
        gw_app = create_app(upstream_url=f"{provider_url}/v1/chat/completions")
        with live_server(gw_app) as gateway_url:
            body = {"model": "m", "stream": True, "scenario": "default", "chunk_delay": 0.025}
            async with httpx.AsyncClient(timeout=30.0) as client:
                start = time.perf_counter()
                first_content_at = None
                async with client.stream("POST", f"{gateway_url}/v1/chat/completions", json=body) as response:
                    assert response.status_code == 200
                    decoder = SSEDecoder()
                    async for raw in response.aiter_bytes():
                        for event in decoder.feed(raw):
                            if event.is_done:
                                continue
                            payload = json.loads(event.data)
                            if payload.get("choices", [{}])[0].get("delta", {}).get("content"):
                                if first_content_at is None:
                                    first_content_at = time.perf_counter() - start
                total = time.perf_counter() - start

    assert first_content_at is not None, "no content delta ever arrived"
    assert total > 0.2, "the provider did not actually stream slowly; the test would be vacuous"
    assert first_content_at < total / 2, (
        f"first token at {first_content_at:.3f}s of a {total:.3f}s stream - the gateway is buffering"
    )


async def test_output_arrives_incrementally_over_a_real_socket():
    """Deltas must be spread across the stream, not delivered in one burst."""
    with live_server(mock_llm.app) as provider_url:
        gw_app = create_app(upstream_url=f"{provider_url}/v1/chat/completions")
        with live_server(gw_app) as gateway_url:
            body = {"model": "m", "stream": True, "scenario": "clean", "chunk_delay": 0.05}
            arrivals = []
            async with httpx.AsyncClient(timeout=30.0) as client:
                start = time.perf_counter()
                async with client.stream("POST", f"{gateway_url}/v1/chat/completions", json=body) as response:
                    decoder = SSEDecoder()
                    async for raw in response.aiter_bytes():
                        for event in decoder.feed(raw):
                            if not event.is_done:
                                payload = json.loads(event.data)
                                if payload["choices"][0]["delta"].get("content"):
                                    arrivals.append(time.perf_counter() - start)

    assert len(arrivals) >= 3, f"expected several deltas, saw {len(arrivals)}"
    spread = arrivals[-1] - arrivals[0]
    assert spread > 0.1, f"all deltas arrived within {spread:.3f}s - the gateway batched them"


# --------------------------------------------------------------------------- #
# Every model-authored field, not just delta.content
# --------------------------------------------------------------------------- #
async def drive_chunks(chunks, extra_events=()):
    """Feed a scripted chunk list through the gateway; return the raw SSE body."""
    async def upstream(request):
        async def body():
            for chunk in chunks:
                yield b"data: " + json.dumps(chunk).encode() + b"\n\n"
            for raw in extra_events:
                yield raw
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(upstream_url="http://p/v1/chat/completions", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        async with gw.stream("POST", "/v1/chat/completions", json={"stream": True}) as response:
            body = b"".join([c async for c in response.aiter_bytes()]).decode()
    await client.aclose()
    return body, app


def delta(index, payload, finish=None):
    return {"choices": [{"index": index, "delta": payload, "finish_reason": finish}]}


async def test_tool_call_arguments_are_redacted_across_chunks():
    """A tool call's ``arguments`` is a JSON string the model wrote, and
    routinely carries exactly the fields being redacted. It used to pass
    through completely untouched."""
    body, app = await drive_chunks([
        delta(0, {"role": "assistant"}),
        delta(0, {"tool_calls": [{"index": 0, "function": {"arguments": '{"email":"ad'}}]}),
        delta(0, {"tool_calls": [{"index": 0, "function": {"arguments": 'a@example.com","ssn":"123-'}}]}),
        delta(0, {"tool_calls": [{"index": 0, "function": {"arguments": '45-6789"}'}}]}),
        delta(0, {}, finish="stop"),
    ])
    assert "ada@example.com" not in body
    assert "123-45-6789" not in body
    assert "[REDACTED]" in body
    assert app.state.last_stats.counts.get("email") == 1
    assert app.state.last_stats.counts.get("ssn") == 1


async def test_reasoning_content_is_redacted():
    body, _ = await drive_chunks([delta(0, {"reasoning_content": "card is 4111 1111 1111 1111 ok"})])
    assert "4111" not in body
    assert "[REDACTED]" in body


async def test_every_choice_is_redacted_not_just_the_first():
    """``n > 1`` produces several choices; only ``choices[0]`` was rewritten."""
    body, _ = await drive_chunks([
        delta(0, {"content": "mail ada@example.com"}),
        delta(1, {"content": "mail grace@example.com"}),
        delta(2, {"content": "mail alan@example.com"}),
    ])
    for address in ("ada@example.com", "grace@example.com", "alan@example.com"):
        assert address not in body, f"{address} leaked from a non-zero choice"


async def test_parallel_streams_are_not_spliced_together():
    """Interleaved deltas for different choices are separate strings.

    Sharing one redactor across them would join ``"ada"`` from choice 0 to
    ``"@example.com"`` from choice 1 and invent a match across a boundary that
    does not exist.
    """
    body, _ = await drive_chunks([
        delta(0, {"content": "ada"}),
        delta(1, {"content": "@example.com "}),
        delta(0, {"content": " is a name"}),
        delta(1, {"content": "is a domain"}),
    ])
    assert "[REDACTED]" not in body, "an address was invented across two independent streams"
    # Both streams still arrive intact.
    assert "ada" in body and "@example.com" in body




async def test_bare_json_string_events_are_redacted():
    body, _ = await drive_chunks(
        [], extra_events=[b'data: "my ssn is 123-45-6789"\n\n']
    )
    assert "123-45-6789" not in body


async def test_non_json_events_are_redacted_not_forwarded_blind():
    body, _ = await drive_chunks([], extra_events=[b"data: contact ada@example.com\n\n"])
    assert "ada@example.com" not in body


async def test_usage_and_role_chunks_survive_untouched():
    """A chunk with no model text must not be dropped or altered."""
    body, _ = await drive_chunks([
        delta(0, {"role": "assistant"}),
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7}},
        delta(0, {"content": "hello"}, finish="stop"),
    ])
    assert '"role": "assistant"' in body or '"role":"assistant"' in body
    assert "prompt_tokens" in body
    assert "hello" in body


# --------------------------------------------------------------------------- #
# SSE decoding edge cases
# --------------------------------------------------------------------------- #
def test_leading_bom_does_not_swallow_the_first_event():
    """Left in, the BOM became part of the field name ("\ufeffdata") and the
    entire first event was silently discarded."""
    decoder = SSEDecoder()
    events = list(decoder.feed(b'\xef\xbb\xbfdata: {"x": 1}\n\n'))
    assert [e.data for e in events] == ['{"x": 1}']


def test_crlf_split_across_a_chunk_boundary_keeps_one_event():
    """Converting a dangling "\r" before its "\n" arrived split one multi-line
    event into two."""
    decoder = SSEDecoder()
    events = list(decoder.feed(b"data: x\r")) + list(decoder.feed(b"\ndata: y\r\n\r\n"))
    assert [e.data for e in events] == ["x\ny"]




def test_sse_id_and_retry_fields_survive_the_gateway():
    """``id:`` and ``retry:`` are how a client resumes a dropped stream and
    controls its reconnect delay. They were parsed and then dropped."""
    decoder = SSEDecoder()
    event = next(iter(decoder.feed(b"id: 42\nretry: 3000\nevent: msg\ndata: hi\n\n")))
    assert (event.id, event.retry, event.event, event.data) == ("42", "3000", "msg", "hi")

    reencoded = encode_sse(event.data, event.event, event.id, event.retry)
    round_tripped = next(iter(SSEDecoder().feed(reencoded)))
    assert (round_tripped.id, round_tripped.retry, round_tripped.event) == ("42", "3000", "msg")


def test_normalisation_cost_is_linear_not_quadratic():
    """The whole buffer was re-normalised on every chunk: 1MB took 0.038s and
    4MB took 0.779s - 4x the data for 20x the time."""
    def cost(megabytes):
        decoder = SSEDecoder(max_event_bytes=64 * 1024 * 1024)
        blob = b"x" * (megabytes * 1024 * 1024)
        start = time.perf_counter()
        for offset in range(0, len(blob), 65536):
            list(decoder.feed(blob[offset : offset + 65536]))
        return time.perf_counter() - start

    one, four = cost(1), cost(4)
    # Linear would be ~4x. Allow generous slack for timer noise, but 20x - the
    # quadratic signature - has to fail.
    assert four < one * 10 + 0.2, f"1MB={one:.3f}s 4MB={four:.3f}s looks quadratic"


def test_decoder_buffer_is_bounded():
    """A stream that never sends a blank line used to buffer without bound."""
    decoder = SSEDecoder(max_event_bytes=4096)
    with pytest.raises(ValueError, match="without a terminator"):
        for _ in range(100):
            list(decoder.feed(b"x" * 256))


# --------------------------------------------------------------------------- #
# Unicode evasions
# --------------------------------------------------------------------------- #


def test_control_characters_are_not_stripped_before_matching():
    """Newlines are invisible too, but dropping them merges lines: it turned
    ``"...6789\n0..."`` into ``"...67890..."``, whose trailing digit defeats the
    SSN lookahead, so a real SSN stopped being redacted."""
    assert redact_text("ssn 123-45-6789\n0 next") == "ssn [REDACTED]\n0 next"
    assert redact_text("line1\nada@example.com") == "line1\n[REDACTED]"


def test_left_context_is_carried_across_emissions():
    """Every pattern is anchored by a lookbehind. Emitting text throws that
    context away, so a buffer *starting* at ``_@Z9X.ac`` matched as an address
    while the same characters in context cannot."""
    text = "c_ada@example.com_@Z9X.ac done"
    redactor = StreamingRedactor()
    streamed = "".join(redactor.feed(c) for c in text) + redactor.flush()
    assert streamed == redact_text(text)
    assert streamed.count("[REDACTED]") == 1, "the stream invented a second match"


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


async def test_interrupted_stream_does_not_flush_the_holdback():
    """A dead connection must not release a half-seen credit card."""

    async def truncated(request):
        async def body():
            payload = {"choices": [{"index": 0, "delta": {"content": "Card 4111 1111 11"}, "finish_reason": None}]}
            yield b"data: " + json.dumps(payload).encode() + b"\n\n"
            raise httpx.ReadError("connection reset")

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(truncated))
    app = create_app(upstream_url="http://provider/v1/chat/completions", client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
        async with client.stream("POST", "/v1/chat/completions", json={"stream": True}) as response:
            body = b"".join([c async for c in response.aiter_bytes()]).decode()
    await upstream.aclose()
    assert "4111" not in body, "held-back digits were released when the stream broke"
    assert "upstream_error" in body and "[DONE]" in body




async def test_unrecognised_chunk_shapes_pass_through(gateway):
    """A provider extension chunk must not be dropped or crash the stream."""
    async def odd_stream(request):
        async def body():
            yield b"data: " + json.dumps({"custom": "provider-extension"}).encode() + b"\n\n"
            yield b": keep-alive\n\n"
            yield b"data: " + json.dumps(
                {"choices": [{"index": 0, "delta": {"content": "hi ada@example.com"}, "finish_reason": None}]}
            ).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(odd_stream))
    app = create_app(upstream_url="http://provider/v1/chat/completions", client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
        async with client.stream("POST", "/v1/chat/completions", json={"stream": True}) as response:
            body = b"".join([c async for c in response.aiter_bytes()]).decode()
    await upstream.aclose()
    assert "provider-extension" in body
    assert "ada@example.com" not in body
    assert "[REDACTED]" in body


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #
def test_redact_text_helper_matches_streaming():
    """Note: this is a self-consistency check, not an oracle.

    ``redact_text`` is ``feed() + flush()``, so it cannot detect a fault in the
    matching semantics themselves - only a difference between chunkings. The
    tests that catch matching faults are the explicit corpora above.
    """
    assert redact_text(SAMPLE) == stream_through(SAMPLE, chunks=list(SAMPLE))[0]




def test_non_ascii_digits_are_not_skipped_by_the_fast_reject():
    r"""The hold-back scan skips indices whose character cannot begin a pattern,
    but only decides that for ASCII - ``\d`` is Unicode-aware, so Arabic-Indic
    and Bengali digits must still be examined."""
    for text in ("ssn \u0660\u0661\u0662-45-6789 ok", "ssn \u09e7\u09e8\u09e9-45-6789 ok"):
        # Streamed one character at a time: the fast reject lives in the
        # hold-back scan, which a single whole-text call never exercises.
        redactor = StreamingRedactor()
        out = "".join(redactor.feed(c) for c in text) + redactor.flush()
        assert "[REDACTED]" in out, f"{text!r} was not examined"


def test_the_shipped_default_holdback_is_bounded():
    """Every memory test passes ``max_holdback`` explicitly, so the default the
    gateway actually runs with was unpinned."""
    import task3_streaming_guardrail.gateway as gw

    assert gw.MAX_HOLDBACK <= 4096
    redactor = StreamingRedactor()          # the shipped default
    assert redactor._max_holdback <= 4096, (
        f"the redactor's own default hold-back is {redactor._max_holdback}"
    )
    peak = 0
    for _ in range(2000):
        redactor.feed("1234567890")
        peak = max(peak, redactor.pending)
    assert peak <= gw.MAX_HOLDBACK


def test_max_match_length_actually_bounds_every_pattern():
    """A hand-written constant with nothing linking it to the patterns: widening
    any pattern silently re-arms the "cap slices a value in half" leak with a
    green suite."""
    from task3_streaming_guardrail.redactor import DEFAULT_PATTERNS, MAX_MATCH_LENGTH
    import regex as regex_module

    longest = 0
    probes = [
        # Long enough to catch a widened local-part bound, not just the current
        # one: a probe shorter than the bound cannot detect it being raised.
        "a" * 5000 + "@" + ".".join("b" * 63 for _ in range(9)) + ".com",
        "a" * 200 + "@" + ".".join("b" * 63 for _ in range(9)) + ".com",
        "4" * 19,
        "4111 " * 6,
        "1" * 3 + "-" + "1" * 2 + "-" + "1" * 4,
        "+1 555-123-4567",
        "AKIA" + "A" * 16,
        "sk-" + "a" * 64,
    ]
    combined = regex_module.compile(
        "|".join(f"(?P<{p.name}>{p.expression})" for p in DEFAULT_PATTERNS)
    )
    for probe in probes:
        for match in combined.finditer(probe):
            longest = max(longest, len(match.group()))
    assert longest <= MAX_MATCH_LENGTH, (
        f"a pattern can match {longest} characters but MAX_MATCH_LENGTH is {MAX_MATCH_LENGTH}"
    )


# --------------------------------------------------------------------------- #
# Buffer state: bounds, resets and neighbouring characters
# --------------------------------------------------------------------------- #


def test_a_neighbouring_character_never_breaks_a_redaction():
    """A character sitting against a value must not stop it being redacted.

    Swept over every character that expands under NFKC, because those are the
    ones that historically perturbed the match: they are multi-character in
    normalised form, so they are the widest available probe for "does an
    adjacent character change the outcome". Nothing here normalises - the
    property is simply that the neighbour is irrelevant.
    """
    expanders = [
        chr(c) for c in range(0x80, 0x3000)
        if len(unicodedata.normalize("NFKC", chr(c))) >= 2
        and unicodedata.normalize("NFKC", chr(c)).isprintable()
    ]
    assert len(expanders) > 100, "the sweep found nothing to test"
    values = ["ada@example.com", "4111 1111 1111 1111", "123-45-6789"]
    for character in expanders:
        for value in values:
            for text in (f"x {character}{value} y", f"x {value}{character} y"):
                assert value not in redact_text(text), f"{text!r} leaked"


def test_zero_width_characters_cannot_grow_the_buffer_past_its_cap():
    """The hold-back cap is a memory bound, so it has to hold against input
    chosen to defeat it. A stream of zero-width characters is the adversarial
    case: invisible, unbounded in length, and each one a character the buffer
    must account for. 20,000 of them once held 20,001 characters against a
    1,024 cap and took 19.7 seconds doing it.
    """
    redactor = StreamingRedactor(max_holdback=1024)
    redactor.feed("4")
    started = time.perf_counter()
    peak = 0
    for _ in range(20000):
        redactor.feed("​")
        peak = max(peak, redactor.pending)
    elapsed = time.perf_counter() - started
    assert peak <= 1024, f"source buffer grew to {peak} against a 1024 cap"
    assert elapsed < 5.0, f"20,000 zero-width characters took {elapsed:.1f}s"


def test_context_is_fully_reset_when_flush_finds_an_empty_buffer():
    """Reusing a redactor after ``flush`` must behave exactly like a fresh one.

    ``flush`` once cleared the carried context but left a stale offset beside
    it, so the next feed began scanning eight characters in - a blind spot at
    the head of the reused stream that let an SSN through.
    """
    redactor = StreamingRedactor()
    redactor.feed("Hello there. ")
    redactor.flush()
    out = redactor.feed("ssn 123-45-6789 and mail ada@example.com card 4111 1111 1111 1111 ")
    assert "123-45-6789" not in out, "the first characters after an empty flush were skipped"
    # Assert the reuse is equivalent to a fresh redactor, not merely that one
    # value survived: the weaker assertion passed against a mutant that left
    # the offset stale.
    fresh = StreamingRedactor()
    expected = fresh.feed("ssn 123-45-6789 and mail ada@example.com card 4111 1111 1111 1111 ")
    assert out == expected, "a reused redactor does not behave like a fresh one"




def test_a_local_part_character_before_an_address_does_not_hide_it():
    """The lookbehind forces a match to start at the first local-part
    character, so ``say .`` + 64 letters + ``@example.com`` needed 65 and
    matched nothing."""
    for prefix in (".", "-", "_", "q", "9"):
        text = "say " + prefix + "q" * 64 + "@example.com end"
        assert "@example.com" not in redact_text(text), f"prefix {prefix!r} hid the address"




@pytest.mark.parametrize(
    "text",
    ["Order 1234567890123456 shipped.", "ip 192.168.100.1 ok", "build 10.15.7 ok",
     "ts 2024-01-15 ok", "ref 100-200-3000 ok", "Version 1.2.3 released."],
)
def test_widening_the_patterns_did_not_add_false_positives(text):
    assert "[REDACTED]" not in redact_text(text), f"{text!r} was wrongly redacted"












async def test_event_ids_survive_the_gateway():
    """The fix landed in the decoder and encoder but not in the hot path, so a
    client still could not resume a dropped stream."""
    body, _ = await drive_chunks([])   # warm the helper's import path

    async def upstream(request):
        async def gen():
            yield (b"id: 42\nretry: 3000\nevent: msg\ndata: "
                   + json.dumps(delta(0, {"content": "hello world "})).encode() + b"\n\n")
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=gen())

    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(upstream_url="http://p/v1", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        async with gw.stream("POST", "/v1/chat/completions", json={"stream": True}) as response:
            out = b"".join([c async for c in response.aiter_bytes()]).decode()
    await client.aclose()
    assert "id: 42" in out and "retry: 3000" in out and "event: msg" in out


async def test_an_unterminated_upstream_event_is_an_in_band_error():
    """The decoder's size cap raised ValueError straight through the handler,
    aborting the response with no [DONE] and a traceback through Starlette."""
    async def endless(request):
        async def gen():
            yield b"data: " + b"x" * (2 * 1024 * 1024)

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=gen())

    client = httpx.AsyncClient(transport=httpx.MockTransport(endless))
    app = create_app(upstream_url="http://p/v1", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        async with gw.stream("POST", "/v1/chat/completions", json={"stream": True}) as response:
            out = b"".join([c async for c in response.aiter_bytes()]).decode()
    await client.aclose()
    assert "upstream_error" in out and "[DONE]" in out
    assert "Traceback" not in out


# --------------------------------------------------------------------------- #
# Proxy behaviour: status, ordering, cancellation, event shape
# --------------------------------------------------------------------------- #
def _upstream_returning(status: int):
    """An upstream that fails before any body, leaking internals in the body."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse as _JSON

    application = FastAPI()

    @application.post("/v1/chat/completions")
    async def fail():
        return _JSON(
            status_code=status,
            content={
                "error": {
                    "message": "upstream detail",
                    "dsn": "postgres://u:pw@internal-db/prod",
                    "stack": "Traceback (most recent call last): File /srv/app/x.py",
                }
            },
        )

    @application.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    return application


@pytest.mark.parametrize("status", [401, 429, 500, 503])
async def test_the_upstream_status_is_mirrored_not_flattened_to_200(status):
    """Every upstream failure used to arrive as HTTP 200 with an in-band error,
    so a client could not tell "retry with backoff" (429) from "your key is
    dead" (401). The upstream request is opened before a status is committed."""
    with live_server(_upstream_returning(status)) as upstream:
        app = create_app(upstream_url=f"{upstream}/v1/chat/completions")
        with live_server(app) as gateway_url:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{gateway_url}/v1/chat/completions", json={"model": "m", "stream": True}
                )
    assert response.status_code == status
    for secret in ("internal-db", "Traceback", "/srv/app", "upstream detail"):
        assert secret not in response.text, f"upstream internals leaked: {secret}"


async def test_held_back_text_is_delivered_before_the_finish_chunk(gateway):
    """A client that treats ``finish_reason`` as end-of-message must already
    have the redacted tail. Residue used to be emitted after it, so the client
    rendered the text without its redaction marker."""
    _, chunks, _ = await collect(gateway, scenario="pii_at_end")
    order = []
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason") is not None:
                order.append("finish")
            elif choice.get("delta", {}).get("content"):
                order.append(choice["delta"]["content"])
    assert "finish" in order, "the stream never reported a finish_reason"
    redaction = next(i for i, part in enumerate(order) if PLACEHOLDER in part)
    assert redaction < order.index("finish"), f"residue arrived after finish: {order}"




def test_residue_from_a_non_chunk_stream_keeps_its_own_shape():
    """``__raw__`` and ``__bare__`` streams were never OpenAI chunks. Reshaping
    their residue into one emitted ``"index": "__raw__"`` - a string where the
    schema requires an integer - and corrupted non-OpenAI SSE passthrough."""
    from task3_streaming_guardrail.gateway import _residue_events

    raw = _residue_events([(("__raw__",), "tail text")], None)
    assert raw == [b"data: tail text\n\n"], raw

    bare = _residue_events([(("__bare__",), "tail text")], None)
    assert bare == [b'data: "tail text"\n\n'], bare

    chunk = _residue_events([((0, "choices", 0, "delta", "content"), "tail")], None)
    payload = json.loads(chunk[0].decode().removeprefix("data: ").strip())
    assert payload["choices"][0]["index"] == 0
    assert isinstance(payload["choices"][0]["index"], int)
    assert payload["choices"][0]["delta"]["content"] == "tail"

    # A legacy completions stream comes back in ITS shape, not forced into a
    # chat delta.
    legacy = _residue_events([((0, "choices", 0, "text"), "tail")], None)
    payload = json.loads(legacy[0].decode().removeprefix("data: ").strip())
    assert payload["choices"][0]["text"] == "tail"


@pytest.mark.parametrize(
    "text, must_not_contain",
    [
        ("Contact ada@example.com-please use it.", "ada@example.com"),
        ("SSN: 123-45-6789-A filed.", "123-45-6789"),
        ("Card 4111 1111 1111 1111-01 expired.", "4111 1111 1111 1111"),
        ("Card 4111111111111111- ok", "4111111111111111"),
        ("Card 4111\n1111\n1111\n1111 ok.", "4111\n1111\n1111\n1111"),
        ("Card 4111\t1111\t1111\t1111 ok.", "4111\t1111\t1111\t1111"),
    ],
)
def test_a_trailing_separator_does_not_defeat_redaction(text, must_not_contain):
    """All three trailing lookaheads excluded ``-``, so a following hyphen did
    not narrow the match - it destroyed it, and the value streamed out whole.

    The bare card branch had already been corrected for exactly this; the
    grouped one and both others had not, which made it an inconsistency rather
    than a position. "More input may still be coming" is the hold-back's job,
    decided by partial matching, not something a lookahead can express.
    """
    out = redact_text(text)
    assert must_not_contain not in out, f"PII leaked: {out!r}"
    assert PLACEHOLDER in out


def test_both_addresses_in_a_hyphen_joined_pair_are_redacted():
    """The scan resumed on a character the leading lookbehind then vetoed, so
    the first address was redacted and the second left in the clear."""
    out = redact_text("ada@example.com-bob@test.org")
    assert "ada@example.com" not in out
    assert "bob@test.org" not in out, f"the second address leaked: {out!r}"
    assert out.count(PLACEHOLDER) == 2


def test_a_hyphenated_local_part_is_still_one_address():
    """Widening the lookbehind must not split a legitimate address in two."""
    assert redact_text("mail foo-bar@example.com now") == f"mail {PLACEHOLDER} now"


@pytest.mark.parametrize(
    "text",
    ["see item 1\n2\n3\n4 below", "steps:\n1234\n5678\nend", "Order 1234 5678 9012 3456 ref"],
)
def test_widening_the_separator_class_does_not_invent_redactions(text):
    assert redact_text(text) == text


@pytest.mark.parametrize(
    "text, value",
    [
        ("His SSN is 123456789.", "123456789"),
        ("SSN: 123456789", "123456789"),
        ("ssn 123456789 filed", "123456789"),
        ("SSN#123456789 ok", "123456789"),
        ("Social Security Number: 123456789", "123456789"),
        ("social security no. 123456789", "123456789"),
        ("her SSN 078051120 on file", "078051120"),
    ],
)
def test_a_bare_nine_digit_ssn_is_redacted_when_the_text_says_it_is_one(text, value):
    """A bare SSN is a real format the brief names. Matching nine digits
    unconditionally would redact every order number, so the keyword is what
    makes it safe to match at all."""
    out = redact_text(text)
    assert value not in out, f"SSN leaked: {out!r}"
    assert PLACEHOLDER in out


@pytest.mark.parametrize(
    "text",
    [
        "order 123456789 shipped",
        "part 123456789",
        "invoice 987654321 due",
        "tracking 123456789012 sent",
        "zip 12345 6789",
        "ref 000000000 void",
        "SSN redacted; unrelated order 123456789 shipped",
    ],
)
def test_nine_digits_without_the_keyword_are_left_alone(text):
    assert redact_text(text) == text


def test_a_keyword_ssn_split_across_chunks_is_still_redacted():
    """The separate engine has to be consulted by the hold-back too, or a value
    split at the wrong byte streams out in the clear."""
    text = "His SSN is 123456789 on file."
    whole = redact_text(text)
    assert "123456789" not in whole
    for index in range(len(text) + 1):
        redactor = StreamingRedactor()
        streamed = redactor.feed(text[:index]) + redactor.feed(text[index:]) + redactor.flush()
        assert streamed == whole, f"split at {index} changed the result: {streamed!r}"


def test_an_email_with_an_ip_literal_domain_is_redacted():
    assert redact_text("mail ada@192.168.1.1 now") == f"mail {PLACEHOLDER} now"
    # A bare IP that is not an address must not be touched.
    assert redact_text("see host 10.0.0.1 for logs") == "see host 10.0.0.1 for logs"


def test_the_holdback_consults_the_keyword_ssn_engine():
    """The keyword SSN is compiled separately, so the hold-back has to ask it
    as well - otherwise a value split mid-digits streams out in the clear.

    Exercised with the email AND card patterns removed. Both hold a run of
    digits for their own reasons - the card's bare 13-to-19-digit branch, and
    the email's local part, which any alphanumeric run is a prefix of - so
    with either present this check can be deleted and the test still passes.
    Two earlier versions proved nothing for exactly that reason.
    """
    from task3_streaming_guardrail.redactor import DEFAULT_PATTERNS

    patterns = tuple(p for p in DEFAULT_PATTERNS if p.name not in ("credit_card", "email"))
    text = "His SSN is 123456789 on file."
    reference = StreamingRedactor(patterns=patterns)
    whole = reference.feed(text) + reference.flush()
    assert "123456789" not in whole, "the fixture does not redact; the test proves nothing"

    for index in range(len(text) + 1):
        redactor = StreamingRedactor(patterns=patterns)
        streamed = redactor.feed(text[:index]) + redactor.feed(text[index:]) + redactor.flush()
        assert streamed == whole, f"split at {index} leaked: {streamed!r}"




async def test_the_gateway_always_streams_and_never_accumulates():
    """The task requires that the full response is never accumulated before
    forwarding, so the gateway streams whatever the caller asked for.

    A client that sends ``stream: false`` gets SSE. That is a deliberate
    reading of the task over OpenAI compatibility: assembling a single JSON
    body means holding the whole response, which is the one thing the task
    rules out.
    """
    with live_server(mock_llm.app) as upstream:
        app = create_app(upstream_url=f"{upstream}/v1/chat/completions")
        with live_server(app) as gateway_url:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for body in ({"model": "m", "stream": False}, {"model": "m"},
                             {"model": "m", "stream": True}):
                    response = await client.post(f"{gateway_url}/v1/chat/completions", json=body)
                    assert response.status_code == 200
                    assert response.headers["content-type"].startswith("text/event-stream"), body
                    assert "ada.lovelace@example.com" not in response.text
                    assert PLACEHOLDER in response.text


@pytest.mark.parametrize(
    "text, must_not_contain",
    [
        ("Your card on file is 4111 1111 1111 11115 - thanks", "4111 1111 1111 1111"),
        ("Card 41111111111111115.", "4111111111111111"),
        ("Amex 3782822463100055.", "378282246310005"),
        ("Write to ada@example.com2024 please", "ada@example.com"),
        ("04111 1111 1111 1111\t3782 822463 10005 x", "3782 822463 10005"),
        ("94111 1111 1111 1111 4111 1111 1111 1111ada@example.com", "ada@example.com"),
        ("SSN 123–45–6789.", "123–45–6789"),
        ("SSN 123‑45‑6789", "123‑45‑6789"),
    ],
)
def test_a_value_glued_to_its_neighbour_is_still_redacted(text, must_not_contain):
    """Three separate causes, one symptom: PII adjacent to other content.

    * A trailing digit made the greedy card match a 17-digit run that fails
      Luhn, and the real PAN ends mid-group - the endpoint the boundary-only
      rescan skipped.
    * An email's trailing lookahead rejected a digit, so ``ada@example.com2024``
      matched nothing at all.
    * A spurious Luhn-valid run covered a real Amex that started inside it and
      was then itself discarded by overlap resolution, taking the Amex with it.
    * ``1111-ada@example.com`` is a valid address starting INSIDE a card, so
      keeping the card threw the whole email away.
    """
    out = redact_text(text)
    assert must_not_contain not in out, f"PII leaked: {out!r}"
    assert PLACEHOLDER in out


@pytest.mark.parametrize(
    "text",
    [
        "Order 1234567890123456 ref",
        "id 12345678901234567890 ok",
        "Ref 1234 5678 9012 3456",
        "invoice 98765432109876 due",
        "tracking 123456789012345678 sent",
        "call 555-123-4567",
        "version 1.2.3",
        "range 10–20 items",
    ],
)
def test_the_adjacency_fixes_do_not_redact_ordinary_numbers(text):
    """Trimming a trailing digit off a bare run needs a second discriminator:
    Luhn alone passes one random 16-digit string in ten, and allowing a trim to
    any PAN length redacted ~40% of random 17-to-19-digit runs. Requiring an
    issuer prefix brought that to ~16% and left every string here untouched."""
    assert redact_text(text) == text


def test_a_re_anchored_pan_must_resolve_to_a_consistent_scheme():
    """Prefix and length together, not either alone.

    A prefix check by itself does not hold: ``34567890123456`` carries an Amex
    prefix and passes Luhn, and accepting it rewrote ``Order 1234567890123456``
    as ``Order 12[REDACTED]``. It is only inconsistent - Amex is 15 digits and
    that is 14.
    """
    from task3_streaming_guardrail.redactor import _card_scheme

    assert _card_scheme("4111111111111111") == "visa"
    assert _card_scheme("4222222222222") == "visa"          # 13-digit legacy
    assert _card_scheme("378282246310005") == "amex"
    assert _card_scheme("5555555555554444") == "mastercard"
    assert _card_scheme("6011111111111117") == "discover"
    assert _card_scheme("30569309025904") == "diners"
    assert _card_scheme("3530111333300000") == "jcb"

    # Amex prefix, wrong length - the exact substring that caused the regression.
    assert _card_scheme("34567890123456") is None
    assert _card_scheme("1234567890123456") is None
    assert _card_scheme("9876543210987654") is None


@pytest.mark.parametrize(
    "text, value",
    [
        ("9visa 94111111111111111 x", "4111111111111111"),
        ("ref 9378282246310005 end", "378282246310005"),
        ("94111111111111111\nada@example.com x", "4111111111111111"),
    ],
)
def test_a_stray_digit_in_front_of_a_bare_pan_does_not_hide_it(text, value):
    """No separator-derived offset can reach past a leading stray digit, so
    offsets inside each digit token are tried as well. Safe only because a bare
    re-anchor must resolve to a consistent card scheme."""
    out = redact_text(text)
    assert value not in out, f"PAN leaked: {out!r}"


@pytest.mark.parametrize(
    "text",
    [
        "visa 4111111111111111", "amex 378282246310005", "mc 5555555555554444",
        "disc 6011111111111117", "diners 30569309025904", "jcb 3530111333300000",
    ],
)
def test_every_major_scheme_is_still_redacted(text):
    assert PLACEHOLDER in redact_text(text)


@pytest.mark.parametrize(
    "event, secret",
    [
        # Legacy /v1/completions shape.
        ('{"choices":[{"index":0,"text":"ada@example.com and 4111111111111111"}]}',
         "ada@example.com"),
        # Content-parts array.
        ('{"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"ada@example.com"}]}}]}',
         "ada@example.com"),
        # Non-streaming message container mid-stream.
        ('{"choices":[{"index":0,"message":{"role":"assistant","content":"card 4111111111111111"}}]}',
         "4111111111111111"),
        # Anthropic-style content_block_delta.
        ('{"type":"content_block_delta","delta":{"type":"text_delta","text":"ada@example.com"}}',
         "ada@example.com"),
        # Somewhere nobody enumerated.
        ('{"choices":[],"usage":{"note":"ada@example.com"}}', "ada@example.com"),
    ],
)
async def test_provider_shapes_outside_the_known_fields_are_still_redacted(event, secret):
    """The field list was fail-open: only ``content``/``reasoning_content``/
    ``refusal`` and tool-call arguments were redacted, so every other shape a
    provider emits went through verbatim. Each of these leaked an address or a
    card end-to-end.

    Inverted to a deny-list of structural keys, so an unenumerated shape is
    redacted by default rather than forwarded.
    """
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse as _Streaming

    provider = FastAPI()

    @provider.post("/v1/chat/completions")
    async def emit():
        async def generate():
            yield f"data: {event}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return _Streaming(generate(), media_type="text/event-stream")

    @provider.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    with live_server(provider) as upstream:
        app = create_app(upstream_url=f"{upstream}/v1/chat/completions")
        with live_server(app) as gateway_url:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{gateway_url}/v1/chat/completions", json={"model": "m", "stream": True}
                )
    assert secret not in response.text, f"PII leaked verbatim: {response.text!r}"


async def test_structural_fields_are_not_redacted(gateway):
    """Inverting the list must not start redacting ids, models and roles."""
    _, chunks, _ = await collect(gateway, scenario="clean")
    assert chunks, "no chunks captured"
    for chunk in chunks:
        assert PLACEHOLDER not in chunk.get("id", "")
        assert PLACEHOLDER not in chunk.get("model", "")
        for choice in chunk.get("choices", []):
            role = (choice.get("delta") or {}).get("role")
            if role is not None:
                assert role == "assistant", f"role was rewritten: {role!r}"


@pytest.mark.parametrize(
    "text, redacts",
    [
        ("Contact user@192.168.1.1.", True),
        ("Contact user@192.168.1.1 now", True),
        ("mail a@10.0.0.1, ok", True),
        ("see host 10.0.0.1 for logs", False),
        ("version 1.2.3.4 released", False),
    ],
)
def test_an_ip_literal_address_survives_sentence_punctuation(text, redacts):
    """``(?![\\d.])`` treated a full stop as if it continued the literal, so an
    address at the end of a sentence matched nothing at all."""
    out = redact_text(text)
    assert (PLACEHOLDER in out) is redacts, out


def _sse_provider(events):
    from fastapi import FastAPI as _F
    from fastapi.responses import StreamingResponse as _S

    application = _F()

    @application.post("/v1/chat/completions")
    async def emit():
        async def generate():
            for event in events:
                yield f"data: {event}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return _S(generate(), media_type="text/event-stream")

    @application.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    return application


async def _through_gateway(events):
    with live_server(_sse_provider(events)) as upstream:
        app = create_app(upstream_url=f"{upstream}/v1/chat/completions")
        with live_server(app) as gateway_url:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{gateway_url}/v1/chat/completions", json={"model": "m", "stream": True}
                )
    return response.text


async def test_logprobs_does_not_bypass_redaction():
    """``logprobs`` was on the structural deny-list, so the same card that was
    redacted in ``content`` sat in the clear one field over - and a client can
    ask for it by sending ``{"logprobs": true}``.

    The deny-list is hand-written, which is exactly the failure mode inverting
    the allow-list was meant to remove.
    """
    body = await _through_gateway([
        '{"choices":[{"index":0,"delta":{"content":"Card 4111111111111111 ok"},'
        '"logprobs":{"content":[{"token":"4111111111111111","logprob":-0.1,'
        '"top_logprobs":[{"token":"ada@example.com","logprob":-2.0}]}]},'
        '"finish_reason":null}]}'
    ])
    assert "4111111111111111" not in body, f"card leaked via logprobs: {body!r}"
    assert "ada@example.com" not in body, f"email leaked via logprobs: {body!r}"


async def test_repeated_metadata_is_not_emptied_and_concatenated():
    """Hold-back only makes sense for a value delivered piece by piece. Applied
    to a per-chunk ``request_id`` it emptied every chunk and re-emitted the
    pieces joined together in a trailing chunk with no ``choices`` key."""
    chunk = (
        '{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",'
        '"request_id":"req-0001","choices":[{"index":0,"delta":{"content":"hi "},'
        '"finish_reason":null}]}'
    )
    body = await _through_gateway([chunk] * 3)
    ids = [
        json.loads(line[6:]).get("request_id")
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert ids == ["req-0001"] * 3, f"metadata was mangled: {ids}"
    assert "req-0001req-0001" not in body


async def test_a_single_shot_structured_field_is_not_split_across_chunks():
    """An annotation URL arrives complete, so hold-back truncated it and
    delivered the remainder as two separate later chunks."""
    body = await _through_gateway([
        '{"choices":[{"index":0,"delta":{"annotations":[{"type":"url_citation",'
        '"url_citation":{"url":"https://example.com/a","title":"Ex"}}]},'
        '"finish_reason":null}]}'
    ])
    chunks = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert len(chunks) == 1, f"one annotation arrived as {len(chunks)} chunks"
    citation = chunks[0]["choices"][0]["delta"]["annotations"][0]["url_citation"]
    assert citation["url"] == "https://example.com/a"
    assert citation["title"] == "Ex"


async def test_pii_inside_a_single_shot_field_is_still_redacted():
    """Atomic redaction must still redact."""
    body = await _through_gateway([
        '{"choices":[{"index":0,"delta":{"annotations":[{"url_citation":'
        '{"url":"https://x.com/ada@example.com","title":"card 4111111111111111"}}]},'
        '"finish_reason":null}]}'
    ])
    assert "ada@example.com" not in body
    assert "4111111111111111" not in body


def _luhn_pan(prefix: str, length: int, rng) -> str | None:
    """A Luhn-valid PAN with the given prefix and length."""
    from task3_streaming_guardrail.redactor import _luhn_ok

    body = prefix + "".join(rng.choice("0123456789") for _ in range(length - len(prefix) - 1))
    for check in "0123456789":
        if _luhn_ok(body + check):
            return body + check
    return None


@pytest.mark.parametrize(
    "label, prefix, length",
    [
        ("visa-16", "4", 16), ("visa-13", "4", 13), ("visa-19", "4", 19),
        ("amex", "37", 15), ("diners", "36", 14), ("mastercard", "55", 16),
        ("discover", "6011", 16), ("jcb", "3530", 16), ("unionpay", "62", 16),
    ],
)
def test_every_scheme_redacts_in_groups_of_four(label, prefix, length):
    """Groups of four is how a card is actually written, and it was the one
    rendering the pattern could not see.

    Every group after the first had to be 4-6 digits, so any card whose length
    is not a multiple of four ended in a 1-3 digit group and matched nothing:
    Amex, Diners and 13-digit Visa leaked 200/200 in a sweep, while the SAME
    numbers redacted correctly bare or as 4-6-5. The tests missed it because
    they only ever exercised those schemes in the renderings that worked.
    """
    import random as _random

    rng = _random.Random(f"{label}-{length}")
    for _ in range(40):
        pan = _luhn_pan(prefix, length, rng)
        assert pan, f"could not generate a {label}"
        grouped = " ".join(pan[i:i + 4] for i in range(0, len(pan), 4))
        for rendering in (grouped, grouped.replace(" ", "-"), pan):
            out = redact_text(f"The card on file is {rendering}.")
            assert rendering not in out, f"{label} leaked as {rendering!r}: {out!r}"
            assert PLACEHOLDER in out


@pytest.mark.parametrize(
    "text",
    [
        "range 1000 2000 3000 4000",
        "ticket 8888 9999 0000 1111",
        "Ref 1234 5678 9012 3456",
        "Order 1234567890123456 ref",
        "call 555-123-4567",
        "isbn 978-0-13-235088-4",
        "invoice 1,234,567.89 due",
        "ids 11 22 33 44 55",
        "date 2024-01-15",
    ],
)
def test_widening_the_card_pattern_did_not_invent_redactions(text):
    """Luhn alone passes about one random digit run in ten, so the wider
    pattern needs the scheme check on the MAIN scan, not only when re-anchoring.
    Without it, "range 1000 2000 3000 4000" and "ticket 8888 9999 0000 1111"
    both redacted - Luhn-valid, and no scheme issues them."""
    assert redact_text(text) == text


def test_the_card_validator_requires_luhn_and_a_scheme():
    from task3_streaming_guardrail.redactor import _is_card

    assert _is_card("3782 8224 6310 005")
    assert _is_card("30569309025904")
    assert not _is_card("1000200030004000")     # Luhn-valid, no scheme
    assert not _is_card("4111111111111112")     # scheme, fails Luhn
    assert not _is_card("411111111111")         # too short to be any PAN

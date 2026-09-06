"""Incremental PII redaction over a character stream.

The problem
-----------
A model streams ``"Contact ada"``, then ``"@exampl"``, then ``"e.com now"``.
Nothing in any single chunk looks like an email; the concatenation does. Any
redactor that examines chunks independently misses it, and one that buffers the
whole response to be safe destroys the reason for streaming in the first place.

The approach: hold-back
-----------------------
Keep a *bounded* tail buffer and, on every chunk, answer one question:
**how much of what I am holding can never change?**

1. Find the earliest index ``h`` from which the text could still match
   differently once more input arrives - i.e. the earliest position whose match
   attempt runs to the end of the buffer. The ``regex`` module's ``partial=True``
   answers this directly, and it covers both "this could still grow into a
   match" and the subtler "this *is* a match but more input could extend it".
2. Anything before ``h`` is final. Redact the complete matches inside it and
   emit. Keep ``[h:]`` raw, so the next chunk re-examines it in full context.

Step 1 is the whole game. Redacting a match the moment it looks complete is
wrong, and wrong in the direction that leaks: ``ada@example.co`` is a valid
email, so an eager redactor emits ``[REDACTED]`` and then passes the ``m`` of
``.com`` through as stray text. Holding until the following character proves
the match cannot grow is what makes every chunking of a stream produce the same
output as redacting the text whole - which is the invariant the tests assert
over hundreds of random splits.

Consequences that matter:

* **TTFT.** Text is released as soon as it is provably safe, not on a timer and
  not at chunk boundaries. Prose flushes essentially immediately, because
  ``"Your order shipped"`` cannot be extended into any pattern - the engine
  fails at the space and the whole buffer goes out.
* **Memory is O(max_holdback)**, independent of response length. A 100k-token
  response holds the same bytes as a 10-token one.
* **A match cannot be split into safety.** ``ada@`` + ``example.com`` and
  ``a`` + ``d`` + ``a`` + ``@`` + ... redact identically; the buffer only ever
  sees one logical string.

The hold-back is capped. A stream that is one long viable prefix - a million
digits, say - would otherwise buffer without bound, which is a memory-exhaustion
vector. At the cap the oldest byte is released even though it is still
theoretically part of a pending match; the default cap is above the longest
legal email (RFC 5321 puts that at 254 characters) so this only fires on
adversarial input.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

import regex

PLACEHOLDER = "[REDACTED]"

#: Characters of already-emitted text kept purely as left context for matching.
#:
#: Every pattern here is anchored by a lookbehind (``(?<![\d\-])`` and friends)
#: that stops a match beginning in the middle of a token. Emitting text throws
#: that context away, so a buffer that *starts* at ``_@Z9X.ac`` matches as an
#: address, while the same characters in ``...example.com_@Z9X.ac`` cannot -
#: the ``m`` before the underscore fails the lookbehind. Without this window the
#: stream invents redactions the whole-text pass does not make.
#:
#: Must be at least as wide as the widest lookbehind, or the streamed result
#: stops matching the whole-text one. The widest is the bare-SSN keyword
#: ("social security number" plus separators, 26 characters); 48 leaves room.
_CONTEXT_CHARS = 48

#: Characters that may separate the groups of a written SSN or card number.
#:
#: Wider than the obvious space/dash/dot: a model writes ``4111,1111,1111,1111``
#: and ``4111_1111_1111_1111`` often enough, a non-breaking space is what you
#: get when the number was pasted out of a rendered document, and a newline is
#: what you get when it is written down a markdown list or table column.
_SEP = r"[ \-.,_/\u00a0\t\n\r‐‑‒–—−]"

#: Characters that can appear inside one written numeric value. Used to find
#: the runs the rescue pass re-examines.
_RUN_CHARS = "0123456789 -.,_/\u00a0\t\n\r‐‑‒–—−"

#: A bare nine-digit SSN, matched ONLY where the surrounding text says that is
#: what it is.
#:
#: Nine digits alone are an order number, a zip+4 or a part code, so matching
#: them unconditionally trades one leak for a flood of false positives. The
#: keyword makes the common real case ("SSN: 123456789") a hit and leaves bare
#: identifiers alone.
#:
#: Compiled SEPARATELY rather than added to the main alternation, and that is a
#: measured decision. Inside it, this branch's variable-width lookbehind and the
#: email branch's 128-character local part interacted under ``partial=True`` and
#: took the redactor from 400k chars/s to 5k, with the hold-back buffer pinned
#: at its 1024 cap so prose stopped flushing. Either branch alone was fine; only
#: the combination was pathological. Two engines, each cheap, is the fix.
KEYWORD_SSN_EXPRESSION = (
    r"(?<=(?i:ssn|social\s+security(?:\s+number|\s+no\.?|\s+#)?)[^\d]{0,8})"
    r"\d{9}(?!\d)"
)

def _card_scheme(digits: str) -> str | None:
    """The card scheme these digits belong to, or ``None``.

    Prefix **and** length together, not either alone. That pairing is what
    makes re-anchoring inside a digit run safe enough to do at all.

    Luhn is a one-in-ten filter, so applying it at every offset of a long run
    turns coincidences into redactions: it flagged 24-35% of random
    17-to-19-digit runs and rewrote ``Order 1234567890123456`` as
    ``Order 12[REDACTED]``. A prefix check alone does not fix that - the
    offending substring there was ``34567890123456``, which carries an Amex
    prefix. It is only *inconsistent*: Amex is 15 digits and that is 14. Every
    real PAN satisfies both halves; coincidental windows rarely do.
    """
    length = len(digits)
    if not digits.isdigit():
        return None
    head4 = int(digits[:4]) if length >= 4 else -1
    head3 = int(digits[:3]) if length >= 3 else -1
    if digits[0] == "4" and length in (13, 16, 19):
        return "visa"
    if digits[:2] in {"34", "37"} and length == 15:
        return "amex"
    if (digits[:2] in {"51", "52", "53", "54", "55"} or 2221 <= head4 <= 2720) and length == 16:
        return "mastercard"
    if (digits[:4] == "6011" or digits[:2] == "65" or 644 <= head3 <= 649) and length in (16, 19):
        return "discover"
    if (digits[:2] in {"36", "38"} or 300 <= head3 <= 305) and length == 14:
        return "diners"
    if 3528 <= head4 <= 3589 and length == 16:
        return "jcb"
    if digits[:2] == "62" and 16 <= length <= 19:
        return "unionpay"
    return None


#: Digit counts a real card actually has: Visa legacy 13, Diners 14, Amex 15,
#: most 16, Maestro 19. Only these are tried when trimming a trailing digit off
#: a bare run, which keeps the extra attempts to five and, more importantly,
#: keeps 17 and 18 - lengths no scheme issues - from being invented.
_PAN_LENGTHS = (16, 15, 19, 14, 13)

#: Characters of preceding text searched for an SSN keyword. Must cover the
#: widest lookbehind ("social security number" plus its 8-character gap).
_KEYWORD_WINDOW = 40

#: How many leading characters of a digit token may be a stray prefix.
_MAX_STRAY_PREFIX = 3

#: How many times the rescue pass may re-run after overlap resolution.
#:
#: Each pass can free a region by discarding a span that was suppressing a
#: candidate, so one pass is not always enough. It converges in two on every
#: case seen; the bound stops a pathological input from looping.
_RESCUE_PASSES = 4

#: Patterns the rescue pass re-anchors. Only the validated numeric ones: a
#: rescued match must survive Luhn or the SSA rules, which is what keeps the
#: extra scanning from inventing matches.
_RESCUE_PATTERNS = ("ssn", "credit_card")

#: The longest string any pattern can match. Every pattern is length-bounded so
#: this number exists at all: the email local part is capped at 128 and the
#: domain at 9 labels of 63, which is the widest of them by a distance.
#: ``StreamingRedactor`` refuses to run with a hold-back below this, because a
#: hold-back smaller than the longest match is not a memory bound - it is a
#: guarantee that a long enough value gets cut in half and its front emitted in
#: the clear.
MAX_MATCH_LENGTH = 128 + 1 + 63 + (9 * 64) + 25  # = 793


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pattern:
    name: str
    expression: str
    #: Optional second-stage check. Regex alone over-matches on numeric data -
    #: a 16-digit order id is not a credit card - so a validator can veto.
    validator: object = None


def _luhn_ok(text: str) -> bool:
    """Luhn checksum over a plausible PAN length.

    The upper bound matters as much as the lower one. Without it a 20-digit
    run that happens to satisfy Luhn was redacted as a card - and the grouped
    pattern admits up to 28 digits, so repetitive numeric output produced
    false positives. Real PANs run 13 (Visa legacy) to 19 (Maestro) digits.
    """
    digits = [int(c) for c in text if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _ssn_ok(text: str) -> bool:
    """Reject SSN shapes the SSA never issues, to cut false positives."""
    digits = "".join(c for c in text if c.isdigit())
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    return area not in ("000", "666") and not area.startswith("9") and group != "00" and serial != "0000"


def _is_card(text: str) -> bool:
    """Luhn AND a scheme whose prefix and length agree.

    Both halves are needed now that the pattern is layout-agnostic. Luhn alone
    passes about one random digit run in ten, so on its own the wider pattern
    redacted "range 1000 2000 3000 4000" and "ticket 8888 9999 0000 1111" -
    Luhn-valid, and no card scheme issues them. The scheme check was already
    written and already applied when re-anchoring inside a run; it simply was
    not applied on the main scan.
    """
    digits = "".join(character for character in text if character.isdigit())
    # Ordered cheapest-first: a length test rejects most candidates without
    # touching the scheme table, and the checksum runs last. On grouped-digit
    # text the candidate set is large, so the order is worth stating.
    if not 13 <= len(digits) <= 19:
        return False
    if _card_scheme(digits) is None:
        return False
    return _luhn_ok(digits)


#: The three classes the brief names: emails, SSNs, credit cards.
DEFAULT_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        "email",
        # Two deliberate details, both about worst-case time rather than
        # matching power:
        #   * a left lookbehind, so a failed attempt is not retried at every
        #     offset inside the local part - that retry is what turns one
        #     failure into O(n^2) and froze the event loop for 18s on a 16KB
        #     delta. A hyphen is deliberately NOT in it: with "-" excluded,
        #     "ada@example.com-bob@test.org" redacted the first address and
        #     left the second in the clear, because the scan resumed on a
        #     character the lookbehind then vetoed;
        #   * RFC-derived length bounds (local part 128, label 63, 9 labels), so
        #     a match has a finite maximum length. MAX_MATCH_LENGTH depends on
        #     that being true.
        # Note: no possessive quantifiers. They would also bound backtracking,
        # but they silently break `partial=True` - the engine stops considering
        # the "more input is coming" continuations, so a long address streams
        # out in the clear instead of being held.
        # Character classes are ASCII on purpose. Widening them to \p{L}/\p{N}
        # to catch an internationalised local part was tried and reverted: it
        # cost ~100x throughput on PII-dense text (3.4k chars/s, below the
        # floor this file asserts), and \p{N} in the trailing lookahead
        # reintroduced a one-character bypass, because a fraction like "1/4"
        # counts as a number and blocked the match. A narrow miss is worth more
        # than a bypass plus a performance collapse. See "Known gaps".
        # Digits are NOT in the lookbehind. A local part may contain them, so
        # excluding a digit-preceded start meant an address glued to the end of
        # a card ("4111 1111 1111 1111ada@example.com") could not be matched at
        # all once the card claimed its span - the address reached the client
        # whole. Letters and the local-part punctuation still block a mid-token
        # start, which is what keeps the retry bounded.
        r"(?<![A-Za-z._%+])"
        r"[A-Za-z0-9._%+\-]{1,128}"
        r"@(?:"
        # Named domain: labels, then a letter TLD.
        r"[A-Za-z0-9](?:[A-Za-z0-9\-]{0,62}[A-Za-z0-9])?"
        r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]{0,62}[A-Za-z0-9])?){0,8}"
        # A TLD is alphabetic, so only a LETTER can continue it. Forbidding a
        # trailing digit did not narrow the match, it destroyed it:
        # "ada@example.com2024" matched nothing and the address streamed out
        # whole - the same failure the numeric patterns had with a trailing dash.
        r"\.[A-Za-z]{2,24}(?![A-Za-z])"
        r"|"
        # Or a bare IPv4 literal, which has no TLD to anchor on. ASCII, cheap,
        # and "ada@192.168.1.1" is an address a model does write.
        # Here a digit or a dot DOES continue the address, so those stay.
        # ``(?!\d)(?!\.\d)`` rather than ``(?![\d.])``: a further digit, or a
        # dot followed by one, means a longer literal we have not seen the end
        # of - but a sentence-ending period does not, and rejecting it meant
        # "Contact user@192.168.1.1." matched nothing at all.
        r"\d{1,3}(?:\.\d{1,3}){3}(?!\d)(?!\.\d)"
        r")",
    ),
    Pattern(
        "ssn",
        # Word boundaries keep this from firing inside a longer digit run.
        r"(?<![\d\-.])\d{3}" + _SEP + r"\d{2}" + _SEP + r"\d{4}(?!\d)",
        validator=_ssn_ok,
    ),
    Pattern(
        "credit_card",
        # Grouped or bare, but not an arbitrary run of digits-and-separators.
        # The old free-form run was greedy, so "4111 1111 1111 1111 9" matched
        # as 17 digits, failed Luhn, and the rescan - which restarts one
        # character later, never shorter at the same place - never tried the
        # valid 16-digit card inside it. Any digit after a card leaked it.
        # Both branches end in ``(?!\d)`` - no dash. Excluding a trailing dash
        # did not narrow the match, it destroyed it: "4111 1111 1111 1111-01"
        # matched nothing at all and the PAN streamed out in full. The bare
        # branch had already been corrected for exactly this; the grouped one
        # had not, which made it an inconsistency rather than a position.
        # "More input may still be coming" is the hold-back's job, decided by
        # partial matching - not something a lookahead can express.
        # Layout-agnostic on purpose. The old branch required every group after
        # the first to be 4-6 digits, so any card whose length is not a
        # multiple of four was invisible in the commonest rendering there is:
        # "3782 8224 6310 005" (Amex), "3056 9309 0259 04" (Diners) and
        # "4222 2222 2222 2" (13-digit Visa) all leaked whole, 200/200 in a
        # sweep, while the SAME numbers redacted fine bare or as 4-6-5.
        #
        # The shape that actually matters is a SHORT TRAILING GROUP: a card
        # whose length is not a multiple of four ends in 1-3 digits. Groups in
        # the middle stay 4-6.
        #
        # Fully layout-agnostic was tried first (any group 1-6) and is not
        # shippable: it took grouped-digit scanning from 191ms to 3823ms per
        # 32k characters and started swallowing SSNs, because almost every
        # digit run became a candidate. The decision still rests with the
        # validators - Luhn AND scheme, see _is_card - but the pattern has to
        # stay narrow enough to keep the candidate set small.
        r"(?<![\d\-.])(?:\d{4,6}(?:" + _SEP + r"\d{4,6}){1,4}(?:" + _SEP + r"\d{1,3})?(?!\d)"
        r"|\d{13,19}(?!\d))",
        validator=_is_card,
    ),
)


def _resolve_overlaps(found: list) -> list:
    """Sorted, non-overlapping spans, preferring the earliest and longest.

    ``_emit`` walks spans with a single cursor, so an out-of-order or
    overlapping span is silently skipped - which drops a redaction and lets the
    value through.
    """
    ordered = sorted(found, key=lambda m: (m.start(), -(m.end() - m.start())))
    disjoint, cursor = [], -1
    for match in ordered:
        if match.start() >= cursor:
            disjoint.append(match)
            cursor = match.end()
    return disjoint


@dataclass
class RedactionStats:
    """Per-stream observability. Cheap to keep, and the only durable record -
    the redactor deliberately never retains the values it removed."""

    counts: dict[str, int] = field(default_factory=dict)
    characters_in: int = 0
    characters_out: int = 0

    def record(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.counts.values())


# --------------------------------------------------------------------------- #
# Redactor
# --------------------------------------------------------------------------- #
_FIRST_CHARS_CACHE: dict[tuple[str, ...], frozenset[str]] = {}


def _first_chars_for(patterns: tuple[Pattern, ...], combined) -> frozenset[str]:
    # Keyed on the pattern text, not ``id(patterns)``: a tuple's id is reused
    # after it is collected, so a caller passing fresh tuples could be served
    # another pattern set's answer.
    key = tuple(p.expression for p in patterns)
    cached = _FIRST_CHARS_CACHE.get(key)
    if cached is None:
        cached = frozenset(
            chr(c) for c in range(32, 127) if combined.fullmatch(chr(c), partial=True) is not None
        )
        _FIRST_CHARS_CACHE[key] = cached
    return cached


class StreamingRedactor:
    """Feed text in, get safe text out. One instance per response stream."""

    def __init__(
        self,
        patterns: tuple[Pattern, ...] = DEFAULT_PATTERNS,
        placeholder: str = PLACEHOLDER,
        max_holdback: int = 1024,
    ) -> None:
        if max_holdback < MAX_MATCH_LENGTH:
            raise ValueError(
                f"max_holdback={max_holdback} is below the longest possible match "
                f"({MAX_MATCH_LENGTH}); a smaller window would emit the front half of "
                f"a long value in the clear"
            )
        self._patterns = patterns
        self._placeholder = placeholder
        self._max_holdback = max_holdback
        self._buffer = ""
        self._context = ""
        self.stats = RedactionStats()

        # One alternation, compiled once. Named groups let a single pass
        # identify which pattern fired, so scanning cost does not grow with
        # the number of patterns the way sequential ``sub`` calls would.
        self._combined = regex.compile(
            "|".join(f"(?P<{p.name}>{p.expression})" for p in patterns)
        )
        self._validators = {p.name: p.validator for p in patterns if p.validator is not None}
        self._keyword_ssn = regex.compile(f"(?P<ssn_bare>{KEYWORD_SSN_EXPRESSION})")
        self._validators.setdefault("ssn_bare", _ssn_ok)

        # A second alternation over the numeric patterns with their leading
        # lookbehind removed. The lookbehind is what stops a match starting in
        # the middle of a token - necessary, but it also means a real value
        # GLUED to a preceding digit, dot or dash is never even attempted:
        # "04111 1111 1111 1111" and "0123-45-6789" both streamed out in full,
        # because the scan can only ever move forward and the card starts one
        # character *before* the first position the lookbehind allows.
        rescue = [p for p in patterns if p.name in _RESCUE_PATTERNS]
        self._rescue = (
            regex.compile(
                "|".join(
                    f"(?P<{p.name}>{regex.sub(r'^\(\?<!\[[^\]]*\]\)', '', p.expression)})"
                    for p in rescue
                )
            )
            if rescue
            else None
        )

        # Characters that can begin some pattern. Used only as a fast reject in
        # the hold-back scan, so it must be a superset - it is derived by
        # probing the compiled alternation rather than hand-maintained, which
        # keeps it correct when a pattern is added. Cached per pattern set:
        # probing 95 characters costs ~0.09ms per construction, which one
        # redactor per stream turns into real blocked-event-loop time.
        self._first_chars = _first_chars_for(patterns, self._combined)

    # -- internals ---------------------------------------------------------- #
    def _shorter_valid_match(self, text: str, start: int, end: int, pattern=None):
        """Look for a shorter validated match anchored at ``start``.

        The scan finds the *greedy* match at a position; if its validator
        vetoes it, moving straight on to ``start + 1`` never asks whether a
        shorter match at the same place would validate. So
        ``"4111 1111 1111 1111 12500"`` matched as five groups, failed Luhn,
        and the card inside streamed out in full - 59 of 75 Luhn-valid PANs
        followed by a trailing group leaked, including in sentences an
        assistant would ordinarily write.

        ``fullmatch`` with a shrinking ``endpos`` asks the engine for exactly
        that: the longest match anchored at ``start`` that ends at or before
        each candidate boundary. Truncating deliberately relaxes the trailing
        lookahead, which is correct here - the point is to find a valid value
        *inside* a longer run.
        """
        engine = pattern if pattern is not None else self._combined
        if not any(ch in _RUN_CHARS and not ch.isdigit() for ch in text[start:end]):
            # An unbroken run of digits. It must not be scanned window-by-window
            # - a long order reference is not a card because some 13-digit slice
            # inside it happens to satisfy Luhn - but it cannot be refused
            # outright either: one digit glued to the end ("41111111111111115")
            # makes the greedy match fail Luhn and leaked the PAN whole.
            #
            # So: trim from the END only, keeping the start anchored, and only
            # to a length that is an actual PAN length. That is at most five
            # attempts instead of a sliding window, and it cannot invent a match
            # that does not begin where the run begins.
            return self._trimmed_pan(text, start, end, engine)

        # Only boundaries are candidate endpoints. A match cannot end in the
        # middle of a digit group, so walking every index re-asks the engine the
        # same question a dozen times per group; on PII-dense text that cost
        # ~15% of throughput for nothing.
        # Two passes. The first tries only group boundaries, which is where a
        # written value almost always ends and keeps the common case cheap.
        #
        # The second tries every endpoint, and it is not optional: a stray
        # digit glued to the end ("4111 1111 1111 11115") makes the greedy
        # match a 17-digit run that fails Luhn, and the real 16-digit PAN ends
        # in the MIDDLE of the final group - precisely the endpoint the
        # boundary pass skips. That leaked the card, and an Amex, and it is
        # trivially triggered by asking a model to print a card followed by a
        # digit. The fallback only runs after a veto and a boundary miss, so
        # ordinary text never pays for it.
        for boundaries_only in (True, False):
            stop = end - 1
            while stop > start:
                if (
                    boundaries_only
                    and text[stop - 1].isalnum()
                    and stop < len(text)
                    and text[stop].isalnum()
                ):
                    stop -= 1
                    continue
                match = engine.fullmatch(text, start, stop)
                if match is None:
                    stop -= 1
                    continue
                name = match.lastgroup
                validator = self._validators.get(name)
                if validator is None or validator(match.group()):
                    return match
                stop = match.end() - 1
        return None

    def _trimmed_pan(self, text: str, start: int, end: int, engine):
        """A valid PAN occupying the front of a bare digit run, if there is one."""
        for length in _PAN_LENGTHS:
            stop = start + length
            if stop >= end:
                continue
            match = engine.fullmatch(text, start, stop)
            if match is None:
                continue
            digits = "".join(c for c in match.group() if c.isdigit())
            if _card_scheme(digits) is None:
                continue
            validator = self._validators.get(match.lastgroup)
            if validator is None or validator(match.group()):
                return match
        return None

    def _valid_matches(self, text: str, start: int = 0) -> list:
        """Every complete match in ``text`` that also survives its validator.

        On a veto the scan does two things, and it needs both. It re-examines
        the vetoed span for a shorter match that does validate (a real card
        inside a longer digit run), and failing that it advances by a single
        character rather than past the whole span (a real card *after* a stray
        leading digit). Skipping either one leaks a PAN.

        Scanning begins at ``start`` - the end of the carried left context, so
        already-decided text is not re-scanned. ``search(text, pos)`` still lets
        a lookbehind inspect characters before ``pos``, which slicing would not,
        so the context does its job without being re-examined.

        Matches come back sorted and non-overlapping: the cursor only ever
        moves forward, and a shorter inner match starts where the vetoed one
        did and ends no later.
        """
        return self._scan_to_fixed_point(text, start)

    def _scan(self, text: str, start: int, limit: int) -> list:
        """The forward scan, restricted to ``[start, limit)``.

        ``search`` is given both bounds rather than a slice so a lookbehind can
        still see the characters before ``start`` - the carried context has to
        keep working.
        """
        matches = []
        position = start
        length = limit
        while position <= length:
            match = self._combined.search(text, position, limit)
            if match is None:
                break
            name = match.lastgroup
            if name is None:  # pragma: no cover - every branch is a named group
                break
            validator = self._validators.get(name)
            if validator is not None and not validator(match.group()):
                inner = self._shorter_valid_match(text, match.start(), match.end())
                if inner is not None:
                    matches.append(inner)
                    position = max(inner.end(), inner.start() + 1)
                    continue
                # Nothing valid anchored here. Step forward by one so a genuine
                # value starting later inside the rejected span is still seen.
                position = match.start() + 1
                continue
            matches.append(match)
            position = max(match.end(), match.start() + 1)
        return matches

    @staticmethod
    def _gaps(covered: list[tuple[int, int]], start: int, end: int):
        """Regions between selected spans, where a discarded match may have been."""
        cursor = start
        for span_start, span_end in covered:
            if span_start > cursor:
                yield cursor, span_start
            cursor = max(cursor, span_end)
        if cursor < end:
            yield cursor, end

    def _scan_to_fixed_point(self, text: str, start: int) -> list:
        """Scan, resolve overlaps, then rescan whatever the resolution freed.

        Resolving an overlap discards the losing span, and that span may have
        been the only thing found in the region it covered - so dropping it
        silently drops a redaction. Two real leaks came from this. A spurious
        Luhn-valid run swallowed a real Amex that started inside it, then was
        itself discarded in favour of an earlier card. And "1111-ada@example.com"
        is a syntactically valid address that starts INSIDE a card, so keeping
        the card threw the whole email away and it reached the client in the
        clear.

        Rescanning the vacated regions is the fix, and it has to include the
        ordinary forward scan, not just the rescue pass: in the second case the
        email is an ordinary match that simply lost.
        """
        length = len(text)
        keyword = [
            m for m in self._keyword_ssn.finditer(text, start) if _ssn_ok(m.group())
        ]
        selected = _resolve_overlaps(self._scan(text, start, length) + keyword)

        # Rescue against the spans that SURVIVE overlap resolution, and repeat
        # until it stops finding anything.
        #
        # Gating on the pre-resolution list leaked a full Amex. The forward
        # scan produced a spurious 16-digit "card" spanning "1111 1111
        # 1111\t3782" - Luhn-valid by coincidence - which covered the real Amex
        # that started inside it, so the rescue skipped that position. The
        # resolver then dropped the spurious span in favour of an earlier real
        # card, and the Amex went with it. A span that is discarded must not
        # get to suppress a candidate.
        for _ in range(_RESCUE_PASSES):
            covered = [(m.start(), m.end()) for m in selected]
            found = self._rescue_matches(text, start, covered)
            for gap_start, gap_end in self._gaps(covered, start, length):
                found += self._scan(text, gap_start, gap_end)
            if not found:
                break
            merged = _resolve_overlaps(selected + found)
            if [(m.start(), m.end()) for m in merged] == covered:
                break
            selected = merged
        return selected

    def _numeric_runs(self, text: str, start: int):
        """Maximal spans of characters that can make up one written number."""
        index, length = start, len(text)
        while index < length:
            if text[index] not in _RUN_CHARS:
                index += 1
                continue
            run_start = index
            while index < length and text[index] in _RUN_CHARS:
                index += 1
            yield run_start, index

    def _rescue_at(self, text: str, position: int):
        """A validated match anchored at ``position``, ignoring the lookbehind.

        Accepted only if the value is grouped, or begins immediately after a
        separator. That restriction is what stops the extra scanning turning a
        long bare digit run into a false positive: a 19-digit order reference
        has no separator to anchor after and no separator inside, so no window
        of it is ever offered - which is the same rule ``_shorter_valid_match``
        already applies.
        """
        match = self._rescue.match(text, position)
        if match is None:
            return None
        name = match.lastgroup
        validator = self._validators.get(name)
        if validator is not None and not validator(match.group()):
            match = self._shorter_valid_match(text, position, match.end(), pattern=self._rescue)
            if match is None:
                return None
        text_of = match.group()
        grouped = any(ch in _RUN_CHARS and not ch.isdigit() for ch in text_of)

        # Re-anchoring INSIDE a digit group, on a grouped CARD, needs its
        # leading group to look like a card's. Every real rendering opens with
        # at least four digits, so requiring that costs nothing real and stops
        # the widened pattern finding coincidences: "Ref 1234 5678 9012 3456"
        # re-anchored two digits in and matched "34 5678 9012 3456", which is
        # Luhn-valid with a Diners prefix and is not a card.
        #
        # Cards only. An SSN's leading group is three digits by definition, so
        # applying this to every grouped value stopped "His SSN is
        # 0123-45-6789" being re-anchored past the stray leading zero.
        if (
            grouped
            and match.lastgroup == "credit_card"
            and position > 0
            and text[position - 1].isdigit()
        ):
            leading = 0
            for character in text_of:
                if not character.isdigit():
                    break
                leading += 1
            if leading < 4:
                return None
        after_separator = position > 0 and text[position - 1] in _RUN_CHARS and not text[position - 1].isdigit()
        if not (grouped or after_separator):
            # A bare run re-anchored mid-digits. Luhn alone is far too weak
            # here - a 1-in-10 filter applied at every offset - so the digits
            # must also form a scheme whose prefix and length agree.
            digits = "".join(c for c in text_of if c.isdigit())
            if _card_scheme(digits) is None:
                return None
        return match

    #: How far back from a separator a written value can begin. The numeric
    #: patterns open with a fixed-width first group - ``\d{3}`` for an SSN,
    #: ``\d{4}`` for a card - so a grouped value that uses the separator at
    #: index ``i`` starts at exactly ``i - 3`` or ``i - 4``. Nowhere else.
    _RESCUE_BACKSETS = (3, 4)

    def _rescue_candidates(self, text: str, run_start: int, run_end: int, start: int):
        """The only positions inside a run where a rescued value can begin.

        This is a constant-factor reduction, not the asymptotic fix - the
        quadratic term lived in the coverage test in ``_rescue_matches``, and
        the binary search there is what removed it. Anchoring off separators
        cuts the number of regex attempts by roughly 5x on grouped digits,
        which is worth having on the hot path but is not what made it linear.
        """
        seen = set()
        if run_start >= start:
            seen.add(run_start)
        # A stray digit in FRONT hides a value completely, and no
        # separator-derived offset reaches past it: "94111111111111111" is a
        # 16-digit Visa behind one 9. So inside each digit-only token, try the
        # first few offsets too. Bounded to _MAX_STRAY_PREFIX because a run-on
        # is a character or two, not twenty, and safe only because a bare
        # re-anchor must resolve to a consistent card scheme.
        index = max(run_start, start)
        while index < run_end:
            if not text[index].isdigit():
                index += 1
                continue
            token_start = index
            while index < run_end and text[index].isdigit():
                index += 1
            for offset in range(1, _MAX_STRAY_PREFIX + 1):
                if token_start + offset < index:
                    seen.add(token_start + offset)
        if not any(text[i] in _RUN_CHARS and not text[i].isdigit() for i in range(run_start, run_end)):
            # A bare run has no separators to anchor off, so a stray digit in
            # FRONT hides the value completely: "94111111111111111" is a
            # 16-digit Visa behind one 9, and every candidate rule based on
            # separators generates nothing for it. Offsets are tried directly.
            # Safe only because a bare rescue additionally requires an issuer
            # prefix (see _rescue_at) on top of Luhn and a real PAN length -
            # without that this is the sliding window over digit runs that
            # turns order numbers into cards.
            seen.update(range(max(run_start, start), run_end))
            return sorted(seen)
        for index in range(max(run_start, start), run_end):
            if text[index].isdigit():
                continue
            # A bare value written straight after a separator ("100.4111...").
            if index + 1 < run_end and index + 1 >= start:
                seen.add(index + 1)
            # A grouped value whose first group ends at this separator.
            for back in self._RESCUE_BACKSETS:
                candidate = index - back
                if candidate >= max(run_start, start):
                    seen.add(candidate)
        return sorted(seen)

    def _rescue_matches(self, text: str, start: int, covered: list[tuple[int, int]]) -> list:
        """Re-examine numeric runs the forward scan could not start inside."""
        if self._rescue is None:
            return []
        # ``covered`` arrives sorted and disjoint (the forward scan only moves
        # forward), so a binary search answers "is this position already taken"
        # in log time. Scanning the list per candidate was quadratic.
        starts = [a for a, _ in covered]
        ends = [b for _, b in covered]
        found = []
        for run_start, run_end in self._numeric_runs(text, start):
            # Candidates within a run are ascending and rescued matches cannot
            # overlap, so one cursor covers everything this pass adds.
            cursor = 0
            for position in self._rescue_candidates(text, run_start, run_end, start):
                if position < cursor:
                    continue
                index = bisect_right(starts, position) - 1
                if index >= 0 and position < ends[index]:
                    continue
                match = self._rescue_at(text, position)
                if match is None:
                    continue
                found.append(match)
                cursor = match.end()
        return found

    def _holdback_index(self, text: str, start: int = 0) -> int:
        """Earliest index from which more input could still change the outcome.

        The predicate is ``fullmatch(..., partial=True)``: could the *entire*
        remainder of the buffer sit inside a single match? If yes, more input
        might extend or complete that match, so nothing from there on is final.

        ``fullmatch`` rather than ``search`` is the load-bearing detail, and the
        difference is not academic. Given ``"04111 1111 1111 "``, a search finds
        a *complete* 13-digit match that stops at the trailing space - it does
        not reach the end, so a "match reaches the end" test concludes the text
        is final and emits it. Four digits later that same run is a valid card,
        and its first group has already gone out in the clear. ``fullmatch``
        asks the right question: the remainder is still consumable by the
        pattern, so hold it.

        The scan is bounded by ``max_holdback``, and on ordinary prose almost
        every index fails in the first character or two.
        """
        length = len(text)
        # Never look further back than the carried context, or the hold-back
        # window: both are already-settled text.
        first = max(start, length - self._max_holdback)
        for index in range(first, length):
            # Cheap reject: skip indices whose character cannot begin any
            # pattern, so prose costs a set lookup rather than a regex call.
            # Only ASCII is decided this way - `\d` is Unicode-aware, so any
            # non-ASCII character is passed through to the engine rather than
            # assumed harmless.
            character = text[index]
            if character.isascii() and character not in self._first_chars:
                continue
            if self._combined.fullmatch(text, index, partial=True) is not None:
                return index
            if (
                character.isdigit()
                and self._keyword_precedes(text, index)
                and self._keyword_ssn.fullmatch(text, index, partial=True) is not None
            ):
                return index
            # The same blind spot as the forward scan: without this, the front
            # of "04111 1111 1111 1111" is emitted as settled text while the
            # rest is still arriving, and the card is split across the boundary
            # in the clear.
            if (
                self._rescue is not None
                and character.isdigit()
                and self._rescue.fullmatch(text, index, partial=True) is not None
            ):
                return index
        return length

    @staticmethod
    def _keyword_precedes(text: str, index: int) -> bool:
        """Whether an SSN keyword sits just before ``index``.

        A cheap substring test, and it is load-bearing rather than an
        optimisation. ``fullmatch(..., partial=True)`` against a variable-width
        lookbehind reports a partial match even where the lookbehind is not
        satisfied, so the keyword-SSN branch claimed that ANY digit might grow
        into a match. The hold-back then never drained: ordinary numbered prose
        ("Step 1: do the thing.") pinned the buffer at its 1024 cap, produced
        output on 11 of 77 deltas, and put ~1 second of dead air mid-response -
        the exact opposite of the TTFT the guardrail is supposed to preserve.

        Checking that the keyword is actually there first makes the partial
        test meaningful. The window covers the widest lookbehind in the file.
        """
        window = text[max(0, index - _KEYWORD_WINDOW):index].lower()
        return "ssn" in window or "social" in window

    def _emit(self, text: str, final: bool) -> str:
        """Split ``text`` into (safe, retained), redact the safe part, return it.

        ``self._buffer`` is left holding raw, un-redacted text - that is what
        makes re-examination in the next chunk's context correct.
        """
        context = self._context
        full = context + text
        base = len(context)
        length = len(full)

        spans = [
            (m.start(), m.end(), m.lastgroup) for m in self._valid_matches(full, start=base)
        ]
        hold = length if final else self._holdback_index(full, start=base)

        if not final:
            # Bound the buffer. Safe because ``max_holdback`` is enforced to be
            # at least ``MAX_MATCH_LENGTH``, so the forced cut is always further
            # back than any match could reach.
            if length - hold > self._max_holdback:
                hold = max(base, length - self._max_holdback)
            # A match may straddle the hold point. Push past it rather than
            # pulling back: a straddling match ends before the buffer end (one
            # reaching the end would have set the hold at its own start), so it
            # is final and can be redacted now.
            for span_start, span_end, _ in spans:
                if span_start < hold < span_end:
                    hold = span_end

        safe, self._buffer = full[base:hold], full[hold:]
        # Carry the tail of what just went out as context for the next call.
        self._context = full[:hold][-_CONTEXT_CHARS:]
        if not safe:
            return ""

        out: list[str] = []
        cursor = base
        for span_start, span_end, name in spans:
            if span_end > hold:
                break  # belongs to the retained region; re-examined next time
            out.append(full[cursor:span_start])
            out.append(self._placeholder)
            cursor = span_end
            self.stats.record(name)
        if cursor == base:
            emitted = safe
        else:
            out.append(full[cursor:hold])
            emitted = "".join(out)
        self.stats.characters_out += len(emitted)
        return emitted

    # -- public API --------------------------------------------------------- #
    def feed(self, chunk: str) -> str:
        """Consume a chunk; return the text that is now safe to emit."""
        if not chunk:
            return ""
        self.stats.characters_in += len(chunk)
        return self._emit(self._buffer + chunk, final=False)

    def flush(self) -> str:
        """End of stream: nothing more is coming, so everything held is final.

        This is where a PII value sitting at the very end of a response gets
        caught - it was withheld precisely because it might still have grown.
        """
        if not self._buffer:
            self._context = ""
            return ""
        remaining = self._emit(self._buffer, final=True)
        self._context = ""
        return remaining

    @property
    def pending(self) -> int:
        """Characters currently held back. Bounded by ``max_holdback``."""
        return len(self._buffer)


def redact_text(text: str, **kwargs) -> str:
    """Convenience for non-streaming callers and for oracle-testing the stream."""
    redactor = StreamingRedactor(**kwargs)
    return redactor.feed(text) + redactor.flush()

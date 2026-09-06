"""Strict input schemas for the two tools.

Every model is deliberately hostile to malformed input:

* ``extra="forbid"``      - unknown argument keys are a validation error, not
                            silently ignored. Stops an agent from smuggling
                            fields past the tool contract.
* ``strict=True``         - no coercion. ``"120.50"`` is not a float and
                            ``"CUST-00001 "`` is not a customer id. Pydantic
                            still accepts ``int`` where a ``float`` is declared,
                            which is the one coercion we want (``amount: 100``).
* ``allow_inf_nan=False`` - ``Infinity``/``NaN`` are valid JSON numbers to many
                            encoders and are not positive floats.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: ``CUST-XXXXX`` read as the literal prefix plus five ASCII digits.
#:
#: ``[0-9]``, not ``\d``: Python's ``\d`` is Unicode-aware and would accept
#: ``CUST-٠١٢٣٤``, while the same string is published as the tool's JSON Schema
#: ``pattern``, where ECMA-262 ``\d`` is ASCII-only. Client-side validation and
#: server-side enforcement have to agree about what a valid id is.
CUSTOMER_ID_PATTERN = re.compile(r"^CUST-[0-9]{5}$")

MIN_REASON_LENGTH = 10
#: A generous ceiling. The brief sets a floor and no cap, but the value is
#: echoed back in the receipt, so an unbounded one is amplification: a
#: 1,000,000-character reason was accepted and returned in full. Set well above
#: any real dispute narrative - one reviewer read 2,048 as an unrequested
#: restriction, another read the absence of any cap as the defect.
MAX_REASON_LENGTH = 8_192

#: ECMA-262 form of the reason rule, for the published schema: at least
#: MIN_REASON_LENGTH characters that are not control characters, and at
#: least one that is not whitespace. Kept beside the validator that
#: enforces it so the two cannot drift apart unnoticed.
REASON_PATTERN = (
    r"^(?=(?:[\u0000-\u001F\u007F]*[^\u0000-\u001F\u007F]){" + str(MIN_REASON_LENGTH) + r"})"
    r"(?=[\s\S]*[^\s])[\s\S]*$"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        frozen=True,
    )


class GetCustomerRecordInput(StrictModel):
    # Plain prose, no RST: this docstring is published verbatim as the tool's
    # schema description and is read by a model, not by Sphinx. The double
    # backticks were reaching the model's context as literal characters.
    """Arguments for get_customer_record: look up one customer billing record."""

    customer_id: str = Field(
        description="Customer identifier in the form CUST-XXXXX (five digits).",
        json_schema_extra={"pattern": CUSTOMER_ID_PATTERN.pattern, "examples": ["CUST-00042"]},
    )

    @field_validator("customer_id")
    @classmethod
    def _validate_customer_id(cls, value: str) -> str:
        if not CUSTOMER_ID_PATTERN.fullmatch(value):
            raise ValueError("customer_id must match CUST-XXXXX where X is a digit (e.g. CUST-00042)")
        return value


class TriggerRefundInput(StrictModel):
    # Published verbatim to the model - keep it plain. See above.
    """Arguments for trigger_refund: issue a refund against a customer's refundable balance."""

    customer_id: str = Field(
        description="Customer identifier in the form CUST-XXXXX (five digits).",
        json_schema_extra={"pattern": CUSTOMER_ID_PATTERN.pattern, "examples": ["CUST-00042"]},
    )
    amount: float = Field(gt=0, description="Refund amount in USD. Must be a positive number.")
    reason: str = Field(
        min_length=MIN_REASON_LENGTH,
        max_length=MAX_REASON_LENGTH,
        # Published so a client validating against this tool's own schema gets
        # the SAME answer as the server. Advertising only `minLength: 10` while
        # enforcing a printable-character count meant "Dup\tcharge" passed
        # client-side validation and was refused server-side - the exact
        # mismatch the [0-9]-not-\d choice above exists to avoid, reintroduced
        # one field over.
        json_schema_extra={"pattern": REASON_PATTERN},
        description=(
            f"Why the refund is issued. At least {MIN_REASON_LENGTH} characters, "
            "including at least one that is not whitespace."
        ),
    )

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        """At least one character a person could actually read.

        ``min_length`` counts code points, so ten spaces, ten newlines, ten NUL
        bytes and ten zero-width spaces all satisfied it - each one accepted,
        money moved, and the receipt echoed the whitespace back. The field
        exists for an audit trail, and none of those is one.

        Stated as a requirement rather than a deny-list - characters must BE
        printable and non-space - so there is no list of forbidden categories
        to keep in step, and one nobody thought of cannot slip past.

        Two rules, because one is not enough. Non-printable characters do not
        count toward the length, so ``"a" + "\x00" * 9`` is one character, not
        ten - requiring merely *some* visible character still accepted it. And
        at least one character must be non-space, so ten spaces is refused.
        Spaces themselves count toward the length: "Dup charge" is ten
        characters and the brief asks for ten.
        """
        # Non-printable characters do not count toward the length, and at least
        # one character must be visible. Spaces DO count - "Dup charge" is ten
        # characters and the brief asks for ten.
        printable = [character for character in value if character.isprintable()]
        if len(printable) < MIN_REASON_LENGTH:
            raise ValueError(
                f"reason must be at least {MIN_REASON_LENGTH} printable characters"
            )
        if not any(not character.isspace() for character in printable):
            raise ValueError("reason must contain at least one non-whitespace character")
        return value

    @field_validator("customer_id")
    @classmethod
    def _validate_customer_id(cls, value: str) -> str:
        if not CUSTOMER_ID_PATTERN.fullmatch(value):
            raise ValueError("customer_id must match CUST-XXXXX where X is a digit (e.g. CUST-00042)")
        return value


def json_schema_for(model: type[BaseModel]) -> dict:
    """Return the wire ``inputSchema`` for a tool.

    Pydantic emits ``title`` noise that clients do not need; the published
    schema is trimmed to the object shape an agent actually reads.
    """
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    schema["additionalProperties"] = False
    return schema


# --------------------------------------------------------------------------- #
# Output schemas
# --------------------------------------------------------------------------- #
# MCP says a server returning ``structuredContent`` SHOULD publish an
# ``outputSchema`` describing it - otherwise a client is handed structured data
# it has no way to validate. Both tools return structured results, so both
# declare one.
#
# These describe the SUCCESS shape only. A tool-execution failure comes back as
# ``isError: true`` with the detail in ``content`` and no ``structuredContent``
# at all, precisely so an error payload is never checked against a schema that
# was written for a success.
CUSTOMER_RECORD_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "customer_id": {"type": "string", "pattern": CUSTOMER_ID_PATTERN.pattern},
        "name": {"type": "string"},
        "email": {"type": "string"},
        "plan": {"type": "string"},
        "status": {"type": "string"},
        "refundable_balance": {"type": "number", "minimum": 0},
        "refund_count": {"type": "integer", "minimum": 0},
    },
    "required": [
        "customer_id", "name", "email", "plan", "status",
        "refundable_balance", "refund_count",
    ],
    "additionalProperties": False,
}

REFUND_RECEIPT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "refund_id": {"type": "string"},
        "customer_id": {"type": "string", "pattern": CUSTOMER_ID_PATTERN.pattern},
        "amount": {"type": "number", "exclusiveMinimum": 0},
        "reason": {"type": "string", "minLength": MIN_REASON_LENGTH},
        "status": {"type": "string"},
        "created_at": {"type": "string"},
        "remaining_refundable_balance": {"type": "number", "minimum": 0},
    },
    "required": [
        "refund_id", "customer_id", "amount", "reason",
        "status", "created_at", "remaining_refundable_balance",
    ],
    "additionalProperties": False,
}

"""In-memory stand-in for the billing system the real server would call.

Kept deliberately small: the point of the exercise is the protocol edge, not
the persistence layer. It does model the two business failures a refund tool
has to express - unknown customer, and a refund larger than what was paid -
because those are *tool execution* errors, not protocol errors, and the server
has to report them differently.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


class CustomerNotFoundError(LookupError):
    """No customer with that id. A valid request about a non-existent entity."""


def _money(amount: float) -> str:
    """Format an amount for a human-readable message, without a 309-digit tail."""
    return f"{amount:.2f}" if abs(amount) < 1e15 else f"{amount:.6g}"


class RefundTooSmallError(Exception):
    """The amount is positive but rounds to nothing at cent precision."""


class RefundExceedsBalanceError(ValueError):
    """The refund is larger than the customer's refundable balance."""



@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    plan: str
    status: str
    refundable_balance: float
    refunds: list[dict] = field(default_factory=list)


_SEED = [
    Customer("CUST-00042", "Ada Lovelace", "ada@example.com", "enterprise", "active", 500.00),
    Customer("CUST-01337", "Grace Hopper", "grace@example.com", "pro", "active", 129.99),
    Customer("CUST-00007", "Alan Turing", "alan@example.com", "free", "suspended", 0.00),
]


class CustomerStore:
    """Thread-safe because a future HTTP transport would serve concurrently."""

    def __init__(self, customers: list[Customer] | None = None) -> None:
        self._lock = threading.Lock()
        self._customers = {c.customer_id: c for c in (customers if customers is not None else _SEED)}

    def get(self, customer_id: str) -> dict:
        with self._lock:
            customer = self._customers.get(customer_id)
            if customer is None:
                raise CustomerNotFoundError(customer_id)
            return {
                "customer_id": customer.customer_id,
                "name": customer.name,
                "email": customer.email,
                "plan": customer.plan,
                "status": customer.status,
                "refundable_balance": round(customer.refundable_balance, 2),
                "refund_count": len(customer.refunds),
            }

    def refund(self, customer_id: str, amount: float, reason: str) -> dict:
        with self._lock:
            customer = self._customers.get(customer_id)
            if customer is None:
                raise CustomerNotFoundError(customer_id)
            # Compare the floats BEFORE converting to cents. ``amount * 100``
            # overflows to infinity for any amount above ~1.8e306, and
            # ``round(inf)`` raises OverflowError - so a large but perfectly
            # finite amount escaped the schema as an unhandled exception and
            # surfaced as -32603 rather than as the refusal it is. The float
            # comparison cannot overflow; the cents comparison that follows is
            # what stops 0.1 + 0.2 drift approving a fraction of a cent over.
            if amount > customer.refundable_balance or round(amount * 100) > round(
                customer.refundable_balance * 100
            ):
                # ``:.2f`` on 1e308 writes out all 309 digits, so a rejected
                # amount produced a 437-character message. The input echo is
                # truncated everywhere else; this path was the exception.
                raise RefundExceedsBalanceError(
                    f"refund of {_money(amount)} exceeds refundable balance of "
                    f"{_money(customer.refundable_balance)}"
                )
            # A positive amount that rounds to zero cents moves no money. It
            # was previously answered with a receipt saying "accepted" and
            # "amount": 0.0, which is a lie to whatever reads it - the balance
            # was untouched and the refund count still went up.
            # ONE quantisation, and both published numbers derive from it.
            #
            # The receipt used to round the amount while the balance rounded
            # the difference, independently - so a refund of 19.185 returned a
            # receipt saying 19.18 while 19.19 actually left the balance. 1,564
            # amounts in a 200,000 sweep disagreed, in both directions. On the
            # one tool that moves money, the receipt handed back has to be what
            # the ledger did.
            charged_cents = round(amount * 100)
            if charged_cents == 0:
                # A positive amount that rounds to nothing moves no money. It
                # was previously answered with a receipt saying "accepted" and
                # "amount": 0.0, which is a lie to whatever reads it.
                raise RefundTooSmallError(
                    f"refund of {amount!r} rounds to zero cents; nothing would be refunded"
                )
            balance_cents = round(customer.refundable_balance * 100) - charged_cents
            charged = charged_cents / 100
            # ``+ 0.0`` normalises -0.0, which otherwise renders as "-0".
            customer.refundable_balance = balance_cents / 100 + 0.0
            receipt = {
                "refund_id": f"RFND-{uuid.uuid4().hex[:12].upper()}",
                "customer_id": customer_id,
                "amount": charged,
                "reason": reason,
                "status": "accepted",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "remaining_refundable_balance": customer.refundable_balance,
            }
            customer.refunds.append(receipt)
            return receipt

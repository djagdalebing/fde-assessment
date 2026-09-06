"""The authorization rule the brief specifies.

    "If params.name starts with admin_ (e.g. admin_reset_key), verify that the
     token role is admin."

One rule, expressed as a table rather than an ``if``, because the moment a
second rule exists an inline check stops being reviewable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolRule:
    """A tool-name pattern and the role required to call anything matching it."""

    pattern: re.Pattern[str]
    required_role: str
    description: str

    def matches(self, tool_name: str) -> bool:
        return self.pattern.match(tool_name) is not None


DEFAULT_TOOL_RULES: tuple[ToolRule, ...] = (
    ToolRule(
        pattern=re.compile(r"^admin_"),
        required_role="admin",
        description="Tools prefixed admin_ perform privileged operations.",
    ),
)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""
    required_role: str = ""


class Policy:
    def __init__(self, tool_rules: tuple[ToolRule, ...] = DEFAULT_TOOL_RULES) -> None:
        self._rules = tool_rules

    def check_tool_call(self, tool_name: str, role: str) -> Decision:
        for rule in self._rules:
            if not rule.matches(tool_name):
                continue
            if role == rule.required_role:
                return Decision(allowed=True)
            return Decision(allowed=False, reason=rule.description, required_role=rule.required_role)
        return Decision(allowed=True)

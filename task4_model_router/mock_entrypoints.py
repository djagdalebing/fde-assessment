"""Importable ASGI apps for the two mock providers, for uvicorn and run.sh."""

from __future__ import annotations

import os

from task4_model_router.providers import MockBehaviour, make_mock_provider

PRIMARY_BEHAVIOUR = MockBehaviour(
    name="primary",
    status=int(os.environ.get("PRIMARY_MOCK_STATUS", os.environ.get("MOCK_STATUS", "200"))),
    delay_seconds=float(os.environ.get("PRIMARY_MOCK_DELAY", os.environ.get("MOCK_DELAY", "0"))),
)
SECONDARY_BEHAVIOUR = MockBehaviour(
    name="secondary",
    status=int(os.environ.get("SECONDARY_MOCK_STATUS", "200")),
    delay_seconds=float(os.environ.get("SECONDARY_MOCK_DELAY", "0")),
)

primary = make_mock_provider(PRIMARY_BEHAVIOUR)
secondary = make_mock_provider(SECONDARY_BEHAVIOUR)

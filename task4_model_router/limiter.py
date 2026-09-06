"""Token-aware sliding-window rate limiter backed by on-disk SQLite.

Why a true sliding window
-------------------------
A fixed one-minute bucket lets a tenant spend the full 50k at 11:59:59 and the
full 50k again at 12:00:00 - 100k tokens in one second, which is exactly the
burst the limit exists to prevent. This keeps individual usage events with
timestamps and sums the trailing 60 seconds, so the limit holds across every
window, not just the aligned ones.

Why reserve-then-reconcile
--------------------------
The token cost of a completion is not known until it finishes, but the limit
has to be enforced *before* the call. So each request reserves an estimate
(prompt tokens + the requested completion ceiling), and reconciles to the
provider's reported usage afterwards. A request that is refused or fails
releases its reservation entirely.

Without this, N concurrent requests all read the same "current usage", all see
room, and all proceed - the classic check-then-act race. Reserving inside the
same transaction as the check is what closes it.

Why SQLite on disk
------------------
State survives a restart. An in-memory counter means a crash-looping gateway
grants every tenant a fresh 50k on each restart, which is unbounded spend
against a provider bill. WAL mode plus ``BEGIN IMMEDIATE`` gives the
serialisation the check-and-reserve needs; the blocking driver is kept off the
event loop on the limiter's own thread pool, so it cannot queue behind
unrelated blocking work in the shared default executor.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import os
import random
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("model-router.limiter")

WINDOW_SECONDS = 60
DEFAULT_LIMIT_TOKENS_PER_MINUTE = 50_000
#: Absolute ceiling on what one response may be charged, as a multiple of the
#: tenant's own limit.
#:
#: A flat 10,000,000 was 200x a 50,000/minute budget, so a single malformed
#: ``usage`` block wrote a number the budget cannot even express and reported
#: it back to the client as ``tokens_charged``. Scaling to the limit keeps the
#: "charge real spend, loudly" intent - the tenant is still refused for the
#: rest of the window - without letting one broken response invent an
#: arbitrary figure. Also keeps the value far below 2**63, where the sqlite
#: driver raises OverflowError and turns a paid-for completion into a 500.
MAX_CHARGE_MULTIPLE = 2
#: Rows evicted per reserve. Bounded so clearing a large backlog cannot hold the
#: global write lock; the remainder goes on the next call.
_EVICTION_BATCH = 1_000
#: Retries for the one-off WAL conversion when several workers race to create
#: the same fresh database.
_WAL_ATTEMPTS = 8

#: One statement per entry rather than one script.
#:
#: ``executescript`` issues an implicit COMMIT before it runs, which does not
#: honour ``busy_timeout`` the way an ordinary statement does - so workers
#: racing to create the same fresh database contended here and died at boot.
#: Splitting a commented script on ";" is not an option either: a semicolon
#: inside a SQL comment silently cuts a statement in half.
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS token_usage (
        id           TEXT PRIMARY KEY,
        tenant_key   TEXT NOT NULL,
        created_ms   INTEGER NOT NULL,
        tokens       INTEGER NOT NULL,
        settled      INTEGER NOT NULL DEFAULT 0
    )
    """,
    # The hot query is "sum tokens for this tenant since T". Leading with
    # tenant_key and then created_ms makes that an index range scan, and
    # carrying `tokens` keeps it covering - off the table entirely.
    "CREATE INDEX IF NOT EXISTS idx_usage_tenant_time ON token_usage (tenant_key, created_ms, tokens)",
    # Eviction sweeps by time across all tenants, so it needs its own index.
    "CREATE INDEX IF NOT EXISTS idx_usage_time ON token_usage (created_ms)",
)


@dataclass(frozen=True)
class Reservation:
    """A held claim on a tenant's budget. Settle or release it - never neither."""

    id: str
    tenant_key: str
    tokens: int


@dataclass(frozen=True)
class LimitDecision:
    allowed: bool
    reservation: Reservation | None
    used_in_window: int
    limit: int
    requested: int
    retry_after_seconds: float = 0.0


class TokenRateLimiter:
    def __init__(
        self,
        db_path: str | os.PathLike[str],
        default_limit: int = DEFAULT_LIMIT_TOKENS_PER_MINUTE,
        window_seconds: int = WINDOW_SECONDS,
    ) -> None:
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._default_limit = default_limit
        # A dedicated pool, not ``asyncio.to_thread``'s default executor.
        # That one is shared with every other blocking call in the process -
        # JSON parsing included - so limiter work queued behind unrelated work
        # and the queue wait landed inside the per-attempt deadline, which can
        # manufacture a "timeout" and attribute it to a healthy provider.
        # Sized for SQLite: writes serialise on one lock, so more threads only
        # deepen the queue.
        self._executor = ThreadPoolExecutor(
            max_workers=int(os.environ.get("ROUTER_LIMITER_THREADS", "8")),
            thread_name_prefix="limiter",
        )
        atexit.register(self._executor.shutdown, wait=False)
        self._window_ms = window_seconds * 1000
        # sqlite3 connections are not safe to share across threads, and
        # calls run on a thread pool, so each takes its own connection.
        # The OS page cache makes this cheap and it sidesteps a whole class of
        # cross-thread state bug.
        self._init_db()

    # -- plumbing ----------------------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=5.0, isolation_level=None)
        # WAL lets readers run while a writer holds the lock, which is what
        # keeps concurrent requests from serialising on the limiter.
        # busy_timeout FIRST. SQLite does not invoke the busy handler for a
        # journal-mode conversion, so several workers starting against a fresh
        # database raced and some died with "database is locked" at boot.
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _init_db(self) -> None:
        """Create the schema, tolerating other workers doing the same thing.

        Every statement is ``IF NOT EXISTS``, so retrying is safe and losing
        the race to another worker is harmless.
        """
        for attempt in range(_WAL_ATTEMPTS):
            connection = self._connect()
            try:
                for statement in _SCHEMA_STATEMENTS:
                    connection.execute(statement)
                return
            except sqlite3.OperationalError:
                if attempt == _WAL_ATTEMPTS - 1:
                    raise
                time.sleep(0.05 * (attempt + 1) + random.random() * 0.05)
            finally:
                # ``with sqlite3.connect(...)`` commits but does not close.
                connection.close()

    def _now_ms(self) -> int:
        """The limiter's clock, in wall-clock milliseconds.

        Wall time rather than a monotonic counter because rows written by
        different worker processes have to be comparable, and because the
        window has to survive a restart - both of which a per-process
        monotonic origin breaks.
        """
        return int(time.time() * 1000)

    # -- synchronous core --------------------------------------------------- #
    def _try_reserve_sync(self, tenant_key: str, tokens: int) -> LimitDecision:
        now = self._now_ms()
        cutoff = now - self._window_ms
        connection = self._connect()
        try:
            # IMMEDIATE takes the write lock up front, so the SUM below cannot
            # be invalidated by another request between reading and inserting.
            # A deferred transaction would upgrade at the INSERT and could be
            # rolled back as a busy/deadlock, which is the race in disguise.
            # Evict BEFORE the transaction, so it commits on its own. Run
            # inside it, the eviction was rolled back along with a refusal - so
            # a tenant sitting at its limit, which is exactly when the table is
            # largest, evicted nothing until it dropped back under. Bounded per
            # call so clearing a backlog cannot hold the write lock for long.
            connection.execute(
                "DELETE FROM token_usage WHERE id IN ("
                "  SELECT id FROM token_usage WHERE created_ms < ? LIMIT ?)",
                (cutoff, _EVICTION_BATCH),
            )

            connection.execute("BEGIN IMMEDIATE")
            try:
                limit = self._default_limit
                used = int(
                    connection.execute(
                        "SELECT COALESCE(SUM(tokens), 0) FROM token_usage "
                        "WHERE tenant_key = ? AND created_ms >= ?",
                        (tenant_key, cutoff),
                    ).fetchone()[0]
                )

                if used + tokens > limit:
                    # Release the write lock BEFORE working out the retry hint.
                    # Computing it inside the transaction made the *denial* path
                    # ~10x more expensive than the allow path at 50k rows, while
                    # holding the global write lock - so one tenant hitting its
                    # limit took unrelated tenants from 7ms to 7.2s and produced
                    # 500s from the limiter itself. The failure mode was
                    # inverted: it degraded precisely when it started working.
                    connection.execute("ROLLBACK")
                    retry_after = self._retry_after_sync(connection, tenant_key, cutoff, used, tokens, limit)
                    return LimitDecision(False, None, used, limit, tokens, retry_after)

                reservation_id = uuid.uuid4().hex
                connection.execute(
                    "INSERT INTO token_usage (id, tenant_key, created_ms, tokens, settled) "
                    "VALUES (?, ?, ?, ?, 0)",
                    (reservation_id, tenant_key, now, tokens),
                )
                connection.execute("COMMIT")
                return LimitDecision(
                    True, Reservation(reservation_id, tenant_key, tokens), used + tokens, limit, tokens
                )
            except Exception:
                with contextlib.suppress(sqlite3.Error):
                    connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()

    def _retry_after_sync(
        self,
        connection: sqlite3.Connection,
        tenant_key: str,
        cutoff: int,
        used: int,
        tokens: int,
        limit: int,
    ) -> float:
        """When will enough tokens age out for this request to fit?

        One windowed query rather than pulling the whole window into Python:
        SQLite walks the covering index and stops at the first row whose running
        total covers the deficit. A flat "try again in 60s" is correct but
        needlessly punishing - it tells a client to wait a minute for headroom
        that arrives in two seconds.

        Runs on a rolled-back connection, outside the write transaction.
        """
        deficit = used + tokens - limit
        if deficit <= 0:
            return 0.0
        if tokens > limit:
            # Larger than the entire budget; no amount of waiting will help.
            return float(self._window_ms) / 1000.0
        try:
            row = connection.execute(
                "SELECT created_ms FROM ("
                "  SELECT created_ms, SUM(tokens) OVER (ORDER BY created_ms) AS running"
                "  FROM token_usage WHERE tenant_key = ? AND created_ms >= ?"
                ") WHERE running >= ? LIMIT 1",
                (tenant_key, cutoff, deficit),
            ).fetchone()
        except sqlite3.Error:  # pragma: no cover - SQLite without window functions
            return float(self._window_ms) / 1000.0
        if row is None:
            return float(self._window_ms) / 1000.0
        return max(0.0, (int(row[0]) + self._window_ms - self._now_ms()) / 1000.0)

    def _settle_sync(self, reservation: Reservation, actual_tokens: int) -> int:
        # Clamped for two reasons. A value at or beyond 2**63 raised
        # OverflowError out of the driver, turning a completion the tenant had
        # already paid for into a 500 and charging them nothing. And a provider
        # reporting far more than the tenant's whole budget is broken or
        # hostile; the charge still applies, but bounded to something the
        # budget can express, and the discrepancy is logged.
        ceiling = max(self._default_limit * MAX_CHARGE_MULTIPLE, reservation.tokens)
        reported = max(0, min(int(actual_tokens), ceiling))
        # Charge what was actually spent. Capping the charge at the reservation
        # looked like the safe direction and was the opposite: it enforced the
        # limit against the *estimate* instead of against spend, so an ordinary
        # request whose completion ran past a low estimate bought far more than
        # the ledger admitted - measured at 7.9x on a request that simply omitted
        # `max_tokens`, and invisible - the ledger still read 49,920/50,000.
        #
        # A loud overspend is recoverable; a silent one is not. The defence
        # against overspend belongs in the reservation being a genuine ceiling
        # (see `estimate_request_tokens`), not in falsifying the ledger.
        charged = reported
        if reported > reservation.tokens:
            logger.warning(
                "provider reported %d tokens against a %d reservation for tenant %s; "
                "charging actual spend - the estimate was too low",
                reported, reservation.tokens, reservation.tenant_key,
            )
        connection = self._connect()
        try:
            cursor = connection.execute(
                "UPDATE token_usage SET tokens = ?, settled = 1 WHERE id = ?",
                (charged, reservation.id),
            )
            if not cursor.rowcount:
                # The reservation row was evicted before the response landed -
                # reachable whenever a request outlives the window. The UPDATE
                # matched nothing, so the charge silently vanished while the
                # client was still told what it had been charged: a completion
                # that really happened, absent from the ledger. Re-insert it,
                # stamped now, so the spend is recorded in the window it
                # actually settled in.
                logger.warning(
                    "reservation %s for %s was evicted before settling; re-recording %d tokens",
                    reservation.id, reservation.tenant_key, charged,
                )
                connection.execute(
                    "INSERT OR REPLACE INTO token_usage "
                    "(id, tenant_key, created_ms, tokens, settled) VALUES (?, ?, ?, ?, 1)",
                    (reservation.id, reservation.tenant_key, self._now_ms(), charged),
                )
        finally:
            connection.close()
        return charged

    def _release_sync(self, reservation: Reservation) -> None:
        connection = self._connect()
        try:
            connection.execute("DELETE FROM token_usage WHERE id = ?", (reservation.id,))
        finally:
            connection.close()

    def _usage_sync(self, tenant_key: str) -> int:
        cutoff = self._now_ms() - self._window_ms
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant_key = ? AND created_ms >= ?",
                (tenant_key, cutoff),
            ).fetchone()
            return int(row[0])
        finally:
            connection.close()

    def _row_count_sync(self) -> int:
        connection = self._connect()
        try:
            return int(connection.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0])
        finally:
            connection.close()

    # -- async surface ------------------------------------------------------ #
    # sqlite3 blocks. Running it inline would stall every other request on the
    # event loop for the duration of the write, which under load is the whole
    # ballgame.
    async def _run(self, function, *args):
        """Run a blocking limiter call on the limiter's own executor."""
        return await asyncio.get_running_loop().run_in_executor(self._executor, function, *args)

    async def try_reserve(self, tenant_key: str, tokens: int) -> LimitDecision:
        return await self._run(self._try_reserve_sync, tenant_key, tokens)

    async def settle(self, reservation: Reservation, actual_tokens: int) -> int:
        """Reconcile the hold and return what was *actually* charged.

        The caller reports this to the client, so it must be the ledger figure
        rather than whatever the provider claimed - otherwise a response could
        tell a tenant it spent 10,000,000 tokens while 15 were recorded.
        """
        return await self._run(self._settle_sync, reservation, actual_tokens)

    async def release(self, reservation: Reservation) -> None:
        await self._run(self._release_sync, reservation)

    async def usage(self, tenant_key: str) -> int:
        return await self._run(self._usage_sync, tenant_key)

    async def row_count(self) -> int:
        return await self._run(self._row_count_sync)

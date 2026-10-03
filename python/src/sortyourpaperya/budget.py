"""A ceiling on what the tool may spend at the API, and a record of what it has.

The watcher is installed as a launchd agent with `KeepAlive`, which means the
one thing that reliably restarts it is failure. A pass that dies partway through
comes back, re-reads the same folder, and pays for the same documents again —
and nothing anywhere else in the pipeline counts, so that loop runs until
somebody notices it on a bill.

So every model request passes through a ledger first. It records requests and
tokens in hourly buckets, keeps a rolling day of them, and refuses a request
that would start beyond the ceiling. Refusing *before* the call is the point:
a limit checked afterwards has already been paid.

The ledger is machine-wide rather than per-library, because the cost is
machine-wide: two libraries on one API key spend the same money. It is shared
between processes, so writes take a lock and land whole — a counter that loses
increments under contention would drift the wrong way, undercounting exactly
when the most is being spent.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DEFAULT_MAX_REQUESTS_PER_DAY,
    DEFAULT_MAX_TOKENS_PER_DAY,
    env_int,
    state_dir,
)

WINDOW_SECONDS = 24 * 60 * 60
BUCKET_SECONDS = 60 * 60


class BudgetExceeded(RuntimeError):
    """Raised when a request would go past the day's ceiling."""


@dataclass(frozen=True)
class Limits:
    """What a rolling day is allowed to cost. Zero or less means no ceiling."""

    requests_per_day: int = DEFAULT_MAX_REQUESTS_PER_DAY
    tokens_per_day: int = DEFAULT_MAX_TOKENS_PER_DAY

    @property
    def unlimited(self) -> bool:
        return self.requests_per_day <= 0 and self.tokens_per_day <= 0


@dataclass(frozen=True)
class Usage:
    """What the rolling day has cost so far."""

    requests: int = 0
    tokens: int = 0


def resolve_limits() -> Limits:
    """Read the ceilings from `SYP_MAX_REQUESTS_PER_DAY` / `SYP_MAX_TOKENS_PER_DAY`.

    Set either to `0` to lift that ceiling. There is deliberately no way to
    spell "unlimited" other than saying so explicitly for each.
    """
    return Limits(
        requests_per_day=env_int(
            "SYP_MAX_REQUESTS_PER_DAY", DEFAULT_MAX_REQUESTS_PER_DAY
        ),
        tokens_per_day=env_int("SYP_MAX_TOKENS_PER_DAY", DEFAULT_MAX_TOKENS_PER_DAY),
    )


def ledger_path() -> Path:
    """Where the spend record lives: with the rest of this machine's state."""
    return state_dir() / "spend.json"


class Budget:
    """The rolling-day ledger, and the ceilings it is checked against."""

    def __init__(
        self,
        path: Path | None = None,
        limits: Limits | None = None,
        *,
        token_setting: str = "SYP_MAX_TOKENS_PER_DAY",
    ) -> None:
        self.path = path or ledger_path()
        self.limits = limits if limits is not None else resolve_limits()
        # The variable that sets this ledger's token ceiling, so the refusal
        # names the one that would lift it -- the embedding ledger has its own.
        self.token_setting = token_setting

    def check(self, what: str = "request") -> None:
        """Refuse a request that would start past the ceiling.

        Raises:
            BudgetExceeded: naming which ceiling, what it is, and when the
                window frees up — an error a person can act on without having
                to go and read the ledger.
        """
        if self.limits.unlimited:
            return
        used = self.usage()
        if 0 < self.limits.requests_per_day <= used.requests:
            raise BudgetExceeded(
                f"the last 24 hours have used {used.requests} of "
                f"{self.limits.requests_per_day} allowed API requests, so this "
                f"{what} was not sent. Raise SYP_MAX_REQUESTS_PER_DAY, or wait "
                "for the window to move."
            )
        if 0 < self.limits.tokens_per_day <= used.tokens:
            raise BudgetExceeded(
                f"the last 24 hours have used {used.tokens} of "
                f"{self.limits.tokens_per_day} allowed tokens, so this {what} "
                f"was not sent. Raise {self.token_setting}, or wait for the "
                "window to move."
            )

    def record(self, *, requests: int = 1, tokens: int = 0) -> None:
        """Add one request's cost to the ledger.

        Never raises: a ledger that cannot be written is worth a wrong count,
        not a failed ingest. The next `check` then reads low, which is why the
        write is the small, locked, whole-file kind rather than a read followed
        by a hopeful write.
        """
        try:
            self._update(requests, tokens)
        except OSError:
            pass

    def usage(self) -> Usage:
        """What the rolling day has cost, as of now."""
        buckets = _prune(_read(self.path), time.time())
        return Usage(
            requests=sum(int(entry.get("requests", 0)) for entry in buckets.values()),
            tokens=sum(int(entry.get("tokens", 0)) for entry in buckets.values()),
        )

    def reset(self) -> None:
        """Forget the recorded spend. For starting a window over deliberately."""
        self.path.unlink(missing_ok=True)

    def _update(self, requests: int, tokens: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusively(self.path):
            now = time.time()
            buckets = _prune(_read(self.path), now)
            key = str(int(now // BUCKET_SECONDS) * BUCKET_SECONDS)
            entry = buckets.setdefault(key, {"requests": 0, "tokens": 0})
            entry["requests"] = int(entry.get("requests", 0)) + requests
            entry["tokens"] = int(entry.get("tokens", 0)) + tokens
            _write(self.path, buckets)


class _exclusively:
    """Hold a lock beside the ledger for the length of a read-modify-write.

    The lock is a separate file that is never replaced, because the ledger
    itself is written by rename: a lock taken on a file that is about to be
    replaced guards an inode nobody will look at again.
    """

    def __init__(self, path: Path) -> None:
        self._lock_path = path.with_suffix(path.suffix + ".lock")
        self._fd = -1

    def __enter__(self) -> None:
        self._fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(self._fd, fcntl.LOCK_EX)

    def __exit__(self, *_exc: object) -> None:
        os.close(self._fd)  # drops the flock


def _read(path: Path) -> dict[str, dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # No ledger yet, or one written by a process that died mid-write. Either
        # way there is nothing to count; a corrupt file is replaced by the next
        # write rather than stopping the pass.
        return {}
    buckets = payload.get("buckets")
    return buckets if isinstance(buckets, dict) else {}


def _write(path: Path, buckets: dict[str, dict]) -> None:
    staged = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    staged.write_text(
        json.dumps({"version": 1, "buckets": buckets}), encoding="utf-8"
    )
    os.replace(staged, path)  # whole or not at all


def _prune(buckets: dict[str, dict], now: float) -> dict[str, dict]:
    """Drop buckets that have fallen out of the rolling window."""
    cutoff = now - WINDOW_SECONDS
    kept: dict[str, dict] = {}
    for key, entry in buckets.items():
        try:
            started = float(key)
        except (TypeError, ValueError):
            continue
        if started + BUCKET_SECONDS > cutoff and isinstance(entry, dict):
            kept[key] = entry
    return kept

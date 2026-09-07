"""Asking the watcher to act, when there is one.

Every call answers one question first: is a watcher serving this library right
now? A socket file is not an answer -- a killed watcher leaves one behind -- so
`serving()` connects rather than calling `exists()`.

Nothing here starts a watcher. A read command that silently launched a
background process which spends money would be the opposite of the point.
"""

from __future__ import annotations

import asyncio
import contextlib

import duckdb
import json
import socket
from pathlib import Path
from typing import Any, Callable

from . import protocol

# A live watcher answers a local socket immediately; this only has to be long
# enough to cross a loaded machine, not to wait out an ingest pass.
CONNECT_TIMEOUT_SECONDS = 2.0

# Ops are as slow as what they do -- an ingest pass calls a model repeatedly --
# so the reply is not on a clock. A watcher that dies mid-op closes the socket,
# which is what actually ends the wait.
REPLY_TIMEOUT_SECONDS: float | None = None


class NoWatcher(RuntimeError):
    """No watcher is serving this library."""


def serving(library_root: Path) -> bool:
    """Whether a watcher answers for this library.

    Connects rather than trusting the path: a `kill -9` leaves the socket file
    behind, and treating that as "serving" would hang every later command.
    """
    path = protocol.socket_path(library_root)
    if not path.exists():
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(CONNECT_TIMEOUT_SECONDS)
        sock.connect(str(path))
        return True
    except (OSError, socket.timeout):
        return False
    finally:
        sock.close()


def call(
    library_root: Path,
    op: str,
    args: dict[str, Any] | None = None,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """Run one op on the watcher and return its result.

    `on_event` receives each line the daemon sends before the result, so a long
    op prints as it happens rather than in a lump at the end. Raises `NoWatcher`
    when nothing is serving; a daemon-side failure is re-raised as its own type.
    """
    return asyncio.run(call_async(library_root, op, args, on_event=on_event))


async def call_async(
    library_root: Path,
    op: str,
    args: dict[str, Any] | None = None,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    path = protocol.socket_path(library_root)
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(path), limit=protocol.MAX_FRAME_BYTES),
            CONNECT_TIMEOUT_SECONDS,
        )
    except (OSError, asyncio.TimeoutError) as err:
        raise NoWatcher(f"no watcher is serving {library_root}") from err

    try:
        writer.write(protocol.encode(protocol.request(op, args)))
        await writer.drain()

        while True:
            line = await _read_line(reader)
            if not line:
                raise NoWatcher(
                    f"the watcher serving {library_root} closed the connection "
                    "without answering; it may have stopped mid-request"
                )
            frame = protocol.decode(line)
            if "event" in frame:
                if on_event is not None:
                    on_event(frame)
                continue
            if frame.get("ok"):
                # Decoded here rather than by each caller: `_ingest_via_watcher`
                # forgot, and got the raw marker dict where it expected a report.
                # `load` is idempotent, so the proxies decoding again is harmless.
                return protocol.load(frame.get("result"))
            protocol.raise_for(
                frame.get("kind", "RuntimeError"),
                frame.get("message", "the watcher reported a failure"),
            )
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # pragma: no cover - shutdown races
            pass


async def _read_line(reader: asyncio.StreamReader) -> bytes:
    if REPLY_TIMEOUT_SECONDS is None:
        return await reader.readline()
    return await asyncio.wait_for(reader.readline(), REPLY_TIMEOUT_SECONDS)


class RemoteDb:
    """A `PaperDb` that answers from the watcher instead of the file.

    Stands in for the real one when a pass holds the lock, so the commands
    above it -- which only ever call read methods -- carry on unchanged rather
    than each learning that the library is busy.
    """

    def __init__(self, library_root: Path) -> None:
        self._root = library_root

    def __getattr__(self, method: str):
        from .daemon import DB_WRITE_METHODS

        # Which op depends on what the method does, not on which proxy holds it:
        # `attr` reads and writes through the same handle, so a single `PaperDb`
        # stand-in has to reach both sides.
        op = "write" if method in DB_WRITE_METHODS else "db"

        def invoke(*args, **kwargs):
            return protocol.load(
                call(
                    self._root,
                    op,
                    {
                        "method": method,
                        "args": protocol.dump(list(args)),
                        "kwargs": protocol.dump(kwargs),
                    },
                )
            )

        return invoke

    def release(self) -> None:
        """Nothing is held here; the watcher owns the connection."""

    def close(self) -> None:
        """As `release`: there is no local connection to close."""


class RemoteLibrary:
    """A `Library` whose changes are made by the watcher holding the lock.

    Paths are arithmetic and stay local; anything that touches the database or
    the store goes over the socket. The commands above it are unchanged -- they
    call the same method names on what looks like the same object.
    """

    def __init__(self, library_root: Path) -> None:
        from .library import Library

        self._local = Library(library_root, read_only=True)
        self._root = library_root
        self.root = library_root
        self.store_dir = self._local.store_dir
        self.tree_dir = self._local.tree_dir
        self.db = RemoteDb(library_root)

    def __getattr__(self, method: str):
        from .daemon import WRITE_METHODS

        if method not in WRITE_METHODS:
            # Path arithmetic and the like: no lock involved, so no round trip.
            return getattr(self._local, method)

        def invoke(*args, **kwargs):
            return protocol.load(
                call(
                    self._root,
                    "write",
                    {
                        "method": method,
                        "args": protocol.dump(list(args)),
                        "kwargs": protocol.dump(kwargs),
                    },
                )
            )

        return invoke

    def release(self) -> None:
        self._local.release()

    def close(self) -> None:
        self._local.close()

    def __enter__(self) -> "RemoteLibrary":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@contextlib.contextmanager
def writing(library_root: Path):
    """A library to change, whoever is holding it.

    Yields a `RemoteLibrary` when the watcher is serving -- it owns the write
    connection, so it does the write -- and `None` otherwise, leaving the caller
    to take the lock itself the way it always has.
    """
    if not serving(library_root):
        yield None
        return
    remote = RemoteLibrary(library_root)
    try:
        yield remote
    finally:
        remote.close()


@contextlib.contextmanager
def reading(library_root: Path):
    """A read-only handle on the library, or `None` when a writer holds it.

    Reads do not go through the watcher. Several read-only connections coexist
    happily, so a read command works with no watcher running at all -- which is
    the point: the library stays readable when nothing is serving it, and an
    agent asking what it holds never depends on a background process.

    The exception is a writer mid-pass, which excludes readers entirely. Rather
    than wait out an ingest, this yields `None` and the caller asks the watcher
    with `call()` -- it is holding the lock, so it is exactly who can answer.
    """
    from .db import Locked
    from .library import DB_FILE, Library

    if not (library_root / DB_FILE).exists():
        # A library nobody has written yet. Read-only cannot create it, so this
        # is the caller's ordinary read-write path -- `sypy list` on a fresh
        # library answered "empty" before this change and must still do so.
        yield None
        return

    library = Library(library_root, read_only=True, fail_on_lock=True)
    try:
        # Touched here so a held lock surfaces now, rather than from inside
        # whatever the caller was part-way through doing with the handle.
        library.db.count()
    except duckdb.Error:
        # A schema older than this build: migrations are DDL, which a read-only
        # connection cannot run. Hand it back to the read-write path, which
        # migrates on open. Otherwise the first read after an upgrade is a raw
        # Catalog Error and the library stays unmigrated until something writes.
        library.close()
        yield None
        return
    except Locked:
        library.close()
        if not serving(library_root):
            # Nothing to ask. The caller decides whether to wait or give up.
            yield None
            return
        # The watcher holds the lock, so it is exactly who can answer. Paths
        # still resolve locally; only the database questions cross the socket.
        remote = Library(library_root, read_only=True)
        remote.db = RemoteDb(library_root)
        yield remote
        return
    try:
        yield library
    finally:
        library.close()

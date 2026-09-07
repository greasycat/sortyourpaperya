"""The watcher, answering for the library it already owns.

`sypy watch` holds the write connection and -- once the key is confined to it --
the only credential that can spend. Other commands ask it to act rather than
opening the database behind its back.

It is not a separate process. `watchlock` already guarantees one watcher per
library, and that invariant is what makes "the daemon" a well-defined thing to
address: the socket is served by whoever holds the claim, or by nobody.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import protocol
from .library import Library

log = logging.getLogger(__name__)

# An op is `(library, args, emit) -> result`. `emit` sends a line the client
# shows immediately; ops that finish quickly never call it.
Emit = Callable[[dict[str, Any]], Awaitable[None]]
Op = Callable[[Library, dict[str, Any], Emit], Awaitable[Any]]

OPS: dict[str, Op] = {}


def op(name: str) -> Callable[[Op], Op]:
    def register(fn: Op) -> Op:
        OPS[name] = fn
        return fn

    return register


@op("ping")
async def _ping(library: Library, args: dict[str, Any], emit: Emit) -> dict[str, Any]:
    """Prove the socket is served, and by what.

    `pid` is what tells a stale socket from a live one: a path that answers is
    answering from somewhere, and this says where.
    """
    from .config import ConfigError, resolve_api_key

    # Whether the watcher has a key is a question only it can answer: the stored
    # key is its to read, so a client asking the keychain would both prompt and
    # get the wrong answer. Never the key itself -- only that there is one.
    try:
        resolve_api_key()
        has_key = True
    except ConfigError:
        has_key = False
    return {
        "pid": os.getpid(),
        "library": str(library.root),
        "v": protocol.VERSION,
        "has_key": has_key,
    }


# Read-only `PaperDb` methods a client may ask the watcher to run on its behalf.
# A whitelist rather than a blocklist: a method added to `PaperDb` later is not
# reachable over the socket until someone decides it should be, which is the
# safe direction for a list that governs what a client can invoke.
READ_METHODS = frozenset(
    {
        "get",
        "search",
        "count",
        "all_papers",
        "attributes",
        "attributes_for",
        "category_counts",
        "select",
        "model_answers",
        "count_model_answers",
    }
)


@op("db")
async def _db(library: Library, args: dict[str, Any], emit: Emit) -> Any:
    """Run one read-only database method for a client that cannot read directly.

    Reached only when a client found the database locked -- by this watcher, in
    the middle of a pass. It is the one process that can answer just then, which
    is the whole reason this op exists.
    """
    method = args.get("method")
    if method not in READ_METHODS:
        raise protocol.ProtocolError(f"not a readable method: {method!r}")
    if method == "select":
        # The watcher's connection is read-write, so `select` here is running on
        # a handle that can delete. `is_read_only` looks only at the leading
        # keyword, and `WITH x AS (...) DELETE FROM papers` gets past it -- which
        # is harmless on the caller's own read-only connection and is not here.
        from .db import is_read_only

        statement = (args.get("args") or [None])[0] or args.get("kwargs", {}).get("sql")
        if not isinstance(statement, str) or not is_read_only(statement):
            raise protocol.ProtocolError("only a reading statement may be run here")
        if any(
            word in statement.lower()
            for word in (" delete ", " insert ", " update ", " drop ", " alter ", " create ")
        ):
            raise protocol.ProtocolError(
                "a reading statement may not contain a writing one"
            )
    fn = getattr(library.db, method)
    return protocol.dump(
        fn(*protocol.load(args.get("args", [])), **protocol.load(args.get("kwargs", {})))
    )


# What a client may ask the watcher to change on its behalf. Derived from what
# the write commands actually call, not from what `Library` happens to expose:
# a method reachable over the socket is one anybody who can reach the socket can
# invoke, so the list is opened deliberately rather than by default.
WRITE_METHODS = frozenset(
    {
        "retag",
        "remove",
        "rescan",
        "backup",
        "rebuild_tree",
        "migrate_store_layout",
        "refresh_document_names",
        "adopt",
        # Reads that the write commands do alongside their writes, so a command
        # routed here does not have to open the database itself for half of it.
        "orphans",
        "missing_files",
        "tree_litter",
    }
)

DB_WRITE_METHODS = frozenset({"set_attribute", "unset_attribute", "forget_model_answers"})


@op("write")
async def _write(library: Library, args: dict[str, Any], emit: Emit) -> Any:
    """Change the library on a client's behalf, because it cannot take the lock.

    The watcher is holding it. Rather than each command failing or waiting out a
    pass, the one process that can write does the write.
    """
    method = args.get("method")
    if method in WRITE_METHODS:
        target = getattr(library, method)
    elif method in DB_WRITE_METHODS:
        target = getattr(library.db, method)
    else:
        raise protocol.ProtocolError(f"not a writable method: {method!r}")
    result = target(
        *protocol.load(args.get("args", [])), **protocol.load(args.get("kwargs", {}))
    )
    return protocol.dump(result)


@op("ingest")
async def _ingest(library: Library, args: dict[str, Any], emit: Emit) -> Any:
    """File a folder on a client's behalf, spending the watcher's key.

    This is where `sypy login` and `sypy ingest` meet. The stored key belongs to
    this process, so an ingest typed at a prompt is done here rather than there
    -- which also means one process is deciding what is new, and the second
    writer that would otherwise fight for the lock never exists.
    """
    from .config import resolve_api_key, resolve_settings
    from .ingest import ingest_folder
    from .library import FilingMode
    from .llm import OpenAiClient

    settings = resolve_settings(
        Path(args["input_dir"]),
        library.root,
        model=args.get("model"),
    )
    mode = FilingMode(args.get("mode", FilingMode.PREVIEW.value))
    await emit(protocol.event("started", input=str(settings.input_dir), mode=mode.value))
    client = OpenAiClient(resolve_api_key(), settings.model)
    report = await ingest_folder(settings, client, library, mode=mode)
    return protocol.dump(report)


@op("login")
async def _login(library: Library, args: dict[str, Any], emit: Emit) -> Any:
    """Store the key, in the one process that is allowed to read it back.

    Doing this here rather than in the command means the keyring is touched by a
    single long-lived process instead of by every invocation -- a desktop
    keyring confirms per process, so storing from the CLI is a dialog each time.
    The watcher picks the new key up on its next request; nothing restarts.

    The key crosses a socket that is `0600` in the user's own runtime directory,
    to a process already entitled to read it. It is never logged.
    """
    from .config import store_api_key

    key = (args.get("key") or "").strip()
    if not key:
        raise protocol.ProtocolError("no key given")
    store_api_key(key)
    return {"stored": True}


@op("logout")
async def _logout(library: Library, args: dict[str, Any], emit: Emit) -> Any:
    """Forget the stored key, from the process that owns the keyring access."""
    from .config import forget_api_key

    return {"removed": forget_api_key()}


@op("probe")
async def _probe(library: Library, args: dict[str, Any], emit: Emit) -> Any:
    """Ask the API whether the watcher's key is accepted.

    The watcher is the only process that can: the key is its to read. Listing
    models is free, so `doctor` still costs nothing, and it is the difference
    between "a key is set" and "a key that works" -- an expired one looks
    exactly like a working one until the first document.
    """
    from .config import ConfigError, resolve_api_key

    try:
        key = resolve_api_key()
    except ConfigError as err:
        return {"ok": False, "detail": str(err)}
    try:
        from openai import OpenAI

        OpenAI(api_key=key, max_retries=0, timeout=15).models.list()
    except Exception as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"}
    return {"ok": True}


class Server:
    """Serves one library's socket for as long as the watcher runs."""

    def __init__(self, library: Library, path: Path | None = None) -> None:
        self.library = library
        self.path = path or protocol.socket_path(library.root)
        self._server: asyncio.AbstractServer | None = None
        # One op at a time. Each connection is its own task and an ingest awaits
        # the model, so two passes otherwise interleave -- and `ingest_folder`
        # decides what is new long before it writes, so both conclude the same
        # PDF is new and file it twice. That leaves two store folders for one
        # database row, silently. The watch loop's own pass takes this too.
        self.busy = asyncio.Lock()

    async def start(self) -> None:
        """Listen, replacing a socket left behind by a watcher that was killed.

        Taking over an abandoned path is safe here and only here: the caller
        holds the watch claim, so no live watcher can be serving this library.
        Binding without that claim could steal a working socket.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        if self.path.exists():
            log.debug("replacing a socket left at %s", self.path)
            self.path.unlink()
        self._server = await asyncio.start_unix_server(
            self._serve,
            path=str(self.path),
            # Without this asyncio caps a line at 64 KiB, well under the frame
            # size this protocol allows: a `list` of ~90 papers already exceeds
            # it, and the failure is a bare ValueError rather than a refusal.
            limit=protocol.MAX_FRAME_BYTES,
        )
        # Nobody else's business, even in a shared state directory.
        os.chmod(self.path, 0o600)
        log.info("serving %s", self.path)

    async def stop(self) -> None:
        """Stop listening and take the path away, so nothing dials a dead socket."""
        if self._server is None:
            # Never bound, so the path belongs to somebody else. Removing it
            # would take a live watcher's socket away from it.
            return
        self._server.close()
        try:
            await self._server.wait_closed()
        except Exception:  # pragma: no cover - shutdown races
            pass
        self._server = None
        self.path.unlink(missing_ok=True)

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """One connection, one request, one terminal frame.

        A failing op must not take the watcher down with it: the client is told
        what went wrong and the loop keeps watching. That is the whole reason
        the reply is built inside a `try`.
        """
        try:
            try:
                # Inside the try: an unreadable request must still get an answer.
                # Outside it, an oversized line escaped as an unhandled traceback
                # and the client was told the watcher had died, which it had not.
                line = await reader.readline()
                if not line:
                    return
                frame = protocol.decode(line)
                protocol.check_version(frame)
                name = frame.get("op")
                handler = OPS.get(name)
                if handler is None:
                    raise protocol.ProtocolError(f"unknown op: {name!r}")

                async def emit(payload: dict[str, Any]) -> None:
                    writer.write(protocol.encode(payload))
                    await writer.drain()

                async with self.busy:
                    result = await handler(self.library, frame.get("args") or {}, emit)
                reply = protocol.ok(result)
            except Exception as err:
                log.debug("op failed: %s", err)
                reply = protocol.failed(err)
            writer.write(protocol.encode(reply))
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass  # the client went away mid-answer; nothing to do
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # pragma: no cover - shutdown races
                pass

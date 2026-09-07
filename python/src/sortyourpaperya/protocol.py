"""The wire between a `sypy` command and the watcher serving its library.

Newline-delimited JSON over a Unix socket. No HTTP and no dependency: the two
halves are the same program on the same machine, and a frame is one line.

Both halves import this module, so the frame shape, the op names, and the error
mapping cannot drift apart in an upgrade -- which is the failure this file
exists to prevent. When they do differ, `VERSION` says so before anything is
interpreted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .budget import BudgetExceeded
from .config import ConfigError, runtime_dir
from .library import LibraryError

# Bumped when a frame's meaning changes, never for a new op: an older daemon
# answering `unknown op` is a clear message, where a changed shape is not.
VERSION = 1

MAX_FRAME_BYTES = 8 * 1024 * 1024


class ProtocolError(RuntimeError):
    """A frame that could not be understood, or a version that cannot be served."""


# Errors that mean something specific to the caller, so `cli.py` can keep
# catching what it already catches. Anything not here crosses as a plain
# RuntimeError: an unexpected daemon-side failure should not be mistaken for a
# condition the client knows how to handle.
ERROR_KINDS: dict[str, type[Exception]] = {
    "LibraryError": LibraryError,
    "ConfigError": ConfigError,
    "BudgetExceeded": BudgetExceeded,
    "ProtocolError": ProtocolError,
    "NotADirectoryError": NotADirectoryError,
    "FileNotFoundError": FileNotFoundError,
    "ValueError": ValueError,
}


def kind_of(exc: BaseException) -> str:
    """The wire name for an exception, or `RuntimeError` when it has none."""
    name = type(exc).__name__
    return name if name in ERROR_KINDS else "RuntimeError"


def raise_for(kind: str, message: str) -> None:
    """Re-raise a daemon-side failure client-side, as its own type where known."""
    raise ERROR_KINDS.get(kind, RuntimeError)(message)


def socket_path(library_root: Path) -> Path:
    """Where the watcher for `library_root` listens.

    One socket per library rather than per machine, because the registry allows
    several watches and each owns a different database. The library's own path
    names it, so a client and the watcher derive the same socket without
    consulting anything that could disagree.

    Kept short on purpose. A Unix socket path is capped near 104 bytes -- the
    whole path -- so the library path is hashed rather than spelled, and it lives
    in the runtime directory rather than under the home directory where the rest
    of the state goes. A deep library or a deep home would otherwise fail at
    `bind` with nothing but "path too long" to go on.
    """
    import hashlib

    root = library_root.expanduser().resolve()
    ident = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    path = runtime_dir() / f"{ident}.sock"
    if len(str(path)) >= 104:
        raise ProtocolError(
            f"the socket path is too long for this system ({len(str(path))} bytes): "
            f"{path}. Set SORTYOURPAPERYA_RUNTIME_DIR to somewhere shorter."
        )
    return path


def _registry() -> dict[str, type]:
    """The classes a result may contain, by name.

    A registry rather than "rebuild whatever class the frame names": the client
    must not be able to make the daemon -- or itself -- construct an arbitrary
    type because a frame said so.
    """
    from .db import ModelAnswer, Paper
    from .ingest import IngestReport
    from .library import BackupReport, PlannedFiling, RescanReport

    return {
        c.__name__: c
        for c in (
            Paper,
            ModelAnswer,
            RescanReport,
            BackupReport,
            PlannedFiling,
            IngestReport,
        )
    }


def dump(value: Any) -> Any:
    """Turn a library result into something JSON can carry.

    Paths, the report dataclasses, and tuples all survive the trip -- a caller
    that does `.name` on a returned path, or unpacks a tuple, must not have to
    know the answer came over a socket.
    """
    from dataclasses import fields, is_dataclass

    if isinstance(value, Path):
        return {"__path__": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if name in _registry():
            # Field by field rather than `asdict`, which recurses itself and
            # flattens a nested dataclass into a bare dict -- losing the marker
            # that says what to rebuild it as. `IngestReport` always carries a
            # `RescanReport`, so that mistake broke every routed ingest.
            return {
                "__obj__": name,
                "fields": {f.name: dump(getattr(value, f.name)) for f in fields(value)},
            }
        raise ProtocolError(f"{name} cannot cross the wire; add it to the registry")
    if isinstance(value, tuple):
        # JSON has no tuples, and `refresh_document_names` returns a list of
        # them for the caller to unpack.
        return {"__tuple__": [dump(v) for v in value]}
    if isinstance(value, list):
        return [dump(v) for v in value]
    if isinstance(value, dict):
        return {k: dump(v) for k, v in value.items()}
    return value


def load(value: Any) -> Any:
    """Rebuild what `dump` took apart."""
    if isinstance(value, dict):
        if "__path__" in value:
            return Path(value["__path__"])
        if "__tuple__" in value:
            return tuple(load(v) for v in value["__tuple__"])
        if "__obj__" in value:
            cls = _registry().get(value["__obj__"])
            if cls is None:
                raise ProtocolError(f"unknown type on the wire: {value['__obj__']!r}")
            return cls(**load(value["fields"]))
        return {k: load(v) for k, v in value.items()}
    if isinstance(value, list):
        return [load(v) for v in value]
    return value


def encode(frame: dict[str, Any]) -> bytes:
    """One frame, as a line."""
    # No `default=`: a value `dump` did not handle must fail here rather than
    # arrive as its `str()`. A datetime column read over the socket would
    # otherwise come back as text where the same query locally gives a datetime.
    return (json.dumps(frame, separators=(",", ":")) + "\n").encode()


def decode(line: bytes) -> dict[str, Any]:
    """Parse one line into a frame, refusing anything that is not an object."""
    if len(line) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame larger than {MAX_FRAME_BYTES} bytes")
    try:
        frame = json.loads(line)
    except json.JSONDecodeError as err:
        raise ProtocolError(f"not JSON: {err}") from err
    if not isinstance(frame, dict):
        raise ProtocolError(f"expected an object, got {type(frame).__name__}")
    return frame


def request(op: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"v": VERSION, "op": op, "args": args or {}}


def ok(result: Any = None) -> dict[str, Any]:
    return {"ok": True, "result": result}


def failed(exc: BaseException) -> dict[str, Any]:
    return {"ok": False, "kind": kind_of(exc), "message": str(exc)}


def event(name: str, /, **fields: Any) -> dict[str, Any]:
    """A line the client should show now, rather than when the op finishes.

    `name` is positional-only because the fields are the caller's to choose and
    `name` is the obvious one for them to pick -- a filed document has a name.
    Without the `/` that call is a TypeError about duplicate arguments.
    """
    return {"event": name, **fields}


def check_version(frame: dict[str, Any]) -> None:
    """Refuse a frame this build cannot interpret, saying what to do about it."""
    got = frame.get("v")
    if got != VERSION:
        raise ProtocolError(
            f"protocol version {got!r}, but this build speaks {VERSION}. "
            "The running watcher and this command are different versions -- "
            "restart the watcher."
        )

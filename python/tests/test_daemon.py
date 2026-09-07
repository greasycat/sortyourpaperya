"""The watcher answering for its library, and the client asking.

The cases worth pinning are the ones that fail quietly: a socket left behind by
a killed watcher, a failing op taking the watcher down with it, and a reply that
never comes. A test that only proves `ping` works would miss all three.
"""

from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path

import pytest

from sortyourpaperya import client, daemon, protocol
from sortyourpaperya.library import Library, LibraryError


@pytest.fixture(autouse=True)
def _short_runtime_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runtime dir short enough to bind, wherever pytest put tmp_path.

    A socket path is capped near 104 bytes and pytest's tmp paths are long, so
    these cannot use the usual fixture without testing the cap instead of the
    code. `tempfile.mkdtemp` under the system temp root stays well inside it.
    """
    import shutil
    import tempfile

    short = tempfile.mkdtemp(prefix="sypyt-")
    monkeypatch.setenv("SORTYOURPAPERYA_RUNTIME_DIR", short)
    yield
    shutil.rmtree(short, ignore_errors=True)


@pytest.fixture
def library(tmp_path: Path) -> Library:
    return Library(tmp_path / "lib")


async def _serving(library: Library):
    server = daemon.Server(library)
    await server.start()
    return server


@pytest.mark.asyncio
async def test_ping_says_which_process_is_answering(library: Library) -> None:
    server = await _serving(library)
    try:
        answer = await client.call_async(library.root, "ping")
        assert answer["pid"] == os.getpid()
        assert answer["library"] == str(library.root)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_the_socket_is_not_readable_by_anyone_else(library: Library) -> None:
    server = await _serving(library)
    try:
        assert protocol.socket_path(library.root).stat().st_mode & 0o777 == 0o600
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_stopping_takes_the_socket_away(library: Library) -> None:
    # A path that outlives its server has every later command dialling
    # something that cannot answer.
    server = await _serving(library)
    await server.stop()
    assert not protocol.socket_path(library.root).exists()
    assert client.serving(library.root) is False


@pytest.mark.asyncio
async def test_a_socket_left_by_a_killed_watcher_is_not_mistaken_for_a_live_one(
    library: Library,
) -> None:
    """`kill -9` leaves the path behind. Trusting it would hang every command."""
    path = protocol.socket_path(library.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()

    assert path.exists()
    assert client.serving(library.root) is False
    with pytest.raises(client.NoWatcher):
        await client.call_async(library.root, "ping")


@pytest.mark.asyncio
async def test_a_new_watcher_takes_over_an_abandoned_socket(library: Library) -> None:
    path = protocol.socket_path(library.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()

    server = await _serving(library)
    try:
        assert (await client.call_async(library.root, "ping"))["pid"] == os.getpid()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_nothing_serving_is_a_NoWatcher_not_a_hang(library: Library) -> None:
    with pytest.raises(client.NoWatcher):
        await client.call_async(library.root, "ping")


@pytest.mark.asyncio
async def test_an_unknown_op_is_answered_rather_than_ignored(library: Library) -> None:
    server = await _serving(library)
    try:
        with pytest.raises(protocol.ProtocolError, match="unknown op"):
            await client.call_async(library.root, "no-such-op")
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_a_failing_op_keeps_its_type_and_leaves_the_server_up(
    library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The watcher must survive a bad request: it is watching for everyone."""

    async def explode(lib, args, emit):
        raise LibraryError("no such document: zzz")

    monkeypatch.setitem(daemon.OPS, "explode", explode)
    server = await _serving(library)
    try:
        with pytest.raises(LibraryError, match="no such document: zzz"):
            await client.call_async(library.root, "explode")
        # still serving
        assert (await client.call_async(library.root, "ping"))["pid"] == os.getpid()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_events_arrive_before_the_result(
    library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An ingest pass prints per-document lines as it goes; they must reach the
    # client while it runs, not in a lump once it finishes.
    async def chatty(lib, args, emit):
        for name in ("one", "two"):
            await emit(protocol.event("filed", name=name))
        return {"filed": 2}

    monkeypatch.setitem(daemon.OPS, "chatty", chatty)
    server = await _serving(library)
    seen: list[str] = []
    try:
        result = await client.call_async(
            library.root, "chatty", on_event=lambda f: seen.append(f["name"])
        )
        assert seen == ["one", "two"]
        assert result == {"filed": 2}
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_a_watcher_that_dies_mid_request_does_not_hang_the_client(
    library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def vanish(lib, args, emit):
        raise asyncio.CancelledError

    monkeypatch.setitem(daemon.OPS, "vanish", vanish)
    server = await _serving(library)
    try:
        with pytest.raises((client.NoWatcher, asyncio.CancelledError)):
            await asyncio.wait_for(client.call_async(library.root, "vanish"), 5)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_two_libraries_are_served_independently(tmp_path: Path) -> None:
    a, b = Library(tmp_path / "a"), Library(tmp_path / "b")
    server_a = await _serving(a)
    try:
        assert (await client.call_async(a.root, "ping"))["library"] == str(a.root)
        # b has no watcher of its own, and a's does not answer for it
        assert client.serving(b.root) is False
    finally:
        await server_a.stop()


@pytest.mark.asyncio
async def test_two_ops_never_overlap(library: Library, monkeypatch) -> None:
    """Ops are serialized, because an interleaved pass files a document twice.

    Each connection is its own task and an ingest awaits the model, so without a
    lock two passes both decide the same PDF is new -- `ingest_folder` works out
    what the library holds long before it writes. The result is two store
    folders for one database row, silently.
    """
    order: list[str] = []

    async def slow(lib, args, emit):
        order.append(f"{args['tag']} in")
        await asyncio.sleep(0.05)
        order.append(f"{args['tag']} out")
        return args["tag"]

    monkeypatch.setitem(daemon.OPS, "slow", slow)
    server = await _serving(library)
    try:
        await asyncio.gather(
            client.call_async(library.root, "slow", {"tag": "a"}),
            client.call_async(library.root, "slow", {"tag": "b"}),
        )
    finally:
        await server.stop()

    # Whichever ran first, it finished before the other started.
    assert order in (
        ["a in", "a out", "b in", "b out"],
        ["b in", "b out", "a in", "a out"],
    ), order


@pytest.mark.asyncio
async def test_a_result_larger_than_asyncios_default_line_limit_survives(
    library: Library, monkeypatch
) -> None:
    """asyncio caps a line at 64 KiB unless told otherwise.

    A `list --json` of ~90 papers already exceeds that, and the failure was a
    bare ValueError from deep inside the stream reader rather than a refusal.
    """
    big = "x" * 200_000

    async def bulky(lib, args, emit):
        return big

    monkeypatch.setitem(daemon.OPS, "bulky", bulky)
    server = await _serving(library)
    try:
        assert await client.call_async(library.root, "bulky") == big
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_a_large_request_is_carried_rather_than_refused(library: Library) -> None:
    # The request direction had the same 64 KiB cap as the reply, where it
    # escaped the handler as an unhandled traceback in the watcher's log and the
    # client was told the watcher had died -- which it had not.
    server = await _serving(library)
    try:
        answer = await client.call_async(library.root, "ping", {"pad": "y" * 300_000})
        assert answer["pid"] == os.getpid()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_login_stores_through_the_watcher(library: Library, monkeypatch) -> None:
    """The keyring is touched by one long-lived process, not by every command.

    A desktop keyring confirms per process, so storing from the CLI is a dialog
    each time -- which is the whole reason this op exists.
    """
    stored: dict = {}
    monkeypatch.setattr(
        "sortyourpaperya.config.store_api_key", lambda k: stored.update(key=k)
    )
    server = await _serving(library)
    try:
        assert await client.call_async(library.root, "login", {"key": "sk-x"}) == {
            "stored": True
        }
        assert stored == {"key": "sk-x"}
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_login_without_a_key_is_refused(library: Library) -> None:
    server = await _serving(library)
    try:
        with pytest.raises(protocol.ProtocolError, match="no key given"):
            await client.call_async(library.root, "login", {"key": "   "})
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_logout_forgets_through_the_watcher(library: Library, monkeypatch) -> None:
    monkeypatch.setattr("sortyourpaperya.config.forget_api_key", lambda: True)
    server = await _serving(library)
    try:
        assert await client.call_async(library.root, "logout") == {"removed": True}
    finally:
        await server.stop()

"""Reading and changing the library while someone else is writing it.

Reads deliberately do not go through the watcher: several read-only connections
coexist, so the library stays readable when nothing is serving it. The watcher
is the fallback for the one case that excludes readers -- a pass holding the
write lock -- and these pin both halves of that.

The contention cases need a second *process*: DuckDB refuses two differently
configured connections inside one process, which is a different error than the
lock this is about.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from sortyourpaperya import client
from sortyourpaperya.db import Paper
from sortyourpaperya.library import Library


@pytest.fixture(autouse=True)
def _short_runtime_dir(monkeypatch: pytest.MonkeyPatch):
    import shutil

    short = tempfile.mkdtemp(prefix="sypyr-")
    monkeypatch.setenv("SORTYOURPAPERYA_RUNTIME_DIR", short)
    yield short
    shutil.rmtree(short, ignore_errors=True)


@pytest.fixture
def library(tmp_path: Path) -> Path:
    root = tmp_path / "lib"
    lib = Library(root)
    lib.db.upsert(
        Paper(
            file_id="abc123",
            content_hash="h",
            store_name="abc123__AI",
            document_name="vaswani.pdf",
            title="Attention Is All You Need",
            year=2017,
        )
    )
    lib.release()
    return root


def _holder(root: Path, runtime: str, *, serve: bool) -> subprocess.Popen:
    """A process holding the write lock, optionally serving as the watcher does."""
    script = textwrap.dedent(
        f"""
        import asyncio, sys
        from pathlib import Path
        from sortyourpaperya.library import Library
        from sortyourpaperya import daemon

        async def main():
            lib = Library(Path({str(root)!r}))
            lib.db.count()                     # take the write lock and keep it
            if {serve!r}:
                server = daemon.Server(lib)
                await server.start()
            print("ready", flush=True)
            await asyncio.sleep(60)

        asyncio.run(main())
        """
    )
    env = {**__import__("os").environ, "SORTYOURPAPERYA_RUNTIME_DIR": runtime}
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, env=env
    )
    assert proc.stdout is not None
    proc.stdout.readline()  # wait for "ready"
    return proc


def test_a_free_library_is_read_directly(library: Path) -> None:
    # The no-watcher case, which is most of the time and must not need one.
    with client.reading(library) as lib:
        assert type(lib.db).__name__ == "PaperDb"
        assert [p.title for p in lib.db.search("attention")] == [
            "Attention Is All You Need"
        ]


def test_a_direct_read_cannot_write(library: Path) -> None:
    with client.reading(library) as lib:
        with pytest.raises(Exception):
            lib.db.upsert(Paper(file_id="x", content_hash="y", store_name="z"))


def test_a_library_nobody_has_written_yet_is_left_to_the_caller(tmp_path: Path) -> None:
    """A fresh library is the first thing a new user reads, and it must answer.

    Read-only cannot create the database, so this hands back nothing and the
    caller opens it read-write -- which is what `sypy list` on an empty library
    always did.
    """
    with client.reading(tmp_path / "nothing") as lib:
        assert lib is None


def test_a_schema_older_than_this_build_is_left_to_the_caller(
    library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Migrations are DDL, which a read-only connection cannot run.

    Without this, the first read after an upgrade is a raw Catalog Error and the
    library stays unmigrated until something happens to write to it.
    """
    import duckdb

    from sortyourpaperya.db import PaperDb

    def stale(self):
        raise duckdb.CatalogException("Table with name model_answers does not exist!")

    monkeypatch.setattr(PaperDb, "count", stale)
    with client.reading(library) as lib:
        assert lib is None


def test_a_held_lock_with_a_watcher_is_answered_by_it(
    library: Path, _short_runtime_dir: str
) -> None:
    """The case the fallback exists for: a pass is running and a read arrives."""
    holder = _holder(library, _short_runtime_dir, serve=True)
    try:
        started = time.monotonic()
        with client.reading(library) as lib:
            assert type(lib.db).__name__ == "RemoteDb"
            papers = lib.db.search("attention")
            # Answered, not waited out: the 30s lock wait is the thing avoided.
            assert time.monotonic() - started < 5
            assert [p.title for p in papers] == ["Attention Is All You Need"]
            # Rebuilt as objects, not handed over as bare dicts.
            assert isinstance(papers[0], Paper)
            # Paths still resolve locally; only questions cross the socket.
            assert lib.store_path(papers[0]).name == "vaswani.pdf"
    finally:
        holder.kill()
        holder.wait()


def test_the_read_proxy_refuses_a_write(library: Path, _short_runtime_dir: str) -> None:
    # The whitelist is what stops a client reaching a writing method through
    # the socket; without it the proxy would be a way around the read-only open.
    holder = _holder(library, _short_runtime_dir, serve=True)
    try:
        with client.reading(library) as lib:
            with pytest.raises(Exception, match="not a readable method"):
                lib.db.upsert(Paper(file_id="x", content_hash="y", store_name="z"))
    finally:
        holder.kill()
        holder.wait()


def test_a_held_lock_with_no_watcher_hands_back_nothing(
    library: Path, _short_runtime_dir: str
) -> None:
    """With nobody to ask, the caller is told so and decides for itself.

    `cli._reading` then waits, which is what every command did before there was
    a watcher to ask at all.
    """
    holder = _holder(library, _short_runtime_dir, serve=False)
    try:
        with client.reading(library) as lib:
            assert lib is None
    finally:
        holder.kill()
        holder.wait()


# ---- writes ----------------------------------------------------------------


def test_with_no_watcher_a_write_is_left_to_the_caller(library: Path) -> None:
    # Nothing to route to, so the command takes the lock itself, exactly as it
    # did before there was a watcher at all.
    with client.writing(library) as lib:
        assert lib is None


def test_the_watcher_makes_the_change_when_it_holds_the_lock(
    library: Path, _short_runtime_dir: str
) -> None:
    holder = _holder(library, _short_runtime_dir, serve=True)
    try:
        with client.writing(library) as lib:
            assert type(lib).__name__ == "RemoteLibrary"
            lib.db.set_attribute("abc123", "doi", "10.1000/xyz")
            assert lib.db.attributes("abc123") == {"doi": "10.1000/xyz"}
    finally:
        holder.kill()
        holder.wait()

    # And it is really in the database, not just in the answer.
    with client.reading(library) as lib:
        assert lib.db.attributes("abc123") == {"doi": "10.1000/xyz"}


def test_one_handle_reaches_both_reads_and_writes(
    library: Path, _short_runtime_dir: str
) -> None:
    """`attr` reads and writes through the same handle, so the proxy must too.

    Routing every call through the read op made `set_attribute` look like an
    unreadable method, which is a confusing way to say "wrong op".
    """
    holder = _holder(library, _short_runtime_dir, serve=True)
    try:
        with client.writing(library) as lib:
            lib.db.set_attribute("abc123", "venue", "NeurIPS")
            assert lib.db.get("abc123").title == "Attention Is All You Need"
            assert lib.db.unset_attribute("abc123", "venue") is True
            assert lib.db.attributes("abc123") == {}
    finally:
        holder.kill()
        holder.wait()


def test_a_method_outside_the_whitelist_is_refused(
    library: Path, _short_runtime_dir: str
) -> None:
    # Anyone who can reach the socket can invoke what it exposes, so what it
    # exposes is opened deliberately rather than by default.
    holder = _holder(library, _short_runtime_dir, serve=True)
    try:
        with client.writing(library) as lib:
            with pytest.raises(Exception, match="not a readable method"):
                lib.db.delete("abc123")
    finally:
        holder.kill()
        holder.wait()


def test_paths_do_not_cross_the_socket(library: Path, _short_runtime_dir: str) -> None:
    # Path arithmetic needs no lock and no watcher; a round trip for it would be
    # latency bought for nothing.
    holder = _holder(library, _short_runtime_dir, serve=True)
    try:
        with client.writing(library) as lib:
            assert lib.store_dir == library / "store"
            assert lib.root == library
    finally:
        holder.kill()
        holder.wait()

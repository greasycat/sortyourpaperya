"""Long-running watcher that ingests the input folder as papers arrive.

Filesystem events are a wake-up hint only: the folder scan is the single source
of truth for what needs ingesting. That keeps the loop correct when events are
coalesced, missed, or caused by the pipeline's own writes.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import Settings
from .daemon import Server
from .discovery import Snapshot, snapshot_input
from .ingest import IngestReport, ingest_folder
from .library import FilingMode, Library, PlannedFiling
from .llm import LlmClient
from .notify import notify
from .watchlock import claim as claim_folders

log = logging.getLogger(__name__)

# How long the folder must stay unchanged before a run starts, so PDFs still
# being copied in are not ingested half-written.
SETTLE_SECONDS = 3.0

# Rescan interval when no event arrives, so a missed event delays a run rather
# than stalling the watcher.
IDLE_RESCAN_SECONDS = 60.0

# How many documents a notification names before it summarises the rest. A
# notification is read at a glance, and a pass over a folder of fifty is not
# something a person reads off their screen.
NOTIFY_LINES = 3


class _WakeHandler(FileSystemEventHandler):
    """Turns any filesystem event into a single wake-up."""

    def __init__(self, wake: asyncio.Event, loop: asyncio.AbstractEventLoop) -> None:
        self._wake = wake
        self._loop = loop

    def on_any_event(self, event) -> None:
        # Watchdog calls this from its own thread; hop back to the loop thread.
        self._loop.call_soon_threadsafe(self._wake.set)


async def watch(
    settings: Settings,
    client: LlmClient,
    library: Library,
    *,
    mode: FilingMode = FilingMode.PREVIEW,
    max_passes: int | None = None,
) -> list[IngestReport]:
    """Ingest the input folder whenever new PDFs settle in it.

    Runs until cancelled. A failed pass is reported and the watcher keeps going,
    but the same unchanged input is not retried: the files stay put until they
    change or a new PDF arrives.

    Args:
        max_passes: stop after this many ingest passes. For tests; ``None``
            runs forever.
    """
    if not settings.input_dir.is_dir():
        raise NotADirectoryError(
            f"watch input folder does not exist: {settings.input_dir}"
        )

    # Claimed before anything else happens. Two watchers sharing either folder
    # file the same document twice, because each decides what is new before
    # either writes.
    claim = claim_folders(settings.input_dir, library.root)

    # Served only once the claim is held: the socket says "the watcher for this
    # library is here", and two of them saying it would make that a lie. Taking
    # over a path an earlier watcher left behind is safe for the same reason.
    server = Server(library)
    await server.start()

    wake = asyncio.Event()
    observer = _start_observer(settings, wake)
    log.info(
        "watching %s -> %s (%d already in the library, mode=%s)",
        settings.input_dir,
        library.root,
        library.db.count(),
        mode.value,
    )

    reports: list[IngestReport] = []
    last_run: Snapshot | None = None
    try:
        while max_passes is None or len(reports) < max_passes:
            pending = _pending(settings, library)
            if not pending or pending == last_run:
                # Nothing to do, so hold no database lock while waiting or the
                # other `sortyourpaperya` commands could never run against this library.
                library.release()
                await _wait_for_change(wake)
                continue

            library.release()
            await asyncio.sleep(SETTLE_SECONDS)
            if _pending(settings, library) != pending:
                # Still arriving. Re-measure rather than ingesting a partial copy.
                continue

            # Drop events this pass is about to cause.
            wake.clear()
            log.info("ingesting %d pending PDF(s)", len(pending))
            last_run = pending

            try:
                # The same lock a client op takes. Without it the watcher's own pass
                # and a routed `sypy ingest` overlap: both decide what is new before
                # either writes, so both file the same PDF and the store is left with
                # an orphan folder for a row that exists once.
                async with server.busy:
                    report = await ingest_folder(settings, client, library, mode=mode)
            except Exception as err:  # a bad pass must not kill the watcher
                log.error("ingest failed: %s; waiting for the next change", err)
                continue

            reports.append(report)
            if report.rescan and report.rescan.changed:
                log.info(
                    "%d stored file(s) changed on disk; hashes refreshed",
                    len(report.rescan.changed),
                )
            if report.rescan and report.rescan.missing:
                log.warning(
                    "%d document(s) missing from the store",
                    len(report.rescan.missing),
                )
            # What was filed and where, before the counts. A count says a pass
            # happened; where a document landed is the part that can be wrong,
            # and the only part worth reading a category off.
            verb = "filed" if mode.writes else "would file"
            for filing in report.filed or report.planned:
                log.info(
                    "%s %s under %s",
                    verb,
                    filing.link_path.name,
                    _category(library, filing) or "no category",
                )
            log.info(
                "%s %d document(s), %d already known, %d failed",
                verb,
                report.processed,
                report.skipped_already_known,
                len(report.failed),
            )
            if report.filed:
                # Off the loop thread: a notifier that hangs holds its timeout,
                # not the watcher. Preview files nothing, so it announces
                # nothing.
                await asyncio.to_thread(_announce, library, report.filed)
            # The counts above are all a service leaves behind, and a count is
            # not something anyone can act on: a disk that filled, a key that
            # expired, and a corrupt PDF all read as "1 failed". The reason is
            # the only part worth waking up for, so each one is named.
            for path, reason in report.failed:
                log.warning("could not file %s: %s", path.name, reason)
            for path in report.skipped_oversized:
                log.warning(
                    "skipped %s: larger than the %d MB limit",
                    path.name,
                    settings.max_file_size_mb,
                )
    finally:
        observer.stop()
        observer.join(timeout=5)
        # The socket goes before the claim: a path that outlived its server
        # would have later commands dialling something that cannot answer.
        await server.stop()
        claim.release()

    return reports


def _category(library: Library, filing: PlannedFiling) -> str:
    """The tags a document was filed under, as the tree spells them."""
    relative = filing.link_path.parent.relative_to(library.tree_dir)
    return "" if str(relative) == "." else str(relative)


def _announce(library: Library, filed: list[PlannedFiling]) -> None:
    """Put a pass on screen, for a watcher nobody is watching."""
    lines = [
        f"{filing.link_path.name} \u2192 {_category(library, filing) or 'no category'}"
        for filing in filed[:NOTIFY_LINES]
    ]
    if len(filed) > NOTIFY_LINES:
        lines.append(f"+{len(filed) - NOTIFY_LINES} more")
    notify(
        f"Filed {len(filed)} document{'s' if len(filed) != 1 else ''}",
        "\n".join(lines),
    )


def _start_observer(settings: Settings, wake: asyncio.Event) -> Observer:
    observer = Observer()
    observer.schedule(
        _WakeHandler(wake, asyncio.get_running_loop()),
        str(settings.input_dir),
        recursive=settings.recursive,
    )
    observer.start()
    return observer


def _pending(settings: Settings, library: Library) -> Snapshot:
    return snapshot_input(
        settings.input_dir,
        library.root,
        recursive=settings.recursive,
        max_file_size_mb=settings.max_file_size_mb,
    )


async def _wait_for_change(wake: asyncio.Event) -> None:
    """Wait for the next event, giving up after the idle interval so we rescan."""
    try:
        await asyncio.wait_for(wake.wait(), timeout=IDLE_RESCAN_SECONDS)
    except asyncio.TimeoutError:
        return
    finally:
        wake.clear()

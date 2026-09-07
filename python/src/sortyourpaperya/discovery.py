"""Finding the PDFs a folder is currently offering, and noticing when that changes.

The watcher treats filesystem events as a wake-up hint only, so this scan is the
single source of truth for what needs ingesting. That keeps the loop correct when
events are coalesced, missed, or caused by the pipeline's own writes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# Path to size in bytes. Comparing two snapshots answers both watcher questions:
# has the folder settled, and has anything changed since the last run.
Snapshot = dict[Path, int]

_FILE_ID_CHARS = 16
_HASH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class PdfCandidate:
    """A PDF sitting in the input folder."""

    path: Path
    size_bytes: int


def discover_pdfs(root: Path, recursive: bool = False) -> list[PdfCandidate]:
    """List the PDFs ``root`` is offering, sorted by path so batching is stable.

    ``root`` is usually a folder. A single PDF is accepted as the list of one it
    is, so filing one document and filing the folder it sits in are the same
    request at different scope, and neither needs its own pipeline.

    A watched folder changes while it is being read: a download completes, a
    file is renamed, something is deleted between listing and measuring. A file
    that goes missing mid-scan is skipped rather than raised — the scan runs on
    every pass of a long-lived watcher, so an exception here would take the
    whole service down over an ordinary event. Which is also why a ``root`` that
    is neither is empty rather than an error: by the time a scan sees it, the
    folder has been deleted out from under a running watcher. A path typed by
    hand is checked where it is typed, in `cli`, and refused there.
    """
    if root.is_file():
        return _candidates([root])

    pattern = "**/*" if recursive else "*"
    return _candidates(sorted(root.glob(pattern)))


def _candidates(paths: list[Path]) -> list[PdfCandidate]:
    candidates: list[PdfCandidate] = []
    for path in paths:
        if path.suffix.lower() != ".pdf":
            continue
        try:
            if not path.is_file():
                continue
            candidates.append(PdfCandidate(path=path, size_bytes=path.stat().st_size))
        except OSError:
            continue  # vanished, or briefly unreadable; the next pass will see it
    return candidates


def snapshot_input(
    input_dir: Path,
    output_dir: Path,
    *,
    recursive: bool = False,
    max_file_size_mb: int | None = None,
) -> Snapshot:
    """Snapshot the PDFs waiting to be ingested.

    Anything inside the output folder is excluded, so a library nested under the
    watched folder is never re-ingested. Oversized files are excluded too, so
    their presence does not make the folder look permanently pending.
    """
    max_bytes = None if max_file_size_mb is None else max_file_size_mb * 1024 * 1024
    snapshot: Snapshot = {}
    for candidate in discover_pdfs(input_dir, recursive=recursive):
        if _is_within(candidate.path, output_dir):
            continue
        if max_bytes is not None and candidate.size_bytes > max_bytes:
            continue
        snapshot[candidate.path] = candidate.size_bytes
    return snapshot


def file_id(path: Path) -> str:
    """Stable identifier for a PDF, derived from its bytes.

    Content-addressed rather than path-addressed so the same paper filed twice,
    or renamed between runs, is recognised as already ingested.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()[:_FILE_ID_CHARS]


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents

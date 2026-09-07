"""The library on disk: a folder per document, and a symlink tree over them.

Every document has exactly one home, `store/<id>__<Tag>__<Tag>/`, holding the
document itself and whatever its owner keeps beside it — notes, figures,
supplements. That folder is the durable thing.

`tree/` is a view and nothing else: it can be deleted and rebuilt at any time.
Each document appears there as a single symlink to its store folder, named for
its author, year, and title, under the folders its tags make. Opening a link
leads into the store, where the document and anything kept beside it live.

Anything real found sitting in the tree is reported rather than quietly kept —
the tree is not where work is safe: see `tree_litter`.

Re-tagging renames the store folder, which carries its contents along, and the
link is re-pointed. Links are relative, so the whole library can be moved.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Sequence

import duckdb

from .bib import BIBS_DIR
from .db import Paper, PaperDb
from .discovery import file_id as hash_file
from .naming import disambiguate, link_name, parse_store_name, store_name
from .notes import DEFAULT_NOTE, NoteError, note_name, notes_in

log = logging.getLogger(__name__)

STORE_DIR = "store"
TREE_DIR = "tree"
DB_FILE = "papers.duckdb"


class FilingMode(str, Enum):
    """What filing is allowed to do to the source file.

    `COPY` exists so a real folder someone else owns — a Downloads folder, a
    shared drive — can be indexed without being rearranged underneath them.
    """

    PREVIEW = "preview"
    COPY = "copy"
    MOVE = "move"

    @property
    def writes(self) -> bool:
        return self is not FilingMode.PREVIEW


class LibraryError(RuntimeError):
    """Raised when the library on disk cannot be brought to the intended state."""


@dataclass
class RescanReport:
    """What a rescan of the store found."""

    checked: int = 0
    rehashed: int = 0
    changed: list[tuple[str, str, str]] = field(default_factory=list)
    missing: list[Paper] = field(default_factory=list)


@dataclass(frozen=True)
class BackupReport:
    """What a backup copied."""

    destination: Path
    database: Path
    documents: int
    bytes_copied: int
    bibliographies: int = 0


@dataclass(frozen=True)
class PlannedFiling:
    """What filing one paper would do, before anything is touched."""

    file_id: str
    source: Path
    store_path: Path
    link_path: Path

    def describe(self) -> str:
        return f"{self.source.name} -> {self.store_path.name} @ {self.link_path.parent}"


class Library:
    """The store folder, the symlink tree, and the database that describes them."""

    def __init__(
        self, root: Path, *, read_only: bool = False, fail_on_lock: bool = False
    ) -> None:
        self.root = root
        self.read_only = read_only
        self.store_dir = root / STORE_DIR
        self.tree_dir = root / TREE_DIR
        self.db = PaperDb(root / DB_FILE, read_only=read_only, fail_on_lock=fail_on_lock)

    def close(self) -> None:
        self.db.close()

    def release(self) -> None:
        """Drop the database lock while idle, so other commands can run."""
        self.db.release()

    def __enter__(self) -> "Library":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def document_dir(self, paper: Paper) -> Path:
        """The document's folder in the store: its home, and its owner's."""
        return self.store_dir / paper.store_name

    def store_path(self, paper: Paper) -> Path:
        """The document file itself, inside its folder."""
        return self.document_dir(paper) / paper.document_name

    def plan_filing(self, paper: Paper, source: Path) -> PlannedFiling:
        """Where a paper would land, without touching the filesystem."""
        return PlannedFiling(
            file_id=paper.file_id,
            source=source,
            store_path=self.store_path(paper),
            link_path=self._link_path(paper, taken=set()),
        )

    def place_file(
        self, paper: Paper, source: Path, mode: FilingMode = FilingMode.MOVE
    ) -> Path:
        """Create the document's folder and put its file inside.

        Touches no database, so a pass can place every file of a batch before
        taking the write lock once for all of them.

        Raises:
            LibraryError: in preview mode, or if the folder already exists.
        """
        if mode is FilingMode.PREVIEW:
            raise LibraryError("place_file called in preview mode")

        directory = self.document_dir(paper)
        if directory.exists():
            raise LibraryError(f"store already holds {directory.name}")
        directory.mkdir(parents=True)

        target = directory / paper.document_name
        # `shutil` rather than `os.replace` because the source folder is often
        # on a different filesystem than the library, which `os.replace`
        # refuses.
        if mode is FilingMode.COPY:
            shutil.copy2(source, target)
        else:
            shutil.move(str(source), str(target))

        stat = target.stat()
        paper.size_bytes = stat.st_size
        paper.stored_mtime_ms = int(stat.st_mtime * 1000)
        return target

    def record(self, papers: Sequence[Paper]) -> None:
        """Write documents to the database, in one transaction."""
        self.db.upsert_many(papers)

    def link_paper(self, paper: Paper, taken: set[Path] | None = None) -> Path:
        """Link a document into the tree. Touches no database."""
        return self._link(paper, taken)

    def file_paper(
        self, paper: Paper, source: Path, mode: FilingMode = FilingMode.MOVE
    ) -> PlannedFiling:
        """Place, record, and link one document.

        The three steps in sequence, for a caller handling a single document.
        Ingest drives the same steps in phases across a whole batch so the
        database is only touched once the file work is done.

        The file lands before the database write, so a crash leaves a file in
        the store with no row rather than a row pointing at nothing — a folder
        that exists but is not claimed, rather than a claim on nothing.

        Re-ingesting does not heal it: ids are minted fresh, so a second attempt
        builds a second folder and the first is left behind. `sortyourpaperya fsck` is what
        finds those folders and adopts them.
        """
        target = self.place_file(paper, source, mode)
        self.record([paper])
        link = self.link_paper(paper)
        return PlannedFiling(
            file_id=paper.file_id, source=source, store_path=target, link_path=link
        )

    def retag(
        self, file_id: str, tags: list[str], keywords: list[str] | None = None
    ) -> Paper:
        """Give a document new tags, and optionally new keywords.

        Renames its folder in the store, so notes and supplements kept beside it
        travel with it, and re-points the link. Nothing is copied and nothing in
        the folder is touched.

        Keywords are only replaced when given, so re-tagging by hand leaves what
        the library already knew about the document alone.

        Raises:
            LibraryError: if the document is unknown or its folder is missing.
        """
        paper = self.db.get(file_id)
        if paper is None:
            raise LibraryError(f"no document with id {file_id}")

        old_dir = self.document_dir(paper)
        new_name = store_name(paper.file_id, tags, suffix="")
        new_dir = self.store_dir / new_name

        if not old_dir.is_dir():
            raise LibraryError(f"store is missing {paper.store_name}")
        if new_dir != old_dir and new_dir.exists():
            raise LibraryError(f"store already holds {new_name}")

        self._unlink(paper)
        if new_dir != old_dir:
            os.replace(old_dir, new_dir)
        self.db.set_tags(file_id, tags, new_name, keywords)

        updated = self.db.get(file_id)
        assert updated is not None  # just written
        self._link(updated)
        return updated

    def remove(self, file_id: str) -> Paper:
        """Take a document out of the library: link, folder, and rows.

        The link goes first and the rows last, so an interruption leaves the
        library holding less than it claims rather than claiming more than it
        holds. A stale row pointing at a deleted folder is the harder state to
        notice.

        Raises:
            LibraryError: if the document is unknown.
        """
        paper = self.db.get(file_id)
        if paper is None:
            raise LibraryError(f"no document with id {file_id}")

        self._unlink(paper)

        # The whole folder goes, including anything kept beside the document.
        # That is why removal asks first.
        directory = self.document_dir(paper)
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass  # already gone; the row is still worth clearing
        except OSError as err:
            # The row is the only thing that says this folder belongs to a
            # document. Dropping it here would leave the folder on disk with
            # nothing pointing at it and no command able to find it, so the
            # document stays in the library and the failure is reported. The
            # link is restored by `sortyourpaperya tree`.
            raise LibraryError(f"could not remove {directory}: {err}") from err

        self.db.delete(file_id)
        return paper

    def rescan(self, *, write: bool = True) -> RescanReport:
        """Bring the recorded content hashes back in line with the store.

        A file edited in place — annotated, re-saved — keeps its name but no
        longer matches its recorded hash. Left alone that hash goes stale, and
        since it is the key that recognises a document the library already has,
        the edited copy arriving later would be ingested a second time.

        Runs in phases: read the rows, drop the lock, stat and hash, then write
        once. The file work is the slow part and holds no lock while it happens.
        Size and mtime are checked first, so an unchanged library reads nothing.

        Args:
            write: record what was found. A preview is supposed to leave the
                library exactly as it found it, and a refreshed hash is a real
                change to it, so preview asks for the report without the write.
                The report is the same either way, which is what lets a preview
                still agree with the apply that follows it.
        """
        report = RescanReport()
        papers = self.db.all_papers()
        self.db.release()

        updates: list[tuple[str, str, int, int]] = []
        for paper in papers:
            path = self.store_path(paper)
            if not path.exists():
                report.missing.append(paper)
                continue

            report.checked += 1
            stat = path.stat()
            size, mtime_ms = stat.st_size, int(stat.st_mtime * 1000)
            if paper.size_bytes == size and paper.stored_mtime_ms == mtime_ms:
                continue

            # The stat moved, so the content might have. Only now is it worth
            # reading the file.
            report.rehashed += 1
            digest = hash_file(path)
            if digest != paper.content_hash:
                report.changed.append((paper.file_id, paper.content_hash, digest))
            # Recorded even when the hash is unchanged — a touched file should
            # not be re-read on every future scan.
            updates.append((paper.file_id, digest, size, mtime_ms))

        if write:
            self.db.set_stored_file_states(updates)
        return report

    def backup(self, destination: Path) -> "BackupReport":
        """Copy everything durable in the library to `destination`.

        The store and the database are only useful together: the store without
        the database is a folder of documents nothing can find, and the database
        without the store is a catalogue of files that are gone. So they are
        copied by one command, in one place, rather than left as two things to
        remember.

        `bibs/` goes with them. A bibliography is hand-made and derivable from
        nothing, so leaving it out of the backup would make this command the
        thing that loses it.

        **The database goes first.** Between the two copies a watcher may file
        another document, and which half is behind decides what the copy is
        worth: a folder with no row is an orphan, which `sortyourpaperya fsck --adopt`
        brings back, while a row with no folder is a document that no longer
        exists anywhere. Taking the database first makes the recoverable
        mistake the only one available.

        `tree/` is not copied. It holds nothing but links and is rebuilt from
        the database by `sortyourpaperya tree`.

        Raises:
            LibraryError: if the destination is unusable, or lies inside the
                library — which would copy the backup into itself.
        """
        destination = destination.expanduser()
        if destination.exists() and any(destination.iterdir()):
            raise LibraryError(f"{destination} already exists and is not empty")
        if _is_within(destination, self.root):
            raise LibraryError(
                f"{destination} is inside the library; back up somewhere else"
            )

        destination.mkdir(parents=True, exist_ok=True)
        try:
            self.db.backup_to(destination / DB_FILE)
        except duckdb.Error as err:
            raise LibraryError(f"could not copy the database: {err}") from err
        self.release()

        documents = 0
        copied_bytes = 0
        if self.store_dir.is_dir():
            target = destination / STORE_DIR
            try:
                # `symlinks=True` so a link kept beside a document is copied as
                # a link rather than followed and copied as its contents.
                shutil.copytree(self.store_dir, target, symlinks=True)
            except OSError as err:
                raise LibraryError(f"could not copy the store: {err}") from err
            documents = sum(1 for entry in target.iterdir() if entry.is_dir())
            copied_bytes = _tree_bytes(target)

        bibliographies = 0
        bibs = self.root / BIBS_DIR
        if bibs.is_dir():
            try:
                # `symlinks=True` for the same reason the store uses it, and
                # more sharply: a bibliography's books are links into the
                # store, and following them would copy every book a second
                # time and restore as real files what were links.
                shutil.copytree(bibs, destination / BIBS_DIR, symlinks=True)
            except OSError as err:
                raise LibraryError(f"could not copy the bibliographies: {err}") from err
            bibliographies = sum(
                1 for entry in (destination / BIBS_DIR).iterdir() if entry.is_dir()
            )

        return BackupReport(
            destination=destination,
            database=destination / DB_FILE,
            documents=documents,
            bytes_copied=copied_bytes,
            bibliographies=bibliographies,
        )

    def notes(self, paper: Paper) -> list[Path]:
        """Every note kept beside a document, in the order they read.

        The document itself is never a note, even when it was filed as markdown
        or JSON: it is the thing the notes are about.
        """
        return notes_in(self.document_dir(paper), besides=[paper.document_name])

    def note_path(self, paper: Paper, name: str = DEFAULT_NOTE) -> Path:
        """Where a note of that name belongs, whether or not it exists yet.

        `notes` owns what may be called a note; what is added here is that the
        document's own file is not one, even when it was filed as markdown or
        JSON: it is the thing the notes are about.

        Raises:
            LibraryError: if the name is not a plain filename, does not name a
                note format, or is the document's own file.
        """
        try:
            filename = note_name(name)
        except NoteError as err:
            raise LibraryError(str(err)) from err
        if filename == paper.document_name:
            raise LibraryError(f"{name!r} is the document itself, not a note about it")
        return self.document_dir(paper) / filename

    def migrate_store_layout(self) -> list[str]:
        """Move documents from the old flat store into a folder each.

        Libraries written before documents had folders hold
        `store/<id>__<Tag>.pdf`. Each becomes `store/<id>__<Tag>/<name>.pdf`, so
        the document has somewhere to keep notes beside it.

        Idempotent and resumable: a document already in a folder is left alone,
        so an interrupted run is finished by running it again. Returns the ids
        it moved.
        """
        moved: list[str] = []
        for paper in self.db.all_papers():
            if paper.document_name and self.store_path(paper).is_file():
                continue

            flat = self.store_dir / paper.store_name
            if not flat.is_file():
                continue  # already a folder, or genuinely missing

            folder_name = Path(paper.store_name).stem
            suffix = Path(paper.store_name).suffix or ".pdf"
            document_name = link_name(
                fallback=f"{folder_name}{suffix}",
                authors=paper.authors,
                year=paper.year,
                title=paper.title,
                suffix=suffix,
            )
            folder = self.store_dir / folder_name
            folder.mkdir(parents=True, exist_ok=True)
            os.replace(flat, folder / document_name)
            self.db.set_store_layout(paper.file_id, folder_name, document_name)
            moved.append(paper.file_id)
        return moved

    def refresh_document_names(self) -> list[tuple[str, str, str]]:
        """Rename stored files the naming rules would now spell differently.

        A document's filename is derived from its title, authors, and year, so
        changing how that derivation works leaves existing files spelled the old
        way — and the tree, which recomputes the name on every rebuild, spelled
        the new one. Renaming the file keeps the two saying the same thing.

        Returns `(file_id, old, new)` for each rename.
        """
        renamed: list[tuple[str, str, str]] = []
        for paper in self.db.all_papers():
            directory = self.document_dir(paper)
            current = directory / paper.document_name
            if not paper.document_name or not current.is_file():
                continue

            suffix = Path(paper.document_name).suffix or ".pdf"
            wanted = link_name(
                fallback=f"{paper.store_name}{suffix}",
                authors=paper.authors,
                year=paper.year,
                title=paper.title,
                suffix=suffix,
            )
            if wanted == paper.document_name or (directory / wanted).exists():
                continue

            os.replace(current, directory / wanted)
            self.db.set_store_layout(paper.file_id, paper.store_name, wanted)
            renamed.append((paper.file_id, paper.document_name, wanted))
        return renamed

    def existing_categories(self, limit: int | None = None) -> list[str]:
        """Category paths already in use, for steering a new document."""
        return self.db.tag_paths(limit=limit)

    def rebuild_tree(self) -> int:
        """Rebuild every symlink from the database, discarding the old tree.

        Only the links are discarded. Anything else found in the tree — a note
        filed beside a document, a folder someone made — is left alone, because
        a rebuild converging the links is not a reason to delete work.
        """
        _clear_links(self.tree_dir)

        taken: set[Path] = set()
        linked = 0
        for paper in self.db.all_papers():
            if not self.store_path(paper).exists():
                continue
            try:
                self._link(paper, taken=taken)
            except LibraryError as err:
                # One document whose place in the tree is occupied must not
                # cost the rest of the library its links. `tree_litter` names
                # what is in the way.
                log.warning("%s", err)
                continue
            linked += 1
        return linked

    def tree_litter(self) -> list[Path]:
        """Real files sitting in the tree, which is not where work is safe.

        The tree is rebuildable and is not part of a backup of the store and the
        database, so a file kept here is one deleted tree away from being gone.
        Reported rather than removed: it is not this tool's to delete.

        Dotfiles are skipped — `.DS_Store` and its kind are the filesystem's
        litter, not the owner's work.
        """
        found: list[Path] = []
        if not self.tree_dir.exists():
            return found
        for parent, dirnames, filenames in os.walk(self.tree_dir, followlinks=False):
            here = Path(parent)
            dirnames[:] = [name for name in dirnames if not (here / name).is_symlink()]
            for name in filenames:
                entry = here / name
                if not entry.is_symlink() and not name.startswith("."):
                    found.append(entry)
        return sorted(found)

    def orphans(self) -> list[Path]:
        """Store folders with no row — documents the library cannot see.

        The counterpart to `missing_files`. Filing puts the file down before
        recording it, so anything that interrupts the pass between the two
        leaves a folder nothing points at: absent from `list`, from the tree,
        from dedupe, and from `remove`. Without this it is found only by
        looking in the store by hand.
        """
        if not self.store_dir.is_dir():
            return []
        known = {paper.store_name for paper in self.db.all_papers()}
        return sorted(
            entry
            for entry in self.store_dir.iterdir()
            if entry.is_dir() and not entry.is_symlink() and entry.name not in known
        )

    def adopt(self, directory: Path) -> Paper:
        """Give an orphaned store folder a row, so the library can see it again.

        Identity, tags, and the content hash come back — the folder name carries
        the first two and the file carries the third. Title, authors, and year
        do not: they only ever lived in the database. The document becomes
        visible, de-duplicated, and re-taggable, named by its file.

        Raises:
            LibraryError: if the folder is not named like a store folder, or
                does not hold exactly one document.
        """
        try:
            file_id, tags = parse_store_name(directory.name)
        except ValueError as err:
            raise LibraryError(f"{directory.name} is not a store folder") from err

        documents = [
            entry
            for entry in sorted(directory.iterdir())
            if entry.is_file() and entry.suffix.lower() == ".pdf"
        ]
        if len(documents) != 1:
            raise LibraryError(
                f"{directory.name} holds {len(documents)} documents; "
                "expected exactly one"
            )

        document = documents[0]
        stat = document.stat()
        paper = Paper(
            file_id=file_id,
            content_hash=hash_file(document),
            store_name=directory.name,
            document_name=document.name,
            original_name=document.name,
            source_path=str(document),
            size_bytes=stat.st_size,
            stored_mtime_ms=int(stat.st_mtime * 1000),
            tags=tags,
        )
        self.db.upsert(paper)
        self._link(paper)
        return paper

    def missing_files(self) -> list[Paper]:
        """Papers the database knows about whose file is gone from the store."""
        return [
            paper
            for paper in self.db.all_papers()
            if not self.store_path(paper).exists()
        ]

    # ---- links -------------------------------------------------------------

    def _link_path(self, paper: Paper, taken: set[Path]) -> Path:
        """Where this document's link belongs, under the folders its tags make."""
        parent = self.tree_dir.joinpath(*paper.tags) if paper.tags else self.tree_dir
        name = link_name(
            fallback=paper.store_name,
            authors=paper.authors,
            year=paper.year,
            title=paper.title,
            suffix="",
        )
        candidate = parent / name
        if candidate in taken:
            candidate = parent / disambiguate(name, paper.file_id)
        return candidate

    def _link(self, paper: Paper, taken: set[Path] | None = None) -> Path:
        """Point this document's link at its store folder.

        Only ever replaces a symlink. A real file or folder sitting where the
        link belongs is somebody's work — `tree_litter` reports it and tells
        them to move it — and unlinking whatever is in the way would delete it
        to make room for something rebuildable. So the document takes its
        id-decorated name instead, which no other document can want.

        Raises:
            LibraryError: if that name is occupied by something real too. The
                link is not placed rather than a file being destroyed for it.
        """
        taken = taken if taken is not None else set()
        path = self._link_path(paper, taken)
        link = self._place_link(paper, path)
        if link is None:
            path = path.parent / disambiguate(path.name, paper.file_id)
            link = self._place_link(paper, path)
        if link is None:
            raise LibraryError(
                f"cannot link {paper.file_id}: {path} is a real file, not a "
                "link. Move it into the document's folder in the store to keep it."
            )
        taken.add(path)
        return link

    def _place_link(self, paper: Paper, path: Path) -> Path | None:
        """Put the link at `path`, or None if something real is in its place."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            path.unlink()  # ours, and pointing at a previous answer
        elif path.exists():
            return None
        path.symlink_to(
            os.path.relpath(self.document_dir(paper), path.parent),
            target_is_directory=True,
        )
        return path

    def _unlink(self, paper: Paper) -> None:
        """Remove this document's link, and any branches it leaves empty.

        Found by where it points, not by what it is called. Two documents whose
        author, year, and title agree want the same name, so one of them gets an
        id appended — and computing the undecorated name here would delete the
        *other* document's link and leave this one dangling.
        """
        parent = self.tree_dir.joinpath(*paper.tags) if paper.tags else self.tree_dir
        if not parent.is_dir():
            return

        target = os.path.realpath(self.document_dir(paper))
        for entry in list(parent.iterdir()):
            if entry.is_symlink() and os.path.realpath(entry) == target:
                entry.unlink()
        _prune_empty(parent, stop=self.tree_dir)


def _clear_links(root: Path) -> None:
    """Remove every symlink under `root`, and the folders left empty.

    Never descends into a link: they point at store folders, and walking into
    one would mean walking the library's real contents.
    """
    if not root.exists():
        return
    directories: list[Path] = []
    for parent, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(parent)
        directories.append(here)
        for name in list(dirnames):
            entry = here / name
            if entry.is_symlink():
                entry.unlink()
                dirnames.remove(name)
        for name in filenames:
            entry = here / name
            if entry.is_symlink():
                entry.unlink()

    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        if directory == root:
            continue
        try:
            directory.rmdir()
        except OSError:
            pass  # holds something that is not ours


def _tree_bytes(root: Path) -> int:
    """Total size of the real files under `root`. Links count as nothing."""
    total = 0
    for parent, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            entry = Path(parent) / name
            if not entry.is_symlink():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    return total


def _is_within(path: Path, parent: Path) -> bool:
    """Whether `path` is `parent` or sits under it, without either having to exist."""
    try:
        resolved, root = path.resolve(), parent.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def _prune_empty(directory: Path, stop: Path) -> None:
    current = directory
    while current != stop and stop in current.parents:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent

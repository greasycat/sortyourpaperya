from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import write_pdf

from sortyourpaperya.db import Paper
from sortyourpaperya.library import FilingMode, Library, LibraryError
from sortyourpaperya.naming import link_name, new_id, store_name


def _paper(tags: list[str], **overrides) -> Paper:
    paper_id = overrides.pop("file_id", new_id())
    fields = {
        "file_id": paper_id,
        "content_hash": "hash-" + paper_id,
        # The folder carries id and tags; the file inside carries the readable
        # name, exactly as ingest builds them.
        "store_name": store_name(paper_id, tags, suffix=""),
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani"],
        "year": 2017,
        "tags": tags,
    }
    fields.update(overrides)
    fields.setdefault(
        "document_name",
        link_name(
            fallback=f"{fields['store_name']}.pdf",
            authors=fields.get("authors") or [],
            year=fields.get("year"),
            title=fields.get("title"),
        ),
    )
    return Paper(**fields)


def test_filing_gives_the_document_its_own_folder_in_the_store(
    library: Library, tmp_path: Path
) -> None:
    source = write_pdf(tmp_path / "src" / "raw.pdf", "text")
    paper = _paper(["AI", "Transformers"])

    filing = library.file_paper(paper, source)

    assert not source.exists(), "the source should have been moved, not copied"
    # The folder is the document's home; the file sits inside it.
    assert filing.store_path.parent == library.store_dir / paper.store_name
    assert filing.store_path.name == paper.document_name
    assert filing.store_path.is_file()
    assert paper.store_name.startswith(paper.file_id)
    assert paper.store_name.endswith("__AI__Transformers")


def test_the_tree_is_symlinks_pointing_back_into_the_store(
    library: Library, tmp_path: Path
) -> None:
    source = write_pdf(tmp_path / "src" / "raw.pdf", "text")
    paper = _paper(["AI", "Transformers"])

    filing = library.file_paper(paper, source)

    link = filing.link_path
    # One link per document, named for it, under the folders its tags make.
    assert link == (
        library.tree_dir / "AI" / "Transformers" / "vaswani_2017_attention-is-all-you-need"
    )
    assert link.is_symlink()
    assert link.resolve() == library.document_dir(paper).resolve()
    assert (link / paper.document_name).is_file()


def test_links_are_relative_so_the_library_can_be_moved(
    library: Library, tmp_path: Path
) -> None:
    source = write_pdf(tmp_path / "src" / "raw.pdf", "text")
    filing = library.file_paper(_paper(["AI"]), source)
    library.close()

    moved = tmp_path / "moved-library"
    library.root.rename(moved)
    relocated = moved / filing.link_path.relative_to(library.root)

    assert relocated.is_symlink()
    assert relocated.resolve().is_dir(), "relative link should survive the move"
    assert any(relocated.iterdir()), "and still lead to the document"


def test_a_paper_with_no_usable_metadata_links_under_its_store_name(
    library: Library, tmp_path: Path
) -> None:
    source = write_pdf(tmp_path / "src" / "raw.pdf", "text")
    paper = _paper(["AI"], title=None, authors=[], year=None)

    filing = library.file_paper(paper, source)

    assert filing.link_path.name == paper.store_name


def test_retagging_renames_the_file_and_moves_the_link(
    library: Library, tmp_path: Path
) -> None:
    source = write_pdf(tmp_path / "src" / "raw.pdf", "text")
    filing = library.file_paper(_paper(["AI", "Transformers"]), source)
    old_store_path = filing.store_path.parent

    updated = library.retag(filing.file_id, ["Systems", "Databases"])

    assert not old_store_path.exists()
    assert (library.store_dir / updated.store_name).is_dir()
    assert (library.store_dir / updated.store_name / updated.document_name).is_file()
    assert updated.store_name.endswith("__Systems__Databases")
    assert updated.store_name.startswith(filing.file_id), "the id must not change"
    new_link = library.tree_dir / "Systems" / "Databases" / filing.link_path.name
    assert new_link.is_symlink()
    assert not (library.tree_dir / "AI").exists(), "the emptied branch should be pruned"


def test_retagging_an_unknown_paper_is_refused(library: Library) -> None:
    with pytest.raises(LibraryError, match="no document"):
        library.retag("ffffffffffff", ["AI"])


def test_filing_over_an_existing_store_entry_is_refused(
    library: Library, tmp_path: Path
) -> None:
    paper = _paper(["AI"])
    library.file_paper(paper, write_pdf(tmp_path / "a" / "raw.pdf", "one"))
    second = write_pdf(tmp_path / "b" / "raw.pdf", "two")

    with pytest.raises(LibraryError, match="already holds"):
        library.file_paper(paper, second)

    assert second.exists(), "a refused filing must not consume the source"


def test_rebuilding_the_tree_restores_it_after_deletion(
    library: Library, tmp_path: Path
) -> None:
    for index in range(3):
        library.file_paper(
            _paper(["AI", f"Sub{index}"], title=f"Paper {index}"),
            write_pdf(tmp_path / f"src{index}" / "raw.pdf", f"text {index}"),
        )

    import shutil

    shutil.rmtree(library.tree_dir)
    linked = library.rebuild_tree()

    assert linked == 3
    links = [p for p in library.tree_dir.rglob("*") if p.is_symlink()]
    assert len(links) == 3
    assert all(link.resolve().is_dir() for link in links)


def test_rebuilding_disambiguates_two_papers_that_want_the_same_link_name(
    library: Library, tmp_path: Path
) -> None:
    # Same author, year, and title, so both want the identical link name.
    for index in range(2):
        library.file_paper(
            _paper(["AI"]), write_pdf(tmp_path / f"src{index}" / "raw.pdf", f"t{index}")
        )

    library.rebuild_tree()

    links = sorted(p.name for p in (library.tree_dir / "AI").iterdir())
    assert len(links) == 2, f"one link overwrote the other: {links}"
    assert len({link for link in links}) == 2


def test_missing_files_are_reported_not_linked(
    library: Library, tmp_path: Path
) -> None:
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    filing.store_path.unlink()

    linked = library.rebuild_tree()

    assert linked == 0
    assert [paper.file_id for paper in library.missing_files()] == [filing.file_id]


def test_removing_takes_the_link_the_file_and_the_record(
    library: Library, tmp_path: Path
) -> None:
    filing = library.file_paper(
        _paper(["AI", "Transformers"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )

    removed = library.remove(filing.file_id)

    assert removed.file_id == filing.file_id
    assert not filing.store_path.exists(), "the stored file should be gone"
    assert not filing.link_path.exists(), "the link should be gone"
    assert library.db.get(filing.file_id) is None
    assert not (library.tree_dir / "AI").exists(), "the emptied branch is pruned"


def test_removing_leaves_the_other_documents_alone(
    library: Library, tmp_path: Path
) -> None:
    keep = library.file_paper(
        _paper(["AI"], title="Kept"), write_pdf(tmp_path / "a" / "raw.pdf", "one")
    )
    drop = library.file_paper(
        _paper(["Systems"], title="Dropped"), write_pdf(tmp_path / "b" / "raw.pdf", "two")
    )

    library.remove(drop.file_id)

    assert keep.store_path.is_file()
    assert keep.link_path.is_symlink()
    assert [p.file_id for p in library.db.all_papers()] == [keep.file_id]


def test_removing_an_unknown_document_is_refused(library: Library) -> None:
    with pytest.raises(LibraryError, match="no document"):
        library.remove("ffffffffffff")


def test_existing_categories_are_the_paths_already_in_use(
    library: Library, tmp_path: Path
) -> None:
    library.file_paper(
        _paper(["Medicine", "Radiology"]), write_pdf(tmp_path / "a" / "r.pdf", "one")
    )
    library.file_paper(
        _paper(["Medicine", "Radiology"]), write_pdf(tmp_path / "b" / "r.pdf", "two")
    )
    library.file_paper(
        _paper(["Finance", "Bills"]), write_pdf(tmp_path / "c" / "r.pdf", "three")
    )

    assert library.existing_categories() == ["Finance/Bills", "Medicine/Radiology"]


def test_editing_a_stored_file_in_place_refreshes_its_hash(
    library: Library, tmp_path: Path
) -> None:
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    original_hash = library.db.get(filing.file_id).content_hash

    # Same name, different bytes — an annotation, or a re-save.
    filing.store_path.write_bytes(filing.store_path.read_bytes() + b"\n% annotated\n")
    report = library.rescan()

    refreshed = library.db.get(filing.file_id)
    assert refreshed.content_hash != original_hash
    assert [(f, o, n) for f, o, n in report.changed] == [
        (filing.file_id, original_hash, refreshed.content_hash)
    ]


def test_a_refreshed_hash_stops_the_edited_copy_being_ingested_twice(
    library: Library, tmp_path: Path
) -> None:
    # The stale hash matters because it is the key that recognises a document
    # the library already holds.
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    filing.store_path.write_bytes(filing.store_path.read_bytes() + b"\n% annotated\n")
    edited_hash = _sha_of(filing.store_path)

    assert library.db.find_by_content_hash(edited_hash) is None, "stale before rescan"
    library.rescan()
    assert library.db.find_by_content_hash(edited_hash) == filing.file_id


def test_an_unchanged_library_is_not_reread(library: Library, tmp_path: Path) -> None:
    # The whole point of recording size and mtime: a quiet library costs one
    # stat per document, not a full read of every file.
    for index in range(3):
        library.file_paper(
            _paper(["AI"], title=f"Paper {index}"),
            write_pdf(tmp_path / f"src{index}" / "raw.pdf", f"text {index}"),
        )

    report = library.rescan()

    assert report.checked == 3
    assert report.rehashed == 0, "nothing changed, so nothing should be reread"
    assert report.changed == []


def test_a_touched_but_unchanged_file_is_reread_once_then_settles(
    library: Library, tmp_path: Path
) -> None:
    import os

    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    # The fixture fabricates a hash rather than reading the file, and the stat
    # gate means a rescan would never notice. Record the real one, so a moved
    # mtime is the only thing this test changes.
    stat = filing.store_path.stat()
    library.db.set_stored_file_state(
        filing.file_id,
        _sha_of(filing.store_path),
        stat.st_size,
        int(stat.st_mtime * 1000),
    )
    os.utime(filing.store_path, (stat.st_atime, stat.st_mtime + 5))

    first = library.rescan()
    second = library.rescan()

    assert first.rehashed == 1, "a moved mtime must be checked"
    assert first.changed == [], "the bytes did not change"
    assert second.rehashed == 0, "the new stat should have been recorded"


def test_rescan_reports_a_file_that_vanished_rather_than_failing(
    library: Library, tmp_path: Path
) -> None:
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    filing.store_path.unlink()

    report = library.rescan()

    assert [paper.file_id for paper in report.missing] == [filing.file_id]
    assert report.checked == 0


def _sha_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def test_a_note_filed_beside_a_document_survives_a_rebuild(
    library: Library, tmp_path: Path
) -> None:
    """The reason each document gets a folder: things can be kept in it.

    A rebuild converges the links. It is not a licence to delete work someone
    filed alongside a document.
    """
    filing = library.file_paper(
        _paper(["AI", "Transformers"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    note = filing.link_path.parent / "my-notes.md"
    note.write_text("what I thought of this paper")

    library.rebuild_tree()

    assert note.is_file(), "a rebuild must not delete files it did not create"
    assert note.read_text() == "what I thought of this paper"
    assert filing.link_path.is_symlink(), "the link is still rebuilt"


def test_a_note_follows_its_document_when_it_is_retagged(
    library: Library, tmp_path: Path
) -> None:
    filing = library.file_paper(
        _paper(["AI", "Transformers"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    (filing.link_path / "my-notes.md").write_text("notes")

    library.retag(filing.file_id, ["Systems", "Databases"])

    moved = (
        library.tree_dir / "Systems" / "Databases" / filing.link_path.name / "my-notes.md"
    )
    assert moved.is_file(), "the note lives in the store folder and moves with it"
    assert moved.read_text() == "notes"
    assert not (library.tree_dir / "AI").exists(), "the emptied branch is pruned"


def test_the_link_still_resolves_after_a_retag_changes_depth(
    library: Library, tmp_path: Path
) -> None:
    # Links are relative, so a folder that moves to a different depth needs its
    # link re-pointed rather than carried along as it was.
    filing = library.file_paper(
        _paper(["AI", "Vision", "Detection"]),
        write_pdf(tmp_path / "src" / "raw.pdf", "text"),
    )

    updated = library.retag(filing.file_id, ["Systems"])

    link = library.tree_dir / "Systems" / filing.link_path.name
    assert link.is_symlink()
    assert link.resolve() == (library.store_dir / updated.store_name).resolve()
    assert (link / updated.document_name).is_file(), "and still reaches the document"


def test_removing_leaves_behind_a_file_stranded_in_the_tree(
    library: Library, tmp_path: Path
) -> None:
    # Written in the wrong place, so it was never the document's to begin with;
    # removing the document does not make it this command's to delete either.
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    stray = filing.link_path.parent / "draft-thoughts.md"
    stray.write_text("written in the wrong place")

    library.remove(filing.file_id)

    assert not filing.link_path.exists(), "the link goes"
    assert not filing.store_path.exists(), "the stored file goes"
    assert stray.is_file(), "someone else's file is not this command's to delete"


def test_removing_takes_the_notes_kept_with_the_document(
    library: Library, tmp_path: Path
) -> None:
    # Notes live in the document's store folder, so they are part of it and go
    # with it. That is why removal asks first.
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    paper = library.db.get(filing.file_id)
    note = library.note_path(paper)
    note.write_text("why I saved this")

    library.remove(filing.file_id)

    assert not note.exists()
    assert not library.document_dir(paper).exists()


def test_removing_takes_the_folder_when_it_is_empty(
    library: Library, tmp_path: Path
) -> None:
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )

    library.remove(filing.file_id)

    assert not filing.link_path.parent.exists()
    assert not (library.tree_dir / "AI").exists()


def test_migrating_a_flat_store_gives_each_document_a_folder(
    library: Library, tmp_path: Path
) -> None:
    """A library written before documents had folders is moved into them."""
    paper = _paper(["AI", "Transformers"])
    # Recreate the old layout: the document loose in the store, no inner name.
    flat_name = f"{paper.store_name}.pdf"
    library.store_dir.mkdir(parents=True, exist_ok=True)
    write_pdf(library.store_dir / flat_name, "text")
    library.db.upsert(replace(paper, store_name=flat_name, document_name=""))

    moved = library.migrate_store_layout()

    assert moved == [paper.file_id]
    migrated = library.db.get(paper.file_id)
    assert migrated.store_name == paper.store_name, "the folder keeps id and tags"
    assert migrated.document_name == "vaswani_2017_attention-is-all-you-need.pdf"
    assert library.store_path(migrated).is_file()
    assert not (library.store_dir / flat_name).exists(), "the flat file is gone"


def test_migrating_twice_changes_nothing_the_second_time(
    library: Library, tmp_path: Path
) -> None:
    # Idempotent, so an interrupted migration is finished by running it again.
    paper = _paper(["AI"])
    flat_name = f"{paper.store_name}.pdf"
    library.store_dir.mkdir(parents=True, exist_ok=True)
    write_pdf(library.store_dir / flat_name, "text")
    library.db.upsert(replace(paper, store_name=flat_name, document_name=""))

    first = library.migrate_store_layout()
    second = library.migrate_store_layout()

    assert first == [paper.file_id]
    assert second == [], "a document already in a folder is left alone"


def test_migration_leaves_an_already_foldered_document_untouched(
    library: Library, tmp_path: Path
) -> None:
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    note = filing.store_path.parent / "notes.md"
    note.write_text("keep me")

    assert library.migrate_store_layout() == []
    assert note.read_text() == "keep me"


def test_a_note_lives_in_the_store_so_it_outlives_the_tree(
    library: Library, tmp_path: Path
) -> None:
    """The reason the folder moved into the store.

    A note kept only in the tree is lost when the tree is deleted, and the tree
    is disposable by design.
    """
    import shutil as _shutil

    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    paper = library.db.get(filing.file_id)
    library.note_path(paper).write_text("why I saved this")

    _shutil.rmtree(library.tree_dir)
    library.rebuild_tree()

    assert library.note_path(paper).read_text() == "why I saved this"
    assert (filing.link_path / "notes.md").is_file(), "and is reachable through the tree"


def test_a_note_is_any_markdown_or_json_beside_the_document(
    library: Library, tmp_path: Path
) -> None:
    """The name is the owner's to choose; the format is what makes it a note."""
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    paper = library.db.get(filing.file_id)
    folder = library.document_dir(paper)
    (folder / "reading-log.md").write_text("prose")
    (folder / "extracted.json").write_text("{}")
    (folder / "figure-1.png").write_bytes(b"not a note")
    (folder / "scratch").mkdir()

    assert [entry.name for entry in library.notes(paper)] == [
        "extracted.json",
        "reading-log.md",
    ]


def test_the_document_is_not_a_note_about_itself(
    library: Library, tmp_path: Path
) -> None:
    """A markdown or JSON document sits in the same folder as its notes."""
    paper = _paper(["AI"], document_name="the-contract.json")
    filing = library.file_paper(paper, write_pdf(tmp_path / "src" / "raw.pdf", "text"))
    filed = library.db.get(filing.file_id)
    (library.document_dir(filed) / "notes.md").write_text("about the contract")

    assert [entry.name for entry in library.notes(filed)] == ["notes.md"]
    with pytest.raises(LibraryError, match="the document itself"):
        library.note_path(filed, "the-contract.json")


def test_a_bare_note_name_is_markdown(library: Library, tmp_path: Path) -> None:
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    paper = library.db.get(filing.file_id)

    assert library.note_path(paper, "reading-log").name == "reading-log.md"
    assert library.note_path(paper, "reading-log.md").name == "reading-log.md"
    assert library.note_path(paper, "extracted.json").name == "extracted.json"
    assert library.note_path(paper).name == "notes.md", "the default is unchanged"


@pytest.mark.parametrize("name", ["notes.txt", "notes.pdf", "v1.2"])
def test_a_note_that_is_not_markdown_or_json_is_refused(
    library: Library, tmp_path: Path, name: str
) -> None:
    """Renaming it would file it under a name its owner will not look for."""
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    paper = library.db.get(filing.file_id)

    with pytest.raises(LibraryError, match="a note is"):
        library.note_path(paper, name)


@pytest.mark.parametrize("name", ["../escape.md", "sub/note.md", "..", "."])
def test_a_note_name_cannot_lead_out_of_the_document_folder(
    library: Library, tmp_path: Path, name: str
) -> None:
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    paper = library.db.get(filing.file_id)

    with pytest.raises(LibraryError):
        library.note_path(paper, name)


def test_notes_of_any_name_travel_with_a_re_tagged_document(
    library: Library, tmp_path: Path
) -> None:
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    paper = library.db.get(filing.file_id)
    library.note_path(paper, "reading-log").write_text("why I saved this")

    library.retag(filing.file_id, ["Machine Learning", "Attention"])

    moved = library.db.get(filing.file_id)
    assert [entry.name for entry in library.notes(moved)] == ["reading-log.md"]
    assert library.note_path(moved, "reading-log").read_text() == "why I saved this"


def test_a_file_left_in_the_tree_is_reported(library: Library, tmp_path: Path) -> None:
    # A real folder in the tree can be written into, and the tree is rebuilt and
    # not backed up — so anything found there is surfaced rather than kept
    # quietly.
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    stray = filing.link_path.parent / "draft-thoughts.md"
    stray.write_text("written in the wrong place")

    litter = library.tree_litter()

    assert litter == [stray]


def test_finder_droppings_are_not_reported_as_litter(
    library: Library, tmp_path: Path
) -> None:
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    (filing.link_path.parent / ".DS_Store").write_bytes(b"\x00")

    assert library.tree_litter() == [], "the filesystem's litter is not the owner's"


def test_litter_reporting_does_not_walk_into_the_store(
    library: Library, tmp_path: Path
) -> None:
    # Walking through the link would report every document and every note as
    # litter, since they are real files at the other end of it.
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    paper = library.db.get(filing.file_id)
    library.note_path(paper).write_text("a note, safely in the store")

    assert library.tree_litter() == []


def test_a_file_left_in_the_tree_survives_a_rebuild(
    library: Library, tmp_path: Path
) -> None:
    # Reported, but not deleted: it is not this tool's to throw away.
    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    stray = filing.link_path.parent / "draft-thoughts.md"
    stray.write_text("keep me")

    library.rebuild_tree()

    assert stray.read_text() == "keep me"
    assert library.tree_litter() == [stray]


def test_a_failed_folder_delete_keeps_the_document(
    library: Library, tmp_path: Path
) -> None:
    """A document whose folder cannot be deleted must stay in the library.

    Dropping the row would leave the folder on disk with nothing pointing at
    it — invisible to every command, including the one that would delete it.
    """
    import os
    import stat

    filing = library.file_paper(
        _paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text")
    )
    directory = filing.store_path.parent
    os.chmod(directory, stat.S_IRUSR | stat.S_IXUSR)  # deletion will fail
    try:
        with pytest.raises(LibraryError, match="could not remove"):
            library.remove(filing.file_id)
    finally:
        os.chmod(directory, 0o755)

    assert library.db.get(filing.file_id) is not None, "the row must survive"
    assert filing.store_path.is_file(), "and so must the document"


def test_removing_one_of_two_look_alikes_keeps_the_other_linked(
    library: Library, tmp_path: Path
) -> None:
    # Same author, year and title, so they compete for one link name and the
    # loser gets an id appended. `rebuild_tree` names in file_id order, so the
    # ids are pinned: `first` takes the plain name, `second` the decorated one.
    # Removing `second` must not delete the plain link, which is `first`'s.
    first = library.file_paper(
        _paper(["AI"], file_id="aaaaaaaaaaaa"),
        write_pdf(tmp_path / "a" / "raw.pdf", "one"),
    )
    second = library.file_paper(
        _paper(["AI"], file_id="bbbbbbbbbbbb"),
        write_pdf(tmp_path / "b" / "raw.pdf", "two"),
    )
    library.rebuild_tree()
    assert len(list((library.tree_dir / "AI").iterdir())) == 2

    library.remove(second.file_id)

    survivors = list((library.tree_dir / "AI").iterdir())
    assert len(survivors) == 1, f"expected one link left, got {survivors}"
    assert survivors[0].is_symlink()
    assert os.path.realpath(survivors[0]) == os.path.realpath(
        library.document_dir(library.db.get(first.file_id))
    ), "the surviving link must point at the document that was kept"


def test_retagging_one_of_two_look_alikes_moves_the_right_one(
    library: Library, tmp_path: Path
) -> None:
    first = library.file_paper(
        _paper(["AI"], file_id="aaaaaaaaaaaa"),
        write_pdf(tmp_path / "a" / "raw.pdf", "one"),
    )
    second = library.file_paper(
        _paper(["AI"], file_id="bbbbbbbbbbbb"),
        write_pdf(tmp_path / "b" / "raw.pdf", "two"),
    )
    library.rebuild_tree()

    library.retag(second.file_id, ["Systems"])

    remaining = list((library.tree_dir / "AI").iterdir())
    assert len(remaining) == 1, f"first should stay put, found {remaining}"
    # Counting is not enough: the entry left behind must be the document that
    # was not re-tagged, still pointing at its own folder.
    assert os.path.realpath(remaining[0]) == os.path.realpath(
        library.document_dir(library.db.get(first.file_id))
    ), "the entry left in AI must be the untouched document's"

    moved = list((library.tree_dir / "Systems").iterdir())
    assert len(moved) == 1
    assert os.path.realpath(moved[0]) == os.path.realpath(
        library.document_dir(library.db.get(second.file_id))
    )


def test_a_crash_between_filing_and_recording_is_found_and_recovered(
    library: Library, tmp_path: Path
) -> None:
    """The window filing is built around, made recoverable.

    The file is put down before the row is written, on the reasoning that a
    folder with no row is the better half to be left holding. That only holds if
    something can find such a folder — otherwise the document is on disk and
    invisible to every command, and in move mode it is the only copy.
    """
    paper = _paper(["AI", "Transformers"])
    source = write_pdf(tmp_path / "src" / "raw.pdf", "text")

    # Exactly the crash: the file lands, the row never does.
    library.place_file(paper, source, FilingMode.MOVE)
    assert not source.exists(), "move mode: the store copy is the only one"
    assert library.db.count() == 0

    assert library.db.all_papers() == []
    assert [d.name for d in library.orphans()] == [paper.store_name]

    adopted = library.adopt(library.document_dir(paper))

    assert adopted.file_id == paper.file_id, "the id in the folder name is kept"
    assert adopted.tags == ["AI", "Transformers"], "and so are the tags"
    assert library.store_path(adopted).is_file()
    assert library.db.find_by_content_hash(adopted.content_hash) == paper.file_id
    assert library.orphans() == [], "nothing orphaned once adopted"


def test_an_adopted_document_is_reachable_through_the_tree(
    library: Library, tmp_path: Path
) -> None:
    paper = _paper(["AI"])
    library.place_file(paper, write_pdf(tmp_path / "src" / "raw.pdf", "text"), FilingMode.MOVE)

    adopted = library.adopt(library.document_dir(paper))

    links = [p for p in library.tree_dir.rglob("*") if p.is_symlink()]
    assert len(links) == 1
    assert os.path.realpath(links[0]) == os.path.realpath(library.document_dir(adopted))


def test_a_healthy_library_reports_no_orphans(library: Library, tmp_path: Path) -> None:
    library.file_paper(_paper(["AI"]), write_pdf(tmp_path / "src" / "raw.pdf", "text"))

    assert library.orphans() == []
    assert library.missing_files() == []


def test_a_folder_that_is_not_a_store_folder_is_refused(
    library: Library, tmp_path: Path
) -> None:
    library.store_dir.mkdir(parents=True, exist_ok=True)
    stray = library.store_dir / "notes-i-put-here"
    stray.mkdir()

    assert [d.name for d in library.orphans()] == ["notes-i-put-here"]
    with pytest.raises(LibraryError, match="not a store folder"):
        library.adopt(stray)


def test_an_ambiguous_folder_is_refused_rather_than_guessed(
    library: Library, tmp_path: Path
) -> None:
    paper = _paper(["AI"])
    library.place_file(paper, write_pdf(tmp_path / "src" / "raw.pdf", "text"), FilingMode.MOVE)
    # A second PDF beside it: which one is the document is not this tool's guess.
    write_pdf(library.document_dir(paper) / "appendix.pdf", "extra")

    with pytest.raises(LibraryError, match="expected exactly one"):
        library.adopt(library.document_dir(paper))


# ---- a real file where a link belongs --------------------------------------


def test_a_real_file_in_the_tree_is_not_deleted_to_make_room_for_a_link(
    library: Library, tmp_path: Path
) -> None:
    """The tree holds links, but the folders in it are real and can be written to.

    Replacing whatever sits at the link path would delete somebody's work to
    make room for something the database can rebuild in a second.
    """
    source = write_pdf(tmp_path / "raw" / "a.pdf", "attention")
    paper = _paper(["AI"])
    filing = library.file_paper(paper, source)
    link = filing.link_path
    link.unlink()
    link.write_text("a draft that is not a link", encoding="utf-8")

    library.rebuild_tree()

    assert link.read_text() == "a draft that is not a link"


def test_the_document_is_still_linked_beside_the_file_that_blocked_it(
    library: Library, tmp_path: Path
) -> None:
    # Refusing to delete the file must not mean the document silently loses its
    # place in the tree.
    source = write_pdf(tmp_path / "raw" / "a.pdf", "attention")
    paper = _paper(["AI"])
    filing = library.file_paper(paper, source)
    filing.link_path.unlink()
    filing.link_path.write_text("not a link", encoding="utf-8")

    library.rebuild_tree()

    links = [
        entry
        for entry in (library.tree_dir / "AI").rglob("*")
        if entry.is_symlink()
    ]
    assert len(links) == 1
    assert os.path.realpath(links[0]) == str(library.document_dir(paper).resolve())
    assert paper.file_id in links[0].name, "the id-decorated name"


def test_a_stale_link_is_still_replaced(library: Library, tmp_path: Path) -> None:
    # Only real files are protected: a link left pointing at an old answer is
    # this tool's to fix, and a rebuild that could not would never converge.
    source = write_pdf(tmp_path / "raw" / "a.pdf", "attention")
    filing = library.file_paper(_paper(["AI"]), source)
    filing.link_path.unlink()
    filing.link_path.symlink_to(tmp_path / "somewhere-else")

    library.rebuild_tree()

    assert os.path.realpath(filing.link_path) != str(tmp_path / "somewhere-else")


# ---- backups ---------------------------------------------------------------


def test_a_backup_holds_the_store_and_the_database(
    library: Library, tmp_path: Path
) -> None:
    # They are only useful together: the store alone is documents nothing can
    # find, and the database alone is a catalogue of files that are gone.
    paper = _paper(["AI"])
    library.file_paper(paper, write_pdf(tmp_path / "raw" / "a.pdf", "attention"))

    report = library.backup(tmp_path / "backup")

    assert (report.destination / "papers.duckdb").is_file()
    assert (report.destination / "store" / paper.store_name).is_dir()
    assert report.documents == 1


def test_a_backed_up_database_can_be_opened_on_its_own(
    library: Library, tmp_path: Path
) -> None:
    """A file copy of a busy database can restore short of its most recent rows.

    So the copy goes through DuckDB, and this is what says it did.
    """
    from sortyourpaperya.db import PaperDb

    paper = _paper(["AI"])
    library.file_paper(paper, write_pdf(tmp_path / "raw" / "a.pdf", "attention"))

    report = library.backup(tmp_path / "backup")
    library.close()

    with PaperDb(report.database) as restored:
        assert restored.get(paper.file_id) is not None


def test_bibliographies_are_backed_up(library: Library, tmp_path: Path) -> None:
    """A bibliography is hand-made and derivable from nothing.

    Leaving it out would make this command the thing that loses it.
    """
    from sortyourpaperya import bib

    bib.create(library.root, "Thesis")

    report = library.backup(tmp_path / "backup")

    assert (report.destination / "bibs" / "thesis" / "bib.toml").is_file()
    assert report.bibliographies == 1


def test_notes_kept_beside_a_document_are_backed_up(
    library: Library, tmp_path: Path
) -> None:
    paper = _paper(["AI"])
    library.file_paper(paper, write_pdf(tmp_path / "raw" / "a.pdf", "attention"))
    library.note_path(paper).write_text("# mine\n", encoding="utf-8")

    report = library.backup(tmp_path / "backup")

    assert (
        report.destination / "store" / paper.store_name / "notes.md"
    ).read_text() == "# mine\n"


def test_the_tree_is_not_backed_up(library: Library, tmp_path: Path) -> None:
    # It is nothing but links and `sortyourpaperya tree` rebuilds it exactly.
    library.file_paper(_paper(["AI"]), write_pdf(tmp_path / "raw" / "a.pdf", "x"))

    report = library.backup(tmp_path / "backup")

    assert not (report.destination / "tree").exists()


def test_the_database_is_copied_before_the_store(
    library: Library, tmp_path: Path, monkeypatch
) -> None:
    """Which half is behind decides what a backup taken mid-pass is worth.

    A folder with no row is an orphan, which `fsck --adopt` brings back. A row
    with no folder is a document that no longer exists anywhere. Taking the
    database first makes the recoverable mistake the only one available.
    """
    import shutil as shutil_module

    import sortyourpaperya.library as library_module

    library.file_paper(_paper(["AI"]), write_pdf(tmp_path / "raw" / "a.pdf", "first"))
    real_copytree = shutil_module.copytree

    def file_another_one_midway(*args, **kwargs):
        library.file_paper(
            _paper(["AI"]), write_pdf(tmp_path / "raw" / "b.pdf", "second")
        )
        return real_copytree(*args, **kwargs)

    monkeypatch.setattr(library_module.shutil, "copytree", file_another_one_midway)
    report = library.backup(tmp_path / "backup")
    library.close()

    restored = Library(report.destination)
    try:
        assert restored.db.count() == 1, "the database was taken after the store"
        assert len(restored.orphans()) == 1, "the late document should be adoptable"
        assert restored.missing_files() == [], "no row may point at a missing file"
    finally:
        restored.close()


def test_backing_up_into_the_library_is_refused(
    library: Library, tmp_path: Path
) -> None:
    # It would copy the backup into itself.
    with pytest.raises(LibraryError, match="inside the library"):
        library.backup(library.root / "backup")


def test_backing_up_over_something_is_refused(
    library: Library, tmp_path: Path
) -> None:
    destination = tmp_path / "backup"
    destination.mkdir()
    (destination / "important.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(LibraryError, match="not empty"):
        library.backup(destination)

    assert (destination / "important.txt").exists()

"""Bibliographies: the record, the .bib generated from it, and what cites what."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

from sortyourpaperya import bib
from sortyourpaperya.db import Paper
from sortyourpaperya.naming import cite_key, new_id


def make_paper(**overrides) -> Paper:
    paper_id = overrides.pop("file_id", new_id())
    fields = {
        "content_hash": "hash-" + paper_id,
        "store_name": paper_id,
        "document_name": "paper.pdf",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "year": 2017,
    }
    fields.update(overrides)
    return Paper(file_id=paper_id, **fields)


def test_a_new_bibliography_is_a_folder_with_both_files(tmp_path: Path) -> None:
    made = bib.create(tmp_path, "PhD Thesis")

    assert made.path == tmp_path / "bibs" / "phd-thesis"
    assert made.record_path.is_file() and made.bib_path.is_file()
    # The name as given is kept: a slug is a filename, not a title.
    assert made.name == "PhD Thesis"
    assert bib.load(made.path).id == made.id


def test_a_bibliography_is_found_by_slug_or_by_id(tmp_path: Path) -> None:
    """An agent that recorded the id last week must still find it."""
    made = bib.create(tmp_path, "Thesis")

    assert bib.open_bib(tmp_path, "thesis").path == made.path
    assert bib.open_bib(tmp_path, made.id).path == made.path
    with pytest.raises(bib.BibError, match="no bibliography"):
        bib.open_bib(tmp_path, "nothing-of-the-sort")


def test_a_name_that_slugifies_to_nothing_is_refused(tmp_path: Path) -> None:
    with pytest.raises(bib.BibError):
        bib.create(tmp_path, "!!!")


def test_a_second_bibliography_of_the_same_name_is_refused(tmp_path: Path) -> None:
    """Silently reopening it would let `bib init` empty one that has entries."""
    bib.create(tmp_path, "Thesis")
    with pytest.raises(bib.BibError, match="already exists"):
        bib.create(tmp_path, "thesis")


def test_a_cited_document_is_written_to_both_files(tmp_path: Path) -> None:
    made = bib.create(tmp_path, "Thesis")
    paper = make_paper()

    made.add(bib.source_from_paper(paper, {"doi": "10.1000/xyz", "journal": "NeurIPS"}))
    made.save()

    record = tomllib.loads(made.record_path.read_text())
    (source,) = record["source"]
    assert source["key"] == "vaswani2017attention"
    assert source["file_id"] == paper.file_id
    assert source["author"] == ["Ashish Vaswani", "Noam Shazeer"]

    written = made.bib_path.read_text()
    assert "@article{vaswani2017attention," in written
    # BibTeX has no lists: names are one field joined by ` and `.
    assert "author = {Ashish Vaswani and Noam Shazeer}," in written
    assert "doi = {10.1000/xyz}," in written
    assert "year = {2017}," in written


def test_only_attributes_that_name_a_bibtex_field_are_carried(tmp_path: Path) -> None:
    """A reader's own notes on a document are not part of a citation."""
    source = bib.source_from_paper(
        make_paper(), {"doi": "10.1000/xyz", "verdict": "worth re-reading", "read-on": "May"}
    )

    assert source.fields["doi"] == "10.1000/xyz"
    assert "verdict" not in source.fields and "read-on" not in source.fields


def test_an_empty_attribute_is_left_out(tmp_path: Path) -> None:
    """A blank journal renders as a citation with a blank journal."""
    source = bib.source_from_paper(make_paper(), {"journal": "  ", "doi": None})

    assert "journal" not in source.fields and "doi" not in source.fields
    assert source.entry_type == "misc"


def test_the_entry_type_follows_the_fields_and_is_overridable() -> None:
    assert bib.source_from_paper(make_paper(), {"journal": "NeurIPS"}).entry_type == "article"
    assert bib.source_from_paper(make_paper(), {"booktitle": "ICML"}).entry_type == "inproceedings"
    assert bib.source_from_paper(make_paper(), {"school": "MIT"}).entry_type == "phdthesis"
    # A document filed by this tool is not assumed to be a paper.
    assert bib.source_from_paper(make_paper(), {}).entry_type == "misc"
    assert bib.source_from_paper(make_paper(), {}, entry_type="online").entry_type == "online"


def test_two_papers_wanting_one_key_get_different_ones(tmp_path: Path) -> None:
    """Same author, same year, same first word — which BibTeX answers with a letter."""
    made = bib.create(tmp_path, "Thesis")

    first = made.add(bib.source_from_paper(make_paper(), {}))
    second = made.add(bib.source_from_paper(make_paper(), {}))
    third = made.add(bib.source_from_paper(make_paper(), {}))

    assert [first.key, second.key, third.key] == [
        "vaswani2017attention",
        "vaswani2017attentionb",
        "vaswani2017attentionc",
    ]


def test_a_document_already_cited_is_found_by_its_id(tmp_path: Path) -> None:
    made = bib.create(tmp_path, "Thesis")
    paper = make_paper()
    made.add(bib.source_from_paper(paper, {}))
    made.save()

    reopened = bib.open_bib(tmp_path, "thesis")
    assert reopened.source_for(paper.file_id).key == "vaswani2017attention"
    assert reopened.source_for("nothing-like-it") is None


def test_tex_characters_in_a_title_survive_the_round_trip(tmp_path: Path) -> None:
    """A title carrying `&` or `_` is text, and must not read as an instruction."""
    made = bib.create(tmp_path, "Thesis")
    paper = make_paper(title="Cost & Effect: 50% of a_b", authors=["Ada Lovelace"])

    made.add(bib.source_from_paper(paper, {}))
    made.save()

    # The record keeps it verbatim; only the projection is escaped.
    (source,) = tomllib.loads(made.record_path.read_text())["source"]
    assert source["title"] == "Cost & Effect: 50% of a_b"
    assert r"title = {{Cost \& Effect: 50\% of a\_b}}," in made.bib_path.read_text()


def test_a_title_is_braced_so_a_style_cannot_lowercase_it(tmp_path: Path) -> None:
    made = bib.create(tmp_path, "Thesis")
    made.add(bib.source_from_paper(make_paper(), {"journal": "NeurIPS"}))
    made.save()

    written = made.bib_path.read_text()
    assert "title = {{Attention Is All You Need}}," in written
    # Only the title. Everything else is a single pair.
    assert "journal = {NeurIPS}," in written


def test_the_bib_is_rebuilt_from_a_hand_edited_record(tmp_path: Path) -> None:
    """The TOML is the truth, which is only true if editing it is what works."""
    made = bib.create(tmp_path, "Thesis")
    made.add(bib.source_from_paper(make_paper(), {}))
    made.save()

    record = made.record_path.read_text().replace(
        'type = "misc"', 'type = "article"'
    ) + '\njournal = "Journal of Corrections"\n'
    made.record_path.write_text(record)

    bib.open_bib(tmp_path, "thesis").save()

    written = made.bib_path.read_text()
    assert "@article{vaswani2017attention," in written
    assert "journal = {Journal of Corrections}," in written


def test_a_source_added_by_hand_needs_no_document(tmp_path: Path) -> None:
    """A bibliography cites what a paper cites, not only what this library holds."""
    made = bib.create(tmp_path, "Thesis")
    made.record_path.write_text(
        made.record_path.read_text()
        + '\n[[source]]\nkey = "knuth1984tex"\ntype = "book"\n'
        'title = "The TeXbook"\nauthor = ["Donald Knuth"]\nyear = 1984\n'
    )

    reopened = bib.open_bib(tmp_path, "thesis")
    reopened.save()

    assert reopened.sources[0].file_id is None
    assert "@book{knuth1984tex," in made.bib_path.read_text()


def test_a_record_with_no_key_is_refused(tmp_path: Path) -> None:
    """An entry with no key is not an entry, and would be written out broken."""
    made = bib.create(tmp_path, "Thesis")
    made.record_path.write_text(
        made.record_path.read_text() + '\n[[source]]\ntitle = "Nameless"\n'
    )

    with pytest.raises(bib.BibError, match="no `key`"):
        bib.load(made.path)


def test_a_broken_bibliography_does_not_hide_the_others(tmp_path: Path) -> None:
    """One unreadable record must not stop the rest being listed or picked from."""
    bib.create(tmp_path, "Good One")
    broken = bib.create(tmp_path, "Broken")
    broken.record_path.write_text("this is not toml = = =")

    assert [entry.slug for entry in bib.all_bibs(tmp_path)] == ["good-one"]


def test_a_document_with_nothing_known_is_cited_by_its_id() -> None:
    """A bare year names nothing, so it is not a key."""
    paper = make_paper(title=None, authors=[], year=2020)

    assert bib.source_from_paper(paper, {}).key == paper.file_id


def test_a_citation_key_is_the_spelling_a_reference_manager_produces() -> None:
    assert (
        cite_key(
            authors=["Ashish Vaswani"],
            year=2017,
            title="Attention Is All You Need",
            fallback="x",
        )
        == "vaswani2017attention"
    )
    # A leading stop word names nothing, so the next word is taken.
    assert cite_key(authors=["Ada Lovelace"], year=1843, title="On the Engine", fallback="x") == "lovelace1843engine"
    # A hyphenated surname is run together: a key is typed, not read.
    assert cite_key(authors=["Jean-Luc Picard"], year=1999, title="Warp", fallback="x") == "picard1999warp"


# ---- linking the folder somewhere it is written against ---------------------


def test_a_bibliography_is_linked_into_a_manuscript_directory(tmp_path: Path) -> None:
    """One name reaching the record, the .bib, the notes, and the books."""
    made = bib.create(tmp_path, "Thesis")
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()

    link = bib.link_into(made, manuscript)

    assert link == manuscript / "thesis"
    assert link.is_symlink() and link.resolve() == made.path.resolve()
    assert (link / "references.bib").is_file()


def test_the_link_is_absolute_so_the_manuscript_can_move(tmp_path: Path) -> None:
    """The two ends are independent trees and move for unrelated reasons."""
    made = bib.create(tmp_path, "Thesis")
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()

    link = bib.link_into(made, manuscript)

    assert Path(os.readlink(link)).is_absolute()


def test_linking_twice_is_the_wanted_state_not_an_error(tmp_path: Path) -> None:
    made = bib.create(tmp_path, "Thesis")
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()

    first = bib.link_into(made, manuscript)

    assert bib.link_into(made, manuscript) == first


def test_nothing_in_the_way_is_replaced(tmp_path: Path) -> None:
    """Deciding someone else's file is stale is not this tool's call."""
    made = bib.create(tmp_path, "Thesis")
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    (manuscript / "thesis").write_text("mine", encoding="utf-8")

    with pytest.raises(bib.BibError, match="already exists"):
        bib.link_into(made, manuscript)
    assert (manuscript / "thesis").read_text() == "mine"


def test_a_link_pointing_elsewhere_is_refused(tmp_path: Path) -> None:
    made = bib.create(tmp_path, "Thesis")
    other = bib.create(tmp_path, "Paper")
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    (manuscript / "thesis").symlink_to(other.path)

    with pytest.raises(bib.BibError, match="link to something else"):
        bib.link_into(made, manuscript)


# ---- books, which you write alongside ---------------------------------------


def book(**overrides) -> Paper:
    fields = {"title": "The TeXbook", "authors": ["Donald Knuth"], "year": 1984}
    fields.update(overrides)
    return make_paper(**fields)


def shelve(made, paper, store: Path):
    """Cite `paper` as a book and shelve it, the way `bib add` does."""
    source = made.add(
        bib.source_from_paper(paper, {"publisher": "Addison-Wesley"})
    )
    folder = store / paper.store_name
    folder.mkdir(parents=True, exist_ok=True)
    return bib.shelve(made, source, folder, paper.document_name)


def test_a_cited_book_is_linked_in_under_its_author_and_year(tmp_path: Path) -> None:
    made = bib.create(tmp_path, "Thesis")
    paper = book()

    link = shelve(made, paper, tmp_path / "store")

    assert link.parent.name == "knuth_1984"
    assert link.name == paper.document_name
    # At the document's folder, as the tree links: the book and what is beside it.
    assert link.is_symlink() and link.resolve().is_dir()


def test_only_a_book_is_shelved(tmp_path: Path) -> None:
    """A paper is read once and cited; a book you go back to while writing."""
    assert bib.source_from_paper(book(), {"publisher": "X"}).entry_type == bib.SHELVED_TYPE
    assert bib.source_from_paper(make_paper(), {}).entry_type != bib.SHELVED_TYPE


def test_a_second_book_by_one_author_in_one_year_adds_its_title(tmp_path: Path) -> None:
    made = bib.create(tmp_path, "Thesis")
    store = tmp_path / "store"

    first = shelve(made, book(), store)
    second = shelve(made, book(title="Concrete Mathematics"), store)

    assert first.parent.name == "knuth_1984"
    assert second.parent.name == "knuth_1984_concrete-mathematics"


def test_a_book_nothing_tells_apart_is_refused(tmp_path: Path) -> None:
    """One folder would quietly hold two different books."""
    made = bib.create(tmp_path, "Thesis")
    store = tmp_path / "store"
    shelve(made, book(), store)
    shelve(made, book(), store)

    with pytest.raises(bib.BibError, match="nothing in the record tells them apart"):
        shelve(made, book(), store)


def test_a_book_with_no_author_is_shelved_under_its_citation_key(tmp_path: Path) -> None:
    """A year alone names nothing, and a key is unique within a bibliography."""
    made = bib.create(tmp_path, "Thesis")

    link = shelve(made, book(authors=[], title="A Field Manual"), tmp_path / "store")

    assert link.parent.name == "1984field"


# ---- notes -------------------------------------------------------------------


def test_a_bibliography_takes_notes_on_the_manuscript(tmp_path: Path) -> None:
    made = bib.create(tmp_path, "Thesis")

    assert made.note_path() == made.path / "notes.md"
    assert made.note_path("outline") == made.path / "outline.md"
    assert made.notes() == []

    made.note_path().write_text("# Thesis\n", encoding="utf-8")
    assert made.notes() == [made.path / "notes.md"]


def test_the_record_and_the_bib_are_never_mistaken_for_notes(tmp_path: Path) -> None:
    """Neither is a note format, so nothing has to be excluded by name."""
    made = bib.create(tmp_path, "Thesis")

    assert made.notes() == []


def test_a_note_on_a_cited_source_is_named_by_its_key(tmp_path: Path) -> None:
    made = bib.create(tmp_path, "Thesis")
    source = made.add(bib.source_from_paper(make_paper(), {}))

    assert made.source_note_path(source) == (
        made.path / "notes" / "vaswani2017attention.md"
    )
    # A second one carries the key too: they share a folder, and `outline.md`
    # in there would say nothing about whose outline it is.
    assert made.source_note_path(source, "ch3").name == "vaswani2017attention-ch3.md"


def test_a_source_sees_only_its_own_notes(tmp_path: Path) -> None:
    made = bib.create(tmp_path, "Thesis")
    one = made.add(bib.source_from_paper(make_paper(), {}))
    two = made.add(bib.source_from_paper(book(), {}))

    for path in (made.source_note_path(one), made.source_note_path(one, "ch3"),
                 made.source_note_path(two)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    assert [p.name for p in made.source_notes(one)] == [
        "vaswani2017attention-ch3.md",
        "vaswani2017attention.md",
    ]
    assert [p.name for p in made.source_notes(two)] == ["knuth1984texbook.md"]


def test_a_bib_note_follows_the_same_rules_as_a_document_note(tmp_path: Path) -> None:
    made = bib.create(tmp_path, "Thesis")

    assert made.note_path("extracted.json").name == "extracted.json"
    with pytest.raises(bib.BibError, match="a note is"):
        made.note_path("outline.txt")
    with pytest.raises(bib.BibError, match="not a path"):
        made.note_path("../escape.md")


# ---- the records, read backwards --------------------------------------------


def test_citations_say_which_bibliographies_cite_which_documents(tmp_path: Path) -> None:
    thesis = bib.create(tmp_path, "PhD Thesis")
    review = bib.create(tmp_path, "Review 2026")
    cited, uncited = make_paper(), make_paper()

    for made in (thesis, review):
        made.add(bib.source_from_paper(cited, {}))
        made.save()

    found = bib.citations(tmp_path)

    assert sorted(c.bib_slug for c in found[cited.file_id]) == ["phd-thesis", "review-2026"]
    assert all(c.key == "vaswani2017attention" for c in found[cited.file_id])
    assert uncited.file_id not in found
    assert bib.citations_of(tmp_path, uncited.file_id) == []


def test_a_hand_edited_record_is_answered_with_nothing_to_rebuild(tmp_path: Path) -> None:
    """Which is the whole reason nothing is stored: it cannot be stale."""
    made = bib.create(tmp_path, "Thesis")
    paper = make_paper()

    made.record_path.write_text(
        made.record_path.read_text()
        + f'\n[[source]]\nkey = "byhand"\ntype = "misc"\n'
        f'file_id = "{paper.file_id}"\ntitle = "Added By Hand"\n'
    )

    assert [c.key for c in bib.citations_of(tmp_path, paper.file_id)] == ["byhand"]


def test_a_source_that_is_not_in_the_library_cites_nothing_in_it(tmp_path: Path) -> None:
    """A book cited by hand has no file_id, so it names no document here."""
    made = bib.create(tmp_path, "Thesis")
    made.record_path.write_text(
        made.record_path.read_text()
        + '\n[[source]]\nkey = "knuth1984tex"\ntype = "book"\ntitle = "The TeXbook"\n'
    )

    assert bib.citations(tmp_path) == {}


def test_a_citation_carries_what_a_caller_would_use_to_find_it(tmp_path: Path) -> None:
    made = bib.create(tmp_path, "PhD Thesis")
    paper = make_paper()
    made.add(bib.source_from_paper(paper, {}))
    made.save()

    (citation,) = bib.citations_of(tmp_path, paper.file_id)

    assert citation.bib_slug == "phd-thesis"
    assert citation.bib_id == made.id
    assert citation.bib_name == "PhD Thesis"

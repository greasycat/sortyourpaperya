"""Bibliographies: the record, the .bib generated from it, and what cites what."""

from __future__ import annotations

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

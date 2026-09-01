from __future__ import annotations

from pathlib import Path

import pytest
from conftest import write_pdf, write_scanned_pdf

from sortyourpaperya.extract import ExtractionError, extract_paper_text


def test_extracts_the_text_layer(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "a.pdf", "Attention Is All You Need")

    paper = extract_paper_text(path, "id1", page_cutoff=1)

    assert "Attention Is All You Need" in paper.text
    assert paper.has_text_layer
    assert paper.pages_read == 1


def test_a_scanned_pdf_opens_but_reports_no_text_layer(tmp_path: Path) -> None:
    path = write_scanned_pdf(tmp_path / "scan.pdf")

    paper = extract_paper_text(path, "id1", page_cutoff=1)

    assert not paper.has_text_layer


def test_an_unreadable_pdf_is_an_extraction_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf at all")

    with pytest.raises(ExtractionError):
        extract_paper_text(path, "id1", page_cutoff=1)


# ---- reading a document to be read ------------------------------------------


def _pages(path: Path, count: int, tmp_path: Path) -> Path:
    """A PDF of `count` pages, each saying which one it is."""
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for number in range(1, count + 1):
        writer.append(
            PdfReader(str(write_pdf(tmp_path / f"p{number}.pdf", f"page {number} here")))
        )
    writer.write(str(path))
    return path


def test_reads_every_page_by_default(tmp_path: Path) -> None:
    from sortyourpaperya.extract import read_document

    document = read_document(_pages(tmp_path / "m.pdf", 4, tmp_path))

    assert document.total_pages == 4
    assert len(document.pages) == 4
    assert document.first_page == 1
    assert "page 1 here" in document.text and "page 4 here" in document.text


def test_a_range_reads_only_those_pages(tmp_path: Path) -> None:
    from sortyourpaperya.extract import read_document

    document = read_document(_pages(tmp_path / "m.pdf", 4, tmp_path), 2, 3)

    assert [p for p in document.pages] == ["page 2 here", "page 3 here"]
    assert document.first_page == 2
    assert document.total_pages == 4, "the whole document is still reported"


def test_a_range_past_the_end_is_clipped_not_refused(tmp_path: Path) -> None:
    """Naming a page past the end is a reasonable way to mean "the rest"."""
    from sortyourpaperya.extract import read_document

    document = read_document(_pages(tmp_path / "m.pdf", 3, tmp_path), 2, 99)

    assert len(document.pages) == 2
    assert document.total_pages == 3


def test_a_first_page_past_the_end_is_an_error(tmp_path: Path) -> None:
    """An empty answer would read as a document with nothing in it."""
    from sortyourpaperya.extract import ExtractionError, read_document

    with pytest.raises(ExtractionError, match="has 3 page"):
        read_document(_pages(tmp_path / "m.pdf", 3, tmp_path), 9)


def test_page_zero_is_an_error(tmp_path: Path) -> None:
    from sortyourpaperya.extract import ExtractionError, read_document

    with pytest.raises(ExtractionError, match="numbered from 1"):
        read_document(_pages(tmp_path / "m.pdf", 2, tmp_path), 0)


def test_a_scan_reports_no_text_layer_rather_than_failing(tmp_path: Path) -> None:
    from sortyourpaperya.extract import read_document

    document = read_document(write_scanned_pdf(tmp_path / "scan.pdf"))

    assert not document.has_text_layer
    assert document.total_pages == 1


def test_reading_keeps_the_lines_the_page_has(tmp_path: Path) -> None:
    """The difference from what ingest reads, which collapses to one line."""
    from sortyourpaperya.extract import extract_paper_text, read_document

    path = _pages(tmp_path / "m.pdf", 2, tmp_path)

    assert "\n" in read_document(path).text
    assert "\n" not in extract_paper_text(path, "id1", page_cutoff=2).text

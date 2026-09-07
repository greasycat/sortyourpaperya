"""Reading a PDF into text.

Two readings, because they are read by different things. `extract_paper_text`
takes the first pages and collapses them to one line: it is what goes into a
request, where layout is noise that costs money. `read_document` keeps the lines
the document has and takes as many pages as asked for: it is what a person or an
agent reads, where layout is most of what makes a page legible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

_WHITESPACE = re.compile(r"\s+")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")


class ExtractionError(RuntimeError):
    """Raised when a PDF cannot be opened or read at all."""


@dataclass(frozen=True)
class PaperText:
    """Text pulled from one PDF, ready to hand to the model."""

    file_id: str
    path: Path
    text: str
    pages_read: int
    # Whether the text was read off rendered page images rather than a text
    # layer. Worth carrying: everything downstream is then a model's reading of
    # a picture, not the document's own words.
    from_page_images: bool = False

    @property
    def has_text_layer(self) -> bool:
        """Whether the PDF carried any extractable text.

        A scanned paper opens fine and yields nothing, which is a different
        problem from a corrupt file and is reported separately.
        """
        return bool(self.text)


@dataclass(frozen=True)
class DocumentText:
    """A document read to be read: its pages, as laid out, and where they sit."""

    pages: list[str]
    # 1-based number of the first page read, so a caller quoting from a range
    # can say which page a line came from.
    first_page: int
    total_pages: int

    @property
    def has_text_layer(self) -> bool:
        """Whether any page read carried extractable text.

        A scan opens fine and yields nothing, which is not a broken file. What
        it says is only in its pictures, and is not this module's to recover.
        """
        return any(page.strip() for page in self.pages)

    @property
    def text(self) -> str:
        """The pages run together, one blank line between them."""
        return "\n\n".join(self.pages).strip()


def read_document(path: Path, first: int = 1, last: int | None = None) -> DocumentText:
    """Read `path` from page `first` to page `last`, keeping its layout.

    `last` beyond the end is clipped rather than refused: asking for the rest of
    a document by naming a page past it is a reasonable thing to mean, and the
    result says how many pages there were.

    Raises:
        ExtractionError: if the PDF cannot be opened, or `first` is past its
            last page — which is not a clipping but an empty answer, and one a
            caller would otherwise read as a document with nothing in it.
    """
    if first < 1:
        raise ExtractionError(f"pages are numbered from 1, not {first}")
    try:
        reader = PdfReader(str(path))
        total = len(reader.pages)
        end = total if last is None else min(last, total)
        chunks = [page.extract_text() or "" for page in reader.pages[first - 1 : end]]
    except ExtractionError:
        raise
    except Exception as err:  # pypdf raises a wide range of parse errors
        raise ExtractionError(f"could not read {path.name}: {err}") from err

    if first > total:
        raise ExtractionError(
            f"{path.name} has {total} page(s); there is no page {first}"
        )
    return DocumentText(
        pages=[_tidy(chunk) for chunk in chunks],
        first_page=first,
        total_pages=total,
    )


def _tidy(text: str) -> str:
    """Clean up a page without reflowing it.

    Trailing spaces and long runs of blank lines are extraction artefacts and go.
    The line breaks themselves stay: they are the page, and guessing which of
    them were the document's and which the column's is how a paragraph gets
    mangled.
    """
    return _BLANK_RUN.sub("\n\n", _TRAILING_SPACE.sub("", text)).strip()


def extract_paper_text(path: Path, file_id: str, page_cutoff: int) -> PaperText:
    """Extract text from the first ``page_cutoff`` pages of ``path``.

    Raises:
        ExtractionError: if the PDF cannot be opened or its pages cannot be read.
    """
    try:
        reader = PdfReader(str(path))
        pages = reader.pages[:page_cutoff]
        chunks = [page.extract_text() or "" for page in pages]
    except ExtractionError:
        raise
    except Exception as err:  # pypdf raises a wide range of parse errors
        raise ExtractionError(f"could not read {path.name}: {err}") from err

    return PaperText(
        file_id=file_id,
        path=path,
        text=_normalize(" ".join(chunks)),
        pages_read=len(chunks),
    )


def _normalize(text: str) -> str:
    """Collapse the whitespace PDF extraction leaves behind."""
    return _WHITESPACE.sub(" ", text).strip()

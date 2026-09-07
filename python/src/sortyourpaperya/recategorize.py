"""Re-asking the model where one document belongs.

Ingest labels a document once, steered towards the paths the library already
uses so that two utility bills do not open two unrelated branches. That steering
is also how a document ends up somewhere wrong: a computational cognitive
science paper joins `Psychology/Research Methods` because the library holds two
research-methods documents and nothing closer. `sortyourpaperya retag` is the way back, and
this is what it asks.

Nothing here touches the database or the library. What the model needs is read
once by the caller and passed in, so the whole exchange — which waits on a
person, and can go around as many times as they like — happens with no
connection open. A command sitting on the write lock at a prompt would stop the
watcher, which waits 30 seconds and then fails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from .db import Paper
from .extract import ExtractionError, PaperText, extract_paper_text
from .llm import CategorySuggestion, LlmClient

log = logging.getLogger(__name__)


async def suggest(
    client: LlmClient,
    paper: Paper,
    store_path: Path,
    *,
    page_cutoff: int,
    existing_categories: Sequence[str] = (),
    rejected: Sequence[str] = (),
    guidance: str = "",
) -> CategorySuggestion:
    """Ask where `paper` belongs, avoiding anything in `rejected`.

    `guidance` is what the person watching said about this document — where
    they think it goes, or what the model keeps missing about it. One request.
    Raises `LlmError` if the model cannot be reached or says nothing usable.
    """
    return await client.suggest_category(
        PaperText(
            file_id=paper.file_id,
            path=store_path,
            text=describe(paper, store_path, page_cutoff),
            pages_read=0,
        ),
        current="/".join(paper.tags),
        existing_categories=existing_categories,
        rejected=rejected,
        guidance=guidance,
    )


def describe(paper: Paper, store_path: Path, page_cutoff: int) -> str:
    """What the model gets to read: what the library knows, then the document.

    The library's own record comes first and always. It is free, it is already
    the distilled version of this document, and it is the only thing left when
    the file cannot be read — a scan carries no text layer, and a document whose
    file is missing from the store carries nothing at all. Both still deserve an
    answer, and neither is worth a second request or a poppler dependency to
    get one.
    """
    known = [
        f"title: {paper.title}" if paper.title else "",
        f"authors: {', '.join(paper.authors)}" if paper.authors else "",
        f"year: {paper.year}" if paper.year else "",
        f"keywords: {', '.join(paper.keywords)}" if paper.keywords else "",
        f"filename: {paper.original_name}" if paper.original_name else "",
    ]
    sections = [line for line in known if line]

    page_text = _first_pages(store_path, paper.file_id, page_cutoff)
    if page_text:
        sections.append(f"\nfirst pages:\n{page_text}")
    return "\n".join(sections)


def _first_pages(store_path: Path, file_id: str, page_cutoff: int) -> str:
    """The document's own words, or nothing.

    Best effort on purpose: a document that cannot be read is still one the
    library has a record of, so failing to read it costs detail rather than the
    answer.
    """
    try:
        return extract_paper_text(store_path, file_id, page_cutoff).text
    except (ExtractionError, OSError) as err:
        log.debug("could not read %s for re-tagging: %s", store_path, err)
        return ""

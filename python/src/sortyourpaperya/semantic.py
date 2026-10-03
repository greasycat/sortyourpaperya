"""Search by meaning: what documents and their notes say, embedded and compared.

`find` matches words against what the library knows *about* a document -- its
title, authors, tags. This answers the other question, what is *in* one, and
answers it by meaning rather than by spelling, so "learning from reward" finds
the paper that only ever says "reinforcement".

Each source a document has -- the document's own text, and each of its notes --
is cut into chunks, the chunks are embedded by the OpenAI embeddings API, and the
vectors are kept in DuckDB beside everything else. A search embeds the query and
compares it with every chunk. Nothing here needs an extension or a second store.

A source is embedded once per fingerprint: the document's content hash, or the
note's bytes, together with the model and the chunking. A pass over a library
that has not changed costs a read of the fingerprints and nothing else, which is
what lets the watcher run one after every filing without anyone paying for it.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Protocol, Sequence

import duckdb

from . import extract
from .budget import Budget, BudgetExceeded, Limits, ledger_path
from .config import DEFAULT_LLM_MAX_RETRIES, DEFAULT_LLM_TIMEOUT_SECONDS, env_int
from .db import Paper
from .extract import ExtractionError
from .library import Library
from .llm import LlmError

log = logging.getLogger(__name__)

# Small, cheap, and good enough to tell a paper about attention from one about
# reward. `SYP_EMBEDDING_MODEL` swaps it; chunks from the old model are left out
# of searches and re-embedded by the next index pass.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Words per chunk, and how far each one starts after the last. The overlap is
# what keeps a sentence cut in half by a boundary findable from either side.
CHUNK_WORDS = 200
CHUNK_STRIDE = 160
# A "word" is whatever sits between spaces, and a PDF can hold a single one that
# is a page long -- an embedded font, a run of base64. Characters are capped too
# so no input can pass the model's token ceiling.
CHUNK_MAX_CHARS = 2000
# Bumped when the chunking above changes, so every source is cut again.
CHUNKER_VERSION = 1

# Inputs per embeddings request. The API takes 2048; requests are what the daily
# budget counts, so fewer and fuller ones keep a backfill from spending it.
EMBED_BATCH = 256

# The document's own text, as a source. Notes are named by their filename.
DOCUMENT = ""


# What embedding may spend in a rolling day. Its own ledger, not the chat one:
# an embedding token costs about a hundredth of a chat token, and the first index
# of a library of a couple of hundred papers is ~5M of them -- five times the
# chat ceiling, for about ten cents. Sharing the ceiling would either stop that
# backfill a fifth of the way in or need a chat ceiling too loose to catch
# anything. 20M is ~$0.40 at text-embedding-3-small's price.
DEFAULT_MAX_EMBEDDING_TOKENS_PER_DAY = 20_000_000


def embedding_budget() -> Budget:
    """The embedding ledger: `SYP_MAX_EMBEDDING_TOKENS_PER_DAY`, 0 for no ceiling.

    Tokens only. A request ceiling guards against a loop re-sending the same
    thing, and fingerprints already make an unchanged source cost nothing.
    """
    return Budget(
        ledger_path().with_name("embedding-spend.json"),
        Limits(
            requests_per_day=0,
            tokens_per_day=env_int(
                "SYP_MAX_EMBEDDING_TOKENS_PER_DAY", DEFAULT_MAX_EMBEDDING_TOKENS_PER_DAY
            ),
        ),
    )


def embedding_model() -> str:
    return (os.environ.get("SYP_EMBEDDING_MODEL") or "").strip() or DEFAULT_EMBEDDING_MODEL


@dataclass
class Chunk:
    position: int
    # 1-based page the chunk came from; None for a note, or for a scan whose
    # text is a model's reading of several pages at once.
    page: int | None
    text: str


@dataclass
class SearchHit:
    """The best place in one document that a query matched."""

    file_id: str
    title: str | None
    # `""` for the document itself, else the note's filename.
    source: str
    page: int | None
    text: str
    score: float


@dataclass
class IndexReport:
    """What one index pass embedded, skipped, and dropped."""

    # Sources embedded this pass, as `file_id/source`.
    indexed: list[str] = field(default_factory=list)
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)


# ---- chunking ---------------------------------------------------------------


def chunk_pages(pages: Sequence[str], first_page: int | None = 1) -> list[Chunk]:
    """Cut text into overlapping windows of words, each kept within its page.

    Within a page rather than across: a chunk that says which page it is from
    is one a reader can go and check, and a page is long enough that little is
    lost at its edges. `first_page=None` numbers nothing.
    """
    chunks: list[Chunk] = []
    for offset, page in enumerate(pages):
        words = page.split()
        number = None if first_page is None else first_page + offset
        start = 0
        while start < len(words):
            text = " ".join(words[start : start + CHUNK_WORDS])[:CHUNK_MAX_CHARS]
            chunks.append(Chunk(position=len(chunks), page=number, text=text))
            if start + CHUNK_WORDS >= len(words):
                break
            start += CHUNK_STRIDE
    return chunks


def fingerprint(identity: str, model: str) -> str:
    return f"{identity}:{model}:{CHUNKER_VERSION}"


# ---- the embeddings API -----------------------------------------------------


class Embedder(Protocol):
    @property
    def model(self) -> str: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAiEmbedder:
    """`Embedder` backed by the OpenAI embeddings API, inside the daily budget.

    Each request is reserved against the `Budget` before it is sent and its
    tokens recorded after, the same as every chat request, so a backfill of a
    large library is stopped by the same ceiling as a restart loop.
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        *,
        budget: Budget | None = None,
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
        timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key, max_retries=max_retries, timeout=timeout_seconds
        )
        self._model = model or embedding_model()
        self._budget = budget if budget is not None else embedding_budget()

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = list(texts[start : start + EMBED_BATCH])
            self._budget.check(f"embedding {len(batch)} chunk(s)")
            self._budget.record(requests=1)
            try:
                response = await self._client.embeddings.create(
                    model=self._model, input=batch
                )
            except Exception as err:  # surface SDK/transport failures uniformly
                raise LlmError(f"embedding request failed: {err}") from err
            total = getattr(getattr(response, "usage", None), "total_tokens", None)
            if isinstance(total, int) and total > 0:
                self._budget.record(requests=0, tokens=total)
            # Ordered by `index` rather than trusted to arrive in order: the
            # vectors are matched back to their chunks by position.
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors


class LazyOpenAiEmbedder:
    """An `OpenAiEmbedder` that resolves the key when a request needs it.

    For the watcher, for the reason `LazyOpenAiClient` gives: it must start and
    serve without a key, and pick up one `sypy login` stores while it runs.
    """

    def __init__(self, model: str | None = None, *, budget: Budget | None = None) -> None:
        # None is the embedding ledger, chosen by `OpenAiEmbedder` when built.
        self._model = model or embedding_model()
        self._budget = budget
        self._embedder: OpenAiEmbedder | None = None
        self._generation: int | None = None

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from .config import key_generation, resolve_api_key

        if self._embedder is None or self._generation != key_generation():
            self._generation = key_generation()
            self._embedder = OpenAiEmbedder(
                resolve_api_key(), self._model, budget=self._budget
            )
        return await self._embedder.embed(texts)


# ---- indexing ---------------------------------------------------------------

# `(done, total, what)`: sources handled out of those that need embedding, and a
# few words on what is happening now. Numbers rather than a line so a caller can
# draw a bar -- and so the same three fields cross the watcher's socket as-is.
Progress = Callable[[int, int, str], Awaitable[None]]


@dataclass
class _Pending:
    file_id: str
    source: str
    fingerprint: str
    chunks: list[Chunk]
    # What is embedded: the chunk with the document's title in front of it, so
    # a chunk that never names its subject still lands near it.
    inputs: list[str]


def _document_text(library: Library, paper: Paper) -> tuple[list[str], int | None] | None:
    """The document's pages and the number of the first, or None if unreadable.

    A scan has no text layer; what the model read off its pages at ingest is
    used instead, as one block with no page number -- it is one answer covering
    several pages, not a page each.
    """
    path = library.store_path(paper)
    if not path.is_file():
        return None
    try:
        document = extract.read_document(path)
    except ExtractionError as err:
        raise LlmError(str(err)) from err
    if document.has_text_layer:
        return document.pages, document.first_page
    banked = library.db.model_answers([paper.content_hash], max_age_ms=None).get(
        paper.content_hash
    )
    text = (banked.page_text or "").strip() if banked else ""
    return ([text] if text else []), None


def _note_identity(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


async def index_library(
    library: Library,
    embedder: Embedder,
    *,
    progress: Progress | None = None,
) -> IndexReport:
    """Embed every source whose fingerprint has changed, and drop vanished ones.

    Text is read and embedded with no database connection held, and each batch
    is written in one short burst -- the same shape as an ingest pass, for the
    same reason: a watcher that held the lock through minutes of requests would
    shut every other command out for all of them.
    """
    report = IndexReport()
    model = embedder.model
    held = {(f, s): fp for f, s, fp in library.db.text_fingerprints()}
    papers = library.db.all_papers()
    library.release()

    wanted: set[tuple[str, str]] = set()
    stale: list[tuple[Paper, str, str, Path | None]] = []
    for paper in papers:
        sources: list[tuple[str, str, Path | None]] = [
            (DOCUMENT, fingerprint(paper.content_hash, model), None)
        ]
        for note in library.notes(paper):
            try:
                sources.append((note.name, fingerprint(_note_identity(note), model), note))
            except OSError as err:
                report.failed.append((f"{paper.file_id}/{note.name}", str(err)))
        for source, print_, note in sources:
            wanted.add((paper.file_id, source))
            if held.get((paper.file_id, source)) == print_:
                report.unchanged += 1
            else:
                stale.append((paper, source, print_, note))

    vanished = [key for key in held if key not in wanted]
    for file_id, source in vanished:
        library.db.drop_text_source(file_id, source)
    report.removed = len(vanished)
    library.release()

    total = len(stale)

    async def tell(done: int, what: str) -> None:
        if progress is not None:
            await progress(done, total, what)

    await tell(0, "reading")
    batch: list[_Pending] = []
    for done, (paper, source, print_, note) in enumerate(stale):
        label = f"{paper.file_id}/{source}" if source else paper.file_id
        name = (paper.title or paper.original_name or paper.file_id)[:48]
        await tell(done, f"reading {source + ' of ' if source else ''}{name}")
        try:
            if note is None:
                read = _document_text(library, paper)
                library.release()
                if read is None:
                    report.failed.append((label, "missing from the store"))
                    continue
                pages, first = read
                chunks = chunk_pages(pages, first)
                heading = paper.title or paper.original_name or ""
            else:
                chunks = chunk_pages([note.read_text(encoding="utf-8", errors="replace")], None)
                heading = f"Notes on {paper.title or paper.original_name or ''}"
        except (LlmError, OSError) as err:
            report.failed.append((label, str(err)))
            continue
        inputs = [f"{heading}\n\n{chunk.text}" if heading else chunk.text for chunk in chunks]
        batch.append(_Pending(paper.file_id, source, print_, chunks, inputs))
        waiting = sum(len(p.inputs) for p in batch)
        if waiting >= EMBED_BATCH:
            await tell(done + 1, f"embedding {waiting} chunk(s)")
            await _flush(library, embedder, batch, report)
            batch = []
    if batch:
        await tell(total, f"embedding {sum(len(p.inputs) for p in batch)} chunk(s)")
        await _flush(library, embedder, batch, report)
    await tell(total, "done")
    return report


async def _flush(
    library: Library,
    embedder: Embedder,
    batch: list[_Pending],
    report: IndexReport,
) -> None:
    """Embed a batch of sources together, then write them in one burst.

    A failed request fails the sources in it and nothing else: they keep their
    old fingerprint, or none, so the next pass tries them again.
    """
    inputs = [text for pending in batch for text in pending.inputs]
    try:
        vectors = await embedder.embed(inputs) if inputs else []
    except (LlmError, BudgetExceeded) as err:
        for pending in batch:
            report.failed.append((_label(pending), str(err)))
        return
    cursor = 0
    for pending in batch:
        mine = vectors[cursor : cursor + len(pending.chunks)]
        cursor += len(pending.chunks)
        library.db.replace_text_chunks(
            pending.file_id,
            pending.source,
            pending.fingerprint,
            embedder.model,
            [(c.position, c.page, c.text, v) for c, v in zip(pending.chunks, mine)],
        )
        report.indexed.append(_label(pending))
        report.chunks += len(pending.chunks)
    library.release()


def _label(pending: _Pending) -> str:
    return f"{pending.file_id}/{pending.source}" if pending.source else pending.file_id


# ---- searching --------------------------------------------------------------


async def search(
    library: Library, embedder: Embedder, query: str, limit: int = 10
) -> list[SearchHit]:
    """The documents whose text or notes come closest to `query` in meaning."""
    query = query.strip()
    if not query:
        return []
    (vector,) = await embedder.embed([query])
    try:
        rows = library.db.nearest_chunks(vector, embedder.model, limit)
    except duckdb.CatalogException:
        # A library no writer has opened since this was added: nothing is
        # indexed, which is an empty answer rather than an error.
        return []
    hits = []
    for file_id, source, page, text, score in rows:
        paper = library.db.get(file_id)
        hits.append(
            SearchHit(
                file_id=file_id,
                title=(paper.title or paper.original_name) if paper else None,
                source=source,
                page=page,
                text=text,
                score=float(score),
            )
        )
    return hits

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

import asyncio
import contextlib
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
from .llm import LlmError, key_refused

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

# Inputs per embeddings request. The API takes 2048; fewer, fuller requests
# make a backfill quicker.
EMBED_BATCH = 256
# And characters per request, because the API also refuses a request over ~300k
# tokens in total. Characters stand in for tokens since no tokenizer is to hand,
# and a token is almost never shorter than a character -- so this holds for
# text that runs a token per character, CJK or base64, where counting inputs
# alone let 256 two-thousand-character chunks ask for ~600k tokens at once.
EMBED_BATCH_CHARS = 250_000

# The document's own text, as a source. Notes are named by their filename.
DOCUMENT = ""

# Suffixes on a fingerprint recorded for a document that was looked at and has
# nothing to embed. Recording it is what stops every pass reopening the same
# corrupt PDF to fail on it again; each suffix says what would change the
# answer, and the planning pass looks for exactly that. Only the fingerprint is
# recorded: whatever chunks the document already had are kept.
#
# The file would not parse. Its fingerprint carries its content hash, so a
# replaced file is a new fingerprint and is tried again by itself.
UNREADABLE = ":unreadable"
# The file is not in the store. Tried again once it is back.
MISSING = ":missing"
# A scan with no reading of its pages banked yet. Tried again once there is one.
UNREAD = ":unread"


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
        token_setting="SYP_MAX_EMBEDDING_TOKENS_PER_DAY",
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
    # Why the pass gave up early -- the network down, the budget spent -- and how
    # many sources it never got to. Every later request would have failed the
    # same way, each after its own retries and timeout.
    stopped: str | None = None
    left: int = 0


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


def _heading(paper: Paper) -> str:
    """What every chunk of a document is embedded under: its title."""
    return paper.title or paper.original_name or ""


def _heading_tag(paper: Paper) -> str:
    """The heading, short, for a fingerprint: a retitled document is re-embedded.

    Each input starts with the title, so the title is part of what a vector
    means -- one embedded under a title since corrected would keep the old one.
    """
    return hashlib.sha256(_heading(paper).encode()).hexdigest()[:8]


# ---- the embeddings API -----------------------------------------------------


class EmbeddingRejected(LlmError):
    """The API refused the input itself, so sending it again would not help.

    Distinct from a failure of the request -- the network, a 5xx, a timeout --
    which says nothing about the input and everything about the next request.
    """


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
        for batch in _requests(texts):
            self._budget.check(f"embedding {len(batch)} chunk(s)")
            self._budget.record(requests=1)
            try:
                response = await self._client.embeddings.create(
                    model=self._model, input=batch
                )
            except Exception as err:  # surface SDK/transport failures uniformly
                if getattr(err, "status_code", None) in (400, 413, 422):
                    raise EmbeddingRejected(f"embedding request refused: {err}") from err
                raise LlmError(f"embedding request failed: {err}") from err
            total = getattr(getattr(response, "usage", None), "total_tokens", None)
            if isinstance(total, int) and total > 0:
                self._budget.record(requests=0, tokens=total)
            # Ordered by `index` rather than trusted to arrive in order: the
            # vectors are matched back to their chunks by position.
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors


def _requests(texts: Sequence[str]) -> list[list[str]]:
    """`texts` split into requests within `EMBED_BATCH` and `EMBED_BATCH_CHARS`."""
    requests: list[list[str]] = []
    current: list[str] = []
    size = 0
    for text in texts:
        if current and (len(current) >= EMBED_BATCH or size + len(text) > EMBED_BATCH_CHARS):
            requests.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        requests.append(current)
    return requests


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
        try:
            return await self._embedder.embed(texts)
        except LlmError as err:
            key_refused(err.__cause__)
            raise


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


def _read_store(path: Path) -> extract.DocumentText | None:
    """The document in the store, or None if it is not there. Blocks: a thread's."""
    if not path.is_file():
        return None
    return extract.read_document(path)


def _note_identity(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _plan(
    library: Library,
    papers: list[Paper],
    held: dict[tuple[str, str], str],
    model: str,
    read_since: set[str],
    report: IndexReport,
) -> tuple[set[tuple[str, str]], list[tuple[Paper, str, str, Path | None]]]:
    """Which sources exist, and which of them need embedding. Blocks: a thread's.

    Every note is hashed and every recorded-missing document looked for, which
    is file I/O across the whole library -- off the watcher's loop, so the
    socket keeps answering while it happens.
    """
    wanted: set[tuple[str, str]] = set()
    stale: list[tuple[Paper, str, str, Path | None]] = []
    for paper in papers:
        tag = _heading_tag(paper)
        sources: list[tuple[str, str, Path | None]] = [
            (DOCUMENT, fingerprint(f"{paper.content_hash}.{tag}", model), None)
        ]
        for note in library.notes(paper):
            # Wanted before it is read: a note that cannot be read just now --
            # mid-save, or a permissions slip -- still exists, and must keep the
            # chunks it has rather than be dropped as gone and paid for again.
            wanted.add((paper.file_id, note.name))
            try:
                identity = f"{_note_identity(note)}.{tag}"
                sources.append((note.name, fingerprint(identity, model), note))
            except OSError as err:
                report.failed.append((f"{paper.file_id}/{note.name}", str(err)))
        for source, print_, note in sources:
            wanted.add((paper.file_id, source))
            if _settled(library, paper, held.get((paper.file_id, source)), print_, read_since):
                report.unchanged += 1
            else:
                stale.append((paper, source, print_, note))
    return wanted, stale


def _settled(
    library: Library, paper: Paper, held: str | None, print_: str, read_since: set[str]
) -> bool:
    """Whether what is recorded for a source still stands."""
    if held in (print_, print_ + UNREADABLE):
        return True
    if held == print_ + MISSING:
        return not library.store_path(paper).is_file()
    if held == print_ + UNREAD:
        return paper.content_hash not in read_since
    return False


async def index_library(
    library: Library,
    embedder: Embedder,
    *,
    progress: Progress | None = None,
    exclusive: asyncio.Lock | None = None,
) -> IndexReport:
    """Embed every source whose fingerprint has changed, and drop vanished ones.

    Text is read and embedded with no database connection held, and each batch
    is written in one short burst -- the same shape as an ingest pass, for the
    same reason: a watcher that held the lock through minutes of requests would
    shut every other command out for all of them.

    `exclusive` is the watcher's op lock, and it is held only for those bursts.
    Held for the whole pass instead, a first index -- minutes of requests -- had
    `status`, `doctor`, `search`, and every routed command hanging behind it
    with no timeout and nothing said. Every database visit is under it, since
    the watcher's own ingest pass shares the connection. File reads -- PDFs
    parsed, notes hashed -- go to a thread, so they do not stall the loop.
    """
    report = IndexReport()
    model = embedder.model

    async with _held(exclusive):
        held = {(f, s): fp for f, s, fp in library.db.text_fingerprints()}
        papers = library.db.all_papers()
        waiting = [
            paper.content_hash
            for paper in papers
            if held.get((paper.file_id, DOCUMENT), "").endswith(UNREAD)
        ]
        read_since = {
            content_hash
            for content_hash, answer in library.db.model_answers(
                waiting, max_age_ms=None
            ).items()
            if (answer.page_text or "").strip()
        }
        library.release()

    wanted, stale = await asyncio.to_thread(
        _plan, library, papers, held, model, read_since, report
    )

    vanished = [key for key in held if key not in wanted]
    async with _held(exclusive):
        for file_id, source in vanished:
            library.db.drop_text_source(file_id, source)
        # What the model read off each document's pages at ingest, fetched once
        # here rather than per scan: each visit waits for the op lock, which an
        # ingest pass holds through all its requests.
        banked = {
            content_hash: (answer.page_text or "").strip()
            for content_hash, answer in library.db.model_answers(
                [paper.content_hash for paper, source, _, _ in stale if source == DOCUMENT],
                max_age_ms=None,
            ).items()
        }
        library.release()
    report.removed = len(vanished)
    # `(file_id, fingerprint)` for documents with nothing to embed; see
    # `UNREADABLE`. Written in one burst at the end, for the same reason.
    marks: list[tuple[str, str]] = []

    total = len(stale)

    async def tell(done: int, what: str) -> None:
        if progress is not None:
            await progress(done, total, what)

    async def flush(done: int, batch: list[_Pending]) -> str | None:
        await tell(done, f"embedding {sum(len(p.inputs) for p in batch)} chunk(s)")
        return await _flush(library, embedder, batch, report, exclusive)

    await tell(0, "reading")
    batch: list[_Pending] = []
    stop: str | None = None
    for done, (paper, source, print_, note) in enumerate(stale):
        label = f"{paper.file_id}/{source}" if source else paper.file_id
        heading = _heading(paper)
        await tell(done, f"reading {source + ' of ' if source else ''}{heading[:48] or paper.file_id}")
        try:
            if note is None:
                document = await asyncio.to_thread(_read_store, library.store_path(paper))
                if document is None:
                    marks.append((paper.file_id, print_ + MISSING))
                    report.failed.append((label, "missing from the store"))
                    continue
                if document.has_text_layer:
                    chunks = chunk_pages(document.pages, document.first_page)
                else:
                    # A scan has no text layer; what the model read off its
                    # pages at ingest is used instead, as one block with no page
                    # number -- one answer covering several pages, not a page each.
                    text = banked.get(paper.content_hash, "")
                    if not text:
                        marks.append((paper.file_id, print_ + UNREAD))
                        continue
                    chunks = chunk_pages([text], None)
            else:
                text = await asyncio.to_thread(
                    note.read_text, encoding="utf-8", errors="replace"
                )
                chunks = chunk_pages([text], None)
                heading = f"Notes on {heading}"
        except ExtractionError as err:
            # Only a file that was read and would not parse is unreadable for
            # good. One that could not be read just now -- a permissions slip,
            # an I/O error, a copy mid-replace -- is tried again next pass.
            if not isinstance(err.__cause__, (OSError, MemoryError)):
                marks.append((paper.file_id, print_ + UNREADABLE))
            report.failed.append((label, str(err)))
            continue
        except OSError as err:
            report.failed.append((label, str(err)))
            continue
        inputs = [f"{heading}\n\n{chunk.text}" if heading else chunk.text for chunk in chunks]
        batch.append(_Pending(paper.file_id, source, print_, chunks, inputs))
        if (
            sum(len(p.inputs) for p in batch) >= EMBED_BATCH
            or sum(len(t) for p in batch for t in p.inputs) >= EMBED_BATCH_CHARS
        ):
            stop = await flush(done + 1, batch)
            batch = []
            if stop is not None:
                report.left = total - done - 1
                break
    if batch and stop is None:
        stop = await flush(total, batch)
    if marks:
        async with _held(exclusive):
            for file_id, print_ in marks:
                if library.db.get(file_id) is not None:  # not removed meanwhile
                    library.db.mark_text_source(file_id, DOCUMENT, print_)
            library.release()
    report.stopped = stop
    await tell(total, "stopped" if stop else "done")
    return report


@contextlib.asynccontextmanager
async def _held(lock: asyncio.Lock | None):
    if lock is None:
        yield
    else:
        async with lock:
            yield


async def _flush(
    library: Library,
    embedder: Embedder,
    batch: list[_Pending],
    report: IndexReport,
    exclusive: asyncio.Lock | None = None,
) -> str | None:
    """Embed a batch of sources together, then write them in one burst.

    A batch that fails is tried again one source at a time, so a source the API
    will not take fails alone. Without that, it failed every unrelated source
    batched with it -- and since a pass batches in the same order each time,
    failed them again on every pass after. That holds for a timeout or a 413
    as much as a 400: a batch can be too big without any one source being so.

    A single source that fails for any reason but its input -- the network,
    the service -- says the next request will too, so it stops the pass rather
    than letting every remaining source wait out its retries and timeout to fail
    the same way. A spent budget stops it at once. The stop reason is returned.
    A source that fails keeps its old fingerprint, or none, so the next pass
    tries it again.
    """
    inputs = [text for pending in batch for text in pending.inputs]
    try:
        vectors = await embedder.embed(inputs) if inputs else []
    except BudgetExceeded as err:
        report.failed.extend((_label(pending), str(err)) for pending in batch)
        return str(err)
    except LlmError as err:
        if len(batch) == 1:
            report.failed.append((_label(batch[0]), str(err)))
            return None if isinstance(err, EmbeddingRejected) else str(err)
        for at, pending in enumerate(batch):
            stop = await _flush(library, embedder, [pending], report, exclusive)
            if stop is not None:
                report.failed.extend((_label(p), stop) for p in batch[at + 1 :])
                return stop
        return None

    async with _held(exclusive):
        cursor = 0
        for pending in batch:
            mine = vectors[cursor : cursor + len(pending.chunks)]
            cursor += len(pending.chunks)
            if library.db.get(pending.file_id) is None:
                # Removed while its text was being embedded -- the lock is not
                # held through the request, so a `remove` can land in between.
                # Writing now would leave chunks with no document to point at.
                continue
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
    return None


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
    return [
        SearchHit(
            file_id=file_id,
            title=title,
            source=source,
            page=page,
            text=text,
            score=float(score),
        )
        for file_id, title, source, page, text, score in rows
    ]

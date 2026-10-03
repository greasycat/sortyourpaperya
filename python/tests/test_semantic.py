"""Semantic search: chunking, indexing only what changed, and finding by text or note."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import FakeLlmClient, write_pdf
from sortyourpaperya import semantic
from sortyourpaperya.config import Settings
from sortyourpaperya.ingest import ingest_folder
from sortyourpaperya.library import FilingMode, Library
from sortyourpaperya.llm import LlmError
from sortyourpaperya.semantic import chunk_pages, index_library, search


class FakeEmbedder:
    """Bag-of-words vectors: texts sharing words point the same way."""

    model = "fake-embedding"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False

    async def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise LlmError("embedding request failed: refused")
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * 64
        for word in text.lower().split():
            slot = int(hashlib.sha256(word.encode()).hexdigest(), 16) % 64
            vector[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


async def _file(settings: Settings, library: Library, **texts: str) -> None:
    for name, text in texts.items():
        write_pdf(settings.input_dir / f"{name}.pdf", text)
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.COPY)


def _paper_with(library: Library, word: str):
    for paper in library.db.all_papers():
        text = library.store_path(paper).read_bytes().decode("latin-1")
        if word in text:
            return paper
    raise AssertionError(f"no paper says {word!r}")


# ---- chunking ---------------------------------------------------------------


def test_a_short_page_is_one_chunk_that_knows_its_page() -> None:
    chunks = chunk_pages(["one two three", "", "four"], first_page=3)
    assert [(c.page, c.text) for c in chunks] == [(3, "one two three"), (5, "four")]
    assert [c.position for c in chunks] == [0, 1]


def test_a_long_page_is_cut_into_overlapping_windows() -> None:
    words = [f"w{i}" for i in range(400)]
    chunks = chunk_pages([" ".join(words)])
    assert chunks[0].text.split()[0] == "w0"
    assert chunks[1].text.split()[0] == f"w{semantic.CHUNK_STRIDE}"
    # Nothing is dropped off the end.
    assert chunks[-1].text.split()[-1] == "w399"
    assert all(len(c.text.split()) <= semantic.CHUNK_WORDS for c in chunks)


def test_a_wordless_run_cannot_make_an_oversized_chunk() -> None:
    (chunk,) = chunk_pages(["x" * 50_000])
    assert len(chunk.text) == semantic.CHUNK_MAX_CHARS


# ---- indexing ---------------------------------------------------------------


async def test_search_finds_the_document_whose_text_matches(
    settings: Settings, library: Library
) -> None:
    await _file(settings, library, a="reward shaping in reinforcement learning",
                b="attention heads in transformer language models")
    embedder = FakeEmbedder()
    report = await index_library(library, embedder)
    assert len(report.indexed) == 2 and report.failed == []

    (best, *_) = await search(library, embedder, "transformer attention heads")
    assert best.file_id == _paper_with(library, "transformer").file_id
    assert best.source == "" and best.page == 1
    assert "attention" in best.text


async def test_an_unchanged_library_costs_nothing_to_index_again(
    settings: Settings, library: Library
) -> None:
    await _file(settings, library, a="reward shaping")
    embedder = FakeEmbedder()
    await index_library(library, embedder)
    embedder.calls.clear()

    report = await index_library(library, embedder)
    assert report.indexed == [] and report.unchanged == 1
    assert embedder.calls == []


async def test_a_note_is_searchable_and_says_it_is_a_note(
    settings: Settings, library: Library
) -> None:
    await _file(settings, library, a="reward shaping", b="attention heads")
    paper = _paper_with(library, "reward")
    note = library.note_path(paper, "notes")
    note.write_text("compare with the dopamine prediction error literature")
    embedder = FakeEmbedder()
    await index_library(library, embedder)

    (best, *_) = await search(library, embedder, "dopamine prediction error")
    assert (best.file_id, best.source, best.page) == (paper.file_id, "notes.md", None)


async def test_an_edited_note_is_embedded_again_and_a_deleted_one_forgotten(
    settings: Settings, library: Library
) -> None:
    await _file(settings, library, a="reward shaping")
    paper = _paper_with(library, "reward")
    note = library.note_path(paper, "notes")
    note.write_text("first thoughts")
    embedder = FakeEmbedder()
    await index_library(library, embedder)

    note.write_text("second thoughts entirely")
    report = await index_library(library, embedder)
    assert report.indexed == [f"{paper.file_id}/notes.md"]

    note.unlink()
    report = await index_library(library, embedder)
    assert report.removed == 1
    assert (paper.file_id, "notes.md") not in {
        (f, s) for f, s, _ in library.db.text_fingerprints()
    }


async def test_a_failed_request_is_retried_by_the_next_pass(
    settings: Settings, library: Library
) -> None:
    await _file(settings, library, a="reward shaping")
    embedder = FakeEmbedder()
    embedder.fail = True
    report = await index_library(library, embedder)
    assert report.indexed == [] and len(report.failed) == 1
    assert library.db.text_fingerprints() == []

    embedder.fail = False
    report = await index_library(library, embedder)
    assert len(report.indexed) == 1


async def test_another_models_vectors_are_not_compared(
    settings: Settings, library: Library
) -> None:
    await _file(settings, library, a="reward shaping")
    await index_library(library, FakeEmbedder())

    other = FakeEmbedder()
    other.model = "another-model"
    assert await search(library, other, "reward shaping") == []
    assert library.db.unindexed_count("another-model") == 1


async def test_removing_a_document_takes_its_chunks_with_it(
    settings: Settings, library: Library
) -> None:
    await _file(settings, library, a="reward shaping")
    await index_library(library, FakeEmbedder())
    paper = _paper_with(library, "reward")

    library.remove(paper.file_id)
    assert library.db.text_fingerprints() == []
    assert library.db.select("SELECT count(*) FROM text_chunks")[1] == [(0,)]


# ---- the command --------------------------------------------------------------


def test_the_search_command_prints_the_passage_and_its_page(
    settings: Settings, library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from sortyourpaperya.cli import app

    asyncio.run(_file(settings, library, a="reward shaping in reinforcement learning"))
    embedder = FakeEmbedder()
    asyncio.run(index_library(library, embedder))
    library.release()
    monkeypatch.setattr("sortyourpaperya.cli._local_embedder", lambda: embedder)

    result = CliRunner().invoke(
        app, ["search", "reinforcement reward", "--library", str(library.root)]
    )
    assert result.exit_code == 0, result.output
    assert "p. 1: reward shaping in reinforcement learning" in result.output


# ---- the watcher ----------------------------------------------------------------


async def test_the_watcher_indexes_what_a_pass_filed(
    settings: Settings, library: Library
) -> None:
    import asyncio

    from sortyourpaperya.watch import watch

    write_pdf(settings.input_dir / "a.pdf", "reward shaping")
    embedder = FakeEmbedder()
    await asyncio.wait_for(
        watch(
            settings, FakeLlmClient(), library,
            mode=FilingMode.MOVE, max_passes=1, embedder=embedder,
        ),
        timeout=10,
    )
    assert len(library.db.text_fingerprints()) == 1


async def test_an_indexing_failure_does_not_stop_the_watcher(
    settings: Settings, library: Library
) -> None:
    import asyncio

    from sortyourpaperya.watch import watch

    write_pdf(settings.input_dir / "a.pdf", "reward shaping")

    class Broken(FakeEmbedder):
        async def embed(self, texts):
            raise RuntimeError("no key")

    reports = await asyncio.wait_for(
        watch(
            settings, FakeLlmClient(), library,
            mode=FilingMode.MOVE, max_passes=1, embedder=Broken(),
        ),
        timeout=10,
    )
    assert reports[0].processed == 1


# ---- the real embedder, against the SDK's own types -----------------------------


async def test_the_openai_embedder_orders_by_index_and_spends_from_the_budget(
    tmp_path: Path,
) -> None:
    from openai.types import CreateEmbeddingResponse

    from sortyourpaperya.budget import Budget, Limits

    budget = Budget(tmp_path / "ledger.json", Limits(requests_per_day=0, tokens_per_day=0))
    embedder = semantic.OpenAiEmbedder("sk-test", "text-embedding-3-small", budget=budget)

    async def create(*, model, input):
        # Returned out of order, which the API is allowed to do.
        return CreateEmbeddingResponse.model_validate({
            "object": "list",
            "model": model,
            "data": [
                {"object": "embedding", "index": i, "embedding": [float(i)]}
                for i in reversed(range(len(input)))
            ],
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        })

    embedder._client.embeddings.create = create
    assert await embedder.embed(["a", "b", "c"]) == [[0.0], [1.0], [2.0]]
    assert budget.usage().requests == 1 and budget.usage().tokens == 7


def test_embedding_spends_from_its_own_ledger_not_the_chat_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sortyourpaperya.budget import ledger_path

    monkeypatch.setenv("SYP_MAX_EMBEDDING_TOKENS_PER_DAY", "123")
    budget = semantic.embedding_budget()
    assert budget.path != ledger_path()
    assert budget.limits.tokens_per_day == 123
    # Requests are not capped: a source is only ever embedded once per change.
    assert budget.limits.requests_per_day == 0


async def test_a_spent_embedding_budget_fails_the_pass_without_recording(
    settings: Settings, library: Library, tmp_path: Path
) -> None:
    from sortyourpaperya.budget import Budget, Limits

    await _file(settings, library, a="reward shaping")
    spent = Budget(tmp_path / "e.json", Limits(requests_per_day=0, tokens_per_day=1))
    spent.record(requests=0, tokens=5)
    embedder = semantic.OpenAiEmbedder("sk-test", "fake-embedding", budget=spent)

    report = await index_library(library, embedder)
    assert report.indexed == [] and len(report.failed) == 1
    assert library.db.text_fingerprints() == []


# ---- progress -------------------------------------------------------------------


async def test_progress_counts_up_to_the_sources_that_needed_embedding(
    settings: Settings, library: Library
) -> None:
    await _file(settings, library, a="reward shaping", b="attention heads")
    note = library.note_path(_paper_with(library, "reward"), "notes")
    note.write_text("a note")
    seen: list[tuple[int, int, str]] = []

    async def progress(done: int, total: int, what: str) -> None:
        seen.append((done, total, what))

    await index_library(library, FakeEmbedder(), progress=progress)

    assert {total for _, total, _ in seen} == {3}
    dones = [done for done, _, _ in seen]
    assert dones == sorted(dones) and dones[0] == 0 and dones[-1] == 3
    assert any(what.startswith("reading notes.md of") for _, _, what in seen)
    assert any(what.startswith("embedding") for _, _, what in seen)
    assert seen[-1][2] == "done"


async def test_an_unchanged_library_reports_nothing_to_do(
    settings: Settings, library: Library
) -> None:
    await _file(settings, library, a="reward shaping")
    await index_library(library, FakeEmbedder())
    seen: list[tuple[int, int, str]] = []

    async def progress(done: int, total: int, what: str) -> None:
        seen.append((done, total, what))

    await index_library(library, FakeEmbedder(), progress=progress)
    assert {total for _, total, _ in seen} == {0}


def test_the_index_command_prints_its_summary(
    settings: Settings, library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from sortyourpaperya.cli import app

    asyncio.run(_file(settings, library, a="reward shaping"))
    library.release()
    monkeypatch.setattr("sortyourpaperya.cli._local_embedder", lambda: FakeEmbedder())

    result = CliRunner().invoke(app, ["index", "--library", str(library.root)])
    assert result.exit_code == 0, result.output
    assert "indexed 1 source(s)" in result.output

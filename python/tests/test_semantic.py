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


# ---- review fixes ---------------------------------------------------------------


async def test_the_watcher_answers_while_an_index_pass_is_embedding(
    settings: Settings, library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first index is minutes of requests; `status` must not wait them out."""
    import asyncio
    import shutil
    import tempfile

    from sortyourpaperya import client, daemon
    from sortyourpaperya.watch import _index

    short = tempfile.mkdtemp(prefix="sypyt-")
    monkeypatch.setenv("SORTYOURPAPERYA_RUNTIME_DIR", short)
    await _file(settings, library, a="reward shaping")

    release = asyncio.Event()
    started = asyncio.Event()

    class Gated(FakeEmbedder):
        async def embed(self, texts):
            started.set()
            await release.wait()
            return await super().embed(texts)

    server = daemon.Server(library)
    await server.start()
    try:
        indexing = asyncio.create_task(_index(server, library, Gated()))
        await asyncio.wait_for(started.wait(), timeout=5)
        answer = await asyncio.wait_for(client.call_async(library.root, "ping"), timeout=2)
        assert answer["library"] == str(library.root)
        release.set()
        await asyncio.wait_for(indexing, timeout=5)
        assert len(library.db.text_fingerprints()) == 1
    finally:
        release.set()
        await server.stop()
        shutil.rmtree(short, ignore_errors=True)


def test_requests_are_split_by_size_as_well_as_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic, "EMBED_BATCH_CHARS", 10)
    assert semantic._requests(["aaaa", "bbbb", "cccc", "d"]) == [["aaaa", "bbbb"], ["cccc", "d"]]
    # One input larger than the cap still goes, alone.
    assert semantic._requests(["x" * 50, "y"]) == [["x" * 50], ["y"]]


async def test_a_source_the_api_refuses_fails_alone(
    settings: Settings, library: Library
) -> None:
    await _file(settings, library, a="reward shaping", b="poison text here")

    class Picky(FakeEmbedder):
        async def embed(self, texts):
            if any("poison" in text for text in texts):
                raise semantic.EmbeddingRejected("embedding request refused: too many tokens")
            return await super().embed(texts)

    report = await index_library(library, Picky())
    assert report.indexed == [_paper_with(library, "reward").file_id]
    assert [label for label, _ in report.failed] == [_paper_with(library, "poison").file_id]


def test_a_spent_embedding_budget_names_its_own_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sortyourpaperya.budget import BudgetExceeded

    monkeypatch.setenv("SYP_MAX_EMBEDDING_TOKENS_PER_DAY", "1")
    budget = semantic.embedding_budget()
    budget.record(requests=0, tokens=5)
    with pytest.raises(BudgetExceeded, match="SYP_MAX_EMBEDDING_TOKENS_PER_DAY"):
        budget.check("embedding")


async def test_a_note_that_cannot_be_read_just_now_keeps_its_chunks(
    settings: Settings, library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _file(settings, library, a="reward shaping")
    paper = _paper_with(library, "reward")
    library.note_path(paper, "notes").write_text("dopamine thoughts")
    await index_library(library, FakeEmbedder())

    def unreadable(path):
        raise PermissionError("mid-save")

    monkeypatch.setattr(semantic, "_note_identity", unreadable)
    report = await index_library(library, FakeEmbedder())
    assert report.removed == 0 and len(report.failed) == 1
    assert (paper.file_id, "notes.md") in {(f, s) for f, s, _ in library.db.text_fingerprints()}


# ---- second review ----------------------------------------------------------------


async def test_the_watcher_keeps_filing_while_it_indexes(
    settings: Settings, library: Library
) -> None:
    """A long index pass runs beside the watch loop, not in front of it."""
    import asyncio

    from sortyourpaperya.watch import watch

    release = asyncio.Event()

    class Gated(FakeEmbedder):
        async def embed(self, texts):
            await release.wait()
            return await super().embed(texts)

    write_pdf(settings.input_dir / "a.pdf", "reward shaping")
    watching = asyncio.create_task(
        watch(
            settings, FakeLlmClient(), library,
            mode=FilingMode.MOVE, max_passes=2, embedder=Gated(),
        )
    )
    try:
        for _ in range(200):
            if library.db.count() == 1:
                break
            await asyncio.sleep(0.05)
        write_pdf(settings.input_dir / "b.pdf", "dopamine prediction error")
        for _ in range(200):
            if library.db.count() == 2:
                break
            await asyncio.sleep(0.05)
        assert library.db.count() == 2, "the second PDF waited for the index pass"
    finally:
        release.set()
        await asyncio.wait_for(watching, timeout=20)
    assert len(library.db.text_fingerprints()) == 2


async def test_a_failed_request_stops_the_pass_after_one_probe(
    settings: Settings, library: Library
) -> None:
    """A network failure says nothing about the input; one request per source
    would each wait out their retries to fail the same way. The batch, then one
    source alone, then stop."""
    await _file(settings, library, a="reward shaping", b="dopamine error")
    embedder = FakeEmbedder()
    embedder.fail = True
    report = await index_library(library, embedder)
    assert len(embedder.calls) == 2
    assert report.stopped and len(report.failed) == 2
    assert library.db.text_fingerprints() == []


async def test_an_unreadable_pdf_is_reported_once_then_left_alone(
    settings: Settings, library: Library
) -> None:
    await _file(settings, library, a="reward shaping")
    (paper,) = library.db.all_papers()
    library.store_path(paper).write_bytes(b"not a pdf at all")
    report = await index_library(library, FakeEmbedder())
    assert [label for label, _ in report.failed] == [paper.file_id]

    report = await index_library(library, FakeEmbedder())
    assert report.failed == [] and report.unchanged == 1


async def test_a_missing_document_is_tried_again_once_it_is_back(
    settings: Settings, library: Library
) -> None:
    await _file(settings, library, a="reward shaping")
    (paper,) = library.db.all_papers()
    path = library.store_path(paper)
    kept = path.read_bytes()
    path.unlink()
    report = await index_library(library, FakeEmbedder())
    assert report.failed == [(paper.file_id, "missing from the store")]
    # Not indexed before, so nothing to keep; see the test below for that.
    report = await index_library(library, FakeEmbedder())
    assert report.failed == [] and report.unchanged == 1

    path.write_bytes(kept)
    report = await index_library(library, FakeEmbedder())
    assert report.indexed == [paper.file_id]


async def test_a_scan_is_indexed_once_its_reading_is_banked(
    settings: Settings, library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sortyourpaperya.db import ModelAnswer
    from sortyourpaperya.extract import DocumentText

    await _file(settings, library, a="placeholder")
    (paper,) = library.db.all_papers()
    monkeypatch.setattr(
        semantic, "_read_store", lambda path: DocumentText([""], 1, 1)
    )
    report = await index_library(library, FakeEmbedder())
    assert report.indexed == [] and report.failed == []
    assert (await index_library(library, FakeEmbedder())).unchanged == 1

    library.db.remember_model_answers(
        [ModelAnswer(content_hash=paper.content_hash, page_text="hippocampal replay")]
    )
    report = await index_library(library, FakeEmbedder())
    assert report.indexed == [paper.file_id]
    (hit,) = await search(library, FakeEmbedder(), "hippocampal replay")
    assert hit.file_id == paper.file_id


async def test_a_retitled_document_is_embedded_again(
    settings: Settings, library: Library
) -> None:
    import dataclasses

    await _file(settings, library, a="reward shaping")
    await index_library(library, FakeEmbedder())
    (paper,) = library.db.all_papers()
    library.db.upsert(dataclasses.replace(paper, title="A better title"))
    report = await index_library(library, FakeEmbedder())
    assert report.indexed == [paper.file_id]
    (hit,) = await search(library, FakeEmbedder(), "reward shaping")
    assert hit.title == "A better title"


def test_a_refused_key_is_read_again_next_time(monkeypatch: pytest.MonkeyPatch) -> None:
    from sortyourpaperya import config
    from sortyourpaperya.llm import key_refused

    before = config.key_generation()
    key_refused(type("E", (Exception,), {"status_code": 500})())
    assert config.key_generation() == before
    key_refused(type("E", (Exception,), {"status_code": 401})())
    assert config.key_generation() == before + 1


def test_budget_shows_and_resets_the_embedding_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sortyourpaperya.cli import app

    monkeypatch.setenv("SORTYOURPAPERYA_STATE_DIR", str(tmp_path))
    semantic.embedding_budget().record(requests=1, tokens=1234)
    result = CliRunner().invoke(app, ["budget"])
    assert "embeddings" in result.output and "1234" in result.output
    CliRunner().invoke(app, ["budget", "--reset"])
    assert semantic.embedding_budget().usage().tokens == 0


# ---- third review -----------------------------------------------------------------


async def test_a_document_missing_for_one_pass_keeps_its_chunks(
    settings: Settings, library: Library
) -> None:
    """An unmounted store must not cost the library its embeddings."""
    import dataclasses

    await _file(settings, library, a="reward shaping")
    await index_library(library, FakeEmbedder())
    (paper,) = library.db.all_papers()
    # Retitled, so the pass looks at it again -- and finds the file gone.
    library.db.upsert(dataclasses.replace(paper, title="Renamed"))
    path = library.store_path(paper)
    kept = path.read_bytes()
    path.unlink()
    await index_library(library, FakeEmbedder())
    assert await search(library, FakeEmbedder(), "reward shaping")

    path.write_bytes(kept)
    report = await index_library(library, FakeEmbedder())
    assert report.indexed == [paper.file_id]


async def test_a_pdf_that_could_not_be_read_just_now_is_tried_again(
    settings: Settings, library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sortyourpaperya.extract import ExtractionError

    await _file(settings, library, a="reward shaping")
    real = semantic._read_store

    def flaky(path):
        raise ExtractionError("could not read a.pdf: I/O error") from OSError(5, "EIO")

    monkeypatch.setattr(semantic, "_read_store", flaky)
    assert len((await index_library(library, FakeEmbedder())).failed) == 1
    monkeypatch.setattr(semantic, "_read_store", real)
    assert len((await index_library(library, FakeEmbedder())).indexed) == 1


async def test_a_batch_that_times_out_is_split_before_the_pass_stops(
    settings: Settings, library: Library
) -> None:
    """Too big together is not too big alone; only a lone failure stops."""
    await _file(settings, library, a="reward shaping", b="dopamine error")

    class Choking(FakeEmbedder):
        async def embed(self, texts):
            if len(texts) > 1:
                raise LlmError("embedding request failed: timed out")
            return await super().embed(texts)

    report = await index_library(library, Choking())
    assert len(report.indexed) == 2 and report.stopped is None


def test_a_refused_key_is_not_read_again_on_every_request() -> None:
    """A key still wrong is refused by every request; one dialog, not N."""
    from sortyourpaperya import config
    from sortyourpaperya.llm import key_refused

    before = config.key_generation()
    refused = type("E", (Exception,), {"status_code": 401})()
    for _ in range(5):
        key_refused(refused)
    assert config.key_generation() == before + 1


async def test_a_refused_key_is_dropped_by_the_lazy_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy clients are the ones holding the cached key, so they drop it."""
    from sortyourpaperya import config

    class Refused(Exception):
        status_code = 401

    class Raising:
        model = "m"

        async def embed(self, texts):
            raise LlmError("embedding request failed: 401") from Refused()

    monkeypatch.setattr(config, "resolve_api_key", lambda: "sk-old")
    monkeypatch.setattr(semantic, "OpenAiEmbedder", lambda *a, **k: Raising())
    before = config.key_generation()
    with pytest.raises(LlmError):
        await semantic.LazyOpenAiEmbedder("m").embed(["x"])
    assert config.key_generation() == before + 1

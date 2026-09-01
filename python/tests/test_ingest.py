from __future__ import annotations

import shutil
from pathlib import Path
from unittest import mock

import pytest

from conftest import FailingLlmClient, FakeLlmClient, write_pdf, write_scanned_pdf

from sortyourpaperya.llm import LlmError
from sortyourpaperya.render import PageImage, RenderError


def _fake_render(path, page_cutoff):
    """Stand in for pdftoppm, so the suite does not need poppler."""
    return [PageImage(data=b"\x89PNG fake")]


from sortyourpaperya.config import Settings
import sortyourpaperya.ingest as ingest_module
from sortyourpaperya.ingest import ingest_folder
from sortyourpaperya.library import FilingMode, Library


async def test_ingests_every_pending_pdf(settings: Settings, library: Library) -> None:
    write_pdf(settings.input_dir / "a.pdf", "attention transformers")
    write_pdf(settings.input_dir / "b.pdf", "genome sequencing")
    client = FakeLlmClient()

    report = await ingest_folder(settings, client, library, mode=FilingMode.MOVE)

    assert report.processed == 2
    assert {filing.source.name for filing in report.filed} == {"a.pdf", "b.pdf"}
    assert all(filing.store_path.is_file() for filing in report.filed)


async def test_batches_respect_the_configured_size(settings: Settings, library: Library) -> None:
    for index in range(5):
        write_pdf(settings.input_dir / f"p{index}.pdf", f"paper {index}")
    client = FakeLlmClient()

    await ingest_folder(settings, client, library, mode=FilingMode.MOVE)

    # batch size 2 over 5 papers
    assert [len(batch) for batch in client.batches] == [2, 2, 1]


async def test_filing_empties_the_inbox_so_a_second_pass_has_nothing_to_do(
    settings: Settings, library: Library
) -> None:
    write_pdf(settings.input_dir / "a.pdf", "attention")
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.MOVE)

    second_client = FakeLlmClient()
    report = await ingest_folder(settings, second_client, library, mode=FilingMode.MOVE)

    assert second_client.batches == []
    assert report.processed == 0


async def test_the_same_content_arriving_again_costs_no_model_call(
    settings: Settings, library: Library
) -> None:
    source = write_pdf(settings.input_dir / "a.pdf", "attention")
    content = source.read_bytes()
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.MOVE)

    # The same paper downloaded a second time, under a different name.
    (settings.input_dir / "a-copy.pdf").write_bytes(content)
    client = FakeLlmClient()
    report = await ingest_folder(settings, client, library, mode=FilingMode.MOVE)

    assert client.batches == []
    assert report.skipped_already_known == 1
    assert library.db.count() == 1


async def test_identity_survives_renaming_because_it_is_content_addressed(
    settings: Settings, library: Library
) -> None:
    source = write_pdf(settings.input_dir / "a.pdf", "attention")
    content = source.read_bytes()
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.MOVE)
    stored_id = library.db.all_papers()[0].file_id

    (settings.input_dir / "totally-different-name.pdf").write_bytes(content)
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.MOVE)

    assert [paper.file_id for paper in library.db.all_papers()] == [stored_id]


async def test_the_library_is_never_reingested(settings: Settings, library: Library) -> None:
    write_pdf(settings.output_dir / "AI" / "filed.pdf", "already filed")
    write_pdf(settings.input_dir / "pending.pdf", "pending")
    settings = Settings(
        input_dir=settings.input_dir,
        output_dir=settings.output_dir,
        recursive=True,
        keyword_batch_size=settings.keyword_batch_size,
    )
    client = FakeLlmClient()

    report = await ingest_folder(settings, client, library, mode=FilingMode.MOVE)

    assert [filing.source.name for filing in report.filed] == ["pending.pdf"]


async def test_a_scanned_pdf_is_read_from_its_pages_and_filed(
    settings: Settings, library: Library
) -> None:
    # A scan opens fine and yields nothing, so its pages are rendered and read;
    # the text that comes back stands in for the text it does not carry.
    write_scanned_pdf(settings.input_dir / "scan.pdf")
    client = FakeLlmClient()

    with mock.patch.object(ingest_module, "render_pages", _fake_render):
        report = await ingest_folder(settings, client, library, mode=FilingMode.COPY)

    assert client.pages_read == [1], "the pages should have been read once"
    assert report.failed == []
    assert report.processed == 1
    paper = library.db.all_papers()[0]
    assert paper.from_page_images is True, "provenance should be recorded"
    assert paper.title, "the stand-in text should have produced a title"


async def test_a_scanned_pdf_is_labelled_by_the_same_batched_call(
    settings: Settings, library: Library
) -> None:
    # Reading the pages only supplies text; labelling stays one path, so a scan
    # and an ordinary document travel together and are steered the same way.
    write_scanned_pdf(settings.input_dir / "scan.pdf")
    write_pdf(settings.input_dir / "text.pdf", "An ordinary paper with a text layer")
    client = FakeLlmClient()

    with mock.patch.object(ingest_module, "render_pages", _fake_render):
        await ingest_folder(settings, client, library, mode=FilingMode.COPY)

    assert len(client.batches) == 1, "both documents in one labelling batch"
    assert len(client.batches[0]) == 2


async def test_a_scan_whose_pages_will_not_render_is_reported(
    settings: Settings, library: Library
) -> None:
    write_scanned_pdf(settings.input_dir / "scan.pdf")
    write_pdf(settings.input_dir / "fine.pdf", "a readable document")

    def refuse(path, page_cutoff):
        raise RenderError("pdftoppm failed")

    with mock.patch.object(ingest_module, "render_pages", refuse):
        report = await ingest_folder(
            settings, FakeLlmClient(), library, mode=FilingMode.COPY
        )

    assert [p.name for p, _ in report.failed] == ["scan.pdf"]
    assert "pdftoppm failed" in report.failed[0][1]
    assert report.processed == 1, "the readable document should still be filed"


async def test_a_scan_the_model_cannot_read_is_reported(
    settings: Settings, library: Library
) -> None:
    write_scanned_pdf(settings.input_dir / "scan.pdf")

    class RefusingClient(FakeLlmClient):
        async def describe_pages(self, images):
            raise LlmError("could not read the pages")

    with mock.patch.object(ingest_module, "render_pages", _fake_render):
        report = await ingest_folder(
            settings, RefusingClient(), library, mode=FilingMode.COPY
        )

    assert report.processed == 0
    assert "could not read the pages" in report.failed[0][1]


async def test_oversized_files_are_skipped_not_failed(settings: Settings, library: Library) -> None:
    big = settings.input_dir / "big.pdf"
    write_pdf(big, "big")
    big.write_bytes(big.read_bytes() + b"\x00" * (17 * 1024 * 1024))

    report = await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.MOVE)

    assert [path.name for path in report.skipped_oversized] == ["big.pdf"]
    assert report.failed == []


async def test_a_failed_batch_does_not_discard_the_batches_that_worked(
    settings: Settings, library: Library
) -> None:
    write_pdf(settings.input_dir / "a.pdf", "a")
    client = FailingLlmClient()

    report = await ingest_folder(settings, client, library, mode=FilingMode.MOVE)

    assert report.processed == 0
    assert len(report.failed) == 1
    assert "rate limited" in report.failed[0][1]
    # Nothing was recorded, so the next pass retries rather than skipping.
    assert library.db.count() == 0


async def test_everything_known_about_a_paper_lands_in_the_database(
    settings: Settings, library: Library
) -> None:
    write_pdf(settings.input_dir / "a.pdf", "a")

    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.MOVE)

    papers = library.db.all_papers()
    assert len(papers) == 1
    paper = papers[0]
    assert paper.tags == ["AI", "Transformers"]
    assert paper.authors == ["Ashish Vaswani"]
    assert paper.year == 2017
    assert paper.title == "Attention Is All You Need"
    assert sorted(paper.keywords) == ["alpha", "beta"]
    assert paper.original_name == "a.pdf"


async def test_a_preview_run_writes_nothing(settings: Settings, library: Library) -> None:
    source = write_pdf(settings.input_dir / "a.pdf", "a")

    report = await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.PREVIEW)

    assert len(report.planned) == 1
    assert report.filed == []
    assert source.exists(), "preview must leave the input folder untouched"
    assert library.db.count() == 0
    assert not library.store_dir.exists()


async def test_filed_papers_land_in_the_store_and_the_tree(
    settings: Settings, library: Library
) -> None:
    write_pdf(settings.input_dir / "a.pdf", "a")

    report = await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.MOVE)

    filing = report.filed[0]
    # The document lands in its own folder, and the tree links to the folder.
    assert filing.store_path.parent.name.endswith("__AI__Transformers")
    assert filing.store_path.parent.parent == library.store_dir
    assert filing.link_path.is_symlink()
    assert filing.link_path.resolve() == filing.store_path.parent.resolve()
    assert (filing.link_path / filing.store_path.name).is_file()


async def test_copy_mode_leaves_the_source_folder_untouched(
    settings: Settings, library: Library
) -> None:
    # The point of copy mode: index a folder someone else owns without
    # rearranging it underneath them.
    source = write_pdf(settings.input_dir / "a.pdf", "attention")
    before = source.read_bytes()

    report = await ingest_folder(
        settings, FakeLlmClient(), library, mode=FilingMode.COPY
    )

    filing = report.filed[0]
    assert source.exists(), "copy mode must not remove the source"
    assert source.read_bytes() == before
    assert filing.store_path.is_file()
    assert filing.store_path.read_bytes() == before
    assert filing.link_path.is_symlink()


async def test_copying_the_same_folder_twice_costs_one_model_call(
    settings: Settings, library: Library
) -> None:
    # In copy mode the source never drains, so re-scanning must be free or a
    # watched folder would pay for the same documents on every pass.
    write_pdf(settings.input_dir / "a.pdf", "attention")
    first = FakeLlmClient()
    await ingest_folder(settings, first, library, mode=FilingMode.COPY)

    second = FakeLlmClient()
    report = await ingest_folder(settings, second, library, mode=FilingMode.COPY)

    assert len(first.batches) == 1
    assert second.batches == [], "the second pass must not call the model"
    assert report.skipped_already_known == 1
    assert library.db.count() == 1


async def test_move_mode_works_across_filesystems(
    settings: Settings, library: Library, monkeypatch
) -> None:
    # os.replace refuses a cross-device move; shutil.move falls back to a copy.
    import shutil as _shutil

    write_pdf(settings.input_dir / "a.pdf", "attention")
    real_move = _shutil.move
    calls: list[str] = []

    def tracking_move(src, dst, *args, **kwargs):
        calls.append(str(src))
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr("sortyourpaperya.library.shutil.move", tracking_move)
    report = await ingest_folder(
        settings, FakeLlmClient(), library, mode=FilingMode.MOVE
    )

    assert calls, "move mode must go through shutil.move, not os.replace"
    assert report.filed[0].store_path.is_file()


async def test_a_new_document_is_shown_the_categories_already_in_use(
    settings: Settings, library: Library
) -> None:
    # Without this each document invents its own taxonomy and related documents
    # land under unrelated top-level branches.
    write_pdf(settings.input_dir / "first.pdf", "one")
    await ingest_folder(
        settings,
        FakeLlmClient(category="Medicine/Radiology"),
        library,
        mode=FilingMode.COPY,
    )

    write_pdf(settings.input_dir / "second.pdf", "two")
    client = FakeLlmClient(category="Medicine/Radiology")
    await ingest_folder(settings, client, library, mode=FilingMode.COPY)

    assert client.seen_categories == [["Medicine/Radiology"]]


async def test_the_first_document_into_an_empty_library_is_steered_by_nothing(
    settings: Settings, library: Library
) -> None:
    write_pdf(settings.input_dir / "a.pdf", "one")
    client = FakeLlmClient()

    await ingest_folder(settings, client, library, mode=FilingMode.COPY)

    assert client.seen_categories == [[]]


async def test_steering_is_read_once_so_concurrent_batches_see_the_same_library(
    settings: Settings, library: Library
) -> None:
    # Batches run concurrently and cannot inform each other; every batch in a
    # pass is steered by the library as it stood when the pass began.
    await ingest_folder(
        settings,
        FakeLlmClient(category="Medicine/Radiology"),
        library,
        mode=FilingMode.COPY,
    )
    write_pdf(settings.input_dir / "seed.pdf", "seed")
    await ingest_folder(
        settings,
        FakeLlmClient(category="Medicine/Radiology"),
        library,
        mode=FilingMode.COPY,
    )

    for index in range(4):
        write_pdf(settings.input_dir / f"n{index}.pdf", f"doc {index}")
    client = FakeLlmClient(category="Finance/Bills")
    await ingest_folder(settings, client, library, mode=FilingMode.COPY)

    assert len(client.seen_categories) == 2, "batch size 2 over 4 documents"
    assert all(seen == ["Medicine/Radiology"] for seen in client.seen_categories)


async def test_an_edited_document_is_recognised_without_a_manual_scan(
    settings: Settings, library: Library
) -> None:
    # The whole point of reconciling on the way in: nobody has to remember to
    # run `sortyourpaperya scan` for the library to recognise what it already holds.
    source = write_pdf(settings.input_dir / "a.pdf", "attention")
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.COPY)
    source.unlink()

    paper = library.db.all_papers()[0]
    stored = library.store_path(paper)
    stored.write_bytes(stored.read_bytes() + b"\n% annotated\n")
    # The annotated copy comes back, as it would from a re-download.
    (settings.input_dir / "annotated.pdf").write_bytes(stored.read_bytes())

    client = FakeLlmClient()
    report = await ingest_folder(settings, client, library, mode=FilingMode.COPY)

    assert client.batches == [], "the edited copy should not reach the model"
    assert report.skipped_already_known == 1
    assert library.db.count() == 1, "no duplicate was created"
    assert report.rescan is not None and len(report.rescan.changed) == 1


async def test_reconciling_a_quiet_library_rereads_nothing(
    settings: Settings, library: Library
) -> None:
    write_pdf(settings.input_dir / "a.pdf", "attention")
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.COPY)

    write_pdf(settings.input_dir / "b.pdf", "second")
    report = await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.COPY)

    assert report.rescan is not None
    assert report.rescan.checked == 1, "the one already-filed document"
    assert report.rescan.rehashed == 0, "nothing changed, so nothing is reread"


async def test_an_unreadable_store_does_not_stop_the_pass(
    settings: Settings, library: Library, monkeypatch
) -> None:
    # Failing to reconcile risks a duplicate; refusing to run files nothing at
    # all. The pass continues.
    write_pdf(settings.input_dir / "a.pdf", "attention")

    def boom(**_kwargs) -> None:
        raise OSError("store is unreadable")

    monkeypatch.setattr(library, "rescan", boom)
    report = await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.COPY)

    assert report.rescan is None
    assert report.processed == 1, "the document should still have been filed"


async def test_no_database_connection_is_held_during_the_model_calls(
    settings: Settings, library: Library
) -> None:
    """The model calls are the slow part of a pass and need no database.

    DuckDB allows one writing process, so a connection left open across them
    would shut every other `sortyourpaperya` command out for the length of the pass.
    """
    observed: list[bool] = []

    class WatchingClient(FakeLlmClient):
        async def extract_keywords(self, batch, existing_categories=()):
            observed.append(library.db._connection is not None)
            return await super().extract_keywords(batch, existing_categories)

    for index in range(3):
        write_pdf(settings.input_dir / f"p{index}.pdf", f"document {index}")

    await ingest_folder(settings, WatchingClient(), library, mode=FilingMode.COPY)

    assert observed, "the model should have been called"
    assert not any(observed), "a connection was open while the model was working"


async def test_no_database_connection_is_held_while_files_are_copied(
    settings: Settings, library: Library
) -> None:
    # Copying is the other slow phase: it runs after the model returns and
    # before the single write burst that records the whole batch.
    observed: list[bool] = []
    real_copy = shutil.copy2

    def watching_copy(src, dst, *args, **kwargs):
        observed.append(library.db._connection is not None)
        return real_copy(src, dst, *args, **kwargs)

    for index in range(3):
        write_pdf(settings.input_dir / f"p{index}.pdf", f"document {index}")

    with mock.patch.object(shutil, "copy2", watching_copy):
        await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.COPY)

    assert len(observed) == 3, "every document should have been copied"
    assert not any(observed), "a connection was open while files were being copied"


async def test_a_document_already_held_is_never_parsed(
    settings: Settings, library: Library
) -> None:
    # Parsing is expensive and pointless for a document the library has. In copy
    # mode the source never drains, so every later pass would pay for it again.
    source = write_pdf(settings.input_dir / "a.pdf", "attention")
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.COPY)
    assert source.exists(), "copy mode leaves the source in place"

    parsed: list[str] = []
    real_extract = ingest_module.extract_paper_text

    def watching_extract(path, file_id, page_cutoff):
        parsed.append(path.name)
        return real_extract(path, file_id, page_cutoff)

    with mock.patch.object(ingest_module, "extract_paper_text", watching_extract):
        report = await ingest_folder(
            settings, FakeLlmClient(), library, mode=FilingMode.COPY
        )

    assert report.skipped_already_known == 1
    assert parsed == [], "an already-held document should not be parsed again"


async def test_a_busy_database_does_not_stop_the_pass(
    settings: Settings, library: Library
) -> None:
    # duckdb.IOException is not an OSError, so catching OSError alone let a
    # lock held past the wait kill the pass this is meant to survive.
    import duckdb

    write_pdf(settings.input_dir / "a.pdf", "attention")

    def busy(**_kwargs) -> None:
        raise duckdb.IOException("Could not set lock on file")

    with mock.patch.object(library, "rescan", busy):
        report = await ingest_folder(
            settings, FakeLlmClient(), library, mode=FilingMode.COPY
        )

    assert report.rescan is None
    assert report.processed == 1, "the document should still have been filed"


async def test_a_preview_does_not_change_what_the_library_holds(
    settings: Settings, library: Library, tmp_path: Path
) -> None:
    """`--mode preview` promises to leave the library as it found it.

    A hash refreshed on the way in is a row rewritten, which is a real change to
    a library the caller asked not to touch.

    What a preview does write is the bank of model answers, which says nothing
    about what the library holds — only that a request has already been paid
    for. Throwing that away would mean the apply after a preview buys the same
    answer twice.
    """
    write_pdf(settings.input_dir / "a.pdf", "attention")
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.COPY)
    paper = library.db.all_papers()[0]
    stored = library.store_path(paper)
    stored.write_bytes(stored.read_bytes() + b"% annotated\n")
    before = library.db.get(paper.file_id)

    write_pdf(settings.input_dir / "b.pdf", "another")
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.PREVIEW)

    after = library.db.get(paper.file_id)
    assert after.content_hash == before.content_hash, "a preview rewrote a row"
    assert after.stored_mtime_ms == before.stored_mtime_ms
    assert library.db.count() == 1, "a preview filed something"
    assert library.orphans() == [], "a preview put something in the store"


async def test_a_preview_still_sees_what_the_reconciliation_found(
    settings: Settings, library: Library, tmp_path: Path
) -> None:
    """A preview that cannot record still has to agree with the apply.

    Otherwise it calls an edited copy new, and the apply that follows recognises
    it — the preview would be showing filing that is never going to happen.
    """
    write_pdf(settings.input_dir / "a.pdf", "attention")
    # Moved in, so the only thing left in the watched folder is what arrives next.
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.MOVE)
    paper = library.db.all_papers()[0]
    stored = library.store_path(paper)
    edited = stored.read_bytes() + b"% annotated\n"
    stored.write_bytes(edited)

    # The same edited bytes come back in the watched folder.
    (settings.input_dir / "annotated.pdf").write_bytes(edited)
    report = await ingest_folder(
        settings, FakeLlmClient(), library, mode=FilingMode.PREVIEW
    )

    assert report.planned == [], "the edited copy is already in the library"
    assert report.skipped_already_known == 1


async def test_an_apply_does_record_what_it_found(
    settings: Settings, library: Library
) -> None:
    # The counterpart: outside preview the refreshed hash is written down, so
    # the next pass does not have to read the file again.
    write_pdf(settings.input_dir / "a.pdf", "attention")
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.COPY)
    paper = library.db.all_papers()[0]
    stored = library.store_path(paper)
    before = library.db.get(paper.file_id).content_hash
    stored.write_bytes(stored.read_bytes() + b"% annotated\n")

    write_pdf(settings.input_dir / "b.pdf", "another")
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.COPY)

    assert library.db.get(paper.file_id).content_hash != before


# ---- paying once -----------------------------------------------------------


async def test_a_crash_between_paying_and_filing_does_not_pay_again(
    settings: Settings, library: Library, monkeypatch
) -> None:
    """The failure the bank exists for.

    The model call comes before the file work, so a pass that dies in between
    has spent money and recorded nothing. Under `KeepAlive` a pass that dies is
    a pass that starts again — so without this, a crash that repeats is a charge
    that repeats with it.
    """
    write_pdf(settings.input_dir / "a.pdf", "attention")
    client = FakeLlmClient()

    def die_while_copying(*_args, **_kwargs):
        raise KeyboardInterrupt("killed mid-pass")

    monkeypatch.setattr(Library, "place_file", die_while_copying)
    with pytest.raises(KeyboardInterrupt):
        await ingest_folder(settings, client, library, mode=FilingMode.COPY)
    assert len(client.batches) == 1, "the first attempt should have paid once"

    monkeypatch.undo()
    report = await ingest_folder(settings, client, library, mode=FilingMode.COPY)

    assert len(client.batches) == 1, "the restart bought the same answer again"
    assert report.reused_answers == 1
    assert report.processed == 1, "and it still gets filed"


async def test_the_answer_is_banked_before_the_file_is_touched(
    settings: Settings, library: Library, monkeypatch
) -> None:
    # Banking it after the copy would leave exactly the window this closes.
    write_pdf(settings.input_dir / "a.pdf", "attention")
    banked: list[int] = []
    real_place = Library.place_file

    def watch_place(self, *args, **kwargs):
        banked.append(self.db.count_model_answers())
        return real_place(self, *args, **kwargs)

    monkeypatch.setattr(Library, "place_file", watch_place)
    await ingest_folder(settings, FakeLlmClient(), library, mode=FilingMode.COPY)

    assert banked == [1], "the answer was not banked before the file was copied"


async def test_a_failed_batch_does_not_lose_the_page_text_it_was_given(
    settings: Settings, library: Library, monkeypatch
) -> None:
    """Reading a scan is the most expensive thing a pass does.

    A rate limit on the labelling call that followed would otherwise throw that
    reading away and buy it again on the next pass.
    """
    write_scanned_pdf(settings.input_dir / "scan.pdf")
    monkeypatch.setattr(ingest_module, "render_pages", _fake_render)

    class _ReadsThenFails(FakeLlmClient):
        async def extract_keywords(self, batch, existing_categories=()):
            raise LlmError("rate limited")

    reader = _ReadsThenFails()
    await ingest_folder(settings, reader, library, mode=FilingMode.COPY)
    assert reader.pages_read == [1], "the scan should have been read once"

    working = FakeLlmClient()
    report = await ingest_folder(settings, working, library, mode=FilingMode.COPY)

    assert working.pages_read == [], "the scan was read a second time"
    assert report.processed == 1


async def test_a_preview_and_the_apply_after_it_pay_once(
    settings: Settings, library: Library
) -> None:
    # Looking at what would happen, then letting it happen, is one decision.
    write_pdf(settings.input_dir / "a.pdf", "attention")
    client = FakeLlmClient()

    await ingest_folder(settings, client, library, mode=FilingMode.PREVIEW)
    report = await ingest_folder(settings, client, library, mode=FilingMode.COPY)

    assert len(client.batches) == 1, "the apply asked the model again"
    assert report.processed == 1


async def test_a_banked_answer_past_its_age_is_not_reused(
    settings: Settings, library: Library, monkeypatch
) -> None:
    """A label is a choice made against the categories the library had then.

    Worth reusing for a while; not forever.
    """
    monkeypatch.setenv("SYP_LABEL_CACHE_DAYS", "7")
    write_pdf(settings.input_dir / "a.pdf", "attention")
    client = FakeLlmClient()
    await ingest_folder(settings, client, library, mode=FilingMode.PREVIEW)

    eight_days = 8 * 24 * 60 * 60 * 1000
    library.db._conn.execute(
        "UPDATE model_answers SET created_at_ms = created_at_ms - ?", [eight_days]
    )
    library.release()

    await ingest_folder(settings, client, library, mode=FilingMode.COPY)

    assert len(client.batches) == 2, "a stale answer was reused"


async def test_reuse_can_be_turned_off(
    settings: Settings, library: Library, monkeypatch
) -> None:
    monkeypatch.setenv("SYP_LABEL_CACHE_DAYS", "0")
    write_pdf(settings.input_dir / "a.pdf", "attention")
    client = FakeLlmClient()

    await ingest_folder(settings, client, library, mode=FilingMode.PREVIEW)
    await ingest_folder(settings, client, library, mode=FilingMode.COPY)

    assert len(client.batches) == 2
    assert library.db.count_model_answers() == 0, "nothing should have been banked"


async def test_an_unreadable_banked_answer_is_only_a_miss(
    settings: Settings, library: Library
) -> None:
    # The worst a broken cache may cost is the request it was meant to save.
    write_pdf(settings.input_dir / "a.pdf", "attention")
    client = FakeLlmClient()
    await ingest_folder(settings, client, library, mode=FilingMode.PREVIEW)
    library.db._conn.execute("UPDATE model_answers SET labels = '{not json'")
    library.release()

    report = await ingest_folder(settings, client, library, mode=FilingMode.COPY)

    assert len(client.batches) == 2
    assert report.processed == 1


async def test_naming_one_pdf_files_that_one_and_leaves_the_rest(
    settings: Settings, library: Library
) -> None:
    """One document filed without first putting it in a folder of its own."""
    from dataclasses import replace

    wanted = write_pdf(settings.input_dir / "wanted.pdf", "attention")
    write_pdf(settings.input_dir / "other.pdf", "something else entirely")

    report = await ingest_folder(
        replace(settings, input_dir=wanted), FakeLlmClient(), library, mode=FilingMode.COPY
    )

    assert report.processed == 1
    assert [paper.original_name for paper in library.db.all_papers()] == ["wanted.pdf"]
    # And the one not named is left exactly as it was.
    assert (settings.input_dir / "other.pdf").is_file()

"""The command layer: how a running service is set up to be read afterwards."""

from __future__ import annotations

import logging
import subprocess
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from conftest import FailingLlmClient, FakeLlmClient
from sortyourpaperya.cli import LOG_BACKUP_COUNT, LOG_MAX_BYTES, _configure


@pytest.fixture(autouse=True)
def _restore_logging():
    """Put the root logger back, so configuring it here cannot leak into other tests."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_every_line_carries_a_timestamp(monkeypatch, capsys) -> None:
    """The watcher's log is read hours later, and often after a restart.

    An untimed line cannot say whether it is from this run or the one that
    crashed — which is exactly the question a restart loop raises.
    """
    monkeypatch.delenv("SORTYOURPAPERYA_LOG_FILE", raising=False)
    _configure(verbose=False)

    logging.getLogger("sortyourpaperya.test").info("something happened")

    line = capsys.readouterr().err.strip()
    assert line.endswith("INFO something happened")
    assert line[:4].isdigit(), f"no timestamp: {line!r}"


def test_the_service_log_is_rotated(monkeypatch, tmp_path: Path) -> None:
    """A watcher left running for months must not be able to fill the disk."""
    log_file = tmp_path / "logs" / "sortyourpaperya.log"
    monkeypatch.setenv("SORTYOURPAPERYA_LOG_FILE", str(log_file))

    _configure(verbose=False)

    handlers = logging.getLogger().handlers
    rotating = [h for h in handlers if isinstance(h, RotatingFileHandler)]
    assert len(rotating) == 1
    assert rotating[0].maxBytes == LOG_MAX_BYTES
    assert rotating[0].backupCount == LOG_BACKUP_COUNT


def test_the_log_is_not_also_written_somewhere_nothing_trims(
    monkeypatch, tmp_path: Path
) -> None:
    """Under launchd, stderr goes to a file that nothing rotates.

    Writing every line to both would leave that unbounded copy growing exactly
    as it did before the rotation was added, so the rotating handler replaces
    the stream one rather than joining it.
    """
    monkeypatch.setenv("SORTYOURPAPERYA_LOG_FILE", str(tmp_path / "sortyourpaperya.log"))

    _configure(verbose=False)

    streams = [
        handler
        for handler in logging.getLogger().handlers
        if type(handler) is logging.StreamHandler
    ]
    assert streams == []


def test_the_log_file_is_written_where_it_was_asked_for(
    monkeypatch, tmp_path: Path
) -> None:
    log_file = tmp_path / "nested" / "sortyourpaperya.log"
    monkeypatch.setenv("SORTYOURPAPERYA_LOG_FILE", str(log_file))
    _configure(verbose=False)

    logging.getLogger("test").warning("into the file")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert "into the file" in log_file.read_text(encoding="utf-8")


def test_a_record_carries_the_path_to_the_document(library) -> None:
    """A search result whose file cannot be opened is only half an answer."""
    from sortyourpaperya.cli import _describe
    from sortyourpaperya.db import Paper

    paper = Paper(
        file_id="aaa111aaa111",
        content_hash="sha-1",
        store_name="aaa111aaa111__AI__Transformers",
        document_name="vaswani_2017_attention.pdf",
        title="Attention Is All You Need",
        tags=["AI", "Transformers"],
    )

    folder = library.document_dir(paper)
    folder.mkdir(parents=True)
    (folder / "reading-log.md").write_text("what I made of it", encoding="utf-8")

    record = _describe(library, paper)

    assert Path(record["document"]).is_absolute()
    assert record["document"].endswith(
        "store/aaa111aaa111__AI__Transformers/vaswani_2017_attention.pdf"
    )
    assert Path(record["notes"][0]).is_absolute()
    assert record["notes"] == [str(folder.resolve() / "reading-log.md")], (
        "a note is found by being markdown beside the document, not by its name"
    )
    assert record["category"] == "AI/Transformers"


def test_note_path_never_opens_an_editor(library, monkeypatch, capsys) -> None:
    """A caller writing the notes itself would be stuck holding an editor open."""
    from sortyourpaperya import cli
    from sortyourpaperya.db import Paper

    paper = Paper(
        file_id="aaa111aaa111",
        content_hash="sha-1",
        store_name="aaa111aaa111__AI",
        document_name="paper.pdf",
        title="A Paper",
        tags=["AI"],
    )
    library.db.upsert(paper)
    library.document_dir(paper).mkdir(parents=True)
    library.close()

    monkeypatch.setenv("EDITOR", "vim")
    monkeypatch.setattr(
        cli.subprocess, "call", lambda *a, **k: pytest.fail("opened an editor")
    )

    # Called directly rather than through Typer, so the defaults it would have
    # filled in are passed by hand.
    cli.note("aaa111aaa111", name=None, library_dir=library.root, path_only=True)

    assert capsys.readouterr().out.strip().endswith("notes.md")


def _crowded(library, count: int = 25):
    """A library with more documents than a default search will show."""
    from sortyourpaperya.db import Paper

    library.db.upsert_many(
        [
            Paper(
                file_id=f"{i:012d}",
                content_hash=f"sha-{i}",
                store_name=f"{i:012d}__AI",
                document_name="paper.pdf",
                title=f"Paper {i}",
                tags=["AI"],
                keywords=["shared"],
            )
            for i in range(count)
        ]
    )
    root = library.root
    library.close()
    return root


def _invoke(root, *args, **kwargs):
    from typer.testing import CliRunner

    from sortyourpaperya.cli import app

    return CliRunner().invoke(app, [*args, "--library", str(root)], **kwargs)


def test_running_sortyourpaperya_with_no_arguments_shows_the_help() -> None:
    """A bare `sortyourpaperya` is someone asking what this does.

    Click's default answer is "Missing command", which names neither the
    commands there are nor the one that is missing.
    """
    from typer.testing import CliRunner

    from sortyourpaperya.cli import app

    result = CliRunner().invoke(app, [])

    assert "Usage: " in result.output
    assert "ingest" in result.output and "find" in result.output


def test_a_capped_search_says_it_was_capped(library) -> None:
    """A cap that says nothing reads as the whole answer.

    The reader stops looking, and the document that was cut off is one they
    now believe the library does not hold.
    """
    root = _crowded(library)

    result = _invoke(root, "find", "shared", "--limit", "3")

    assert result.exit_code == 0, result.output
    assert len([line for line in result.stdout.splitlines() if line[:12].isdigit()]) == 3
    assert "more than 3 matched" in result.stderr
    assert "--limit 0" in result.stderr


def test_the_cap_notice_leaves_the_records_parseable(library) -> None:
    """It goes to stderr, so `--json` stdout is still only JSON."""
    import json

    root = _crowded(library)

    result = _invoke(root, "find", "shared", "--limit", "3", "--json")

    assert len(json.loads(result.stdout)) == 3
    assert "more than 3 matched" in result.stderr


def test_a_search_that_fits_says_nothing_about_a_cap(library) -> None:
    root = _crowded(library, count=3)

    result = _invoke(root, "find", "shared", "--limit", "3")

    assert result.exit_code == 0, result.output
    assert "more than" not in result.stderr


def test_no_limit_shows_everything(library) -> None:
    root = _crowded(library)

    result = _invoke(root, "find", "shared", "--limit", "0")

    assert "25 document(s)" in result.stdout
    assert "more than" not in result.stderr


def test_a_negative_limit_is_a_usage_error(library) -> None:
    """Not a DuckDB binder error surfacing from three layers down."""
    root = _crowded(library, count=1)

    result = _invoke(root, "find", "shared", "--limit", "-1")

    assert result.exit_code == 2
    assert "--limit" in result.stderr


# ---- re-asking for a category ----------------------------------------------


def _misfiled(library, tags=("Psychology", "Research Methods")):
    """A document filed where steering put it, with a real folder in the store."""
    from sortyourpaperya.db import Paper

    paper = Paper(
        file_id="78c64b3b8ef6",
        content_hash="sha-kahn",
        store_name="78c64b3b8ef6__" + "__".join(tags),
        document_name="kahn_2025_successor-representations.pdf",
        title="Trial-by-trial learning of successor representations",
        authors=["Ari E. Kahn", "Nathaniel D. Daw"],
        year=2025,
        tags=list(tags),
        keywords=["successor representations", "temporal difference learning"],
    )
    library.db.upsert(paper)
    folder = library.document_dir(paper)
    folder.mkdir(parents=True)
    (folder / paper.document_name).write_bytes(b"%PDF-1.4 not a real pdf")
    (folder / "notes.md").write_text("kept beside it", encoding="utf-8")
    library.rebuild_tree()
    root = library.root
    library.close()
    return root


def _retag(root, fake, *args, **kwargs):
    """Invoke `retag` with the fake client standing in for the real one."""
    from typer.testing import CliRunner

    from sortyourpaperya import cli

    original = cli._build
    cli._build = lambda *a, **k: (original(*a, **k)[0], fake)
    try:
        return CliRunner().invoke(
            cli.app, ["retag", *args, "--library", str(root)], **kwargs
        )
    finally:
        cli._build = original


def test_accepting_a_suggestion_moves_the_document(library, settings) -> None:
    """Tags, folder, link, and keywords all move together."""
    from sortyourpaperya.library import Library

    root = _misfiled(library)
    fake = FakeLlmClient()

    result = _retag(root, fake, "78c64b3b8ef6", input="a\n")

    assert result.exit_code == 0, result.output
    with Library(root) as after:
        paper = after.db.get("78c64b3b8ef6")
        assert paper.tags == ["Cognitive Science", "Computational Modelling"]
        assert after.document_dir(paper).is_dir()
        assert (after.document_dir(paper) / "notes.md").exists(), "notes travel"
        assert "shared" in paper.keywords, "keywords were refreshed too"


def test_regenerating_never_offers_the_same_category_twice(library) -> None:
    """The whole point of asking again.

    Each round must carry what came before, or the model has no reason to
    answer differently and the loop cannot end except by cancelling.
    """
    root = _misfiled(library)
    fake = FakeLlmClient()

    result = _retag(root, fake, "78c64b3b8ef6", input="r\nr\na\n")

    assert result.exit_code == 0, result.output
    assert len(fake.suggestions) == 3
    assert fake.suggestions[0].rejected == []
    assert fake.suggestions[1].rejected == ["Cognitive Science/Computational Modelling"]
    assert fake.suggestions[2].rejected == [
        "Cognitive Science/Computational Modelling",
        "Neuroscience/Learning",
    ]


def test_a_steer_is_carried_into_the_next_suggestion(library) -> None:
    """Refusing says only "not that". Someone who has read the document knows more."""
    root = _misfiled(library)
    fake = FakeLlmClient()

    result = _retag(
        root, fake, "78c64b3b8ef6", input="s\nit is about the maths, not the clinic\na\n"
    )

    assert result.exit_code == 0, result.output
    assert len(fake.suggestions) == 2
    assert fake.suggestions[0].guidance == ""
    assert fake.suggestions[1].guidance == "it is about the maths, not the clinic"
    assert fake.suggestions[1].rejected == ["Cognitive Science/Computational Modelling"], (
        "a steer is still a refusal of what was on screen"
    )
    assert "it is about the maths" in result.output, "shown with the answer it produced"


def test_a_later_steer_replaces_the_one_before_it(library) -> None:
    """Someone correcting their own instruction means the newer sentence."""
    root = _misfiled(library)
    fake = FakeLlmClient()

    _retag(root, fake, "78c64b3b8ef6", input="s\nthe maths\ns\nno, the hardware\nc\n")

    assert [call.guidance for call in fake.suggestions] == [
        "",
        "the maths",
        "no, the hardware",
    ]


def test_saying_nothing_at_the_steer_prompt_costs_no_request(library) -> None:
    """An empty answer is not a decision, so the menu comes back."""
    root = _misfiled(library)
    fake = FakeLlmClient()

    result = _retag(root, fake, "78c64b3b8ef6", input="s\n\nc\n")

    assert result.exit_code == 0, result.output
    assert len(fake.suggestions) == 1, "nothing was asked again"


def test_the_model_is_told_where_the_document_sits_now(library) -> None:
    root = _misfiled(library)
    fake = FakeLlmClient()

    _retag(root, fake, "78c64b3b8ef6", input="c\n")

    call = fake.suggestions[0]
    assert call.current == "Psychology/Research Methods"
    assert "successor representations" in call.text, "the library's own keywords"
    assert "Ari E. Kahn" in call.text


def test_cancelling_changes_nothing(library) -> None:
    from sortyourpaperya.library import Library

    root = _misfiled(library)

    result = _retag(root, FakeLlmClient(), "78c64b3b8ef6", input="c\n")

    assert result.exit_code == 0, "cancelling is a decision, not an error"
    with Library(root) as after:
        assert after.db.get("78c64b3b8ef6").tags == ["Psychology", "Research Methods"]


def test_no_one_at_the_keyboard_cancels_rather_than_crashing(library) -> None:
    """From cron, or with stdin closed, this must stop cleanly."""
    from sortyourpaperya.library import Library

    root = _misfiled(library)

    result = _retag(root, FakeLlmClient(), "78c64b3b8ef6", input="")

    assert result.exit_code == 0, result.output
    assert "cancelled" in result.output
    with Library(root) as after:
        assert after.db.get("78c64b3b8ef6").tags == ["Psychology", "Research Methods"]


def test_a_document_whose_file_is_gone_still_gets_a_suggestion(library) -> None:
    """Its record is what the model reads from anyway."""
    root = _misfiled(library)
    from sortyourpaperya.library import Library

    with Library(root) as lib:
        paper = lib.db.get("78c64b3b8ef6")
        (lib.document_dir(paper) / paper.document_name).unlink()
    fake = FakeLlmClient()

    result = _retag(root, fake, "78c64b3b8ef6", input="c\n")

    assert result.exit_code == 0, result.output
    assert "successor representations" in fake.suggestions[0].text


def test_an_unknown_id_is_reported_before_anything_is_asked(library) -> None:
    root = _misfiled(library)
    fake = FakeLlmClient()

    result = _retag(root, fake, "nosuchdocument", input="a\n")

    assert result.exit_code == 1
    assert fake.suggestions == [], "no request is paid for"


def test_a_failing_model_does_not_leave_the_document_half_moved(library) -> None:
    from sortyourpaperya.library import Library

    root = _misfiled(library)

    result = _retag(root, FailingLlmClient(), "78c64b3b8ef6", input="a\n")

    assert result.exit_code == 1
    with Library(root) as after:
        assert after.db.get("78c64b3b8ef6").tags == ["Psychology", "Research Methods"]


def test_retag_with_a_category_never_asks_the_model(library) -> None:
    """The manual path is unchanged, and free."""
    from sortyourpaperya.library import Library

    root = _misfiled(library)
    fake = FakeLlmClient()

    result = _retag(root, fake, "78c64b3b8ef6", "Cognitive Science/Computation")

    assert result.exit_code == 0, result.output
    assert fake.suggestions == []
    with Library(root) as after:
        paper = after.db.get("78c64b3b8ef6")
        assert paper.tags == ["Cognitive Science", "Computation"]
        assert paper.keywords == [
            "successor representations",
            "temporal difference learning",
        ], "a hand re-tag leaves what the library knew alone"


def test_only_our_own_lines_are_turned_up(monkeypatch, capsys) -> None:
    """`sortyourpaperya retag` waits at a prompt; a request log lands in the middle of it.

    Named by what is ours rather than by what is noisy: the OpenAI SDK vendors
    its HTTP client, so the logger doing the narrating is called `httpx2` today
    and something else after the next release.
    """
    monkeypatch.delenv("SORTYOURPAPERYA_LOG_FILE", raising=False)
    _configure(verbose=False)

    logging.getLogger("httpx2._client").info("HTTP Request: POST ...")
    logging.getLogger("sortyourpaperya.ingest").info("filed 3 documents")
    logging.getLogger("httpx2._client").warning("connection retried")

    err = capsys.readouterr().err
    assert "HTTP Request" not in err
    assert "filed 3 documents" in err
    assert "connection retried" in err, "warnings still come through"


def test_verbose_brings_everything_back(monkeypatch, capsys) -> None:
    monkeypatch.delenv("SORTYOURPAPERYA_LOG_FILE", raising=False)
    _configure(verbose=True)

    logging.getLogger("httpx2._client").info("HTTP Request: POST ...")

    assert "HTTP Request" in capsys.readouterr().err


# ---- naming a document without reciting its id -----------------------------


def _two_papers(library):
    """Two documents that share a word, so a bare word is ambiguous."""
    from sortyourpaperya.db import Paper

    for file_id, title, tags, keywords in (
        ("78c64b3b8ef6", "Successor representations in human behavior",
         ["Psychology", "Research Methods"], ["kahn", "learning"]),
        ("aa11bb22cc33", "Successor features for transfer",
         ["Machine Learning", "RL"], ["barreto", "learning"]),
    ):
        paper = Paper(
            file_id=file_id, content_hash=f"sha-{file_id}",
            store_name=f"{file_id}__" + "__".join(tags),
            document_name="doc.pdf", title=title, tags=tags, keywords=keywords,
        )
        library.db.upsert(paper)
        library.document_dir(paper).mkdir(parents=True)
        (library.document_dir(paper) / "doc.pdf").write_bytes(b"%PDF")
    library.rebuild_tree()
    root = library.root
    library.close()
    return root


@pytest.fixture(autouse=True)
def _no_fzf(monkeypatch):
    """fzf is interactive and may be installed; never launch it by accident."""
    from sortyourpaperya import cli

    monkeypatch.setattr(cli, "_fzf_available", lambda: False)


def test_a_word_is_enough_to_name_a_document(library) -> None:
    from sortyourpaperya.library import Library

    root = _two_papers(library)

    result = _invoke(root, "retag", "kahn", "Cognitive Science/Computation")

    assert result.exit_code == 0, result.output
    with Library(root) as after:
        assert after.db.get("78c64b3b8ef6").tags == ["Cognitive Science", "Computation"]


def test_what_a_word_resolved_to_is_reported_on_stderr(library) -> None:
    """`sortyourpaperya note <id> --path` is composed as `$(...)`; stdout stays the path."""
    root = _two_papers(library)

    result = _invoke(root, "note", "kahn", "--path")

    assert result.stdout.strip().endswith("notes.md")
    assert result.stdout.strip().count("\n") == 0, "only the path is on stdout"
    assert "78c64b3b8ef6" in result.stderr


def test_a_named_note_is_created_under_that_name(library) -> None:
    root = _two_papers(library)

    result = _invoke(root, "note", "78c64b3b8ef6", "reading-log", "--path")

    assert result.exit_code == 0, result.output
    created = Path(result.stdout.strip())
    assert created.name == "reading-log.md", "a bare word is markdown"
    assert created.read_text().startswith("# "), "and is opened with a heading"


def test_a_json_note_is_created_as_valid_json(library) -> None:
    """A markdown heading in a `.json` note breaks the only thing it is for."""
    import json

    root = _two_papers(library)

    result = _invoke(root, "note", "78c64b3b8ef6", "extracted.json", "--path")

    assert result.exit_code == 0, result.output
    assert json.loads(Path(result.stdout.strip()).read_text()) == {}


def test_the_only_note_is_opened_whatever_it_is_called(library) -> None:
    root = _two_papers(library)
    _invoke(root, "note", "78c64b3b8ef6", "reading-log", "--path")

    result = _invoke(root, "note", "78c64b3b8ef6", "--path")

    assert result.exit_code == 0, result.output
    assert result.stdout.strip().endswith("reading-log.md")


def test_several_notes_are_listed_rather_than_one_being_guessed(library) -> None:
    """The wrong guess is written into by a caller that asked for "the" notes."""
    root = _two_papers(library)
    _invoke(root, "note", "78c64b3b8ef6", "reading-log", "--path")
    _invoke(root, "note", "78c64b3b8ef6", "extracted.json", "--path")

    result = _invoke(root, "note", "78c64b3b8ef6", "--path")

    assert result.exit_code == 1
    assert "several notes" in result.stderr
    assert "reading-log.md" in result.stderr and "extracted.json" in result.stderr
    assert result.stdout.strip() == "", "nothing that reads stdout gets a path"


def test_a_note_that_is_not_markdown_or_json_is_refused(library) -> None:
    root = _two_papers(library)

    result = _invoke(root, "note", "78c64b3b8ef6", "notes.txt", "--path")

    assert result.exit_code == 1
    assert "a note is" in result.stderr
    assert list((Path(root) / "store").rglob("notes.*")) == [], "and nothing was made"


def test_an_exact_id_is_never_searched_for(library) -> None:
    """So a document can always be named unambiguously.

    Here one document's id is also a word in another's keywords, which would
    make the id ambiguous if it went through the search.
    """
    from sortyourpaperya.db import Paper
    from sortyourpaperya.library import Library

    root = _two_papers(library)
    with Library(root) as lib:
        other = lib.db.get("aa11bb22cc33")
        lib.db.upsert(Paper(**{**other.__dict__, "keywords": ["78c64b3b8ef6"]}))

    result = _invoke(root, "retag", "78c64b3b8ef6", "Cognitive Science/Computation")

    assert result.exit_code == 0, result.output
    with Library(root) as after:
        assert after.db.get("78c64b3b8ef6").tags == ["Cognitive Science", "Computation"]
        assert after.db.get("aa11bb22cc33").tags == ["Machine Learning", "RL"]


def test_a_word_matching_nothing_is_an_error(library) -> None:
    root = _two_papers(library)

    result = _invoke(root, "note", "nothinglikethis", "--path")

    assert result.exit_code == 1
    assert "matches" in result.stderr


def test_an_ambiguous_word_lists_the_matches_and_changes_nothing(library) -> None:
    """With no picker available, printing them beats acting on a guess."""
    from sortyourpaperya.library import Library

    root = _two_papers(library)

    result = _invoke(root, "retag", "learning", "Cognitive Science/Computation")

    assert result.exit_code == 2
    assert "78c64b3b8ef6" in result.stderr and "aa11bb22cc33" in result.stderr
    assert "name one by its id" in result.stderr
    with Library(root) as after:
        assert after.db.get("78c64b3b8ef6").tags == ["Psychology", "Research Methods"]


def test_fzf_picks_among_the_matches(library, monkeypatch) -> None:
    from sortyourpaperya import cli
    from sortyourpaperya.library import Library

    root = _two_papers(library)
    monkeypatch.setattr(cli, "_fzf_available", lambda: True)
    shown: dict = {}

    def fake_run(argv, **kwargs):
        shown["argv"] = argv
        shown["lines"] = kwargs["input"]
        return subprocess.CompletedProcess(
            argv, 0, stdout="aa11bb22cc33\tMachine Learning / RL\tSuccessor features\n"
        )

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = _invoke(root, "retag", "learning", "Cognitive Science/Computation")

    assert result.exit_code == 0, result.output
    assert shown["argv"][0] == "fzf"
    assert "--with-nth" in shown["argv"], "the id column is not displayed"
    assert shown["lines"].count("\n") == 1, "both matches were offered"
    with Library(root) as after:
        assert after.db.get("aa11bb22cc33").tags == ["Cognitive Science", "Computation"]
        assert after.db.get("78c64b3b8ef6").tags == ["Psychology", "Research Methods"]


def test_backing_out_of_fzf_changes_nothing(library, monkeypatch) -> None:
    from sortyourpaperya import cli
    from sortyourpaperya.library import Library

    root = _two_papers(library)
    monkeypatch.setattr(cli, "_fzf_available", lambda: True)
    # 130 is what fzf exits with on an interrupt.
    monkeypatch.setattr(
        cli.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 130, stdout=""),
    )

    result = _invoke(root, "retag", "learning", "Cognitive Science/Computation")

    assert result.exit_code == 0, "backing out is a decision, not an error"
    with Library(root) as after:
        assert after.db.get("78c64b3b8ef6").tags == ["Psychology", "Research Methods"]


def test_unattended_delete_will_not_act_on_a_word(library) -> None:
    """Which document a word matches changes as the library grows."""
    from sortyourpaperya.library import Library

    root = _two_papers(library)

    result = _invoke(root, "remove", "kahn", "--yes")

    assert result.exit_code == 2
    assert "exact id" in result.stderr
    with Library(root) as after:
        assert after.db.get("78c64b3b8ef6") is not None


def test_delete_by_word_shows_what_it_found_before_asking(library) -> None:
    from sortyourpaperya.library import Library

    root = _two_papers(library)

    result = _invoke(root, "remove", "kahn", input="y\n")

    assert result.exit_code == 0, result.output
    assert "Successor representations in human behavior" in result.stdout
    with Library(root) as after:
        assert after.db.get("78c64b3b8ef6") is None
        assert after.db.get("aa11bb22cc33") is not None, "the other one is untouched"


def _stocked(library):
    """A small library with categories worth grouping and a year worth sorting."""
    from sortyourpaperya.db import Paper

    library.db.upsert_many(
        [
            Paper(
                file_id="aaaaaaaaaaaa",
                content_hash="sha-a",
                store_name="aaaaaaaaaaaa__AI__Transformers",
                document_name="paper.pdf",
                title="Attention Is All You Need",
                year=2017,
                size_bytes=900,
                pages_read=2,
                tags=["AI", "Transformers"],
                authors=["Ashish Vaswani"],
            ),
            Paper(
                file_id="bbbbbbbbbbbb",
                content_hash="sha-b",
                store_name="bbbbbbbbbbbb__Home__Utilities",
                document_name="bill.pdf",
                title="March Electricity Bill",
                year=2026,
                size_bytes=100,
                tags=["Home", "Utilities"],
            ),
            Paper(
                file_id="cccccccccccc",
                content_hash="sha-c",
                store_name="cccccccccccc__Home__Utilities",
                document_name="bill.pdf",
                title="April Electricity Bill",
                year=2026,
                size_bytes=200,
                tags=["Home", "Utilities"],
            ),
        ]
    )
    root = library.root
    library.close()
    return root


def test_categories_group_what_had_to_be_counted_by_hand(library) -> None:
    """A re-tag is decided from this; before, it meant reading the whole library."""
    root = _stocked(library)

    result = _invoke(root, "categories")

    assert "AI/Transformers" in result.output
    assert "2  Home/Utilities" in result.output


def test_a_record_carries_when_it_was_filed_and_how_large_it_is(library) -> None:
    """Without these the library has no time dimension: no "what did I file today"."""
    import json as _json

    root = _stocked(library)

    result = _invoke(root, "find", "attention", "--json")

    record = _json.loads(result.output)[0]
    assert record["size_bytes"] == 900
    assert record["pages_read"] == 2
    assert record["filed_at"].startswith("20") and record["filed_at"].endswith("+00:00")
    assert record["attributes"] == {}


def test_sorting_puts_the_answer_first_instead_of_a_hash_order(library) -> None:
    root = _stocked(library)

    result = _invoke(root, "list", "--sort", "size")

    lines = [line for line in result.output.splitlines() if line.strip()]
    assert lines[0].startswith("aaaaaaaaaaaa"), result.output


def test_an_unknown_sort_is_refused_by_name(library) -> None:
    root = _stocked(library)

    result = _invoke(root, "list", "--sort", "sideways")

    assert result.exit_code == 2
    assert "unknown sort" in result.output


def test_an_attribute_written_by_a_reader_survives_a_retag(library) -> None:
    """The point of attributes: what a reader records outlives what a model said."""
    import json as _json

    root = _stocked(library)
    _invoke(root, "attr", "aaaaaaaaaaaa", "verdict", "worth re-reading")

    _invoke(root, "retag", "aaaaaaaaaaaa", "AI/Attention")
    result = _invoke(root, "find", "attention", "--json")

    assert _json.loads(result.output)[0]["attributes"] == {
        "verdict": "worth re-reading"
    }


def test_one_attribute_prints_alone_so_it_can_be_captured(library) -> None:
    # `$(sortyourpaperya attr <id> doi)` has to yield the value and nothing else.
    root = _stocked(library)
    _invoke(root, "attr", "aaaaaaaaaaaa", "doi", "10.1000/xyz")

    result = _invoke(root, "attr", "aaaaaaaaaaaa", "doi")

    assert result.output.strip() == "10.1000/xyz"


def test_forgetting_an_attribute_that_is_not_there_is_an_error(library) -> None:
    root = _stocked(library)

    result = _invoke(root, "attr", "aaaaaaaaaaaa", "doi", "--unset")

    assert result.exit_code == 1
    assert "no 'doi'" in result.output


def test_sql_reaches_what_no_other_command_reports(library) -> None:
    import json as _json

    root = _stocked(library)

    result = _invoke(
        root, "sql", "SELECT title FROM papers ORDER BY size_bytes DESC", "--json"
    )

    assert _json.loads(result.output)[0]["title"] == "Attention Is All You Need"


def test_sql_refuses_to_write_and_says_why(library) -> None:
    root = _stocked(library)

    result = _invoke(root, "sql", "DELETE FROM papers")

    assert result.exit_code == 2
    assert "single reading statement" in result.output
    assert "3 document(s)" in _invoke(root, "list").output


def test_sql_says_when_it_capped_the_rows(library) -> None:
    # The same trap `find` has: a capped result that says nothing reads as all of it.
    root = _stocked(library)

    result = _invoke(root, "sql", "SELECT title FROM papers", "--limit", "1")

    assert "more than 1 rows matched" in result.output

"""Command line entry points for `sortyourpaperya`."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Sequence

import typer

from . import bib as bibs
from .bib import BibError, Bibliography
from .budget import Budget
from .config import (
    MAX_STEERING_CATEGORIES,
    DEFAULT_LABEL_CACHE_DAYS,
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    ConfigError,
    env_float,
    env_int,
    resolve_api_key,
    resolve_settings,
)
from .db import SORTS, Paper
from .ingest import ingest_folder
import os
import shutil
import subprocess

from .library import FilingMode, Library, LibraryError
from .llm import LlmError, OpenAiClient
from . import recategorize
from .naming import split_category
from .registry import RegistryError, WatchEntry, load_registry, registry_path
from .watch import watch as watch_loop
from .watchlock import WatchConflict

app = typer.Typer(
    help="sortyourpaperya: LLM ingest and folder watcher.",
    # A bare `sortyourpaperya` is someone asking what this does, so answer that rather
    # than an error saying a command is missing without naming one.
    no_args_is_help=True,
)


# Kept small enough that the whole history stays cheap to keep and to read:
# four files of 2MB is roughly a fortnight of a busy watcher.
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUP_COUNT = 3


@app.callback()
def _configure(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Set up logging for whichever command follows.

    Lines carry a timestamp because the watcher's log is read long after the
    fact, and an untimed line cannot say whether it is from this run or the one
    that crashed an hour ago — which is exactly the question a restart loop
    raises.

    With `SORTYOURPAPERYA_LOG_FILE` set — which is how the service runs — the log goes to
    that file through a rotating handler instead of to stderr, so a watcher left
    running for months cannot fill the disk. Instead of, not as well as: under
    launchd stderr is itself redirected to a file that nothing rotates, and
    writing every line to both would leave the unbounded copy growing exactly as
    before. What still reaches stderr is what logging never sees — a traceback
    from a crash — which is the one thing worth keeping unrotated.

    Only the service sets it. A rotating handler is safe for a single writer,
    and two processes rotating one file can lose each other's lines.
    """
    log_file = (os.environ.get("SORTYOURPAPERYA_LOG_FILE") or "").strip()
    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers: list[logging.Handler] = [
            RotatingFileHandler(
                path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
            )
        ]
    else:
        handlers = [logging.StreamHandler()]

    # Everything gets a handler, but only this package's own lines are turned
    # up to INFO. The HTTP client underneath logs a line per request at INFO,
    # which says nothing this tool does not already report and which lands in
    # the middle of `sortyourpaperya retag`'s prompt while it waits for an answer.
    #
    # Quieting the root and raising ours, rather than naming the loggers to
    # silence: the OpenAI SDK vendors its HTTP client, so that logger is called
    # `httpx2` today and whatever the next release vendors tomorrow. A rule
    # about our own name cannot go stale that way. Third-party warnings and
    # errors still come through, which is the half worth reading.
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger(__package__).setLevel(
        logging.DEBUG if verbose else logging.INFO
    )


@app.command()
def ingest(
    input_dir: Path = typer.Option(None, "--input", "-i", help="Folder of PDFs."),
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
    recursive: bool = typer.Option(None, "--recursive", "-r"),
    page_cutoff: int = typer.Option(None, "--page-cutoff", "-p"),
    batch_size: int = typer.Option(None, "--batch-size"),
    model: str = typer.Option(None, "--model", "-m"),
    mode: FilingMode = typer.Option(
        FilingMode.PREVIEW.value,
        "--mode",
        help="preview (default), copy to leave the source in place, or move.",
    ),
) -> None:
    """Read every not-yet-known document and file it into the library."""
    settings, client = _build(
        input_dir, library_dir, recursive, page_cutoff, batch_size, model
    )
    with Library(settings.output_dir) as library:
        report = asyncio.run(ingest_folder(settings, client, library, mode=mode))

        verb = "filed" if mode.writes else "would file"
        typer.echo(
            f"{verb} {report.processed} document(s); "
            f"{report.skipped_already_known} already known; "
            f"{len(report.skipped_oversized)} oversized; {len(report.failed)} failed"
        )
        if report.rescan and report.rescan.changed:
            typer.echo(
                f"  ({len(report.rescan.changed)} stored file(s) changed on disk; "
                "hashes refreshed)"
            )
        for filing in report.filed or report.planned:
            typer.echo(f"  {filing.describe()}")
        for path, reason in report.failed:
            typer.echo(f"  ! {path.name}: {reason}", err=True)
        if not mode.writes and report.planned:
            typer.echo("\nnothing was written; re-run with --mode copy or --mode move")

    if report.failed:
        raise typer.Exit(code=1)


@app.command()
def watch(
    name: str = typer.Argument(
        None, help="A watch declared in the registry. Omit to use the default."
    ),
    input_dir: Path = typer.Option(None, "--input", "-i", help="Folder to watch."),
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
    recursive: bool = typer.Option(None, "--recursive", "-r"),
    page_cutoff: int = typer.Option(None, "--page-cutoff", "-p"),
    batch_size: int = typer.Option(None, "--batch-size"),
    model: str = typer.Option(None, "--model", "-m"),
    mode: FilingMode = typer.Option(
        FilingMode.PREVIEW.value,
        "--mode",
        help="preview (default), copy to leave the source in place, or move.",
    ),
) -> None:
    """File new documents into the library whenever they settle in the input folder."""
    entry = _watch_entry(name) if (name or not input_dir) else None
    if entry is not None:
        input_dir = input_dir or entry.input_dir
        library_dir = library_dir or entry.library_dir
        if mode is FilingMode.PREVIEW:
            mode = entry.mode
    elif name:
        typer.echo(f"error: no watch named {name!r}", err=True)
        raise typer.Exit(code=2)

    settings, client = _build(
        input_dir, library_dir, recursive, page_cutoff, batch_size, model
    )
    try:
        with Library(settings.output_dir) as library:
            asyncio.run(watch_loop(settings, client, library, mode=mode))
    except WatchConflict as err:
        typer.echo(f"error: {err}", err=True)
        typer.echo(
            "  two watchers sharing a folder file the same document twice; "
            "stop the other one first.",
            err=True,
        )
        raise typer.Exit(code=2) from err
    except KeyboardInterrupt:
        typer.echo("stopped")


@app.command()
def tree(
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
) -> None:
    """Rebuild the symlink tree from the database."""
    settings = _settings(None, library_dir)
    with Library(settings.output_dir) as library:
        linked = library.rebuild_tree()
        typer.echo(f"linked {linked} document(s) under {library.tree_dir}")
        for paper in library.missing_files():
            typer.echo(
                f"  ! {paper.file_id}: {paper.store_name} is missing from the store",
                err=True,
            )
        litter = library.tree_litter()
        if litter:
            typer.echo(
                f"\n  {len(litter)} file(s) live in the tree, which is rebuilt and "
                "is not backed up with the library:",
                err=True,
            )
            for path in litter:
                typer.echo(f"    {path.relative_to(library.tree_dir)}", err=True)
            typer.echo(
                "  move them into the document's folder "
                "(`sortyourpaperya note <id>` opens one) to keep them.",
                err=True,
            )


def _resolve(library: Library, needle: str) -> Paper:
    """The document a command should act on, named by id or by words.

    Nobody remembers `78c64b3b8ef6`. `sortyourpaperya retag kahn` should be enough, and the
    search behind `sortyourpaperya find` already matches ids, titles, authors, keywords,
    tags, and years — so this is that search with a single answer required.

    An exact id wins outright and is never searched for, so a document can
    always be named unambiguously even when its id happens to read like a word
    in another document's title.
    """
    exact = library.db.get(needle)
    if exact is not None:
        return exact

    matches = library.db.search(needle)
    if not matches:
        typer.echo(f"error: nothing in the library matches {needle!r}", err=True)
        raise typer.Exit(code=1)
    if len(matches) == 1:
        found = matches[0]
        # To stderr: `sortyourpaperya note <id> --path` is composed as `$(...)`, and a
        # second line on stdout would land in the path.
        typer.echo(f"{found.file_id}  {_label(found)}", err=True)
        return found
    return _choose(matches, needle)


def _label(paper: Paper) -> str:
    return paper.title or paper.original_name or paper.store_name


def _choose(matches: list[Paper], needle: str) -> Paper:
    """Pick one of several matches, through fzf when it can be used."""
    if _fzf_available():
        chosen = _pick_with_fzf(matches, needle)
        if chosen is None:
            typer.echo("nothing chosen", err=True)
            raise typer.Exit(code=0)  # backing out is a decision, not an error
        return chosen

    typer.echo(f"error: {needle!r} matches {len(matches)} documents:", err=True)
    for paper in matches:
        tags = " / ".join(paper.tags) or "-"
        typer.echo(f"  {paper.file_id}  {tags:<40}  {_label(paper)}", err=True)
    typer.echo("\nadd a word to narrow it, or name one by its id.", err=True)
    raise typer.Exit(code=2)


def _fzf_available() -> bool:
    """Whether a picker can be shown at all.

    Both halves are needed: fzf draws its interface on the terminal, so with no
    terminal to draw on — cron, a pipeline, a test — there is nothing to pick
    with and the caller falls back to printing the matches.
    """
    if shutil.which("fzf") is None:
        return False
    try:
        with open("/dev/tty"):
            return True
    except OSError:
        return False


def _pick_with_fzf(matches: list[Paper], needle: str) -> Paper | None:
    """Show the matches in fzf and return the one chosen, or None.

    The id leads each line so the choice can be read back from it, and
    `--with-nth` keeps it out of what is displayed and searched. Only stdout is
    captured: fzf paints its interface on the terminal it opens itself, and
    capturing stderr as well would swallow it.
    """
    lines = "\n".join(
        f"{paper.file_id}\t{' / '.join(paper.tags) or '-'}\t{_label(paper)}"
        for paper in matches
    )
    try:
        result = subprocess.run(
            [
                "fzf",
                "--prompt", f"{needle}> ",
                "--delimiter", "\t",
                "--with-nth", "2..",
                "--height", "40%",
                "--reverse",
            ],
            input=lines,
            stdout=subprocess.PIPE,
            text=True,
        )
    except OSError as err:  # fzf vanished between the check and the call
        typer.echo(f"error: could not run fzf: {err}", err=True)
        raise typer.Exit(code=1) from err

    # 0 chose something; 130 is an interrupt, 1 no match, and both mean no.
    if result.returncode != 0 or not result.stdout.strip():
        return None
    chosen = result.stdout.split("\t", 1)[0].strip()
    return next((paper for paper in matches if paper.file_id == chosen), None)


@app.command()
def retag(
    file_id: str = typer.Argument(..., help="Document id, or words from its title, authors, or keywords."),
    category: str = typer.Argument(
        None, help="New category path, e.g. 'AI/Transformers'. Omit to ask the model."
    ),
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
    model: str = typer.Option(None, "--model", "-m"),
) -> None:
    """Give a document new tags, renaming its folder and moving its link.

    With a category, it is applied. Without one the model is asked where the
    document belongs and nothing is written until you accept — which is the way
    back from a label that steering got wrong. Each round you may say what the
    model is missing, and it answers again knowing it.
    """
    if category is None:
        _retag_by_asking(file_id, library_dir, model)
        return

    settings = _settings(None, library_dir)
    tags = split_category(category)
    if not tags:
        typer.echo(f"error: {category!r} has no usable tags", err=True)
        raise typer.Exit(code=2)

    with Library(settings.output_dir) as library:
        _apply_retag(library, _resolve(library, file_id).file_id, tags)


def _apply_retag(
    library: Library, file_id: str, tags: list[str], keywords: list[str] | None = None
) -> None:
    try:
        paper = library.retag(file_id, tags, keywords)
    except Exception as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    typer.echo(f"{paper.store_name}  [{' / '.join(paper.tags)}]")


def _retag_by_asking(file_id: str, library_dir: Path | None, model: str | None) -> None:
    """Ask the model where a document belongs, until the answer is accepted.

    Every suggestion is one request, so the loop is deliberately hand-driven:
    nothing is asked for, and nothing is spent, without someone saying so.

    The database is let go before the first request. What follows waits on a
    person and can go around any number of times, and holding the write lock
    across that would stop the watcher, which waits 30 seconds and then fails.
    """
    settings, client = _build(None, library_dir, None, None, None, model)

    with Library(settings.output_dir) as library:
        paper = _resolve(library, file_id)
        store_path = library.store_path(paper)
        steering = library.existing_categories(MAX_STEERING_CATEGORIES)
        library.release()

        label = paper.title or paper.original_name or paper.store_name
        typer.echo(f"{paper.file_id}  {label}")
        typer.echo(f"  now:        {' / '.join(paper.tags) or '-'}")

        rejected: list[str] = []
        # What the person last said about this document. The latest replaces
        # the one before it rather than piling up: someone correcting their own
        # steer — "no, the hardware side" — means the newer sentence, and two
        # contradictory instructions in one prompt are worth less than either.
        guidance = ""
        while True:
            try:
                suggestion = asyncio.run(
                    recategorize.suggest(
                        client,
                        paper,
                        store_path,
                        page_cutoff=settings.page_cutoff,
                        existing_categories=steering,
                        rejected=rejected,
                        guidance=guidance,
                    )
                )
            except LlmError as err:
                typer.echo(f"error: {err}", err=True)
                raise typer.Exit(code=1) from err

            tags = split_category(suggestion.category)
            if not tags:
                typer.echo(
                    f"  the model returned {suggestion.category!r}, which is not a "
                    "usable path; asking again",
                    err=True,
                )
                rejected.append(suggestion.category)
                continue

            typer.echo(
                f"\nsuggestion {len(rejected) + 1}: {' / '.join(tags)}"
            )
            if suggestion.keywords:
                typer.echo(f"  keywords:   {', '.join(suggestion.keywords)}")
            if guidance:
                # Named alongside the answer it produced, so a suggestion that
                # ignored what you said is visibly that rather than a mystery.
                typer.echo(f"  asked for:  {guidance}")

            choice, direction = _ask_what_to_do()
            if choice == "accept":
                with Library(settings.output_dir) as writable:
                    _apply_retag(
                        writable, paper.file_id, tags, suggestion.keywords or None
                    )
                return
            if choice == "cancel":
                typer.echo("nothing changed")
                return
            if direction:
                guidance = direction
            rejected.append(suggestion.category)


def _ask_what_to_do() -> tuple[str, str]:
    """Accept, say what the model is missing, ask again, or cancel.

    Returns the choice and, when one was given, the direction to ask under.
    Steering is read here rather than by the caller because an empty answer is
    not a decision: the menu comes back, and no request has been spent.
    """
    while True:
        answer = _prompt("[a]ccept, [s]teer, [r]egenerate, [c]ancel", default="a")
        first = answer.strip()[:1].lower()
        if first == "a":
            return "accept", ""
        if first == "r":
            return "regenerate", ""
        if first == "c":
            return "cancel", ""
        if first == "s":
            # A steer is a re-ask that carries a sentence, so the caller has
            # nothing extra to handle: it regenerates either way.
            direction = _prompt(
                "  what is it about, or where should it go?", default=""
            ).strip()
            if direction:
                return "regenerate", direction
            continue
        typer.echo("  please answer a, s, r, or c", err=True)


def _prompt(text: str, default: str = "") -> str:
    """Ask a question, treating an end of input as a cancel rather than a crash.

    Running this with no one at the keyboard — from cron, or with stdin closed
    — stops cleanly instead of failing with a traceback.
    """
    try:
        return typer.prompt(text, default=default)
    except (typer.Abort, EOFError):
        # Typer vendors click, so its re-export is the one public name for
        # the exception a closed stdin raises.
        typer.echo("\ncancelled; nothing changed")
        raise typer.Exit(code=0) from None


@app.command()
def note(
    file_id: str = typer.Argument(..., help="Document id, or words from its title, authors, or keywords."),
    name: str = typer.Argument(
        None,
        help="Which note. A bare word is markdown, so `reading-log` means "
        "`reading-log.md`. Left out: the document's only note, or `notes.md` "
        "when it has none.",
    ),
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
    path_only: bool = typer.Option(
        False, "--path", help="Print where the notes are instead of opening them."
    ),
) -> None:
    """Open one of a document's notes, creating it if it does not exist yet.

    A document may keep as many notes as its owner likes — any markdown or JSON
    file in its folder is one — so a name says which. Without one, a document
    with a single note opens that note whatever it is called, and a document
    with none gets `notes.md`. A document with several is ambiguous, so they are
    listed instead of one being picked.

    Notes live in the document's folder in the store, so they are backed up with
    it, follow it when it is re-tagged, and are reachable through the tree.

    `--path` prints the file and stops. A caller that means to write the notes
    itself has no use for an editor, and one inheriting an `$EDITOR` it cannot
    drive would be left holding a terminal open forever.
    """
    settings = _settings(None, library_dir)
    with Library(settings.output_dir) as library:
        paper = _resolve(library, file_id)
        if not library.document_dir(paper).is_dir():
            typer.echo(f"error: {paper.store_name} is missing from the store", err=True)
            raise typer.Exit(code=1)

        if name:
            try:
                path = library.note_path(paper, name)
            except LibraryError as err:
                typer.echo(f"error: {err}", err=True)
                raise typer.Exit(code=1)
        else:
            existing = library.notes(paper)
            if len(existing) > 1:
                # Picking one would be a guess, and the wrong guess is written
                # into by a caller that asked for "the" notes and got another.
                typer.echo(
                    f"error: {paper.file_id} has several notes. Name one:", err=True
                )
                for entry in existing:
                    typer.echo(f"  {entry.name}", err=True)
                raise typer.Exit(code=1)
            path = existing[0] if existing else library.note_path(paper)

        if not path.exists():
            # An empty JSON note has to parse, or it is broken for the only
            # thing JSON is kept for.
            if path.suffix.lower() == ".json":
                path.write_text("{}\n", encoding="utf-8")
            else:
                title = paper.title or paper.original_name or paper.store_name
                path.write_text(f"# {title}\n\n", encoding="utf-8")

    editor = None if path_only else (os.environ.get("VISUAL") or os.environ.get("EDITOR"))
    if editor:
        # Split so EDITOR="code -w" works, and let the editor own the terminal.
        subprocess.call([*editor.split(), str(path)])
    else:
        # No editor configured, so print the path for the caller to use.
        typer.echo(path)


bib_app = typer.Typer(
    help="Bibliographies: what you cite, and the .bib LaTeX reads.",
    no_args_is_help=True,
)
app.add_typer(bib_app, name="bib")


@bib_app.command("init")
def bib_init(
    name: str = typer.Argument(..., help="What the bibliography is called, e.g. 'PhD Thesis'."),
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
) -> None:
    """Start a new bibliography under the library's `bibs/` folder.

    A library holds as many as its owner writes papers: one per manuscript,
    keeping its own citation keys, so a key disambiguated in one does not
    change under another's feet. The folder is named by the slug of `name`, and its
    `bib.toml` records the name as given.
    """
    settings = _settings(None, library_dir)
    try:
        bibliography = bibs.create(settings.output_dir, name)
    except BibError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err

    typer.echo(f"{bibliography.slug}  {bibliography.id}  {bibliography.name}")
    typer.echo(f"  record  {bibliography.record_path}")
    typer.echo(f"  bib     {bibliography.bib_path}")


@bib_app.command("list")
def bib_list(
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
    as_json: bool = typer.Option(
        False, "--json", help="Print records instead of a table, for a program to read."
    ),
) -> None:
    """Every bibliography in the library, with its slug, id, and size."""
    settings = _settings(None, library_dir)
    found = bibs.all_bibs(settings.output_dir)
    if as_json:
        typer.echo(
            json.dumps(
                [
                    {
                        "id": entry.id,
                        "slug": entry.slug,
                        "name": entry.name,
                        "sources": len(entry.sources),
                        "path": str(entry.path),
                        "bib": str(entry.bib_path),
                        "created_at": _timestamp(entry.created_at_ms),
                    }
                    for entry in found
                ],
                indent=2,
            )
        )
        return
    if not found:
        typer.echo("no bibliographies yet; `sortyourpaperya bib init <name>` starts one")
        return
    for entry in found:
        typer.echo(
            f"{entry.slug:<24}  {entry.id}  {len(entry.sources):>4} source(s)  {entry.name}"
        )


@bib_app.command("add")
def bib_add(
    lib: str = typer.Option(
        None, "--lib", help="Which bibliography, by slug or id. Asked for if left out."
    ),
    cite: str = typer.Option(
        None,
        "--cite",
        help="The document to cite: its id, or words from it. Asked for if left out.",
    ),
    entry_type: str = typer.Option(
        None, "--type", help="BibTeX entry type. Worked out from the fields if left out."
    ),
    key: str = typer.Option(
        None, "--key", help="Citation key. Built from author, year, and title if left out."
    ),
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
) -> None:
    """Cite one of the library's documents in one of its bibliographies.

    Either half may be left off and is asked for: which bibliography, and which
    document. The document is named the way every other command names one — an
    id, or words from its title, authors, or keywords.

    Title, authors, and year come from the library; a DOI, a journal, a page
    range, and the rest come from the document's attributes, which is where
    `sortyourpaperya attr` puts what the model was never asked for. What the
    library does not know is added by editing `bib.toml` and running
    `sortyourpaperya bib build`.
    """
    settings = _settings(None, library_dir)

    # Both questions are asked before the database is opened. They wait on a
    # person, and holding the write lock across that would stop the watcher,
    # which waits 30 seconds and then fails.
    bibliography = _pick_bib(settings.output_dir, lib)
    needle = cite or _prompt("which document? (id, or words from it)")

    with Library(settings.output_dir) as library:
        paper = _resolve(library, needle)

        already = bibliography.source_for(paper.file_id)
        if already is not None:
            # Not an error: the document is cited, which is what was asked for.
            typer.echo(
                f"{paper.file_id} is already in {bibliography.slug} as {already.key}"
            )
            return

        source = bibs.source_from_paper(
            paper,
            library.db.attributes(paper.file_id),
            entry_type=entry_type,
            key=key,
        )

    written = bibliography.add(source)
    try:
        bibliography.save()
    except BibError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err

    typer.echo(f"{written.key}  @{written.entry_type}  {_label(paper)}")
    if key and written.key != key:
        typer.echo(f"  {key!r} was taken; cited as {written.key}", err=True)
    typer.echo(f"  {bibliography.bib_path}")


@bib_app.command("build")
def bib_build(
    lib: str = typer.Option(
        None, "--lib", help="Which bibliography, by slug or id. Asked for if left out."
    ),
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
) -> None:
    """Regenerate a bibliography's `.bib` from its `bib.toml`.

    The TOML is the record and the `.bib` is a projection of it, so this is what
    you run after editing the TOML by hand — to correct a title, to add a
    journal the library never knew. Nothing in the `.bib` survives it.
    """
    settings = _settings(None, library_dir)
    bibliography = _pick_bib(settings.output_dir, lib)
    try:
        bibliography.save()
    except BibError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    typer.echo(
        f"wrote {len(bibliography.sources)} entr"
        f"{'y' if len(bibliography.sources) == 1 else 'ies'} to {bibliography.bib_path}"
    )


def _pick_bib(library_root: Path, needle: str | None) -> Bibliography:
    """The bibliography a command should write to, asking when it was not told.

    Named ones are looked up and never guessed at. Unnamed, the choices are
    printed and one is asked for, with the first offered as the default — so a
    library with a single bibliography takes one keystroke, and a library with
    several cannot have the wrong one picked for it.
    """
    if needle:
        try:
            return bibs.open_bib(library_root, needle)
        except BibError as err:
            typer.echo(f"error: {err}", err=True)
            raise typer.Exit(code=1) from err

    found = bibs.all_bibs(library_root)
    if not found:
        typer.echo(
            "error: this library has no bibliography yet; "
            "`sortyourpaperya bib init <name>` starts one",
            err=True,
        )
        raise typer.Exit(code=1)

    for entry in found:
        typer.echo(f"  {entry.slug:<24}  {entry.id}  {entry.name}", err=True)
    answer = _prompt("which bibliography? (slug or id)", default=found[0].slug)
    try:
        return bibs.open_bib(library_root, answer)
    except BibError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err


@app.command("migrate-store")
def migrate_store(
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
) -> None:
    """Bring the store's layout and filenames in line with the current rules."""
    settings = _settings(None, library_dir)
    with Library(settings.output_dir) as library:
        moved = library.migrate_store_layout()
        for file_id in moved:
            paper = library.db.get(file_id)
            typer.echo(f"  folder: {file_id} -> {paper.store_name}/{paper.document_name}")

        renamed = library.refresh_document_names()
        for _, old, new in renamed:
            typer.echo(f"  rename: {old}\n       -> {new}")

        if not moved and not renamed:
            typer.echo("nothing to do; the store already matches the current rules")
            return
        typer.echo(
            f"moved {len(moved)} document(s) into folders, renamed {len(renamed)}"
        )
        library.rebuild_tree()
        typer.echo("tree rebuilt")


@app.command()
def fsck(
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
    adopt: bool = typer.Option(
        False, "--adopt", help="Give orphaned folders a row instead of only listing them."
    ),
) -> None:
    """Check the store and the database against each other.

    Reports documents whose file is gone, and folders the database does not know
    about — which is what an interrupted pass leaves behind, and what nothing
    else can find.
    """
    settings = _settings(None, library_dir)
    with Library(settings.output_dir) as library:
        missing = library.missing_files()
        orphans = library.orphans()

        for paper in missing:
            typer.echo(
                f"  missing: {paper.file_id} -> {paper.store_name} is not in the store",
                err=True,
            )

        for directory in orphans:
            if not adopt:
                typer.echo(f"  orphan : {directory.name}", err=True)
                continue
            try:
                paper = library.adopt(directory)
            except LibraryError as err:
                typer.echo(f"  orphan : {directory.name} — {err}", err=True)
                continue
            typer.echo(f"  adopted: {paper.file_id}  {paper.document_name}")

        if not missing and not orphans:
            typer.echo(f"{library.db.count()} document(s); store and database agree")
            return

        typer.echo(
            f"\n{library.db.count()} document(s); "
            f"{len(missing)} missing, {len(orphans)} orphaned"
        )
        if orphans and not adopt:
            typer.echo("re-run with --adopt to bring the orphaned folders back in.")
            typer.echo(
                "Their title, authors, and year are not recoverable — those "
                "lived only in the database."
            )
        raise typer.Exit(code=1)


@app.command()
def scan(
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
) -> None:
    """Re-check stored files and refresh the hashes of any edited in place."""
    settings = _settings(None, library_dir)
    with Library(settings.output_dir) as library:
        report = library.rescan()
        typer.echo(
            f"checked {report.checked} document(s); "
            f"{report.rehashed} restatted; {len(report.changed)} changed; "
            f"{len(report.missing)} missing"
        )
        for file_id, old, new in report.changed:
            typer.echo(f"  {file_id}: {old[:12]} -> {new[:12]}")
        for paper in report.missing:
            typer.echo(
                f"  ! {paper.file_id}: {paper.store_name} is missing from the store",
                err=True,
            )


@app.command()
def remove(
    file_id: str = typer.Argument(..., help="Document id, or words from its title, authors, or keywords. An exact id with --yes."),
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Delete a document from the library: its link, its file, and its record.

    Words find a document here as they do elsewhere, and what they found is
    shown before anything is deleted. With `--yes` there is nothing to show, so
    the document must be named by its exact id: which document a word matches
    changes as the library grows, and an unattended delete must not.
    """
    settings = _settings(None, library_dir)
    with Library(settings.output_dir) as library:
        if yes and library.db.get(file_id) is None:
            typer.echo(
                f"error: --yes needs an exact id, and {file_id!r} is not one.",
                err=True,
            )
            typer.echo(
                "  what a word matches changes as the library grows; "
                "run it without --yes to see what this finds.",
                err=True,
            )
            raise typer.Exit(code=2)

        paper = _resolve(library, file_id)
        if not yes:
            typer.echo(f"{paper.store_name}\n  {_label(paper)}")
            # The stored file is the only copy when it arrived by move, so this
            # is not undoable.
            typer.confirm("permanently delete this document?", abort=True)

        library.remove(paper.file_id)
        typer.echo(f"removed {paper.file_id}")


@app.command()
def backup(
    destination: Path = typer.Argument(..., help="Empty folder to copy the library into."),
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
) -> None:
    """Copy the store and the database somewhere safe, together.

    They are only useful as a pair, and the tree is not copied because `sortyourpaperya
    tree` rebuilds it. Run it from cron or a launchd agent for a nightly copy.
    """
    settings = _settings(None, library_dir)
    with Library(settings.output_dir) as library:
        try:
            report = library.backup(destination)
        except LibraryError as err:
            typer.echo(f"error: {err}", err=True)
            raise typer.Exit(code=1) from err

    typer.echo(f"backed up {report.documents} document(s) to {report.destination}")
    typer.echo(f"  database  {report.database.name}")
    typer.echo(f"  store     {report.bytes_copied / 1e6:.1f} MB")
    if report.bibliographies:
        plural = "y" if report.bibliographies == 1 else "ies"
        typer.echo(f"  bibs      {report.bibliographies} bibliograph{plural}")
    typer.echo("  the tree is not copied; `sortyourpaperya tree` rebuilds it")


@app.command()
def cache(
    forget: bool = typer.Option(
        False, "--forget", help="Throw the banked answers away."
    ),
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
) -> None:
    """Show the model answers this library has already paid for.

    An answer is kept against the document's contents, so a pass that dies
    between paying and filing does not buy the same one again when it restarts.
    `--forget` clears them, which costs money rather than losing anything: the
    next pass asks the model again.
    """
    settings = _settings(None, library_dir)
    with Library(settings.output_dir) as library:
        if forget:
            typer.echo(f"forgot {library.db.forget_model_answers()} banked answer(s)")
            return
        typer.echo(f"{library.db.count_model_answers()} banked answer(s)")

    days = env_int("SYP_LABEL_CACHE_DAYS", DEFAULT_LABEL_CACHE_DAYS)
    if days == 0:
        typer.echo("reuse is off (SYP_LABEL_CACHE_DAYS=0); every pass asks again")
    elif days < 0:
        typer.echo("kept indefinitely (SYP_LABEL_CACHE_DAYS is negative)")
    else:
        typer.echo(
            f"reused for {days} day(s), then asked again — a label is a choice "
            "made against the categories the library had at the time"
        )


@app.command()
def budget(
    reset: bool = typer.Option(False, "--reset", help="Forget the recorded spend."),
) -> None:
    """Show what the last 24 hours have cost at the API, against the ceiling."""
    ledger = Budget()
    if reset:
        ledger.reset()
        typer.echo("spend record cleared")
        return

    used = ledger.usage()
    limits = ledger.limits
    typer.echo(f"last 24 hours, from {ledger.path}")
    typer.echo(f"  requests  {used.requests:>8} / {_ceiling(limits.requests_per_day)}")
    typer.echo(f"  tokens    {used.tokens:>8} / {_ceiling(limits.tokens_per_day)}")
    if limits.unlimited:
        typer.echo(
            "\nboth ceilings are off. The watcher restarts on failure, so "
            "nothing here would stop a loop paying for the same folder."
        )
    else:
        typer.echo(
            "\nraise with SYP_MAX_REQUESTS_PER_DAY / SYP_MAX_TOKENS_PER_DAY; "
            "0 turns a ceiling off."
        )


def _ceiling(limit: int) -> str:
    return "unlimited" if limit <= 0 else str(limit)


@app.command("watch-target", hidden=True)
def watch_target(
    name: str = typer.Argument(None, help="Watch name; omit for the default."),
) -> None:
    """Print a watch's input and library, one per line.

    For the service script, so it reads the registry through the same code the
    rest of the tool does instead of parsing TOML in shell.
    """
    entry = _watch_entry(name)
    if entry is None:
        typer.echo(
            "no watch to install: declare one in "
            f"{registry_path()} (see `sortyourpaperya watches`), or pass the folders",
            err=True,
        )
        raise typer.Exit(code=2)
    typer.echo(entry.input_dir)
    typer.echo(entry.library_dir)


@app.command()
def watches() -> None:
    """Show the declared watches, and which are running."""
    from .watchlock import locks_dir

    try:
        registry = load_registry()
    except RegistryError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=2) from err

    if not registry.watches:
        typer.echo(f"no watches declared in {registry.path}\n")
        typer.echo("Declare one:\n")
        typer.echo('  [watch.downloads]')
        typer.echo('  input   = "~/Downloads"')
        typer.echo('  library = "~/Documents/sortyourpaperya-library"')
        typer.echo('  mode    = "copy"')
        return

    claimed = _claimed_folders(locks_dir())
    default = registry.default()
    for entry in registry.watches.values():
        marks = []
        if default is not None and entry.name == default.name:
            marks.append("default")
        holder = claimed.get(entry.input_dir) or claimed.get(entry.library_dir)
        marks.append(f"running pid {holder}" if holder else "stopped")
        typer.echo(f"{entry.name}  [{', '.join(marks)}]")
        typer.echo(f"    input   {entry.input_dir}")
        typer.echo(f"    library {entry.library_dir}")
        typer.echo(f"    mode    {entry.mode.value}")
    typer.echo(f"\n{registry.path}")


def _claimed_folders(directory: Path) -> dict[Path, int]:
    """Folders claimed by a watcher that is still running, by path."""
    from .watchlock import live_claims

    return live_claims(directory)


_SORT_HELP = f"Order of the results: {', '.join(SORTS)}."


@app.command("list")
def list_papers(
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
    sort: str = typer.Option("id", "--sort", "-s", help=_SORT_HELP),
    as_json: bool = typer.Option(
        False, "--json", help="Print records instead of a table, for a program to read."
    ),
) -> None:
    """List what the library holds."""
    settings = _settings(None, library_dir)
    _check_sort(sort)
    with Library(settings.output_dir) as library:
        _report(library, library.db.all_papers(sort=sort), as_json=as_json)


def _check_sort(sort: str) -> None:
    """Refuse an unknown sort at the CLI, where the name was typed."""
    if sort not in SORTS:
        typer.echo(
            f"error: unknown sort {sort!r}; one of {', '.join(SORTS)}", err=True
        )
        raise typer.Exit(code=2)


@app.command()
def find(
    query: str = typer.Argument(
        ..., help="Words to match against titles, authors, keywords, tags, and ids."
    ),
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
    limit: int = typer.Option(
        20, "--limit", "-n", min=0, help="Most to show; 0 for all."
    ),
    sort: str = typer.Option("id", "--sort", "-s", help=_SORT_HELP),
    as_json: bool = typer.Option(
        False, "--json", help="Print records instead of a table, for a program to read."
    ),
) -> None:
    """Find documents by title, author, keyword, tag, year, or id.

    Every word has to match somewhere, so adding one narrows the result. The
    records `--json` prints carry the path to each document, which is how a
    reader gets from a search to the file itself.
    """
    settings = _settings(None, library_dir)
    _check_sort(sort)
    with Library(settings.output_dir) as library:
        # One more than will be shown, which is how the cap is noticed. A
        # capped result that says nothing about it reads as the whole answer,
        # and the reader stops looking for the document that was cut off.
        papers = library.db.search(
            query, limit=limit + 1 if limit else None, sort=sort
        )
        capped = bool(limit) and len(papers) > limit
        _report(library, papers[:limit] if capped else papers, as_json=as_json)
        if capped:
            typer.echo(
                f"more than {limit} matched; narrow the search, "
                "or pass --limit 0 for all of them",
                err=True,
            )


@app.command()
def categories(
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
    as_json: bool = typer.Option(
        False, "--json", help="Print records instead of a table, for a program to read."
    ),
) -> None:
    """Show every category the library uses, and how many documents are under it.

    What a re-tag should be decided from. The alternative was listing the whole
    library and counting it by hand, which is a lot of reading to answer a
    question the database can group.
    """
    settings = _settings(None, library_dir)
    with Library(settings.output_dir) as library:
        counted = library.db.category_counts()
    if as_json:
        typer.echo(
            json.dumps(
                [{"category": path, "documents": count} for path, count in counted],
                indent=2,
            )
        )
        return
    for path, count in counted:
        typer.echo(f"{count:>5}  {path}")
    typer.echo(f"\n{len(counted)} categor{'y' if len(counted) == 1 else 'ies'}")


@app.command()
def attr(
    file_id: str = typer.Argument(..., help="Document id, or words from its title, authors, or keywords."),
    key: str = typer.Argument(None, help="Attribute to read or write. Omit for all of them."),
    value: str = typer.Argument(None, help="What to set it to. Omit to read it."),
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
    unset: bool = typer.Option(False, "--unset", help="Forget the attribute instead."),
    as_json: bool = typer.Option(
        False, "--json", help="Print records instead of a table, for a program to read."
    ),
) -> None:
    """Read and write a document's own fields: anything the library has no column for.

    A DOI, a venue, a verdict, the day it was read — whatever the reader wants
    kept about a document that the model was never asked for. They live in the
    database beside the document's own labels, so they survive re-tagging and a
    rebuilt tree, are deleted with the document, and are not touched by ingest:
    a document re-read or re-hashed keeps everything written here.

    Keys are the caller's to choose and are not interpreted. Storing a value
    replaces what was there.
    """
    settings = _settings(None, library_dir)
    with Library(settings.output_dir) as library:
        paper = _resolve(library, file_id)

        if unset:
            if key is None:
                typer.echo("error: --unset needs the attribute to forget", err=True)
                raise typer.Exit(code=2)
            if not library.db.unset_attribute(paper.file_id, key):
                typer.echo(f"error: {paper.file_id} has no {key!r}", err=True)
                raise typer.Exit(code=1)
            return

        if value is not None:
            library.db.set_attribute(paper.file_id, key, value)
            return

        held = library.db.attributes(paper.file_id)
        if key is not None:
            if key not in held:
                typer.echo(f"error: {paper.file_id} has no {key!r}", err=True)
                raise typer.Exit(code=1)
            # Alone on stdout, so `$(sortyourpaperya attr <id> doi)` yields just the value.
            typer.echo(held[key] if held[key] is not None else "")
            return

    if as_json:
        typer.echo(json.dumps(held, indent=2))
        return
    for name, held_value in held.items():
        typer.echo(f"{name}\t{held_value if held_value is not None else ''}")


@app.command()
def sql(
    statement: str = typer.Argument(..., help="One reading statement, e.g. \"SELECT title FROM papers\"."),
    library_dir: Path = typer.Option(None, "--library", "-o", help="Library folder."),
    limit: int = typer.Option(
        200, "--limit", "-n", min=0, help="Most rows to show; 0 for all."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Print records instead of a table, for a program to read."
    ),
) -> None:
    """Query the library database directly, for what no other command reports.

    The commands above answer the questions worth having a command for. This is
    the rest of the database: when a document was filed, how large it is, how
    many pages were read, and the page text banked in `model_answers`.

    Only a single reading statement is accepted, which is a guard against a
    query meant to count documents deleting them — not a sandbox. The tables
    are `papers`, `paper_tags`, `paper_authors`, `paper_keywords`,
    `paper_attributes`, and `model_answers`; `DESCRIBE <table>` shows the
    columns of any of them.

    Opening the file directly is not an alternative while a watcher is running:
    DuckDB allows one process, and it refuses a second connection even to read.
    """
    settings = _settings(None, library_dir)
    with Library(settings.output_dir) as library:
        try:
            # One more than will be shown, which is how the cap is noticed.
            columns, rows = library.db.select(
                statement, limit=limit + 1 if limit else None
            )
        except ValueError as err:
            typer.echo(f"error: {err}", err=True)
            raise typer.Exit(code=2) from err
        except Exception as err:  # whatever DuckDB made of the statement
            typer.echo(f"error: {err}", err=True)
            raise typer.Exit(code=1) from err

    capped = bool(limit) and len(rows) > limit
    shown = rows[:limit] if capped else rows

    if as_json:
        typer.echo(
            json.dumps(
                [dict(zip(columns, row)) for row in shown],
                indent=2,
                default=str,
            )
        )
    else:
        typer.echo("\t".join(columns))
        for row in shown:
            typer.echo("\t".join("" if cell is None else str(cell) for cell in row))
        typer.echo(f"\n{len(shown)} row(s)")
    if capped:
        typer.echo(
            f"more than {limit} rows matched; add a LIMIT, "
            "or pass --limit 0 for all of them",
            err=True,
        )


def _report(library: Library, papers: Sequence[Paper], *, as_json: bool) -> None:
    """Show what was found, either to a person or to whatever called this."""
    if as_json:
        # One query for the whole page rather than one per document: `list
        # --json` describes the entire library, and each record already costs
        # three queries to hydrate.
        held = library.db.attributes_for([paper.file_id for paper in papers])
        typer.echo(
            json.dumps(
                [
                    _describe(library, paper, held.get(paper.file_id, {}))
                    for paper in papers
                ],
                indent=2,
            )
        )
        return
    for paper in papers:
        tags = " / ".join(paper.tags) or "-"
        typer.echo(f"{paper.file_id}  {tags:<40}  {paper.title or paper.original_name}")
    typer.echo(f"\n{len(papers)} document(s) in {library.root}")


def _describe(
    library: Library, paper: Paper, attributes: dict[str, str | None] | None = None
) -> dict:
    """One document as a record, with the paths that lead to it.

    Paths are absolute because the caller is not standing anywhere in
    particular, and a record whose file cannot be opened from it is only half
    an answer.

    `attributes` is passed in when the caller has already fetched them for a
    whole page of results; left out, this fetches the one document's.
    """
    return {
        "id": paper.file_id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "category": "/".join(paper.tags),
        "tags": paper.tags,
        "keywords": paper.keywords,
        "document": str(library.store_path(paper).resolve()),
        "folder": str(library.document_dir(paper).resolve()),
        "notes": [str(entry.resolve()) for entry in library.notes(paper)],
        "original_name": paper.original_name,
        "from_page_images": paper.from_page_images,
        "filed_at": _timestamp(paper.created_at_ms),
        "updated_at": _timestamp(paper.updated_at_ms),
        "size_bytes": paper.size_bytes,
        "pages_read": paper.pages_read,
        "attributes": (
            library.db.attributes(paper.file_id) if attributes is None else attributes
        ),
    }


def _timestamp(milliseconds: int | None) -> str | None:
    """A recorded time as UTC ISO 8601, which is the same everywhere it is read."""
    if milliseconds is None:
        return None
    moment = datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc)
    return moment.replace(microsecond=0).isoformat()


def _watch_entry(name: str | None) -> WatchEntry | None:
    """The declared watch a command should act on, if the registry names one."""
    try:
        registry = load_registry()
    except RegistryError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=2) from err
    return registry.get(name) if name else registry.default()


def _resolved_library(library_dir: Path | None, watch_name: str | None = None) -> Path | None:
    """Where a command should look, when it was not told.

    CLI beats the environment beats the registry, which is the order the rest of
    the settings already resolve in.
    """
    if library_dir is not None or os.environ.get("SYP_OUTPUT"):
        return library_dir
    try:
        entry = _watch_entry(watch_name)
    except typer.Exit:
        raise
    return entry.library_dir if entry else None


def _settings(input_dir: Path | None, library_dir: Path | None, watch_name: str | None = None):
    try:
        return resolve_settings(input_dir, _resolved_library(library_dir, watch_name))
    except ConfigError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=2) from err


def _build(
    input_dir: Path | None,
    library_dir: Path | None,
    recursive: bool | None,
    page_cutoff: int | None,
    batch_size: int | None,
    model: str | None,
):
    try:
        settings = resolve_settings(
            input_dir,
            _resolved_library(library_dir),
            recursive=recursive,
            page_cutoff=page_cutoff,
            keyword_batch_size=batch_size,
            model=model,
        )
        return settings, OpenAiClient(
            resolve_api_key(),
            settings.model,
            budget=Budget(),
            max_retries=env_int("SYP_LLM_MAX_RETRIES", DEFAULT_LLM_MAX_RETRIES),
            timeout_seconds=env_float(
                "SYP_LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS
            ),
        )
    except ConfigError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=2) from err


def main() -> None:
    app()


if __name__ == "__main__":
    main()

"""A bibliography's reading list: what to read first, and why.

An agent ranks what a bibliography cites and says why, and a person walks the
list. The two halves live here the way `pick` splits them: a **model** that
reads, checks, and writes the list without a terminal, and a **screen** at the
bottom that draws it and holds no rule worth testing.

The list is kept beside the record as `reading.toml`. TOML rather than the JSON
the agent hands over, because a JSON file in a bibliography's folder is a note
about the manuscript, and this is not one.

What the agent hands over is replaced whole, never merged: a ranking is a
judgement about the entire list, and splicing a new one into an old one makes an
order nobody decided. The one thing carried across is what has been read
already, because that is the reader's and not the agent's.

    {
      "chain_of_thought": ["how the list as a whole was ordered", "..."],
      "items": [
        {
          "cite": "vaswani2017attention",
          "rank": 1,
          "rationale": "why this one, in a line",
          "chain_of_thought": ["the steps that put it here", "..."]
        }
      ]
    }

`cite` is a citation key or a `file_id` in this bibliography and nothing looser:
an agent writing a list should not have its words searched for. `rank` may be
left off, and is then the item's position.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import tomli_w

from .bib import BibError, Bibliography, Source
from .db import now_ms

READING_FILE = "reading.toml"

# A step that arrives already numbered, or bulleted, loses the marker: the list
# is numbered when it is shown, and keeping both would print "1. 1. ...".
_MARKER = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•])\s+")


@dataclass
class Entry:
    """One thing to read, where it falls, and the reasoning that put it there."""

    key: str
    rank: int
    rationale: str
    chain_of_thought: list[str] = field(default_factory=list)
    read_at_ms: int | None = None


@dataclass
class ReadingList:
    """A bibliography's ranked reading, in rank order."""

    entries: list[Entry] = field(default_factory=list)
    chain_of_thought: list[str] = field(default_factory=list)
    updated_at_ms: int = 0

    def entry(self, key: str) -> Entry | None:
        return next((e for e in self.entries if e.key == key), None)


def reading_path(bibliography: Bibliography) -> Path:
    return bibliography.path / READING_FILE


def load(bibliography: Bibliography) -> ReadingList | None:
    """The bibliography's reading list, or None if no agent has written one.

    Raises:
        BibError: if the file is there and is not a reading list.
    """
    path = reading_path(bibliography)
    if not path.is_file():
        return None
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise BibError(f"could not read {path}: {err}") from err

    declared = raw.get("entry") or []
    if not isinstance(declared, list):
        raise BibError(f"{path}: `entry` must be a list of [[entry]] tables")
    entries = []
    for position, body in enumerate(declared, start=1):
        if not isinstance(body, dict) or not str(body.get("key") or "").strip():
            raise BibError(f"{path}: entry {position} has no `key`")
        rank = body.get("rank", position)
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise BibError(f"{path}: entry {position} has a `rank` that is not a number")
        read_at = body.get("read_at_ms")
        entries.append(
            Entry(
                key=str(body["key"]),
                rank=rank,
                rationale=str(body.get("rationale") or ""),
                chain_of_thought=_stored_steps(body.get("chain_of_thought"), path),
                read_at_ms=read_at if isinstance(read_at, int) else None,
            )
        )
    entries.sort(key=lambda e: e.rank)
    updated = raw.get("updated_at_ms")
    return ReadingList(
        entries=entries,
        chain_of_thought=_stored_steps(raw.get("chain_of_thought"), path),
        updated_at_ms=updated if isinstance(updated, int) else 0,
    )


def _stored_steps(value: object, path: Path) -> list[str]:
    """Steps as written in the file; a lone string is one step, not one per letter."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise BibError(f"{path}: `chain_of_thought` must be a list of strings")
    return [str(step) for step in value]


def save(bibliography: Bibliography, reading: ReadingList) -> Path:
    """Write the list beside the record, and say where."""
    document: dict[str, object] = {"updated_at_ms": reading.updated_at_ms}
    if reading.chain_of_thought:
        document["chain_of_thought"] = reading.chain_of_thought
    tables = []
    for entry in reading.entries:
        table: dict[str, object] = {
            "key": entry.key,
            "rank": entry.rank,
            "rationale": entry.rationale,
            "chain_of_thought": entry.chain_of_thought,
        }
        if entry.read_at_ms is not None:
            table["read_at_ms"] = entry.read_at_ms
        tables.append(table)
    if tables:
        document["entry"] = tables
    path = reading_path(bibliography)
    # Written beside and renamed over, so a reader -- the screen, an agent's
    # `--json` -- never meets a half-written list, and an interruption leaves
    # the old one rather than an empty file.
    # Named per process, so two writers at once -- an agent and the screen --
    # cannot rename each other's half-written file into place.
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(tomli_w.dumps(document), encoding="utf-8")
        os.replace(staged, path)
    except OSError as err:
        staged.unlink(missing_ok=True)
        raise BibError(f"could not write {path}: {err}") from err
    return path


def _steps(value: object, where: str, *, required: bool) -> list[str]:
    """A chain of thought as a list of steps, unnumbered.

    Raises:
        BibError: if it is not a list of strings, or is empty when required.
    """
    if value is None:
        value = []
    if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
        raise BibError(f"{where}: `chain_of_thought` must be a list of strings")
    steps = [_MARKER.sub("", s).strip() for s in value]
    steps = [s for s in steps if s]
    if required and not steps:
        raise BibError(f"{where}: `chain_of_thought` needs at least one step")
    return steps


def _source(bibliography: Bibliography, cite: str) -> Source:
    """The source a `cite` names: exactly its key, or exactly its document."""
    found = next((s for s in bibliography.sources if s.key == cite), None)
    found = found or bibliography.source_for(cite)
    if found is None:
        raise BibError(
            f"{bibliography.slug} does not cite {cite!r}; a reading list ranks "
            f"what the bibliography cites, and `sortyourpaperya bib add --lib "
            f"{bibliography.slug} --cite <id>` adds to it"
        )
    return found


def from_agent(
    bibliography: Bibliography, payload: object, previous: ReadingList | None
) -> ReadingList:
    """Check what an agent handed over and turn it into a reading list.

    Everything is checked before anything is kept, so a list with one bad item
    is refused whole rather than written short. Ranks must be distinct; they are
    then renumbered 1..n, so gaps the agent left (1, 2, 5) read as an order and
    not as missing entries.

    Raises:
        BibError: naming the first thing wrong with it.
    """
    if not isinstance(payload, dict):
        raise BibError("a reading list is a JSON object with an `items` list")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise BibError("a reading list needs a non-empty `items` list")

    ranked: list[tuple[int, Entry]] = []
    seen_keys: dict[str, int] = {}
    seen_ranks: dict[int, int] = {}
    for position, item in enumerate(items, start=1):
        where = f"item {position}"
        if not isinstance(item, dict):
            raise BibError(f"{where}: must be an object")
        cite = str(item.get("cite") or "").strip()
        if not cite:
            raise BibError(f"{where}: needs `cite`, a citation key or a file_id")
        source = _source(bibliography, cite)
        if source.key in seen_keys:
            raise BibError(
                f"{where}: {source.key} is already item {seen_keys[source.key]}"
            )
        seen_keys[source.key] = position

        rank = item.get("rank", position)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise BibError(f"{where}: `rank` must be a whole number from 1")
        if rank in seen_ranks:
            raise BibError(f"{where}: rank {rank} is already item {seen_ranks[rank]}")
        seen_ranks[rank] = position

        rationale = item.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise BibError(f"{where}: needs a `rationale`")

        before = previous.entry(source.key) if previous else None
        ranked.append(
            (
                rank,
                Entry(
                    key=source.key,
                    rank=rank,
                    rationale=rationale.strip(),
                    chain_of_thought=_steps(
                        item.get("chain_of_thought"), where, required=True
                    ),
                    read_at_ms=before.read_at_ms if before else None,
                ),
            )
        )

    ranked.sort(key=lambda pair: pair[0])
    entries = [entry for _, entry in ranked]
    for number, entry in enumerate(entries, start=1):
        entry.rank = number
    return ReadingList(
        entries=entries,
        chain_of_thought=_steps(
            payload.get("chain_of_thought"), "the list", required=False
        ),
        updated_at_ms=now_ms(),
    )


def unranked(bibliography: Bibliography, reading: ReadingList | None) -> list[str]:
    """The keys the bibliography cites that the list does not place."""
    placed = {e.key for e in reading.entries} if reading else set()
    return [s.key for s in bibliography.sources if s.key not in placed]


def mark(bibliography: Bibliography, key: str) -> ReadingList:
    """Toggle one entry read in the list as it is on disk now, and save it.

    Re-read rather than written from memory: the list may be open on a screen
    while an agent replaces it, and writing the screen's copy back would undo
    the agent's ranking for the sake of one checkbox.

    Raises:
        BibError: if the list is gone, unreadable, or no longer has the entry.
    """
    current = load(bibliography)
    if current is None:
        raise BibError("the reading list is gone")
    toggle_read(current, key)
    save(bibliography, current)
    return current


def toggle_read(reading: ReadingList, key: str) -> Entry:
    """Mark one entry read, or unread again."""
    entry = reading.entry(key)
    if entry is None:
        raise BibError(f"{key!r} is not on the reading list")
    entry.read_at_ms = None if entry.read_at_ms is not None else now_ms()
    return entry


def title_of(bibliography: Bibliography, key: str) -> str:
    """What to call an entry: its title, its author and year, and failing all, its key."""
    source = next((s for s in bibliography.sources if s.key == key), None)
    if source is None:
        return f"{key} (no longer cited)"
    title = str(source.fields.get("title") or key)
    authors = source.fields.get("author")
    parts = []
    if isinstance(authors, list) and authors:
        surname = str(authors[0]).split()[-1]
        parts.append(f"{surname} et al." if len(authors) > 1 else surname)
    if source.fields.get("year"):
        parts.append(str(source.fields["year"]))
    byline = ", ".join(parts)
    return f"{title} ({byline})" if byline else title


def numbered(steps: Sequence[str], indent: str = "") -> list[str]:
    return [f"{indent}{n}. {step}" for n, step in enumerate(steps, start=1)]


def as_text(bibliography: Bibliography, reading: ReadingList) -> str:
    """The whole list for reading off a page: order, reasons, and the steps behind them."""
    lines = [f"Reading list for {bibliography.name}"]
    if reading.chain_of_thought:
        lines += ["", "How it was ordered:", *numbered(reading.chain_of_thought, "  ")]
    for entry in reading.entries:
        mark = "[x]" if entry.read_at_ms is not None else "[ ]"
        lines += [
            "",
            f"{mark} {entry.rank}. {entry.key}  {title_of(bibliography, entry.key)}",
            f"      {entry.rationale}",
            *numbered(entry.chain_of_thought, "      "),
        ]
    left = unranked(bibliography, reading)
    if left:
        lines += ["", f"not ranked: {', '.join(left)}"]
    return "\n".join(lines)


# ---- the screen ------------------------------------------------------------


class Screen:
    """Draws the list and turns keystrokes into calls on the model."""

    def __init__(self, bibliography: Bibliography, reading: ReadingList) -> None:
        self.bibliography = bibliography
        self.reading = reading
        self.cursor = 0
        self.expanded: set[str] = set()
        self.message = ""

    def under_cursor(self) -> Entry | None:
        entries = self.reading.entries
        return entries[self.cursor] if 0 <= self.cursor < len(entries) else None

    def handle(self, key: str) -> str | None:
        """Apply one keystroke. Returns "open" or "read" when the key asks for one.

        Both are handed back rather than done here: opening needs the paths and
        marking needs to write the file, and neither belongs to a thing that
        draws.
        """
        entries = self.reading.entries
        if not entries:
            return None
        here = entries[self.cursor]
        if key in ("j",):
            self.cursor = min(self.cursor + 1, len(entries) - 1)
        elif key in ("k",):
            self.cursor = max(self.cursor - 1, 0)
        elif key in ("l", "\n", "\r", " "):
            self.expanded ^= {here.key}
        elif key == "h":
            self.expanded.discard(here.key)
        elif key == "e":
            everything = {e.key for e in entries}
            self.expanded = set() if self.expanded == everything else everything
        elif key == "o":
            return "open"
        elif key == "x":
            return "read"
        return None


FOOTER = "j/k move  enter why  e expand all  o open  x mark read  q quit"


def _view(screen: Screen, height: int, width: int):
    """The list, a status line, and the legend, scrolled to keep the cursor in view."""
    from rich.text import Text

    blocks: list[Text] = []
    for index, entry in enumerate(screen.reading.entries):
        block = Text()
        done = entry.read_at_ms is not None
        block.append("[x] " if done else "[ ] ", style="bold green" if done else "dim")
        block.append(f"{entry.rank:>2}. ", style="bold")
        block.append(
            title_of(screen.bibliography, entry.key), style="dim" if done else ""
        )
        block.append(f"\n       {entry.rationale}", style="italic")
        if entry.key in screen.expanded:
            for line in numbered(entry.chain_of_thought, "         "):
                block.append(f"\n{line}", style="cyan")
        if index == screen.cursor:
            block.stylize("reverse", 0, block.plain.find("\n"))
        blocks.append(block)

    # Scroll by whole entries: the cursor's entry and as many after it as fit,
    # backed up while there is room so the end of the list does not float.
    def lines(block: Text) -> int:
        return sum(1 + len(part) // max(1, width) for part in block.plain.split("\n"))

    header = Text()
    if screen.reading.chain_of_thought:
        header.append(f"{screen.bibliography.name}: how it was ordered\n", style="bold")
        for line in numbered(screen.reading.chain_of_thought, "  "):
            header.append(f"{line}\n", style="dim")
        header.append("\n")

    room = max(1, height - 3 - (lines(header) - 1 if header.plain else 0))
    top = screen.cursor
    used = sum(lines(b) for b in blocks[top:])
    while top > 0 and used + lines(blocks[top - 1]) <= room:
        top -= 1
        used += lines(blocks[top])
    out = header
    used = 0
    for block in blocks[top:]:
        if used and used + lines(block) > room:
            break
        out.append_text(block)
        out.append("\n")
        used += lines(block)
    read = sum(1 for e in screen.reading.entries if e.read_at_ms is not None)
    status = screen.message or f"{read} of {len(screen.reading.entries)} read"
    out.append(status + "\n", style="yellow" if screen.message else "dim")
    out.append(FOOTER, style="dim")
    return out


def run(
    bibliography: Bibliography,
    reading: ReadingList,
    open_entry: Callable[[Entry], None],
) -> None:
    """Show the list until asked to stop.

    `open_entry` opens an entry's document, and is the caller's, which knows
    where the library is. Marking an entry re-reads the list from disk, so an
    agent's new ranking written meanwhile shows up rather than being undone.
    """
    import termios
    import tty

    from rich.console import Console
    from rich.live import Live

    from .pick import QUIT_KEYS, _read_key

    console = Console()
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    screen = Screen(bibliography, reading)
    tty.setcbreak(fd)
    try:
        with Live(console=console, auto_refresh=False, transient=True) as live:
            while True:
                live.update(
                    _view(screen, console.size.height, console.size.width),
                    refresh=True,
                )
                key = _read_key(fd)
                if key in QUIT_KEYS:
                    break
                screen.message = ""
                wanted = screen.handle(key)
                entry = screen.under_cursor()
                if wanted is None or entry is None:
                    continue
                try:
                    if wanted == "open":
                        open_entry(entry)
                        screen.message = f"opened {entry.key}"
                    else:
                        screen.reading = mark(bibliography, entry.key)
                        # Follow the entry: the list may have been re-ranked
                        # under the screen, and the row number with it.
                        keys = [e.key for e in screen.reading.entries]
                        screen.cursor = keys.index(entry.key)
                except Exception as err:  # noqa: BLE001 -- shown, not raised
                    screen.message = f"{entry.key}: {err}"
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)

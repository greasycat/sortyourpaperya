"""What counts as a note, and what a note is called.

Documents keep notes in their store folder and bibliographies keep them in
theirs. The rule is the same in both places — any markdown or JSON file is a
note, a bare word means markdown, anything else is refused rather than renamed —
so it is written once here rather than twice.

Only the rule lives here. Which folder a note belongs in, and which file in that
folder is not a note but the thing the notes are about, is the caller's: a
document's own file is not a note about itself, and neither is a bibliography's
record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

NOTE_SUFFIXES = (".md", ".json")
DEFAULT_NOTE = "notes.md"


class NoteError(RuntimeError):
    """Raised when a name does not name a note."""


def note_name(name: str) -> str:
    """`name` as a note's filename.

    A bare name is taken as markdown, so `reading-log` and `reading-log.md`
    name the same file. A suffix that is neither markdown nor JSON is refused
    rather than corrected: the caller meant a format this does not keep, and
    renaming it for them would file it under a name they will not look for.

    Raises:
        NoteError: if the name is not a plain filename, or does not name a
            note format.
    """
    candidate = Path(name)
    if candidate.name != name:
        raise NoteError(f"a note is named by a filename, not a path: {name!r}")
    if not candidate.suffix:
        candidate = candidate.with_suffix(".md")
    if candidate.suffix.lower() not in NOTE_SUFFIXES:
        kinds = " or ".join(NOTE_SUFFIXES)
        raise NoteError(f"a note is {kinds}, not {candidate.suffix}: {name!r}")
    return candidate.name


def notes_in(folder: Path, *, besides: Sequence[str] = ()) -> list[Path]:
    """Every note in `folder`, in the order they read.

    The name is its owner's to choose — `notes.md` is only the one this tool
    makes when asked for a note and told nothing else. `besides` names the
    files in the folder that are not notes about anything, however they are
    spelled.
    """
    if not folder.is_dir():
        return []
    excluded = set(besides)
    return sorted(
        entry
        for entry in folder.iterdir()
        if entry.is_file()
        and entry.suffix.lower() in NOTE_SUFFIXES
        and entry.name not in excluded
    )


def create(path: Path, heading: str) -> None:
    """Start a note that does not exist yet, if it does not exist yet.

    An empty JSON note has to parse, or it is broken for the only thing JSON is
    kept for; a markdown one opens with what it is about.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text("{}\n", encoding="utf-8")
    else:
        path.write_text(f"# {heading}\n\n", encoding="utf-8")

"""Bibliographies kept beside the library: what you cite, and the file LaTeX reads.

A bibliography is a folder under `<library>/bibs/<slug>/` holding two files:

    bib.toml          the record, and the one you edit
    references.bib    generated from it, and the one LaTeX reads

**The TOML is the source of truth and the `.bib` is a projection of it**, the
same way the database is the truth behind the store's filenames. That is what
makes the `.bib` safe to overwrite on every change, and it is why a field the
library never knew — a journal, a page range, a corrected title — is added by
editing the TOML and running `sortyourpaperya bib build`, not by editing the
`.bib` that the next write would throw away.

A source names the document it came from by `file_id`, so a bibliography is a
list of things in this library rather than a copy of them. Everything else in a
`[[source]]` table is a BibTeX field, carried through as written: the four keys
this module reserves are `key`, `type`, `file_id`, and `added_at_ms`.

Bibliographies sit under the library root, beside `store/`, because they are
durable in the same way — hand-written, not derivable from anything — and so
they are copied by the same backup.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from string import ascii_lowercase

import tomli_w

from .db import Paper, now_ms
from .naming import cite_key, new_id, slugify

BIBS_DIR = "bibs"
RECORD_FILE = "bib.toml"
BIB_FILE = "references.bib"

# What a `[[source]]` table uses for its own bookkeeping. Every other key in it
# is a BibTeX field and is written out as one, so a bibliography can carry a
# field this tool has never heard of without needing to be taught it.
RESERVED = frozenset({"key", "type", "file_id", "added_at_ms"})

# The attribute keys that name a BibTeX field, and so are carried into an entry
# when a document is cited. Everything else a reader keeps on a document — a
# verdict, the day it was read — is theirs, and is not part of a citation.
BIB_ATTRIBUTES = (
    "address",
    "booktitle",
    "chapter",
    "doi",
    "edition",
    "editor",
    "eprint",
    "howpublished",
    "institution",
    "isbn",
    "issn",
    "journal",
    "month",
    "note",
    "number",
    "organization",
    "pages",
    "publisher",
    "school",
    "series",
    "url",
    "volume",
)

DEFAULT_ENTRY_TYPE = "misc"

# What a field's presence says the entry is. A document filed by this tool is
# not assumed to be a paper, so `misc` is what an entry is until something in
# the record says otherwise; `--type` overrules all of it.
_TYPE_BY_FIELD = (
    ("journal", "article"),
    ("booktitle", "inproceedings"),
    ("school", "phdthesis"),
    ("institution", "techreport"),
    ("publisher", "book"),
)

# Characters TeX reads as instructions rather than as text. The backslash is
# replaced first, or the replacements below would themselves be escaped.
_TEX_ESCAPES = (
    ("\\", r"\textbackslash{}"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
)


class BibError(RuntimeError):
    """Raised when a bibliography cannot be read, made, or written."""


@dataclass
class Source:
    """One thing cited: its key, its kind, and its BibTeX fields.

    `file_id` says which document in the library it is, and is what stops the
    same document being cited twice under two keys. A source added by hand to
    the TOML has none, and is written out like any other.
    """

    key: str
    entry_type: str = DEFAULT_ENTRY_TYPE
    file_id: str | None = None
    added_at_ms: int | None = None
    fields: dict[str, object] = field(default_factory=dict)


@dataclass
class Bibliography:
    """One bibliography: the folder, what it is called, and what it cites."""

    path: Path
    id: str
    slug: str
    name: str
    created_at_ms: int
    sources: list[Source] = field(default_factory=list)

    @property
    def record_path(self) -> Path:
        return self.path / RECORD_FILE

    @property
    def bib_path(self) -> Path:
        return self.path / BIB_FILE

    def source_for(self, file_id: str) -> Source | None:
        """The entry citing that document, if it is already cited."""
        return next((s for s in self.sources if s.file_id == file_id), None)

    def add(self, source: Source) -> Source:
        """Add a source, giving it a free key if the one it wants is taken.

        Two papers by the same author in the same year want the same key, which
        is the collision BibTeX itself answers with a trailing letter. The
        source is returned because the key it ends up with is not always the one
        it arrived with, and the caller has to be able to say what was written.
        """
        source.key = _free_key(source.key, {s.key for s in self.sources})
        self.sources.append(source)
        return source

    def save(self) -> None:
        """Write the record, then regenerate the `.bib` from it.

        In that order, so an interruption leaves the truth written and its
        projection stale — which `sortyourpaperya bib build` repairs — rather
        than a `.bib` citing something no record explains.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        try:
            self.record_path.write_text(_dump_record(self), encoding="utf-8")
            self.bib_path.write_text(render(self), encoding="utf-8")
        except OSError as err:
            raise BibError(f"could not write {self.path}: {err}") from err


def bibs_dir(library_root: Path) -> Path:
    """Where a library keeps its bibliographies."""
    return library_root / BIBS_DIR


def create(library_root: Path, name: str) -> Bibliography:
    """Make a new bibliography called `name`, and write its two files.

    The folder is named by the slug of `name`, so `sortyourpaperya bib init
    "PhD Thesis"` is `bibs/phd-thesis/`; the name it was given is kept in the
    record, since a slug is a filename and not a title.

    Raises:
        BibError: if `name` slugifies to nothing, or that folder is taken.
    """
    slug = slugify(name)
    if not slug:
        raise BibError(f"{name!r} does not name a bibliography")
    path = bibs_dir(library_root) / slug
    if path.exists():
        raise BibError(f"{path} already exists")

    bibliography = Bibliography(
        path=path,
        id=new_id(),
        slug=slug,
        name=name.strip(),
        created_at_ms=now_ms(),
    )
    bibliography.save()
    return bibliography


def all_bibs(library_root: Path) -> list[Bibliography]:
    """Every bibliography in the library, by slug.

    A folder that does not read is skipped rather than raising: one broken
    record must not stop the others being listed, or being picked from.
    """
    root = bibs_dir(library_root)
    if not root.is_dir():
        return []
    found = []
    for folder in sorted(root.iterdir()):
        if not (folder / RECORD_FILE).is_file():
            continue
        try:
            found.append(load(folder))
        except BibError:
            continue
    return found


def open_bib(library_root: Path, needle: str) -> Bibliography:
    """The bibliography named by its slug or by its id.

    Both are offered because the slug is what a person reads and types, and the
    id is what survives the folder being renamed — an agent that recorded one
    last week should still find it.

    Raises:
        BibError: if nothing in the library goes by that name.
    """
    wanted = needle.strip()
    direct = bibs_dir(library_root) / wanted
    if (direct / RECORD_FILE).is_file():
        return load(direct)

    for bibliography in all_bibs(library_root):
        if bibliography.id == wanted:
            return bibliography

    known = ", ".join(b.slug for b in all_bibs(library_root)) or "none yet"
    raise BibError(f"no bibliography {wanted!r} in this library ({known})")


def load(folder: Path) -> Bibliography:
    """Read one bibliography's record.

    Raises:
        BibError: if the file is missing, unreadable, or not a record.
    """
    path = folder / RECORD_FILE
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise BibError(f"could not read {path}: {err}") from err

    declared = raw.get("source") or []
    if not isinstance(declared, list):
        raise BibError(f"{path}: `source` must be a list of [[source]] tables")

    sources = []
    for position, body in enumerate(declared, start=1):
        if not isinstance(body, dict):
            raise BibError(f"{path}: source {position} must be a table")
        key = str(body.get("key") or "").strip()
        if not key:
            raise BibError(f"{path}: source {position} has no `key`")
        sources.append(
            Source(
                key=key,
                entry_type=str(body.get("type") or DEFAULT_ENTRY_TYPE),
                file_id=str(body["file_id"]) if body.get("file_id") else None,
                added_at_ms=body.get("added_at_ms")
                if isinstance(body.get("added_at_ms"), int)
                else None,
                fields={k: v for k, v in body.items() if k not in RESERVED},
            )
        )

    return Bibliography(
        path=folder,
        id=str(raw.get("id") or ""),
        slug=str(raw.get("slug") or folder.name),
        name=str(raw.get("name") or folder.name),
        created_at_ms=raw.get("created_at_ms")
        if isinstance(raw.get("created_at_ms"), int)
        else 0,
        sources=sources,
    )


def source_from_paper(
    paper: Paper,
    attributes: dict[str, str | None],
    *,
    entry_type: str | None = None,
    key: str | None = None,
) -> Source:
    """Turn a filed document into a citable source.

    Title, authors, and year come from the library's own columns; everything
    else comes from the document's attributes, which is where a reader puts the
    DOI and the venue that the model was never asked for. An attribute the
    caller set to nothing is left out rather than written as an empty field,
    which BibTeX would render as a citation with a blank journal.
    """
    fields: dict[str, object] = {}
    if paper.title:
        fields["title"] = paper.title
    if paper.authors:
        fields["author"] = list(paper.authors)
    if paper.year:
        fields["year"] = paper.year
    for name in BIB_ATTRIBUTES:
        value = (attributes.get(name) or "").strip()
        if value:
            fields[name] = value

    return Source(
        key=key
        or cite_key(
            authors=paper.authors,
            year=paper.year,
            title=paper.title,
            fallback=paper.file_id,
        ),
        entry_type=entry_type or _infer_type(fields),
        file_id=paper.file_id,
        added_at_ms=now_ms(),
        fields=fields,
    )


def render(bibliography: Bibliography) -> str:
    """The whole bibliography as a BibTeX file.

    The header says where the file came from, because someone who finds it in a
    paper's folder and edits it would otherwise have no way of knowing that the
    next `bib add` overwrites their work.
    """
    lines = [
        f"% {bibliography.name}",
        f"% Generated by sortyourpaperya from {RECORD_FILE}. Edit that, not this.",
        "",
    ]
    for source in bibliography.sources:
        lines.append(_render_entry(source))
    return "\n".join(lines)


def _render_entry(source: Source) -> str:
    """One `@type{key, ...}` entry.

    The title is wrapped in a second pair of braces because BibTeX styles
    lowercase a title they are not told to leave alone, and a title read off the
    document is already capitalized the way its authors capitalized it.
    """
    body = []
    for name, value in source.fields.items():
        if isinstance(value, list):
            # BibTeX has no lists: names are one field joined by ` and `.
            rendered = _escape(" and ".join(str(item) for item in value))
        elif isinstance(value, bool):
            rendered = _escape(str(value).lower())
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            rendered = _escape(str(value))
        if name == "title":
            rendered = "{" + rendered + "}"
        body.append(f"  {name} = {{{rendered}}},")
    return "\n".join([f"@{source.entry_type}{{{source.key},", *body, "}", ""])


def _escape(text: str) -> str:
    for character, replacement in _TEX_ESCAPES:
        text = text.replace(character, replacement)
    return text


def _infer_type(fields: dict[str, object]) -> str:
    for name, entry_type in _TYPE_BY_FIELD:
        if fields.get(name):
            return entry_type
    return DEFAULT_ENTRY_TYPE


def _free_key(key: str, taken: set[str]) -> str:
    """`key`, or the first spelling of it nothing else has claimed."""
    if key not in taken:
        return key
    for letter in ascii_lowercase[1:]:
        if f"{key}{letter}" not in taken:
            return f"{key}{letter}"
    number = 2
    while f"{key}-{number}" in taken:
        number += 1
    return f"{key}-{number}"


def _dump_record(bibliography: Bibliography) -> str:
    """The record as TOML.

    Written through `tomli_w` rather than by hand so that a title carrying a
    quote or a backslash comes back out of `tomllib` as what went in.
    """
    document: dict[str, object] = {
        "id": bibliography.id,
        "slug": bibliography.slug,
        "name": bibliography.name,
        "created_at_ms": bibliography.created_at_ms,
    }
    sources = []
    for source in bibliography.sources:
        table: dict[str, object] = {"key": source.key, "type": source.entry_type}
        if source.file_id:
            table["file_id"] = source.file_id
        if source.added_at_ms is not None:
            table["added_at_ms"] = source.added_at_ms
        table.update(source.fields)
        sources.append(table)
    if sources:
        document["source"] = sources
    return tomli_w.dumps(document)

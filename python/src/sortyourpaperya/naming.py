"""Turning a paper's identity and tags into filenames, and back again.

Two names exist for every paper, and they answer different questions.

The **store name** is `<id>__<Tag>__<Tag>.pdf`. It lives in one flat folder,
carries the paper's permanent id up front, and spells its tags after it. Because
the id never changes and the tags do, re-tagging a paper is a rename.

The **link name** is what the symlink tree shows: `author_year_title.pdf` when
the pieces are known, and the store name when they are not, so a link always has
a name even for a paper nothing is known about.

The **citation key** is what a bibliography cites a document by:
`vaswani2017attention`. Same three pieces, spelled the way BibTeX wants them.

The database is authoritative for tags. These names are a readable projection of
it, which is what lets long tag lists be truncated to fit a filesystem without
losing anything.
"""

from __future__ import annotations

import re
import unicodedata
import uuid

# Two underscores separate the id from the tags and the tags from each other, so
# no single tag may contain one. Sanitizing underscores out entirely is what
# makes that guarantee hold rather than merely usually hold.
TAG_SEPARATOR = "__"

PAPER_ID_CHARS = 12
MAX_TAG_CHARS = 48
MAX_TITLE_SLUG_CHARS = 72

# Comfortably inside the 255-byte limit that ext4, APFS, and NTFS all share,
# leaving room for a `.pdf` suffix and a disambiguating id.
MAX_NAME_CHARS = 200

# Words that name nothing, and so are skipped when a title has to supply the
# one word a citation key carries.
KEY_STOP_WORDS = frozenset(
    {"a", "an", "and", "for", "from", "in", "of", "on", "the", "to", "with"}
)

_ALLOWED_TAG = re.compile(r"[^A-Za-z0-9 -]+")
_ALLOWED_SLUG = re.compile(r"[^a-z0-9-]+")
_DASH_RUN = re.compile(r"-{2,}")
_SPACE_RUN = re.compile(r"\s+")


def new_id() -> str:
    """A fresh permanent id: 12 hex characters of a UUID4.

    Papers and bibliographies are both named by one. They live in separate
    namespaces, so the same generator serves both.
    """
    return uuid.uuid4().hex[:PAPER_ID_CHARS]


def sanitize_tag(tag: str) -> str:
    """Make one category safe to put in a filename.

    Underscores become dashes so the `__` separator can never appear inside a
    tag, and the round trip through `parse_store_name` stays exact.
    """
    folded = _ascii_fold(tag).replace("_", "-").replace("/", " ")
    cleaned = _ALLOWED_TAG.sub("", folded)
    cleaned = _SPACE_RUN.sub(" ", cleaned).strip(" -")
    cleaned = _DASH_RUN.sub("-", cleaned)
    return cleaned[:MAX_TAG_CHARS].strip(" -")


def split_category(category: str) -> list[str]:
    """Split a model's `A/B/C` category text into ordered, sanitized tags."""
    tags = [sanitize_tag(part) for part in category.split("/")]
    return [tag for tag in tags if tag]


def store_name(paper_id: str, tags: list[str], suffix: str = ".pdf") -> str:
    """Build the flat-store filename for a paper.

    Tags are dropped from the end if the name would outgrow the filesystem
    limit. That loses nothing: the database holds the full tag list, and this
    name is only its readable form.
    """
    kept = list(tags)
    while True:
        joined = TAG_SEPARATOR.join([paper_id, *kept]) if kept else paper_id
        name = f"{joined}{suffix}"
        if len(name) <= MAX_NAME_CHARS or not kept:
            return name
        kept.pop()


def parse_store_name(name: str) -> tuple[str, list[str]]:
    """Recover the paper id and tags from a store filename.

    Raises:
        ValueError: if the name does not start with a paper id.
    """
    stem = name.rsplit(".", 1)[0] if "." in name else name
    parts = stem.split(TAG_SEPARATOR)
    paper_id = parts[0]
    if not re.fullmatch(r"[0-9a-f]{%d}" % PAPER_ID_CHARS, paper_id):
        raise ValueError(f"{name!r} does not start with a paper id")
    return paper_id, [part for part in parts[1:] if part]


def link_name(
    *,
    fallback: str,
    authors: list[str] | None = None,
    year: int | None = None,
    title: str | None = None,
    suffix: str = ".pdf",
) -> str:
    """Name for the symlink in the browsable tree.

    Built from whichever of author, year, and title are known, joined in that
    order: `vaswani_2017_attention-is-all-you-need`, `vaswani_attention…` when
    the year is missing, `attention…` when only the title is.

    A year on its own names nothing a person could recognise, so a document with
    neither an author nor a title falls back to `fallback` — the store name —
    rather than being called `2017.pdf`.
    """
    author = _first_author_surname(authors or [])
    slug = title_slug(title)
    if not (author or slug):
        return fallback

    parts = [part for part in (author, str(year) if year else "", slug) if part]
    # Trim the stem, not the finished name: capping afterwards can cut the
    # suffix off a very long author and leave a file with no extension.
    stem = "_".join(parts)[: MAX_NAME_CHARS - len(suffix)]
    return f"{stem}{suffix}"


def disambiguate(name: str, paper_id: str) -> str:
    """Append the paper id to a link name that collided with another paper's."""
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        return f"{name}_{paper_id}"
    return f"{stem}_{paper_id}.{suffix}"


def title_slug(title: str | None) -> str:
    """A title as a slug, cut between words rather than inside one."""
    return _truncate_words(slugify(title or ""), MAX_TITLE_SLUG_CHARS)


def author_year(authors: list[str] | None = None, year: int | None = None) -> str:
    """`knuth_1984` — who and when, which is how a shelf is arranged.

    Empty when there is no author to arrange by: a year on its own names
    nothing, here as much as in `link_name`, and the caller has to fall back
    to something that does.
    """
    surname = _first_author_surname(authors or [])
    if not surname:
        return ""
    return f"{surname}_{year}" if year else surname


def cite_key(
    *,
    fallback: str,
    authors: list[str] | None = None,
    year: int | None = None,
    title: str | None = None,
) -> str:
    """The key a bibliography cites a document by: `vaswani2017attention`.

    Author surname, year, and the first word of the title that names something.
    It is the spelling nearly every reference manager produces, and so the one
    someone will guess at when citing it. Whichever pieces are known are run
    together; a document with none of them falls back to `fallback`, its id,
    because an entry with no key is not an entry.

    Dashes are dropped rather than kept: a hyphenated surname reads perfectly
    well run together, and a stray dash is one more thing to remember about how
    the key was spelled.

    A year on its own names nothing, exactly as it does not for `link_name`, so
    a document with neither an author nor a title falls back rather than being
    cited as `2017`.
    """
    surname = _first_author_surname(authors or []).replace("-", "")
    words = [word for word in slugify(title or "").split("-") if word]
    first = next(
        (word for word in words if word not in KEY_STOP_WORDS),
        words[0] if words else "",
    )
    if not (surname or first):
        return fallback
    parts = [part for part in (surname, str(year) if year else "", first) if part]
    return "".join(parts)


def _truncate_words(slug: str, limit: int) -> str:
    """Shorten a slug to `limit`, cutting between words rather than inside one.

    A title clipped mid-word reads as a typo — `...-and-tomo` for
    "...and Tomosynthesis". Dropping the partial word is shorter and honest.

    A first word already longer than the limit has no boundary to cut at, so it
    is clipped as it stands.
    """
    if len(slug) <= limit:
        return slug.strip("-")
    clipped = slug[:limit]
    boundary = clipped.rfind("-")
    if boundary > 0:
        clipped = clipped[:boundary]
    return clipped.strip("-")


def _first_author_surname(authors: list[str]) -> str:
    """The surname of the first author, which is how papers are cited."""
    for author in authors:
        cleaned = author.strip()
        if not cleaned:
            continue
        # "Ashish Vaswani" -> vaswani; "Vaswani, Ashish" -> vaswani
        surname = cleaned.split(",")[0] if "," in cleaned else cleaned.split()[-1]
        slug = slugify(surname)
        if slug:
            return slug
    return ""


def slugify(text: str) -> str:
    """Text as a lowercase, dash-separated, filename-safe slug."""
    folded = _ascii_fold(text).lower().replace("_", "-")
    slug = _ALLOWED_SLUG.sub("-", folded)
    return _DASH_RUN.sub("-", slug).strip("-")


def _ascii_fold(text: str) -> str:
    """Drop accents so filenames stay portable across filesystems."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))

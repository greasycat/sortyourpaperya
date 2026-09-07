"""DuckDB store holding everything known about each paper.

The database is the source of truth. The flat store's filenames and the symlink
tree are both projections of it and can be rebuilt from it at any time, which is
what makes re-tagging safe: change the rows, then rebuild the names.

Column names follow the Rust `paper-db` crate (`file_id`, `content_hash`,
`created_at_ms`) so the two can be read side by side.

Expanding it means adding a column: `_MIGRATIONS` is an ordered list applied
once each and recorded in `schema_version`. Anything not worth a column yet goes
in `paper_attributes` as a key/value pair.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import duckdb

from .llm import KeywordPair

# DuckDB allows one writing process at a time, so a command run while the
# watcher is mid-pass finds the file locked. Waiting briefly turns that from an
# error into a pause; past this the lock is not contention but a stuck process,
# and saying so is more use than waiting longer.
LOCK_WAIT_SECONDS = 30.0
_LOCK_POLL_SECONDS = 0.25

# Applied in order, each exactly once. Append to expand the schema; never edit
# or reorder an entry that has shipped.
_MIGRATIONS: list[tuple[str, str]] = [
    (
        "0001_initial",
        """
        CREATE TABLE IF NOT EXISTS papers (
            file_id        TEXT PRIMARY KEY,
            content_hash   TEXT NOT NULL,
            store_name     TEXT NOT NULL,
            original_name  TEXT,
            source_path    TEXT,
            size_bytes     BIGINT,
            pages_read     INTEGER,
            title          TEXT,
            year           INTEGER,
            created_at_ms  BIGINT NOT NULL,
            updated_at_ms  BIGINT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS paper_tags (
            file_id  TEXT NOT NULL,
            position INTEGER NOT NULL,
            tag      TEXT NOT NULL,
            PRIMARY KEY (file_id, position)
        );

        CREATE TABLE IF NOT EXISTS paper_authors (
            file_id  TEXT NOT NULL,
            position INTEGER NOT NULL,
            name     TEXT NOT NULL,
            PRIMARY KEY (file_id, position)
        );

        CREATE TABLE IF NOT EXISTS paper_keywords (
            file_id TEXT NOT NULL,
            keyword TEXT NOT NULL,
            PRIMARY KEY (file_id, keyword)
        );

        CREATE TABLE IF NOT EXISTS paper_attributes (
            file_id TEXT NOT NULL,
            key     TEXT NOT NULL,
            value   TEXT,
            PRIMARY KEY (file_id, key)
        );
        """,
    ),
    (
        "0002_stored_file_state",
        """
        ALTER TABLE papers ADD COLUMN IF NOT EXISTS stored_mtime_ms BIGINT;
        """,
    ),
    (
        "0003_page_image_provenance",
        """
        ALTER TABLE papers ADD COLUMN IF NOT EXISTS from_page_images BOOLEAN;
        """,
    ),
    (
        "0004_document_folder",
        """
        ALTER TABLE papers ADD COLUMN IF NOT EXISTS document_name TEXT;
        """,
    ),
    (
        "0005_model_answer_cache",
        """
        CREATE TABLE IF NOT EXISTS model_answers (
            content_hash  TEXT PRIMARY KEY,
            page_text     TEXT,
            labels        TEXT,
            model         TEXT,
            created_at_ms BIGINT NOT NULL,
            updated_at_ms BIGINT NOT NULL
        );
        """,
    ),
]


@dataclass
class ModelAnswer:
    """What the model said about one document's contents, keyed by its hash.

    A receipt for money already spent, not a description of the library. It is
    keyed by content rather than by document id because the id is minted fresh
    on every attempt — the whole point is to be found again by a pass that has
    no memory of the one that paid.

    Either half may be absent: `page_text` only exists for a document with no
    text layer, and `labels` only once the labelling call has come back.
    """

    content_hash: str
    page_text: str | None = None
    labels: "KeywordPair | None" = None
    model: str | None = None


@dataclass
class Paper:
    """Everything the library knows about one PDF."""

    file_id: str
    content_hash: str
    # The document's folder in the store, `<id>__<Tag>__<Tag>`. It holds the
    # document and anything its owner keeps beside it.
    store_name: str
    # The document file inside that folder.
    document_name: str = ""
    original_name: str | None = None
    source_path: str | None = None
    # Size and mtime of the file in the store, as last observed. Together they
    # are a cheap gate: a file whose stat has not moved does not need rehashing.
    size_bytes: int | None = None
    stored_mtime_ms: int | None = None
    pages_read: int | None = None
    # True when the text behind this document's labels was read off rendered
    # page images rather than a text layer.
    from_page_images: bool = False
    title: str | None = None
    year: int | None = None
    tags: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    # When the library first recorded this document and last changed it. Read
    # from the database and never written back: `_write` keeps the original
    # `created_at_ms` of a row it replaces, so a re-tag does not reset the day
    # a document was filed.
    created_at_ms: int | None = None
    updated_at_ms: int | None = None


def now_ms() -> int:
    return int(time.time() * 1000)


def _encode_labels(labels: KeywordPair | None) -> str | None:
    """A label set as JSON. `file_id` is left out: it belongs to one attempt."""
    if labels is None:
        return None
    return json.dumps(
        {
            "keywords": labels.keywords,
            "preliminary_category": labels.preliminary_category,
            "title": labels.title,
            "authors": labels.authors,
            "year": labels.year,
        }
    )


def _decode_labels(content_hash: str, payload: str | None) -> KeywordPair | None:
    """Read a stored label set back, or None if it cannot be used.

    A cache that cannot be read is a cache miss, never an error: the worst it
    costs is the request it was meant to save.
    """
    if not payload:
        return None
    try:
        fields = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(fields, dict):
        return None
    return KeywordPair(
        file_id=content_hash,
        keywords=[str(keyword) for keyword in fields.get("keywords") or []],
        preliminary_category=str(fields.get("preliminary_category") or ""),
        title=str(fields.get("title") or ""),
        authors=[str(author) for author in fields.get("authors") or []],
        year=fields.get("year") if isinstance(fields.get("year"), int) else None,
    )


_PAPER_COLUMNS = (
    "file_id, content_hash, store_name, original_name, source_path, "
    "size_bytes, pages_read, title, year, stored_mtime_ms, "
    "from_page_images, document_name, created_at_ms, updated_at_ms"
)

# What `sort=` may ask for, and the only place a sort reaches the SQL. The
# caller names a key; it never supplies an ORDER BY.
#
# `id` is first because it is the stable one: a hash orders arbitrarily but
# identically every time, which is what makes a capped result reproducible.
# Everything else puts unknowns last, since a document with no year is not the
# oldest one and a reader scanning from the top should not meet it first.
SORTS = {
    "id": "p.file_id",
    "recent": "p.created_at_ms DESC NULLS LAST",
    "updated": "p.updated_at_ms DESC NULLS LAST",
    "title": "lower(coalesce(p.title, p.original_name, p.store_name))",
    "year": "p.year DESC NULLS LAST",
    "size": "p.size_bytes DESC NULLS LAST",
}

# What one search word is checked against. A paper is described in four columns
# and three side tables, and a reader searching for "vaswani" does not know or
# care which of them holds the answer.
_MATCHES_A_WORD = """(
    lower(p.file_id) LIKE ? ESCAPE '!'
    OR lower(coalesce(p.title, '')) LIKE ? ESCAPE '!'
    OR lower(coalesce(p.original_name, '')) LIKE ? ESCAPE '!'
    OR coalesce(CAST(p.year AS VARCHAR), '') LIKE ? ESCAPE '!'
    OR EXISTS (SELECT 1 FROM paper_tags t
               WHERE t.file_id = p.file_id AND lower(t.tag) LIKE ? ESCAPE '!')
    OR EXISTS (SELECT 1 FROM paper_authors a
               WHERE a.file_id = p.file_id AND lower(a.name) LIKE ? ESCAPE '!')
    OR EXISTS (SELECT 1 FROM paper_keywords k
               WHERE k.file_id = p.file_id AND lower(k.keyword) LIKE ? ESCAPE '!')
)"""


def _like(word: str) -> str:
    """One search word as a LIKE pattern, with its wildcards taken literally.

    `_` and `%` turn up in filenames, where they mean themselves rather than
    "any character" — a search for `report_2024` should not also match
    `report-2024`.
    """
    escaped = word.lower().replace("!", "!!").replace("%", "!%").replace("_", "!_")
    return f"%{escaped}%"


def _order_by(sort: str) -> str:
    """The ORDER BY for a named sort. Unknown names are refused, not guessed."""
    try:
        return SORTS[sort]
    except KeyError:
        raise ValueError(
            f"unknown sort {sort!r}; one of {', '.join(SORTS)}"
        ) from None


# What a read-only statement may begin with. All of them read in DuckDB:
# nothing here can insert, update, delete, attach, copy, or set a setting.
_READ_ONLY_HEADS = frozenset(
    {"select", "with", "from", "table", "values", "describe", "summarize", "show"}
)


def _skeleton(sql: str) -> str:
    """`sql` with comments dropped and string bodies blanked.

    Only ever inspected, never executed. Blanking the inside of a literal is
    what stops `SELECT ';DROP'` from reading as two statements, and dropping
    comments is what stops `--` from hiding one.
    """
    out: list[str] = []
    index, end = 0, len(sql)
    while index < end:
        pair = sql[index : index + 2]
        if pair == "--":
            while index < end and sql[index] != "\n":
                index += 1
        elif pair == "/*":
            index += 2
            while index < end and sql[index : index + 2] != "*/":
                index += 1
            index += 2
        elif sql[index] in "\'\"":
            quote = sql[index]
            out.append(" ")
            index += 1
            # A doubled quote closes and reopens, which lands here again.
            while index < end and sql[index] != quote:
                index += 1
            index += 1
        else:
            out.append(sql[index])
            index += 1
    return "".join(out)


def is_read_only(sql: str) -> bool:
    """Whether `sql` is one statement that only reads.

    A guard against a mistake, not a sandbox: it is what stops a query meant to
    count documents from deleting them, and what stops a second statement
    riding in behind the first. Anyone who can run this can already read the
    database file.
    """
    skeleton = _skeleton(sql).strip().rstrip(";").strip()
    words = skeleton.split()
    if not words or ";" in skeleton:
        return False
    return words[0].lower() in _READ_ONLY_HEADS


class Locked(RuntimeError):
    """A read could not start because a writer holds the database.

    Raised instead of waiting only when the caller asked for that, by passing
    `fail_on_lock` -- which it does when it has somewhere else to ask.
    """


class PaperDb:
    """Connection to the library database."""

    def __init__(
        self, path: Path, *, read_only: bool = False, fail_on_lock: bool = False
    ) -> None:
        self.path = path
        self.read_only = read_only
        self.fail_on_lock = fail_on_lock
        if not read_only:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: duckdb.DuckDBPyConnection | None = None

    @property
    def _conn(self) -> duckdb.DuckDBPyConnection:
        """The connection, opened on first use.

        DuckDB takes an exclusive lock on the file, so a long-running watcher
        that held one open would block every other `sortyourpaperya` command for as long as
        it ran. Connecting lazily lets an idle watcher drop the lock — see
        `release` — and pick it up again for the next pass.
        """
        if self._connection is None:
            self._connection = self._connect()
            # A read-only connection cannot run the migration DDL, and does not
            # need to: it is only ever opened against a database some writer
            # already made. Trying would fail with a permissions error that says
            # nothing about the real cause.
            if not self.read_only:
                self._migrate()
        return self._connection

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Open the database, waiting out a lock another process is holding.

        Only lock contention is waited on. Any other IO failure — a missing
        directory, a permission problem, a corrupt file — is raised at once,
        because no amount of waiting fixes it.
        """
        deadline = time.monotonic() + LOCK_WAIT_SECONDS
        while True:
            try:
                return duckdb.connect(str(self.path), read_only=self.read_only)
            except duckdb.IOException as err:
                if "lock" not in str(err).lower():
                    raise
                if self.fail_on_lock:
                    # The caller has somewhere else to ask -- the watcher holding
                    # this lock can answer now, where waiting out an ingest pass
                    # could take minutes.
                    raise Locked(str(err)) from err
                if time.monotonic() >= deadline:
                    raise
                time.sleep(_LOCK_POLL_SECONDS)

    def release(self) -> None:
        """Drop the connection, and the file lock with it.

        Safe to call at any time: the next read or write reconnects.
        """
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def close(self) -> None:
        self.release()

    def __enter__(self) -> "PaperDb":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _migrate(self) -> None:
        conn = self._connection
        assert conn is not None  # only called from the connection property
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                name        TEXT PRIMARY KEY,
                applied_at_ms BIGINT NOT NULL
            )
            """
        )
        applied = {
            row[0] for row in conn.execute("SELECT name FROM schema_version").fetchall()
        }
        for name, sql in _MIGRATIONS:
            if name in applied:
                continue
            conn.execute("BEGIN")
            try:
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO schema_version VALUES (?, ?)", [name, now_ms()]
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def backup_to(self, destination: Path) -> Path:
        """Write a standalone copy of the database to `destination`.

        Copied through DuckDB rather than by copying the file: a database being
        written has changes in a write-ahead log beside it, so a file copy taken
        at the wrong moment restores to a database missing its most recent rows
        — or to one that will not open at all. `COPY FROM DATABASE` reads a
        committed view and writes a complete one.

        Raises:
            duckdb.Error: if the copy cannot be made. A backup that failed has
                to say so; a quiet one is worse than none, because it is
                believed.
        """
        if destination.exists():
            raise duckdb.IOException(f"{destination} already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)

        source = self._conn.execute("SELECT current_database()").fetchone()[0]
        # ATTACH takes a literal, not a parameter, so the path is quoted by
        # hand. Doubling the quote is what keeps a folder name containing one
        # from ending the string early.
        literal = str(destination).replace("'", "''")
        self._conn.execute(f"ATTACH '{literal}' AS sortyourpaperya_backup")
        try:
            self._conn.execute(f'COPY FROM DATABASE "{source}" TO sortyourpaperya_backup')
        finally:
            self._conn.execute("DETACH sortyourpaperya_backup")
        return destination

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._conn.execute("BEGIN")
        try:
            yield
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ---- reads -------------------------------------------------------------

    def find_by_content_hash(self, content_hash: str) -> str | None:
        """The id of the paper with this content, if the library already has it."""
        row = self._conn.execute(
            "SELECT file_id FROM papers WHERE content_hash = ?", [content_hash]
        ).fetchone()
        return row[0] if row else None

    def get(self, file_id: str) -> Paper | None:
        row = self._conn.execute(
            f"SELECT {_PAPER_COLUMNS} FROM papers WHERE file_id = ?",  # noqa: S608
            [file_id],
        ).fetchone()
        if row is None:
            return None
        return self._hydrate(row)

    def all_papers(self, sort: str = "id") -> list[Paper]:
        rows = self._conn.execute(
            f"SELECT {_PAPER_COLUMNS} FROM papers p ORDER BY {_order_by(sort)}"  # noqa: S608
        ).fetchall()
        return [self._hydrate(row) for row in rows]

    def search(
        self, query: str, limit: int | None = None, sort: str = "id"
    ) -> list[Paper]:
        """Every paper matching all of `query`'s words, anywhere it is described.

        A word matches against the id, the title, the original filename, the
        year, and any one tag, author, or keyword. Words are ANDed and fields
        are ORed, so `transformer 2017` narrows while `attention` on its own
        casts wide — the order a reader expects from a search box.

        Matching happens in SQL rather than over `all_papers()` because
        hydrating a paper costs three further queries, and a search that
        hydrates the whole library to discard most of it pays them for nothing.
        """
        words = query.split()
        if not words:
            return []

        clauses = " AND ".join(_MATCHES_A_WORD for _ in words)
        params = [pattern for word in words for pattern in (_like(word),) * 7]
        ceiling = f" LIMIT {int(limit)}" if limit is not None else ""
        rows = self._conn.execute(
            f"SELECT {_PAPER_COLUMNS} FROM papers p "  # noqa: S608
            f"WHERE {clauses} ORDER BY {_order_by(sort)}{ceiling}",
            params,
        ).fetchall()
        return [self._hydrate(row) for row in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT count(*) FROM papers").fetchone()[0]

    def tag_paths(self, limit: int | None = None) -> list[str]:
        """Every distinct category path the library already uses, as `A/B/C`.

        This is what a new document is shown so it can join an existing branch
        instead of inventing a parallel one.
        """
        rows = self._conn.execute(
            """
            SELECT string_agg(tag, '/' ORDER BY position) AS path
            FROM paper_tags
            GROUP BY file_id
            """
        ).fetchall()
        paths = sorted({path for (path,) in rows if path})
        return paths[:limit] if limit else paths

    def known_content_hashes(self, digests: Sequence[str]) -> set[str]:
        """Which of these contents the library already holds.

        One query rather than one per candidate, so the whole already-known
        check is a single short visit to the database.
        """
        if not digests:
            return set()
        unique = list(dict.fromkeys(digests))
        placeholders = ", ".join("?" for _ in unique)
        rows = self._conn.execute(
            f"SELECT content_hash FROM papers WHERE content_hash IN ({placeholders})",  # noqa: S608
            unique,
        ).fetchall()
        return {row[0] for row in rows}

    def model_answers(
        self, content_hashes: Sequence[str], *, max_age_ms: int | None
    ) -> dict[str, ModelAnswer]:
        """What has already been paid for, for these contents.

        Entries past `max_age_ms` are ignored. A label is a decision made
        against the library as it was when the model made it — the categories
        it was steered by have since moved on — so a receipt is worth reusing
        for a while and not forever. `None` keeps them indefinitely.
        """
        if not content_hashes:
            return {}
        unique = list(dict.fromkeys(content_hashes))
        placeholders = ", ".join("?" for _ in unique)
        parameters: list[object] = list(unique)
        clause = ""
        if max_age_ms is not None:
            clause = " AND created_at_ms >= ?"
            parameters.append(now_ms() - max_age_ms)

        rows = self._conn.execute(
            "SELECT content_hash, page_text, labels, model FROM model_answers "  # noqa: S608
            f"WHERE content_hash IN ({placeholders}){clause}",
            parameters,
        ).fetchall()

        answers: dict[str, ModelAnswer] = {}
        for content_hash, page_text, labels, model in rows:
            answers[content_hash] = ModelAnswer(
                content_hash=content_hash,
                page_text=page_text,
                labels=_decode_labels(content_hash, labels),
                model=model,
            )
        return answers

    def remember_model_answers(
        self, answers: Sequence[ModelAnswer], *, max_age_ms: int | None = None
    ) -> None:
        """Record what the model said, so a later pass does not buy it again.

        Written as one transaction, and merged rather than replaced: the page
        text of a scan is recorded as soon as it is read, and its labels arrive
        in a second visit, so a write that dropped the halves it was not given
        would throw away the more expensive one.
        """
        if not answers:
            return
        with self._transaction():
            stamp = now_ms()
            for answer in answers:
                self._conn.execute(
                    """
                    INSERT INTO model_answers
                        (content_hash, page_text, labels, model,
                         created_at_ms, updated_at_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (content_hash) DO UPDATE SET
                        page_text = coalesce(excluded.page_text, model_answers.page_text),
                        labels    = coalesce(excluded.labels, model_answers.labels),
                        model     = coalesce(excluded.model, model_answers.model),
                        updated_at_ms = excluded.updated_at_ms
                    """,
                    [
                        answer.content_hash,
                        answer.page_text,
                        _encode_labels(answer.labels),
                        answer.model,
                        stamp,
                        stamp,
                    ],
                )
            if max_age_ms is not None:
                # Pruned here rather than on a schedule: this is the one place
                # the table is written, and an unbounded cache of text would
                # outgrow the library it describes.
                self._conn.execute(
                    "DELETE FROM model_answers WHERE created_at_ms < ?",
                    [stamp - max_age_ms],
                )

    def forget_model_answers(self) -> int:
        """Empty the cache, and say how many receipts were thrown away."""
        count = self._conn.execute("SELECT count(*) FROM model_answers").fetchone()[0]
        with self._transaction():
            self._conn.execute("DELETE FROM model_answers")
        return count

    def count_model_answers(self) -> int:
        return self._conn.execute("SELECT count(*) FROM model_answers").fetchone()[0]

    def delete(self, file_id: str) -> None:
        """Forget a document entirely."""
        with self._transaction():
            for table in (
                "paper_tags",
                "paper_authors",
                "paper_keywords",
                "paper_attributes",
                "papers",
            ):
                self._conn.execute(
                    f"DELETE FROM {table} WHERE file_id = ?", [file_id]  # noqa: S608
                )

    def select(self, sql: str, limit: int | None = None) -> tuple[list[str], list[tuple]]:
        """Run one read-only statement, returning its column names and rows.

        The whole database is in reach here, including what no command reports:
        when a document was filed, how large it is, and the page text banked in
        `model_answers`.

        Raises:
            ValueError: if `sql` is not a single reading statement.
        """
        if not is_read_only(sql):
            raise ValueError(
                "only a single reading statement is allowed "
                f"({', '.join(sorted(_READ_ONLY_HEADS))})"
            )
        cursor = self._conn.execute(sql)
        columns = [column[0] for column in cursor.description or []]
        rows = cursor.fetchall() if limit is None else cursor.fetchmany(limit)
        return columns, rows

    def category_counts(self) -> list[tuple[str, int]]:
        """Every category path in use and how many documents are under it.

        The question `list --json` was being dumped whole to answer: where a
        new document could join instead of opening a branch beside one.
        """
        rows = self._conn.execute(
            """
            SELECT path, count(*) AS documents FROM (
                SELECT file_id, string_agg(tag, '/' ORDER BY position) AS path
                FROM paper_tags
                GROUP BY file_id
            )
            GROUP BY path
            ORDER BY path
            """
        ).fetchall()
        return [(path, count) for path, count in rows]

    def attributes(self, file_id: str) -> dict[str, str | None]:
        rows = self._conn.execute(
            "SELECT key, value FROM paper_attributes WHERE file_id = ? ORDER BY key",
            [file_id],
        ).fetchall()
        return {key: value for key, value in rows}

    def attributes_for(self, file_ids: Sequence[str]) -> dict[str, dict[str, str | None]]:
        """Every listed document's attributes, in one query.

        `list --json` describes the whole library, and asking per document
        would add a query per document to a path that already has three.
        """
        if not file_ids:
            return {}
        places = ", ".join("?" for _ in file_ids)
        rows = self._conn.execute(
            "SELECT file_id, key, value FROM paper_attributes "  # noqa: S608
            f"WHERE file_id IN ({places}) ORDER BY file_id, key",
            list(file_ids),
        ).fetchall()
        found: dict[str, dict[str, str | None]] = {}
        for file_id, key, value in rows:
            found.setdefault(file_id, {})[key] = value
        return found

    def _hydrate(self, row: tuple) -> Paper:
        file_id = row[0]
        return Paper(
            file_id=file_id,
            content_hash=row[1],
            store_name=row[2],
            original_name=row[3],
            source_path=row[4],
            size_bytes=row[5],
            pages_read=row[6],
            title=row[7],
            year=row[8],
            stored_mtime_ms=row[9],
            from_page_images=bool(row[10]),
            document_name=row[11] or "",
            created_at_ms=row[12],
            updated_at_ms=row[13],
            tags=self._ordered(file_id, "paper_tags", "tag"),
            authors=self._ordered(file_id, "paper_authors", "name"),
            keywords=[
                value
                for (value,) in self._conn.execute(
                    "SELECT keyword FROM paper_keywords WHERE file_id = ? ORDER BY keyword",
                    [file_id],
                ).fetchall()
            ],
        )

    def _ordered(self, file_id: str, table: str, column: str) -> list[str]:
        rows = self._conn.execute(
            f"SELECT {column} FROM {table} WHERE file_id = ? ORDER BY position",  # noqa: S608
            [file_id],
        ).fetchall()
        return [value for (value,) in rows]

    # ---- writes ------------------------------------------------------------

    def upsert(self, paper: Paper) -> None:
        """Write a paper and its multi-valued fields as one atomic change."""
        with self._transaction():
            self._write(paper)

    def _write(self, paper: Paper) -> None:
        """Write one paper. Caller owns the transaction."""
        timestamp = now_ms()
        existing = self._conn.execute(
            "SELECT created_at_ms FROM papers WHERE file_id = ?", [paper.file_id]
        ).fetchone()
        created_at = existing[0] if existing else timestamp
        self._conn.execute("DELETE FROM papers WHERE file_id = ?", [paper.file_id])
        self._conn.execute(
            """
            INSERT INTO papers (
                file_id, content_hash, store_name, document_name,
                original_name, source_path, size_bytes, stored_mtime_ms,
                pages_read, title, year, from_page_images,
                created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                paper.file_id,
                paper.content_hash,
                paper.store_name,
                paper.document_name,
                paper.original_name,
                paper.source_path,
                paper.size_bytes,
                paper.stored_mtime_ms,
                paper.pages_read,
                paper.title,
                paper.year,
                paper.from_page_images,
                created_at,
                timestamp,
            ],
        )
        self._replace_ordered(paper.file_id, "paper_tags", "tag", paper.tags)
        self._replace_ordered(paper.file_id, "paper_authors", "name", paper.authors)
        self._conn.execute(
            "DELETE FROM paper_keywords WHERE file_id = ?", [paper.file_id]
        )
        for keyword in dict.fromkeys(paper.keywords):
            self._conn.execute(
                "INSERT INTO paper_keywords VALUES (?, ?)", [paper.file_id, keyword]
            )

    def upsert_many(self, papers: Sequence[Paper]) -> None:
        """Write several documents in one transaction.

        Filing writes every document of a pass at once so the lock is taken
        once, after the copying is finished, rather than once per document
        while it is still going on — and so a failure part-way leaves none of
        them recorded rather than an arbitrary prefix.
        """
        if not papers:
            return
        with self._transaction():
            for paper in papers:
                self._write(paper)

    def set_stored_file_states(
        self, states: Sequence[tuple[str, str, int, int]]
    ) -> None:
        """Record what several stored files look like now, in one transaction.

        Each entry is ``(file_id, content_hash, size_bytes, mtime_ms)``.
        """
        if not states:
            return
        timestamp = now_ms()
        with self._transaction():
            for file_id, content_hash, size_bytes, mtime_ms in states:
                self._conn.execute(
                    """
                    UPDATE papers
                    SET content_hash = ?, size_bytes = ?, stored_mtime_ms = ?,
                        updated_at_ms = ?
                    WHERE file_id = ?
                    """,
                    [content_hash, size_bytes, mtime_ms, timestamp, file_id],
                )

    def set_tags(
        self,
        file_id: str,
        tags: list[str],
        store_name: str,
        keywords: list[str] | None = None,
    ) -> None:
        """Re-tag a document and record the new name of its store folder.

        Keywords go in the same transaction when they are given, because a
        re-tag that took the model's category and left its keywords behind would
        describe the document as two different things at once.

        They do not go through `_replace_ordered`: `paper_keywords` has no
        `position` column, being a set rather than a sequence.
        """
        with self._transaction():
            self._replace_ordered(file_id, "paper_tags", "tag", tags)
            if keywords is not None:
                self._conn.execute(
                    "DELETE FROM paper_keywords WHERE file_id = ?", [file_id]
                )
                for keyword in keywords:
                    self._conn.execute(
                        "INSERT INTO paper_keywords VALUES (?, ?)", [file_id, keyword]
                    )
            self._conn.execute(
                "UPDATE papers SET store_name = ?, updated_at_ms = ? WHERE file_id = ?",
                [store_name, now_ms(), file_id],
            )

    def set_stored_file_state(
        self, file_id: str, content_hash: str, size_bytes: int, mtime_ms: int
    ) -> None:
        """Record what the stored file looks like now.

        Called after filing, and again whenever a rescan finds the file on disk
        no longer matches what was recorded.
        """
        with self._transaction():
            self._conn.execute(
                """
                UPDATE papers
                SET content_hash = ?, size_bytes = ?, stored_mtime_ms = ?,
                    updated_at_ms = ?
                WHERE file_id = ?
                """,
                [content_hash, size_bytes, mtime_ms, now_ms(), file_id],
            )

    def set_store_layout(
        self, file_id: str, store_name: str, document_name: str
    ) -> None:
        """Record where a document's folder and file now are."""
        with self._transaction():
            self._conn.execute(
                """
                UPDATE papers
                SET store_name = ?, document_name = ?, updated_at_ms = ?
                WHERE file_id = ?
                """,
                [store_name, document_name, now_ms(), file_id],
            )

    def set_attribute(self, file_id: str, key: str, value: str | None) -> None:
        """Record a field that does not have a column yet."""
        with self._transaction():
            self._conn.execute(
                "DELETE FROM paper_attributes WHERE file_id = ? AND key = ?",
                [file_id, key],
            )
            self._conn.execute(
                "INSERT INTO paper_attributes VALUES (?, ?, ?)", [file_id, key, value]
            )

    def unset_attribute(self, file_id: str, key: str) -> bool:
        """Forget one attribute. False if the document did not have it."""
        with self._transaction():
            removed = self._conn.execute(
                "DELETE FROM paper_attributes WHERE file_id = ? AND key = ? "
                "RETURNING key",
                [file_id, key],
            ).fetchall()
        return bool(removed)

    def _replace_ordered(
        self, file_id: str, table: str, column: str, values: list[str]
    ) -> None:
        self._conn.execute(f"DELETE FROM {table} WHERE file_id = ?", [file_id])  # noqa: S608
        for position, value in enumerate(values):
            self._conn.execute(
                f"INSERT INTO {table} (file_id, position, {column}) VALUES (?, ?, ?)",  # noqa: S608
                [file_id, position, value],
            )

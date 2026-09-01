---
name: sortyourpaperya
description: Search and read a sortyourpaperya library — the documents `sortyourpaperya` has filed, with their categories, authors, years, and notes. Use when asked to find, look up, read, cite, summarise, compare, re-tag, or take notes on documents the user has filed, or when a question is about "my papers", "my library", "that bill", "the manual", or a document they say they already have.
---

# Reading a sortyourpaperya library

A library is a folder of filed documents. Every document lives in exactly one
folder in `store/`, holding the document itself and anything kept beside it;
`tree/` is a browsable view of symlinks over that, rebuilt on demand. A DuckDB
database is the source of truth for what is filed and how it is labelled.

Reach it through `sortyourpaperya`, never by walking the folders. The database knows
titles, authors, years, keywords, and categories that the filenames do not.

## Find the document

```bash
sortyourpaperya find "attention transformers" --json
```

Every word has to match, somewhere in the id, title, original filename, year,
tag, author, or keyword — so adding a word narrows and removing one widens. If
a search comes back empty, drop the most specific word and try again rather
than concluding the library does not have it.

Twenty results are shown by default. When more matched, `find` says so on
stderr — read it, because the rest are documents the library holds and this
search did not show you. Add a word to narrow, or pass `--limit 0` for all of
them.

Each record looks like this:

```json
{
  "id": "5112ee75ddcf",
  "title": "Attention Is All You Need",
  "authors": ["Ashish Vaswani"],
  "year": 2017,
  "category": "Machine Learning/Deep Learning/Transformers",
  "tags": ["Machine Learning", "Deep Learning", "Transformers"],
  "keywords": ["attention", "sequence modelling"],
  "document": "/…/store/5112ee75ddcf__…/vaswani_2017_attention-is-all-you-need.pdf",
  "folder": "/…/store/5112ee75ddcf__…",
  "notes": ["/…/store/5112ee75ddcf__…/notes.md"],
  "original_name": "1706.03762v7.pdf",
  "from_page_images": false,
  "filed_at": "2026-08-24T23:06:38+00:00",
  "updated_at": "2026-08-24T23:06:38+00:00",
  "size_bytes": 2215244,
  "pages_read": 2,
  "attributes": {"doi": "10.48550/arXiv.1706.03762"}
}
```

`document` is the file — open it to read the document. `id` is what every
other command takes. Those commands accept words too, but **pass the id you got
from `find`**: a word matching two documents is an error you then have to
resolve, and the id you already have cannot be ambiguous. `from_page_images: true` means the title, authors and year
were a model's reading of a scan rather than the document's own words, so treat
them as approximate and say so if they matter.

`sortyourpaperya list --json` gives every document the same way. Prefer `find`: a large
library is a lot to read, and the whole point of the labels is not having to.

Both take `--sort`, which is how you answer a question about *which* document
rather than which ones match: `id` (the default — a hash, so arbitrary but
stable), `recent`, `updated`, `title`, `year`, `size`. "What did I file this
week" is `sortyourpaperya list --sort recent --json`; without a sort there is no order to
read anything out of.

```bash
sortyourpaperya categories
```

Every category path in use and how many documents are under it. Read this
before re-filing anything, and before telling the user what their library
covers — it is one query, where `list --json` is the whole library to count by
hand.

Both commands, and the ones below, take `--library <path>` when the user names a
library. Without it `sortyourpaperya` resolves the one this machine watches, which is
usually right.

## Record something about it

```bash
sortyourpaperya attr 5112ee75ddcf                       # everything recorded
sortyourpaperya attr 5112ee75ddcf doi                   # one value, alone on stdout
sortyourpaperya attr 5112ee75ddcf doi 10.48550/arXiv.1706.03762   # set it
sortyourpaperya attr 5112ee75ddcf doi --unset           # forget it
```

Attributes are free key/value pairs on a document, for anything the library has
no column for and the model was never asked: a DOI, a venue, a verdict, when it
was read, what it was checked against. Keys are yours to choose.

They live in the database beside the document's own labels, which means they
**survive what everything else does not**: a re-tag, a rescan, a re-ingest of
the same document, and a rebuilt tree all leave them untouched, and they are
deleted only when the document is. That makes them the right place for a
finding you want to still be there next session — a note is prose for a person,
an attribute is a field you can search on and read back exactly.

They come back in every `find --json` and `list --json` record under
`attributes`, so recording one costs nothing to read later.

Use them rather than inventing a side-file. A file you write next to the
library is not backed up with it, does not follow a re-tagged document, and is
not deleted when the document is.

## Take notes on it

```bash
sortyourpaperya note 5112ee75ddcf --path                # its only note, or a new notes.md
sortyourpaperya note 5112ee75ddcf reading-log --path    # a note by name
sortyourpaperya note 5112ee75ddcf extracted.json --path # JSON, for a note you read back
```

Prints the path to that note, creating it if it does not exist — markdown with a
heading, JSON as `{}` — and prints nothing else. Write or append to that file
directly.

A note is any markdown or JSON file in the document's folder, so the name is
yours to choose; `notes.md` is only what a bare `sortyourpaperya note` makes when the
document has none. Every note that exists comes back in the record's `notes`
list, so read that before writing rather than starting a second file about the
same thing.

**Name the note you mean.** A bare `sortyourpaperya note <id>` opens the only note when
there is exactly one, but exits 1 and lists them once there are several rather
than guessing which you meant.

Always pass `--path`. Without it the command opens `$EDITOR`, which will hang.

Notes belong in the document's `folder`, beside the document — anything written
there is backed up with it, follows it when it is re-tagged, and is deleted with
it. Nothing written into `tree/` is durable: it is rebuilt from the database and
not backed up.

## Cite it in a paper

```bash
sortyourpaperya bib list --json                                  # which bibliographies exist
sortyourpaperya bib add --lib thesis --cite 5112ee75ddcf         # cite one document in one
```

A bibliography lives at `bibs/<slug>/` inside the library and holds two files:
`bib.toml`, the record, and `references.bib`, generated from it — the one to
point LaTeX at. `bib add` writes both. It reports the citation key it used,
which is what goes inside `\cite{}`.

**Always pass both `--lib` and `--cite`.** Either one left out is asked for at
a prompt you cannot answer.

Fill in what the entry needs *before* citing it, with `attr`: `doi`, `journal`,
`booktitle`, `publisher`, `volume`, `number`, `pages`, `url`, and the rest of
the BibTeX field names are carried into the entry, and the entry type follows
from them — a document with a `journal` is an `@article`, one with a
`booktitle` an `@inproceedings`, and one with neither `@misc`. Pass `--type` to
say outright. Attributes that are not BibTeX field names stay out of the
bibliography, so a verdict or a reading date is safe to keep on a document.

To correct or add anything afterwards, edit `bib.toml` and run
`sortyourpaperya bib build --lib <slug>`. **Never edit `references.bib`** — it
is regenerated from the TOML and the next `bib add` overwrites it.

Citing the same document twice does nothing and says so; it does not duplicate
the entry. `sortyourpaperya bib init "<name>"` starts a new bibliography, one
per manuscript.

## Re-file a document

```bash
sortyourpaperya retag 5112ee75ddcf "Medicine/Radiology"
```

Only when the user asks, or when they agree a document is filed wrongly. It
renames the document's folder and moves its link; nothing is copied and no
model is called. Run `sortyourpaperya categories` first, so a re-tag joins an existing
branch instead of opening a near-duplicate of one.

**Always pass the category.** `sortyourpaperya retag <id>` with no category asks the model
where the document belongs and then waits at a prompt for a person to accept or
reject it — a prompt you cannot answer, on a command that spends money each time
round. Decide the category yourself and pass it, or tell the user to run the
bare form themselves.

## Ask the database anything else

```bash
sortyourpaperya sql "SELECT title, year FROM papers ORDER BY created_at_ms DESC LIMIT 5" --json
```

The commands above cover the questions worth having a command for. This is the
rest of the database, and it is a lot: `papers` (with `content_hash`,
`size_bytes`, `pages_read`, `stored_mtime_ms`, `created_at_ms`,
`updated_at_ms`), `paper_tags`, `paper_authors`, `paper_keywords`,
`paper_attributes`, and `model_answers` — whose `page_text` column holds the
text already extracted from each document's first pages. Reading that is free
and already paid for; re-parsing the PDF to answer a question about its
contents is not.

`DESCRIBE papers` shows the columns of any of them. Join on `file_id`, except
`model_answers`, which is keyed by `content_hash`.

Only one reading statement is accepted — `SELECT`, `WITH`, `FROM`, `TABLE`,
`VALUES`, `DESCRIBE`, `SUMMARIZE`, `SHOW` — and anything else is refused before
it reaches the database. That is a guard against a query meant to count
documents deleting them, not a permission boundary.

Do not open `papers.duckdb` yourself with a DuckDB client. DuckDB allows one
process at a time and refuses a second connection even to read, so it works
only when no watcher is running — which is worse than not working, because it
fails intermittently and for a reason that has nothing to do with the question.

## What not to run

- **`sortyourpaperya ingest` and `sortyourpaperya watch` file new documents, and both send every
  document's text to OpenAI and cost money.** Never run either to answer a
  question. Run them only when the user asks for documents to be filed, and use
  `--mode copy` unless they ask for `move`, which drains the source folder.
- **`sortyourpaperya remove` deletes the document, its notes, and its record.** When the
  document arrived by move, that is the only copy. Ask first, every time; pass
  `--yes` only after the user has said yes to that document, and only with an
  exact id — `--yes` refuses words, because what a word matches changes as the
  library grows.
- `sortyourpaperya bib init` starts a bibliography and `bib add` edits one. Both
  only write inside `bibs/`, but a bibliography is the user's manuscript:
  run them when asked, not to tidy up.
- `sortyourpaperya fsck`, `scan`, `tree`, `migrate-store`, and `backup` are maintenance.
  They are safe, but run them when asked, not speculatively.

## When something goes wrong

- A command that hangs for a few seconds is waiting on the database lock — a
  watcher may be mid-pass. It waits up to 30 seconds, then fails.
- `sortyourpaperya fsck` reports a document whose file is gone, or a folder the database
  does not know about. `--adopt` brings such a folder back in, losing the
  title, authors, and year, which only ever lived in the database.
- A document the user is sure they filed, that `find` cannot see, is worth one
  `sortyourpaperya fsck` before saying it is not there.

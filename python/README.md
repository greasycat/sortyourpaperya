# sortyourpaperya

Read a folder of PDFs, label each one with an LLM, file it into a library, and
keep doing it as new documents arrive.

Documents are not assumed to be academic papers. Bills, receipts, and manuals
are categorized on their own terms.

The two-stage taxonomy synthesis and placement the Rust implementation had are
not reimplemented. A document's category comes from the single labelling call,
steered by the paths the library already uses.

## What it does

`ingest` walks the input folder, skips anything already known or too large,
pulls text from the first page of each PDF, and sends batches of them to the
model for keywords, a category, and the title, authors, and year the links are
named by — all in one request per batch.

Each request also carries the category paths the library already uses, so a new
document joins an existing branch instead of inventing a parallel one. Without
it an electricity bill and a water bill land under unrelated top-level folders;
with it the second joins the first. Batches within one pass run concurrently and
cannot see each other's choices, so two new documents arriving together can
still diverge — arriving one at a time, as under the watcher, they cannot. Batches run four at a time,
matching the Rust pipeline's ceiling, because the stage is bound by round-trips
rather than tokens.

`watch` does that continuously. Filesystem events are only a wake-up hint — the
folder scan is the source of truth, so the loop stays correct when events are
coalesced, missed, or caused by its own writes. A folder must stop changing for
three seconds before a pass starts, so a PDF still being copied in is never read
half-written.

Every pass says what it filed and where — `filed vaswani_2017_attention-is-all-you-need
under AI/Transformers`, one line per document — and a pass that filed something
also puts it on screen as a desktop notification, since a watcher running as a
service is otherwise only visible in a log nobody is reading at the time. The
category is the part that can be wrong, and this is what makes it noticeable
while it is still one document rather than fifty. Notifications go through
`osascript` on macOS and `notify-send` on Linux; a machine with neither, or a
service that cannot reach a session bus, loses the notification and nothing else.

## What leaves your machine

**Every document you file is sent to OpenAI.** There is no local model and no
offline mode: labelling a document means uploading the text of its first page
(or, for a scan, an image of it) to `api.openai.com`, under the key in your
`.env`. Bills, contracts, medical letters, and anything else that lands in a
watched folder go the same way as a paper does — the watcher does not ask, and
under the service it happens without anyone at the keyboard.

Concretely, each request carries:

- up to 4,000 characters of the document's text, or the first page rendered as
  an image when it carries no text layer
- the filename
- the list of category paths your library already uses, which describes the
  shape of your whole collection

The file itself is never uploaded, and nothing is sent for a document the
library already holds. What OpenAI does with the rest is between you and their
terms; this tool cannot promise anything about it. If that is not acceptable for
a folder, do not point a watch at it.

## What it costs

Two different things bound this, and it is worth keeping them apart. **Paying
twice for the same document** is a defect, and is fixed by banking the model's
answer — the section after this one. **Paying more than you meant to in a day**
is not a defect at all; it is what happens when a watch is pointed somewhere
surprising. That is what the ceiling is for:

```bash
sortyourpaperya budget           # what the last 24 hours cost, against the ceiling
sortyourpaperya budget --reset   # start the window over
```

Requests and tokens are counted in a rolling 24-hour window, machine-wide rather
than per-library, in `~/.local/state/sortyourpaperya/spend.json`. A request that would start
past the ceiling is refused before it is sent, because a limit checked afterwards
has already been paid. Defaults are 500 requests and 1,000,000 tokens a day;
raise them with `SYP_MAX_REQUESTS_PER_DAY` and `SYP_MAX_TOKENS_PER_DAY`, or set
either to `0` to turn that ceiling off.

Each request also has a retry ceiling and a timeout (`SYP_LLM_MAX_RETRIES`,
default 2; `SYP_LLM_TIMEOUT_SECONDS`, default 120), so one failing or hanging
request gives up instead of holding a pass open and billing for every attempt.

The ceilings bound the damage; they do not price it. This tool does not know
what the model costs, and does not try to guess. Nor is the ceiling a cap on how
much one pass may do: stopping a pass halfway would leave the rest of the folder
unfiled and, in copy mode, never looked at again — the watcher decides there is
work by comparing a folder snapshot against the last one it ran, and a folder it
only half-processed looks unchanged. Request size is capped where it belongs
instead: 20 documents per request, 60,000 characters across a batch, 4,000 per
document, four requests at a time.

## Paying once for a document

A pass calls the model, then copies files, then writes rows. Anything that kills
it in between — a crash, a full disk, `launchctl bootout` — has spent money and
recorded nothing, and under `KeepAlive` the thing that reliably restarts the
watcher is failure. It comes back, finds the same documents pending, and pays
for them again. Repeatably, if whatever killed it is repeatable.

So the model's answer is banked against the document's **contents** the moment
it arrives, before a single file is touched:

```bash
sortyourpaperya cache            # how many answers this library has already paid for
sortyourpaperya cache --forget   # throw them away and ask again
```

Keyed by content hash rather than document id, because the id is minted fresh on
every attempt — the point is to be found again by a pass with no memory of the
one that paid. A scan's page text is banked separately and as soon as it is
read, since reading a scan is the most expensive thing a pass does and the
labelling call after it can fail on its own.

A preview banks its answers too, so looking at what would happen and then
letting it happen is one decision and one charge. That is the only thing a
preview writes: it says nothing about what the library holds, only that a
request has already been paid for.

Answers are reused for 7 days, then asked again — `SYP_LABEL_CACHE_DAYS`, `0`
to turn reuse off, negative to keep them indefinitely. Not forever by default,
because a label is a choice made against the categories the library had at the
time, and a library that has grown since would be steered somewhere else.

This is what makes the daily ceiling a backstop rather than a budget: with it,
the same document is not bought twice, so the ceiling only catches what was
never predicted — a watch pointed at a folder of ten thousand PDFs, a key shared
with something else.

## Backups

```bash
sortyourpaperya backup ~/Backups/sortyourpaperya-2026-08-22
```

The store and the database are only useful together — the store alone is a
folder of documents nothing can find, the database alone is a catalogue of files
that are gone — so one command copies both, along with `bibs/`, which is
hand-made and derivable from nothing. `tree/` is skipped; `sortyourpaperya tree`
rebuilds it.

The database is copied first, and through DuckDB rather than as a file: a
database being written has changes in a log beside it, and a file copy taken at
the wrong moment restores short of its most recent rows or will not open at all.
The ordering matters for the same reason filing writes the file before the row.
If the watcher files something between the two halves, the copy holds a folder
with no row — an orphan, which `sortyourpaperya fsck --adopt` brings back — rather than a
row pointing at a document that no longer exists anywhere.

For a nightly copy, run it from cron or a launchd agent.

## How the library is laid out

Every PDF lives in exactly one place — a single flat folder — and the browsable
folder tree is made only of symlinks over it:

```
library/
  papers.duckdb
  store/
    5112ee75ddcf__Machine Learning__Deep Learning__Transformers/
      vaswani_2017_attention-is-all-you-need.pdf
      notes.md              <- yours, and as durable as the document
      reading-log.md        <- likewise, and a note by any other name
      figure-3.png          <- likewise, though not a note
  tree/
    Machine Learning/Deep Learning/Transformers/
      vaswani_2017_attention-is-all-you-need -> ../../../store/5112ee75ddcf__...
  bibs/
    phd-thesis/
      bib.toml              <- what you cite, and the file you edit
      references.bib        <- generated from it, and the file LaTeX reads
```

Every document has one home: a folder in the store holding the document and
whatever you keep beside it. **That folder is the durable thing** — it is what
gets backed up, what re-tagging renames, and what removal deletes.

`tree/` is a view and nothing else: delete it and `sortyourpaperya tree` rebuilds it
exactly. Each document appears there as a single link to its store folder, so
opening one from a category takes you straight to the document and its notes.

Only the category folders are real, and the tree is neither backed up nor
preserved across a rebuild-from-scratch. So `sortyourpaperya tree` reports any file it
finds living there and tells you to move it into the document's folder. It never
deletes it — that is not this tool's call.

The store filename is `<id>__<Tag>__<Tag>.pdf`. The id is permanent and the tags
are not, so **re-tagging a paper is a rename plus a moved link** — no file is
ever copied and nothing has to be found again. Tags are sanitized so no tag can
contain `__`, which keeps the name unambiguous to parse.

Links are named from whichever of author, year, and title are known —
`vaswani_2017_attention-is-all-you-need`, or `vaswani_attention…` when the year
is missing, or `attention…` when only the title is. A document with neither an
author nor a title falls back to the store name, since a bare year names
nothing. A long title is cut between words rather than inside one, so a name
never trails off mid-word.

Because those names are derived, changing how they are derived leaves existing
files spelled the old way. `sortyourpaperya migrate-store` renames them to match. They are relative, so
the whole library can be moved without breaking.

**DuckDB is the source of truth.** Filenames and the links are projections of
it, which is why `sortyourpaperya tree` can discard the links and rebuild them exactly, and
why a tag list too long for a filename loses nothing. Papers are keyed by a hash of
their contents, so re-running costs nothing and the same paper arriving twice
under different names is recognised.

Expanding the schema means appending to `_MIGRATIONS` in `db.py`; anything not
worth a column yet goes in `paper_attributes` as a key/value pair.

`bibs/` is durable in the same way the store is — hand-made, derivable from
nothing — so `sortyourpaperya backup` copies it too.

## Scanned documents

A PDF with no text layer opens fine and yields nothing to read. Its pages are
rendered with `pdftoppm` and handed to the model, which writes plain text
standing in for the text the document does not carry — the title, the authors,
the date, and a short description of what it covers.

That text then goes through the ordinary labelling call, so a scan is batched,
steered, and named exactly like any other document. It costs one extra request
per scanned document, and `from_page_images` records which documents were read
this way, since their metadata is a model's reading of a picture rather than the
document's own words.

Needs poppler (`brew install poppler`), the same dependency the Rust pipeline
already has. Without it, a scan is reported as a failure naming what to install.

launchd gives an agent a bare `PATH` that does not include Homebrew, so the
service is installed with poppler's directory added explicitly. Without that,
scans fail only under the service while working by hand — install warns if
`pdftoppm` cannot be found.

## Naming a document

Nobody remembers `78c64b3b8ef6`. Every command that takes a document — `retag`,
`note`, `remove` — also takes words, matched against the same things `sortyourpaperya find`
searches: ids, titles, original filenames, years, tags, authors, and keywords.

```bash
sortyourpaperya note kahn --path
sortyourpaperya retag "successor representations" "Cognitive Science/Computation"
```

What a word resolved to is printed **on stderr**, so `$(sortyourpaperya note kahn --path)`
still yields nothing but the path.

An exact id is never searched for, so a document can always be named
unambiguously — even one whose id happens to appear in another document's
keywords.

When several documents match, `fzf` opens on them if it is installed and there
is a terminal to draw on; the id is carried on each line but hidden from what
you see and type against. Without a picker, the matches are printed and nothing
is done:

```
error: 'psychology' matches 3 documents:
  4d402aa8e499  Psychology / Research Methods   Experimental Design and Analysis
  689c4699c74c  Psychology / Research Methods   PSY 389: Advanced Methods
  78c64b3b8ef6  Psychology / Research Methods   Trial-by-trial learning of …

add a word to narrow it, or name one by its id.
```

`sortyourpaperya remove --yes` is the one exception: it wants an exact id and refuses a
word. The confirmation is what shows you which document a word found, and with
`--yes` there is no confirmation — while which document a word matches changes
as the library grows.

## When a document is filed wrongly

```bash
sortyourpaperya retag <id> "Cognitive Science/Computational Modelling"   # you decide
sortyourpaperya retag <id>                                               # ask the model
```

Steering is what makes a second utility bill join the first, and it is also how
a document ends up somewhere wrong: a paper on computational cognitive science
joins `Psychology/Research Methods` because the library holds two
research-methods documents and nothing closer. With a category, `retag` applies
it. Without one it asks the model where the document belongs:

```
78c64b3b8ef6  Trial-by-trial learning of successor representations in human behavior
  now:        Psychology / Research Methods

suggestion 1: Cognitive Science / Computational Modelling
  keywords:   successor representations, temporal difference learning, …

[a]ccept, [s]teer, [r]egenerate, [c]ancel [a]:
```

`r` asks again, and each round is told every category already turned down — so
the model has to reconsider rather than reword. Without that the inputs would be
identical each time and the answer would be too.

`s` is for when refusing is not enough. It asks for a sentence — "it is about
the maths, not the clinic", "file it near the tax papers" — and the next round
is asked under it:

```
[a]ccept, [s]teer, [r]egenerate, [c]ancel [a]: s
  what is it about, or where should it go?: it is about the maths, not the clinic

suggestion 2: Mathematics / Probability
  keywords:   markov decision processes, temporal difference learning, …
  asked for:  it is about the maths, not the clinic
```

You are the one who has read the document, so what you say outranks everything
else in the prompt, the rejected list included — a steer that walks back a path
you turned down two rounds ago is allowed to. Saying it again replaces it
rather than piling up; an empty answer costs nothing and brings the menu back.
There is no cap on how many times you may ask; each one is a request, numbered
on screen so the count is visible, and the daily ceiling is what bounds it.

Accepting replaces the tags **and** the keywords, in one transaction, since
taking the model's category and keeping its old keywords would describe the
document as two things at once. A re-tag you type yourself leaves the keywords
alone. Title, authors, and year are never touched: they name the link, and may
have been corrected by hand.

The question the model is asked is not the one ingest asks. Ingest is told to
*prefer* an existing path, which is what misfiled the document; here the
existing paths are context to weigh, and opening a new one is a valid answer.
What it reads is the library's own record — title, authors, keywords — followed
by the document's first pages, so a scan with no text layer and a document whose
file has gone missing both still get an answer, with no extra request.

Nothing is written until you accept, and the database is let go before the first
request: the exchange waits on a person, and holding the write lock across that
would stop the watcher.

## Notes

```bash
sortyourpaperya note <id>              # the document's only note, or a new notes.md
sortyourpaperya note <id> reading-log  # a note by name; a bare word is markdown
sortyourpaperya note <id> extracted.json
```

A note is **any markdown or JSON file in the document's folder** — the name is
yours, and `notes.md` is only what you get when you ask for a note and say
nothing else. A markdown note is created with the title as a heading, a JSON one
as `{}`; nothing else is a note, so `notes.txt` is refused rather than quietly
renamed. With no `$EDITOR` set the command prints the path instead, so
`$(sortyourpaperya note <id>)` composes.

Naming one is only needed once there is more than one. A document with a single
note opens it whatever it is called, and a document with several lists them
rather than picking, because the wrong pick is written into by a caller that
asked for "the" notes.

Notes are just files in the folder, so anything else you put there — figures,
supplements, a scanned appendix — gets the same treatment, minus being reported
as a note. All of it is backed up with the document, follows it through a
re-tag, and is deleted with it, which is why `sortyourpaperya remove` asks first.

## Bibliographies

A library keeps the papers you cite; `bib` keeps what you cite them in. One
bibliography per manuscript, under `bibs/<slug>/`:

```bash
sortyourpaperya bib init "PhD Thesis"                       # bibs/phd-thesis/
sortyourpaperya bib add --lib phd-thesis --cite 5112ee75ddcf
```

Leave off `--lib`, `--cite`, or both and you are asked, with the bibliographies
listed and the first offered as the default — so a library with one takes a
keystroke, and a library with several cannot have the wrong one picked for it.
A bibliography answers to its slug or to its id, so a script that recorded the
id keeps working across a rename.

Each one holds two files:

```toml
# bibs/phd-thesis/bib.toml — the record, and the one you edit
id = "add9918a7e03"
slug = "phd-thesis"
name = "PhD Thesis"

[[source]]
key = "vaswani2017attention"
type = "article"
file_id = "5112ee75ddcf"
title = "Attention Is All You Need"
author = ["Ashish Vaswani", "Noam Shazeer"]
year = 2017
doi = "10.48550/arXiv.1706.03762"
journal = "NeurIPS"
```

```bibtex
% bibs/phd-thesis/references.bib — generated from it, and the one LaTeX reads
@article{vaswani2017attention,
  title = {{Attention Is All You Need}},
  author = {Ashish Vaswani and Noam Shazeer},
  year = {2017},
  doi = {10.48550/arXiv.1706.03762},
  journal = {NeurIPS},
}
```

**The TOML is the source of truth and the `.bib` is a projection of it**, the
same way the database is the truth behind the store's filenames. So a field the
library never knew — a page range, a corrected title, a source that is not in
the library at all — is added by editing `bib.toml` and running
`sortyourpaperya bib build`. Nothing you write into `references.bib` survives
the next write; the file says so in its own header.

Every key in a `[[source]]` table except `key`, `type`, `file_id`, and
`added_at_ms` is a BibTeX field, carried through as written, so the record can
hold a field this tool has never heard of.

Title, authors, and year come from the library's own columns. Everything else
comes from the document's attributes, so this is what `sortyourpaperya attr`
is for:

```bash
sortyourpaperya attr 5112ee75ddcf doi 10.48550/arXiv.1706.03762
sortyourpaperya attr 5112ee75ddcf journal NeurIPS
```

Only attribute keys that name a BibTeX field are carried across — `doi`,
`journal`, `booktitle`, `publisher`, `volume`, `pages`, `url` and the rest — so
a verdict or a reading date kept on the same document stays out of the
bibliography. The entry type follows from what is there: a `journal` makes it
an `@article`, a `booktitle` an `@inproceedings`, a `school` a `@phdthesis`,
and nothing at all leaves it `@misc`, since a document filed by this tool is
not assumed to be a paper. `--type` overrules all of it.

The citation key is `vaswani2017attention` — surname, year, and the first word
of the title that names something, which is the spelling nearly every reference
manager produces and so the one you will guess at when typing `\cite{`. Two
papers by the same author in the same year get `…attention` and `…attentionb`,
the way BibTeX itself answers a collision. `--key` names one outright, and is
told if the name was taken.

Citing a document twice does nothing: a source records the `file_id` it came
from, so the second `bib add` reports the key it already has.

TeX's special characters are escaped on the way into the `.bib` and left alone
in the record, so a title carrying `&`, `%`, or `_` is text in both. Titles are
double-braced, because a BibTeX style will otherwise lowercase a title that was
already capitalized the way its authors capitalized it.

## Reading the library from a program

`sortyourpaperya find` searches everything a document is described by — its id, title,
original filename, year, tags, authors, and keywords. Every word has to match
somewhere, so a second word narrows rather than widens.

```bash
sortyourpaperya find "vaswani attention" --json
```

`find` shows 20 matches by default and says on stderr when more matched, since
a cap that says nothing reads as the whole answer; `--limit 0` lifts it.

`--json`, on `find` and on `list`, prints records instead of a table. Each
carries the absolute path to the document, its folder, and every note beside it,
because a result whose file the caller cannot open is only half an answer, and a
note it cannot find is one it will write a second copy of, along with when
it was filed, how large it is, how many pages were read, and its attributes.

`--sort` orders both: `id` (the default — a hash, so arbitrary but stable),
`recent`, `updated`, `title`, `year`, `size`. Without it there is no order to
read an answer out of, so "the three most recent" is not a question the library
could be asked.

`sortyourpaperya categories` lists every category path in use with a count under each. It
is what a re-tag should be decided from, and the alternative was listing the
whole library and grouping it by hand.

`sortyourpaperya attr <id> [key] [value]` reads and writes free key/value pairs on a
document — a DOI, a venue, a verdict, the day it was read. They live in the
database beside the document's own labels and are not part of what ingest
writes, so a rescan, a re-tag, and a re-ingest of the same document all leave
them alone; only removing the document removes them. `--unset` forgets one.
They come back in every `--json` record under `attributes`.

`sortyourpaperya sql "<statement>"` is the rest of the database, for what no command
reports: `stored_mtime_ms`, `content_hash`, and the `page_text` in
`model_answers` — the text already extracted from each document, which a reader
would otherwise re-parse the PDF to get. Only a single reading statement is
accepted (`SELECT`, `WITH`, `FROM`, `TABLE`, `VALUES`, `DESCRIBE`, `SUMMARIZE`,
`SHOW`); anything else is refused before it reaches the database. That is a
guard against a query written to count documents deleting them, not a
permission boundary — whoever can run it can already read the file.

Opening `papers.duckdb` directly is not an alternative. DuckDB allows one
process, and refuses a second connection even read-only, so it works only while
no watcher is running.

`sortyourpaperya note <id> [name] --path` prints where the note lives — creating it if it
does not exist — and stops, for a caller that means to write it itself. Without
`--path` the command opens `$EDITOR`, which a program that cannot drive one
would be left holding open. A caller writing its own file should name it, since
a bare `sortyourpaperya note` refuses to choose once a document has several.

`skills/sortyourpaperya/` is an agent skill over all of it — what to run to find a document, how to read and annotate
it, and which commands cost money or delete things and so are not to be run to
answer a question. `./install.sh` links it into `~/.claude/skills`, alongside
putting `sortyourpaperya` on PATH: a skill an agent cannot find is no more use than a
command that is not on PATH. It is a symlink into the project, so editing it
takes effect without reinstalling.

The link is only made into a skills directory whose parent already exists.
A machine with no `~/.claude` will never read a skill put there, and inventing
another tool's config folder to hold one is litter — so the install says how to
place it instead. `SORTYOURPAPERYA_SKILLS_DIR` names the directory for anything that is not
Claude Code, and is taken at its word.

## When the two halves disagree

```bash
sortyourpaperya fsck            # what is wrong
sortyourpaperya fsck --adopt    # bring orphaned folders back in
```

Filing puts a document's file down before writing its row, so that an
interruption leaves a folder nothing points at rather than a row pointing at
nothing. That is the better half to be left holding, but only because `fsck`
can find it: an unclaimed folder is otherwise absent from `list`, from the tree,
from de-duplication, and from `remove` — and under `--mode move` it is the only
copy of the document.

`--adopt` gives such a folder a row. The id and tags come back from its name and
the hash from its bytes; title, authors, and year do not, because they only ever
lived in the database. Re-ingesting does **not** heal an orphan — ids are minted
fresh, so it just files a second copy.

## Editing a stored file

Annotating or re-saving a file in the store keeps its name but changes its
bytes, which makes the recorded hash stale. That matters because the hash is how
the library recognises a document it already holds: left stale, the edited copy
arriving in a watched folder later would be ingested a second time.

Every ingest reconciles the store before deciding what is already known, so
this is handled without anyone remembering to do it — including under the
watcher. It compares each stored file's size and mtime against what was recorded
and only reads the ones that moved, so a quiet library costs one stat per
document rather than a full re-read. Changed files get a fresh hash; a file that
vanished is reported rather than silently dropped. A store that cannot be read is
reported and stepped over, because failing to reconcile risks a duplicate while
refusing to run files nothing at all.

`sortyourpaperya scan` runs the same reconciliation on its own, for checking a library
without ingesting into it.

## Installing

```bash
./install.sh              # macOS or Linux
./install.sh --check      # what is missing, changing nothing
./install.sh --service    # ...and run the watcher in the background
./install.sh --uninstall  # take it back off
```

It finds a Python 3.11 or newer — trying `python3.14` down to `python3`, since
a distribution's `python3` is often older than the newest it also ships —
builds a virtualenv inside the project, links the command into `~/.local/bin`
under both its names — `sortyourpaperya` and the short `sypy`, which run the
same thing — and links the agent skill into
`~/.claude/skills`. Nothing is written outside the project, those two
directories, and (with `--service`) the supervisor's config — and the last two
get a symlink each.

Two prerequisites are checked by name rather than left to fail obscurely later:
Debian and its derivatives ship `venv` as a separate package, so a working
`python3` is not on its own enough; and `pdftoppm` is only needed for documents
with no text layer, so a missing one is a warning naming the package to install
rather than a refusal.

Override where things go with `SORTYOURPAPERYA_VENV_DIR`, `SORTYOURPAPERYA_BIN_DIR`, and
`SORTYOURPAPERYA_SKILLS_DIR`.

## Usage

Every command below is spelled `sortyourpaperya` in full. `sypy` is the same
command under a shorter name — both are installed, and the short one is what
you will actually type.

```bash

sortyourpaperya ingest --input ./inbox                 # preview: nothing is written
sortyourpaperya ingest --input ./inbox --mode copy     # copy in, leave the source alone
sortyourpaperya ingest --input ./inbox --mode move     # move in, draining the source
sortyourpaperya watch  --input ./inbox --mode copy     # keep doing it as documents arrive

sortyourpaperya list                              # what the library holds
sortyourpaperya list --json                       # ...as records, for a program to read
sortyourpaperya list --sort recent                # id (default), recent, updated, title, year, size
sortyourpaperya find "attention 2017"             # by title, author, keyword, tag, year, or id
sortyourpaperya categories                        # every category in use, and how many are under it
sortyourpaperya attr <id>                         # free key/value pairs kept on a document
sortyourpaperya attr <id> doi 10.1000/xyz         # ...set one; --unset forgets it
sortyourpaperya sql "SELECT ..."                  # the database directly, reading only
sortyourpaperya retag <id> "Systems/Databases"    # re-tag: renames the folder, moves the link
sortyourpaperya retag <id>                        # ...or ask the model, and confirm
sortyourpaperya note kahn                         # id or words: any command taking a document
sortyourpaperya note <id>                         # open this document's notes ($EDITOR)
sortyourpaperya note <id> reading-log             # ...a note by name; .md unless you say .json
sortyourpaperya note <id> --path                  # ...or just say where they are
sortyourpaperya bib init "PhD Thesis"             # start a bibliography under bibs/
sortyourpaperya bib add --lib thesis --cite <id>  # cite a document in it
sortyourpaperya bib add                           # ...or be asked which, and which
sortyourpaperya bib build --lib thesis            # re-generate the .bib from bib.toml
sortyourpaperya bib list                          # every bibliography in the library
sortyourpaperya remove <id>                       # delete link, folder, and record (asks first)
sortyourpaperya scan                              # refresh hashes of files edited in place
sortyourpaperya fsck [--adopt]                    # check the store and database agree
sortyourpaperya tree                              # rebuild the symlink tree from the database
sortyourpaperya backup <dir>                      # copy the store and the database, together
sortyourpaperya budget                            # what the last 24 hours cost at the API
sortyourpaperya cache [--forget]                  # model answers already paid for

./install.sh --uninstall               # remove the link
```

Nothing is written without `--mode`. Use `copy` for a folder you did not create
— a Downloads folder keeps its files and the library gets copies. Re-run `wire`
after changing dependencies.

## The registry

One file says what this machine watches, at
`~/.config/sortyourpaperya/config.toml`:

```toml
default = "downloads"

[watch.downloads]
input   = "~/Downloads"
library = "~/Documents/sortyourpaperya-library"
mode    = "copy"
```

```bash
sortyourpaperya watches        # what is declared, and which are running
sortyourpaperya watch          # run the default watch
sortyourpaperya watch papers   # run a named one
```

With a registry, the other commands stop needing `--library`: it resolves
CLI > `SYP_OUTPUT` > the registry's default watch > `./sorted`. A single
declared watch is the default without saying so; past that, `default` has to
name one, because picking would be a guess.

Two watches sharing an input folder or a library are refused when the file is
read, naming both — so the mistake surfaces while it is being made rather than
hours later when the second watcher will not start.

Nothing needs the registry. Passing `--input` and `--library` still works, and a
missing file is an empty registry rather than an error.

## One watcher per folder

A watcher claims its input folder and its library before it starts, and refuses
if either is already claimed:

```
error: ~/Documents/sortyourpaperya-library is already being watched as the library of a
       watcher running as pid 63643
```

Two watchers sharing either folder would file the same document twice: each
decides what the library already holds before either writes, and the database
lock is not held across that gap. Claims are kept in `~/.local/state/sortyourpaperya`, not
in the folders themselves, so they work across libraries and leave no litter.

A claim whose owner is gone is taken over rather than respected, so a crash does
not lock a folder out. That is also what cleans up after `sortyourpaperya-service
uninstall`: stopping the agent sends `SIGTERM`, which does not run the release,
so the claims are left behind until the next watcher takes them over.

Who owns a claim is decided by an `flock` on the claim file, not by the pid
written inside it. A pid is not an identity — it gets reused — so a claim left by
a crashed watcher would otherwise name whatever process was handed that number
next, and reading liveness off it locks the folder out for as long as that
stranger lives. The kernel drops an `flock` when the holder dies, however it
dies. The pid is still recorded, because "already watched by pid 63643" is what
makes the refusal actionable, but nothing is decided from it.

`sortyourpaperya ingest` is not covered by this. Running one by hand while a watcher is
going can still file a document twice, because both check before either writes.

## Running it as a service

```bash
./install.sh --service                              # the registry's default watch

./python/scripts/sortyourpaperya-service install papers     # a named watch
./python/scripts/sortyourpaperya-service status
./python/scripts/sortyourpaperya-service logs
./python/scripts/sortyourpaperya-service uninstall
```

**launchd** on macOS, **systemd --user** on Linux. The two are written the way
each platform expects rather than one being emulated on the other: they differ
in where the unit lives, how it is loaded, what happens to its output, and
whether it survives logout.

It watches in copy mode, so the watched folder is indexed but never rearranged.
`SORTYOURPAPERYA_MODE=move` overrides that. Both units carry poppler's directory on `PATH`
explicitly — a service gets a bare one that includes neither Homebrew nor
`/usr/local`, and without it every scanned document fails under the service
while the same command works by hand.

Restarting is how a transient failure recovers, and both stop rather than spin
when it is not transient: launchd throttles, and the systemd unit gives up after
five starts in five minutes, leaving the reason in the journal. A restart is
cheap now in any case — the model's answers were banked before the file work,
so a pass that comes back does not buy them again.

On Linux a `--user` service stops when your last session ends. To keep it
running after logout, `loginctl enable-linger $USER`; the installer says so if
linger is off.

Logs go through a rotating handler — 2MB a file, three kept — so a service left
running for months cannot fill the disk: `~/Library/Logs/sortyourpaperya/sortyourpaperya.log` on
macOS, `~/.local/state/sortyourpaperya/logs/sortyourpaperya.log` on Linux. Every line carries a
timestamp, because the log is read hours later and often after a restart, and
every document that failed is named along with why: a full disk, an expired key,
and a corrupt PDF all read the same as "1 failed".

The supervisor's own capture of the process goes to `sortyourpaperya.crash.log` beside it
and holds only what logging never sees — a traceback from a crash.
`sortyourpaperya-service logs` shows the tail of both.

There is one service, so installing for a different folder is refused rather
than silently replacing the running one; `uninstall` first. Reinstalling the
same folder is how configuration changes are picked up.

DuckDB allows one writing process at a time, so a running service could shut
every other command out. Three things stop it.

The watcher drops its lock whenever it is idle. A pass runs in phases, so
hashing, parsing, calling the model, and copying all happen with no connection
open and the database is visited in three short bursts between them. And a
command that still collides waits rather than failing, up to 30 seconds, past
which the problem is a stuck process rather than contention.

The phases are what keep this true as the library grows: the bursts are
proportional to the number of documents, not to the bytes being read. On twelve
4MB documents the share of a pass with the lock unavailable is 3%, against 66%
when the file work was done with the connection held.

`install.sh` delegates the virtualenv to `python/scripts/sortyourpaperya-path wire`,
which installs the dependencies from `requirements.lock` and then the package
itself in editable mode, so source edits take effect without reinstalling.

The lock is what stops two machines set up a week apart running different code,
which is what makes a break arriving with a dependency indistinguishable from
one arriving with a commit. Move it forward deliberately:

```bash
./python/scripts/sortyourpaperya-path relock    # re-resolve, then review the diff
```

It resolves in a throwaway virtualenv, so the test dependencies and whatever a
debugging session left behind stay out of it. Versions are pinned, not hashes:
it says what is installed, and does not try to prove the index handed over the
same bytes as last time. It refuses to replace a `sortyourpaperya` it did not create,
and `unwire` refuses to delete one, so an unrelated command of the same name
survives both; a skill of the same name someone else wrote survives the same
way, though as a warning rather than a refusal, since by then the command is
already installed. Override the locations with `SORTYOURPAPERYA_VENV_DIR`,
`SORTYOURPAPERYA_BIN_DIR`, and `SORTYOURPAPERYA_SKILLS_DIR`.

Without wiring, run it through the project directly:

```bash
uv run --project python sortyourpaperya ingest --input ./inbox
```

Running the tests needs the dev extras, which `wire` does not install:

```bash
python/.venv/bin/python -m pip install -e "python[dev]"
python/.venv/bin/python -m pytest
```

Every test is capped at 60 seconds by `pytest-timeout`, using the signal method
so it can interrupt a loop that never yields. Several tests drive the watcher,
and `asyncio.wait_for` cannot preempt a coroutine that stops awaiting — without
the cap, a change that removes an `await` from the watch loop spins at full CPU
until the machine is unusable.

Without `--input` the current directory is watched, and the library defaults to
`sorted` inside it.

## Configuration

Settings resolve CLI > environment > defaults, reusing the Rust pipeline's
`SYP_*` names so a folder set up for one reads the same to the other:
`SYP_INPUT`, `SYP_OUTPUT`, `SYP_RECURSIVE`, `SYP_MAX_FILE_SIZE_MB`,
`SYP_PAGE_CUTOFF`, `SYP_KEYWORD_BATCH_SIZE`, `SYP_LLM_MODEL`.

What it may spend, and how hard it tries:
`SYP_MAX_REQUESTS_PER_DAY`, `SYP_MAX_TOKENS_PER_DAY`, `SYP_LLM_MAX_RETRIES`,
`SYP_LLM_TIMEOUT_SECONDS`, `SYP_LABEL_CACHE_DAYS`.

Where it keeps what one invocation leaves for the next — watch claims and the
spend ledger — is `SORTYOURPAPERYA_STATE_DIR`, defaulting to `~/.local/state/sortyourpaperya`. The
registry is `SORTYOURPAPERYA_CONFIG_DIR`, and `SORTYOURPAPERYA_LOG_FILE` turns on the rotating log
(the service sets it; a second process rotating the same file can lose lines).

The API key is read from `OPENAI_API_KEY`, `SYP_API_KEY`, or `OEPNAI_API_KEY`,
including from the repository-root `.env`. The third spelling is a typo this
repo's `.env` currently carries; it is accepted so the tool works as-is,
and the correct spelling wins when both are set.

## Known gaps

- Only OpenAI is wired up, because that is the key the repo carries.
- No resumable run state beyond the database: an interrupted pass keeps the
  papers it filed and redoes the batch it was in the middle of — though it no
  longer pays for it, because the answers were banked before the file work.
- The bank closes the window between the model call and the rows. The narrower
  one inside it — a crash between reading a scan's pages and labelling them, in
  the same breath, with no file work in between — is not covered, because the
  page text is banked when the reading stage finishes rather than per document.
- Banked answers hold the model's reading of a document's first page, so
  clearing them (`sortyourpaperya cache --forget`) is the way to stop that text sitting in
  the library.
- `remove` deletes the stored file, which is the only copy when the document
  arrived by move. There is no unfile-but-keep option.
- Category steering sends at most 200 existing paths; a larger library sends its
  alphabetically first.
- Only watchers claim folders. A hand-run `sortyourpaperya ingest` racing a watcher can
  still file the same document twice.
- Steering can also mislead. A 1990 radiology paper joined an existing
  `Machine Learning/Explainable AI/Medical Imaging` branch because it was the
  nearest thing present, where an unsteered run put it under
  `Medicine/Radiology`. `sortyourpaperya retag` is the fix when it happens.
- Moving a link by hand still does not re-tag anything: links are relative, so
  moving one to a different depth breaks it, and a rebuild puts it back. Use
  `sortyourpaperya retag`. Files you add to a document's folder are safe either way.
- `sortyourpaperya remove` deletes the document's whole store folder, including notes and
  anything else kept in it. It confirms first.
- A file written into the tree rather than the document's folder is reported by
  `sortyourpaperya tree`, not moved or deleted. It is not durable where it sits. If it is
  sitting exactly where a document's link belongs, that document is linked under
  an id-decorated name beside it instead; the file is never replaced.
- The spend ceiling counts requests and tokens, not money. It bounds the damage
  from a restart loop; it does not know what the model costs.
- Four batches run at once and each reserves its request before sending, so the
  ceiling is enforced at the request that crosses it — the tokens that request
  turns out to cost are recorded after, and can carry the day slightly past the
  token ceiling.
- `sortyourpaperya backup` copies; it does not rotate, prune, or verify old backups.
- The lock pins versions, not hashes.
- A library made before documents had folders needs `sortyourpaperya migrate-store` once.

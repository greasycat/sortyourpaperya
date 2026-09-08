# sortyourpaperya

Use an LLM to sort documents into a folder tree you can browse.

Point it at a folder, and it reads each document, labels it, files it into a
store, and builds a symlink tree over that store by category. Run it once, or
leave it watching a folder — a Downloads folder, say — and it files things as
they arrive.

Documents are not assumed to be academic papers. Bills, receipts, and manuals
are categorized on their own terms.

## Layout

- **`python/`** — the tool. A Python ingest pipeline and folder watcher,
  installable as `sortyourpaperya`, with DuckDB as the source of truth. Start at
  [`python/README.md`](python/README.md).
- **`python/skills/`** — an agent skill for searching and reading a library
  through `sortyourpaperya find`, so an LLM can answer questions from what you have filed.
  `./install.sh` links it into `~/.claude/skills`.
- **`assets/testsets/`** — committed test-set manifests, kept as data. The
  tooling that built them is gone from `main`; it is in the history.
- **`docs/`** — measurements and design notes; `docs/archive/` keeps historical
  planning material.
- **`CHANGELOG.md`** — a verbose, newest-first log of what changed and why.

## Quick start

```bash
./install.sh                              # macOS or Linux; installs `sortyourpaperya`

sypy login                                           # store your API key in the keychain
sypy doctor                                          # check the watcher and the model
sypy ingest --input ./inbox                          # preview: nothing is written
sypy ingest --input ./inbox --mode copy              # copy in, leave the source alone
sypy watch  --input ./inbox --mode copy              # keep doing it as documents arrive
sypy pick                                            # choose several in a tree, and act on them
```

Everything else — the store layout, the registry of watched folders, running it
as a background service, what leaves your machine, and what it costs — is in
[`python/README.md`](python/README.md).

## History

This started as a Rust workspace: a `syp` binary over `syp-core`, `syp-ai`,
`syp-library`, `syp-workflow`, and `paper-db`, with a two-stage taxonomy and
placement pipeline. That implementation is preserved on the **`old-rust`**
branch, along with the evaluation harness the measurements in `docs/` came
from. It is not maintained.

The work continues in Python, which is what `main` now holds.

"""Runtime settings, resolved from CLI arguments then environment then defaults.

The Rust implementation this began as layered CLI > ENV > folder config > XDG
config > defaults. This keeps the first two layers and the same ``SYP_*``
variable names, so a folder set up for either reads the same to the other.
"""

from __future__ import annotations

import os
from time import monotonic
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# Mirrors the Rust defaults so the two pipelines read the same folder the same way.
DEFAULT_OUTPUT_DIR = "sorted"
DEFAULT_MAX_FILE_SIZE_MB = 16
DEFAULT_PAGE_CUTOFF = 1
DEFAULT_KEYWORD_BATCH_SIZE = 20
DEFAULT_MODEL = "gpt-6-luna"

# Text budget per request, matching `MAX_TOTAL_BATCH_TEXT_CHARS` and
# `MAX_TEXT_CHARS_PER_FILE` in crates/syp-library/src/papers/taxonomy/mod.rs.
MAX_TOTAL_BATCH_TEXT_CHARS = 60_000
MAX_TEXT_CHARS_PER_FILE = 4_000

# Concurrent LLM requests, matching MAX_CONCURRENT_BATCH_REQUESTS in the Rust
# llm batch layer.
MAX_CONCURRENT_REQUESTS = 4

# How many existing category paths a new document is shown so it can join a
# branch rather than invent one. Capped because the whole list rides in every
# request; a library past this many branches sends its alphabetically first.
MAX_STEERING_CATEGORIES = 200

# What the tool is allowed to spend at the API in a rolling day, and how
# hard it tries before giving a request up. The watcher runs unattended under
# `KeepAlive`, so nothing else in the process would ever stop it paying.
#
# There is no ceiling on requests: a day's filing is however many documents
# turned up that day, and a number picked here is wrong for anyone whose
# backlog is larger than the person who picked it. What the ceiling exists to
# catch is a restart loop re-reading the same folder forever, and tokens catch
# that just as well while costing nothing to a person merely filing a lot --
# they are also what the bill is actually denominated in. Set
# `SYP_MAX_REQUESTS_PER_DAY` to put the request ceiling back.
DEFAULT_MAX_REQUESTS_PER_DAY = 0
DEFAULT_MAX_TOKENS_PER_DAY = 1_000_000
DEFAULT_LLM_MAX_RETRIES = 2
DEFAULT_LLM_TIMEOUT_SECONDS = 120.0

# How long a model answer stays worth reusing. Its labels were chosen against
# the categories the library used at the time, so a receipt is worth keeping for
# a while and not forever; `0` turns the cache off entirely.
DEFAULT_LABEL_CACHE_DAYS = 7


def state_dir() -> Path:
    """Where this machine keeps what one `sortyourpaperya` invocation leaves for the next.

    Watch claims and the spend ledger, neither of which belongs to any one
    library. `SORTYOURPAPERYA_STATE_DIR` overrides it, which is how the tests keep out of
    the real user directories.
    """
    state = os.environ.get("SORTYOURPAPERYA_STATE_DIR") or os.environ.get("XDG_STATE_HOME")
    base = Path(state) if state else Path.home() / ".local" / "state"
    return Path(base) / "sortyourpaperya"


def runtime_dir() -> Path:
    """Where this machine keeps sockets, as opposed to state it keeps between runs.

    A Unix socket path is capped near 104 bytes -- the whole path, not the name --
    and `state_dir()` is nested under a home directory that can be arbitrarily
    deep. XDG puts sockets in the runtime directory for exactly this reason, and
    it is short by construction (`/run/user/1000`). macOS has no such variable, so
    `TMPDIR` stands in; it is also short, and a socket is worthless after a reboot
    anyway.

    `SORTYOURPAPERYA_RUNTIME_DIR` overrides it, which is how the tests keep out of
    the real user directories without reintroducing the length problem.
    """
    override = os.environ.get("SORTYOURPAPERYA_RUNTIME_DIR")
    if override:
        return Path(override) / "sortyourpaperya"
    base = os.environ.get("XDG_RUNTIME_DIR") or os.environ.get("TMPDIR") or "/tmp"
    return Path(base) / "sortyourpaperya"


class ConfigError(RuntimeError):
    """Raised when the resolved configuration cannot be used."""


@dataclass(frozen=True)
class Settings:
    """Everything one ingest or watch invocation needs."""

    input_dir: Path
    output_dir: Path
    recursive: bool = False
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB
    page_cutoff: int = DEFAULT_PAGE_CUTOFF
    keyword_batch_size: int = DEFAULT_KEYWORD_BATCH_SIZE
    model: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        for name in ("page_cutoff", "keyword_batch_size"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be greater than 0")
        if self.max_file_size_mb < 0:
            raise ConfigError("max_file_size_mb must be 0 (no limit) or greater")

    @property
    def max_file_size_bytes(self) -> int | None:
        """The size past which a PDF is skipped, or None when there is no limit.

        0 is "no limit", as it is for the spend ceilings: the limit exists to
        keep a stray scan of a whole book from being read, and someone filing
        one on purpose should not have to guess a number larger than it.
        """
        if self.max_file_size_mb == 0:
            return None
        return self.max_file_size_mb * 1024 * 1024


def resolve_settings(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    recursive: bool | None = None,
    page_cutoff: int | None = None,
    keyword_batch_size: int | None = None,
    model: str | None = None,
    max_file_size_mb: int | None = None,
) -> Settings:
    """Resolve settings from explicit arguments, then ``SYP_*`` variables, then defaults.

    The input folder defaults to the current directory and the library to
    ``sorted`` inside it, so pointing the watcher at a folder needs no config.
    """
    load_dotenv(_repo_dotenv(), override=False)

    resolved_input = _first_path(input_dir, os.getenv("SYP_INPUT"), Path.cwd())
    # `sorted` goes beside a single PDF rather than inside it: an input naming
    # one document still has a folder it lives in, and that folder is where a
    # library would have gone had the whole of it been named.
    beside = resolved_input.parent if resolved_input.is_file() else resolved_input
    resolved_output = _first_path(
        output_dir, os.getenv("SYP_OUTPUT"), beside / DEFAULT_OUTPUT_DIR
    )
    return Settings(
        input_dir=resolved_input,
        output_dir=resolved_output,
        recursive=_first_bool(recursive, os.getenv("SYP_RECURSIVE"), False),
        max_file_size_mb=_first_int(
            max_file_size_mb,
            os.getenv("SYP_MAX_FILE_SIZE_MB"),
            DEFAULT_MAX_FILE_SIZE_MB,
        ),
        page_cutoff=_first_int(
            page_cutoff, os.getenv("SYP_PAGE_CUTOFF"), DEFAULT_PAGE_CUTOFF
        ),
        keyword_batch_size=_first_int(
            keyword_batch_size,
            os.getenv("SYP_KEYWORD_BATCH_SIZE"),
            DEFAULT_KEYWORD_BATCH_SIZE,
        ),
        model=model or os.getenv("SYP_LLM_MODEL") or DEFAULT_MODEL,
    )


KEYRING_SERVICE = "sortyourpaperya"
KEYRING_USERNAME = "openai"

# Bumped whenever the stored key changes. A long-running watcher builds its API
# client once and keeps it; without this, a `login` while it runs would not take
# effect until someone restarted the service.
_KEY_GENERATION = 0


def key_generation() -> int:
    """A number that changes when the stored key does."""
    return _KEY_GENERATION


# What the keychain last said, and at which generation it said it. A desktop
# keyring confirms per *access*, not per process: KeePassXC has no "allow this
# program from now on", so every read is another dialog. The watcher reads on
# every `ping` -- which is what `sypy doctor` and `sypy status` each do -- and a
# person who cannot answer one of those is a person whose watcher stalls. One
# read per key, then, and `_key_changed` below is what makes a `login` land.
_CACHED_KEY: tuple[int, str | None] = (-1, None)

# How long "no key" is believed. A found key stands until `login` or `logout`
# changes it, but a miss cannot be trusted that long: the watcher starts at login
# and may ask before KeePassXC is running or unlocked, and a miss remembered
# until the next `login` left it keyless for the whole session -- while the
# keychain sat there unlocked, with the key in it. A minute keeps the dialog
# from being asked for on every `ping`, and the next request after it looks again.
KEY_MISS_SECONDS = 60.0
_MISS_UNTIL = 0.0


def _key_changed() -> None:
    global _KEY_GENERATION, _CACHED_KEY
    _KEY_GENERATION += 1
    _CACHED_KEY = (-1, None)


def forget_cached_key() -> None:
    """Read the key afresh next time: the API has just refused this one.

    A key found in the keychain is otherwise believed until `login` or
    `logout` -- but it can be rotated in the keyring itself, or stored by a
    `login` that never reached this process, and a watcher holding the old one
    would fail every request until restarted. A refusal is the one sign that
    the cached key is wrong, and it costs one keychain read to act on.
    """
    _key_changed()


# Whether this process may spend the key `login` stored. False everywhere until
# the watcher says otherwise; see `becomes_the_spender`.
_MAY_USE_KEYCHAIN = False


def becomes_the_spender() -> None:
    """Declare this process the one allowed to use the stored key.

    Called by the watcher, and only by it. `sypy login` puts a key in the
    keychain so the *service* has one -- a background process cannot be handed a
    key any other way, since neither launchd nor systemd inherits a shell. It
    does not follow that every command run by hand should quietly spend it, and
    the surprise there is expensive: an `ingest` typed at a prompt drawing on the
    service's credential looks free until the bill.

    The environment is untouched by this. `OPENAI_API_KEY=... sypy ingest` is a
    deliberate act with a key the person chose for that run, which is the
    opposite of the case above.
    """
    global _MAY_USE_KEYCHAIN
    _MAY_USE_KEYCHAIN = True


def resolve_api_key() -> str:
    """The OpenAI key: the environment, plus the keychain in the watcher only.

    The keychain is read only by the process that called `becomes_the_spender`,
    which is the watcher. Everywhere else this is the environment and `.env`
    alone -- see that function for why, and note it is consulted *first* in the
    watcher, so `login` is not shadowed by a stale export.

    The keychain matters because it is the one place the key is neither in
    a file next to the code nor in a shell profile that no supervisor inherits.
    `sypy login` puts it there; macOS Keychain and the Linux Secret
    Service both unlock at login, so the watcher has it after a reboot without
    anything being typed.

    The environment still works and is still what a container or a CI job uses,
    so nothing that worked before stops working.

    Last, and in the watcher only, `watcher_env_path`: the key for a session
    with no keychain to keep it in.

    ``OEPNAI_API_KEY`` is accepted because the repository's own ``.env`` spells
    it that way; the correct spelling wins when both are set.
    """
    if _MAY_USE_KEYCHAIN:
        stored = key_from_keychain()
        if stored:
            return stored
    load_dotenv(_repo_dotenv(), override=False)
    for name in ("OPENAI_API_KEY", "SYP_API_KEY", "OEPNAI_API_KEY"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    if _MAY_USE_KEYCHAIN:
        fallback = key_from_env_file()
        if fallback:
            return fallback
    if not _MAY_USE_KEYCHAIN:
        # Deliberately without asking the keychain whether it holds one. Reading
        # it is what makes a locked keyring prompt, and a command that cannot
        # use the stored key has no business waking that dialog to write a
        # better error message. A process that may not spend the stored key
        # never touches it at all.
        raise ConfigError(
            "no API key for this command. `sypy login` stores a key for the "
            "watcher, which is what spends it -- start the watcher and let it do "
            "the work (`sypy watch`), or set OPENAI_API_KEY for this run."
        )
    raise ConfigError(
        "no API key found; run `sypy login` so the watcher has one, "
        f"or put OPENAI_API_KEY=... in {watcher_env_path()} for a machine "
        "with no keychain"
    )


def watcher_env_path() -> Path:
    """The file the watcher falls back to when there is no keychain to read.

    Beside the watch registry, because both belong to the person rather than to
    the checkout. It exists for the session the keychain is absent from -- a
    KeePassXC that is not running, a headless box -- where `login` has nowhere
    to put the key and the service, inheriting no shell, has no other way to
    be handed one.
    """
    from .registry import registry_path

    return registry_path().parent / "env"


def key_from_env_file() -> str | None:
    """The key in `watcher_env_path`, or None when there is no such file or key.

    Read with `dotenv_values` rather than loaded, so the file answers for the
    key and nothing else in it leaks into the process environment.
    """
    path = watcher_env_path()
    if not path.is_file():
        return None
    values = dotenv_values(path)
    for name in ("OPENAI_API_KEY", "SYP_API_KEY"):
        value = (values.get(name) or "").strip()
        if value:
            return value
    return None


def _keyring():
    """The `keyring` module, with a backend chosen now rather than at import.

    `keyring` picks its backend the first time it is used and keeps that choice
    for the life of the process. The watcher is started at login and may well
    beat the desktop keyring to the bus -- KeePassXC registers
    org.freedesktop.secrets only while it is running, and nothing on the bus
    can activate it -- so the choice made in that second is "no backend", and
    it would otherwise stand for the whole session. A long-lived process asks
    again, but only when the cached answer was a failure, so a machine that
    genuinely has no keychain still pays one D-Bus probe per call and no more.
    """
    import keyring

    try:
        from keyring.backends import fail

        if isinstance(keyring.get_keyring(), fail.Keyring):
            keyring.core.init_backend()
    except Exception:
        # A stub in place of the real module, or a `keyring` whose internals
        # have moved: whatever is there is what gets used, unexamined.
        pass
    return keyring


def key_from_keychain() -> str | None:
    """The stored key, or None when there is none or no keychain to ask.

    Answered from the last read until the key changes; see `_CACHED_KEY`. The
    absence of a key is cached too, but only for `KEY_MISS_SECONDS` -- a watcher
    started before `login` is asked for one by every `ping`, and prompting each
    time is the same dialog storm for a keychain that has nothing to give, while
    remembering it for good misses a keychain that comes up after the watcher.

    A machine with no keychain — a container, a headless box with no Secret
    Service running — is not an error here: it is the ordinary case for the
    environment variable below, so a missing backend falls through rather than
    refusing to start.
    """
    global _CACHED_KEY, _MISS_UNTIL
    generation, cached = _CACHED_KEY
    if generation == _KEY_GENERATION and (
        cached is not None or monotonic() < _MISS_UNTIL
    ):
        return cached
    value = _read_keychain()
    _CACHED_KEY = (_KEY_GENERATION, value)
    _MISS_UNTIL = monotonic() + KEY_MISS_SECONDS if value is None else 0.0
    return value


def _read_keychain() -> str | None:
    """Ask the keychain itself. Prompts, on a keyring that confirms per access."""
    try:
        keyring = _keyring()

        value = (keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME) or "").strip()
        if value:
            return value
    except Exception:
        return None
    return _from_any_collection()


def _from_any_collection() -> str | None:
    """The key from a Secret Service collection other than the default one.

    `keyring` searches only the collection aliased "default", so a key kept in
    any other one -- an ordinary thing to do, and what a renamed or second
    keyring leaves you with -- reads back as "no key found" while sitting there
    perfectly intact.

    **Locked collections are skipped, never unlocked.** Reading a locked item is
    what raises a password prompt, and the process that needs this most is a
    background service with nobody watching it: prompting there hangs the
    watcher rather than failing it. A locked collection is treated as one that
    does not have the key.

    Returns None wherever Secret Service is not the backend -- macOS Keychain
    has no collections to search, so `keyring` either found it or it is absent.
    """
    try:
        import secretstorage
    except Exception:
        return None
    if secretstorage is None:  # blocked, as the test suite does
        return None
    try:
        connection = secretstorage.dbus_init()
        for collection in secretstorage.get_all_collections(connection):
            if collection.is_locked():
                continue
            for item in collection.search_items(
                {"service": KEYRING_SERVICE, "username": KEYRING_USERNAME}
            ):
                if item.is_locked():
                    continue
                found = (item.get_secret() or b"").decode().strip()
                if found:
                    return found
    except Exception:
        return None
    return None


def store_api_key(key: str) -> None:
    """Put the key in the platform keychain. Raises ConfigError if there is none."""
    try:
        keyring = _keyring()

        keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, key)
    except Exception as exc:
        raise ConfigError(
            f"no keychain available to store the key: {exc}. Without one, the "
            f"watcher reads OPENAI_API_KEY from {watcher_env_path()} instead"
        ) from exc
    _key_changed()


def forget_api_key() -> bool:
    """Remove the stored key. True if there was one, False if there was nothing.

    A backend can refuse the delete while allowing a write: a Secret Service
    collection that is locked raises on `delete_password`, and gnome-keyring
    routinely has a locked collection beside an unlocked one. Logging out has
    to work anyway, so the key is overwritten with nothing when the delete is
    refused — `key_from_keychain` reads a blank entry as no key, so the effect
    on every later run is the same. Only if that also fails is it an error.
    """
    try:
        keyring = _keyring()
    except Exception as exc:
        raise ConfigError(f"no keychain available: {exc}") from exc

    try:
        had_one = bool((keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME) or "").strip())
    except Exception:
        # A backend that refuses the read may still accept the delete, and
        # logging out has to work regardless of which half is unavailable.
        # Whether there was a key becomes unknown, not a reason to stop.
        had_one = True

    if not had_one:
        return False

    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception:
        try:
            keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, "")
        except Exception as exc:
            raise ConfigError(f"could not clear the stored key: {exc}") from exc
    _key_changed()
    return True


def _repo_dotenv() -> Path:
    """The repository-root ``.env``, which is where this repo keeps its key."""
    return Path(__file__).resolve().parents[3] / ".env"


def label_cache_max_age_ms() -> int | None:
    """How old a cached model answer may be, or None when nothing expires.

    `SYP_LABEL_CACHE_DAYS=0` disables reuse, so every document is labelled
    afresh; a negative value keeps answers indefinitely.
    """
    days = env_int("SYP_LABEL_CACHE_DAYS", DEFAULT_LABEL_CACHE_DAYS)
    if days == 0:
        return 0
    if days < 0:
        return None
    return days * 24 * 60 * 60 * 1000


def env_int(name: str, default: int) -> int:
    """An integer read from the environment, or `default` when it is not set."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as err:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from err


def env_float(name: str, default: float) -> float:
    """A number read from the environment, or `default` when it is not set."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as err:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from err


def _first_path(*candidates: Path | str | None) -> Path:
    for candidate in candidates:
        if candidate:
            return Path(candidate).expanduser().resolve()
    raise ConfigError("no path candidate resolved")


def _first_int(*candidates: int | str | None) -> int:
    for candidate in candidates:
        if candidate is None or candidate == "":
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError) as err:
            raise ConfigError(f"expected an integer, got {candidate!r}") from err
    raise ConfigError("no integer candidate resolved")


def _first_bool(*candidates: bool | str | None) -> bool:
    for candidate in candidates:
        if candidate is None or candidate == "":
            continue
        if isinstance(candidate, bool):
            return candidate
        lowered = str(candidate).strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"expected a boolean-like value, got {candidate!r}")
    raise ConfigError("no boolean candidate resolved")

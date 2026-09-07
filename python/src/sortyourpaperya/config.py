"""Runtime settings, resolved from CLI arguments then environment then defaults.

The Rust implementation this began as layered CLI > ENV > folder config > XDG
config > defaults. This keeps the first two layers and the same ``SYP_*``
variable names, so a folder set up for either reads the same to the other.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Mirrors the Rust defaults so the two pipelines read the same folder the same way.
DEFAULT_OUTPUT_DIR = "sorted"
DEFAULT_MAX_FILE_SIZE_MB = 16
DEFAULT_PAGE_CUTOFF = 1
DEFAULT_KEYWORD_BATCH_SIZE = 20
DEFAULT_MODEL = "gpt-5.6-terra"

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
DEFAULT_MAX_REQUESTS_PER_DAY = 500
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
        for name in ("max_file_size_mb", "page_cutoff", "keyword_batch_size"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be greater than 0")


def resolve_settings(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    recursive: bool | None = None,
    page_cutoff: int | None = None,
    keyword_batch_size: int | None = None,
    model: str | None = None,
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
            None, os.getenv("SYP_MAX_FILE_SIZE_MB"), DEFAULT_MAX_FILE_SIZE_MB
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


def resolve_api_key() -> str:
    """The OpenAI key: from the keychain if `login` put one there, else the environment.

    The keychain comes first because it is the one place the key is neither in
    a file next to the code nor in a shell profile that no supervisor inherits.
    `sypy login` puts it there; macOS Keychain and the Linux Secret
    Service both unlock at login, so the watcher has it after a reboot without
    anything being typed.

    The environment still works and is still what a container or a CI job uses,
    so nothing that worked before stops working.

    ``OEPNAI_API_KEY`` is accepted because the repository's own ``.env`` spells
    it that way; the correct spelling wins when both are set.
    """
    stored = key_from_keychain()
    if stored:
        return stored
    load_dotenv(_repo_dotenv(), override=False)
    for name in ("OPENAI_API_KEY", "SYP_API_KEY", "OEPNAI_API_KEY"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    raise ConfigError(
        "no API key found; run `sypy login`, "
        "or set OPENAI_API_KEY in the environment or in .env"
    )


def key_from_keychain() -> str | None:
    """The stored key, or None when there is none or no keychain to ask.

    A machine with no keychain — a container, a headless box with no Secret
    Service running — is not an error here: it is the ordinary case for the
    environment variable below, so a missing backend falls through rather than
    refusing to start.
    """
    try:
        import keyring

        return (keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME) or "").strip() or None
    except Exception:
        return None


def store_api_key(key: str) -> None:
    """Put the key in the platform keychain. Raises ConfigError if there is none."""
    try:
        import keyring

        keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, key)
    except Exception as exc:
        raise ConfigError(f"no keychain available to store the key: {exc}") from exc


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
        import keyring
    except Exception as exc:
        raise ConfigError(f"no keychain available: {exc}") from exc

    try:
        if not (keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME) or "").strip():
            return False
    except Exception as exc:
        raise ConfigError(f"cannot read the keychain: {exc}") from exc

    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception:
        try:
            keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, "")
        except Exception as exc:
            raise ConfigError(f"could not clear the stored key: {exc}") from exc
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

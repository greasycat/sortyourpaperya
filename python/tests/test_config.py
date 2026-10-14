"""Where the API key comes from.

The key is the one setting that cannot live in the config file, because a
config file is the thing you back up and share. `sypy login` puts it
in the platform keychain instead; the environment still works for a container
or a CI job. These pin the order, since a key source that fails quietly is a
watcher that spends a night failing on every document.
"""

from __future__ import annotations

import pytest

from sortyourpaperya import config
from sortyourpaperya.config import ConfigError, resolve_api_key


@pytest.fixture
def keychain(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A keychain in memory, standing in for Keychain / Secret Service."""
    store: dict[tuple[str, str], str] = {}

    class FakeKeyring:
        @staticmethod
        def get_password(service, user):
            return store.get((service, user))

        @staticmethod
        def set_password(service, user, value):
            store[(service, user)] = value

        @staticmethod
        def delete_password(service, user):
            del store[(service, user)]

    monkeypatch.setitem(__import__("sys").modules, "keyring", FakeKeyring)
    return store


@pytest.fixture
def no_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine with no keychain at all — a container, or a headless box."""
    monkeypatch.setattr(config, "key_from_keychain", lambda: None)


@pytest.fixture
def spender(monkeypatch: pytest.MonkeyPatch) -> None:
    """This process is the watcher, so it may spend the stored key."""
    monkeypatch.setattr(config, "_MAY_USE_KEYCHAIN", True)


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # The developer's own .env and exported key must not decide these.
    for name in ("OPENAI_API_KEY", "SYP_API_KEY", "OEPNAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)


def test_login_stores_the_key_and_the_watcher_uses_it(
    keychain: dict, spender: None
) -> None:
    config.store_api_key("sk-from-login")
    assert resolve_api_key() == "sk-from-login"


def test_the_keychain_wins_over_a_stale_export(
    keychain: dict, spender: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Otherwise `login` would appear to do nothing on a machine whose .zshrc
    # still exports last year's key.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stale-export")
    config.store_api_key("sk-current")
    assert resolve_api_key() == "sk-current"


def test_logout_removes_it_and_says_whether_there_was_one(keychain: dict) -> None:
    config.store_api_key("sk-x")
    assert config.forget_api_key() is True
    assert config.key_from_keychain() is None
    assert config.forget_api_key() is False


def test_logout_falls_back_to_the_environment(
    keychain: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    config.store_api_key("sk-keychain")
    config.forget_api_key()
    assert resolve_api_key() == "sk-env"


def test_surrounding_whitespace_is_not_part_of_the_key(
    keychain: dict, spender: None
) -> None:
    # A key pasted from a browser arrives with a newline more often than not.
    config.store_api_key("sk-pasted\n")
    assert resolve_api_key() == "sk-pasted"


def test_an_empty_keychain_entry_is_not_a_key(keychain: dict) -> None:
    config.store_api_key("   ")
    assert config.key_from_keychain() is None


def test_logout_works_when_the_backend_refuses_to_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A locked Secret Service collection raises on delete but allows a write.

    gnome-keyring routinely has a locked collection beside an unlocked one, so
    this is the ordinary case on Linux, not an exotic one — and `logout` that
    dies with a traceback there is `logout` that does not work.
    """
    store = {("sortyourpaperya", "openai"): "sk-x"}

    class RefusesDelete:
        @staticmethod
        def get_password(service, user):
            return store.get((service, user))

        @staticmethod
        def set_password(service, user, value):
            store[(service, user)] = value

        @staticmethod
        def delete_password(service, user):
            raise RuntimeError("Item is locked!")

    monkeypatch.setitem(__import__("sys").modules, "keyring", RefusesDelete)
    assert config.forget_api_key() is True
    assert config.key_from_keychain() is None       # blank reads as no key
    assert config.forget_api_key() is False         # and stays gone


def test_no_keychain_is_not_an_error_when_the_environment_has_one(
    no_keychain: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A container has no Secret Service. That is the ordinary case for the
    # environment variable, not a reason to refuse to start.
    monkeypatch.setenv("SYP_API_KEY", "sk-plain-env")
    assert resolve_api_key() == "sk-plain-env"


def test_a_keychain_that_raises_falls_through_rather_than_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Exploding:
        @staticmethod
        def get_password(service, user):
            raise RuntimeError("no D-Bus session")

    monkeypatch.setitem(__import__("sys").modules, "keyring", Exploding)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert resolve_api_key() == "sk-env"


def test_storing_without_a_keychain_says_so_rather_than_pretending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Exploding:
        @staticmethod
        def set_password(service, user, value):
            raise RuntimeError("no backend")

    monkeypatch.setitem(__import__("sys").modules, "keyring", Exploding)
    with pytest.raises(ConfigError, match="no keychain available"):
        config.store_api_key("sk-x")


def test_no_key_anywhere_names_the_login_command(no_keychain: None) -> None:
    with pytest.raises(ConfigError, match="sypy login"):
        resolve_api_key()


# ---- who may spend the stored key ------------------------------------------


def test_a_command_run_by_hand_does_not_spend_the_watchers_key(
    keychain: dict,
) -> None:
    """The requirement this whole arrangement exists for.

    `sypy login` stores a key so the *background service* has one -- a process
    started by launchd or systemd cannot be handed a key any other way. It does
    not follow that an `ingest` typed at a prompt should quietly spend it, and
    the surprise is expensive: it looks free until the bill.
    """
    config.store_api_key("sk-the-watchers-key")
    with pytest.raises(ConfigError, match="which is what spends it"):
        resolve_api_key()


def test_the_refusal_says_what_to_do_instead(keychain: dict) -> None:
    config.store_api_key("sk-the-watchers-key")
    with pytest.raises(ConfigError, match="sypy watch"):
        resolve_api_key()


def test_a_key_chosen_for_this_run_is_still_honoured(
    keychain: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Setting OPENAI_API_KEY for one command is a deliberate act with a key the
    # person picked for it, which is the opposite of quietly drawing on the
    # service's credential. It must keep working.
    config.store_api_key("sk-the-watchers-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mine-for-this-run")
    assert resolve_api_key() == "sk-mine-for-this-run"


def test_the_watcher_spends_the_stored_key(keychain: dict, spender: None) -> None:
    config.store_api_key("sk-the-watchers-key")
    assert resolve_api_key() == "sk-the-watchers-key"


# ---- the watcher's fallback when there is no keychain ----------------------


def _write_env_file(text: str):
    path = config.watcher_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_the_watcher_falls_back_to_its_env_file(no_keychain: None, spender: None) -> None:
    # KeePassXC not running, a headless box: `login` has nowhere to put the key,
    # and a service inherits no shell, so this file is how it is handed one.
    _write_env_file("OPENAI_API_KEY=sk-from-the-file\n")
    assert resolve_api_key() == "sk-from-the-file"


def test_the_keychain_still_wins_over_the_env_file(keychain: dict, spender: None) -> None:
    _write_env_file("OPENAI_API_KEY=sk-from-the-file\n")
    config.store_api_key("sk-the-watchers-key")
    assert resolve_api_key() == "sk-the-watchers-key"


def test_a_command_run_by_hand_does_not_spend_the_env_file(no_keychain: None) -> None:
    # The file is the watcher's credential, the same as the keychain entry it
    # stands in for, so the spender rule covers it too.
    _write_env_file("OPENAI_API_KEY=sk-from-the-file\n")
    with pytest.raises(ConfigError, match="which is what spends it"):
        resolve_api_key()


def test_the_env_file_leaks_nothing_into_the_environment(
    no_keychain: None, spender: None
) -> None:
    import os

    _write_env_file("OPENAI_API_KEY=sk-from-the-file\nSYP_LLM_MODEL=other\n")
    resolve_api_key()
    assert "SYP_LLM_MODEL" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ


def test_no_keychain_to_store_in_names_the_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    class Exploding:
        @staticmethod
        def set_password(service, user, value):
            raise RuntimeError("no backend")

    monkeypatch.setitem(__import__("sys").modules, "keyring", Exploding)
    with pytest.raises(ConfigError, match=str(config.watcher_env_path())):
        config.store_api_key("sk-x")


def test_a_client_never_reads_the_keychain_at_all(monkeypatch) -> None:
    """Not even to write a better error message.

    Reading it is what makes a locked keyring put a password dialog in front of
    whoever is sitting there, and a command that may not spend the stored key
    has no business waking that.
    """
    touched = []

    class Loud:
        @staticmethod
        def get_password(service, username):
            touched.append((service, username))
            return "sk-should-not-be-read"

    monkeypatch.setitem(__import__("sys").modules, "keyring", Loud)
    with pytest.raises(ConfigError):
        resolve_api_key()
    assert touched == []


# ---- finding a key kept outside the default collection ---------------------


def _fake_secretstorage(monkeypatch, collections):
    """A Secret Service with the given collections, for the fallback lookup."""
    import types

    class Item:
        def __init__(self, attrs, secret, locked=False):
            self._a, self._s, self._locked = attrs, secret, locked
            self.read = False

        def is_locked(self):
            return self._locked

        def get_secret(self):
            self.read = True
            return self._s.encode()

    class Collection:
        def __init__(self, label, items, locked=False):
            self._label, self._items, self._locked = label, items, locked

        def get_label(self):
            return self._label

        def is_locked(self):
            return self._locked

        def search_items(self, attrs):
            return [i for i in self._items if i._a == attrs]

    module = types.SimpleNamespace(
        dbus_init=lambda: object(),
        get_all_collections=lambda _c: collections,
    )
    monkeypatch.setitem(__import__("sys").modules, "secretstorage", module)
    return Collection, Item


def test_a_key_in_a_non_default_collection_is_found(monkeypatch) -> None:
    """`keyring` searches only the collection aliased "default".

    A key kept in any other one — what a renamed or second keyring leaves you
    with — read back as "no key found" while sitting there intact.
    """

    class NoDefaultEntry:
        @staticmethod
        def get_password(service, username):
            return None

    monkeypatch.setitem(__import__("sys").modules, "keyring", NoDefaultEntry)
    Collection, Item = _fake_secretstorage(monkeypatch, [])
    item = Item({"service": "sortyourpaperya", "username": "openai"}, "sk-elsewhere")
    _fake_secretstorage(monkeypatch, [Collection("other", [item])])

    assert config.key_from_keychain() == "sk-elsewhere"


def test_a_locked_collection_is_skipped_rather_than_unlocked(monkeypatch) -> None:
    """Reading a locked item is what raises a password prompt.

    The process that needs this most is a background service with nobody
    watching it, where a prompt hangs the watcher instead of failing it.
    """

    class NoDefaultEntry:
        @staticmethod
        def get_password(service, username):
            return None

    monkeypatch.setitem(__import__("sys").modules, "keyring", NoDefaultEntry)
    Collection, Item = _fake_secretstorage(monkeypatch, [])
    item = Item({"service": "sortyourpaperya", "username": "openai"}, "sk-locked")
    _fake_secretstorage(monkeypatch, [Collection("locked one", [item], locked=True)])

    assert config.key_from_keychain() is None
    assert item.read is False  # never touched, so never prompted


def test_the_default_collection_still_wins(monkeypatch) -> None:
    class HasIt:
        @staticmethod
        def get_password(service, username):
            return "sk-from-default"

    monkeypatch.setitem(__import__("sys").modules, "keyring", HasIt)
    Collection, Item = _fake_secretstorage(monkeypatch, [])
    other = Item({"service": "sortyourpaperya", "username": "openai"}, "sk-elsewhere")
    _fake_secretstorage(monkeypatch, [Collection("other", [other])])

    assert config.key_from_keychain() == "sk-from-default"
    assert other.read is False  # no need to look further


def test_a_backend_chosen_before_the_keyring_existed_is_reconsidered(monkeypatch) -> None:
    """`keyring` picks a backend once and keeps it for the life of the process.

    The watcher starts at login and routinely beats the desktop keyring onto
    the bus — KeePassXC registers org.freedesktop.secrets only while running,
    and nothing can activate it — so the choice made in that second is "no
    backend", and `sypy login` through a day-old watcher fails against a
    keyring that has been sitting there unlocked for hours.
    """
    import sys
    import types

    class Failing:
        """Stands in for keyring.backends.fail.Keyring."""

    stored: dict = {}
    working = types.SimpleNamespace(
        get_password=lambda s, u: stored.get((s, u)),
        set_password=lambda s, u, v: stored.__setitem__((s, u), v),
    )
    backend = [Failing()]

    module = types.SimpleNamespace(
        get_keyring=lambda: backend[0],
        core=types.SimpleNamespace(
            init_backend=lambda: backend.__setitem__(0, working)
        ),
        get_password=lambda s, u: backend[0].get_password(s, u),
        set_password=lambda s, u, v: backend[0].set_password(s, u, v),
    )
    monkeypatch.setitem(sys.modules, "keyring", module)
    fail = types.SimpleNamespace(Keyring=Failing)
    monkeypatch.setitem(sys.modules, "keyring.backends", types.SimpleNamespace(fail=fail))
    monkeypatch.setitem(sys.modules, "keyring.backends.fail", fail)

    config.store_api_key("sk-after-the-keyring-woke-up")
    assert stored[(config.KEYRING_SERVICE, config.KEYRING_USERNAME)] == (
        "sk-after-the-keyring-woke-up"
    )


def test_the_keychain_is_read_once_per_key_not_once_per_question(monkeypatch) -> None:
    """A desktop keyring confirms per access, not per process.

    KeePassXC has no "allow this program from now on", so every read is another
    dialog -- and the watcher is asked for the key by every `ping`, which is
    what `sypy doctor` and `sypy status` each do. A `login` must still land
    without a restart, which is what the generation counter is for.
    """
    import sys
    import types

    reads = []
    stored = {"key": "sk-first"}
    module = types.SimpleNamespace(
        get_password=lambda s, u: (reads.append((s, u)), stored["key"])[1],
        set_password=lambda s, u, v: stored.__setitem__("key", v),
    )
    monkeypatch.setitem(sys.modules, "keyring", module)
    monkeypatch.setattr(config, "_MAY_USE_KEYCHAIN", True)

    assert [config.resolve_api_key() for _ in range(5)] == ["sk-first"] * 5
    assert len(reads) == 1, "one dialog, not five"

    config.store_api_key("sk-second")
    assert config.resolve_api_key() == "sk-second"
    assert len(reads) == 2, "a login lands without restarting the watcher"


def test_having_no_key_is_remembered_too(monkeypatch) -> None:
    """A watcher started before `login` is asked for a key by every `ping`.

    Prompting each time is the same dialog storm for a keychain that has
    nothing to give.
    """
    import sys
    import types

    reads = []
    monkeypatch.setitem(
        sys.modules,
        "keyring",
        types.SimpleNamespace(get_password=lambda s, u: (reads.append(1), None)[1]),
    )
    monkeypatch.setitem(sys.modules, "secretstorage", None)
    monkeypatch.setattr(config, "_MAY_USE_KEYCHAIN", True)

    for _ in range(5):
        assert config.key_from_keychain() is None
    assert len(reads) == 1


def test_a_keychain_that_comes_up_later_is_found_once_the_miss_expires(monkeypatch) -> None:
    """The watcher starts at login, often before KeePassXC is up or unlocked.

    A miss remembered until the next `login` left it keyless all session with
    the key sitting in an unlocked keychain.
    """
    import sys
    import types

    stored: list[str | None] = [None]
    monkeypatch.setitem(
        sys.modules,
        "keyring",
        types.SimpleNamespace(get_password=lambda s, u: stored[0]),
    )
    monkeypatch.setitem(sys.modules, "secretstorage", None)
    monkeypatch.setattr(config, "_MAY_USE_KEYCHAIN", True)
    clock = [1000.0]
    monkeypatch.setattr(config, "monotonic", lambda: clock[0])

    assert config.key_from_keychain() is None
    stored[0] = "sk-unlocked-later"
    assert config.key_from_keychain() is None          # still inside the minute
    clock[0] += config.KEY_MISS_SECONDS + 1
    assert config.key_from_keychain() == "sk-unlocked-later"


# ---- the size limit ---------------------------------------------------------


def test_the_size_limit_comes_from_the_argument_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYP_MAX_FILE_SIZE_MB", "50")
    assert config.resolve_settings(max_file_size_mb=200).max_file_size_bytes == 200 * 1024 * 1024
    assert config.resolve_settings().max_file_size_mb == 50


def test_a_size_limit_of_zero_is_no_limit() -> None:
    assert config.resolve_settings(max_file_size_mb=0).max_file_size_bytes is None


def test_a_negative_size_limit_is_refused() -> None:
    with pytest.raises(ConfigError, match="max_file_size_mb"):
        config.resolve_settings(max_file_size_mb=-1)

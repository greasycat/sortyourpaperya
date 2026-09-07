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

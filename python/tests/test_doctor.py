"""`doctor`: what is set up, what is running, what works.

It exists to answer "why did nothing get filed" in one command instead of four,
so what matters is that a real problem exits non-zero and names the fix. A
doctor that reports "ok" while the watcher is dead is worse than no doctor.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sortyourpaperya import cli
from sortyourpaperya.cli import app


def _run(*args: str):
    return CliRunner().invoke(app, ["doctor", *args])


@pytest.fixture(autouse=True)
def _no_network_or_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch the API or the user's real service files from a test."""
    monkeypatch.setattr(cli, "_probe_api", lambda key: None)
    monkeypatch.setattr(cli, "_service_file", lambda: None)
    # No watcher unless a test says otherwise, so these never dial a real socket.
    monkeypatch.setattr(cli, "_serving_watcher", lambda: None)


@pytest.fixture
def watching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A registry with one healthy, stopped watch."""
    inbox, library = tmp_path / "inbox", tmp_path / "library"
    inbox.mkdir()
    library.mkdir()
    (library / "papers.duckdb").touch()
    config = tmp_path / "config"
    (config / "sortyourpaperya").mkdir(parents=True)
    (config / "sortyourpaperya" / "config.toml").write_text(
        f'[watch.papers]\ninput = "{inbox}"\nlibrary = "{library}"\nmode = "copy"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("SORTYOURPAPERYA_CONFIG_DIR", str(config))
    monkeypatch.setenv("SORTYOURPAPERYA_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    # Patch where it is *looked up*, not only where doctor imported it: the
    # resolution order runs inside config, so patching the cli name alone left
    # these reading the developer's own keychain -- passing or failing on
    # whether whoever ran them happened to be logged in.
    monkeypatch.setattr(cli, "key_from_keychain", lambda: None)
    monkeypatch.setattr("sortyourpaperya.config.key_from_keychain", lambda: None)
    return inbox, library


def test_a_healthy_setup_exits_zero(watching) -> None:
    result = _run()
    assert result.exit_code == 0, result.output
    assert "nothing to fix" in result.output


def test_no_watcher_and_no_key_is_a_warning_not_a_failure(
    watching, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing serving, the watcher's credential is not knowable from here.

    The stored key belongs to the watcher and a client may not read it, so
    "I cannot see a key" says nothing about whether the service has one. Calling
    that a failure would have `doctor` cry wolf on a perfectly fine setup.
    """
    for name in ("OPENAI_API_KEY", "SYP_API_KEY", "OEPNAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("sortyourpaperya.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("sortyourpaperya.config.key_from_keychain", lambda: None)

    result = _run()
    assert "no watcher running" in result.output
    assert "sypy login" in result.output
    assert result.exit_code == 0


def test_a_watcher_without_a_key_is_a_failure(
    watching, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This is the broken setup worth shouting about: something is running and
    # will fail on the first document it is handed.
    monkeypatch.setattr(cli, "_serving_watcher", lambda: {"has_key": False})
    result = _run()
    assert result.exit_code == 1
    assert "the watcher has no API key" in result.output


def test_a_watcher_with_a_key_is_reported_without_reading_the_keychain(
    watching, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode():
        raise AssertionError("doctor must not read the keychain; it asks the watcher")

    monkeypatch.setattr("sortyourpaperya.config.key_from_keychain", explode)
    monkeypatch.setattr(cli, "_serving_watcher", lambda: {"has_key": True})
    result = _run("--offline")
    assert "the watcher has a key" in result.output
    assert result.exit_code == 0


def test_a_watcher_whose_key_the_api_refuses_is_a_failure(
    watching, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"Has a key" is not "has a key that works".

    An expired key looks exactly like a working one until the first document,
    and only the watcher can find out which it is -- the key is its to read.
    """
    monkeypatch.setattr(cli, "_serving_watcher", lambda: {"has_key": True})
    monkeypatch.setattr(
        "sortyourpaperya.client.call", lambda *a, **k: {"ok": False, "detail": "401"}
    )
    result = _run()
    assert result.exit_code == 1
    assert "refused the watcher's key" in result.output


def test_a_watcher_whose_key_works_passes(
    watching, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_serving_watcher", lambda: {"has_key": True})
    monkeypatch.setattr("sortyourpaperya.client.call", lambda *a, **k: {"ok": True})
    result = _run()
    assert result.exit_code == 0
    assert "the API accepts the watcher's key" in result.output


def test_a_key_the_api_refuses_fails(watching, monkeypatch: pytest.MonkeyPatch) -> None:
    # The check that a key merely *exists* is the one that lulls you: an expired
    # key looks exactly like a working one until the first document.
    monkeypatch.setattr(cli, "_probe_api", lambda key: "AuthenticationError: 401")
    result = _run()
    assert result.exit_code == 1
    assert "the API refused" in result.output


def test_offline_skips_the_api_call(watching, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(key):
        raise AssertionError("--offline must not reach the network")

    monkeypatch.setattr(cli, "_probe_api", explode)
    assert _run("--offline").exit_code == 0


def test_a_missing_input_folder_is_a_failure(watching) -> None:
    inbox, _ = watching
    inbox.rmdir()
    result = _run()
    assert result.exit_code == 1
    assert "input folder is missing" in result.output


def test_no_watches_declared_fails_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("SORTYOURPAPERYA_CONFIG_DIR", str(config))
    monkeypatch.setenv("SORTYOURPAPERYA_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    result = _run()
    assert result.exit_code == 1
    assert "no watches declared" in result.output


def test_an_installed_service_that_is_not_running_is_a_failure(
    watching, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Installed-but-dead is the case worth catching: the folder looks watched,
    # and nothing has been filed for a week.
    monkeypatch.setattr(cli, "_service_file", lambda: Path("/tmp/sortyourpaperya.service"))
    result = _run()
    assert result.exit_code == 1
    assert "nothing is running" in result.output


def test_stopped_without_a_service_is_only_a_warning(watching) -> None:
    # Running it by hand is a legitimate way to use this.
    result = _run()
    assert result.exit_code == 0
    assert "no background service is installed" in result.output

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
    monkeypatch.setattr(cli, "key_from_keychain", lambda: None)
    return inbox, library


def test_a_healthy_setup_exits_zero(watching) -> None:
    result = _run()
    assert result.exit_code == 0, result.output
    assert "nothing to fix" in result.output


def test_a_missing_key_fails_and_names_login(
    watching, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("OPENAI_API_KEY", "SYP_API_KEY", "OEPNAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("sortyourpaperya.config.load_dotenv", lambda *a, **k: False)

    result = _run()
    assert result.exit_code == 1
    assert "sypy login" in result.output


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

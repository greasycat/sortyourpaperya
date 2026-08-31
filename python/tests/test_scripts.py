"""The install and service scripts.

They are the first thing anyone runs and the only part of the project that
touches a supervisor, so what they generate is worth pinning. The systemd path
cannot run on macOS and the launchd path cannot run on Linux, so both are
exercised with the supervisor command stubbed on PATH: the unit the script
writes is the thing being checked, not whether this machine can load it.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
INSTALL = PROJECT / "install.sh"
SERVICE = PROJECT / "python" / "scripts" / "sortyourpaperya-service"
WIRE = PROJECT / "python" / "scripts" / "sortyourpaperya-path"


def _stub(directory: Path, name: str, body: str = "exit 0") -> Path:
    """A command that answers instead of doing anything."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _run(script: Path, *args: str, env: dict, cwd: Path | None = None):
    return subprocess.run(
        [str(script), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        cwd=str(cwd or PROJECT),
    )


@pytest.fixture
def home(tmp_path: Path) -> Path:
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


@pytest.fixture
def watched(tmp_path: Path) -> tuple[Path, Path]:
    inbox, library = tmp_path / "inbox", tmp_path / "library"
    inbox.mkdir()
    return inbox, library


def _service_env(home: Path, tmp_path: Path, init: str, extra: dict | None = None) -> dict:
    """Point the script at a throwaway HOME with a stubbed supervisor."""
    stubs = tmp_path / "stubs"
    _stub(stubs, "systemctl", 'echo "systemctl $*" >> "$SORTYOURPAPERYA_TEST_CALLS"')
    _stub(stubs, "launchctl", 'echo "launchctl $*" >> "$SORTYOURPAPERYA_TEST_CALLS"; exit 1')
    _stub(stubs, "loginctl")
    env = {
        "HOME": str(home),
        "SORTYOURPAPERYA_INIT_SYSTEM": init,
        "SORTYOURPAPERYA_UNIT_DIR": str(home / "units"),
        "SORTYOURPAPERYA_LOG_DIR": str(home / "logs"),
        "SORTYOURPAPERYA_VENV_DIR": str(tmp_path / "venv"),
        "SORTYOURPAPERYA_TEST_CALLS": str(tmp_path / "calls.txt"),
        "PATH": f"{stubs}:{os.environ['PATH']}",
    }
    env.update(extra or {})
    # A `sortyourpaperya` for the script to find. It never runs: both folders are passed.
    fake_venv_bin = tmp_path / "venv" / "bin"
    _stub(fake_venv_bin, "sortyourpaperya")
    return env


# ---- syntax ----------------------------------------------------------------


@pytest.mark.parametrize("script", [INSTALL, SERVICE, WIRE], ids=lambda p: p.name)
def test_the_script_parses(script: Path) -> None:
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


@pytest.mark.parametrize("script", [INSTALL, SERVICE, WIRE], ids=lambda p: p.name)
def test_the_script_is_executable(script: Path) -> None:
    # It is documented as `./install.sh`, which only works if it is.
    assert os.access(script, os.X_OK)


# ---- what the service scripts generate -------------------------------------


def test_the_systemd_unit_runs_the_watcher_on_the_named_folders(
    home: Path, tmp_path: Path, watched: tuple[Path, Path]
) -> None:
    inbox, library = watched
    env = _service_env(home, tmp_path, "systemd")

    result = _run(SERVICE, "install", str(inbox), str(library), env=env)

    assert result.returncode == 0, result.stderr
    unit = (home / "units" / "sortyourpaperya.service").read_text()
    assert f'--input "{inbox}"' in unit
    assert f'--library "{library}"' in unit
    assert "--mode copy" in unit


def test_the_systemd_unit_restarts_but_gives_up_on_a_crashloop(
    home: Path, tmp_path: Path, watched: tuple[Path, Path]
) -> None:
    """Restarting is how a transient failure recovers.

    A service that fails instantly and forever should stop instead of spinning,
    leaving the reason somewhere it can be read.
    """
    inbox, library = watched
    _run(SERVICE, "install", str(inbox), str(library), env=_service_env(home, tmp_path, "systemd"))

    unit = (home / "units" / "sortyourpaperya.service").read_text()
    assert "Restart=always" in unit
    assert "RestartSec=" in unit
    assert "StartLimitBurst=" in unit


def test_the_systemd_unit_carries_poppler_and_the_rotating_log(
    home: Path, tmp_path: Path, watched: tuple[Path, Path]
) -> None:
    # A user service gets a bare PATH; without pdftoppm on it every scanned PDF
    # fails under the service while working by hand. Without SORTYOURPAPERYA_LOG_FILE the
    # log lands in a capture file nothing trims.
    inbox, library = watched
    poppler = tmp_path / "poppler"
    _stub(poppler, "pdftoppm")
    env = _service_env(home, tmp_path, "systemd")
    env["PATH"] = f"{poppler}:{env['PATH']}"

    _run(SERVICE, "install", str(inbox), str(library), env=env)

    unit = (home / "units" / "sortyourpaperya.service").read_text()
    assert f"Environment=PATH={poppler}:" in unit
    assert f"Environment=SORTYOURPAPERYA_LOG_FILE={home}/logs/sortyourpaperya.log" in unit


def test_installing_a_systemd_service_enables_it(
    home: Path, tmp_path: Path, watched: tuple[Path, Path]
) -> None:
    # Written but never enabled is a service that does nothing until reboot.
    inbox, library = watched
    env = _service_env(home, tmp_path, "systemd")

    _run(SERVICE, "install", str(inbox), str(library), env=env)

    calls = Path(env["SORTYOURPAPERYA_TEST_CALLS"]).read_text()
    assert "systemctl --user daemon-reload" in calls
    assert "systemctl --user enable --now sortyourpaperya.service" in calls


def test_the_launchd_plist_runs_the_watcher_on_the_named_folders(
    home: Path, tmp_path: Path, watched: tuple[Path, Path]
) -> None:
    import plistlib

    inbox, library = watched
    env = _service_env(home, tmp_path, "launchd")

    _run(SERVICE, "install", str(inbox), str(library), env=env)

    plist = home / "Library" / "LaunchAgents" / "com.sortyourpaperya.watcher.plist"
    payload = plistlib.loads(plist.read_bytes())
    args = payload["ProgramArguments"]
    assert args[args.index("--input") + 1] == str(inbox)
    assert args[args.index("--library") + 1] == str(library)
    assert payload["KeepAlive"] is True
    assert payload["EnvironmentVariables"]["SORTYOURPAPERYA_LOG_FILE"].endswith("sortyourpaperya.log")


def test_a_watched_folder_with_a_space_survives_the_unit(
    home: Path, tmp_path: Path
) -> None:
    """`~/My Documents` is an ordinary thing to watch.

    systemd splits ExecStart on whitespace, so an unquoted path would reach the
    watcher as two arguments and the service would never start.
    """
    inbox = tmp_path / "My Downloads"
    inbox.mkdir()
    library = tmp_path / "My Library"
    env = _service_env(home, tmp_path, "systemd")

    _run(SERVICE, "install", str(inbox), str(library), env=env)
    unit = (home / "units" / "sortyourpaperya.service").read_text()
    assert f'--input "{inbox}"' in unit

    # And it reads back as one argument, which is what refuses a second folder.
    other = tmp_path / "other"
    other.mkdir()
    result = _run(SERVICE, "install", str(other), str(library), env=env)
    assert "My Downloads" in result.stderr


@pytest.mark.parametrize("init", ["systemd", "launchd"])
def test_installing_over_a_different_folder_is_refused(
    home: Path, tmp_path: Path, watched: tuple[Path, Path], init: str
) -> None:
    """One service, one name.

    Installing for a different folder would replace the running one without
    saying so, leaving someone believing both folders were watched.
    """
    inbox, library = watched
    other = tmp_path / "other-inbox"
    other.mkdir()
    env = _service_env(home, tmp_path, init)
    if init == "launchd":
        # `is_loaded` asks the supervisor; say yes so the guard is reached.
        _stub(tmp_path / "stubs", "launchctl", 'echo "launchctl $*" >> "$SORTYOURPAPERYA_TEST_CALLS"')
    _run(SERVICE, "install", str(inbox), str(library), env=env)

    result = _run(SERVICE, "install", str(other), str(library), env=env)

    assert result.returncode != 0
    assert "already watching" in result.stderr
    assert str(inbox) in result.stderr


def test_reinstalling_the_same_folder_is_how_settings_are_picked_up(
    home: Path, tmp_path: Path, watched: tuple[Path, Path]
) -> None:
    inbox, library = watched
    env = _service_env(home, tmp_path, "systemd")
    _run(SERVICE, "install", str(inbox), str(library), env=env)

    result = _run(SERVICE, "install", str(inbox), str(library), env={**env, "SORTYOURPAPERYA_MODE": "move"})

    assert result.returncode == 0, result.stderr
    assert "--mode move" in (home / "units" / "sortyourpaperya.service").read_text()


def test_an_unknown_mode_is_refused_before_anything_is_written(
    home: Path, tmp_path: Path, watched: tuple[Path, Path]
) -> None:
    inbox, library = watched
    env = _service_env(home, tmp_path, "systemd", {"SORTYOURPAPERYA_MODE": "delete-everything"})

    result = _run(SERVICE, "install", str(inbox), str(library), env=env)

    assert result.returncode != 0
    assert "SORTYOURPAPERYA_MODE" in result.stderr
    assert not (home / "units" / "sortyourpaperya.service").exists()


def test_uninstalling_removes_the_unit(
    home: Path, tmp_path: Path, watched: tuple[Path, Path]
) -> None:
    inbox, library = watched
    env = _service_env(home, tmp_path, "systemd")
    _run(SERVICE, "install", str(inbox), str(library), env=env)
    assert (home / "units" / "sortyourpaperya.service").exists()

    result = _run(SERVICE, "uninstall", env=env)

    assert result.returncode == 0, result.stderr
    assert not (home / "units" / "sortyourpaperya.service").exists()


# ---- the installer ---------------------------------------------------------


def test_check_reports_a_missing_interpreter_and_changes_nothing(
    home: Path, tmp_path: Path
) -> None:
    # An empty PATH but for the shell: no python of any name is findable.
    bare = tmp_path / "bare"
    bare.mkdir()
    for name in ("bash", "uname", "dirname", "basename", "command", "id"):
        found = shutil.which(name)
        if found:
            (bare / name).symlink_to(found)

    result = _run(
        INSTALL, "--check", env={"HOME": str(home), "PATH": str(bare), "NO_COLOR": "1"}
    )

    assert result.returncode != 0
    assert "no Python" in result.stdout + result.stderr
    assert not (tmp_path / "venv").exists()


def test_a_python_too_old_to_run_this_is_refused(home: Path, tmp_path: Path) -> None:
    """Present is not the same as usable.

    A distribution's `python3` is often older than the newest it also ships, so
    finding one says nothing until its version has been asked for.
    """
    stubs = tmp_path / "stubs"
    for name in ("python3.14", "python3.13", "python3.12", "python3.11", "python3"):
        _stub(stubs, name, "exit 1")  # fails the >= 3.11 check

    result = _run(
        INSTALL,
        "--check",
        env={
            "HOME": str(home),
            "PATH": f"{stubs}:{os.environ['PATH']}",
            "NO_COLOR": "1",
        },
    )

    assert result.returncode != 0
    assert "no Python 3.11 or newer" in result.stdout + result.stderr


def test_check_names_the_package_that_provides_venv(home: Path, tmp_path: Path) -> None:
    """Debian ships `venv` separately, so a working python3 is not enough.

    Without this the failure is an opaque one from inside ensurepip.
    """
    stubs = tmp_path / "stubs"
    without_venv = (
        'if [ "$1" = "-c" ]; then '
        'case "$2" in *ensurepip*) exit 1;; *version_info*) exit 0;; esac; fi; exit 0'
    )
    # Every name the installer tries, or it finds a real interpreter instead.
    for name in ("python3.14", "python3.13", "python3.12", "python3.11", "python3"):
        _stub(stubs, name, without_venv)
    _stub(stubs, "apt-get")
    env = {
        "HOME": str(home),
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "NO_COLOR": "1",
    }

    result = _run(INSTALL, "--check", env=env)

    assert result.returncode != 0
    assert "python3-venv" in result.stdout + result.stderr


def test_a_missing_pdftoppm_is_a_warning_not_a_refusal(
    home: Path, tmp_path: Path
) -> None:
    # Only scanned documents need it, and that failure is reported per document
    # rather than stopping a pass — so it must not block the install.
    bare = tmp_path / "bare"
    bare.mkdir()
    for name in ("bash", "uname", "dirname", "basename", "id", "python3", "apt-get"):
        found = shutil.which(name)
        if found and not (bare / name).exists():
            (bare / name).symlink_to(found)

    result = _run(INSTALL, "--check", env={"HOME": str(home), "PATH": str(bare), "NO_COLOR": "1"})

    output = result.stdout + result.stderr
    assert "pdftoppm not found" in output
    assert result.returncode == 0, output


def test_the_installer_says_how_to_put_it_on_path(home: Path, tmp_path: Path) -> None:
    # The one step that is not automatic, so it has to be spelled out.
    result = _run(
        INSTALL, "--help", env={"HOME": str(home), "NO_COLOR": "1"}
    )

    assert "SORTYOURPAPERYA_BIN_DIR" in result.stderr


# ---- the agent skill -------------------------------------------------------


def _wired(home: Path, tmp_path: Path, extra: dict | None = None) -> dict:
    """A wire that installs nothing: the venv is stubbed, so no pip, no network."""
    venv_bin = tmp_path / "venv" / "bin"
    _stub(venv_bin, "python")  # answers the version check and every pip call
    _stub(venv_bin, "sortyourpaperya")
    _stub(venv_bin, "sypy")
    env = {
        "HOME": str(home),
        "SORTYOURPAPERYA_PYTHON": str(_stub(tmp_path / "stubs", "python3")),
        "SORTYOURPAPERYA_VENV_DIR": str(tmp_path / "venv"),
        "SORTYOURPAPERYA_BIN_DIR": str(home / "bin"),
        "SORTYOURPAPERYA_FROM_INSTALLER": "1",
    }
    env.update(extra or {})
    return env


def test_both_command_names_land_on_path(home: Path, tmp_path: Path) -> None:
    """`sypy` is the name anyone types; it has to be a real link, not an alias.

    A shell alias would be invisible to the service, to a script, and to an
    agent — none of which read your rc file.
    """
    result = _run(WIRE, "wire", env=_wired(home, tmp_path))

    assert result.returncode == 0, result.stderr
    for name in ("sortyourpaperya", "sypy"):
        link = home / "bin" / name
        assert link.is_symlink(), f"{name} was not linked"
        assert link.resolve() == tmp_path / "venv" / "bin" / name


def test_unwiring_takes_both_names_back_off(home: Path, tmp_path: Path) -> None:
    env = _wired(home, tmp_path)
    _run(WIRE, "wire", env=env)

    result = _run(WIRE, "unwire", env=env)

    assert result.returncode == 0, result.stderr
    for name in ("sortyourpaperya", "sypy"):
        link = home / "bin" / name
        assert not link.exists() and not link.is_symlink()


def test_a_sypy_someone_else_wrote_does_not_strand_our_own_link(
    home: Path, tmp_path: Path
) -> None:
    """Refusing over one name must still take the other off, and still fail."""
    env = _wired(home, tmp_path)
    _run(WIRE, "wire", env=env)
    (home / "bin" / "sypy").unlink()
    _stub(home / "bin", "sypy")  # a real file now, not our symlink

    result = _run(WIRE, "unwire", env=env)

    assert result.returncode != 0, "the foreign link should be reported"
    assert not (home / "bin" / "sortyourpaperya").exists(), "ours still came off"
    assert (home / "bin" / "sypy").is_file(), "theirs to keep"


def test_wiring_links_the_skill_where_the_agent_looks(home: Path, tmp_path: Path) -> None:
    """A skill an agent cannot find is no more use than a command off PATH."""
    (home / ".claude").mkdir()

    result = _run(WIRE, "wire", env=_wired(home, tmp_path))

    assert result.returncode == 0, result.stderr
    link = home / ".claude" / "skills" / "sortyourpaperya"
    assert link.is_symlink()
    assert link.resolve() == (PROJECT / "python" / "skills" / "sortyourpaperya")
    assert (link / "SKILL.md").is_file()


def test_no_skills_directory_is_invented_for_an_agent_that_is_not_there(
    home: Path, tmp_path: Path
) -> None:
    """A machine with no ~/.claude will never read a skill put there.

    Creating another tool's config folder to hold one is litter, so the install
    says how to place it by hand instead.
    """
    result = _run(WIRE, "wire", env=_wired(home, tmp_path))

    assert result.returncode == 0, result.stderr
    assert not (home / ".claude").exists()
    assert "SORTYOURPAPERYA_SKILLS_DIR" in result.stdout


def test_skills_dir_is_taken_at_its_word(home: Path, tmp_path: Path) -> None:
    """Anything that is not Claude Code says where it looks, and is believed."""
    elsewhere = tmp_path / "agent" / "skills"

    result = _run(
        WIRE, "wire", env=_wired(home, tmp_path, {"SORTYOURPAPERYA_SKILLS_DIR": str(elsewhere)})
    )

    assert result.returncode == 0, result.stderr
    assert (elsewhere / "sortyourpaperya" / "SKILL.md").is_file()


def test_a_skill_of_the_same_name_someone_else_wrote_survives(
    home: Path, tmp_path: Path
) -> None:
    """Theirs to keep — and warned about, not silently replaced."""
    mine = home / ".claude" / "skills" / "sortyourpaperya"
    mine.mkdir(parents=True)
    (mine / "SKILL.md").write_text("mine", encoding="utf-8")

    wired = _run(WIRE, "wire", env=_wired(home, tmp_path))
    unwired = _run(WIRE, "unwire", env=_wired(home, tmp_path))

    assert wired.returncode == 0, wired.stderr
    assert "leaving it alone" in wired.stdout + unwired.stdout
    assert (mine / "SKILL.md").read_text() == "mine"


def test_unwiring_takes_the_skill_back_off(home: Path, tmp_path: Path) -> None:
    (home / ".claude").mkdir()
    env = _wired(home, tmp_path)
    _run(WIRE, "wire", env=env)
    link = home / ".claude" / "skills" / "sortyourpaperya"
    assert link.is_symlink()

    result = _run(WIRE, "unwire", env=env)

    assert result.returncode == 0, result.stderr
    assert not link.exists() and not link.is_symlink()


def test_the_skill_comes_off_even_when_the_command_link_is_not_ours(
    home: Path, tmp_path: Path
) -> None:
    """`unwire` refuses outright over a `sortyourpaperya` it did not create.

    Doing the skill first is what stops that refusal leaving the skill behind.
    """
    (home / ".claude").mkdir()
    env = _wired(home, tmp_path)
    _run(WIRE, "wire", env=env)
    link = home / ".claude" / "skills" / "sortyourpaperya"
    (home / "bin" / "sortyourpaperya").unlink()
    _stub(home / "bin", "sortyourpaperya")  # a real file now, not our symlink

    result = _run(WIRE, "unwire", env=env)

    assert result.returncode != 0, "the command link should still be refused"
    assert not link.exists() and not link.is_symlink()


def test_the_installer_says_where_the_skill_goes(home: Path) -> None:
    result = _run(INSTALL, "--help", env={"HOME": str(home), "NO_COLOR": "1"})

    assert "SORTYOURPAPERYA_SKILLS_DIR" in result.stderr

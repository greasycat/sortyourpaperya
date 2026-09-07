"""The wire between a command and the watcher.

Both halves import this module, so a mistake here is a mistake in both at once
and shows up as a hang or a wrong answer rather than a traceback. These pin the
parts that would fail quietly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sortyourpaperya import protocol
from sortyourpaperya.budget import BudgetExceeded
from sortyourpaperya.config import ConfigError
from sortyourpaperya.library import LibraryError


def test_a_request_survives_the_round_trip() -> None:
    frame = protocol.request("attr_set", {"id": "abc", "key": "doi"})
    assert protocol.decode(protocol.encode(frame)) == frame


def test_a_path_crosses_as_a_path() -> None:
    # Paths are the most common argument here and are not JSON on their own.
    # `dump` is what makes them carriable -- `encode` deliberately refuses
    # anything `dump` did not handle rather than quietly stringifying it.
    frame = {"v": 1, "op": "note", "args": protocol.dump({"path": Path("/tmp/x.md")})}
    back = protocol.load(protocol.decode(protocol.encode(frame))["args"])
    assert back["path"] == Path("/tmp/x.md")


def test_a_value_dump_did_not_handle_is_refused_not_stringified() -> None:
    # A datetime column read over the socket would otherwise come back as text
    # where the same query run locally gives a datetime.
    import datetime

    with pytest.raises(TypeError):
        protocol.encode({"v": 1, "r": datetime.datetime(2024, 1, 2)})


@pytest.mark.parametrize(
    "exc", [LibraryError("no such document"), ConfigError("no key"), BudgetExceeded("spent")]
)
def test_a_known_failure_keeps_its_type_and_text(exc: Exception) -> None:
    # `cli.py` catches these by type; crossing the wire must not turn them into
    # something its `except` clauses no longer match.
    frame = protocol.failed(exc)
    with pytest.raises(type(exc), match=str(exc)):
        protocol.raise_for(frame["kind"], frame["message"])


def test_an_unexpected_failure_does_not_impersonate_a_known_one() -> None:
    # A daemon-side bug must not arrive looking like a condition the client
    # knows how to handle and recovers from.
    frame = protocol.failed(ZeroDivisionError("boom"))
    assert frame["kind"] == "RuntimeError"
    with pytest.raises(RuntimeError):
        protocol.raise_for(frame["kind"], frame["message"])


def test_a_version_this_build_cannot_read_says_what_to_do() -> None:
    with pytest.raises(protocol.ProtocolError, match="restart the watcher"):
        protocol.check_version({"v": protocol.VERSION + 1})


def test_the_current_version_passes() -> None:
    protocol.check_version(protocol.request("ping"))


@pytest.mark.parametrize("line", [b"not json", b'"a string"', b"[1,2]", b"null"])
def test_anything_that_is_not_an_object_is_refused(line: bytes) -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(line)


def test_an_oversized_frame_is_refused_before_it_is_parsed() -> None:
    with pytest.raises(protocol.ProtocolError, match="larger than"):
        protocol.decode(b"x" * (protocol.MAX_FRAME_BYTES + 1))


def test_an_event_field_may_be_called_name(tmp_path) -> None:
    # A filed document has a name, so this is the field a caller reaches for
    # first; the frame builder must not claim the word for itself.
    frame = protocol.event("filed", name="vaswani_2017.pdf", under="AI")
    assert frame == {"event": "filed", "name": "vaswani_2017.pdf", "under": "AI"}


def test_each_library_gets_its_own_socket(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    assert protocol.socket_path(a) != protocol.socket_path(b)
    assert protocol.socket_path(a) == protocol.socket_path(a)


def test_the_socket_name_does_not_grow_with_the_library_path(tmp_path: Path) -> None:
    # macOS caps a Unix socket path near 104 bytes. Hashing the library path is
    # what keeps a library nested a few folders down from blowing through it, so
    # the name must be the same size however deep the library is.
    deep = tmp_path.joinpath(*[f"folder-with-a-long-name-{i}" for i in range(12)])
    assert len(str(protocol.socket_path(deep))) == len(str(protocol.socket_path(tmp_path)))
    assert len(str(protocol.socket_path(deep))) < 104


def test_a_runtime_dir_too_deep_to_bind_says_so_before_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `bind` reports only "path too long", which names neither the setting that
    # caused it nor the fix.
    monkeypatch.setenv("SORTYOURPAPERYA_RUNTIME_DIR", "/" + "x" * 120)
    with pytest.raises(protocol.ProtocolError, match="SORTYOURPAPERYA_RUNTIME_DIR"):
        protocol.socket_path(tmp_path)

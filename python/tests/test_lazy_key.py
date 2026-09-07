"""The watcher does not need a key to start.

It needs one to *spend*, but not to hold the write lock or serve the socket --
and those are what the other commands now depend on it for. Resolving the key at
startup meant a watcher with no key could not come up, so it could not answer a
read during a pass either, and the service failed on a ten-second timer until
systemd parked it.
"""

from __future__ import annotations

import pytest

from sortyourpaperya.config import ConfigError
from sortyourpaperya.llm import LazyOpenAiClient


@pytest.fixture(autouse=True)
def _no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "SYP_API_KEY", "OEPNAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("sortyourpaperya.config.load_dotenv", lambda *a, **k: False)


def test_building_it_resolves_nothing() -> None:
    # The whole point: this is what the watcher constructs at startup.
    LazyOpenAiClient("gpt-5.6-terra")


@pytest.mark.asyncio
async def test_the_key_is_resolved_at_the_first_request(monkeypatch) -> None:
    asked: list[str] = []
    monkeypatch.setattr(
        "sortyourpaperya.config.resolve_api_key",
        lambda: asked.append("asked") or "sk-x",
    )
    built: list[tuple] = []
    monkeypatch.setattr(
        "sortyourpaperya.llm.OpenAiClient",
        lambda *a, **k: built.append(a) or _Stub(),
    )
    client = LazyOpenAiClient("m")
    assert asked == []
    await client.suggest_category(object())
    assert asked == ["asked"]


@pytest.mark.asyncio
async def test_a_missing_key_surfaces_at_the_request_not_at_startup() -> None:
    client = LazyOpenAiClient("m")
    with pytest.raises(ConfigError):
        await client.suggest_category(object())


@pytest.mark.asyncio
async def test_a_key_stored_while_it_runs_is_picked_up_without_a_restart(
    monkeypatch,
) -> None:
    """A failure is not cached, so logging in does not require a restart."""
    client = LazyOpenAiClient("m")
    with pytest.raises(ConfigError):
        await client.suggest_category(object())

    monkeypatch.setattr("sortyourpaperya.config.resolve_api_key", lambda: "sk-later")
    monkeypatch.setattr("sortyourpaperya.llm.OpenAiClient", lambda *a, **k: _Stub())
    assert await client.suggest_category(object()) == "ok"


class _Stub:
    async def suggest_category(self, *a, **k):
        return "ok"

    async def extract_keywords(self, *a, **k):
        return "ok"

    async def describe_pages(self, *a, **k):
        return "ok"


@pytest.mark.asyncio
async def test_a_login_while_it_runs_replaces_the_live_key(monkeypatch) -> None:
    """A watcher builds its client once and keeps it.

    Without invalidation a `sypy login` would not take effect until someone
    restarted the service -- and it would go on spending the key it replaced.
    """
    import sortyourpaperya.config as config
    import sortyourpaperya.llm as llm

    keys = iter(["sk-first", "sk-second"])
    monkeypatch.setattr(config, "resolve_api_key", lambda: next(keys))
    monkeypatch.setattr(llm, "OpenAiClient", lambda key, *a, **k: _Keyed(key))

    client = LazyOpenAiClient("m")
    assert await client.suggest_category(object()) == "sk-first"
    assert await client.suggest_category(object()) == "sk-first"  # cached

    config._key_changed()
    assert await client.suggest_category(object()) == "sk-second"


class _Keyed:
    def __init__(self, key):
        self.key = key

    async def suggest_category(self, *a, **k):
        return self.key

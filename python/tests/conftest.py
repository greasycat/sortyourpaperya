from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest
from pypdf import PdfWriter

from sortyourpaperya.config import Settings
from sortyourpaperya.extract import PaperText
from sortyourpaperya.llm import CategorySuggestion, KeywordPair, LlmError
from sortyourpaperya.render import PageImage


@dataclass
class SuggestionCall:
    """One re-ask, recorded so a test can see what the model was told."""

    text: str
    current: str
    existing_categories: list[str]
    rejected: list[str]
    guidance: str


class FakeLlmClient:
    """Answers every batch, recording what it was asked."""

    def __init__(
        self,
        category: str = "AI/Transformers",
        *,
        title: str = "Attention Is All You Need",
        authors: list[str] | None = None,
        year: int | None = 2017,
    ) -> None:
        self.category = category
        self.title = title
        self.authors = ["Ashish Vaswani"] if authors is None else authors
        self.year = year
        self.batches: list[list[str]] = []
        self.seen_categories: list[list[str]] = []
        self.pages_read: list[int] = []
        # What `suggest_category` offers, in order, skipping anything rejected.
        self.categories = [
            "Cognitive Science/Computational Modelling",
            "Neuroscience/Learning",
            "Psychology/Decision Making",
        ]
        self.suggestions: list[SuggestionCall] = []

    async def extract_keywords(
        self, batch: Sequence[PaperText], existing_categories: Sequence[str] = ()
    ) -> list[KeywordPair]:
        self.batches.append([paper.file_id for paper in batch])
        self.seen_categories.append(list(existing_categories))
        return [
            KeywordPair(
                file_id=paper.file_id,
                keywords=["alpha", "beta"],
                preliminary_category=self.category,
                title=self.title,
                authors=list(self.authors),
                year=self.year,
            )
            for paper in batch
        ]


    async def describe_pages(self, images: Sequence[PageImage]) -> str:
        self.pages_read.append(len(images))
        return "A Scanned Report\nJane Doe\n2024\nA report about scanned things."

    async def suggest_category(
        self,
        paper: PaperText,
        *,
        current: str = "",
        existing_categories: Sequence[str] = (),
        rejected: Sequence[str] = (),
        guidance: str = "",
    ) -> CategorySuggestion:
        """Answer a re-ask, never repeating one that was turned down.

        Standing in for the real thing's most load-bearing behaviour: a client
        that returned the same category every time would make "give me another"
        a loop with no way out.
        """
        self.suggestions.append(
            SuggestionCall(
                text=paper.text,
                current=current,
                existing_categories=list(existing_categories),
                rejected=list(rejected),
                guidance=guidance,
            )
        )
        offered = [c for c in self.categories if c not in set(rejected)]
        category = offered[0] if offered else f"Fallback/Round {len(rejected)}"
        return CategorySuggestion(
            category=category, keywords=[f"keyword-{len(rejected)}", "shared"]
        )


class FailingLlmClient:
    """Fails every batch, to exercise the error path."""

    def __init__(self, message: str = "rate limited") -> None:
        self.message = message
        self.calls = 0

    async def extract_keywords(
        self, batch: Sequence[PaperText], existing_categories: Sequence[str] = ()
    ) -> list[KeywordPair]:
        self.calls += 1
        raise LlmError(self.message)

    async def describe_pages(self, images: Sequence[PageImage]) -> str:
        self.calls += 1
        raise LlmError(self.message)

    async def suggest_category(self, paper: PaperText, **_kwargs) -> CategorySuggestion:
        self.calls += 1
        raise LlmError(self.message)


def write_pdf(path: Path, text: str) -> Path:
    """Write a single-page PDF carrying `text` as a real text layer."""
    # A minimal hand-built PDF is the only way to get an extractable text layer
    # without pulling in a rendering dependency just for the tests.
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n"
        "%%EOF\n"
    ).encode()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


def write_scanned_pdf(path: Path) -> Path:
    """A valid PDF with no text layer, standing in for a scan."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


@pytest.fixture(autouse=True)
def _isolated_watch_locks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep watcher claims and the registry out of the real user directories."""
    monkeypatch.setenv("SORTYOURPAPERYA_STATE_DIR", str(tmp_path / "state"))
    # And the registry, so a real one on this machine cannot steer a test.
    monkeypatch.setenv("SORTYOURPAPERYA_CONFIG_DIR", str(tmp_path / "config"))


class _NoKeychain:
    """A keychain that is not there, which is what a test machine should look like."""

    _WHY = "the keychain is not available to the test suite"

    @staticmethod
    def get_password(service: str, username: str) -> None:
        raise RuntimeError(_NoKeychain._WHY)

    @staticmethod
    def set_password(service: str, username: str, value: str) -> None:
        raise RuntimeError(_NoKeychain._WHY)

    @staticmethod
    def delete_password(service: str, username: str) -> None:
        raise RuntimeError(_NoKeychain._WHY)


@pytest.fixture(autouse=True)
def _no_real_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's own keychain out of the suite, for the same reason.

    `resolve_api_key` consults the keychain before the environment, so without
    this every test that needs a key reaches into the real one -- and a desktop
    keyring asks the person sitting there to confirm each new process. `pytest`
    is a new process every run, so that is a dialog per run, forever.

    It also stops a developer who happens to be logged in from getting a
    different result than CI, which is the kind of difference nobody debugs
    until it has wasted an afternoon.

    Tests that exercise keychain behaviour install their own fake over this one.
    """
    monkeypatch.setitem(sys.modules, "keyring", _NoKeychain)
    # The spender flag is a module global, so a test that sets it would leave
    # every later test in this process able to spend the stored key -- turning
    # the tests that check the opposite into ones that pass for the wrong
    # reason, depending on order.
    from sortyourpaperya import config as _config

    monkeypatch.setattr(_config, "_MAY_USE_KEYCHAIN", False)
    # With no keychain and no .env, commands that resolve a key would exit 2
    # before reaching what they are being tested for. They all inject a fake
    # model client, so the key only has to exist -- it is never spent, and a
    # value this obviously fake fails loudly if anything ever does send it.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key-for-tests")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    return Settings(
        input_dir=inbox,
        output_dir=inbox / "library",
        keyword_batch_size=2,
    )


@pytest.fixture
def library(settings: Settings):
    from sortyourpaperya.library import Library

    lib = Library(settings.output_dir)
    try:
        yield lib
    finally:
        lib.close()

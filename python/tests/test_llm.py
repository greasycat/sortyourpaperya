from __future__ import annotations

import json
from pathlib import Path

import pytest

from sortyourpaperya.config import MAX_TEXT_CHARS_PER_FILE
from sortyourpaperya.extract import PaperText
from sortyourpaperya.llm import LlmError, build_prompt, parse_response


def _paper(file_id: str, text: str = "some text") -> PaperText:
    return PaperText(
        file_id=file_id, path=Path(f"/tmp/{file_id}.pdf"), text=text, pages_read=1
    )


def test_prompt_truncates_per_file_text_so_a_big_batch_still_fits() -> None:
    batch = [_paper(f"f{index}", "x" * 100_000) for index in range(4)]

    prompt = build_prompt(batch)

    files = json.loads(prompt.split("files:\n", 1)[1])
    assert all(len(entry["text"]) <= MAX_TEXT_CHARS_PER_FILE for entry in files)


def test_parses_pairs_back_into_batch_order() -> None:
    batch = [_paper("a"), _paper("b")]
    content = json.dumps(
        {
            "pairs": [
                {
                    "file_id": "b",
                    "keywords": ["k2"],
                    "preliminary_categories_k_depth": "Systems",
                },
                {
                    "file_id": "a",
                    "keywords": ["k1"],
                    "preliminary_categories_k_depth": "AI",
                },
            ]
        }
    )

    pairs = parse_response(content, batch)

    assert [pair.file_id for pair in pairs] == ["a", "b"]
    assert pairs[0].preliminary_category == "AI"


def test_a_missing_file_is_rejected_rather_than_silently_dropped() -> None:
    batch = [_paper("a"), _paper("b")]
    content = json.dumps(
        {
            "pairs": [
                {
                    "file_id": "a",
                    "keywords": [],
                    "preliminary_categories_k_depth": "AI",
                }
            ]
        }
    )

    with pytest.raises(LlmError, match="missing file_id"):
        parse_response(content, batch)


def test_an_unrequested_file_is_rejected() -> None:
    content = json.dumps(
        {
            "pairs": [
                {
                    "file_id": "zzz",
                    "keywords": [],
                    "preliminary_categories_k_depth": "AI",
                }
            ]
        }
    )

    with pytest.raises(LlmError, match="unexpected file_id"):
        parse_response(content, [_paper("a")])


def test_malformed_json_is_an_llm_error() -> None:
    with pytest.raises(LlmError, match="invalid JSON"):
        parse_response("not json", [_paper("a")])


# ---- re-asking for a category ----------------------------------------------


def _text(body: str = "successor representations, temporal difference learning"):
    from sortyourpaperya.extract import PaperText

    return PaperText(file_id="abc", path=Path("a.pdf"), text=body, pages_read=1)


def test_the_re_ask_tells_the_model_what_was_already_turned_down() -> None:
    """Without this, asking twice returns the same answer twice.

    The inputs are otherwise identical, so "give me another" would never move
    off the first suggestion.
    """
    from sortyourpaperya.llm import build_category_prompt

    prompt = build_category_prompt(
        _text(),
        "Psychology/Research Methods",
        ["Psychology/Research Methods"],
        ["Psychology/Research Methods", "Psychology/Statistics"],
    )

    assert "already offered and rejected" in prompt
    assert "Psychology/Statistics" in prompt
    assert "Do not return any of them" in prompt


def test_a_steer_outranks_the_paths_already_turned_down() -> None:
    """The person has read the document; they may be walking back a refusal."""
    from sortyourpaperya.llm import build_category_prompt

    prompt = build_category_prompt(
        _text(),
        "Psychology/Research Methods",
        [],
        ["Psychology/Statistics"],
        "it is about the maths, not the clinic",
    )

    assert "it is about the maths, not the clinic" in prompt
    assert "where it conflicts with anything above, it wins" in prompt
    assert prompt.index("already offered and rejected") < prompt.index("has read it")


def test_the_re_ask_weighs_existing_paths_rather_than_preferring_them() -> None:
    """The ingest prompt's preference is what misfiles a document.

    A paper on computational cognitive science joins `Psychology/Research
    Methods` because two research-methods documents are already there. Someone
    re-tagging has seen that answer and said no, so the same rule must not
    decide it again.
    """
    from sortyourpaperya.llm import build_category_prompt, build_prompt

    ingest = build_prompt([_text()], ["Psychology/Research Methods"])
    re_ask = build_category_prompt(_text(), "", ["Psychology/Research Methods"])

    assert "Prefer a path from existing_categories" in ingest
    assert "Prefer a path from existing_categories" not in re_ask
    assert "do not force this document into one" in re_ask
    assert "its own subject decides" in re_ask


def test_the_re_ask_does_not_ask_for_title_authors_or_year() -> None:
    """Those name the link and may have been fixed by hand; a re-tag keeps them."""
    from sortyourpaperya.llm import build_category_prompt

    prompt = build_category_prompt(_text())

    for field in ("title", "authors", "year"):
        assert f'"{field}"' not in prompt


def test_a_category_response_is_read_and_trimmed() -> None:
    from sortyourpaperya.llm import parse_category_response

    suggestion = parse_category_response(
        '{"category":"  Cognitive Science/Computation  ","keywords":["a","  ","b"]}'
    )

    assert suggestion.category == "Cognitive Science/Computation"
    assert suggestion.keywords == ["a", "b"], "blank keywords are dropped"


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        '["a list"]',
        '{"keywords":["a"]}',
        '{"category":"","keywords":[]}',
        '{"category":"   ","keywords":[]}',
        '{"category":"AI","keywords":"not a list"}',
    ],
)
def test_an_unusable_category_response_is_an_error(payload: str) -> None:
    from sortyourpaperya.llm import LlmError, parse_category_response

    with pytest.raises(LlmError):
        parse_category_response(payload)

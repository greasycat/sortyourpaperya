"""A bibliography's reading list: what an agent hands over, and what is shown."""

from __future__ import annotations

import json
import tomllib

import pytest

from sortyourpaperya import bib, reading
from sortyourpaperya.bib import BibError, Source


@pytest.fixture
def thesis(tmp_path):
    made = bib.create(tmp_path, "Thesis")
    for key, title in (
        ("vaswani2017attention", "Attention Is All You Need"),
        ("devlin2019bert", "BERT"),
        ("brown2020language", "Language Models are Few-Shot Learners"),
    ):
        made.add(
            Source(
                key=key,
                file_id=f"id-{key}",
                fields={"title": title, "author": ["A. Author"], "year": 2020},
            )
        )
    made.save()
    return made


def _item(cite, rank=None, **extra):
    item = {"cite": cite, "rationale": f"read {cite}", "chain_of_thought": ["1. a", "b"]}
    if rank is not None:
        item["rank"] = rank
    item.update(extra)
    return item


def test_an_agent_list_is_ranked_renumbered_and_unnumbered(thesis) -> None:
    payload = {
        "chain_of_thought": ["- foundations first", "2) then applications"],
        "items": [_item("devlin2019bert", 5), _item("id-vaswani2017attention", 1)],
    }

    made = reading.from_agent(thesis, payload, None)

    assert [(e.rank, e.key) for e in made.entries] == [
        (1, "vaswani2017attention"),
        (2, "devlin2019bert"),
    ]
    # Numbering is added on display, so the agent's own is stripped.
    assert made.chain_of_thought == ["foundations first", "then applications"]
    assert made.entries[0].chain_of_thought == ["a", "b"]
    assert reading.unranked(thesis, made) == ["brown2020language"]


def test_rank_defaults_to_position(thesis) -> None:
    made = reading.from_agent(
        thesis, {"items": [_item("brown2020language"), _item("devlin2019bert")]}, None
    )
    assert [e.key for e in made.entries] == ["brown2020language", "devlin2019bert"]


@pytest.mark.parametrize(
    ("items", "complaint"),
    [
        ([], "non-empty"),
        ([_item("nobody2000")], "does not cite"),
        ([_item("devlin2019bert"), _item("id-devlin2019bert")], "already item"),
        ([_item("devlin2019bert", 1), _item("brown2020language", 1)], "rank 1"),
        ([_item("devlin2019bert", 0)], "whole number"),
        ([_item("devlin2019bert", rationale="  ")], "rationale"),
        ([_item("devlin2019bert", chain_of_thought=[])], "at least one step"),
        ([_item("devlin2019bert", chain_of_thought="one")], "list of strings"),
    ],
)
def test_a_bad_list_is_refused_whole(thesis, items, complaint) -> None:
    with pytest.raises(BibError, match=complaint):
        reading.from_agent(thesis, {"items": items}, None)


def test_a_new_ranking_keeps_what_was_read(thesis) -> None:
    first = reading.from_agent(thesis, {"items": [_item("devlin2019bert")]}, None)
    reading.toggle_read(first, "devlin2019bert")
    reading.save(thesis, first)

    again = reading.from_agent(
        thesis,
        {"items": [_item("brown2020language"), _item("devlin2019bert")]},
        reading.load(thesis),
    )

    assert again.entry("devlin2019bert").read_at_ms is not None
    assert again.entry("brown2020language").read_at_ms is None


def test_the_list_round_trips_and_is_not_a_note(thesis) -> None:
    made = reading.from_agent(
        thesis, {"chain_of_thought": ["why"], "items": [_item("devlin2019bert")]}, None
    )
    path = reading.save(thesis, made)

    assert path.name == "reading.toml"
    assert reading.load(thesis) == made
    assert path not in thesis.notes()
    tomllib.loads(path.read_text())  # what is written parses


def test_the_screen_expands_and_asks_for_actions(thesis) -> None:
    made = reading.from_agent(
        thesis, {"items": [_item("devlin2019bert"), _item("brown2020language")]}, None
    )
    screen = reading.Screen(thesis, made)

    screen.handle("j")
    assert screen.under_cursor().key == "brown2020language"
    screen.handle("\n")
    assert screen.expanded == {"brown2020language"}
    screen.handle("e")
    assert screen.expanded == {"devlin2019bert", "brown2020language"}
    assert screen.handle("o") == "open"
    assert screen.handle("x") == "read"
    screen.handle("j")  # the cursor stops at the end
    assert screen.under_cursor().key == "brown2020language"


def test_text_numbers_the_reasoning(thesis) -> None:
    made = reading.from_agent(thesis, {"items": [_item("devlin2019bert")]}, None)
    text = reading.as_text(thesis, made)

    assert "[ ] 1. devlin2019bert  BERT (Author, 2020)" in text
    assert "      1. a\n      2. b" in text
    assert "not ranked: vaswani2017attention, brown2020language" in text


def test_marking_read_keeps_a_ranking_written_meanwhile(thesis) -> None:
    """A screen open on the old list must not undo the agent's new one."""
    old = reading.from_agent(thesis, {"items": [_item("devlin2019bert")]}, None)
    reading.save(thesis, old)
    reading.save(
        thesis,
        reading.from_agent(
            thesis, {"items": [_item("brown2020language"), _item("devlin2019bert")]}, old
        ),
    )

    marked = reading.mark(thesis, "devlin2019bert")

    assert [e.key for e in marked.entries] == ["brown2020language", "devlin2019bert"]
    assert reading.load(thesis).entry("devlin2019bert").read_at_ms is not None


@pytest.mark.parametrize(
    "body",
    [
        'entry = "x"',
        '[[entry]]\nkey = "a"\nrank = "first"',
        '[[entry]]\nkey = "a"\nchain_of_thought = 3',
    ],
)
def test_a_hand_broken_list_is_a_bib_error(thesis, body) -> None:
    reading.reading_path(thesis).write_text(body)
    with pytest.raises(BibError):
        reading.load(thesis)


def test_a_lone_string_step_is_one_step(thesis) -> None:
    reading.reading_path(thesis).write_text(
        '[[entry]]\nkey = "devlin2019bert"\nchain_of_thought = "one step"'
    )
    assert reading.load(thesis).entries[0].chain_of_thought == ["one step"]

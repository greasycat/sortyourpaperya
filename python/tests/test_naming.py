from __future__ import annotations

import pytest

from sortyourpaperya.naming import (
    MAX_NAME_CHARS,
    disambiguate,
    link_name,
    new_id,
    parse_store_name,
    sanitize_tag,
    split_category,
    store_name,
)


def test_store_name_round_trips_id_and_tags() -> None:
    paper_id = new_id()

    name = store_name(paper_id, ["Machine Learning", "Transformers"])

    assert name == f"{paper_id}__Machine Learning__Transformers.pdf"
    assert parse_store_name(name) == (paper_id, ["Machine Learning", "Transformers"])


def test_a_tag_can_never_contain_the_separator() -> None:
    # Underscores are what would break parsing, so they must not survive.
    assert "_" not in sanitize_tag("deep__learning")
    assert "_" not in sanitize_tag("a_b_c")

    paper_id = new_id()
    name = store_name(paper_id, [sanitize_tag("deep__learning"), "Vision"])

    assert parse_store_name(name) == (paper_id, ["deep-learning", "Vision"])


def test_split_category_turns_a_model_path_into_ordered_tags() -> None:
    assert split_category("Machine Learning/Transformers") == [
        "Machine Learning",
        "Transformers",
    ]
    assert split_category("AI//  /Vision") == ["AI", "Vision"]


def test_tags_are_dropped_to_fit_the_filesystem_limit() -> None:
    paper_id = new_id()
    tags = [f"Category number {index}" for index in range(40)]

    name = store_name(paper_id, tags)

    assert len(name) <= MAX_NAME_CHARS
    recovered_id, recovered = parse_store_name(name)
    assert recovered_id == paper_id
    # A prefix of the tags survives; the database holds the rest.
    assert recovered == tags[: len(recovered)]


def test_a_name_without_a_paper_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_store_name("not-an-id__AI.pdf")


def test_link_name_is_author_year_title() -> None:
    name = link_name(
        fallback="x.pdf",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        title="Attention Is All You Need",
    )

    assert name == "vaswani_2017_attention-is-all-you-need.pdf"


def test_link_name_handles_surname_first_authors_and_accents() -> None:
    name = link_name(
        fallback="x.pdf", authors=["Bengio, Yoshua"], year=2003, title="Neural Modèles"
    )

    assert name == "bengio_2003_neural-modeles.pdf"


@pytest.mark.parametrize(
    "authors,year,title,expected",
    [
        (["Ashish Vaswani"], 2017, "A Title", "vaswani_2017_a-title.pdf"),
        # A missing piece drops out rather than costing the whole name.
        (["Ashish Vaswani"], None, "A Title", "vaswani_a-title.pdf"),
        ([], 2017, "A Title", "2017_a-title.pdf"),
        ([], None, "A Title", "a-title.pdf"),
        (["Ashish Vaswani"], 2017, "", "vaswani_2017.pdf"),
        (["Ashish Vaswani"], None, "", "vaswani.pdf"),
    ],
)
def test_link_name_uses_whichever_pieces_are_known(
    authors, year, title, expected
) -> None:
    assert (
        link_name(fallback="abc.pdf", authors=authors, year=year, title=title)
        == expected
    )


@pytest.mark.parametrize("year", [None, 2017])
def test_link_name_falls_back_without_an_author_or_a_title(year) -> None:
    # A bare year names nothing anyone could recognise, so the store name wins.
    assert (
        link_name(fallback="abc123__AI.pdf", authors=[], year=year, title="")
        == "abc123__AI.pdf"
    )


def test_disambiguate_keeps_the_suffix() -> None:
    assert disambiguate("a_2017_b.pdf", "deadbeef") == "a_2017_b_deadbeef.pdf"


@pytest.mark.parametrize(
    "title,expected",
    [
        # Clipped between words, not inside one: "...and Tomosynthesis" loses
        # the whole word rather than leaving "-tomo" looking like a typo.
        (
            "Comparing the Subsequent Search Miss Effect Between Mammography and Tomosynthesis",
            "doe_2020_comparing-the-subsequent-search-miss-effect-between-mammography-and.pdf",
        ),
        # Already ends on a boundary, so nothing is dropped.
        (
            "Advancing AI Interpretability in Medical Imaging: A Comparative Analysis of "
            "Pixel-Level Interpretability",
            "doe_2020_advancing-ai-interpretability-in-medical-imaging-a-comparative-analysis.pdf",
        ),
        # Short enough to keep whole.
        ("A Short Title", "doe_2020_a-short-title.pdf"),
    ],
)
def test_a_long_title_is_cut_between_words(title, expected) -> None:
    assert (
        link_name(fallback="x.pdf", authors=["Jane Doe"], year=2020, title=title)
        == expected
    )


def test_a_single_over_long_word_has_no_boundary_to_cut_at() -> None:
    # Nothing to cut between, so it is clipped where it stands rather than
    # collapsing to nothing.
    name = link_name(
        fallback="x.pdf", authors=["Jane Doe"], year=2020, title="Supercalifragilistic" * 8
    )

    assert name.startswith("doe_2020_supercalifragilistic")
    assert len(name) <= MAX_NAME_CHARS


def test_a_cut_title_never_ends_in_a_dash() -> None:
    for words in range(1, 30):
        name = link_name(
            fallback="x.pdf",
            authors=["Jane Doe"],
            year=2020,
            title=" ".join(["word"] * words),
        )
        assert not name.removesuffix(".pdf").endswith("-"), name


def test_a_very_long_name_keeps_its_extension() -> None:
    # The cap trims the stem, not the finished name: capping afterwards can cut
    # the suffix off and leave a file with no extension at all.
    name = link_name(fallback="x.pdf", authors=["A" * 260], year=2020, title="T")

    assert name.endswith(".pdf"), name
    assert len(name) <= MAX_NAME_CHARS

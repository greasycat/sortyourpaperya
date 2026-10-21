"""Choosing several documents at once.

Everything asserted here is in the model half of `pick.py`, which needs no
terminal. That is the reason the module is split: the drawing half decides
nothing, so there is nothing in it a test would want to reach.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sortyourpaperya import pick
from sortyourpaperya.db import Paper
from sortyourpaperya.library import Library, LibraryError


def _paper(file_id: str, title: str, *tags: str) -> Paper:
    return Paper(
        file_id=file_id,
        content_hash=f"h{file_id}",
        store_name=f"{file_id}__x",
        document_name=f"{file_id}.pdf",
        title=title,
        tags=list(tags),
    )


@pytest.fixture
def papers() -> list[Paper]:
    return [
        _paper("a", "Guided Search", "Psychology", "Cognitive"),
        _paper("b", "Attentional Guidance", "Psychology", "Cognitive"),
        _paper("c", "Research Methods in X", "Psychology", "Methods"),
        _paper("d", "Radiology Errors", "Medicine"),
    ]


@pytest.fixture
def roots(papers: list[Paper]) -> list[pick.Node]:
    return pick.build_tree(papers)


# ---- the tree --------------------------------------------------------------


def test_the_tree_follows_the_librarys_own_tags(roots) -> None:
    assert [n.label for n in roots] == ["Medicine", "Psychology"]
    psychology = roots[1]
    assert [n.label for n in psychology.children] == ["Cognitive", "Methods"]


def test_a_category_knows_everything_beneath_it(roots) -> None:
    assert len(roots[1].papers()) == 3
    assert len(roots[0].papers()) == 1


def test_an_untagged_document_is_still_shown() -> None:
    """A document you cannot see is one you cannot delete.

    Untagged documents are exactly the ones worth finding, so they get a branch
    rather than being dropped out of the tree.
    """
    roots = pick.build_tree([_paper("x", "No tags at all")])
    assert [n.label for n in roots] == ["(uncategorised)"]
    assert len(roots[0].papers()) == 1


def test_a_document_with_no_title_is_named_by_its_file(roots) -> None:
    # Grant sections and chapters have no title; they still have to be
    # recognisable, and the original filename is what their owner knows.
    paper = Paper(
        file_id="z",
        content_hash="h",
        store_name="z__x",
        original_name="Research Strategy.pdf",
        tags=["Medicine"],
    )
    assert pick.label(paper) == "Research Strategy.pdf"


def test_it_opens_collapsed_to_the_top_level(roots) -> None:
    screen = pick.Screen(roots, pick.Selection())
    assert [n.label for n in screen.rows()] == ["Medicine", "Psychology"]


# ---- selection -------------------------------------------------------------


def test_selecting_a_category_takes_everything_under_it(roots) -> None:
    selection = pick.Selection()
    selection.toggle(roots[1])  # Psychology
    assert len(selection) == 3
    assert selection.state(roots[1]) == "all"


def test_a_partly_selected_category_says_so(roots) -> None:
    selection = pick.Selection()
    selection.toggle(roots[1].children[0])  # Psychology/Cognitive, 2 of 3
    assert selection.state(roots[1]) == "some"
    assert selection.state(roots[1].children[0]) == "all"
    assert selection.state(roots[1].children[1]) == "none"


def test_space_on_a_partly_selected_category_completes_it(roots) -> None:
    # Which is what a half-filled box invites; emptying it would be the other
    # reading, and the more surprising one.
    selection = pick.Selection()
    selection.toggle(roots[1].children[0])
    selection.toggle(roots[1])
    assert selection.state(roots[1]) == "all"


def test_deselecting_a_category_clears_only_its_own(roots) -> None:
    selection = pick.Selection()
    selection.set_all(roots, True)
    selection.toggle(roots[1])  # all of Psychology, so this clears it
    assert len(selection) == 1
    assert selection.state(roots[0]) == "all"


def test_the_selection_survives_a_rebuild_under_a_filter(papers) -> None:
    """Ids, not flags on nodes -- so re-filtering does not lose what was chosen."""
    selection = pick.Selection()
    selection.toggle(pick.build_tree(papers)[1])  # Psychology
    narrowed = pick.build_tree([p for p in papers if p.file_id in {"a", "d"}])
    assert [p.file_id for p in selection.chosen(narrowed)] == ["a"]


def test_chosen_comes_back_in_the_order_it_is_shown(roots) -> None:
    """The confirmation names documents in the order the eye just ran down them.

    Asserted against what the screen would draw rather than a list written by
    hand -- the hand-written one encoded my assumption about the sort, not the
    tree's actual order, and agreed with nothing.
    """
    for node in roots:  # everything open, so every document is on screen
        node.expanded = True
        for child in node.children:
            child.expanded = True
    selection = pick.Selection()
    selection.set_all(roots, True)

    shown = [n.paper.file_id for n in pick.visible(roots) if n.is_document]
    assert [p.file_id for p in selection.chosen(roots)] == shown


# ---- keys ------------------------------------------------------------------


def test_moving_expanding_and_selecting(roots) -> None:
    screen = pick.Screen(roots, pick.Selection())
    screen.handle("j")                       # onto Psychology
    screen.handle("l")                       # open it
    assert [n.label for n in screen.rows()] == [
        "Medicine",
        "Psychology",
        "Cognitive",
        "Methods",
    ]
    screen.handle("h")                       # close it again
    assert len(screen.rows()) == 2
    screen.handle(" ")
    assert len(screen.selection) == 3


def test_the_cursor_stops_at_the_ends(roots) -> None:
    screen = pick.Screen(roots, pick.Selection())
    for _ in range(10):
        screen.handle("k")
    assert screen.cursor == 0
    for _ in range(10):
        screen.handle("j")
    assert screen.cursor == len(screen.rows()) - 1


def test_a_and_A_select_everything_and_nothing(roots) -> None:
    screen = pick.Screen(roots, pick.Selection())
    screen.handle("a")
    assert len(screen.selection) == 4
    screen.handle("A")
    assert len(screen.selection) == 0


def test_d_asks_for_the_delete_operation(roots) -> None:
    screen = pick.Screen(roots, pick.Selection())
    assert screen.handle("d") is pick.DELETE


def test_an_unbound_key_does_nothing(roots) -> None:
    screen = pick.Screen(roots, pick.Selection())
    before = [screen.cursor, len(screen.selection)]
    assert screen.handle("Z") is None
    assert [screen.cursor, len(screen.selection)] == before


def test_every_operation_is_reachable_and_advertised() -> None:
    """Adding an operation must not need a change to the screen.

    So the binding and the footer both come from the registry, and this fails if
    a future operation is added without either.
    """
    for operation in pick.OPERATIONS:
        assert pick.operation_for(operation.key) is operation
        assert f"{operation.key} {operation.label}" in pick.footer()


def test_no_operation_shadows_a_navigation_key() -> None:
    # A `j` operation would make the tree unnavigable, and the clash is silent.
    reserved = (
        pick.QUIT_KEYS
        | pick.DOWN_KEYS
        | pick.UP_KEYS
        | pick.OPEN_KEYS
        | pick.CLOSE_KEYS
        | {" ", "a", "A", pick.FZF_KEY}
    )
    for operation in pick.OPERATIONS:
        assert operation.key not in reserved


# ---- applying --------------------------------------------------------------


def test_delete_removes_every_selected_document(tmp_path: Path) -> None:
    library = Library(tmp_path / "lib")
    kept, gone = _fill(library)
    outcome = pick.apply(pick.DELETE, library, gone)
    assert len(outcome.done) == 2 and not outcome.failed
    assert {p.file_id for p in library.db.all_papers()} == {kept.file_id}


def test_one_refusal_does_not_strand_the_rest(tmp_path: Path) -> None:
    """The person asked for all of them; stopping halfway leaves a state nobody
    chose, and says nothing about which half it is."""
    library = Library(tmp_path / "lib")
    _, chosen = _fill(library)
    calls: list[str] = []

    def refuse_the_first(lib, paper, _answer):
        calls.append(paper.file_id)
        if len(calls) == 1:
            raise LibraryError("nope")
        lib.remove(paper.file_id)

    operation = pick.Operation(
        key="x", label="test", run=refuse_the_first, confirm=lambda ps: "?"
    )
    outcome = pick.apply(operation, library, chosen)
    assert len(calls) == 2
    assert len(outcome.done) == 1 and len(outcome.failed) == 1
    assert "1 done, 1 failed" in outcome.describe(operation)


def test_the_confirmation_names_what_will_go(roots) -> None:
    papers = pick.Selection()
    papers.set_all(roots, True)
    question = pick.DELETE.confirm(papers.chosen(roots))
    assert "Delete 4 document(s)?" in question
    assert "Guided Search" in question


def test_a_long_confirmation_counts_the_rest_instead_of_listing_them() -> None:
    # A question you have to scroll is one people stop reading.
    many = [_paper(str(i), f"Paper {i}", "X") for i in range(20)]
    question = pick.DELETE.confirm(many)
    assert question.count("\n  ") == pick.CONFIRM_NAMES + 1
    assert f"and {20 - pick.CONFIRM_NAMES} more" in question


def _fill(library: Library) -> tuple[Paper, list[Paper]]:
    """Three documents in the library; returns the one to keep and two to remove."""
    papers = [
        _paper("keep", "Kept", "Medicine"),
        _paper("one", "First", "Psychology"),
        _paper("two", "Second", "Psychology"),
    ]
    for paper in papers:
        library.db.upsert(paper)
        (library.store_dir / paper.store_name).mkdir(parents=True, exist_ok=True)
    return papers[0], papers[1:]


def test_delete_takes_the_store_folder_with_the_row(tmp_path: Path) -> None:
    # A row without its folder, or a folder without its row, is the state `fsck`
    # exists to find. Deleting must not create one.
    library = Library(tmp_path / "lib")
    kept, gone = _fill(library)
    pick.apply(pick.DELETE, library, gone)

    assert (library.store_dir / kept.store_name).is_dir()
    for paper in gone:
        assert not (library.store_dir / paper.store_name).exists()


def test_declining_the_confirmation_runs_nothing(roots) -> None:
    """Nothing destructive happens without a yes."""
    ran: list = []
    message = pick.act(
        pick.DELETE,
        [_paper("a", "One", "X")],
        ask=lambda question: False,
        do=lambda op, papers, answer, category: ran.append(papers) or "ran",
    )
    assert ran == []
    assert message == "delete cancelled"


def test_confirming_runs_it(roots) -> None:
    ran: list = []
    message = pick.act(
        pick.DELETE,
        [_paper("a", "One", "X")],
        ask=lambda question: True,
        do=lambda op, papers, answer, category: ran.append(papers) or "did it",
    )
    assert len(ran) == 1
    assert message == "did it"


def test_an_empty_selection_is_not_worth_asking_about(roots) -> None:
    asked: list = []
    message = pick.act(
        pick.DELETE, [], ask=lambda q: asked.append(q) or True, do=lambda *a: "ran"
    )
    assert asked == []
    assert "nothing selected" in message


def test_the_question_asked_is_the_operations_own(roots) -> None:
    asked: list[str] = []
    papers = [_paper("a", "Guided Search", "X")]
    pick.act(pick.DELETE, papers, ask=lambda q: asked.append(q) or False, do=lambda *a: "")
    assert asked == [pick.DELETE.confirm(papers)]
    assert "Guided Search" in asked[0]


# ---- opening ---------------------------------------------------------------


class _Started:
    """A process that took the document and is still running, which is success."""

    def __init__(self, command, piped: list | None = None) -> None:
        self.command = command
        self.stdin = self
        self._piped = piped

    def write(self, text: str) -> None:
        if self._piped is not None:
            self._piped.append((self.command, text))

    def close(self) -> None:
        pass

    def wait(self, timeout: float) -> int:
        raise pick.subprocess.TimeoutExpired(self.command, timeout)


def test_open_hands_the_document_to_the_desktop(tmp_path: Path, monkeypatch) -> None:
    library = Library(tmp_path / "lib")
    kept, chosen = _fill(library)
    for paper in chosen:  # a folder alone is not a document
        (library.store_dir / paper.store_name / paper.document_name).write_text("pdf")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        pick.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or _Started(cmd)
    )

    outcome = pick.apply(pick.OPEN, library, chosen)
    assert len(outcome.done) == 2 and not outcome.failed
    assert [c[0] for c in calls] == [pick.viewer(), pick.viewer()]
    assert calls[0][1].endswith(chosen[0].document_name)


def test_open_says_which_document_is_missing(tmp_path: Path, monkeypatch) -> None:
    """A row whose file has gone is what `fsck` looks for; opening must name it
    rather than hand a path that is not there to the desktop."""
    library = Library(tmp_path / "lib")
    _, chosen = _fill(library)  # folders, no files in them
    monkeypatch.setattr(pick.subprocess, "Popen", lambda cmd, **kw: pytest.fail("ran"))

    outcome = pick.apply(pick.OPEN, library, chosen)
    assert not outcome.done and len(outcome.failed) == 2
    assert "no file at" in outcome.failed[0][1]


def test_opening_asks_nothing(tmp_path: Path) -> None:
    """A keystroke that undoes itself earns no question; one that does teaches
    people to answer without reading."""
    asked: list[str] = []
    message = pick.act(
        pick.OPEN,
        [_paper("a", "One", "X")],
        ask=lambda q: asked.append(q) or False,
        do=lambda op, papers, answer, category: "opened",
    )
    assert asked == []
    assert message == "opened"


def test_open_with_asks_for_a_command_and_pipes_the_path(tmp_path: Path, monkeypatch) -> None:
    library = Library(tmp_path / "lib")
    _, chosen = _fill(library)
    for paper in chosen:
        (library.store_dir / paper.store_name / paper.document_name).write_text("pdf")
    piped: list[tuple[str, str]] = []

    monkeypatch.setattr(
        pick.subprocess, "Popen", lambda cmd, **kw: _Started(cmd, piped)
    )
    outcome = pick.apply(pick.OPEN_WITH, library, chosen, "xargs -r zathura")

    assert len(outcome.done) == 2 and not outcome.failed
    assert [c for c, _ in piped] == ["xargs -r zathura"] * 2
    assert piped[0][1].strip().endswith(chosen[0].document_name)


def test_open_with_runs_nothing_until_a_command_is_typed() -> None:
    """Escape at the prompt, or an empty line, is the same decision: no."""
    for typed in (None, "", "   "):
        ran: list = []
        message = pick.act(
            pick.OPEN_WITH,
            [_paper("a", "One", "X")],
            ask=lambda q: True,
            do=lambda op, papers, answer, category: ran.append(answer) or "ran",
            request=lambda prompt: typed,
        )
        assert ran == []
        assert message == "open with cancelled"


def test_what_is_typed_reaches_the_operation() -> None:
    got: list[str] = []
    pick.act(
        pick.OPEN_WITH,
        [_paper("a", "One", "X")],
        ask=lambda q: True,
        do=lambda op, papers, answer, category: got.append(answer) or "ran",
        request=lambda prompt: "  wc -l  ",
    )
    assert got == ["wc -l"]  # trimmed: a stray space is not part of the command


def test_the_viewer_is_the_platforms_own(monkeypatch) -> None:
    monkeypatch.setattr(pick.sys, "platform", "darwin")
    assert pick.viewer() == "open"
    monkeypatch.setattr(pick.sys, "platform", "linux")
    assert pick.viewer() == "xdg-open"


def test_opening_takes_no_write_lock() -> None:
    """Reading routes past the watcher: a document should open while a filing
    pass holds the write connection, not queue behind it."""
    assert pick.OPEN.writes is False
    assert pick.DELETE.writes is True


def test_a_mistyped_command_is_reported_rather_than_counted_as_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shell starts happily whatever you type; the failure arrives after. Long
    enough to catch that, short enough not to wait on a reader that stays open.

    The grace is widened here: the shell exiting inside 50 ms is the common case,
    not a promise, and a busy machine is not a broken command. Either way the
    shell may exit before the path is written, and its status, not the broken
    pipe, is what must be reported.
    """
    monkeypatch.setattr(pick, "SPAWN_GRACE", 5.0)
    library = Library(tmp_path / "lib")
    _, chosen = _fill(library)
    paper = chosen[0]
    (library.store_dir / paper.store_name / paper.document_name).write_text("pdf")

    outcome = pick.apply(pick.OPEN_WITH, library, [paper], "no-such-command-xyz")
    assert not outcome.done and len(outcome.failed) == 1
    assert "status 127" in outcome.failed[0][1]


# ---- opening a category ----------------------------------------------------


def test_a_category_knows_its_folder_in_the_tree(roots) -> None:
    assert roots[1].path == ("Psychology",)
    assert roots[1].children[0].path == ("Psychology", "Cognitive")


def test_the_uncategorised_branch_is_a_label_not_a_folder() -> None:
    """Untagged documents link at the top of the tree, so that is where opening
    their branch goes -- not to a `(uncategorised)` folder that never existed."""
    roots = pick.build_tree([_paper("x", "No tags at all")])
    assert roots[0].label == "(uncategorised)" and roots[0].path == ()


def test_o_on_a_category_opens_the_branch_rather_than_the_folder(roots) -> None:
    """Opening a branch on screen is seeing what is in it; the folder it stands
    for is `O`. So `o` here navigates and runs nothing."""
    screen = pick.Screen(roots, pick.Selection())
    screen.cursor = 1  # Psychology, collapsed
    assert screen.handle("o") is None
    assert [n.label for n in screen.rows()][2:] == ["Cognitive", "Methods"]


def test_o_on_a_document_still_asks_for_the_open_operation(roots) -> None:
    screen = pick.Screen(roots, pick.Selection())
    screen.handle("o")               # onto the documents under Medicine
    screen.handle("j")
    assert screen.rows()[screen.cursor].is_document
    assert screen.handle("o") is pick.OPEN


def test_O_on_a_category_opens_the_folder(tmp_path: Path, monkeypatch) -> None:
    library = Library(tmp_path / "lib")
    (library.tree_dir / "Psychology" / "Cognitive").mkdir(parents=True)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        pick.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or _Started(cmd)
    )
    branch = pick.build_tree([_paper("a", "Guided Search", "Psychology", "Cognitive")])[0]

    outcome = pick.apply(pick.OPEN_WITH, library, [], category=branch)
    assert outcome.done == ["Psychology"] and not outcome.failed
    assert calls == [[pick.viewer(), str(library.tree_dir / "Psychology")]]


def test_opening_a_category_needs_no_selection_and_asks_for_nothing(roots) -> None:
    """Not even the command `O` asks a document for: a folder is not piped into
    anything, and the desktop already knows what to do with one."""
    asked: list[str] = []
    aimed: list = []
    message = pick.act(
        pick.OPEN_WITH,
        [],  # nothing selected, which for a folder is not a problem
        ask=lambda q: asked.append(q) or False,
        do=lambda op, papers, answer, category: aimed.append(category) or "opened",
        request=lambda prompt: pytest.fail("asked for a command"),
        category=roots[1],
    )
    assert asked == [] and message == "opened"
    assert aimed == [roots[1]]


def test_an_operation_with_no_folder_sense_still_acts_on_the_selection(roots) -> None:
    """`d` on a category deletes what is selected, as it always did -- only an
    operation that knows what a folder means is aimed at one."""
    aimed: list = []
    pick.act(
        pick.DELETE,
        [_paper("a", "One", "X")],
        ask=lambda q: True,
        do=lambda op, papers, answer, category: aimed.append(category) or "deleted",
        category=roots[1],
    )
    assert aimed == [None]


def test_a_category_with_no_folder_says_so(tmp_path: Path, monkeypatch) -> None:
    # The tree is a view rebuilt from the database and may not be on disk at all.
    library = Library(tmp_path / "lib")
    monkeypatch.setattr(pick.subprocess, "Popen", lambda cmd, **kw: pytest.fail("ran"))
    branch = pick.build_tree([_paper("a", "One", "Psychology")])[0]

    outcome = pick.apply(pick.OPEN_WITH, library, [], category=branch)
    assert not outcome.done and "no folder at" in outcome.failed[0][1]


# ---- retagging -------------------------------------------------------------


def test_retag_moves_every_selected_document_to_the_typed_category(
    tmp_path: Path,
) -> None:
    """One category, asked for once, applied down the selection in turn."""
    library = Library(tmp_path / "lib")
    kept, chosen = _fill(library)

    outcome = pick.apply(pick.RETAG, library, chosen, "Medicine/Radiology")
    assert len(outcome.done) == 2 and not outcome.failed
    moved = {p.file_id: p.tags for p in library.db.all_papers()}
    assert moved["one"] == ["Medicine", "Radiology"]
    assert moved["two"] == ["Medicine", "Radiology"]
    assert moved[kept.file_id] == ["Medicine"]  # untouched: it was not selected


def test_retag_sanitizes_what_was_typed(tmp_path: Path) -> None:
    # The same treatment a model's answer gets, so a shelf typed by hand and one
    # suggested cannot become two spellings of the same place.
    library = Library(tmp_path / "lib")
    _, chosen = _fill(library)
    pick.apply(pick.RETAG, library, chosen[:1], "  Psychology / Vision  ")
    assert library.db.get("one").tags == ["Psychology", "Vision"]


def test_a_line_with_no_usable_tags_moves_nothing(tmp_path: Path) -> None:
    library = Library(tmp_path / "lib")
    _, chosen = _fill(library)
    outcome = pick.apply(pick.RETAG, library, chosen, "///")
    assert not outcome.done and len(outcome.failed) == 2
    assert "no usable tags" in outcome.failed[0][1]
    assert library.db.get("one").tags == ["Psychology"]


def test_retag_asks_for_a_category_and_takes_the_write_seam() -> None:
    assert pick.RETAG.prompt is not None
    assert pick.RETAG.confirm is None  # typing the category is the decision
    assert pick.RETAG.writes is True


# ---- fzf -------------------------------------------------------------------


def test_fzf_returns_what_was_selected(monkeypatch, tmp_path: Path) -> None:
    """The ids on the chosen lines, back as documents, in the order fzf gave."""
    library = Library(tmp_path / "lib")
    _fill(library)
    papers = library.db.all_papers()
    shown = {}

    def fake_run(argv, input, stdout, text):
        shown["argv"], shown["lines"] = argv, input
        chosen = input.splitlines()[:2]
        return SimpleNamespace(returncode=0, stdout="\n".join(chosen) + "\n")

    monkeypatch.setattr(pick.subprocess, "run", fake_run)
    got = pick._fzf_select(papers)

    assert "--multi" in shown["argv"]
    assert [p.file_id for p in got] == [
        line.split("\t")[0] for line in shown["lines"].splitlines()[:2]
    ]
    # The id leads the line but is not what fzf shows or searches.
    assert shown["argv"][shown["argv"].index("--with-nth") + 1] == "2.."


def test_backing_out_of_fzf_selects_nothing(monkeypatch, tmp_path: Path) -> None:
    library = Library(tmp_path / "lib")
    _fill(library)
    monkeypatch.setattr(
        pick.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=130, stdout=""),
    )
    assert pick._fzf_select(library.db.all_papers()) == []


def test_fzf_round_runs_the_operation_on_what_was_chosen(
    monkeypatch, tmp_path: Path
) -> None:
    library = Library(tmp_path / "lib")
    _fill(library)
    papers = library.db.all_papers()
    roots = pick.build_tree(papers)
    done = {}

    monkeypatch.setattr(pick, "_fzf_select", lambda _: papers[:1])
    monkeypatch.setattr("builtins.input", lambda prompt="": pick.OPEN.key)

    def on_apply(operation, chosen, answer, category):
        done["call"] = (operation, list(chosen), category)
        return "opened"

    pick.fzf_round(lambda: roots, on_apply)
    assert done["call"] == (pick.OPEN, papers[:1], None)


def test_an_unknown_key_at_the_fzf_prompt_does_nothing(
    monkeypatch, tmp_path: Path
) -> None:
    library = Library(tmp_path / "lib")
    _fill(library)
    papers = library.db.all_papers()
    monkeypatch.setattr(pick, "_fzf_select", lambda _: papers[:1])
    monkeypatch.setattr("builtins.input", lambda prompt="": "x")
    pick.fzf_round(
        lambda: pick.build_tree(papers),
        lambda *a: pytest.fail("an unbound key ran an operation"),
    )


def test_the_author_and_the_year_are_searchable(monkeypatch) -> None:
    """`knuth 84` is how a document is remembered when its title is not."""
    paper = Paper(
        file_id="abc123",
        content_hash="h",
        store_name="abc123__Computing",
        title="The TeXbook",
        year=1984,
        authors=["Donald Knuth"],
        tags=["Computing"],
    )
    shown = {}

    def fake_run(argv, input, stdout, text):
        shown["argv"], shown["line"] = argv, input
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(pick.subprocess, "run", fake_run)
    pick._fzf_select([paper])

    hidden, *searched = shown["line"].split("\t")
    assert hidden == "abc123"  # read back from, never shown
    assert "Donald Knuth" in searched and "1984" in searched


@pytest.mark.parametrize(("status", "said"), [(127, "status 127"), (0, "did not take")])
def test_a_command_gone_before_the_path_reports_its_status(
    monkeypatch: pytest.MonkeyPatch, status: int, said: str
) -> None:
    """The broken pipe is the symptom; a failing status is the cause. Forced
    here, since against a real shell which is noticed first is a race."""

    class Gone:
        def write(self, _text):
            raise BrokenPipeError(32, "Broken pipe")

        def close(self):
            pass

    class Process:
        stdin = Gone()

        def wait(self, timeout):
            return status

    monkeypatch.setattr(pick.subprocess, "Popen", lambda cmd, **kw: Process())
    with pytest.raises(LibraryError, match=said):
        pick._spawn("whatever", "/some/path")


# ---- notes -----------------------------------------------------------------


def test_the_notes_menu_lists_notes_then_offers_a_new_one(tmp_path: Path) -> None:
    library = Library(tmp_path / "lib")
    paper, _ = _fill(library)
    folder = library.store_dir / paper.store_name
    (folder / "notes.md").write_text("# Kept\n")
    (folder / "reading-log.md").write_text("")

    menu = pick.notes_menu(library, paper)

    assert menu.rows == ["notes.md", "reading-log.md", pick.NEW_NOTE]
    assert menu.handle("\n") == ("open", folder / "notes.md")
    menu.handle("j")
    assert menu.handle("l") == ("open", folder / "reading-log.md")
    menu.handle("j")
    menu.handle("j")  # stops on the last row
    assert menu.handle("\r") == ("new", None)
    assert menu.handle("n") == ("new", None)
    for key in ("h", "q", "\x1b", pick.NOTES_KEY):
        assert menu.handle(key) == ("back", None)


def test_a_document_with_no_notes_offers_only_a_new_one(tmp_path: Path) -> None:
    library = Library(tmp_path / "lib")
    paper, _ = _fill(library)

    menu = pick.notes_menu(library, paper)

    assert menu.rows == [pick.NEW_NOTE]
    assert menu.handle("\n") == ("new", None)


def test_the_notes_menu_refuses_a_document_missing_from_the_store(tmp_path: Path) -> None:
    """Starting a note there would make a folder the store does not account for."""
    library = Library(tmp_path / "lib")
    paper, _ = _fill(library)
    (library.store_dir / paper.store_name).rmdir()

    with pytest.raises(LibraryError, match="missing from the store"):
        pick.notes_menu(library, paper)


def test_a_new_note_is_named_like_sypy_note_and_never_overwritten(tmp_path: Path) -> None:
    library = Library(tmp_path / "lib")
    paper, _ = _fill(library)
    folder = library.store_dir / paper.store_name

    assert pick.new_note(library, paper, "") == folder / "notes.md"
    made = pick.new_note(library, paper, "ch3")
    assert made == folder / "ch3.md"
    assert made.read_text().startswith("# Kept")

    made.write_text("mine")
    pick.new_note(library, paper, "ch3")
    assert made.read_text() == "mine"

    with pytest.raises(LibraryError):
        pick.new_note(library, paper, "../escape")


def test_the_editor_comes_from_visual_then_editor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    assert pick.editor() is None
    monkeypatch.setenv("EDITOR", "code -w")
    assert pick.editor() == ["code", "-w"]
    monkeypatch.setenv("VISUAL", "nvim")
    assert pick.editor() == ["nvim"]


def test_the_footer_names_the_notes_key() -> None:
    assert f"{pick.NOTES_KEY} notes" in pick.footer()


def test_a_note_is_edited_with_the_screen_set_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The editor owns the terminal while it runs, so it runs inside `suspend`."""
    note = tmp_path / "notes.md"
    calls: list[object] = []
    monkeypatch.setenv("VISUAL", "myeditor -w")
    monkeypatch.setattr(pick.subprocess, "call", lambda cmd: calls.append(cmd) or 0)

    def suspend(job):
        calls.append("away")
        job()
        calls.append("back")

    assert pick._open_note(note, suspend) == "edited notes.md"
    assert calls == ["away", ["myeditor", "-w", str(note)], "back"]


def test_with_no_editor_a_note_goes_to_the_desktop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    note = tmp_path / "notes.md"
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    spawned: list[list[str]] = []
    monkeypatch.setattr(pick, "_spawn", lambda cmd: spawned.append(cmd))

    pick._open_note(note, lambda job: pytest.fail("no screen to set aside"))

    assert spawned == [[pick.viewer(), str(note)]]


def _reading_from(library: Library):
    return lambda job: job(library)


def test_fzf_notes_lists_every_chosen_documents_notes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fzf has no cursor, so every chosen document's notes are listed, named."""
    library = Library(tmp_path / "lib")
    _, chosen = _fill(library)
    first = library.store_dir / chosen[0].store_name / "notes.md"
    first.write_text("# First\n")
    shown = {}

    def fake_run(argv, input, stdout, text):
        shown["argv"], shown["lines"] = argv, input.splitlines()
        return SimpleNamespace(returncode=0, stdout=shown["lines"][0] + "\n")

    opened = []
    monkeypatch.setattr(pick.subprocess, "run", fake_run)
    monkeypatch.setattr(pick, "_open_note", lambda path, suspend: opened.append(path) or "ok")

    said = pick.fzf_notes(chosen, _reading_from(library))

    assert [line.split("\t", 1)[1] for line in shown["lines"]] == [
        "notes.md    First",
        f"{pick.NEW_NOTE}  First",
        f"{pick.NEW_NOTE}  Second",
    ]
    assert "--multi" not in shown["argv"]  # one note is opened, not a batch
    assert opened == [first] and said == "ok"


def test_fzf_notes_leads_with_the_note_so_a_long_title_cannot_hide_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A title wider than the terminal pushed the note names off its edge,
    and four notes read as four identical lines."""
    library = Library(tmp_path / "lib")
    _, chosen = _fill(library)
    paper = chosen[0]
    paper.title = "What Distinguishes Optimal Visual Searchers? " * 3
    folder = library.store_dir / paper.store_name
    for name in ("discussion.md", "methods.md"):
        (folder / name).write_text("")
    shown = {}

    def fake_run(argv, input, stdout, text):
        shown["lines"] = input.splitlines()
        return SimpleNamespace(returncode=130, stdout="")

    monkeypatch.setattr(pick.subprocess, "run", fake_run)
    pick.fzf_notes([paper], _reading_from(library))

    shown_from = [line.split("\t", 1)[1] for line in shown["lines"]]
    assert [line[:13] for line in shown_from] == [
        "discussion.md", "methods.md   ", pick.NEW_NOTE + "   "
    ]


def test_fzf_notes_starts_a_note_on_the_document_whose_line_was_chosen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library = Library(tmp_path / "lib")
    _, chosen = _fill(library)
    monkeypatch.setattr(
        pick.subprocess,
        "run",
        lambda argv, input, stdout, text: SimpleNamespace(
            returncode=0, stdout=input.splitlines()[1] + "\n"
        ),
    )
    opened = []
    monkeypatch.setattr(pick, "_open_note", lambda path, suspend: opened.append(path) or "ok")

    pick.fzf_notes(chosen, _reading_from(library), request=lambda prompt: "ch3")

    assert opened == [library.store_dir / chosen[1].store_name / "ch3.md"]
    assert opened[0].is_file()


def test_fzf_notes_names_a_missing_document_and_lists_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library = Library(tmp_path / "lib")
    _, chosen = _fill(library)
    (library.store_dir / chosen[0].store_name).rmdir()
    seen = {}

    def fake_run(argv, input, stdout, text):
        seen["lines"] = input.splitlines()
        return SimpleNamespace(returncode=130, stdout="")

    monkeypatch.setattr(pick.subprocess, "run", fake_run)

    said = pick.fzf_notes(chosen, _reading_from(library))

    assert len(seen["lines"]) == 1 and "Second" in seen["lines"][0]
    assert "First" in said and "missing from the store" in said


def test_p_at_the_fzf_prompt_goes_to_the_notes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library = Library(tmp_path / "lib")
    _fill(library)
    papers = library.db.all_papers()
    monkeypatch.setattr(pick, "_fzf_select", lambda _: papers[:2])
    monkeypatch.setattr("builtins.input", lambda prompt="": pick.NOTES_KEY)
    went = {}
    monkeypatch.setattr(
        pick, "fzf_notes", lambda chosen, reading: went.setdefault("chosen", chosen) and ""
    )

    pick.fzf_round(
        lambda: pick.build_tree(papers),
        lambda *a: pytest.fail("notes are not an operation"),
        _reading_from(library),
    )

    assert went["chosen"] == papers[:2]


def test_the_title_comes_before_the_authors_in_fzf(monkeypatch) -> None:
    """Three papers by one group, led by the author list, were cut off by the
    terminal before anything told them apart."""
    paper = Paper(
        file_id="abc123",
        content_hash="h",
        store_name="abc123__Psychology",
        title="Automated decision aids",
        authors=["Luke Strickland", "Russell J. Boag", "Andrew Heathcote"],
        tags=["Psychology"],
    )
    shown = {}

    def fake_run(argv, input, stdout, text):
        shown["line"] = input
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(pick.subprocess, "run", fake_run)
    pick._fzf_select([paper])

    assert shown["line"].split("\t")[1:3] == ["----", "Automated decision aids"]

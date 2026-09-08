"""Choosing several documents at once, and doing one thing to all of them.

Two layers live here, and the split is the point: the **model** below knows the
tree, the selection, and what operations exist, and needs no terminal to do any
of it. The **screen** at the bottom draws that model and maps keys onto it, and
deliberately holds no rule worth testing -- because it is the half a test cannot
reach.

Adding an operation is an entry in `OPERATIONS`. The screen reads its footer and
its key bindings from there, so nothing in the drawing half changes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .db import Paper
from .library import Library, LibraryError
from .naming import split_category

# How many documents a confirmation names before it counts the rest. A question
# you have to scroll is one people stop reading, which is the opposite of what a
# confirmation is for.
CONFIRM_NAMES = 8

# The branch untagged documents are shown under. A label and not a folder: they
# link at the top of the tree, which is where opening this branch goes.
UNCATEGORISED = "(uncategorised)"


def label(paper: Paper) -> str:
    """What to call a document on screen.

    Mirrors `cli._label`: a document with no title -- a grant section, a chapter
    -- still has to be recognisable, and its original filename is what its owner
    knows it by.
    """
    return paper.title or paper.original_name or paper.store_name


# ---- the tree --------------------------------------------------------------


@dataclass
class Node:
    """One line in the tree: a category, or a document under one."""

    label: str
    depth: int
    paper: Paper | None = None
    children: list["Node"] = field(default_factory=list)
    expanded: bool = False

    # The tags leading here, which is also this category's folder under `tree/`.
    # Empty at the top, and for `(uncategorised)`, whose documents link there.
    path: tuple[str, ...] = ()

    @property
    def is_document(self) -> bool:
        return self.paper is not None

    def papers(self) -> list[Paper]:
        """Every document at or below this node, in display order."""
        if self.paper is not None:
            return [self.paper]
        return [p for child in self.children for p in child.papers()]


def build_tree(papers: Iterable[Paper]) -> list[Node]:
    """The category tree the library's own tags describe.

    Built from `paper.tags` rather than by walking `tree/`, which is a view that
    is rebuilt from the database and may not exist. A document with no tags is
    not dropped -- it goes under `(uncategorised)`, because a document you cannot
    see is one you cannot delete, and those are exactly the ones worth finding.
    """
    roots: list[Node] = []
    by_path: dict[tuple[str, ...], Node] = {}

    for paper in sorted(papers, key=lambda p: (list(p.tags or [UNCATEGORISED]), label(p).lower())):
        names = tuple(paper.tags or (UNCATEGORISED,))
        parent_children = roots
        for depth in range(len(names)):
            branch = names[: depth + 1]
            node = by_path.get(branch)
            if node is None:
                node = Node(
                    label=names[depth],
                    depth=depth,
                    path=branch if paper.tags else (),
                )
                by_path[branch] = node
                parent_children.append(node)
            parent_children = node.children
        parent_children.append(
            Node(label=label(paper), depth=len(names), paper=paper, path=names)
        )
    return roots


def visible(roots: Sequence[Node]) -> list[Node]:
    """The nodes currently on screen, in order.

    A collapsed category hides its subtree, which is what keeps a library of six
    hundred readable when it opens.
    """
    out: list[Node] = []

    def walk(nodes: Sequence[Node]) -> None:
        for node in nodes:
            out.append(node)
            if node.children and node.expanded:
                walk(node.children)

    walk(roots)
    return out


# ---- selection -------------------------------------------------------------


class Selection:
    """Which documents are chosen, as a set of ids.

    Ids rather than a flag on each node: a category's state is then *derived*
    from its documents, so a "partly selected" mark cannot drift out of step with
    what it summarises, and rebuilding the tree under a different filter keeps
    the selection intact.
    """

    def __init__(self) -> None:
        self._ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, paper: Paper) -> bool:
        return paper.file_id in self._ids

    def ids(self) -> set[str]:
        return set(self._ids)

    def chosen(self, roots: Sequence[Node]) -> list[Paper]:
        """The selected documents, in the tree's own order.

        Order matters for the confirmation: what it names should be what the
        eye just ran down.
        """
        return [p for node in roots for p in node.papers() if p.file_id in self._ids]

    def state(self, node: Node) -> str:
        """`all`, `some`, or `none` of what is under this node."""
        papers = node.papers()
        if not papers:
            return "none"
        marked = sum(1 for p in papers if p.file_id in self._ids)
        if marked == 0:
            return "none"
        return "all" if marked == len(papers) else "some"

    def toggle(self, node: Node) -> None:
        """Select everything under this node, or clear it.

        A category that is not already wholly selected becomes wholly selected --
        so pressing space on a partly-selected branch completes it rather than
        emptying it, which is the reading that matches what the eye expects from
        a half-filled box.
        """
        papers = node.papers()
        if self.state(node) == "all":
            self._ids.difference_update(p.file_id for p in papers)
        else:
            self._ids.update(p.file_id for p in papers)

    def set_all(self, roots: Sequence[Node], selected: bool) -> None:
        if selected:
            self._ids.update(p.file_id for node in roots for p in node.papers())
        else:
            self._ids.clear()


# ---- operations ------------------------------------------------------------


@dataclass(frozen=True)
class Operation:
    """One thing that can be done to a selection.

    `run` is per document rather than per batch so a partial failure can be
    reported as one: four removed, one refused, and which. A batch call can only
    say that something went wrong.
    """

    key: str
    label: str
    run: Callable[[Library, Paper, str], None]

    # What to ask before running, or nothing. Only destructive work earns a
    # question: one for an operation that undoes itself -- opening a document --
    # is a keystroke that teaches people to answer without reading.
    confirm: Callable[[Sequence[Paper]], str] | None = None

    # What to do when the cursor is on a category instead of a document. A
    # category is a folder, and opening one means opening the folder -- not
    # opening the several hundred documents beneath it, which nobody asked for.
    # An operation without one acts on the selection wherever the cursor rests.
    on_category: Callable[[Library, Node], None] | None = None

    # A line to ask for first, or nothing. The answer reaches `run` as its third
    # argument; operations that ask for nothing are handed an empty one.
    prompt: str | None = None

    # Whether it changes the library. An operation that only reads must not be
    # made to queue behind the write lock -- opening a document would then wait
    # out a filing pass, for a path it could have worked out itself.
    writes: bool = True


def _names(papers: Sequence[Paper]) -> str:
    """The documents a question is about, named until naming them stops helping."""
    named = "\n".join(f"  {label(p)}" for p in papers[:CONFIRM_NAMES])
    rest = len(papers) - CONFIRM_NAMES
    if rest > 0:
        named += f"\n  ... and {rest} more"
    return named


def _delete_confirm(papers: Sequence[Paper]) -> str:
    return (
        f"Delete {len(papers)} document(s)? This removes the file, its notes, "
        f"and its record.\n{_names(papers)}"
    )


DELETE = Operation(
    key="d",
    label="delete",
    confirm=_delete_confirm,
    run=lambda library, paper, _answer: library.remove(paper.file_id),
)


def viewer() -> str:
    """What hands a file to the desktop.

    Which viewer then opens it is the person's own setting, made once for every
    program they use, and not something a library should have an opinion about.
    """
    return "open" if sys.platform == "darwin" else "xdg-open"


def _path(library: Library, paper: Paper) -> str:
    """The document on disk, refusing a row whose file has gone.

    A row without its file is the state `fsck` exists to find; handing that path
    to a viewer would report the failure as the viewer's, which it is not.
    """
    path = library.store_path(paper)
    if not path.is_file():
        raise LibraryError(f"no file at {path}")
    return str(path)


def _spawn(command: list[str] | str, path: str | None = None, **kwargs) -> None:
    """Start something on the document and do not wait for it.

    Not `run`: what opens a document lives as long as the person reads it, and
    waiting would hold the picker until they close it. Its own session, so
    leaving the terminal does not take the reader down with it, and its output
    goes nowhere -- a viewer's warnings would otherwise land on top of the tree.
    """
    shown = command if isinstance(command, str) else command[0]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if path is not None else subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            **kwargs,
        )
    except OSError as err:  # no xdg-open on a machine with no desktop
        raise LibraryError(f"could not run {shown}: {err}") from err
    if path is not None:
        try:
            process.stdin.write(f"{path}\n")
        except OSError as err:  # a command that exits before it reads
            raise LibraryError(f"{shown} did not take the path: {err}") from err
        finally:
            process.stdin.close()
    # A moment, and only a moment: long enough for a mistyped command to come
    # back 127 and be reported as the failure it is, short enough that a reader
    # that stays open is not waited on. Still running is the answer we want.
    try:
        status = process.wait(timeout=0.05)
    except subprocess.TimeoutExpired:
        return
    if status != 0:
        raise LibraryError(f"{shown} exited with status {status}")


def _open_category(library: Library, node: Node) -> None:
    """Hand a category's folder in `tree/` to the desktop.

    The tree is a view rebuilt from the database and may not be there at all, so
    a missing folder is worth saying plainly -- `fsck` is what puts it back.
    """
    folder = library.tree_dir.joinpath(*node.path)
    if not folder.is_dir():
        raise LibraryError(f"no folder at {folder}")
    _spawn([viewer(), str(folder)])


OPEN = Operation(
    key="o",
    label="open",
    run=lambda library, paper, _answer: _spawn([viewer(), _path(library, paper)]),
    writes=False,
)

OPEN_WITH = Operation(
    key="O",
    label="open with",
    prompt="open with (path is piped in): ",
    # The line is run by a shell, and text mode, because what is typed is a
    # command line -- `xargs -r zathura`, a pipeline -- and not a filename.
    # It is the person's own shell, at their own terminal, doing what they typed.
    run=lambda library, paper, answer: _spawn(
        answer, _path(library, paper), shell=True, text=True
    ),
    # A category has no command to pipe a path into -- it is a folder, and the
    # desktop already knows what to do with one. So `O` on a branch opens it,
    # which is the half of "open" that `o` gave up when it became navigation.
    on_category=_open_category,
    writes=False,
)

def _retag(library: Library, paper: Paper, answer: str) -> None:
    """Move one document to the category typed at the prompt.

    Sanitized the same way a model's answer is, so a category typed by hand and
    one suggested by the model cannot become two spellings of the same shelf.
    """
    tags = split_category(answer)
    if not tags:  # `///` and the like: a line that survives the trim but says nothing
        raise LibraryError(f"no usable tags in {answer!r}")
    library.retag(paper.file_id, tags)


RETAG = Operation(
    key="r",
    label="retag",
    prompt="new category (A/B/C): ",
    # Asked once and applied down the selection, one document at a time. No
    # confirmation: typing a category is already the decision, and a shelf moved
    # by mistake is put back by typing the old one.
    run=_retag,
)

# The registry. A new operation is an entry here and nothing else: the screen
# takes its key bindings and its footer from this tuple.
OPERATIONS: tuple[Operation, ...] = (OPEN, OPEN_WITH, RETAG, DELETE)


def operation_for(key: str) -> Operation | None:
    """The operation a keystroke runs, or None when it runs nothing."""
    for operation in OPERATIONS:
        if operation.key == key:
            return operation
    return None


@dataclass
class Outcome:
    """What an operation did, for a line the person reads afterwards."""

    done: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    def describe(self, operation: Operation) -> str:
        if not self.failed:
            return f"{operation.label}: {len(self.done)} document(s)"
        return (
            f"{operation.label}: {len(self.done)} done, {len(self.failed)} failed"
            + "".join(f"\n  ! {name}: {why}" for name, why in self.failed)
        )


def act(
    operation: Operation,
    chosen: Sequence[Paper],
    ask: Callable[[str], bool],
    do: Callable[[Operation, Sequence[Paper], str, "Node | None"], str],
    request: Callable[[str], str | None] = lambda prompt: None,
    category: "Node | None" = None,
) -> str:
    """Decide whether an operation runs, on what, and say what happened.

    `category` is the branch the cursor rests on, when it rests on one: an
    operation that knows what to do with a folder is aimed at that instead.

    Both rules live here rather than in the screen, where a test cannot reach
    them: nothing destructive happens without a yes, and an empty selection is
    not worth asking about. An operation that wants a line gets one first --
    asked before the confirmation, so what is confirmed is what will happen.
    """
    if category is not None and operation.on_category is not None:
        # Opening a folder undoes itself and names itself; there is nothing a
        # question would add, and no selection it depends on.
        return do(operation, (), "", category)
    if not chosen:
        return f"nothing selected to {operation.label}"
    answer = ""
    if operation.prompt is not None:
        answer = (request(operation.prompt) or "").strip()
        if not answer:  # nothing typed, or backed out: the same decision
            return f"{operation.label} cancelled"
    if operation.confirm is not None and not ask(operation.confirm(chosen)):
        return f"{operation.label} cancelled"
    return do(operation, chosen, answer, None)


def apply(
    operation: Operation,
    library: Library,
    papers: Sequence[Paper],
    answer: str = "",
    category: Node | None = None,
) -> Outcome:
    """Run one operation over the selection, surviving a document that refuses.

    One failure must not strand the rest: the person asked for all of them, and
    stopping halfway leaves a state nobody chose. With a `category`, the aim is
    that one folder rather than the documents.
    """
    if category is not None and operation.on_category is not None:
        jobs = [(category.label, lambda: operation.on_category(library, category))]
    else:
        jobs = [
            (label(paper), lambda paper=paper: operation.run(library, paper, answer))
            for paper in papers
        ]

    outcome = Outcome()
    for name, job in jobs:
        try:
            job()
            outcome.done.append(name)
        except (LibraryError, OSError) as err:
            outcome.failed.append((name, str(err)))
    return outcome


# ---- the screen ------------------------------------------------------------
#
# Everything below draws the model above and maps keys onto it. It holds no rule
# a test would want to assert, which is the deal: the half that cannot be tested
# is the half that decides nothing.

MARKS = {"all": "[x]", "some": "[-]", "none": "[ ]"}

# Which way a category is turned. Triangles rather than `>` and `v`: the pair
# reads as one shape rotating, which is what opening a branch actually is.
ARROW_OPEN, ARROW_SHUT = "\u25be", "\u25b8"  # black down/right-pointing small triangle

# Keys that are not operations. Operations bind themselves, from OPERATIONS.
QUIT_KEYS = frozenset({"q", "\x1b"})
DOWN_KEYS = frozenset({"j"})
UP_KEYS = frozenset({"k"})
OPEN_KEYS = frozenset({"l", "\n", "\r"})
CLOSE_KEYS = frozenset({"h"})

# What an arrow key sends, folded onto the letter that already means it, so the
# model below is asked about one alphabet rather than two.
ARROWS = {"[A": "k", "[B": "j", "[C": "l", "[D": "h"}


def footer() -> str:
    """The key legend, built from the registry so a new operation appears in it."""
    operations = "  ".join(f"{op.key} {op.label}" for op in OPERATIONS)
    return f"j/k move  space select  a all  A none  {operations}  q quit"


class Screen:
    """Draws the tree and turns keystrokes into calls on the model."""

    def __init__(self, roots: list[Node], selection: Selection) -> None:
        self.roots = roots
        self.selection = selection
        self.cursor = 0
        self.message = ""
        for node in roots:  # opens collapsed to the top level
            node.expanded = False

    # -- the part the drawing loop calls --

    def rows(self) -> list[Node]:
        return visible(self.roots)

    def under_cursor(self) -> Node | None:
        rows = self.rows()
        return rows[self.cursor] if 0 <= self.cursor < len(rows) else None

    def line(self, node: Node) -> "Text":
        """One row, styled: a category reads as structure, a document as content."""
        from rich.text import Text

        state = self.selection.state(node)
        text = Text("  " * node.depth)
        text.append(MARKS[state], style="bold green" if state != "none" else "dim")
        if node.is_document:
            text.append(f" {node.label}")
            return text
        text.append(f" {ARROW_OPEN if node.expanded else ARROW_SHUT} ", style="dim")
        text.append(node.label, style="bold")
        text.append(f"  ({len(node.papers())})", style="dim")
        return text

    def handle(self, key: str) -> Operation | None:
        """Apply one keystroke. Returns an operation when the key asks for one.

        The operation is handed back rather than run here: running it needs a
        library and a confirmation, and neither belongs to a thing that draws.
        """
        rows = self.rows()
        if not rows:
            return None
        self.cursor = max(0, min(self.cursor, len(rows) - 1))
        node = rows[self.cursor]

        if key in DOWN_KEYS:
            self.cursor = min(self.cursor + 1, len(rows) - 1)
        elif key in UP_KEYS:
            self.cursor = max(self.cursor - 1, 0)
        elif key in OPEN_KEYS:
            if node.children:
                node.expanded = True
        elif key in CLOSE_KEYS:
            if node.children and node.expanded:
                node.expanded = False
        elif key == OPEN.key and not node.is_document:
            # Opening a branch on screen is seeing what is in it. Opening the
            # folder it stands for is a different thing, and is `O`.
            node.expanded = True
        elif key == " ":
            self.selection.toggle(node)
        elif key == "a":
            self.selection.set_all(self.roots, True)
        elif key == "A":
            self.selection.set_all(self.roots, False)
        else:
            return operation_for(key)
        return None


def _view(screen: Screen, height: int):
    """The tree, a status line, and the legend, as tall as it needs to be.

    It draws in place among whatever else is in the scrollback rather than
    taking the terminal over, so `height` is a ceiling and not a shape: a
    library of four is four lines, and one of six hundred scrolls within what
    the window can spare.
    """
    from rich.text import Text

    rows = screen.rows()
    body = max(1, min(len(rows), height - 3))
    # Keep the cursor on screen without scrolling further than there is content.
    top = max(0, min(screen.cursor - body // 2, max(0, len(rows) - body)))
    out = Text()
    for offset, node in enumerate(rows[top : top + body]):
        line = screen.line(node)
        if top + offset == screen.cursor:
            line.stylize("reverse")
        out.append_text(line)
        out.append("\n")
    status = screen.message or f"{len(screen.selection)} selected"
    out.append(status + "\n", style="yellow" if screen.message else "dim")
    out.append(footer(), style="dim")
    return out


def _question(text: str):
    from rich.text import Text

    out = Text(text + "\n\n")
    out.append("y to confirm, anything else cancels", style="bold yellow")
    return out


def _line(prompt: str, typed: str):
    from rich.text import Text

    out = Text(prompt, style="bold yellow")
    out.append(typed)
    out.append("\u2588", style="dim")  # where the next character lands
    out.append("\n\nenter runs it, escape cancels", style="dim")
    return out


def _read_key(fd: int, arrows: bool = True) -> str:
    """One keystroke, with arrows folded onto the letters that mean the same.

    An arrow arrives as three bytes and a bare escape as one, and nothing but
    the pause between them tells the two apart -- so a short wait is what asks.

    `arrows` off while a line is being typed: there the fold would be wrong, and
    reaching for the left arrow would silently type an `h` into the command.
    """
    import select

    key = os.read(fd, 1).decode(errors="ignore")
    if key != "\x1b":
        return key
    if not select.select([fd], [], [], 0.05)[0]:
        return "\x1b"
    sequence = os.read(fd, 2).decode(errors="ignore")
    return ARROWS.get(sequence, "") if arrows else ""


def _read_line(fd: int, show: Callable[[object], None], prompt: str) -> str | None:
    """A line typed at the picker, or None if it was abandoned.

    Backspace and escape only: a command short enough to type at a prompt is one
    short enough to retype, and a line editor here would be a second, worse copy
    of the one the shell already has.
    """
    typed = ""
    while True:
        show(_line(prompt, typed))
        key = _read_key(fd, arrows=False)
        if key in ("\n", "\r"):
            return typed
        if key == "\x1b":
            return None
        if key in ("\x7f", "\b"):
            typed = typed[:-1]
        elif key.isprintable():
            typed += key


def run(
    load: Callable[[], list[Node]],
    on_apply: Callable[[Operation, Sequence[Paper], str, "Node | None"], str],
) -> None:
    """Show the tree until asked to stop.

    `on_apply` performs an operation and returns the line to show afterwards,
    and `load` re-reads the tree. Both are passed in rather than imported:
    applying needs a library, and how the library is reached -- directly, or
    through the watcher holding the write lock -- is the caller's business, not
    this screen's.

    The tree is reloaded after every operation. A delete leaves what it removed
    on screen otherwise, and any operation a later release adds may change what
    the tree should say just as much.
    """
    import termios
    import tty

    from rich.console import Console
    from rich.live import Live

    console = Console()
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    screen = Screen(load(), Selection())
    # cbreak rather than raw: it leaves the interrupt character alone, so ctrl-c
    # still ends this the way it ends everything else.
    tty.setcbreak(fd)
    try:
        with Live(console=console, auto_refresh=False, transient=True) as live:

            def show(renderable) -> None:
                live.update(renderable, refresh=True)

            while True:
                show(_view(screen, console.size.height))
                key = _read_key(fd)
                if key in QUIT_KEYS:
                    break
                operation = screen.handle(key)
                if operation is None:
                    continue
                chosen = screen.selection.chosen(screen.roots)
                applied = False

                def ask(question: str) -> bool:
                    show(_question(question))
                    return _read_key(fd) == "y"

                def do(
                    op: Operation,
                    papers: Sequence[Paper],
                    answer: str,
                    category: Node | None,
                ) -> str:
                    nonlocal applied
                    applied = True
                    return on_apply(op, papers, answer, category)

                here = screen.under_cursor()
                message = act(
                    operation,
                    chosen,
                    ask,
                    do,
                    request=lambda prompt: _read_line(fd, show, prompt),
                    category=None if here is None or here.is_document else here,
                )
                if not applied:
                    screen.message = message
                    continue
                # Re-read rather than patch the tree: what an operation changed
                # is its own business, and guessing here would be wrong the first
                # time one of them does something other than remove a row.
                cursor = screen.cursor
                screen = Screen(load(), Selection())
                screen.cursor = cursor
                screen.message = message
        # The picker clears itself on the way out; what it did should not go
        # with it, so the last line said is said again where it will stay.
        if screen.message:
            console.print(screen.message)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)

"""The user guide's examples run, and print what the guide says they print.

``docs/user-guide/`` teaches each stage of the library with a short ``python``
example followed by a ``text`` block holding its output. A guide is read long
after it is written, so a rename or a number that moves must fail the gate
rather than leave the page quietly wrong.

Rules for a page:

- every ``python`` block runs on its own, from the repository root;
- when the next fenced block is a ``text`` block, it is the example's
  expected standard output, line for line; prose may sit between the two;
- a last expected line of ``...`` stands for any further lines, so a long
  output can be shown in part.
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "user-guide"

_BLOCK_RE = re.compile(r"^```(\w*)\n(.*?)^```\n", re.MULTILINE | re.DOTALL)


def _examples() -> list[tuple[str, str, str | None]]:
    """Every ``python`` block in the guide, with the ``text`` block that follows it."""
    examples = []
    for page in sorted(GUIDE.glob("*.md")):
        text = page.read_text(encoding="utf-8")
        blocks = list(_BLOCK_RE.finditer(text))
        for index, block in enumerate(blocks):
            if block.group(1) != "python":
                continue
            line = text.count("\n", 0, block.start()) + 1
            expected = None
            if index + 1 < len(blocks) and blocks[index + 1].group(1) == "text":
                expected = blocks[index + 1].group(2)
            examples.append((f"{page.name}:{line}", block.group(2), expected))
    return examples


EXAMPLES = _examples()


def test_the_guide_has_examples() -> None:
    assert EXAMPLES, f"found no python examples under {GUIDE}"


@pytest.mark.parametrize(
    ("source", "expected"),
    [(code, expected) for _, code, expected in EXAMPLES],
    ids=[where for where, _, _ in EXAMPLES],
)
def test_example_prints_what_the_guide_shows(
    source: str, expected: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(source, "<user-guide example>", "exec"), {"__name__": "__main__"})
    if expected is None:
        return
    printed = stdout.getvalue().rstrip("\n").splitlines()
    wanted = expected.rstrip("\n").splitlines()
    if wanted and wanted[-1] == "...":
        wanted = wanted[:-1]
        printed = printed[: len(wanted)]
    assert printed == wanted

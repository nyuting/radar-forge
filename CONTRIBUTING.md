# Contributing to radar-forge

`radar-forge` is built for interns, students and researchers, and a good deal of it will be
*written* by them too. These conventions exist so that a first contribution can be reviewed
for its radar content rather than its formatting.

Everything here is enforced automatically. You should never have to remember a rule — the
hooks will tell you, and they will tell you how to fix it.

---

## 1. Setup

You need [`uv`](https://docs.astral.sh/uv/). Then:

```bash
git clone https://github.com/yutingng/radar-forge.git
cd radar-forge
./scripts/setup-dev.sh
```

That creates the virtual environment (pinned to Python 3.11 by `.python-version`), installs
the dev dependencies, and activates the git hooks by pointing `core.hooksPath` at
`.githooks/`. It is safe to re-run at any time, and you should re-run it after pulling
changes to `pyproject.toml`.

Confirm it worked:

```bash
make check
```

---

## 2. The loop

```bash
git switch -c feat/taylor-tapering     # branch naming: docs/conventions/commits.md
# ... edit ...
make fmt                               # auto-format and auto-fix
make test-fast                         # quick feedback while iterating
git add -u && git commit -m "feat(array): add Taylor tapering for planar arrays"
make check                             # the full gate, before you push
git push -u origin HEAD
```

`make help` lists every target.

---

## 3. What the hooks do

There are three, all plain shell scripts in [`.githooks/`](.githooks/) — no framework, no
extra dependency. Read them; they are short.

### `pre-commit` — fast, staged files only

Runs in well under a second so it never gets in your way:

1. `ruff format --check` and `ruff check` on staged Python.
2. Blocks files over 1 MiB, and any `.npy`/`.npz`/`.h5`/`.mat` outside `tests/data/golden/`.
   A radar repo generates large cubes; keeping them out of history is far cheaper than
   excising them from history later.
3. Blocks leftover `breakpoint()`/`pdb.set_trace()` and merge-conflict markers.
4. Blocks notebooks committed with stored outputs — teaching notebooks are reviewed as
   source, and outputs make diffs unreadable.
5. Runs `scripts/check_conventions.py` on the staged files. That is where the project-specific
   rules live, and it is the answer to "how is this actually enforced?":

   | Rule | Catches |
   | :--- | :--- |
   | R1 `toml` | `setup.py`, `setup.cfg`, `requirements.txt`, standalone linter configs |
   | R2 `layout` | Non-Python files inside the import package |
   | R3 `docstring` | A library module with no module docstring |
   | R4 `constants` | Redefining a shared constant, or hardcoding its literal (`3e8`, `1.38e-23`, …) |
   | R5 `units` | A parameter named `range`, `gain_tx`, `power`… with no unit suffix |
   | R6 `broadcast` | `np.tile`/`np.repeat`/`np.broadcast_to` with no justification comment |

   Each failure prints the rule, the offending line, the fix, and a link to the guide. Rules that
   a tool genuinely cannot judge are tagged **[review]** in
   [style.md](docs/conventions/style.md) — that guide labels every rule with what enforces it.

It deliberately **checks** rather than auto-formats: rewriting files mid-commit would leave
the index disagreeing with your worktree and silently commit content you never saw. When it
complains, run `make fmt` and `git add -u`.

### `commit-msg` — Conventional Commits

Validates the subject line against `<type>(<scope>): <summary>`. Full rules and the list of
types and scopes: [docs/conventions/commits.md](docs/conventions/commits.md). The error
message prints working examples.

### `pre-push` — the full gate

Cheapest check first, so a formatting slip fails in a second rather than a minute:
`ruff format --check` → `ruff check` → `mypy` → conventions → **the full `pytest` suite**.

It also blocks pushing straight to `main`. Open a branch and a PR instead. As the sole
maintainer you can override that for a one-off:

```bash
RADAR_FORGE_ALLOW_MAIN_PUSH=1 git push
```

If the full suite becomes slow enough to hurt, that is a signal to mark the expensive tests
`@pytest.mark.slow` (see [testing.md](docs/conventions/testing.md)) — not to weaken the
push gate. Iterate with `make test-fast`; push still runs everything.

### Bypassing

`git commit --no-verify` and `git push --no-verify` work. They are for genuine emergencies,
not for "I'll fix it later" — CI runs exactly the same checks, so you have deferred the
failure, not avoided it.

---

## 4. The guides

| Guide | Read it when |
| :--- | :--- |
| [conventions/commits.md](docs/conventions/commits.md) | Writing a commit, naming a branch, opening a PR |
| [conventions/style.md](docs/conventions/style.md) | Writing library code — naming, **units**, docstrings, typing |
| [conventions/testing.md](docs/conventions/testing.md) | Writing tests — tolerances, ground truth, golden data, markers |
| [CLAUDE.md](CLAUDE.md) | Directing an AI agent at this repo |

The single best thing to read before writing your first module is
[`src/radar_forge/core/radar_equation.py`](src/radar_forge/core/radar_equation.py). It is
deliberately maintained as the worked example of every convention at once.

---

## 5. Pull requests

- One logical change per PR. A refactor and a feature in the same diff cannot be reviewed.
- The PR title becomes the squash-merge commit subject, so it must itself be a valid
  Conventional Commit.
- New physics or DSP needs a citation — the textbook section or paper it implements — in the
  docstring's `References`, and a test against analytic ground truth where one exists.
- New dependencies need a justification in the PR description. Heavy or optional backends
  (ray-tracing engines, PyTorch) belong in an extra, never in the core dependency list.

## 6. Reporting bugs

Include the `radar-forge` version, Python version, a minimal reproducing script, and — for a
numerical discrepancy — the expected value *and where it comes from*. "The sidelobes look
wrong" cannot be acted on; "first sidelobe is −13.2 dB, Richards §4.3 gives −13.26 dB for an
unweighted ULA" can.

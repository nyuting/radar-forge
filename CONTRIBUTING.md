# Contributing to radar-forge

`radar-forge` is a radar workbench for research and education, much of it written by the
interns and students who read it. The rules, and the check that enforces each, are in
[CLAUDE.md](CLAUDE.md); the conventions behind them are in [`docs/conventions/`](docs/conventions/).
The hooks tell you when you break one, and how to fix it.

## Setup

You need [`uv`](https://docs.astral.sh/uv/). On Windows, run `make` and the hooks from Git Bash.

```bash
git clone https://github.com/nyuting/radar-forge.git && cd radar-forge
./scripts/setup-dev.sh                                  # venv, dev deps, git hooks
uv sync --extra dev --extra visualisation               # add --extra notebooks for Jupyter
make check
```

`uv sync` removes every extra you do not name, so always name them all. Never `--all-extras`:
it pulls in PyTorch and Mitsuba.

## The loop

```bash
git switch -c feat/taylor-tapering
make fmt          # format and auto-fix
make test-fast    # everything not marked slow
git commit -m "feat(array): add Taylor tapering for planar arrays"
make check        # lint, types, conventions and the full suite
git push -u origin HEAD
```

`make help` lists every target.

## The hooks

Plain shell scripts in [`.githooks/`](.githooks/), activated by `setup-dev.sh` (`make hooks`).

- **pre-commit** (staged files, under a second): ruff format and lint; blocks files over 1 MiB,
  binary arrays outside `tests/data/golden/`, breakpoints, conflict markers and notebooks with
  outputs; runs `scripts/check_conventions.py --staged` (rules R1–R12).
- **commit-msg**: the subject is a Conventional Commit
  ([commits.md](docs/conventions/commits.md)).
- **pre-push**: ruff, mypy, conventions, then `pytest -m "not slow"`. It blocks pushing to
  `main`. CI runs all of that plus the slow tests.

`--no-verify` only defers the failure to CI.

## Pull requests

- One logical change per PR. PRs squash-merge, so the title must be a valid Conventional Commit
  and the description becomes the commit body: what changed, why, what to check.
- New physics or DSP cites its source in `References` and is tested against analytic ground
  truth where one exists. A new dependency says what it buys over numpy and scipy.
- Before implementing a spec, check it; fix a wrong spec in a `docs(spec)` commit first.
- Report a numerical bug with the expected value and where it comes from.

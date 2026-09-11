# CLAUDE.md

Instructions for AI agents working in this repository. Humans should read
[CONTRIBUTING.md](CONTRIBUTING.md) instead; it covers the same ground at more length.

## What this is

`radar-forge` is a Python radar simulation and DSP library for **education and research**.
Its readers are interns and students. Code that is clever but opaque fails the brief;
code that is plain, documented, and cites its source passes it.

## Non-negotiables

- **Python ≥ 3.11**, src layout. Library code lives only in `src/radar_forge/`.
- **All configuration is TOML, in `pyproject.toml`.** Never add `setup.py`, `setup.cfg`,
  `.flake8`, `tox.ini`, `requirements*.txt`, or a standalone `ruff.toml`/`mypy.ini`.
  `scripts/check_conventions.py` blocks these.
- **Run everything through `uv`**: `uv run pytest`, `uv run ruff …`. Never `pip install`.
- **`make check` must pass before you say you are done.** It runs lint, types,
  conventions and the full test suite — the same gate as `pre-push` and CI.
- **SI units, with the unit in the name**: `range_m`, `f0_hz`, `chirp_time_s`, `rcs_dbsm`,
  `power_w`. Linear internally; decibels only at API boundaries, and named `_db*`.
- **NumPy-style docstrings on everything public**, including a `References` section citing
  the textbook or paper the block implements, and array shapes written out as
  `(n_chirps, n_samples)`.
- **Complete type annotations**; arrays as `numpy.typing.NDArray[np.float64]` /
  `ArrayLike` in, `NDArray` out. `mypy --strict` must pass.
- **Conventional Commits**: `feat(array): add Taylor tapering`. The `commit-msg` hook enforces it.

## Working agreements

- Vectorise with NumPy; if a Python loop is unavoidable, write a comment saying why.
- Never commit generated arrays (`.npy`, `.h5`, …). Commit the script that generates them.
  Small golden reference data goes in `tests/data/golden/` only.
- Do not weaken a test tolerance to make a test pass. Find out why the number moved.
- Do not add a dependency without saying, in the PR, what it buys and why the existing
  `numpy`/`scipy` stack cannot do it. Heavy or optional backends go in an extra, never core.
- Prefer analytic ground truth in tests (a known null, an exact bin) over recorded output.
- `--no-verify` is not a fix. CI runs the same checks.

## Where things are

| Path | What |
| :--- | :--- |
| `src/radar_forge/core/radar_equation.py` | The style exemplar — copy its shape |
| `docs/conventions/style.md` | Naming, units, docstrings, typing, API surface |
| `docs/conventions/commits.md` | Commit and branch conventions |
| `docs/conventions/testing.md` | Test layout, tolerances, golden data, markers |
| `.githooks/` | pre-commit, commit-msg, pre-push |
| `Makefile` | The single definition of every gate |
| `spec/` | Design specification; the intended end state |

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
  `power_w`. Linear internally; decibels only at API boundaries, and named `_db*`. Dimensionless
  ratios say which: `gain_tx_linear` or `gain_tx_dbi`, never `gain_tx`. A commit hook rejects bare
  names, matching on the leading token.
- **Physical constants come from `radar_forge.core.constants`** — `SPEED_OF_LIGHT_MPS`,
  `BOLTZMANN_JPK`, etc. Never define one elsewhere, never hardcode its value, never write `3e8`
  (0.07% high = 70 cm of range error at 1 km). A commit hook rejects both the rebinding and the
  literal.
- **NumPy-style docstrings on everything public**, including a `References` section citing
  the textbook or paper the block implements, and array shapes written out as
  `(n_pulses, n_samples)`.
- **Complete type annotations**; arrays as `numpy.typing.NDArray[np.float64]` /
  `ArrayLike` in, `NDArray` out. `mypy --strict` must pass.
- **Conventional Commits**: `feat(array): add Taylor tapering`. The `commit-msg` hook enforces it.

## Working agreements

- Vectorise with NumPy; if a Python loop is unavoidable, write a comment saying why.
- Broadcast rather than `np.tile`/`np.repeat`/`np.broadcast_to`. A hook rejects those unless the
  call carries `# broadcast-exempt: <reason>` on its line or the one above.
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
| `docs/conventions/style.md` | Naming, units, data layout, docstrings, typing, API surface |
| `docs/conventions/commits.md` | Commit and branch conventions |
| `docs/conventions/testing.md` | Test layout, tolerances, golden data, markers |
| `src/radar_forge/core/constants.py` | Every physical constant, defined once |
| `.githooks/` | pre-commit, commit-msg, pre-push |
| `scripts/check_conventions.py` | Rules R1–R6: TOML config, layout, docstrings, constants, units, broadcasting |
| `Makefile` | The single definition of every gate |
| `spec/` | Design specification; the intended end state |

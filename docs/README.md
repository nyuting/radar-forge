# radar-forge documentation

## Conventions

These are the rules the git hooks and CI enforce. Read them before your first contribution.

- [**commits.md**](conventions/commits.md) — Conventional Commits, branch naming, PR titles,
  changelog policy.
- [**style.md**](conventions/style.md) — naming, the SI **unit-suffix convention**, the shared
  **constants module**, NumPy docstrings and citation policy, typing, public API surface, NumPy
  practice, errors. Every rule is tagged with what enforces it: `[ruff]`, `[mypy]`, `[hook]` or
  `[review]`.
- [**testing.md**](conventions/testing.md) — test layout, float-comparison tolerances,
  analytic ground truth over golden data, fixtures and seeding, property-based tests, markers.

## Getting started

- [CONTRIBUTING.md](../CONTRIBUTING.md) — setup, the development loop, what each hook does.
- [CLAUDE.md](../CLAUDE.md) — the same conventions, condensed for AI agents.
- [`src/radar_forge/core/radar_equation.py`](../src/radar_forge/core/radar_equation.py) — the
  worked example that demonstrates every convention at once.
- [`src/radar_forge/core/constants.py`](../src/radar_forge/core/constants.py) — every physical
  constant, defined once. Import from here; never redefine, never hardcode.

## Design

- [`spec/starter.md`](../spec/starter.md) — project charter and ecosystem survey.
- [`spec/structure.md`](../spec/structure.md) — file-level module design.
- [`spec/scenario-001-xband.md`](../spec/scenario-001-xband.md) — the first
  vertical slice: trajectory → IQ → range-Doppler map, in three ambiguity variants.
- [`spec/scenario-002-bistatic.md`](../spec/scenario-002-bistatic.md) — the same
  path over a two-site geometry: the Raleigh-Durham illuminator received at the Duke Receiver,
  in an X-band and an S-band variant.
- [`spec/scenario-003-tracking.md`](../spec/scenario-003-tracking.md) — CFAR
  detection, data association and Kalman tracking stacked on scenario 001's S1 variant, in two
  waveform variants. §14 records the seven things building it changed.

## Scenario companions

Visual, orienting walk-throughs for running a scenario for the first time — the geometry, the
pipeline, the pictures and the knobs. They are **not normative**; the `spec/` documents above are.

- [**scenario 001 — X-band**](scenarios/scenario-001-xband.md) — one site, three waveforms, and
  why the three sample rates differ.
- [**scenario 002 — bistatic**](scenarios/scenario-002-bistatic.md) — two sites, the bistatic
  angle, and what stayed identical in the code.
- [**scenario 003 — tracking**](scenarios/scenario-003-tracking.md) — the detection pipeline, the
  track lifecycle, and three things that look like bugs and are not.

## Library guides

API documentation and the intern onboarding tutorials land here as the modules are built.

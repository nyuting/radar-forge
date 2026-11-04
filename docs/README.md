# radar-forge documentation

## User guide

New to the library? Start with the [**user guide**](user-guide/README.md). It explains each stage
of `radar_forge.core`, from geometry to tracks: what it does, a table of its quantities and
equations with worked values, an example to run, and a scenario to try it in.

1. [The radar and its link budget](user-guide/01-radar-and-link-budget.md)
2. [Geometry and propagation](user-guide/02-geometry-and-propagation.md)
3. [Waveforms and signals](user-guide/03-waveforms-and-signals.md)
4. [Range-Doppler processing](user-guide/04-range-doppler-processing.md)
5. [Detection](user-guide/05-detection.md)
6. [Tracking](user-guide/06-tracking.md)

The [scenarios](#scenarios) below are where you try each page out.

## Conventions

The rules, and the check that enforces each, are summarised in [CLAUDE.md](../CLAUDE.md).

- [**style.md**](conventions/style.md) — what each tool enforces, then the judgement calls:
  units, naming, data layout, docstrings and citations, typing, public API, NumPy practice, errors.
- [**testing.md**](conventions/testing.md) — layout, tolerances, ground truth, markers, and which
  radar-domain tests each stage needs.
- [**review.md**](conventions/review.md) — reviewing a PR (also `/pr-review`), auditing code
  against the spec (Q1–Q8) and refactoring (F1–F9).
- [**commits.md**](conventions/commits.md) — Conventional Commits, branches, squash-merged PRs.

## Getting started

- [CONTRIBUTING.md](../CONTRIBUTING.md) — setup, the development loop, what each hook does.
- [CLAUDE.md](../CLAUDE.md) — the same conventions, condensed for AI agents.
- [`src/radar_forge/core/radar_equation.py`](../src/radar_forge/core/radar_equation.py) — the
  worked example that demonstrates every convention at once.
- [`src/radar_forge/core/constants.py`](../src/radar_forge/core/constants.py) — every physical
  constant, defined once. Import from here; never redefine, never hardcode.

## Design


- [`spec/data-001-formats.md`](../spec/data-001-formats.md) — the files a run writes.
- [`spec/tracker-001.md`](../spec/tracker-001.md) — the tracking architecture.

## Scenarios

Each scenario has a specification in `spec/` and, once it runs, a TOML file in `scenarios/`.
Run one with `uv run python scripts/run_scenario.py scenarios/<file>.toml --out out/<name>`.

| Scenario | TOML | Status |
| :--- | :--- | :--- |
| [001, X-band](../spec/scenario-001-xband.md): trajectory to IQ to range-Doppler map, three ambiguity variants | `scenario_001_fmcw_low_prf.toml`, `scenario_001_fmcw_dual_prf.toml`, `scenario_001_pulsed_medium_prf.toml` | Runs |
| [002, bistatic](../spec/scenario-002-bistatic.md): the Raleigh-Durham transmitter received at the Duke receiver | `scenario_002_bistatic_xband.toml`, `scenario_002_bistatic_sband.toml` | Runs |
| [003, tracking](../spec/scenario-003-tracking.md): CFAR, association and tracking on 001 | `scenario_003_tracking.toml`, `scenario_003_tracking_dual_prf.toml`, `scenario_003_ukf_*.toml` | Runs |
| [004, keystone](../spec/scenario-004-keystone.md): fast sounding rocket | none | Specified, not built |
| [005, leakage cancellation](../spec/scenario-005-leakage-cancellation.md): continuous-wave leakage | none | Specified, not built |
| [006, parallel-approach DoA](../spec/scenario-006-parallel-approach-doa.md): two aircraft in one cell | none | Specified, not built |
| [007, interference](../spec/scenario-007-interference.md): strong interferers against a weak target | none | Specified, not built |
| [008, STAP](../spec/scenario-008-stap.md): airborne clutter ridge | none | Specified, not built |
| [009, receiver chain](../spec/scenario-009-receiver-chain.md): front end, DDC, impairments | none | Specified, not built |
| [010, self-calibration](../spec/scenario-010-self-calibration.md): array perturbations | none | Specified, not built |
| [011, terminal-area tracking](../spec/scenario-011-terminal-area-tracking.md): crossings, turns, clutter; IMM, JPDA, MHT, PMBM | none | Specified, not built |

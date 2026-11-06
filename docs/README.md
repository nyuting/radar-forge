# radar-forge documentation

## User guide

New to the library? Start with the [**user guide**](user-guide/README.md). It explains each stage
of the radar chain, from geometry to tracks: what it does, a table of its quantities and
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
- [`src/radar_forge/propagation.py`](../src/radar_forge/propagation.py) — the style exemplar:
  its `received_power_w` is the radar equation, and it shows every convention at once.
- [`src/radar_forge/constants.py`](../src/radar_forge/constants.py) — every physical
  constant, defined once. Import from here; never redefine, never hardcode.

## Design


- [`spec/data-001-formats.md`](../spec/data-001-formats.md) — the files a run writes.
- [`spec/tracker-001.md`](../spec/tracker-001.md) — the tracking architecture.

## Scenarios

A scenario is specified in `spec/` until it is built; then its TOML in `config/` carries its
acceptance numbers in a `[verify]` table and its spec is deleted. Run one with
`uv run python scripts/run_scenario.py config/<file>.toml --frames 3`, which writes
`runs/<file>/` and exits 1 if a check fails.

| Scenario | TOML | Status |
| :--- | :--- | :--- |
| 001, X-band: trajectory to IQ to range-Doppler map, three ambiguity variants | `scenario_001_fmcw_low_prf.toml`, `scenario_001_fmcw_dual_prf.toml`, `scenario_001_pulsed_medium_prf.toml` | Runs; checked by `[verify]` |
| 002, bistatic: the Raleigh-Durham transmitter received at the Duke receiver | `scenario_002_bistatic_xband.toml`, `scenario_002_bistatic_sband.toml` | Runs; checked by `[verify]` |
| 003, tracking: CFAR, association and tracking on 001 | `scenario_003_tracking.toml`, `scenario_003_tracking_dual_prf.toml`, `scenario_003_ukf_*.toml` | Runs; checked by `[verify]` |
| 004, keystone: a fast sounding rocket; range walk, Doppler spread and fold, corrected and estimated on a data chip | `scenario_004_keystone_k1.toml`, `scenario_004_keystone_k2.toml`, `scenario_004_keystone_k3.toml` | Runs; checked by `[verify]` |
| 005, leakage cancellation: a noise radar cancels its own leakage and near clutter | `scenario_005_static_leakage.toml`, `scenario_005_varying_clutter.toml`, `scenario_005_distorted_transmitter.toml` | Runs; checked by `[verify]` |
| [006, parallel-approach DoA](../spec/scenario-006-parallel-approach-doa.md): two aircraft in one cell | none | Specified, not built |
| [007, interference](../spec/scenario-007-interference.md): strong interferers against a weak target | none | Specified, not built |
| [008, STAP](../spec/scenario-008-stap.md): airborne clutter ridge | none | Specified, not built |
| [009, receiver chain](../spec/scenario-009-receiver-chain.md): front end, DDC, impairments | none | Specified, not built |
| [010, self-calibration](../spec/scenario-010-self-calibration.md): array perturbations | none | Specified, not built |
| 011a, terminal-area tracking: turns, crossings, clutter; IMM, JPDA, JIPDA, the plot model | `scenario_011_manoeuvre.toml`, `scenario_011_crossing.toml`, `scenario_011_terminal_dense.toml`, `scenario_011_iq_check.toml` | Runs; checked by `[verify]` |
| 011b, track-oriented MHT and PMBM: the separation sweep | `scenario_011_cst_sweep.toml`; MHT and PMBM also score `scenario_011_crossing.toml` and `scenario_011_terminal_dense.toml` | Runs; checked by `[verify]` |

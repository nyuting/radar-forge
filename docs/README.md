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

The [scenario companions](#scenario-companions) below are where you try each page out.

## Conventions

The rules, and the check that enforces each, are summarised in [CLAUDE.md](../CLAUDE.md).

- [**style.md**](conventions/style.md) — what each tool enforces, then the judgement calls:
  units, naming, data layout, docstrings and citations, typing, public API, NumPy practice, errors.
- [**testing.md**](conventions/testing.md) — layout, tolerances, ground truth, markers, and which
  radar-domain tests each stage needs.
- [**review.md**](conventions/review.md) — reviewing a PR (also `/pr-review`), auditing code
  against the spec (Q1–Q8) and refactoring (F1–F9).
- [**commits.md**](conventions/commits.md) — Conventional Commits, branches, squash-merged PRs.

## Audits

- [**pipelines-audit.md**](audits/pipelines-audit.md), [**tests-audit.md**](audits/tests-audit.md):
  the open items left from the audits of the pipelines and the tests, made by the method in
  [review.md §5](conventions/review.md#5-auditing-existing-code). Each file names the commit that holds its full report.

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
  path over a two-site geometry: the Raleigh-Durham transmitter received at the Duke Receiver,
  in an X-band and an S-band variant.
- [`spec/scenario-003-tracking.md`](../spec/scenario-003-tracking.md) — CFAR
  detection, data association and Kalman tracking stacked on scenario 001's S1 variant, in two
  waveform variants. §14 records the things building it changed.
- [`spec/scenario-004-keystone.md`](../spec/scenario-004-keystone.md) — Fast sounding rocket: keystone, polynomial-phase motion compensation, datachip and fine estimation.
  *Specified, not implemented.*
- [`spec/scenario-005-leakage-cancellation.md`](../spec/scenario-005-leakage-cancellation.md) — Transmitter leakage and near-range clutter in a continuous-wave radar.
  *Specified, not implemented.*
- [`spec/scenario-006-parallel-approach-doa.md`](../spec/scenario-006-parallel-approach-doa.md) — Two aircraft on closely spaced parallel approaches in one range-Doppler cell: spatial super-resolution (MUSIC, ESPRIT, IAA).
  *Specified, not implemented.*
- [`spec/scenario-007-interference.md`](../spec/scenario-007-interference.md) — Strong RF interference against a weak target: SLC, MVDR, LCMV, mainlobe suppression.
  *Specified, not implemented.*
- [`spec/scenario-008-stap.md`](../spec/scenario-008-stap.md) — Airborne clutter ridge: space-time adaptive processing with RASPNet and MCARM radar parameters.
  *Specified, not implemented.*
- [`spec/scenario-009-receiver-chain.md`](../spec/scenario-009-receiver-chain.md) — Receiver chain: front end, DDC and polyphase filtering, hardware impairments, pilot-tone calibration.
  *Specified, not implemented.*
- [`spec/scenario-010-self-calibration.md`](../spec/scenario-010-self-calibration.md) — Array perturbations: self-calibration, self-localisation and robust beamforming.
  *Specified, not implemented.*
- [`spec/scenario-011-terminal-area-tracking.md`](../spec/scenario-011-terminal-area-tracking.md) — Dense terminal-area traffic: crossings, clutter, manoeuvres and closely spaced aircraft (IMM, JPDA, MHT, PMBM).
  *Specified, not implemented.*
- [`spec/tracker-001.md`](../spec/tracker-001.md) — tracking architecture: one tracker loop
  with interchangeable KF, EKF and UKF filters, and how `core/tracking/` meets the pipelines.

## Scenario companions

Visual, orienting walk-throughs for running a scenario for the first time — the geometry, the
pipeline, the pictures and the knobs. They are **not normative**; the `spec/` documents above are.

- [**scenario 001 — X-band**](scenarios/scenario-001-xband.md) — one site, three waveforms, and
  why the three sample rates differ.
- [**scenario 002 — bistatic**](scenarios/scenario-002-bistatic.md) — two sites, the bistatic
  angle, and what stayed identical in the code.
- [**scenario 003 — tracking**](scenarios/scenario-003-tracking.md) — the detection pipeline, the
  track lifecycle, and three things that look like bugs and are not.
- [**scenario 004 — keystone**](scenarios/scenario-004-keystone.md) — fast sounding rocket: keystone, polynomial-phase motion compensation, datachip and fine estimation. Not runnable yet.
- [**scenario 005 — leakage cancellation**](scenarios/scenario-005-leakage-cancellation.md) — a continuous-wave radar hears its own transmitter; least-squares cancellation removes the leakage and near-range clutter before range-Doppler processing. Not runnable yet.
- [**scenario 006 — parallel-approach doa**](scenarios/scenario-006-parallel-approach-doa.md) — two aircraft on closely spaced parallel approaches in one range-Doppler cell: spatial super-resolution (MUSIC, ESPRIT, IAA). Not runnable yet.
- [**scenario 007 — strong interferers**](scenarios/scenario-007-interference.md) — strong RF interference against a weak target: SLC, MVDR, LCMV, mainlobe suppression. Not runnable yet.
- [**scenario 008 — stap**](scenarios/scenario-008-stap.md) — airborne clutter ridge: space-time adaptive processing with RASPNet and MCARM radar parameters. Not runnable yet.
- [**scenario 009 — receiver chain**](scenarios/scenario-009-receiver-chain.md) — receiver chain: front end, DDC and polyphase filtering, hardware impairments, pilot-tone calibration. Not runnable yet.
- [**scenario 010 — self calibration**](scenarios/scenario-010-self-calibration.md) — array perturbations: self-calibration, self-localisation and robust beamforming. Not runnable yet.
- [**scenario 011 — terminal-area tracking**](scenarios/scenario-011-terminal-area-tracking.md) — dense civil traffic around Raleigh-Durham: a holding stack, crossings, turns, birds and rain clutter; IMM, JPDA, MHT and PMBM compared. Not runnable yet.

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
  analytic ground truth over golden data, fixtures and seeding, property-based tests, markers,
  and radar-domain verification: which physics and statistics tests each pipeline stage needs.
- [**review.md**](conventions/review.md) — doing and writing a PR review: where it lives, the order
  the work is done in, its layout, how a finding and a check are phrased, the voice, and what to
  review against.
- [**audit.md**](conventions/audit.md) — auditing existing code against the spec (Q1–Q8), the
  changes an audit may make (C1–C4), and auditing and pruning the test suite.

## Audits

- [**pipelines-audit.md**](audits/pipelines-audit.md), [**tests-audit.md**](audits/tests-audit.md):
  the reports for the pipelines and the tests, made by the method in [audit.md](conventions/audit.md).

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

## Library guides

API documentation and the intern onboarding tutorials land here as the modules are built.

- [**tracking**](tracking/README.md) — what is in `core/tracking/`, the order one scan runs in,
  a short example, choosing the process noise, and the tracker's limitations.

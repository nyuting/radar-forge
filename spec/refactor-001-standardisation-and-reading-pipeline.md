# Refactor 001 — Repository standardisation, specification overhaul and the reading pipeline

Status: **specified, not implemented**. A cross-cutting refactor rather than a vertical slice:
it touches dependencies, naming, scenario configuration, every document in `spec/`, the test
suite, and adds a new derived-output pipeline under `tools/`.

Unlike the scenario specs, this one has no new physics. Every change is either a rename, a
restructure, or a new piece of developer tooling. That makes it unusually safe to do in parallel
and unusually easy to do carelessly — a rename that misses one call site is a broken import, and
a geographic scrub that misses one coordinate is a test that still passes for the wrong reason.

§8 records where the request's premises did not match the repository as it stands. Where §8
contradicts the body, §8 wins.

## 1. Codebase cleanup and dependency pruning

### 1.1 Dependency cleanup

~~Remove `hypothesis` from `pyproject.toml`, the test configuration, and the environment lock.~~

> **Withdrawn — see §8.1.** `hypothesis` is imported by `tests/core/test_detection.py`, where it
> provides property-based coverage of the CFAR detectors. The premise that it is unutilised is
> false, so the dependency stays. No work item remains under §1.1.

### 1.2 Attribution removal

Remove the line beginning *"Attribution note: the FMCW Radar Target Simulator …"* from `README.md`,
`spec/starter.md`, and any other document or file header carrying it.

| Requirement | Target |
| :--- | :--- |
| R1.2.1 | No occurrence of `Attribution note` remains in any tracked `.md`, `.py` or `.toml` file |
| R1.2.2 | Surrounding prose still reads correctly with the line removed — no dangling reference |

## 2. Naming, terminology and API conventions

### 2.1 Terminology and physical units

Enforce explicit unit suffixes across code, `*.toml` configuration, and specifications.

| From | To | Scope |
| :--- | :--- | :--- |
| `chirp_time`, `chirp_time_s` | `chirp_duration_s` | Code, config, specs, `CLAUDE.md` |
| bare `velocity` | `velocity_mps` | Identifiers only; prose may say "velocity" |

Both renames must stay consistent with the unit rule already enforced by
`scripts/check_conventions.py` (rule R5) and documented in `docs/conventions/style.md`.

### 2.2 Expressive function and method naming

Eliminate the generic simulation call signature `radar.simulate(scene)`. Replace it with methods
that name the domain of the signal they return:

```
radar.simulate(scene)          ->  radar.simulate_baseband(scene)
                                   radar.simulate_iq(scene)
```

or a single explicit parameter-driven variant, provided the parameter names the output domain.

### 2.3 API and code-quality audit

Audit `RangeDopplerProduct` and its neighbours in `pipelines/scenarios.py`. Re-evaluate every
variable, function and class name against PEP 8 and against standard radar signal-processing
nomenclature. Record any rename in the decisions section rather than applying it silently.

## 3. Scenario and configuration restructuring

### 3.1 TOML layout

In `scenarios/scenario_002_bistatic_xband.toml` and the S-band equivalent, move `[receiver]` so
that it follows `[[burst]]` — or, preferably, rename it to `[receiver_site]` and place it directly
after `[transmitter_site]`, which is the layout `scenario_003_tracking.toml` already uses.

Current order is `[scenario] [radar] [transmitter_site] [receiver] [target] [trajectory] [[burst]]`.

### 3.2 Scenario 003 state model

In `scenarios/scenario_003_tracking.toml` and `scenario_003_tracking_dual_prf.toml`, restore the
missing `enu_2d` parameters alongside `range_1d` under `state_model`. Both files currently pin
`state_model = "range_1d"` and carry only that model's `sigma_*` values, even though
`core/tracking.py` ships `range_1d`, `enu_2d` and `enu_3d`.

## 4. Specification overhaul and documentation architecture

Adopt **progressive disclosure** across `spec/` and a new `docs/scenarios/`.

### 4.1 Standard structural scaffolding

Every scenario specification follows one flow:

```
Status & Purpose
      |
      v
Overview & Geometry
      |
      v
Parameter Matrix
      |
      v
Core Decisions
      |
      v
Build Sequence
      |
      v
Acceptance Criteria
```

### 4.2 Document-specific restructuring

**`spec/starter.md`**
- Promote *Repo Layout* to section 2, so a reader maps the tree before meeting external
  dependencies.
- Split the monolithic parameter and configuration tables into smaller categorised matrices.

**`spec/structure.md`**
- Open with the architectural philosophy, stated up front:

  > `radar-forge` is built on a lightweight core with optional extras. It prioritises
  > zero-friction imports, clear educational abstractions, and modular backend pluggability.
  >
  > **Core principles**
  > - **Lightweight core** — `radar_forge.core` and `radar_forge.array` rely only on NumPy and SciPy.
  > - **Zero-breakage imports** — `import radar_forge` always succeeds.
  > - **Unified backend contract** — one `raytracing.Scene` powers all backends.
  > - **Clean licensing and attribution** — copyleft or unlicensed code is never vendored.

- Move **Part B** before **Part A**.
- Add a top-level summary table to Part A with jump links to the detailed subsections.
- Update every cross-reference to include Scenario 3.

**`spec/scenario-001-*.md`**
- Add a comparative table of the key differences across the three waveforms.

**`spec/scenario-002-*.md`**
- Add a top-level ASCII geometry diagram:

```
               [ Target Aircraft ]
               /                 \
        R_r (Receive)       R_t (Transmit)
             /                     \
            v                       v
       [ Duke Receiver ] <────────> [ Raleigh-Durham Airport ]
                             Baseline
                           (23.726 km)
```

### 4.3 Content transformation rules

| Rule | From | To |
| :--- | :--- | :--- |
| Parameter matrices | Dense prose descriptions | Tables of physical constraints, waveform parameters, radar bounds |
| Requirements vs. notes | Rationale interleaved with requirements | Theory, history and trade-offs isolated from functional requirements |
| Mathematical contracts | Implied shapes | Explicit tensor dimensions, coordinate frames, type signatures, I/O schemas |
| Workflows and diagrams | Text-heavy sequence prose | ASCII flowcharts for DSP pipelines and track lifecycles |
| Acceptance criteria | Qualitative goals | Verification matrices with numerical bounds, error margins, thresholds |
| Companion docs | — | One `docs/scenarios/*.md` per scenario, carrying the visual diagrams |

The companion documents exist for external collaborators and research partners running a scenario
for the first time; they are visual and orienting, not normative.

## 5. Implementation verification and test-suite integrity

### 5.1 Source reference comparison

Cross-reference each implemented function against the upstream references catalogued in
`spec/structure.md` Part A. Where an upstream implementation is clearer, more accurate, more
concise, faster, or better documented, adopt it — subject to D4 (RadarBook material is written
from the published equations, never vendored) and to the licensing principle in §4.2.

### 5.2 Multidisciplinary test audit

Audit the suite for coverage across three domains:

| Domain | What must be validated |
| :--- | :--- |
| Radar physics | Physical models, propagation delays, power budgets, geometry |
| Signal processing | Transforms, windowing, filtering, broadcasting, ambiguity resolution |
| Software engineering | Edge cases, type safety, modularity, error handling |

## 6. Code minimisation and developer ergonomics

### 6.1 Architectural rules for `docs/conventions/style.md`

| Rule | Statement |
| :--- | :--- |
| Explicit unit suffixes | Every physical variable and argument declares its unit in the identifier: `_mps`, `_hz`, `_w`, `_s`, `_rad`, `_m` |
| Explicit public exports | Every module defines `__all__`, so a reader can skip `_private_func` helpers |
| Vector mechanics isolation | Vectorised NumPy/SciPy code stays distinct from validation guard clauses (`if …: raise ValueError`) |

### 6.2 `tools/generate_reading_view.py`

An AST/regex utility that reads `src/radar_forge/` and writes minimised sources to
`radar_forge_reading/`.

| Strip | Preserve |
| :--- | :--- |
| Top-level and inner docstrings | `import` statements |
| Inline comments | Function and method signatures |
| Type annotations (`: NDArray[np.float64]`, `-> float`) | Vector broadcasting and maths |
| Validation guard blocks (`if …: raise ValueError`) | `return` expressions |

The output is a reading aid: what the code *computes*, with the prose, the typing and the
defensive scaffolding removed.

### 6.3 Dual-output reading pipeline

**Git hygiene.** `radar_forge_reading/` is a derived artifact and is never committed:

```gitignore
# Minimized AST reading view
radar_forge_reading/
```

**Offline access.** A developer runs `python tools/generate_reading_view.py` on demand and reads
the output locally, or syncs the folder to a mobile device (Working Copy, Obsidian, iCloud).

**Online access.** `.github/workflows/deploy-reading-view.yml`, triggered on pushes to `main`:

```
push to main
      |
      v
run tools/generate_reading_view.py
      |
      v
deploy radar_forge_reading/ to GitHub Pages
      |
      v
https://<user>.github.io/<repo>/
```

This gives browser and mobile reading without derived files entering the repository tree.

## 7. Open clarifications

### 7.1 Sampling-rate query

**Audit only — change nothing.** Investigate why the three waveforms in scenario 001 use
different sampling rates, and document the finding in that scenario's overview. Do not alter any
sampling rate as part of this refactor.

## 8. Premise audit

Checked against the tree at the time of writing. Four items in the request do not match the
repository:

### 8.1 `hypothesis` is in use — §1.1 is blocked

`tests/core/test_detection.py` imports `given`, `settings` and `strategies`, and the surrounding
comment explains the choice of interval. Removing the dependency deletes property-based coverage
of the CFAR detectors. Resolve before executing §1.1: either keep the dependency, or rewrite
those tests against fixed inputs and accept the loss of coverage.

### 8.2 `docs/conventions.md` does not exist

The conventions live in three files — `docs/conventions/style.md`, `commits.md` and `testing.md`.
§6.1's rules belong in `style.md`. `CLAUDE.md` already states the unit-suffix rule, so §6.1's
first row is a restatement rather than a new rule, and R5 in `scripts/check_conventions.py`
already enforces it.

### 8.3 `radar.simulate(scene)` exists only in `README.md`

There is no `simulate` method anywhere in `src/radar_forge/`. The call appears once, at
`README.md:89`, in an illustrative snippet. §2.2 is therefore a documentation fix, unless the
intent is to *add* the method — which would be new API surface, not a rename.

### 8.4 `chirp_time` is already `chirp_time_s`; bare `velocity` is prose only

Code and configuration already carry the `_s` suffix, and `CLAUDE.md:21` cites `chirp_time_s` as
an exemplar. §2.1's first row is thus `chirp_time_s -> chirp_duration_s`, a readability rename
that also edits `CLAUDE.md`. `README.md:80` is the one true offender, using bare `chirp_time`.
Bare `velocity` appears only inside docstrings and comments in `core/ambiguity.py`; every
identifier is already `*_velocity_mps` or `velocity_mps`.

## 9. Acceptance criteria

| # | Criterion | Verified by |
| :--- | :--- | :--- |
| A1 | `make check` passes | CI, `pre-push` |
| A2 | No tracked file contains `Attribution note` | `grep` |
| A4 | Both scenario-003 TOMLs carry `enu_2d` parameters | Config load test |
| A5 | Every scenario spec follows the §4.1 flow | Review |
| A6 | Each scenario has a `docs/scenarios/` companion | `ls` |
| A7 | `tools/generate_reading_view.py` runs clean over `src/radar_forge/` and its output imports nothing it did not preserve | New test |
| A8 | `radar_forge_reading/` is git-ignored and untracked | `git check-ignore` |
| A9 | Pages deploy succeeds on a push to `main` | Actions run |
| A10 | The scenario-001 sampling-rate finding is documented, and no sampling rate changed | Review, `git diff` |

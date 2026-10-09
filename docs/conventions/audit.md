# Audits

An audit checks code or tests that already exist, where a [review](review.md)
checks a change. Its output is as much a set of findings as a set of diffs. An honest "no change,
here is why" is a valid result for anything audited.

| Report | Covers |
| :--- | :--- |
| [`pipelines-audit.md`](../audits/pipelines-audit.md) | `pipelines/`, `viz/` and `scripts/run_scenario.py`, code against spec |
| [`tests-audit.md`](../audits/tests-audit.md) | The test suite, classified and pruned |

Not yet audited: the tracking package and its tests. That audit waits for the tracker migration,
and is [`tracker-001` §13](../../spec/tracker-001.md#13-migration-from-todays-code) step 9.

## 1. Auditing code against the spec

The reference is the **specified behaviour**, not a rival implementation:

- the governing equation, as written in the scenario spec or cited from its source;
- the numerical bounds, tolerances and acceptance figures the spec fixes;
- the documented contract: array shapes, coordinate frames, sign conventions, units;
- the decisions in `spec/structure.md` (D1–D8) that constrain the implementation.

Each public name gets two verdicts.

- **Conformance.** Does the code compute what the spec says, to the accuracy the spec claims,
  with the contract it documents? A divergence is a finding whichever side is wrong. The code may
  have drifted, or the spec may have been wrong and the code quietly corrected it.
- **Quality.** Is this the best available form? Judge it on its own merits, not only against the
  spec. Absence of a better rival is not a reason to leave a clumsy implementation standing.

Quality is judged on eight dimensions. Q6–Q8 apply to every function, not only to those with a
numerical problem. A misleading name, an argument order that can be swapped silently, or a return
whose shape is undocumented is as much a defect as a wrong constant.

| # | Dimension | Asks |
| :--- | :--- | :--- |
| Q1 | Mathematical accuracy and radar-physics fidelity | Does it compute the right thing, to the claimed accuracy? |
| Q2 | Clarity | Would an intern follow it without the docstring? |
| Q3 | Speed and vectorisation | Is it vectorised? A speed claim carries a measurement, or it is not a finding |
| Q4 | Conciseness | Is anything here that earns nothing? |
| Q5 | Documentation | Docstring, shapes, units, references |
| Q6 | Naming | Does the name say what it does, in radar terms and PEP 8, with units? |
| Q7 | Inputs | Argument order, keyword-only where it matters, sensible defaults, `ArrayLike` in |
| Q8 | Outputs | Type, shape, units, and a contract a caller can rely on |

## 2. Changes an audit makes

| Rule | Statement |
| :--- | :--- |
| C1 | No change in behaviour without a test that fails against the old code. Run it red before keeping it |
| C2 | No tolerance is weakened. A number that moves is re-derived from first principles and the derivation recorded |
| C3 | Licensing follows `spec/structure.md` B.2 rule 4 and D4: copyleft or unlicensed code is never vendored |
| C4 | Where the spec is the wrong party, the spec is corrected, and the correction recorded in both the spec and the audit |

## 3. Auditing tests

Each collected test is classified **Sound**, **Redundant**, **Weak** or **Wrong**, and acted on:
Redundant ones are deleted, Weak ones rewritten, Wrong ones fixed. The classes, the floor that
pruning may not go below and the checks it must pass are
[`testing.md` §10.10](testing.md#1010-pruning-a-suite-review). Judge each test's
setup against [§10.1](testing.md#101-what-earns-a-place-review) item 5 as well:
a test can pass and still describe a scenario no radar would meet.

## 4. Writing the report

One file per area, in this directory. Each has:

1. **Scope**: what was audited, at which commit, and what was left out and why.
2. **A verdict key** and a **findings index**: one linked row per finding, so a reader jumps to
   the detail ([style.md §12](style.md#12-summary-tables-link-to-their-sections-hook-r7)).
3. **One verdict table per module or test file.**
4. **The findings in detail**: what was wrong, what it was hiding, and the evidence. For code
   that means the red test (C1) or the measurement (Q3). For tests it means the counterexample.
5. **A result**: for a test audit, counts before and after per file, the coverage comparison and
   the mutation table.

When the audited code later moves, add a dated **Status** note at the top, saying what changed.
Do not rewrite the findings as they were recorded.

This method was first set out in `spec/refactor-002-spec-first-audit.md`, which was retired once
its audits had landed. Git keeps the file.

# Tests audit — open items

**Scope:** the test suite, classified and pruned by the [audit method](../conventions/audit.md#3-auditing-tests)
and [testing.md §10.10](../conventions/testing.md#1010-pruning-a-suite-review). Tracking tests are
not audited yet.
**Last checked:** 2026-11-03, at `61e63a9`.

This file lists only what is still to do. When you fix an item, delete it in the same commit. The
October 2026 pruning took the suite from 1563 tests to 1297 without losing any line or branch
coverage. Its full report, with every verdict, the coverage comparison and the mutation tables, is
in git:

```sh
git show 7bfb654:docs/audits/tests-audit.md
```

The patterns behind the weak tests it found are now in
[testing.md §10.11](../conventions/testing.md#1011-ways-a-test-passes-for-the-wrong-reason-review).

## Summary

| # | Problem | Size |
| :--- | :--- | :--- |
| [T1](#t1-resample-has-an-undocumented-raises) | `resample` has an undocumented `Raises` | small, `src/` |
| [T2](#t2-scenario-001-has-no-link-budget-test) | Scenario 001 has no link-budget test | medium, new test |
| [T3](#t3-the-tracking-tests-are-not-audited) | The tracking tests are not audited | waits on tracker-001 step 7 |

## T1. `resample` has an undocumented `Raises`

`resample(trajectory, [t])` with a single time raises `ValueError: a trajectory needs at least two
fixes; got 1.`, from `Trajectory.__post_init__`. The docstring (`pipelines/trajectories.py:293`)
lists only the out-of-span error. No shipped path hits this, because `iterate_frames` pads the time
grid.

**Fix:** either list the error under `Raises` and test it, or let `resample` return a one-fix
track. The first is simpler.

## T2. Scenario 001 has no link-budget test

[testing.md §10.8](../conventions/testing.md#108-end-to-end) asks each pipeline for a link-budget
test and one Pd point. `tests/pipelines/test_scenario_001.py` checks only where peaks land.
Scenario 002 is exempt, because its D8 says it does not assert power.

**Fix:** add the link-budget test: a point target through the whole chain, whose processed SNR
equals the radar equation plus integration gain, minus window and processing losses.

## T3. The tracking tests are not audited

`tests/core/tracking/**`, `tests/core/test_tracking.py`, `tests/pipelines/test_tracking.py` and
`tests/pipelines/test_scenario_003.py` were left out, because the tracker migration replaces most
of the code they test.

**Waits on:** [tracker-001 §13](../../spec/tracker-001.md#13-migration-from-todays-code) step 7.
Step 9 then audits them and records the result here.

## Don't delete

The October pruning kept these on purpose. Another pruning pass could mistake them for redundant.

- **`test_rejects_power_that_is_not_finite[inf]`** (`tests/core/test_detection.py`). The guard is
  `np.isfinite`. If it regresses to `np.isnan`, only the `inf` case fails.
- **`test_a_ring_has_less_cfar_loss_than_a_line_of_the_same_reach`** (`tests/core/test_detection.py`).
  `spec/scenario-003-tracking.md` §13.2 cites it by name as the evidence for its reach claim.

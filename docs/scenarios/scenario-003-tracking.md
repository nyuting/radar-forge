# Scenario 003 — Detection, association and tracking

A visual companion for someone running this scenario for the first time. **Not normative**: where
this disagrees with [`spec/scenario-003-tracking.md`](../../spec/scenario-003-tracking.md), the
specification wins.

## What it does

Scenario 001 stops at a peak on a range-Doppler map. Reading a peak off a map is not detection, and
one peak per frame is not a track. This scenario adds the three stages that turn one into the
other: **CFAR detection → data association → Kalman tracking**.

Two variants ship, and the contrast between them is the point:

| | `scenario_003_tracking.toml` | `scenario_003_tracking_dual_prf.toml` |
| :--- | :--- | :--- |
| Waveform | scenario 001's **S1** low-PRF FMCW | scenario 001's **S3** dual-PRF FMCW |
| Doppler folds at | ±7.6478 m/s | resolved in the waveform, ±191 m/s |
| Velocity unfolding | `track_aided` — the tracker has to do it | `none` — it arrives already unfolded |
| Shows you | **why the problem is hard** | **that the tracker is not what makes it hard** |

## Run it

```bash
uv run python scripts/run_scenario.py scenarios/scenario_003_tracking.toml --out out/t1
uv run python scripts/run_scenario.py scenarios/scenario_003_tracking_dual_prf.toml --out out/t3
```

Both start at `start_time_s = 663.0`, not 0. That is the earliest 120 s window that still crosses a
Doppler fold boundary — the thing the scenario exists to show — while holding the range rate steady
enough either side of the crossing that the bootstrap can resolve it.

Outputs, on top of scenario 001's: `track_{frame:05d}.png` (range against time), `track.mp4`,
`detections.csv`, `tracks.csv`, and `rd_*.png` now carrying detection markers and the track's gate.

## The pipeline

```
  RangeDopplerProduct  (rd_map, range_axis_m, velocity_axis_mps)
            |
            v
  core/detection.py   cfar_detect (CA, along range)  +  cfar_valid_mask
            |             pfa = 1e-5, n_train = 16, n_guard = 4 -> alpha = 11.417 dB
            |             245,760 of 256,000 cells carry a full reference window
            v
            boolean detection map, (256, 1000)
            |
            v
  roll the map so its edge falls on the quietest Doppler row, then cluster
            |    (the Doppler axis wraps and cluster_detections does not)
            v
  keep only the strongest detection per range cell
            |    (one point target cannot give two returns at one range)
            v
  fractional cell index -> metres and m/s via the dsp bin-centre helpers
            |
            v
  velocity unfolding:  track_aided  |  oracle (tests)  |  none
            |
            v
      Measurement  z = [range_m, range_rate_mps]
            |
            v
  core/tracking.py   predict -> gate -> assign (GNN) -> update -> initiate/delete
            |
            v
  tracks.csv, detections.csv, track_*.png, rd_*.png
```

## The track lifecycle

```
                       unassociated measurement
                                |
                                v
                        +----------------+
                        |   TENTATIVE    |  dim 1, range only
                        |  seeded from   |  velocity variance = v_max^2
                        |  range alone   |
                        +----------------+
                         |             |
          4 hits in 5    |             |  fewer than 4 hits in 5 frames
          frames         v             v
                  +-------------+   +---------+
          +-----> |  CONFIRMED  |   | DELETED |
          |       +-------------+   +---------+
          |         |    ^   |
          |         |    |   | miss
          |  hit    |    |   v
          |         |    | +----------+   3 consecutive misses
          |         |    | | COASTING | -------------------+
          |         |    | +----------+                    |
          |         |    |   |                             v
          |         |    +---+ hit                  +--------------+
          |         |                               | REACQUIRING  |
          |         v                               | forget rate, |
          |  range-slope sigma < v_span/6           | widen gate   |
          |  (about 10 frames)                      +--------------+
          |         |                                      |
          |         v                                      | hit
          |   dim 1 -> dim 2: velocity measured  <---------+
          |   and unfolded from the prediction
          +--------------------------------------------------

  Gate: d^2 = nu^T S^-1 nu  <=  chi^2(0.99, dim)
        dim 1 -> 6.635   dim 2 -> 9.210   dim 3 -> 11.345   dim 4 -> 13.277
```

The **`measurement_dim` column in `tracks.csv` is the most informative thing the run produces**.
The frame at which it steps from 1 to 2 is the frame the track became unfoldable, and
`velocity_unfolded_mps` is empty until then.

## The knobs, and which ones matter

| Knob | Value | Why |
| :--- | :--- | :--- |
| `pfa` | 1e-5 | Sets the *difficulty*, not a detector detail: about 2.46 false alarms per frame in front of the associator, while a spurious confirmed track is a once-in-400-runs event. Raise it to 1e-4 and that last number goes up ~7000×. |
| `sigma_range_m` | 21.636 m | One range bin over √12. Quantisation-limited, not SNR-limited. |
| `sigma_velocity_mps` | 0.017248 m/s | One velocity bin over √12. **Five orders of magnitude** below the range sigma — that is what a range-Doppler radar looks like: a coarse ranging measurement and an exquisite velocity one. |
| `sigma_accel_mps2` | 5.0 | Not the 2.0 a light aircraft's lateral acceleration suggests. The target is *simulated* from a CSV with 2-3 s fixes and ADS-B noise, so its range rate moves 3.4 m/s between frames and by as much as 11 m/s. The filter tracks the target as simulated, not as flown. |
| `state_model` | `range_1d` | `enu_2d` and `enu_3d` exist and are unit-tested, but they need angles this radar does not measure. |

## Three things that look like bugs and are not

1. **`range_1d`'s transition matrix carries `−T`, not `+T`.** Closing velocity is positive, so a
   closing target's range must *shrink*. The textbook constant-velocity pair conjugated by
   `diag(1, −1)`. The version with `+T` produces a plausible-looking picture in which the filter
   dead-reckons the target the wrong way down the line of sight.
2. **A track spends its first ~10 frames measuring range only.** Track-aided unfolding needs a
   velocity prediction to select a fold, and a new track has none. It bootstraps on a batch
   range-slope estimator instead.
3. **On S1, the fold index is right about 86 % of the time, and no tracker can do better.** The
   simulated target's range rate moves by more than half a fold span in 3.4 % of frames; the
   information simply is not there. Run the dual-PRF variant and it is right every frame — the
   waveform resolves what the tracker cannot.

## Measured, over the default 120-frame window

| Criterion | Asked for | S1 | S3 dual-PRF |
| :--- | :--- | :--- | :--- |
| One confirmed track, stable id | ≥95 % of frames, one id | 98 %, 2 ids | 98 %, **1 id** |
| Range RMSE | < 74.95 m | **17.5 m** | **2.0 m** |
| Range-rate RMSE | < 2 m/s | 5.8 m/s | **0.60 m/s** |
| Fold index correct | every frame | 86 % | **every frame** |

## Where to look next

- The specification, including the seven things building it changed:
  [`spec/scenario-003-tracking.md`](../../spec/scenario-003-tracking.md) §14
- The slice underneath it: [scenario 001](scenario-001-xband.md)
- The bistatic slice, which this one does **not** build on: [scenario 002](scenario-002-bistatic.md)

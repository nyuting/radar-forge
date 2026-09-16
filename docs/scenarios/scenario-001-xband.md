# Scenario 001 — X-band radar at the Duke Receiver

A visual companion for someone running this scenario for the first time. **Not normative**: where
this disagrees with [`spec/scenario-001-xband.md`](../../spec/scenario-001-xband.md), the
specification wins.

## What it does

Takes a real ADS-B-style aircraft track, puts a radar on a rooftop in Durham NC, and produces one
range-Doppler map per second of scenario time — in three waveform variants that each resolve the
range/Doppler ambiguity a different way.

## Run it

```bash
uv run python scripts/run_scenario.py scenarios/scenario_001_fmcw_low_prf.toml --out out/s1
uv run python scripts/run_scenario.py scenarios/scenario_001_pulsed_medium_prf.toml --out out/s2
uv run python scripts/run_scenario.py scenarios/scenario_001_fmcw_dual_prf.toml --out out/s3
```

Each writes `rd_{frame:05d}.png`, `rd.mp4`, `truth.csv` and `metadata.json` under `--out`. The
default window is 120 frames at 1 Hz, so expect 120 pictures.

## The geometry

```
                       [ Light aircraft, 1500 m AMSL ]
                          .        |         .
                        .          |           .
                     R .           | elevation   . R
                      .            | 3.7-15.7 deg  .
                    .              |                 .
      [ Duke Receiver, 36.00250 N, 78.94100 W, 60 m AMSL ]
                  azimuth 166.4-238.6 deg, clockwise from true north
                  slant range 5.33 - 22.01 km over the full track
```

The aircraft flies a circuit over the North Carolina Piedmont for 4 h 35 min (5422 fixes, 2-8 s
apart). The default window is the first 120 s of it, where the target sits at 16.6-18.4 km and
closes or opens at up to 55 m/s.

## The pipeline

```
  data/flight_coordinates.csv          scenarios/scenario_001_*.toml
      (lat, lon, timestamp)                  (site, waveform, target)
               |                                      |
               v                                      v
  pipelines/trajectories.py   ---->  pipelines/scenarios.py  iterate_frames()
      load -> resample 1 Hz -> ENU              |
                                                v
  core/signal.py   line_of_sight_paths -> PropagationPaths -> baseband + noise
                                                |
                                                v
                              IQ cube (n_pulses, n_samples) complex128
                                                |
                                                v
  core/dsp.py   [matched_filter if pulsed] -> range_fft -> doppler_fft
                                                |
                                                v
                     RangeDopplerProduct -> rd_*.png, truth.csv
```

## The three variants, and what each one shows you

| | **S1** low-PRF FMCW | **S2** medium-PRF pulsed | **S3** dual-PRF FMCW |
| :--- | :--- | :--- | :--- |
| File | `scenario_001_fmcw_low_prf.toml` | `scenario_001_pulsed_medium_prf.toml` | `scenario_001_fmcw_dual_prf.toml` |
| PRF | 1.0 kHz | 25 kHz | 5.0 and 6.0 kHz |
| Unambiguous range | 37.474 km | **5.996 km** | 29.979 / 24.983 km |
| Unambiguous velocity | **±7.6478 m/s** | ±191.194 m/s | ±191.194 m/s after unfolding |
| Sample rate | 1.0 MHz | 2.5 MHz | 4.0 MHz |
| What you will see | The target sits at its true range; the Doppler peak jumps around, because it is folded about five times | The target's velocity is right; its *range* is wrong, wrapped into a 6 km window | Both right — the coprime PRF pair recovers the true velocity |

Range resolution is 74.9481 m in all three: same carrier, same 2 MHz bandwidth.

### Why the sample rates differ

Not resolution — all three resolve to the same 74.95 m. The two families sample different things:

- **Pulsed** samples the 2 MHz signal itself, so `fs >= B`. 2.5 MHz is 1.25× oversampled.
- **FMCW** samples the *dechirped beat*, whose frequency is `2·R·α/c` with `α = B/chirp_duration_s`.
  The constraint is on the **slope**, not the bandwidth. S3's chirps are 5-6× shorter than S1's at
  the same bandwidth, so the slope is 5-6× steeper, so `fs` has to be 4 MHz rather than 1 MHz.

S1 sampling at 0.5× its own RF bandwidth is the giveaway: for a pulsed waveform that would be
aliased and wrong, but after dechirping there is nothing near 2 MHz left to alias.

## Reading a range-Doppler map

```
  velocity (m/s), fftshift-ed, ZERO IN THE MIDDLE, closing POSITIVE
        +v_ua  +-------------------------------------+
               |                                     |
             0 |            *  <- target peak        |
               |                                     |
        -v_ua  +-------------------------------------+
               0                                  R_ua
                     range (m), UNSHIFTED, zero at bin 0
```

The truth marker overlaid on each frame is the **true, unfolded** range and velocity, straight
from `truth.csv`. The gap between it and the peak is not a bug — in S1 and S2 it is the whole
point of the picture.

## What the input data cannot tell you

- **No altitude** in the CSV. It is a scenario parameter, held at 1500 m. Elevation angles and the
  ground-to-slant correction are approximate (3.86 % at the near end, 0.20 % at the far end).
- **Fixes are 2-8 s apart.** Truth velocity is a central difference of an interpolated range, so it
  is a mean rate over several seconds, not an instantaneous Doppler.
- **Possibly more than one aircraft.** The track begins and ends within 416 m of the same point.
  The default 120 s window is safely one continuous flight; the full track is opt-in.

## Where to look next

- The specification: [`spec/scenario-001-xband.md`](../../spec/scenario-001-xband.md)
- The next slice, two sites instead of one: [scenario 002](scenario-002-bistatic.md)
- The slice that turns peaks into tracks: [scenario 003](scenario-003-tracking.md)

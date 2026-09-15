# Scenario 001 — X-band radar at DSO National Laboratories

Status: **implemented**. The first end-to-end simulation in `radar-forge`.

## 1. Purpose

`spec/structure.md` describes about thirty modules. This document specifies the **first vertical
slice** through them: the thinnest path that turns a real aircraft trajectory into baseband IQ and
a range-Doppler map that advances one frame per second.

Two reasons to build a slice before breadth:

1. It produces a picture. A radar library that cannot yet draw an RD map is hard to review, hard to
   teach from, and hard to trust.
2. It tests the module boundaries in `spec/structure.md` against a real scenario. Two of them did
   not survive; see §8.

A second workstream is planning the remaining modules in parallel. §9 records the boundary.

## 2. Scenario

A ground-based X-band radar on the roof of DSO National Laboratories observes a light aircraft
flying a circuit over western Singapore.

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Radar site | 1.29150 °N, 103.78710 °E | DSO National Laboratories, 12 Science Park Drive |
| Antenna height | 60 m AMSL | Assumed rooftop height; configurable |
| Carrier `f0_hz` | 9.8 GHz | Specified. λ = 30.591 mm |
| Bandwidth `bandwidth_hz` | 2.0 MHz | Specified. Range resolution c/2B = **74.95 m** |
| Target track | `data/flight_coordinates.csv` | Real ADS-B-style track, 5422 fixes |
| Track window | 2026-09-03T00:17:56Z → 04:53:21Z | 4 h 35 min, irregular 2–8 s sampling |
| Target altitude | 1500 m AMSL, **constant** | The CSV has no altitude column; see §3 |
| Target RCS | 10 m² (10 dBsm) | Light aircraft, Swerling 0 (non-fluctuating) |
| Frame cadence | 1 Hz | Specified |

Derived geometry, computed from the CSV against the radar site:

| Quantity | Value |
| :--- | :--- |
| Slant range, minimum | **8.39 km** |
| Slant range, maximum | **17.87 km** |
| Ground range span | 8.26 – 17.81 km |

The target never enters the near field and never leaves 18 km. Any waveform for this scenario needs
roughly 20 km of unambiguous range to avoid folding, or must fold predictably.

## 3. Known limitations of the input data

Stated up front, because a teaching repository that hides its assumptions teaches the wrong thing.

- **No altitude.** The CSV carries `timestamp,lat,lon` only. Altitude is a constant scenario
  parameter, default 1500 m AMSL. Elevation angle and the ground-range-to-slant-range correction
  are therefore approximate. The correction is small here (8.26 → 8.39 km, 1.6 %) and shrinks with
  range.
- **Irregular sampling.** Fixes arrive every 2–8 s. Positions are linearly interpolated onto the
  1 Hz frame grid; radial velocity is a central difference of slant range on that grid and is
  therefore **smoothed over several seconds**. It is a good estimate of mean radial velocity across
  a frame, not of instantaneous Doppler.
- **No callsign, heading, or speed.** Nothing cross-checks the derived velocity. Tests must use
  synthetic trajectories with analytic ground truth (`docs/conventions/testing.md` §3); the real
  CSV is exercised for loading and plumbing, not for numerical correctness.
- **Possibly more than one aircraft.** The track starts and ends within ~300 m of the same point
  over 4 h 35 min, consistent with a loiter or circuit, but the file is not guaranteed to be a
  single continuous flight. The scenario treats it as one target and defaults to a **120 s window**,
  which is safely continuous. The full track is opt-in.

## 4. The ambiguity problem, and why there are three variants

At f_c = 9.8 GHz, λ = 30.59 mm, so an 80 m/s closing target produces 5.2 kHz of Doppler.
Unambiguous Doppler at that speed needs PRF ≳ 10.5 kHz, which caps unambiguous range at 14.3 km —
below the 17.9 km the target reaches. A 200 m/s target makes it worse.

**At B = 2 MHz and X-band, this scenario cannot have both range and Doppler unambiguous.** That is
not a defect in the specification; it is the central design tension of pulse-Doppler radar. So the
scenario ships in three variants, one per way of resolving it.

All three share `f0_hz = 9.8e9`, `bandwidth_hz = 2.0e6`, 74.95 m range resolution, a 1 Hz frame
cadence, and a short CPI within each 1 s frame (not a contiguous 1 s CPI — see §5).

### S1 — `fmcw-low-prf`: Doppler folds

| | |
| :--- | :--- |
| Waveform | FMCW sawtooth, `chirp_time_s` = 1.0 ms, sweep rate α = 2.000 GHz/s |
| Receive | Deramp (stretch), `sample_rate_hz` = 1.0 MHz, 1000 samples/chirp |
| PRF | 1.0 kHz (100 % duty) |
| Unambiguous range | **37.47 km** at f_s/2 — covers the whole track |
| Unambiguous velocity | **±7.65 m/s** — an 80 m/s target folds ~5× |
| CPI | 256 chirps = 256.0 ms |
| Velocity resolution | 0.0597 m/s |
| IQ cube | `(256, 1000)` complex128 ≈ 4.1 MB/frame |

Build this one first. The target is unambiguous in range and heavily folded in Doppler; the
truth-label CSV carries the true radial velocity and the RD map shows the aliased one.

### S2 — `pulsed-medium-prf`: range folds

| | |
| :--- | :--- |
| Waveform | LFM pulse, `pulse_width_s` = 10 µs, B = 2 MHz (time-bandwidth product 20) |
| Receive | Matched filter, `sample_rate_hz` = 2.5 MHz, 100 samples/PRI |
| PRF | 25 kHz, PRI 40 µs |
| Unambiguous range | **5.996 km** — the target at 8.4–17.9 km folds 1–2× |
| Unambiguous velocity | **±191.2 m/s** — unfolded |
| CPI | 256 pulses = 10.24 ms |
| IQ cube | `(256, 100)` — 0.4 MB/frame |

The textbook pulse-Doppler case and the exact mirror of S1. The same target appears at a *different*
apparent range, which is the lesson.

### S3 — `fmcw-dual-prf`: the ambiguity is resolved

| | |
| :--- | :--- |
| Waveform | FMCW, alternating blocks of 128 chirps at two chirp rates |
| Burst A | `chirp_time_s` = 200.0 µs, α = 10.000 GHz/s, PRF 5.0 kHz, 800 samples |
| Burst B | `chirp_time_s` = 166.7 µs, α = 12.000 GHz/s, PRF 6.0 kHz, 667 samples |
| Receive | Deramp, `sample_rate_hz` = 4.0 MHz |
| Unambiguous range | 29.98 km (A) / 24.98 km (B) — both cover the track |
| Per-burst unambiguous velocity | ±38.24 m/s (A) / ±45.89 m/s (B) |
| **After unfolding** | **±191.2 m/s** — extension ×5 from the 5:6 PRF ratio |
| CPI | 25.60 ms + 21.33 ms ≈ 47 ms |

The PRF ratio 5:6 is coprime, so the pair of folded velocity estimates identifies the true velocity
uniquely up to 5 × 38.24 = 191.2 m/s — matching S2's limit by a completely different mechanism.
This is the only variant needing an algorithm the others do not: `unfold_doppler_dual_prf`.

> Every number in §4 was derived from the parameters above and confirmed numerically. The
> implementation must nonetheless re-derive each unambiguous limit from `core/waveforms.py` and
> assert it in a test, rather than trusting this table. A spec table is documentation; a test is a
> guarantee.

## 5. Frame and CPI structure

A frame is emitted every 1.0 s of scenario time. Within each frame the radar transmits a **short
coherent processing interval** — 256 ms (S1), 10.24 ms (S2), 47 ms (S3) — and is idle for the
remainder.

This is deliberate, and not merely a compute saving:

- **Range migration.** At 80 m/s radial, a target moves 20.5 m during S1's 256 ms CPI — a quarter of
  a 74.95 m range bin, so migration is negligible and range-Doppler coupling can be ignored. Over a
  contiguous 1 s CPI it would move 80 m, more than a full bin, and the peak would smear.
- **Bounded memory.** The full track is ~16,500 frames; S1 at 100 % duty over 1 s would be 16 MB per
  frame.

The neglect of range migration is an approximation and must be documented in
`core/signal.py`'s `Notes`, with this bound.

## 6. Outputs

Written under `--out <dir>` by `scripts/run_scenario.py`:

| File | Content |
| :--- | :--- |
| `iq_{frame:05d}.npz` | Baseband IQ cube, complex128, `(n_pulses, n_samples)`; slow time on axis 0 |
| `rd_{frame:05d}.png` | Rendered range-Doppler map, dB scale, with the truth marker overlaid |
| `rd.mp4` | The PNG frames assembled at 1 fps (GIF fallback) |
| `truth.csv` | Per frame: `frame, time_s, range_m, radial_velocity_mps, azimuth_deg, elevation_deg` |
| `metadata.json` | Resolved scenario parameters, axes, git commit, seed |

`truth.csv` carries the **true, unfolded** quantities in every variant. The whole point of the
scenario is the discrepancy between it and the map. Its field names and units are chosen to be a
subset of the COCO-parity label schema in `spec/structure.md` D5, so the exporter can later consume
it unchanged.

IQ is `.npz`, already covered by `.gitignore`; `out/` is added to `.gitignore` too. Per
`CLAUDE.md`, the generating script is committed, the arrays are not.

## 7. Module build order

Each module lands with its tests in the same commit, and `make check` passes before the next
begins.

| Step | Module | Notes |
| :--- | :--- | :--- |
| 0 | `core/constants.py` *(edit)* | Add `WGS84_SEMI_MAJOR_AXIS_M`, `WGS84_FLATTENING`. **Load-bearing**: rule R4 in `scripts/check_conventions.py` rejects the literal `6_378_137.0` outside this module, so geodesy cannot be written first. R4's literal map must also be corrected — it currently suggests `EARTH_RADIUS_M` for that literal, which is wrong for an ellipsoid semi-major axis. |
| 1 | `core/geodesy.py` *(new)* | WGS-84 geodetic ↔ ECEF ↔ local ENU; ENU → range/azimuth/elevation. Azimuth 0° at true north, increasing clockwise, per D5. |
| 2 | `core/radar.py` | Frozen `Transmitter`, `Receiver`, `Radar` dataclasses; `wavelength_m`, `unambiguous_range_m`, `unambiguous_velocity_mps` properties. Validation only, no simulation. |
| 3 | `core/targets.py` | `PointTarget`, constant RCS, **Swerling 0 only**. Swerling 1–4 stay unimplemented and are named as such. |
| 4 | `core/signal.py` | `PropagationPaths` exactly as specified in D1, plus `line_of_sight_paths`, `fmcw_deramp_baseband`, `pulsed_baseband`, `thermal_noise`. Reuses `radar_equation.received_power_w` and `waveforms.lfm_chirp` / `beat_frequency_hz` — **adds no new waveform or link-budget mathematics**. |
| 5 | `core/dsp.py`, `core/windows.py` | **Already built by the parallel workstream** (commit 64f1b62) and consumed, not written, by this scenario: `range_fft`, `doppler_fft`, `range_doppler_map`, `matched_filter`, `mti_filter`, `range_bin_centers_m`, `doppler_bin_centers_mps`, `taper`. See §7.1. |
| 5a | `core/ambiguity.py` *(new)* | `unfold_doppler_dual_prf`, needed only by S3. Kept out of `core/dsp.py` because that module belongs to the other workstream and is already near the 400-line guidance of `docs/conventions/style.md` §9. |
| 6 | `pipelines/trajectories.py` | CSV load, resample to the frame grid, transform to the radar frame. |
| 7 | `pipelines/scenarios.py` | Scenario TOML (`tomllib`, stdlib) → `Scenario`; `iterate_frames` is a **generator**, not a list. |
| 8 | `teaching/plotting.py`, `teaching/scopes/rd_map.py` | `matplotlib` lazily imported; `ImportError` names the `teaching` extra. |
| 9 | `scripts/run_scenario.py` | Thin CLI; all logic stays in `pipelines/`. |
| 10 | `__init__.py` wiring | `import radar_forge` must still succeed with zero extras, so `teaching` is **not** imported at top level. |

### 7.1 Integration seam with `core/dsp.py`

`core/signal.py` produces the IQ cube that `core/dsp.py` consumes. Three conventions must line up,
and all three are the *existing* module's, not this scenario's:

- **Cube layout** is `(n_pulses, n_samples)` — slow time on axis 0, fast time on axis 1 — matching
  the canonical `(n_pulses, n_samples, n_rx)` in `docs/conventions/style.md` §4. Emitting that
  layout means `range_doppler_map`'s default axes just work. Every `dsp` function takes an explicit
  `axis`, so a different layout would cost a keyword argument, not a rework.
- **Doppler sign: closing velocity is positive**, per `spec/structure.md` D5.
  `doppler_bin_centers_mps` returns an `fftshift`-ed axis, zero at `n_bins // 2` and increasing. If
  `core/signal.py` generates the opposite sign, closing targets will appear to open — a bug that
  looks like a plausible picture, so §9's acceptance check must catch it rather than the eye.
- **Range is unshifted, Doppler is shifted.** `range_fft` leaves zero range at bin 0; `doppler_fft`
  centres zero Doppler. `teaching/scopes/rd_map.py` must label axes with `range_bin_centers_m` and
  `doppler_bin_centers_mps` rather than deriving them, since those helpers already encode this
  asymmetry and the unambiguous limits.

Scenario configurations live in `scenarios/scenario_001_{fmcw_low_prf,pulsed_medium_prf,fmcw_dual_prf}.toml`,
sharing `[radar]`, `[target]` and `[trajectory]` blocks and differing only in `[waveform]`.

### Where the input data lives

`data/flight_coordinates.csv`, tracked. Not `src/radar_forge/` (rule R2 tolerates `.csv` there, but
library code stays code); not `tests/data/golden/`, which `docs/conventions/testing.md` §6 reserves
for kilobyte-scale regression fixtures with a regeneration script. A 51-row excerpt is committed to
`tests/data/golden/flight_coordinates_head.csv` so the loader test does not depend on the full file.

## 8. Amendments to `spec/structure.md`

> **Applied.** These amendments are folded into `spec/structure.md`; it is the authoritative tree and decision list. This section records what changed and why.

1. **`core/geodesy.py` is new.** `spec/structure.md` gives WGS-84 ↔ ENU no home. It does not belong
   in `pipelines/trajectories.py`, because `raytracing/scene.py` will need it to place scene
   geometry, nor in `config.py`, which is units and global settings. It is a `core` concern.
2. **`core/propagation.py` is deferred.** This slice is free-space only, via the existing
   `received_power_w`. Atmospheric and rain attenuation are negligible for a clear X-band path at
   18 km, and implementing them here would ship untested code.
3. **`line_of_sight_paths` is the zeroth-order analytic backend.** It produces a `PropagationPaths`
   with `bounce_count == 1` and no multipath. Whoever builds `raytracing/backends/analytic.py`
   should build on it rather than duplicate it — D1's decision that `core/signal.py` owns
   path → baseband for *all* backends is what makes that possible, and this slice is its first
   proof.

The open question recorded in `spec/structure.md` — that the analytic backend is the only one under
test until a GPU backend exists — stands unchanged.

## 9. Acceptance criteria

The scenario is done when, for each of the three variants over the default 120 s window, the
range-Doppler peak falls in the bin predicted analytically from `truth.csv`:

- **S1**: peak range within one 74.95 m bin of the true range; peak velocity matches the true
  velocity **wrapped** into ±7.65 m/s.
- **S2**: peak range matches the true range **wrapped** into 5.996 km; peak velocity matches the
  true velocity directly.
- **S3**: after `unfold_doppler_dual_prf`, peak velocity matches the true velocity **unwrapped**,
  within one bin, across every frame.

Encoded as `tests/pipelines/test_scenario_001.py`, marked `slow`, over a 5-frame window so that
`make check` stays fast.

## 10. Workstream boundary

This scenario claims, and another workstream should not plan:

```
data/flight_coordinates.csv                    scenarios/*.toml
scripts/run_scenario.py                        spec/scenario-001-singapore-xband.md
src/radar_forge/core/{geodesy,radar,targets,signal,ambiguity}.py
src/radar_forge/pipelines/{__init__,trajectories,scenarios}.py
src/radar_forge/teaching/{__init__,plotting}.py
src/radar_forge/teaching/scopes/rd_map.py
```

`src/radar_forge/core/constants.py` and `scripts/check_conventions.py` are **shared**: this slice
makes only the additive edits in step 0, and any other workstream's edits must be additive and
coordinated. Note that `check_conventions.py` reads `constants.__all__` *live*, so adding a constant
immediately begins rejecting its literal everywhere else in the tree — step 0 was verified against
the whole tree for exactly this reason.

`src/radar_forge/core/dsp.py` and `src/radar_forge/core/windows.py` belong to the other workstream
and are **consumed, never edited**, by this scenario.

Everything else in `spec/structure.md` — `array/`, `raytracing/`, `core/{propagation, detection,
clutter, tracking}.py`, `pipelines/{generate, datasets}.py`, `pipelines/exporters/`, and the other
`teaching/scopes/` — is outside this scenario, as are `core/dsp.py` and `core/windows.py`.

## References

.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014.
       §4.4 (LFM), §5.3 (pulse-Doppler ambiguity), §8.3 (matched filtering).
.. [2] G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014, §2.5 (FMCW deramp).
.. [3] L. Harrison and G. Andrews, *Introduction to Radar Using Python and MATLAB*, Artech
       House, 2020, ch. 2, ch. 4.
.. [4] National Imagery and Mapping Agency, *Department of Defense World Geodetic System 1984*,
       NIMA TR8350.2, 3rd ed., 2000.

# Scenario 001 — X-band radar at the Duke Receiver

## 1. Status & Purpose

**Status: implemented.** The first end-to-end simulation in `radar-forge`.

`spec/structure.md` describes about thirty modules. This document specifies the **first vertical
slice** through them: the thinnest path that turns a real aircraft trajectory into baseband IQ and
a range-Doppler map that advances one frame per second.

Two reasons to build a slice before breadth:

1. It produces a picture. A radar library that cannot yet draw an RD map is hard to review, hard to
   teach from, and hard to trust.
2. It tests the module boundaries in `spec/structure.md` against a real scenario. Two of them did
   not survive; see §4.4.

A second workstream planned the remaining modules in parallel. §4.5 records the boundary.

| Section | What it fixes |
| :--- | :--- |
| §2 Overview & Geometry | The site, the track, what the three waveforms are for, and what the input data cannot tell you |
| §3 Parameter Matrix | Every number, per variant, with its mathematical contract |
| §4 Core Decisions | Why there are three variants, why the CPI is short, and what moved in `structure.md` |
| §5 Build Sequence | Module order, and the seam with `core/dsp.py` |
| §6 Acceptance Criteria | The verification matrix, with bounds |

---

## 2. Overview & Geometry

A ground-based X-band radar on the roof of the Duke Receiver observes a light aircraft flying a
circuit over the North Carolina Piedmont.

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

The DSP path this scenario builds, end to end:

```
  data/flight_coordinates.csv          scenarios/scenario_001_*.toml
      (lat, lon, timestamp)                  (site, waveform, target)
               |                                      |
               v                                      |
  pipelines/trajectories.py                           |
      load -> resample to 1 Hz -> WGS-84 -> ENU        |
               |                                      |
               v                                      v
        TargetTrack (range_m, az, el, v_r)  <--  pipelines/scenarios.py
               |                                  iterate_frames()
               v
  core/signal.py   line_of_sight_paths -> PropagationPaths
               |          (delay_s, doppler_hz, amplitude)
               v
      fmcw_deramp_baseband  |  pulsed_baseband     + thermal_noise
               |
               v
        IQ cube (n_pulses, n_samples) complex128
               |
               v
  core/dsp.py    range_fft -> doppler_fft -> range_doppler_map
               |            (+ matched_filter first, if pulsed)
               v
        RangeDopplerProduct (rd_map, range_axis_m, velocity_axis_mps)
               |
               +---> core/ambiguity.py  unfold_doppler_dual_prf   (S3 only)
               |
               v
  teaching/scopes/rd_map.py -> rd_{frame}.png,  truth.csv,  metadata.json
```

### 2.1 The three waveforms at a glance

All three variants share the site, the target, the trajectory, the carrier, the bandwidth and the
frame cadence. They differ only in `[[burst]]`. This is the comparison the scenario exists for:

| | **S1** `fmcw-low-prf` | **S2** `pulsed-medium-prf` | **S3** `fmcw-dual-prf` |
| :--- | :--- | :--- | :--- |
| Waveform | FMCW sawtooth | LFM pulse | FMCW, two coprime bursts |
| Receive chain | deramp (stretch) | matched filter | deramp (stretch) |
| `chirp_duration_s` | 1.0 ms | 10 µs | 200.0 µs / 166.667 µs |
| Sweep rate α | 2.000 GHz/s | — (2 MHz in 10 µs) | 10.000 / 12.000 GHz/s |
| `prf_hz` | 1.0 kHz | 25 kHz | 5.0 kHz / 6.0 kHz |
| `sample_rate_hz` | 1.0 MHz | 2.5 MHz | 4.0 MHz |
| `fs / B` | **0.50** | 1.25 | 2.00 |
| Samples per PRI | 1000 | 100 | 800 / 667 |
| `n_pulses` | 256 | 256 | 128 + 128 |
| CPI | 256.0 ms | 10.24 ms | 25.60 + 21.33 ≈ 47 ms |
| Unambiguous range | 37.474 km | **5.996 km** | 29.979 / 24.983 km |
| Unambiguous velocity | **±7.6478 m/s** | ±191.194 m/s | ±38.239 / ±45.887 m/s |
| Resolved by the waveform | range | velocity | **both** |
| What folds | Doppler, ≈5× | range, 1–4× | nothing, after unfolding |
| Extra algorithm needed | — | — | `unfold_doppler_dual_prf` |
| IQ cube per frame | `(256, 1000)`, 4.1 MB | `(256, 100)`, 0.4 MB | 2 × `(128, 800/667)` |

Range resolution is **74.9481 m** in all three, since all three carry `f0_hz = 9.8e9` and
`bandwidth_hz = 2.0e6`.

> **Why the three sampling rates differ** (refactor-001 §7.1, an audit; no rate was changed).
>
> All three share the same 2 MHz RF bandwidth and therefore the same 74.95 m range resolution, so
> the differing rates are **not** about resolution. The two waveform families sample *different
> quantities*, and Nyquist binds on different things.
>
> **Pulsed (S2) samples the signal itself.** The LFM pulse occupies the full 2 MHz, so the
> constraint is `fs >= B`. At 2.5 MHz it carries 1.25×, a modest oversampling margin ahead of the
> matched filter.
>
> **FMCW (S1, S3) samples the dechirped beat.** Mixing against the transmit ramp collapses the
> 2 MHz sweep into a beat tone whose frequency encodes delay,
> `f_beat = 2·R·α/c` with `α = B / chirp_duration_s`, so the constraint is
> `fs >= 2·α·R_max/c` — a function of the **slope**, not of `B`:
>
> | Variant | α | beat per metre | range at full `fs` | `R_ua` from PRF |
> | :--- | ---: | ---: | ---: | ---: |
> | S1 | 2.00e9 Hz/s | 13.34 Hz/m | 74.9 km | 149.9 km |
> | S3 burst A | 1.00e10 Hz/s | 66.71 Hz/m | 60.0 km | 30.0 km |
> | S3 burst B | 1.20e10 Hz/s | 80.06 Hz/m | 50.0 km | 25.0 km |
>
> S3's chirps are 5–6× shorter than S1's, because the dual-PRF design needs high PRFs; a shorter
> chirp at the same bandwidth is a steeper slope, which is 5–6× more beat per metre, which forces
> `fs` up from 1 MHz to 4 MHz to keep a comparable beat horizon.
>
> **The decisive evidence** is that S1 samples at **0.5× its own RF bandwidth**. For a pulsed
> waveform that is straightforwardly aliased and wrong. For FMCW it is unremarkable: after
> dechirping there is nothing near 2 MHz to alias, and 1 MHz of beat band buys 74.9 km — exactly
> half S1's 149.9 km PRF-unambiguous range, so the *sample rate* is the binding range limit for S1,
> not the PRF. That single number is what makes the beat-bandwidth reading the only coherent one.
>
> `core/dsp.py::beat_frequency_hz` is the implementation of the relation above.

### 2.2 Known limitations of the input data

Stated up front, because a teaching repository that hides its assumptions teaches the wrong thing.
These are properties of the input, not defects in the simulation.

| Limitation | Consequence | Mitigation |
| :--- | :--- | :--- |
| **No altitude.** The CSV carries `timestamp,lat,lon` only. | Elevation angle and the ground-to-slant-range correction are approximate. The correction runs from 3.86 % at the near end of the track (5.128 → 5.326 km) to 0.20 % at the far end (21.966 → 22.010 km). | Altitude is an explicit scenario parameter, default 1500 m AMSL. |
| **Irregular sampling**, 2–8 s between fixes. | Positions are linearly interpolated onto the 1 Hz frame grid; radial velocity is a central difference of slant range on that grid, and is therefore **smoothed over several seconds**. | Truth velocity is a good estimate of the mean radial velocity across a frame, not of instantaneous Doppler, and is documented as such. |
| **No callsign, heading or speed.** | Nothing cross-checks the derived velocity. | Tests use synthetic trajectories with analytic ground truth (`docs/conventions/testing.md` §3); the real CSV is exercised for loading and plumbing, not numerical correctness. |
| **Possibly more than one aircraft.** The track starts and ends within 416 m of the same point over 4 h 35 min. | Consistent with a loiter or circuit, but the file is not guaranteed to be one continuous flight. | The scenario treats it as one target and defaults to a **120 s window**, which is safely continuous. The full track is opt-in. |

---

## 3. Parameter Matrix

### 3.1 Site, target and track

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Radar site | 36.00250 °N, 78.94100 °W | the Duke Receiver, 101 Science Drive, Durham NC |
| Antenna height | 60 m AMSL | Assumed rooftop height; configurable |
| `gain_tx_dbi`, `gain_rx_dbi` | 30.0 dBi each | Specified |
| `noise_figure_db` | 3.0 dB | Specified |
| `transmit_power_w` | 100 W (S1, S3), 1 kW (S2) | S2's short duty cycle needs the peak power |
| Target track | `data/flight_coordinates.csv` | Real ADS-B-style track, 5422 fixes |
| Track window | 2026-09-03T00:17:56Z → 04:53:21Z | 16 525 s = 4 h 35 min, irregular 2–8 s sampling |
| Target altitude | 1500 m AMSL, **constant** | The CSV has no altitude column; see §2.2 |
| Target RCS | 10 m² (10 dBsm) | Light aircraft, Swerling 0 (non-fluctuating) |
| Frame cadence | 1 Hz | Specified |
| Default window | `start_time_s = 0.0`, `duration_s = 120.0` | 120 frames; safely one continuous flight |

### 3.2 Waveform-independent derived quantities

| Parameter | Value |
| :--- | :--- |
| Carrier `f0_hz` | 9.8 GHz → λ = **30.5911 mm** |
| Bandwidth `bandwidth_hz` | 2.0 MHz → range resolution c/2B = **74.9481 m** |
| Thermal noise power | 1.598e-14 W (kT₀·B·F at B = **2 MHz**, F = 3 dB) |

> The bandwidth in that product is the **transmitted** bandwidth, 2.0 MHz, which is what
> `core/radar.py`'s `Radar.noise_power_w` uses as the noise bandwidth of a matched receiver.
> An earlier draft of this row said `B = 1 MHz`, which is S1's *sample rate*; the stated
> 1.598e-14 W was always the 2 MHz figure (1 MHz would give 7.989e-15 W), and only the
> parenthetical was wrong. It has to be the transmitted bandwidth for this quantity to sit
> in §3.2 at all: the three variants sample at 1.0, 2.5 and 4.0 MHz, so a sample-rate-based
> noise power would not be waveform-independent. Corrected in refactor-002 §1.3 (R1.3.4).

### 3.3 Derived geometry, computed from the CSV against the radar site

Over the **full** track:

| Quantity | Value |
| :--- | ---: |
| Slant range | **5.326 – 22.010 km** |
| Ground range | 5.128 – 21.966 km |
| Azimuth (0° at true north, clockwise) | 166.4 – 238.6 ° |
| Elevation | 3.7 – 15.7 ° |

Over the **default 120 s window** (`start_time_s = 0`):

| Quantity | Value |
| :--- | ---: |
| Slant range | 16.64 – 18.36 km |
| Radial velocity (closing positive) | −39.3 – +55.3 m/s |

The target never enters the near field and never leaves 22.1 km. Any waveform for this scenario
needs roughly 25 km of unambiguous range to avoid folding over the whole track, or must fold
predictably.

### 3.4 Per-variant waveform matrix

All three share `f0_hz = 9.8e9`, `bandwidth_hz = 2.0e6`, 74.9481 m range resolution, a 1 Hz frame
cadence, and a short CPI within each 1 s frame (not a contiguous 1 s CPI — see §4.2).

**S1 — `fmcw-low-prf`: Doppler folds.** Build this one first. The target is unambiguous in range
and heavily folded in Doppler; the truth-label CSV carries the true radial velocity and the RD map
shows the aliased one.

| Parameter | Value |
| :--- | :--- |
| Waveform | FMCW sawtooth, `chirp_duration_s` = 1.0 ms, sweep rate α = 2.000 GHz/s |
| Receive | Deramp (stretch), `sample_rate_hz` = 1.0 MHz, 1000 samples/chirp |
| PRF | 1.0 kHz (100 % duty) |
| Unambiguous range | **37.474 km** at f_s/2 — covers the whole track |
| Unambiguous velocity | **±7.6478 m/s** — an 80 m/s target folds ≈5× |
| Fold span | 15.29553 m/s |
| CPI | 256 chirps = 256.0 ms |
| Velocity resolution | 0.0597482 m/s |
| IQ cube | `(256, 1000)` complex128 ≈ 4.1 MB/frame |

**S2 — `pulsed-medium-prf`: range folds.** The textbook pulse-Doppler case and the exact mirror of
S1. The same target appears at a *different* apparent range, which is the lesson.

| Parameter | Value |
| :--- | :--- |
| Waveform | LFM pulse, `chirp_duration_s` = 10 µs, B = 2 MHz (time-bandwidth product 20) |
| Receive | Matched filter, `sample_rate_hz` = 2.5 MHz, 100 samples/PRI |
| PRF | 25 kHz, PRI 40 µs |
| Unambiguous range | **5.996 km** — the target at 5.33–22.01 km folds 0–3× |
| Unambiguous velocity | **±191.194 m/s** — unfolded |
| CPI | 256 pulses = 10.24 ms |
| IQ cube | `(256, 100)` ≈ 0.4 MB/frame |

**S3 — `fmcw-dual-prf`: the ambiguity is resolved.** The only variant needing an algorithm the
others do not: `unfold_doppler_dual_prf`.

| Parameter | Value |
| :--- | :--- |
| Waveform | FMCW, alternating blocks of 128 chirps at two chirp rates |
| Burst A | `chirp_duration_s` = 200.0 µs, α = 10.000 GHz/s, PRF 5.0 kHz, 800 samples |
| Burst B | `chirp_duration_s` = 1/6000 s = 166.667 µs, α = 12.000 GHz/s, PRF 6.0 kHz, 667 samples |
| Receive | Deramp, `sample_rate_hz` = 4.0 MHz |
| Unambiguous range | 29.979 km (A) / 24.983 km (B) — both cover the track |
| Per-burst unambiguous velocity | ±38.2388 m/s (A) / ±45.8866 m/s (B) |
| **After unfolding** | **±191.194 m/s** — extension ×5 from the 5:6 PRF ratio |
| CPI | 25.60 ms + 21.33 ms ≈ 47 ms |

The PRF ratio 5:6 is coprime, so the pair of folded velocity estimates identifies the true velocity
uniquely up to 5 × 38.2388 = 191.194 m/s — matching S2's limit by a completely different mechanism.

> Burst B's `chirp_duration_s` is exactly `1/6000` s, not the rounded 166.7 µs. At 6 kHz PRF the
> rounded value is a duty cycle of 1.0002 and `Transmitter` rejects it: the next chirp would start
> before this one finished.

> Every number in §3 is derived from the parameters above and confirmed numerically. The
> implementation must nonetheless re-derive each unambiguous limit from `core/radar.py` and
> `core/waveforms.py` and assert it in a test, rather than trusting this table. A spec table is
> documentation; a test is a guarantee.

### 3.5 Mathematical contracts

The shapes, frames and conventions every module in §5 must agree on.

| Contract | Statement |
| :--- | :--- |
| IQ cube layout | `(n_pulses, n_samples)`, complex128. Slow time on axis 0, fast time on axis 1 — the canonical `(n_pulses, n_samples, n_rx)` of `docs/conventions/style.md` §4 with the receive axis dropped |
| RD map layout | `(n_doppler_bins, n_range_bins)`, complex128 |
| Range axis | `(n_range_bins,)` float64, **unshifted**: zero range is bin 0 |
| Velocity axis | `(n_doppler_bins,)` float64, **`fftshift`-ed**: zero Doppler at `n_bins // 2` |
| Doppler sign | **Closing velocity is positive**, per `spec/structure.md` D5 |
| Coordinate frame | Local ENU tangent plane at the radar site, WGS-84; azimuth 0° at true north, increasing clockwise, per D5 |
| Angle units | Radians inside `core/`, degrees at the TOML and CSV boundaries, with the unit in the name |
| Type signatures | `NDArray[np.float64]` / `ArrayLike` in, `NDArray` out; `mypy --strict` |

### 3.6 Output schema

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

IQ is `.npz`, already covered by `.gitignore`; `out/` is gitignored too. Per `CLAUDE.md`, the
generating script is committed, the arrays are not.

---

## 4. Core Decisions

### 4.1 Why there are three variants

At f_c = 9.8 GHz, λ = 30.591 mm, so an 80 m/s closing target produces 5.2 kHz of Doppler.
Unambiguous Doppler at that speed needs PRF ≳ 10.5 kHz, which caps unambiguous range at 14.3 km —
below the 22.0 km the target reaches. A 200 m/s target makes it worse.

**At B = 2 MHz and X-band, this scenario cannot have both range and Doppler unambiguous.** That is
not a defect in the specification; it is the central design tension of pulse-Doppler radar. So the
scenario ships in three variants, one per way of resolving it: fold Doppler (S1), fold range (S2),
or spend two coprime PRFs and fold neither (S3).

### 4.2 Why the CPI is short, and the frame idle

A frame is emitted every 1.0 s of scenario time. Within each frame the radar transmits a **short
coherent processing interval** — 256 ms (S1), 10.24 ms (S2), 47 ms (S3) — and is idle for the
remainder. This is deliberate, and not merely a compute saving:

- **Range migration.** At 80 m/s radial, a target moves 20.5 m during S1's 256 ms CPI — a quarter of
  a 74.95 m range bin, so migration is negligible and range-Doppler coupling can be ignored. Over a
  contiguous 1 s CPI it would move 80 m, more than a full bin, and the peak would smear.
- **Bounded memory.** The full track is ≈16 500 frames; S1 at 100 % duty over 1 s would be 16 MB per
  frame.

The neglect of range migration is an approximation and must be documented in `core/signal.py`'s
`Notes`, with this bound.

### 4.3 The integration seam with `core/dsp.py`

`core/signal.py` produces the IQ cube that `core/dsp.py` consumes. Three conventions must line up,
and all three are the *existing* module's, not this scenario's. §3.5 states them; what follows is
why each is load-bearing.

- **Cube layout.** Emitting `(n_pulses, n_samples)` means `range_doppler_map`'s default axes just
  work. Every `dsp` function takes an explicit `axis`, so a different layout would cost a keyword
  argument, not a rework.
- **Doppler sign.** `doppler_bin_centers_mps` returns an `fftshift`-ed axis, zero at `n_bins // 2`
  and increasing. If `core/signal.py` generates the opposite sign, closing targets will appear to
  open — a bug that looks like a plausible picture, so §6's acceptance check must catch it rather
  than the eye.
- **Range unshifted, Doppler shifted.** `teaching/scopes/rd_map.py` must label axes with
  `range_bin_centers_m` and `doppler_bin_centers_mps` rather than deriving them, since those helpers
  already encode this asymmetry and the unambiguous limits.

### 4.4 Amendments to `spec/structure.md`

> **Applied.** These amendments are folded into `spec/structure.md`; it is the authoritative tree
> and decision list. This section records what changed and why.

1. **`core/geodesy.py` is new.** `spec/structure.md` gave WGS-84 ↔ ENU no home. It does not belong
   in `pipelines/trajectories.py`, because `raytracing/scene.py` will need it to place scene
   geometry, nor in `config.py`, which is units and global settings. It is a `core` concern.
2. **`core/propagation.py` is deferred.** This slice is free-space only, via the existing
   `received_power_w`. Atmospheric and rain attenuation are negligible for a clear X-band path at
   22 km, and implementing them here would ship untested code.
3. **`line_of_sight_paths` is the zeroth-order analytic backend.** It produces a `PropagationPaths`
   with `bounce_count == 1` and no multipath. Whoever builds `raytracing/backends/analytic.py`
   should build on it rather than duplicate it — D1's decision that `core/signal.py` owns
   path → baseband for *all* backends is what makes that possible, and this slice is its first
   proof.

The open question recorded in `spec/structure.md` — that the analytic backend is the only one under
test until a GPU backend exists — stands unchanged.

### 4.5 Workstream boundary

This scenario claims, and another workstream should not plan:

```
data/flight_coordinates.csv                    scenarios/scenario_001_*.toml
scripts/run_scenario.py                        spec/scenario-001-xband.md
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
`teaching/scopes/` — is outside this scenario.

---

## 5. Build Sequence

Each module lands with its tests in the same commit, and `make check` passes before the next
begins.

```
  0. core/constants.py (edit)   <-- load-bearing: R4 blocks the WGS-84 literal
             |
             v
  1. core/geodesy.py            geodetic <-> ECEF <-> ENU -> range/az/el
             |
             v
  2. core/radar.py              Transmitter / Receiver / Radar; validation only
             |
             v
  3. core/targets.py            PointTarget, Swerling 0
             |
             v
  4. core/signal.py             PropagationPaths -> baseband + noise
             |
             |     [ 5. core/dsp.py, core/windows.py -- already built elsewhere ]
             |          consumed, never edited; see §4.3 for the seam
             v
  5a. core/ambiguity.py         unfold_doppler_dual_prf          (S3 only)
             |
             v
  6. pipelines/trajectories.py  CSV -> frame grid -> radar frame
             |
             v
  7. pipelines/scenarios.py     TOML -> Scenario; iterate_frames() is a generator
             |
             v
  8. teaching/plotting.py, teaching/scopes/rd_map.py   (matplotlib, lazily imported)
             |
             v
  9. scripts/run_scenario.py    thin CLI; all logic stays in pipelines/
             |
             v
 10. __init__.py wiring         import radar_forge must succeed with zero extras
```

| Step | Module | Notes |
| :--- | :--- | :--- |
| 0 | `core/constants.py` *(edit)* | Add `WGS84_SEMI_MAJOR_AXIS_M`, `WGS84_FLATTENING`. **Load-bearing**: rule R4 in `scripts/check_conventions.py` rejects the literal `6_378_137.0` outside this module, so geodesy cannot be written first. R4's literal map must also be corrected — it suggested `EARTH_RADIUS_M` for that literal, which is wrong for an ellipsoid semi-major axis. |
| 1 | `core/geodesy.py` *(new)* | WGS-84 geodetic ↔ ECEF ↔ local ENU; ENU → range/azimuth/elevation. Azimuth 0° at true north, increasing clockwise, per D5. |
| 2 | `core/radar.py` | Frozen `Transmitter`, `Receiver`, `Radar` dataclasses; `wavelength_m`, `unambiguous_range_m`, `unambiguous_velocity_mps` properties. Validation only, no simulation. |
| 3 | `core/targets.py` | `PointTarget`, constant RCS, **Swerling 0 only**. Swerling 1–4 stay unimplemented and are named as such. |
| 4 | `core/signal.py` | `PropagationPaths` exactly as specified in D1, plus `line_of_sight_paths`, `fmcw_deramp_baseband`, `pulsed_baseband`, `thermal_noise`. Reuses `radar_equation.received_power_w` and `waveforms.lfm_chirp` / `beat_frequency_hz` — **adds no new waveform or link-budget mathematics**. |
| 5 | `core/dsp.py`, `core/windows.py` | **Already built by the parallel workstream** (commit 64f1b62) and consumed, not written, by this scenario: `range_fft`, `doppler_fft`, `range_doppler_map`, `matched_filter`, `mti_filter`, `range_bin_centers_m`, `doppler_bin_centers_mps`, `taper`. See §4.3. |
| 5a | `core/ambiguity.py` *(new)* | `unfold_doppler_dual_prf`, needed only by S3. Kept out of `core/dsp.py` because that module belongs to the other workstream and is already near the 400-line guidance of `docs/conventions/style.md` §9. |
| 6 | `pipelines/trajectories.py` | CSV load, resample to the frame grid, transform to the radar frame. |
| 7 | `pipelines/scenarios.py` | Scenario TOML (`tomllib`, stdlib) → `Scenario`; `iterate_frames` is a **generator**, not a list. |
| 8 | `teaching/plotting.py`, `teaching/scopes/rd_map.py` | `matplotlib` lazily imported; `ImportError` names the `teaching` extra. |
| 9 | `scripts/run_scenario.py` | Thin CLI; all logic stays in `pipelines/`. |
| 10 | `__init__.py` wiring | `import radar_forge` must still succeed with zero extras, so `teaching` is **not** imported at top level. |

Scenario configurations live in `scenarios/scenario_001_{fmcw_low_prf,pulsed_medium_prf,fmcw_dual_prf}.toml`,
sharing `[radar]`, `[receiver]`, `[target]` and `[trajectory]` blocks and differing only in
`[[burst]]`.

### 5.1 Where the input data lives

`data/flight_coordinates.csv`, tracked. Not `src/radar_forge/` (rule R2 tolerates `.csv` there, but
library code stays code); not `tests/data/golden/`, which `docs/conventions/testing.md` §6 reserves
for kilobyte-scale regression fixtures with a regeneration script. A 51-row excerpt is committed to
`tests/data/golden/flight_coordinates_head.csv` so the loader test does not depend on the full file.

---

## 6. Acceptance Criteria

The scenario is done when, for each of the three variants over the default 120 s window, the
range-Doppler peak falls in the bin predicted analytically from `truth.csv`.

| # | Variant | Criterion | Bound | Verified by |
| :--- | :--- | :--- | :--- | :--- |
| A1 | S1 | Peak range matches the true range | within one 74.9481 m bin | `tests/pipelines/test_scenario_001.py` |
| A2 | S1 | Peak velocity matches the true velocity **wrapped** into ±7.6478 m/s | within one 0.0597482 m/s bin | same |
| A3 | S2 | Peak range matches the true range **wrapped** into 5.996 km | within one 74.9481 m bin | same |
| A4 | S2 | Peak velocity matches the true velocity directly | within one velocity bin | same |
| A5 | S3 | After `unfold_doppler_dual_prf`, peak velocity matches the **unwrapped** true velocity, in every frame | within one bin | same |
| A6 | all | Each unambiguous limit is re-derived from `core/radar.py`, not read from §3 | exact to `rtol = 1e-4` | `tests/core/test_radar.py`, `tests/pipelines/test_scenarios.py` |
| A7 | all | Closing targets close: the sign of the peak velocity matches the sign of `truth.csv`'s | sign agreement in every frame | `tests/pipelines/test_scenario_001.py` |
| A8 | all | `import radar_forge` succeeds with zero extras installed | no `ImportError` | `tests/test_import.py` |

Encoded as `tests/pipelines/test_scenario_001.py`, marked `slow`, over a 5-frame window so that
`make check` stays fast.

---

## References

.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014.
       §4.4 (LFM), §5.3 (pulse-Doppler ambiguity), §8.3 (matched filtering).
.. [2] G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014, §2.5 (FMCW deramp).
.. [3] L. Harrison and G. Andrews, *Introduction to Radar Using Python and MATLAB*, Artech
       House, 2020, ch. 2, ch. 4.
.. [4] National Imagery and Mapping Agency, *Department of Defense World Geodetic System 1984*,
       NIMA TR8350.2, 3rd ed., 2000.

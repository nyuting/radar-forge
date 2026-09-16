# Scenario 002 — Bistatic radar, Raleigh-Durham illuminator to the Duke Receiver

## 1. Status & Purpose

**Status: implemented.** The second vertical slice, and the first bistatic one.

`spec/scenario-001-xband.md` proved the monostatic path end to end. This document specifies the
**second vertical slice**: the same aircraft, the same receive site, but with the transmitter moved
19.6 km away to Raleigh-Durham Airport, so that transmit and receive ranges are no longer the same
number.

Three reasons to make bistatic the second slice rather than breadth:

1. **It tests decision D1 against a second geometry.** `PropagationPaths` was designed to be the one
   thing every propagation model produces. Until a second model exists, that is an assertion. §4.1
   records what survived: the dataclass is unchanged, which is the strongest evidence so far that
   D1 was drawn in the right place.
2. **It is the geometry `pyAPRiL` exists for.** `spec/starter.md` §3.1 names bistatic processing
   as something `radar-forge` borrows, but Part B of `spec/structure.md` had no home for a two-site
   radar. This closes that gap.
3. **It teaches a real thing badly taught.** Bistatic range resolution depends on where the target
   *is*, not only on bandwidth. Over this track the resolution degrades by up to **2.34×**, which is
   visible on the map rather than merely stated.

| Section | What it fixes |
| :--- | :--- |
| §2 Overview & Geometry | The two sites, the baseline, and what the bistatic angle does over the track |
| §3 Parameter Matrix | Sites, waveform, derived geometry, the two variants, the default window |
| §4 Core Decisions | D6, D7, D8; the `range_tx_m` / `range_rx_m` naming; what moved in `structure.md` |
| §5 Build Sequence | What is new, what is widened, what is untouched |
| §6 Acceptance Criteria | The verification matrix, led by the degeneracy test |

---

## 2. Overview & Geometry

A receiver at the Duke Receiver listens to the Raleigh-Durham Airport air-traffic-control
illuminator and detects a light aircraft flying a circuit over the North Carolina Piedmont. The
transmitter and the receiver stand 19.601 km apart, so the two propagation ranges are independent.

```
               [ Target Aircraft ]
               /                 \
        R_r (Receive)       R_t (Transmit)
             /                     \
            v                       v
       [ Duke Receiver ] <────────> [ Raleigh-Durham Airport ]
                             Baseline
                           (19.601 km)
```

The angle β subtended at the target by the two sites is the whole story. At β = 0 the geometry is
monostatic; as β grows, the iso-range surfaces — ellipsoids with the two sites at their foci —
crowd near the sites and spread near the baseline, and the range resolution `c / (2B·cos(β/2))`
degrades with it.

```
                      beta = 0                          beta -> 180
             monostatic; circles of                 near the baseline;
             constant range, resolution           ellipsoids spread wide,
             c/2B = 74.9481 m                    resolution -> unbounded

   |------------------|------------------|------------------|
   0 deg           40.3 deg          129.3 deg           180 deg
                   ^                  ^
                   |                  |
          track minimum          track maximum
          resolution 79.85 m     resolution 175.21 m  (2.34x the floor)
```

The processing chain is scenario 001's unchanged. Only the geometry block differs:

```
  data/flight_coordinates.csv       scenarios/scenario_002_bistatic_*.toml
            |                          [radar] + [transmitter_site] + [receiver_site]
            v                                        |
  pipelines/trajectories.py                          |
     to_bistatic_radar_frame  <------------------ BistaticRadar
            |
            v
  BistaticTargetTrack (range_tx_m, range_rx_m, bistatic_angle_rad,
                       bisector_velocity_mps, tx/rx az and el)
            |
            v
  core/signal.py  bistatic_line_of_sight_paths -> PropagationPaths
            |        delay_s = (R_t + R_r)/c,  doppler_hz = 2*v_bisector/lambda
            v
     fmcw_deramp_baseband  (RadarLike; no bistatic twin)      + thermal_noise
            |
            v
  core/dsp.py  range_doppler_map          <-- UNCHANGED
            |
            v
     range axis carries BISTATIC MEAN RANGE (R_t + R_r)/2
            |
            v
  teaching/scopes/rd_map.py  (axis LABEL differs; nothing else does)
```

---

## 3. Parameter Matrix

### 3.1 Sites and target

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Transmit site (Raleigh-Durham) | 35.87750 °N, 78.78750 °W, 25 m | Raleigh-Durham Airport reference point. Antenna height assumed; configurable |
| Receive site (Duke Receiver) | 36.00250 °N, 78.94100 °W, 60 m | Unchanged from scenario-001 |
| Baseline `L` | **19.601 km** | Computed from the two sites, not assumed |
| Target track | `data/flight_coordinates.csv` | The same real track as scenario-001 |
| Target altitude | 1500 m AMSL, **constant** | The CSV has no altitude column |
| Target RCS | 10 m² (10 dBsm) | **Bistatic** RCS, held constant; see D8 |
| Frame cadence | 1 Hz | As scenario-001 |

### 3.2 Waveform — identical in both variants but for the carrier

| Parameter | Value |
| :--- | :--- |
| Waveform | FMCW sawtooth, deramp receive |
| `bandwidth_hz` | 2.0 MHz → best-case resolution c/2B = **74.9481 m** |
| `chirp_duration_s` | 1.0 ms, sweep rate α = 2.000 GHz/s |
| `prf_hz` | 1.0 kHz |
| `sample_rate_hz` | 1.0 MHz, 1000 samples/chirp |
| `n_pulses` | 256 → CPI 256.0 ms |
| `transmit_power_w` | 100 W; `gain_tx_dbi` = `gain_rx_dbi` = 30.0; `noise_figure_db` = 3.0 |
| Range-sum limit `c·f_s/2α` | 74.95 km, against a range sum that never exceeds 51.97 km |

| | **B1** `bistatic-xband` | **B2** `bistatic-sband` |
| :--- | :--- | :--- |
| `f0_hz` | 9.8 GHz | 2.8 GHz |
| λ | 30.591 mm | 107.069 mm |
| Unambiguous velocity | ±7.6478 m/s | **±26.7672 m/s** |
| Folds over the default window | ≈8.6× | ≈2.5× |
| Realistic? | No — a fiction for comparability | Yes — terminal ATC PSR is S-band |

### 3.3 Derived geometry over the full track

Computed against both sites:

| Quantity | Value |
| :--- | ---: |
| Transmit range `range_tx_m` (Raleigh-Durham → target) | 13.502 – 30.203 km |
| Receive range `range_rx_m` (target → Duke Receiver) | 5.326 – 22.010 km |
| Range sum `R_t + R_r` | 21.391 – 51.969 km |
| Bistatic mean range `(R_t + R_r)/2` | 10.696 – 25.985 km |
| Bistatic angle β | **40.3° – 129.3°** |
| Range resolution at those angles | 79.85 m – **175.21 m** (1.07× – **2.34×** the floor) |

The bistatic angle exceeds 90° over part of the track. This is past the point where the
monostatic-equivalence theorem is usually invoked, which is exactly why the scenario is worth
having: a slice that stayed under 20° would not exercise anything the monostatic code does not
already cover.

### 3.4 Default window

`start_time_s = 13329.0`, `duration_s = 120.0`, chosen — not inherited — as the 120 s window that
maximises the mean bistatic angle over the track. Across it:

| Quantity | Value |
| :--- | ---: |
| Bistatic angle β | 109.9° – 129.3° |
| Bistatic mean range | 10.74 – 11.81 km |
| Bisector range rate | −61.9 – +65.8 m/s |
| Range resolution | 130.5 – 175.2 m (**1.74× – 2.34×** the β = 0 floor) |
| Range sum | ≤ 23.62 km — unambiguous in both variants |

The window is strongly bistatic throughout. Scenario-001's window (`start_time_s = 0`) would give
β = 40.7° – 44.1° here, where the geometry is nearly monostatic and the slice would prove little.

### 3.5 Mathematical contracts

Scenario 001 §3.5 holds unchanged. What is added or reinterpreted:

| Contract | Statement |
| :--- | :--- |
| `PropagationPaths.delay_s` | `(R_t + R_r)/c` — the whole transmitter → target → receiver route |
| `PropagationPaths.range_m` | `delay_s·c/2` = the **bistatic mean range** `(R_t + R_r)/2` |
| `PropagationPaths.doppler_hz` | `2·v_bisector/λ`, with `v_bisector = −½·d(R_t + R_r)/dt`, positive closing |
| RD map range axis | Bistatic mean range, metres. **The array is identical to the monostatic case; only the label differs.** |
| `BistaticTargetTrack` | `(n_frames,)` float64 per field: `range_tx_m`, `range_rx_m`, `bistatic_angle_rad`, `bisector_velocity_mps`, and tx/rx azimuth and elevation in degrees |
| Coordinate frames | **Two** local ENU tangent planes, one per site. Transmit angles are angles of departure; receive angles are angles of arrival. They genuinely differ — that is what makes the geometry bistatic |
| Type alias | `RadarLike = Radar \| BistaticRadar`; consumers widen to it rather than growing `bistatic_*` twins |
| TOML layout | `[scenario] [radar] [transmitter_site] [receiver_site] [target] [trajectory] [[burst]]` — `[transmitter_site]` and `[receiver_site]` adjacent, matching `scenario_003_tracking.toml` |

Selecting the siting is data, not an argument: `pipelines/scenarios.py` builds a `BistaticRadar`
when a `[transmitter_site]` table is present, so a monostatic scenario file is unchanged by the
feature's existence.

### 3.6 Output schema

As scenario-001 §3.6, with `truth.csv` gaining three columns:

| Column | Meaning |
| :--- | :--- |
| `range_tx_m` | Raleigh-Durham site to target |
| `range_rx_m` | Target to the Duke Receiver site |
| `bistatic_angle_deg` | β at the target |

`range_m` remains present and carries the **bistatic mean range**, so the column set stays a
superset of scenario-001's and the D5 COCO exporter can consume either unchanged.

---

## 4. Core Decisions

### 4.1 D6 — why `PropagationPaths` does not change

`spec/structure.md` D1 fixes the backend contract. The bistatic case was not considered when it was
drawn, so the first question this slice had to answer is whether the dataclass survives. It does,
without an edit:

| Field | Monostatic reading | Bistatic reading |
| :--- | :--- | :--- |
| `delay_s` | `2R/c` | `(R_t + R_r)/c` |
| `range_m` property, `delay_s·c/2` | `R` | the **bistatic mean range** `(R_t + R_r)/2` |
| `doppler_hz` | `2v/λ` | `(Ṙ_t + Ṙ_r)/λ` — the same expression with the bisector range rate |

Both monostatic forms are the special case `R_t = R_r`. No field is added, no generator signature
changes shape, and `core/dsp.py` is untouched: the range axis it produces is a **bistatic mean
range** axis, and only the label differs.

`range_tx_m` and `range_rx_m` are deliberately **not** added to `PropagationPaths`. The two ranges
are geometry, and geometry belongs to `BistaticRadar`. A path set is what a propagation model
produced, not how it was arranged.

### 4.2 D7 — one siting-agnostic pipeline, via `RadarLike`

`RadarLike = Radar | BistaticRadar`. Consumers widen to it rather than growing `bistatic_*` twins.
The signal generators depend only on waveform and receiver attributes that both classes carry, so
`fmcw_deramp_baseband` and `pulsed_baseband` need a type widening and nothing else.

### 4.3 D8 — bistatic RCS is taken as given and angle-independent

`PointTarget` keeps a single `rcs_m2`, used as σ_b. The monostatic-equivalence theorem licenses this
only for smooth bodies at small β away from resonance, and this scenario runs to β = 129.3°, so the
**absolute** power levels here are an approximation. The acceptance criteria in §6 test range,
velocity and resolution — geometry — and deliberately do not assert an absolute SNR.

Forward scatter is not modelled at all. The track's maximum β is 129.3°, short of the ≈135° where
the forward-scatter enhancement begins to matter, so it does not arise here; a scenario that needs
it must say so.

### 4.4 Naming: `range_tx_m` and `range_rx_m`

The `tx`/`rx` token names **the site the range is measured to**, exactly as it names the antenna in
`gain_tx_linear` and `gain_rx_linear`. It is not a property of the transmitter itself.

- `range_tx_m` — the transmit range `R_t`, transmitter site to target.
- `range_rx_m` — the receive range `R_r`, target to receiver site.

Neither name is needed for a monostatic radar, where the two are the same number. These are never
called "legs" or "paths": a path in `core/signal.py` is the whole transmitter–target–receiver route,
and a burst is a waveform configuration, so both terms are taken.

### 4.5 Why the carrier is the only difference between B1 and B2

Both variants use the geometry of §3.1 and the waveform of scenario-001 S1.

**B1 — `bistatic-xband`, f₀ = 9.8 GHz.** Identical in every waveform parameter to
`scenario_001_fmcw_low_prf.toml`, so **geometry is the only variable** between the two scenarios.
That makes the comparison clean and gives §6 its strongest test: shrink the baseline to nothing and
B1 must reproduce scenario-001's map.

> This is a fiction, and the spec says so plainly: Raleigh-Durham's primary surveillance radar is
> not a 9.8 GHz FMCW transmitter. B1 exists for comparability with scenario-001, not for realism.
> B2 is the physically honest one.

**B2 — `bistatic-sband`, f₀ = 2.8 GHz.** The realistic case. Terminal air-traffic-control primary
surveillance radar is S-band, and 2.8 GHz is representative. λ = 107.069 mm, so Doppler folds at
±26.767 m/s rather than ±7.648 m/s. The pair teaches directly what λ buys: the same trajectory, the
same geometry, the same processing, and 3.5× more unambiguous velocity purely from the carrier.
Over the default window the bisector rate reaches 65.8 m/s, so B1 folds about 8.6 times and B2
about 2.5 times — both fold, but B2's map is readable.

### 4.6 Amendments to `spec/structure.md`

> **Applied.** These amendments are folded into `spec/structure.md`; it is the authoritative tree
> and decision list. This section records what changed and why.

1. **`BistaticRadar` joins `core/radar.py`.** Part B listed `Transmitter, Receiver, Radar`. Add
   `BistaticRadar` and the `RadarLike` alias. It is not a new module: it shares validation, waveform
   and noise behaviour with `Radar`, and splitting it would duplicate all three.
2. **`core/radar_equation.py` gains the bistatic form.** `bistatic_received_power_w`, whose
   denominator carries the product `R_t²R_r²` — so contours of constant received power are ovals of
   Cassini rather than circles.
3. **Decisions D6, D7 and D8 are added**, as stated in §4.1–§4.3.
4. **The `range_m` axis produced by `core/dsp.py` is a bistatic mean range axis** in the bistatic
   case. `core/dsp.py` itself is unchanged; only `teaching/scopes/rd_map.py`'s axis label differs.

The open question recorded in `spec/structure.md` — that the analytic backend is the only one under
test until a GPU backend exists — stands unchanged.

### 4.7 Workstream boundary

This scenario claims, and another workstream should not plan:

```
scenarios/scenario_002_bistatic_{xband,sband}.toml
spec/scenario-002-bistatic.md
src/radar_forge/core/{radar,radar_equation,signal}.py
src/radar_forge/pipelines/{trajectories,scenarios}.py
src/radar_forge/teaching/scopes/rd_map.py   (axis label only)
```

`data/flight_coordinates.csv`, `scenarios/scenario_001_*.toml` and
`tests/pipelines/test_scenario_001.py` belong to scenario-001 and are **read, never edited**. The
monostatic path must not shift: `test_scenario_001.py` staying green is a precondition for every
commit in this slice.

`src/radar_forge/core/{dsp,windows,detection,ambiguity}.py` are consumed, never edited.

Everything else in `spec/structure.md` — `array/`, `raytracing/`, `core/{propagation, clutter,
tracking}.py`, `pipelines/{generate, datasets}.py` and `pipelines/exporters/` — is outside this
scenario.

---

## 5. Build Sequence

Nothing here is a new pipeline. Each step is either a widening of an existing signature or a second
geometry alongside the first.

```
  1. core/radar.py (edit)       + BistaticRadar, + RadarLike alias
             |                    shares validation/waveform/noise with Radar
             v
  2. core/radar_equation.py      + bistatic_received_power_w  (R_t^2 R_r^2)
             |
             v
  3. core/signal.py (widen)      + bistatic_line_of_sight_paths
             |                    generators take RadarLike; no bistatic twins
             v
  4. pipelines/trajectories.py   + BistaticTargetTrack, to_bistatic_radar_frame
             |
             v
  5. pipelines/scenarios.py      [transmitter_site] present -> BistaticRadar
             |                    [receiver] -> [receiver_site], moved adjacent
             v
  6. scenarios/scenario_002_bistatic_{xband,sband}.toml
             |
             v
  7. teaching/scopes/rd_map.py   axis LABEL only
             |
             v
  8. tests/pipelines/test_scenario_002.py   degeneracy test first
```

| Step | Module | Notes |
| :--- | :--- | :--- |
| 1 | `core/radar.py` *(edit)* | `BistaticRadar` with both sitings; `RadarLike` union. Per §4.6.1, not a new module. |
| 2 | `core/radar_equation.py` *(edit)* | `bistatic_received_power_w`. Additive; `received_power_w` is unchanged and remains the monostatic special case. |
| 3 | `core/signal.py` *(widen)* | `bistatic_line_of_sight_paths` producing the same `PropagationPaths`; `fmcw_deramp_baseband` and `pulsed_baseband` widen to `RadarLike`. |
| 4 | `pipelines/trajectories.py` *(edit)* | `BistaticTargetTrack` and `to_bistatic_radar_frame`. No `radial_velocity_mps` field: a bistatic target has two radial velocities and the Doppler shift measures neither. |
| 5 | `pipelines/scenarios.py` *(edit)* | Siting selected by the presence of `[transmitter_site]`. `[receiver]` becomes `[receiver_site]`, placed directly after `[transmitter_site]` (refactor-001 §3.1). |
| 6 | `scenarios/*.toml` *(new)* | Two files differing in `[[burst]].f0_hz` alone. |
| 7 | `teaching/scopes/rd_map.py` *(edit)* | Axis label only. The array is the same array. |
| 8 | `tests/pipelines/test_scenario_002.py` *(new)* | Build the degeneracy test first; everything else is meaningless until it passes. |

---

## 6. Acceptance Criteria

Encoded as `tests/pipelines/test_scenario_002.py`, marked `slow`, over a 5-frame window so that
`make check` stays fast. Mirrors `tests/pipelines/test_scenario_001.py`.

| # | Criterion | Bound | Applies to |
| :--- | :--- | :--- | :--- |
| A1 | **Degeneracy — the load-bearing test.** With the transmit site moved to the receive site plus one metre, B1 reproduces `scenario_001_fmcw_low_prf`'s range-Doppler peak | one range bin and one velocity bin | B1 |
| A2 | Peak bistatic mean range matches `(R_t + R_r)/2` from `truth.csv` via `BistaticRadar.target_ranges_m`, in every frame | within one range bin | B1, B2 |
| A3 | Peak velocity matches the true bisector range rate **wrapped** into the variant's unambiguous interval | within one velocity bin of ±7.6478 m/s (B1), ±26.7672 m/s (B2) | B1, B2 |
| A4 | **Resolution.** Two targets separated by `d` metres along the bisector direction are separated in bistatic mean range by `d·cos(β/2)` | `1/cos(β/2)`, i.e. 1.07× – 2.34× the monostatic spatial separation over the track | B1 |
| A5 | The baseline computed from the two sites matches §3.1 | 19.601 km, `rtol = 1e-4` | both |
| A6 | β over the default window stays strongly bistatic | 109.9° – 129.3° | both |
| A7 | Absolute received power | **not asserted**, per D8 | — |

> A4 is **not** a statement about peak width, and an earlier draft of this spec had it wrong. A
> point target has exactly one delay, so the width of its range peak is set by the window and the
> transform length and does not change with β at all. What degrades is the *mapping from space to
> range*: the iso-range surfaces are ellipsoids with the two sites at the foci, and they crowd
> together near the sites and spread out near the baseline. Asserting a widened peak would have
> tested a thing this simulation cannot produce, and would have been made to pass by loosening a
> tolerance.

---

## References

.. [1] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005. §1.3 (geometry and the
       bistatic angle), §2.2 (the bistatic range equation and its ovals of Cassini), §4.2 (range
       resolution), §6.2 (bistatic Doppler).
.. [2] M. C. Jackson, *The geometry of bistatic radar systems*, IEE Proc. F, 133(7), 1986.
.. [3] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014,
       §5.3 (pulse-Doppler ambiguity).
.. [4] National Imagery and Mapping Agency, *Department of Defense World Geodetic System 1984*,
       NIMA TR8350.2, 3rd ed., 2000.

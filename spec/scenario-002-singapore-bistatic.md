# Scenario 002 — Passive bistatic radar, Changi illuminator to DSO receiver

Status: **specification**, pre-implementation. The second vertical slice, and the first bistatic one.

## 1. Purpose

`spec/scenario-001-singapore-xband.md` proved the monostatic path end to end. This document
specifies the **second vertical slice**: the same aircraft, the same DSO site, but with the
transmitter moved 23.7 km away to Changi Airport, so that transmit and receive ranges are no longer
the same number.

Three reasons to make bistatic the second slice rather than breadth:

1. **It tests decision D1 against a second geometry.** `PropagationPaths` was designed to be the one
   thing every propagation model produces. Until a second model exists, that is an assertion. §3
   records what survived: the dataclass is unchanged, which is the strongest evidence so far that
   D1 was drawn in the right place.
2. **It is the geometry `pyAPRiL` exists for.** `spec/starter.md` §2.1 names passive bistatic
   processing as something `radar-forge` borrows, but Part B of `spec/structure.md` has no home for
   a two-site radar. This closes that gap.
3. **It teaches a real thing badly taught.** Bistatic range resolution depends on where the target
   *is*, not only on bandwidth. Over this track the resolution degrades by up to 1.56×, which is
   visible on the map rather than merely stated.

## 2. Scenario

A receive-only site at DSO National Laboratories listens to the Changi Airport air-traffic-control
illuminator and detects a light aircraft flying a circuit over Singapore. Nothing at DSO transmits;
this is a **passive bistatic** configuration.

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Transmit site (Changi) | 1.3592 °N, 103.9894 °E, 25 m | Singapore Changi Airport reference point. Antenna height assumed; configurable |
| Receive site (DSO) | 1.29150 °N, 103.78710 °E, 60 m | Unchanged from scenario-001 |
| Baseline `L` | **23.726 km** | Computed, not assumed |
| Bandwidth `bandwidth_hz` | 2.0 MHz | As scenario-001. Best-case resolution c/2B = 74.95 m |
| Target track | `data/flight_coordinates.csv` | The same real track as scenario-001 |
| Target altitude | 1500 m AMSL, **constant** | The CSV has no altitude column |
| Target RCS | 10 m² (10 dBsm) | **Bistatic** RCS, held constant; see D8 |
| Frame cadence | 1 Hz | As scenario-001 |

Derived geometry over the full track, computed against both sites:

| Quantity | Value |
| :--- | :--- |
| Transmit range `range_tx_m` (Changi → target) | 16.06 – 37.20 km |
| Receive range `range_rx_m` (target → DSO) | 8.36 – 17.84 km |
| Range sum `R_t + R_r` | 30.81 – 53.86 km |
| Bistatic mean range `(R_t + R_r)/2` | 15.41 – 26.93 km |
| Bistatic angle β | **25.8° – 100.5°** |
| Range resolution at those angles | 74.95 m – **117.2 m** |

The bistatic angle exceeds 90° over part of the track. This is past the point where the
monostatic-equivalence theorem is usually invoked, which is exactly why the scenario is worth
having: a slice that stayed under 20° would not exercise anything the monostatic code does not
already cover.

## 3. Why `PropagationPaths` does not change — decision D6

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

### D7 — one siting-agnostic pipeline, via `RadarLike`

`RadarLike = Radar | BistaticRadar`. Consumers widen to it rather than growing `bistatic_*` twins.
The signal generators depend only on waveform and receiver attributes that both classes carry, so
`fmcw_deramp_baseband` and `pulsed_baseband` need a type widening and nothing else.

### D8 — bistatic RCS is taken as given and angle-independent

`PointTarget` keeps a single `rcs_m2`, used as σ_b. The monostatic-equivalence theorem licenses this
only for smooth bodies at small β away from resonance, and this scenario runs to β = 100.5°, so the
**absolute** power levels here are an approximation. The acceptance criteria in §7 test range,
velocity and resolution — geometry — and deliberately do not assert an absolute SNR.

Forward scatter is not modelled at all. The track never approaches the baseline (β never exceeds
101°), so the forward-scatter enhancement does not arise here; a scenario that needs it must say so.

## 4. Naming: `range_tx_m` and `range_rx_m`

The `tx`/`rx` token names **the site the range is measured to**, exactly as it names the antenna in
`gain_tx_linear` and `gain_rx_linear`. It is not a property of the transmitter itself.

- `range_tx_m` — the transmit range `R_t`, transmitter site to target.
- `range_rx_m` — the receive range `R_r`, target to receiver site.

Neither name is needed for a monostatic radar, where the two are the same number. These are never
called "legs" or "paths": a path in `core/signal.py` is the whole transmitter–target–receiver route,
and a burst is a waveform configuration, so both terms are taken.

## 5. Two variants, and why the carrier is the only difference

Both variants use the geometry of §2 and the waveform of scenario-001 S1 — 2 MHz bandwidth, 1 ms
sweep, 1 kHz PRF, 1 MHz sampling, 256 pulses. They differ in the carrier alone.

### B1 — `bistatic-xband`, f₀ = 9.8 GHz

Identical in every waveform parameter to `scenario_001_fmcw_low_prf.toml`, so **geometry is the only
variable** between the two scenarios. That makes the comparison clean and gives §7 its strongest
test: shrink the baseline to nothing and B1 must reproduce scenario-001's map.

λ = 30.591 mm, so Doppler folds at ±7.65 m/s — the same fold as S1.

> This is a fiction, and the spec says so plainly: Changi's primary surveillance radar is not a
> 9.8 GHz FMCW transmitter. B1 exists for comparability with scenario-001, not for realism. B2 is
> the physically honest one.

### B2 — `bistatic-sband`, f₀ = 2.8 GHz

The realistic case. Terminal air-traffic-control primary surveillance radar is S-band, and 2.8 GHz
is representative.

λ = 107.07 mm, so Doppler folds at **±26.77 m/s** rather than ±7.65 m/s. The pair teaches directly
what λ buys: the same trajectory, the same geometry, the same processing, and 3.5× more unambiguous
velocity purely from the carrier. Over the default window the bisector rate reaches 67 m/s, so B1
folds about 8.8 times and B2 about 2.5 times — both fold, but B2's map is readable.

Range is unambiguous in both: the FMCW range-sum limit is `c·f_s/2α = 74.95 km` against a range sum
that never exceeds 53.86 km.

## 6. Default window

`start_time_s = 13360.0`, `duration_s = 120.0`, chosen — not inherited — as the 120 s window that
maximises the mean bistatic angle over the track. Across it:

| Quantity | Value |
| :--- | :--- |
| Bistatic angle β | 91.7° – 100.5° |
| Bistatic mean range | 15.41 – 16.50 km |
| Bisector range rate | −67.0 – +55.9 m/s |
| Range resolution | 107.5 – 117.2 m (**1.43× – 1.56×** the β = 0 floor) |

The window is strongly bistatic throughout. Scenario-001's window (`start_time_s = 0`) would give
β = 34° – 38° here, where the geometry is nearly monostatic and the slice would prove little.

## 7. Acceptance criteria

Encoded as `tests/pipelines/test_scenario_002.py`, marked `slow`, over a 5-frame window so that
`make check` stays fast. Mirrors `tests/pipelines/test_scenario_001.py`.

1. **Degeneracy — the load-bearing test.** With the transmit site moved to the receive site plus one
   metre, B1 must reproduce `scenario_001_fmcw_low_prf`'s range-Doppler peak to within a small
   tolerance. This catches every factor-of-two and sign error in the delay, the phase and the
   Doppler at once, against code that is already trusted.
2. **Range.** The peak's bistatic mean range falls within one range bin of `(R_t + R_r)/2` computed
   from `truth.csv` via `BistaticRadar.target_ranges_m`, in every frame, for both variants.
3. **Velocity.** The peak velocity matches the true bisector range rate **wrapped** into ±7.65 m/s
   (B1) and ±26.77 m/s (B2). `truth.csv` carries the unwrapped rate; the discrepancy is the point.
4. **Resolution.** The measured range-peak width exceeds the β = 0 width by the factor
   `1/cos(β/2)` predicted at that frame's bistatic angle, within tolerance. This is the one
   criterion with no monostatic counterpart.

Absolute received power is **not** an acceptance criterion, per D8.

## 8. Outputs

As scenario-001 §6, with `truth.csv` gaining three columns:

| Column | Meaning |
| :--- | :--- |
| `range_tx_m` | Changi site to target |
| `range_rx_m` | Target to DSO site |
| `bistatic_angle_deg` | β at the target |

`range_m` remains present and carries the **bistatic mean range**, so the column set stays a
superset of scenario-001's and the D5 COCO exporter can consume either unchanged.

## 9. Amendments to `spec/structure.md`

1. **`BistaticRadar` joins `core/radar.py`.** Part B lists `Transmitter, Receiver, Radar`. Add
   `BistaticRadar` and the `RadarLike` alias. It is not a new module: it shares validation, waveform
   and noise behaviour with `Radar`, and splitting it would duplicate all three.
2. **`core/radar_equation.py` gains the bistatic form.** `bistatic_received_power_w`, whose
   denominator carries the product `R_t²R_r²` — so contours of constant received power are ovals of
   Cassini rather than circles.
3. **Decisions D6, D7 and D8 are added**, as stated in §3.
4. **The `range_m` axis produced by `core/dsp.py` is a bistatic mean range axis** in the bistatic
   case. `core/dsp.py` itself is unchanged; only `teaching/scopes/rd_map.py`'s axis label differs.

The open question recorded in `spec/structure.md` — that the analytic backend is the only one under
test until a GPU backend exists — stands unchanged.

## 10. Workstream boundary

This scenario claims, and another workstream should not plan:

```
scenarios/scenario_002_bistatic_{xband,sband}.toml
spec/scenario-002-singapore-bistatic.md
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

## References

.. [1] N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005. §1.3 (geometry and the
       bistatic angle), §2.2 (the bistatic range equation and its ovals of Cassini), §4.2 (range
       resolution), §6.2 (bistatic Doppler).
.. [2] M. C. Jackson, *The geometry of bistatic radar systems*, IEE Proc. F, 133(7), 1986.
.. [3] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014,
       §5.3 (pulse-Doppler ambiguity).
.. [4] National Imagery and Mapping Agency, *Department of Defense World Geodetic System 1984*,
       NIMA TR8350.2, 3rd ed., 2000.

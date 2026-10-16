# Scenario 010 — Array self-calibration and robust beamforming

A visual companion for someone meeting this scenario for the first time. **Not normative**: where
this disagrees with [`spec/scenario-010-self-calibration.md`](../../spec/scenario-010-self-calibration.md),
the specification wins.

> **Not runnable yet.** The scenario is specified, not implemented. The commands below are the
> ones that *will* work once `array/manifold.py`, the self-calibration half of
> `array/calibration.py` and the robust beamformers in `array/adaptive.py` land.

## What it does

Every array method in scenarios 006–009 trusts a steering vector, and the steering vector is a
model. Here the model is wrong in the ways a pilot tone cannot see:

- the elements are a millimetre or so from where the drawing puts them;
- neighbouring elements couple;
- whatever sits between each element and 009's injection point has its own gain and phase, and
  these drift as the site warms up.

The scenario estimates all three from signals from sources already on the air or deployed for the
purpose: the scenario-002 transmitter's line-of-sight signal, aircraft echoes at reported
positions, and calibration beacons at unsurveyed positions. It then hands the refined manifold to
robust beamformers. "Arrival phase" means the **arrival phase** of such a source across the
channels.

## Run it (once implemented)

```bash
# Not yet runnable: these TOMLs arrive with the implementation PR.
uv run python scripts/run_scenario.py scenarios/scenario_010_position_known_source.toml --out out/m1
uv run python scripts/run_scenario.py scenarios/scenario_010_coupling_blind.toml        --out out/m2
uv run python scripts/run_scenario.py scenarios/scenario_010_online_drift.toml          --out out/m3
```

## The geometry

```
                                 N (0 deg)
                                 |
      W ---------------- [ Duke Receiver ] ---------------- E
                    16-element ULA, d = lambda/2 = 15.2955 mm
                    array +x (right, looking out) points to 275 deg
                        /        |         \
                       /         |          \
                      /          |           \
     aircraft track              boresight    Raleigh-Durham transmitter
     166.4-238.6 deg true        185.0 deg    134.99 deg true, el -0.19 deg
     array az -18.6 .. +53.6     (re-pointed  array az -50.01 deg, 19.601 km
     el 3.7-15.7 deg             from 006)    surveyed: 1 m -> 51 urad
```

## The model: three layers on one steering vector

```
  a_true(u) = diag(g) . C . a(u ; p + delta)
                 |      |          |
                 |      |          +-- where the elements really are (M1)
                 |      +------------- mutual coupling, banded symmetric Toeplitz (M2)
                 +-------------------- each channel's gain and phase, and its drift (M2, M3)
```

All three live in one `ArrayManifold` object. The truth is one, and the estimate is another of the
same type, so a test compares them field by field.

## What the data cannot tell you

This is the section to read twice. Some things are invisible to any estimator:

| You cannot see | Because | So the code |
| :--- | :--- | :--- |
| Where the whole array is | Moving every element adds a phase the unknown source phase absorbs | fixes element 0 |
| How it is rotated, if no direction is known | Rotate array and sources together: nothing changes | needs a known source or a reference direction |
| How long a ULA is, if no direction is known | `x · sin(θ)` is all a line array measures | needs two known directions |
| A phase ramp across a ULA | It looks exactly like every source moving together | needs one known direction, or reports DOAs as relative |
| Coupling, gain and direction, from one source | Too many unknowns | needs at least two sources |

Ask an estimator to solve one of these and it raises `NonIdentifiableError`. It does not hand back
a confident wrong answer.

## The three variants

| | **M1** positions | **M2** coupling | **M3** drift |
| :--- | :--- | :--- | :--- |
| What is wrong | Element positions, 1.53 mm rms per axis | Coupling −14 dB / −24 dB, gain 0.5 dB, phase 5° | Phase drift to 14.4° rms, gain to 0.47 dB |
| Calibrated from | Transmitter + 3 aircraft epochs, known directions | 3 calibration beacons, unsurveyed positions, transmitter as anchor | Transmitter, every frame |
| Method | Weighted least squares on arrival phases (Ng & See) | Alternating DOA / coupling / gain (Friedlander & Weiss) | Rank-one subspace tracking (PAST) |
| Result | At the CRB (0.99×) | DOA RMSE 1.00–1.11× the joint CRB | Phase tracked to ≤ 0.23° |

## The picture worth remembering (M1)

SINR loss against the best possible, with the aircraft in the training data and a strong
interferer half a beamwidth away:

```
                                    0 dB          10           20           30
                                    |-------------|------------|------------|
  MPDR, nominal manifold            ##############################  29.97 dB  (self-nulls)
  RCB,  nominal manifold            ####################            19.76 dB  (interferer inside
                                                                               its uncertainty set)
  MPDR, refined manifold            ###                              2.51 dB  (calibration alone)
  RCB,  refined manifold            #                                0.69 dB  (both)
```

Neither calibration nor robustness is enough on its own. Calibration shrinks the uncertainty set
from 7.565 to 0.00206 (in units of ‖a‖² = 16), and only then can RCB be robust *and* still null
the interferer.

And the cautionary row: if the aircraft's reported position is off by 30 m, the position estimate
is 27–47 times worse than its bound. A known direction is only as good as the source's position.
The surveyed transmitter is 51 µrad; the aircraft at 5.3 km is 0.32°.

## M2: why "below 0.1 beamwidth" is not the test

Uncalibrated MUSIC on two beacons 6° apart is biased by 0.32° and 0.24°. That is only 0.05 of a
beamwidth, so a beamwidth criterion says nothing is wrong. It is also **21–28 times the
Cramér–Rao bound**, so the bias, not the noise, sets the error. After calibration the bias is
under 0.005°.

## M3: a calibration goes stale

| Frame | 0 | 30 | 60 | 119 |
| :--- | ---: | ---: | ---: | ---: |
| MPDR, calibration frozen at frame 0 | 0.00 dB | 15.55 dB | 21.69 dB | **27.80 dB** |
| MPDR, PAST-tracked | 0.20 dB | 0.21 dB | 0.19 dB | 0.18 dB |
| RCB, PAST-tracked | 0.13 dB | 0.13 dB | 0.13 dB | 0.13 dB |

The drift is a channel effect, not a geometry one: 10 K of thermal expansion moves the end element
by 0.0017 λ. That is why one known direction, always on the air, is enough to track it.

## Where to look next

- The specification: [`spec/scenario-010-self-calibration.md`](../../spec/scenario-010-self-calibration.md)
- The array frame and steering vector it builds on: [scenario 006](scenario-006-parallel-approach-doa.md)
- The adaptive-weights record and self-nulling from pointing error: [scenario 007](scenario-007-interference.md)
- The pilot tone that runs before this slice: [scenario 009](scenario-009-receiver-chain.md)
- The transmitter's site: [scenario 002](scenario-002-bistatic.md)

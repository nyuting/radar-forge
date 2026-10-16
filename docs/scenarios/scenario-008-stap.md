# Scenario 008 — Airborne ground-clutter ridge and space-time adaptive processing

A visual companion for someone meeting STAP for the first time. **Not normative**: where this
disagrees with [`spec/scenario-008-stap.md`](../../spec/scenario-008-stap.md), the specification
wins.

> **Not runnable yet.** This scenario is specified, not implemented. The commands below are the
> ones that *will* work once `array/stap.py`, the Ward model in `core/clutter.py` and the
> scenario TOMLs land.

## What it does

A radar on an airborne platform looks down at the ground. Each patch of ground returns at a
Doppler set by its direction relative to the platform's motion, so the clutter is a **ridge**
across angle and Doppler, not a blob. A slow mover in the main beam sits right next to that ridge.
A Doppler filter cannot pull it out (the mainlobe clutter has the same Doppler), and a beam
cannot either (the clutter in its own direction is in the beam). A filter that weights the 16
channels and the pulses *together* can put a null along the ridge and keep the target. That is
space-time adaptive processing.

The radar is the one in the public **RASPNet** dataset (an X-band array, 1000 m up, aimed 20 km
south-west). The clutter is synthetic, from Ward's patch model, so every test has an analytic
answer. If you have downloaded a RASPNet scenario, R4 runs the same chain on it. R5 swaps in
the MCARM radar (L-band, 100 m/s), the measured system RASPNet's authors validate against, so
one ridge comes from a real platform's own speed and PRF.

## Run it

```bash
# will work once scenario 008 is implemented
uv run python scripts/run_scenario.py scenarios/scenario_008_stap_raspnet_static.toml   --out out/r1
uv run python scripts/run_scenario.py scenarios/scenario_008_stap_sidelooking_ridge.toml --out out/r2
uv run python scripts/run_scenario.py scenarios/scenario_008_stap_forward_looking.toml   --out out/r3
uv run python scripts/run_scenario.py scenarios/scenario_008_stap_mcarm_sidelooking.toml --out out/r5

# R4 needs your own copy of RASPNet; the test is skipped without it
export RASPNET_DIR=/path/to/RASPNet          # the folder holding CLUTTER/
uv run python scripts/run_scenario.py scenarios/scenario_008_stap_raspnet_data.toml --out out/r4
```

RASPNet is hosted on AFRL SDMS (`https://www.sdms.afrl.af.mil/index.php?collection=raspnet`):
50 sample realisations per scenario download directly; full scenarios are by request. Nothing
from it is ever committed to this repository.

## The geometry

```
                          v_p  -->   (platform velocity; R1: zero, as in RASPNet)
                 [ 48 x 5 element array, 16 subarray channels ]
                  \            h = 1000 m
                   \ depression 2.862 deg at 20 km ground range
                    \
                     \_____________ clutter ring at the cell under test ________
        ground  ......:::::::::::[ mainlobe footprint ]:::::::::::......
                                    ^ slow mover, inside the mainlobe clutter Doppler
```

The one number to remember is **β = 2·v_p·T_r / d**: how far the platform moves in one pulse,
measured in half element spacings. It is the slope of the ridge.

```
      omega (Doppler)                              omega
       +0.5 |            /                        +0.5 |       /      /
            |          /     beta = 1                  |     /      /    beta = 1.3
          0 |------x-----    corner to corner        0 |-x-----/----   runs off the edge at
            |    /           every Doppler holds       | /     /       |theta| = 0.3846 and
       -0.5 |/__________     clutter from ONE     -0.5 |___/________   wraps: some Doppler bins
          -0.5        +0.5   direction               -0.5       +0.5  hold clutter from TWO
                     theta (angle)                                    directions (aliasing)
```

| β | Platform speed at RASPNet's PRF (1100 Hz) | Clutter rank (Brennan) | What the eigenvalues do |
| :--- | ---: | ---: | :--- |
| 0 | 0 (RASPNet as recorded) | 16 | all clutter at zero Doppler |
| 1 | 8.2500 m/s | 31 | 31 big eigenvalues, then a cliff to the noise floor |
| 1.3 | 10.7250 m/s | ≈ 36 | a slope, not a cliff: 40 / 38 / 36 / 35 above 1.01 / 2 / 10 / 100 × noise |

The platform speeds are slow because RASPNet's PRF is kept. Everything here depends only on β
and on normalised frequencies; one unit of normalised Doppler is 16.4886 m/s of radial velocity.

## Two surprises in RASPNet, worth knowing before you read the pictures

- **RASPNet's platform does not move.** The paper says the radar "was assumed as static". Its
  clutter is therefore a line at zero Doppler, with no ridge. The ridge in R2 and R3 comes from
  motion this scenario adds. R1 is RASPNet exactly, and there a plain Doppler filter already
  works for anything moving faster than about 1 m/s.
- **Its 16 channels are 1.5 wavelengths apart.** Each channel is a 3 × 5 subarray, so the channel
  phase centres are 45 mm apart at a 30 mm wavelength. That is spatial aliasing, the angle-domain
  twin of β > 1. The rank tests therefore use element-level channels (R2), where Brennan's rule
  holds exactly.

## The pipeline

```
  Ward clutter patches  -->  covariance R per range bin  -->  synthetic clutter + noise + target
  (core/clutter.py)            (exposes theta, omega, CNR per patch)
                                         |                       (or R4: RASPNet X_clut file)
                                         v
              cube data_rfst (n_pulses, n_range_bins, n_rx)
                                         |
            one range bin  ->  (n_pulses, n_rx) slab  ->  flatten, pulse outer, channel inner
                                         |                 v = kron(temporal, spatial)   [D17]
                                         v
          training bins (guard + train)  ->  covariance estimate
                                         |
                                         v
     optimum | SMI | JDL | PC | MWF | knowledge-aided | sparse recovery   -> weights (returned)
                                         |
                                         v
              apply weights  ->  STAP output map  ->  CFAR  ->  detections
```

The flatten order is the one thing to get right. Flatten the slab the other way round and the
covariance is still perfectly invertible, the filter still runs, and it nulls the wrong ridge.

## What you should see

| Picture | Where | What it shows |
| :--- | :--- | :--- |
| Eigenvalues of R, sorted | R2 | 31 above the floor for β = 1, a cliff; a slope for β = 1.3 |
| Optimum SINR loss vs Doppler | R2 | a −59.59 dB notch at the mainlobe clutter Doppler, −5 dB half-width 0.4988 m/s (the MDV) |
| Same, beam at θ = 0.4 | R2 | β = 1: notch at ω = 0.4; β = 1.3: notch at ω = −0.48, the ridge wrapped round |
| SMI loss vs training size | R2 | −3 dB at K = 2NM = 512 (Reed–Mallett–Brennan); −9.3 dB at K = 288 |
| The slow target, 0.4122 m/s | R2 | STAP output SINR 17.67 dB: detected. Doppler-only −37.82 dB: invisible |
| Range-averaged training | R3 | forward-looking: −9.43 dB mean loss near the clutter; ADC brings it to −1.97 dB |

The slow-target row is the point of the scenario: **the baseline fails where STAP succeeds**.

## The methods, as a ladder

| Rung | Method | Adaptive DOF | Training it needs | Result at K = 288 (slow target) |
| :--- | :--- | ---: | :--- | ---: |
| oracle | optimum, `R⁻¹v` with the true R | 256 | none (it is told R) | 0 dB by definition |
| baseline | SMI | 256 | ≈ 512 | −9.17 dB |
| current | JDL, 3 × 3 beams | 9 | ≈ 18 | −2.26 dB |
| current | eigencanceler, rank 31 | 31 | fewer than SMI | −0.50 dB |
| current | multistage Wiener, 16 stages | 16 | fewer than SMI | −0.49 dB |
| current | knowledge-aided | 256 | few, *if the prior is right* | −0.43 dB at K = 32; −11.81 dB with a 1 % speed error |
| current | sparse recovery (IAA) | 256 | 4 snapshots, *if the ridge is on the grid* | −0.31 dB at K = 4; −40.19 dB off-grid |

The last two rows each come with their failure, on purpose. A method is not understood until you
have seen what breaks it.

## A published moving radar (R5, MCARM)

R5 takes the MCARM parameters tabulated in RASPNet's appendix B: 1.25 GHz, PRF 1984 Hz, 3072 m
altitude, 100 m/s, 16 columns at 0.1092 m, 128 pulses, 630 range bins. Only the parameters are
used. The clutter is still Ward's synthetic model, and no MCARM measured data is read.

| Quantity | R5 value |
| :--- | ---: |
| λ; spacing | 0.239834 m; 0.45531 λ (a little under half a wavelength) |
| β | 0.92314, so the ridge does not alias |
| Clutter rank | 124 eigenvalues above 10 × noise; Brennan's rule says 134 |
| Adaptive DOF NM, and the 2NM training it wants | 2048 and 4096, from **630** range bins |
| MDV (−5 dB) | 4.7583 m/s |
| Slow target at 4.0 m/s | STAP 16.78 dB, detected; Doppler-only −41.40 dB, not detected |
| JDL 3 × 3 trained on 600 bins | −3.18 dB, within 0.2 dB of what it achieves with perfect knowledge |

Two simplifications are stated, not hidden. MCARM records up to 24 channels from its 16 columns
of two four-element subarrays. Which 24, and where they sit, is in a report this spec could not
retrieve, so R5 uses one channel per column. And the side-looking orientation is assumed.

The rank row is the lesson. Spacing under λ/2 leaves part of the angle axis empty, so Brennan's
rule overcounts. Scaling it by the occupied band (0.91063) gives 121.33, closer to the 124 found.
And 630 bins can never train 2048 DOF, which is why every operational STAP is reduced-dimension.

## Range-dependent clutter (R3)

Turn the array so the platform flies along its boresight and steer the beam 60° off the flight
line. The ridge is now a curve whose position depends on elevation, and therefore on range. To
get 512 training bins you must reach from 2.3403 km to 17.6597 km, and across that span the
mainlobe clutter moves in both angle and Doppler. Averaging those bins produces a covariance that
fits no single bin. Doppler warping shifts each bin in Doppler only and barely helps here (−8.54
dB); angle–Doppler compensation shifts both and recovers most of the loss (−1.97 dB).

## Where to look next

- The specification: [`spec/scenario-008-stap.md`](../../spec/scenario-008-stap.md)
- The single-channel range-Doppler slice it builds on: [scenario 001](scenario-001-xband.md)
- The steering-vector convention (scenario 006, D13) and the adaptive-weight API (scenario 007,
  D16), which this scenario uses unchanged.
- RASPNet: arXiv:2406.09638, and the example notebooks at `github.com/shyamven/RASPNet`.
- MCARM: Babu, Torres & Melvin (1996) and Himed, *MCARM/STAP Data Analysis* (AFRL, 1999).

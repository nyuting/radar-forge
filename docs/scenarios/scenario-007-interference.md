# Scenario 007 — Strong RF interference against a weak target

A visual companion for someone meeting adaptive arrays for the first time. **Not normative**: where
this disagrees with [`spec/scenario-007-interference.md`](../../spec/scenario-007-interference.md),
the specification wins.

> **Not runnable yet.** The scenario is specified, not implemented. It also needs scenario 006's
> array geometry and multi-channel cube, which are not built yet either. The commands below are
> the ones that *will* work once both land.

## What it does

A small aircraft (−30 dBsm, about the radar cross-section of a small drone) flies straight at the
Duke Receiver's 16-element receive array, 5.5 km out. Strong wideband RF interferers (for example
a communications transmitter whose emission spills into the radar band) share its range and
Doppler cells, so range and Doppler processing cannot separate them from it. The array's spatial
degrees of freedom can. The scenario shows the classical way to spend them, then the current way,
and where each one breaks.

## Run it

```bash
# Will work once scenarios 006 and 007 are implemented:
uv run python scripts/run_scenario.py scenarios/scenario_007_interference_sidelobe.toml --out out/i1
uv run python scripts/run_scenario.py scenarios/scenario_007_interference_mainlobe_edge.toml --out out/i2
```

Each will write one range-Doppler map per beamforming method, `interference.csv` alongside
`truth.csv`, and the adapted beam patterns. Geometry is fixed: every frame is a fresh noise
realisation of the same picture, which is what lets the tests compare against closed forms.

## The geometry

```
                                   boresight (0 deg)
                                         |
                                         |  small aircraft
                                         |  0 deg, 5516 m, -30 dBsm
                                         |
                                         |    .  mainlobe-edge interferer (I2)
                                         |  .    +4.5 deg, 40 km
                                         |.
                                         |            .  sidelobe wideband interferer
                                         |       .       (I1, I2) +26 deg, 40 km
                                         v  .
        ===o==o==o==o==o==o==o==o==o==o==o==o==o==o==o==o===
             16 elements, half-wavelength spacing
```

| Beam | 3 dB width | First null | Highest sidelobe |
| :--- | ---: | ---: | ---: |
| Uniform (the adaptive methods' starting point) | 6.359° | 7.181° | −13.15 dB |
| Taylor −30 dB (a low-sidelobe beam with no adaptation) | 8.069° | 10.843° | −30.05 dB |

## The numbers that frame everything

| | Value |
| :--- | ---: |
| Aircraft SNR after full processing, no interference | 24.17 dB |
| Wideband interferer, per element, before any processing | **43.65 dB above thermal noise** |
| Interferer over aircraft, per element, per pulse | **68.62 dB** |
| The same, after pulse compression and Doppler (the aircraft gains 37.09 dB, the noise does not) | 31.52 dB |
| SINR on the plain uniform beam | **−11.38 dB** |
| SINR on a −30 dB Taylor beam | −1.08 dB |
| Best possible SINR with the array, closed form | **24.13 dB** (only 0.04 dB lost) |

Integration does not help against noise that fills every cell, and a very good low-sidelobe beam
does not help enough. The array can recover almost everything, if it adapts.

## The pipeline

```
   aircraft paths (two-way)                 wideband interferer (one-way)
   core/signal.py                           core/interference.py
              \                                    /
               +------ 16-channel pulsed cube ----+
                                |
                 matched filter + Doppler FFT, per channel
                                |
           training snapshots: same Doppler bin, 16 range cells
           each side, 2 guard cells, the cell under test EXCLUDED
                                |
  array/adaptive.py  ->  AdaptiveWeights (weights, covariance, diagnostics)
                                |      nothing is applied yet
                                v
                  apply_weights:  y = w^H x
                                |
                      CFAR at Pfa = 1e-6
```

Weights come back as an object you can print and plot before anything is beamformed. That is
decision D16, and it is why every intermediate (the eigenvalues, the blocking matrix, the
mainlobe-interferer angle the method found) is there to inspect.

## The two variants

### I1 — a wideband interferer in a sidelobe

```
   method            what it is                                   SINR (dB)
   ----------------  -------------------------------------------  ---------
   uniform beam      no adaptation                                  -11.38
   Taylor -30 dB     low sidelobes, no adaptation                    -1.08
   SLC, 2 taps       Howells-Applebaum canceller (baseline)          23.67
   SMI-MVDR, K = 32  sample-covariance MVDR (baseline)       21.52 mean
   LCMV / GSC        MVDR with mainlobe-shape constraints            24.125
   optimum           closed form                                     24.13
```

With K = 2N = 32 training snapshots, SMI-MVDR loses on average 2.632 dB against the optimum. That
is the Reed–Mallett–Brennan result, and the tests check it.

**The mistake worth seeing.** Put the aircraft's own cell into the training set and MVDR partly
nulls the aircraft: on average 6.54 dB lost with perfect pointing, 9.57 dB with a 1° pointing error.
INCM (Gu & Leshem) rebuilds the covariance from directions outside the aircraft's sector and gets
23.95 dB back, even with the aircraft in the training data.

### I2 — a second interferer at the mainlobe edge, +4.5°

A null inside the mainlobe has to reshape the mainlobe. MVDR keeps unit gain at 0° but its beam
peak walks to −1.43°. The current method projects the mainlobe interferer out (eigen-projection,
EMP-CMR) and then adds a derivative constraint so that the peak stays at 0°.

```
   pattern near boresight (sketch)

   MVDR                      EMP-CMR + peak constraint
        .--.                          .--.
      /      \                      /      \
     /        \  null              /        \  null
    /          \ at 4.5           /          \ at 4.5
   ----+--------v---           ------+--------v---
      -1.43  0  4.5                  0       4.5
      peak                           peak
```

The price: 2.17 dB of white-noise gain. The return: the response varies by 0.53 dB over ±1°
instead of 2.11 dB, and the beam points where it says it does, which is what angle estimation
downstream needs. Move the interferer inside the 3 dB beamwidth, to +2.5°, and nothing works: the
mainlobe splits. The spec records that as a result, not a bug.

## A caveat worth knowing before you read the pictures

- **The matched filter correlates neighbouring range cells**, so training snapshots in the cube are
  not quite independent. The Reed–Mallett–Brennan check runs on independent snapshots drawn from
  the exact covariance; the end-to-end test checks detection instead.

## Where to look next

- The specification: [`spec/scenario-007-interference.md`](../../spec/scenario-007-interference.md)
- The array, steering vectors and cube it builds on: scenario 006's array, `spec/scenario-006-parallel-approach-doa.md`
- The single-channel slice whose waveform it reuses (S2): [scenario 001](scenario-001-xband.md)

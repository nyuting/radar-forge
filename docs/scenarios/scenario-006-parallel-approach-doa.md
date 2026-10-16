# Scenario 006 — Two aircraft on parallel approaches in one range-Doppler cell

A visual companion for someone meeting array processing in this repository for the first time.
**Not normative**: where this disagrees with
[`spec/scenario-006-parallel-approach-doa.md`](../../spec/scenario-006-parallel-approach-doa.md),
the specification wins.

> **Not runnable yet.** The scenario is specified, not implemented. The commands below are the
> ones that *will* work once `array/` and the scenario TOMLs land.

## What it does

The scenario-001 radar at the Duke Receiver gets a 16-channel receive array. Aircraft A flies
the real track in `data/flight_coordinates.csv`; aircraft B flies a path offset laterally from it,
as on a closely spaced parallel approach, at the same range and the same radial velocity. On the
range-Doppler map they are **one cell**. The array's 16 channels are the only thing that can tell
them apart, and the scenario shows which methods manage it and why the textbook one does not.

## Run it

Once implemented:

```bash
uv run python scripts/run_scenario.py scenarios/scenario_006_parallel_resolved.toml --out out/f1
uv run python scripts/run_scenario.py scenarios/scenario_006_parallel_subrayleigh.toml --out out/f2
```

F3, the separation and SNR sweep, is a notebook: `notebooks/006_parallel_approach_doa.ipynb`.

## The geometry

```
                           aircraft B             aircraft A
                                  *                    *
                                   \   delta_az       /
                                    \<------------->/      R = 16.6 - 18.4 km
                                     \             /
                                        v       v
          x <---------------- [ 16-element ULA, d = lambda/2 ] ----> boresight
                               aperture 0.245 m, boresight 235.8 deg true
```

The array's Rayleigh width is `λ/(n_rx·d)` = 0.125 in direction cosine, 7.18° at broadside. At
17 km that is about **2 km** across the line of sight. The aperture is a quarter of a metre, so
"close" means something very different to the array than to the pilots:

| Lateral offset | Rayleigh widths at 16.64 km |
| :--- | ---: |
| 15 m (less than one airliner's wingspan) | 0.0072 |
| 500 m | 0.2403 |
| 1000 m | 0.4805 |
| 2000 m | 0.9592 |

Closely spaced parallel runways are roughly 760–1300 m apart (Raleigh-Durham's are about 1 km
apart): roughly 0.37–0.62 Rayleigh widths at this range.

## Why the textbook method fails here

Every cell the processor looks at contains *both* echoes multiplied by the *same* range and
Doppler response, because the two aircraft share range and Doppler:

```
   x_k = g_k (a(u_1) s_1 + a(u_2) s_2) + noise          same g_k for both
```

Average as many snapshots as you like: the signal part of the covariance is one vector times its
own conjugate. **Rank one.** MUSIC looks for a two-dimensional signal subspace and finds one
dimension of signal plus one of noise.

```
  covariance eigenvalues (noiseless, co-cell pair)

     lambda_1  ##########################################
     lambda_2  .                                           <- 1e-12 of lambda_1
     ...       .
```

The fixes, in order of how current they are:

| Method | What it does | Cost |
| :--- | :--- | :--- |
| Spatial smoothing (Shan, Wax, Kailath 1985) | Average the covariances of overlapping 12-channel subarrays; each sees the pair with a different relative phase | Aperture: 12 channels, not 16 |
| Forward-backward (Pillai, Kwon 1989) | Average with the conjugate-reversed array | Fails when the echoes are in phase or anti-phase |
| IAA (Yardibi et al. 2010) | Fit a power at every angle to the **one** snapshot, iterating Capon-style weights | No covariance needed; keeps the full aperture |
| Deterministic ML / RELAX (Li, Stoica 1996) | Fit two plane waves exactly | The oracle: as good as the bound allows |

## What the conventional beam sees

```
   equal powers, in phase              equal powers, anti-phase
          ___                              _          _
         /   \                            / \        / \
        /     \                          /   \  /\  /   \
   ____/   ^   \____                ____/  ^  \/  \/  ^  \____
         u1 | u2                       peak  u1 null u2  peak
       one peak, at the                two peaks, both OUTSIDE
         midpoint                       the true pair
```

Below a Rayleigh width the conventional beam and monopulse see one blob. Averaged over looks,
monopulse points at the **power-weighted centroid**. Look by look, it wanders with the relative
phase of the echoes, sometimes outside the pair altogether.

## The three variants

| | **F1** | **F2** | **F3** |
| :--- | :--- | :--- | :--- |
| Separation | 2× Rayleigh | 0.5× Rayleigh | 0.1× to 2× |
| SNR (per channel, one cell) | 38–40 dB (link budget) | 38–40 dB | 0 to 40 dB |
| What you should see | Everything resolves; raw MUSIC is still biased ~90× the bound | Raw MUSIC resolves 7 % of the time; FBSS, IAA and ML 100 % | Probability of resolution and RMSE against the CRB |

## The bound decides, not the algorithm

To resolve two equal echoes 0.1 Rayleigh apart takes about 22 dB of per-channel SNR when they
are in phase and 53 dB when they are in anti-phase. Two aircraft 15 m apart at 17 km need over
67 dB. The radar has 38–40 dB. No estimator closes that gap; a bigger aperture does.

## Conventions this scenario fixes for every array scenario

- Cube `(n_pulses, n_samples, n_rx)`, receive channel last.
- Array frame: x along the face, y boresight, z up; phase centre at the array centre.
- Steering vector `a_n = exp(+j 2π p_n·u / λ)`: the element nearer the source leads. Weights
  apply as `y = wᴴx`.

Scenarios 007–010 cite these as "scenario-006 D13".

## Where to look next

- The specification: [`spec/scenario-006-parallel-approach-doa.md`](../../spec/scenario-006-parallel-approach-doa.md)
- The single-channel radar it extends: [scenario 001](scenario-001-xband.md)

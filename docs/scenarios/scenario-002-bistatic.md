# Scenario 002 — Bistatic radar, Raleigh-Durham to the Duke Receiver

A visual companion for someone running this scenario for the first time. **Not normative**: where
this disagrees with [`spec/scenario-002-bistatic.md`](../../spec/scenario-002-bistatic.md), the
specification wins.

## What it does

The same aircraft and the same receive site as scenario 001, but the transmitter is 19.601 km away
at Raleigh-Durham Airport. The two propagation ranges are now independent numbers, and that single
change is enough to make range resolution depend on where the target *is*.

## Run it

```bash
uv run python scripts/run_scenario.py scenarios/scenario_002_bistatic_xband.toml --out out/b1
uv run python scripts/run_scenario.py scenarios/scenario_002_bistatic_sband.toml --out out/b2
```

The default window starts at `start_time_s = 13329.0` — not at 0, as scenario 001 does. It is the
120 s window that maximises the bistatic angle, which is the thing worth looking at.

## The geometry

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

`β` is the angle the two sites subtend **at the target**. Everything interesting follows from it.

| Quantity | Full track | Default window |
| :--- | ---: | ---: |
| Transmit range `R_t` | 13.502 – 30.203 km | |
| Receive range `R_r` | 5.326 – 22.010 km | |
| Bistatic mean range `(R_t+R_r)/2` | 10.696 – 25.985 km | 10.74 – 11.81 km |
| Bistatic angle β | 40.3° – 129.3° | 109.9° – 129.3° |
| Range resolution | 79.85 – 175.21 m | 130.5 – 175.2 m |
| Degradation vs. the β = 0 floor | 1.07× – **2.34×** | 1.74× – **2.34×** |

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

## What changed in the code, and what did not

```
                            MONOSTATIC              BISTATIC
  delay                     2R/c                    (R_t + R_r)/c
  map's range axis means    slant range R           (R_t + R_r)/2, "bistatic mean range"
  Doppler                   2v/lambda               2*v_bisector/lambda
  PropagationPaths          -------- IDENTICAL DATACLASS --------
  core/dsp.py               -------- IDENTICAL CODE --------
  rd_map.py                 axis label is the only difference
```

That the path dataclass survived a second, quite different geometry unchanged is the strongest
evidence so far that decision D1 in `spec/structure.md` was drawn in the right place.

Selecting the siting is data, not a flag: `pipelines/scenarios.py` builds a `BistaticRadar` when
the TOML carries a `[transmitter_site]` table. Delete that table and the same file is a monostatic
scenario at the Duke Receiver.

## The two variants

| | **B1** `bistatic-xband` | **B2** `bistatic-sband` |
| :--- | :--- | :--- |
| Carrier | 9.8 GHz | 2.8 GHz |
| λ | 30.591 mm | 107.069 mm |
| Unambiguous velocity | ±7.6478 m/s | **±26.7672 m/s** |
| Folds over the window | ≈8.6× | ≈2.5× |
| Realistic? | No — a fiction | Yes — terminal ATC radar is S-band |

B1 is waveform-identical to scenario 001's S1, so **geometry is the only variable** between the two
scenarios; that is what makes the comparison clean. B2 is the physically honest one. Run both and
the pair shows exactly what the carrier buys you: same trajectory, same processing, 3.5× more
unambiguous velocity.

## Naming

- `range_tx_m` — transmitter site to target.
- `range_rx_m` — target to receiver site.

The `tx`/`rx` token names **the site the range is measured to**, the same way it names the antenna
in `gain_tx_linear`. Never "legs" or "paths": a *path* is the whole transmitter-target-receiver
route, and a *burst* is a waveform configuration.

There is no `radial_velocity_mps` on a bistatic track, and the omission is deliberate: a bistatic
target has two radial velocities and the Doppler shift measures neither. What it measures is
`bisector_velocity_mps`.

## Two caveats worth knowing before you read the pictures

- **The RCS is a single number** used as the bistatic RCS at every β, which the
  monostatic-equivalence theorem licenses only for small β. At β = 129° the **absolute** power
  levels are an approximation, so the acceptance criteria test geometry and say nothing about SNR.
- **Resolution degrading is not the peak getting wider.** A point target has one delay, so its peak
  width is set by the window and the transform length. What degrades is the *mapping from space to
  range*: iso-range surfaces are ellipsoids with the two sites at the foci, and they spread out
  near the baseline.

## Where to look next

- The specification: [`spec/scenario-002-bistatic.md`](../../spec/scenario-002-bistatic.md)
- The monostatic slice it builds on: [scenario 001](scenario-001-xband.md)
- Tracking, which is stacked on scenario 001 rather than on this one:
  [scenario 003](scenario-003-tracking.md)

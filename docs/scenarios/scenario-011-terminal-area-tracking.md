# Scenario 011 — Dense terminal-area traffic: crossings, clutter, manoeuvres and closely spaced aircraft

A visual companion for someone meeting multi-target tracking in this repository for the first
time. **Not normative**: where this disagrees with
[`spec/scenario-011-terminal-area-tracking.md`](../../spec/scenario-011-terminal-area-tracking.md),
the specification wins.

> **Not runnable yet.** The scenario is specified, not implemented. The commands below are the
> ones that *will* work once the IMM, JPDA, MHT and PMBM code and the scenario TOMLs land.

## What it does

Scenario 003 tracked one aircraft. This one tracks the civil traffic around Raleigh-Durham
airport, seen by an airport-surveillance-class radar at the Duke Receiver that rotates every
4.8 s. The traffic includes:

- arrivals on two parallel finals, and arrivals turning onto final;
- a go-around;
- four aircraft in a holding stack;
- two overflights;
- two light aircraft flying a circuit;
- an air-ambulance helicopter;
- a flock of birds.

All of it is seen through ground-clutter residue, a drifting rain cell and a scatter of fixed
reflectors.

The radar reports **plots**: one range and azimuth per resolved echo per rotation, with no
Doppler. Everything the trackers know comes from those plots, and the scenario asks four
questions:

1. Can the tracker follow a turn without losing the aircraft?
2. Can it keep two aircraft apart where they cross?
3. Can it keep them apart where they fly close together for minutes?
4. How much clutter can it take before any of that fails?

## Run it

Once implemented:

```bash
uv run python scripts/run_scenario.py scenarios/scenario_011_manoeuvre.toml --out out/v1
uv run python scripts/run_scenario.py scenarios/scenario_011_crossing.toml --out out/v2
uv run python scripts/run_scenario.py scenarios/scenario_011_terminal_dense.toml --out out/v4
```

V3, the separation and clutter sweep, is a notebook: `notebooks/011_terminal_area_tracking.ipynb`.

## The scene

```
   N (km)
   16 |               B  birds, 12 m/s
   12 |  L  light aircraft circuit
   10 |  T1 ====================================> overflight     H  holding stack (4 aircraft)
    4 |                                       R  rain cell, drifting over the stack
    0 +--D  Duke Receiver (the radar) --------------------------- A1, A2 on parallel finals
      |   HE helicopter: hover, then transit                     /
   -8 |                                     T2 crosses A1 at 30 deg
  -14 |                         *  RDU; A3 goes around
      +----------------------------------------------------------------------> E (km)
```

Every difficulty has a cause you can point at:

| Where | What makes it hard |
| :--- | :--- |
| The holding stack | Four aircraft on one racetrack at different heights. The radar cannot see height, so to it they overlap. Two of them are 4 s apart and sit in one resolution cell for ten scans |
| T2 over A1 | A crossing at 30°: the case where a hard-assignment tracker swaps the two tracks |
| Turns | 180° turns every minute in the stack, base and final turns at 3 °/s, the go-around |
| The helicopter, the circuit | Moving across the beam or hovering, they fall into the moving-target filter's notch and are missed |
| The birds | Twelve real echoes in less than one resolution cell: not false alarms, but not aircraft |
| The rain cell | Three clutter plots in every gate it covers, and it drifts over the stack |

## What "closely spaced" means to this radar

The radar's resolution cell is one range cell (115.75 m) by one beamwidth (1.4°). The beam widens
with range:

| Range | 10 km | 20 km | 26.9 km (the stack) | 40 km |
| :--- | ---: | ---: | ---: | ---: |
| Beam width across the line of sight | 244 m | 489 m | 658 m | 977 m |

The two parallel finals are 1 km apart, but from Duke that 1 km lies mostly along the line of
sight, 5–9 range cells. So **they are not closely spaced**. The holding stack is, because its
aircraft share a racetrack in the ground plane.

Two echoes closer than about one cell come out as **one merged plot**, at their centroid. The
model is Koch and van Keuk's: the chance of a merge falls off as `exp(−d²/2)` with the separation
`d` in cells.

## Why one motion model is not enough

A standard-rate turn at 72 m/s is 3.77 m/s² sideways. In one 4.8 s scan that is 43 m off a
straight-line prediction, about one standard deviation. In three scans it is 391 m, which is
gone.

```
   CV filter, quiet                    CV filter, loose                 IMM (quiet CV + turn + agile)
   -------------                       -------------                    -------------
   straight: smooth, accurate          straight: noisy                  straight: like quiet CV
   turn:     lags, then loses it       turn:     follows                turn:     switches to turn mode
                                                                                  within ~3 scans
```

IMM runs all three models side by side and weights them by how well each has predicted lately.
Its mode probabilities are written to `tracks.csv`, so you can watch the turn mode light up as an
aircraft starts its turn and fade as it rolls out. The longer the radar takes to come round, the
more the IMM helps, so V1 runs at 1.2, 2.4 and 4.8 s per rotation.

## Coalescence, in one picture

```
   truth:          A  o-------------------->           two aircraft, one cell apart
                   B  o-------------------->

   GNN:            A  o----------\  /------>           hard assignment: at the crossing
                   B  o----------/  \------>           it may take the wrong plot, and swap

   JPDA:           A  o--------._                      soft assignment: each track takes a
                   B  o--------'  `========>           share of both plots, and the two
                                                       estimates are pulled together

   JPDA* / Set-JPDA, MHT:   two tracks, kept apart
```

| Tracker | What it does with an ambiguous plot | What goes wrong |
| :--- | :--- | :--- |
| GNN (scenario 003) | Gives it to exactly one track | Swaps at a shallow crossing |
| JPDA | Shares it among tracks, by probability | **Coalescence**: two tracks merge onto the midpoint |
| JPDA\*, Set-JPDA | JPDA with the coalescing hypotheses pruned, or with identities dropped inside the cluster | Set-JPDA can lose track identity in the cluster |
| JIPDA | JPDA plus a probability that the track exists | That probability, not a hit count, confirms and deletes |
| MHT | Keeps several association histories for 3 scans, then decides | Memory and run time; the estimate shown at each scan can later be revised |
| PMBM | Models the set of targets as a whole | The reference: the others are approximations of it |

## The variants

| | **V1** | **V2** | **V3** | **V4** | **V5** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| What | Manoeuvres, one aircraft at a time | Crossings at 90°, 30°, 10° | Pairs at fixed separations, in clutter | The whole scene | The real IQ chain |
| Clutter per gate | 0.01 | 1 | 0.1, 1, 3 | 0.01 background, 0.3 near the radar, 3 in the rain | Thermal |
| What you should see | IMM beats either CV filter, more so as the rotation slows | GNN swaps at 10°; MHT does not | A resolution curve per tracker, PMBM on top | GOSPA split into position error, missed and false | The plot model's Pd, false-alarm rate and merging match the real detector |

## How it is scored

With many aircraft, "the error" is no longer one number. The scenario uses **GOSPA**, which
splits the error into three parts you can read separately:

- **localisation**: how far the tracks are from the aircraft they follow;
- **missed**: aircraft with no track;
- **false**: tracks with no aircraft.

It also uses **trajectory GOSPA**, which additionally charges for track switches. A track on the
bird flock counts as false, because the score is over aircraft.

## Where to look next

- The specification: [`spec/scenario-011-terminal-area-tracking.md`](../../spec/scenario-011-terminal-area-tracking.md)
- The tracker architecture it extends: [`spec/tracker-001.md`](../../spec/tracker-001.md)
- The single-target tracking scenario before it: [scenario 003](scenario-003-tracking.md)

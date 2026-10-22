# Scenario 005 — A continuous-wave radar that hears its own transmitter

A visual companion for someone meeting this scenario for the first time. **Not normative**: where
this disagrees with
[`spec/scenario-005-leakage-cancellation.md`](../../spec/scenario-005-leakage-cancellation.md),
the specification wins.

> **Not runnable yet.** The scenario is specified, not implemented. The commands below are the ones
> that *will* work once `core/clutter.py` and the scenario files land.

## What it does

The site, aircraft and window are scenario 001's. The radar now transmits a continuous noise code,
8 MHz wide at 9.8 GHz, instead of chirps. A continuous transmitter never switches off, so the
receiver hears it the whole time it listens:

- **leakage**: transmit power coupling straight across to the receive antenna;
- **near-range clutter**: buildings and terrain 300 m – 3 km away;
- **the aircraft echo**, at 17 km, 71 dB below the two of them.

The radar knows the code it sent. It models the leakage and clutter as delayed, scaled copies of
that code, fits them by least squares, and subtracts the fit. That is Widrow's two-input noise
canceller, with the radar's own transmit waveform as the reference input. Full-duplex radios do
the same with their own transmission.

## Run it

These will work once the scenario is implemented:

```bash
uv run python scripts/run_scenario.py scenarios/scenario_005_static_leakage.toml --out out/p1
uv run python scripts/run_scenario.py scenarios/scenario_005_varying_clutter.toml --out out/p2
uv run python scripts/run_scenario.py scenarios/scenario_005_distorted_transmitter.toml --out out/p3
```

The window is 001's (`start_time_s = 0.0`, 120 s): the aircraft is at 16.6 – 18.4 km.

## The geometry

```
                                   [ Light aircraft, 1500 m, 16.6 - 18.4 km ]
                                                ^
                                                |  echo, -31 dB per sample
                                                |
   [ Duke Receiver ]  tx antenna ---------------+
        |    ^                                  |
        |    |  leakage (lags 0, 1, 2)          |
        v    |                                  |
     rx antenna <------ clutter at 300 m, 750 m, 1.5 km, 3.0 km (lags 16, 40, 80, 160)
```

Everything is timed from the transmit instant, so lag `m` is range `m·c/(2f_s)` = `m × 18.737 m`.

## Why the aircraft is invisible without cancellation

```
   received stream, per sample       range-Doppler map
   (dB re noise)                     (dB re the map's noise floor)

   +40.0 ... +20.0  leakage          +99.1  leakage peak (one cell)
   +30.0 ... +12.0  clutter          +40.9  its sidelobe floor: EVERY cell
     0.0  noise                      +27.2  aircraft peak         <-- 13.7 dB under the floor
   -31.1  aircraft                     0.0  noise floor
```

A noise code correlates to a sharp peak, with a flat sidelobe floor 59.1 dB below it everywhere on
the map. The leakage and clutter are 71 dB above the aircraft, so their floor covers it. No range or
Doppler gate helps, because the floor is everywhere. The fix has to remove them from the stream
*before* correlation.

## The pipeline

```
  received stream (n_stream_samples,)   +   the radar's own code (n_stream_samples,)
     |                                           |
     |  least squares: subtract the best fit of the stream by delayed copies of the code
     |                 (K = 192 taps, lags 0..191); this is a projection
     v
  residual stream
     |
     |  batched correlation: correlate each 50 us batch with the code over 1024 lags,
     |                       then FFT across the 2048 batches
     v
  range-Doppler map  ->  CFAR (pfa 1e-6)
```

**Cancellation is a notch in Doppler, but only inside the taps.** The projection removes what is
constant over the fit inside lags 0–191 (0–3.6 km). The aircraft at lag 887–980 is outside the
taps, so it is never touched. A slow target inside 3.6 km would lose one Doppler bin to the notch
for a full-CPI fit, and `±1/T_B` for a fit per batch of length `T_B`.

## The three variants

| | **P1** static leakage | **P2** varying clutter | **P3** distorted transmitter |
| :--- | :--- | :--- | :--- |
| What changes | Nothing: the channel is static | A 2 Hz copy of the nearest clutter, as strong as it | The power amplifier compresses: every path carries `x + a₃|x|²x`, not `x` |
| No cancellation | SINR −14.3 to −13.7 dB: **missed** | **missed** | **missed** |
| Baseline | Least squares, K = 192: cancels to the noise, detected | Least squares over the full CPI: the 2 Hz part escapes the one-bin notch, SINR 5.4 – 6.0 dB, **missed** | Linear basis: stops at the amplifier's distortion floor, SINR 4.4 – 5.0 dB, **missed** |
| Current | — | Batched (6.4 ms): 1.9 dB from clean, detected. Sliding (50 µs / 6.45 ms): also removes the batched fit's 156 Hz parasitic lines, 28 dB lower | Parallel-Hammerstein basis (`x` and `|x|²x` at every tap): back to clean, detected |
| The lesson | Taps must span the clutter: K = 96 misses the 3 km echo and loses 12.3 dB | Short fits widen the notch, at the cost of slow targets close in | The basis must model what the hardware did |

## The numbers worth remembering

| Quantity | Value |
| :--- | ---: |
| Processing gain, N = 819 200 samples | 59.13 dB |
| Range bin / radial velocity bin | 18.737 m / 0.14937 m/s |
| Cancellation depth, P1, K = 192 (predicted; prototype) | 40.946 dB; 40.943 dB |
| Ceiling with the digital code as the basis | (1 + INR)/(1 − K/N): cancel to the noise |
| Amplifier distortion a linear basis cannot reach, a₃ = −0.07 | −18.8 dB of every path |
| Batched 6.4 ms notch at 9.8 Hz / 78.1 Hz / 156.3 Hz | −37.9 / −4.5 / 0.0 dB |

## Where to look next

- The specification: [`spec/scenario-005-leakage-cancellation.md`](../../spec/scenario-005-leakage-cancellation.md)
- Where the idea comes from: Widrow et al., "Adaptive noise cancelling: principles and applications", *Proc. IEEE*, 1975
- The radar and site it reuses: [scenario 001](scenario-001-xband.md)

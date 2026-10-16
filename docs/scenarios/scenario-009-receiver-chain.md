# Scenario 009 — Receiver chain: front end, DDC, channel impairments and calibration

A visual companion for someone meeting this scenario for the first time. **Not normative**: where
this disagrees with [`spec/scenario-009-receiver-chain.md`](../../spec/scenario-009-receiver-chain.md),
the specification wins.

> **Not runnable yet.** The scenario is specified, not implemented. The commands below are the
> ones that *will* work once `core/frontend.py`, `array/impairments.py` and
> `array/calibration.py` land.

## What it does

Scenarios 001–003 write the samples a perfect receiver would deliver. This one builds the receiver
between the antenna and the cube, for each of the 16 channels of the shared ground array. It shows
four things:

- where the noise goes when you decimate;
- what happens to a strong signal just outside your channel;
- how channel mismatch ruins a low-sidelobe beam, and how one injected reference repairs it;
- what an analog IQ demodulator and a noisy oscillator do, and which of those effects calibration
  can remove.

## Run it (once implemented)

```bash
# Not yet runnable: these TOMLs arrive with the implementation PR.
uv run python scripts/run_scenario.py scenarios/scenario_009_frontend_ddc.toml      --out out/c1
uv run python scripts/run_scenario.py scenarios/scenario_009_channel_mismatch.toml  --out out/c2
uv run python scripts/run_scenario.py scenarios/scenario_009_iq_phase_noise.toml    --out out/c3
```

Delete the `[front_end]` and `[impairments]` tables from any of them and you get the ideal
direct-baseband receiver of scenario 001, bit for bit.

## The chain, with its sample rates

```
  20 MHz real          20 MHz complex             2.5 MHz complex
  ----------           --------------             ---------------
  IF at 5 MHz   --NCO-->  signal at 0 Hz   --LPF + keep 1 in 8-->  cube (256, 100, 16)
  + noise 0..10 MHz       noise over 20 MHz          noise over ENBW 2.31 MHz
  + CW at 8 MHz           CW at +3 MHz and +7 MHz    CW 84.9 dB down
```

What the spectrum looks like at each step, one channel:

```
  ADC input (real, 0 .. 10 MHz)
    noise  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    signal                 [####]                                  4..6 MHz
    CW                                     |                       8 MHz, +50 dB
           0               5               8              10 MHz

  After the NCO (complex, -10 .. +10 MHz)
    signal                 [####]                                  -1..+1 MHz
    CW                           |     |                           +3, +7 MHz
    filter  ________________/    \______________________________   pass +-1, stop >= 1.5 MHz
           -10             0    +3    +7              +10 MHz

  After decimating by 8 (complex, -1.25 .. +1.25 MHz)
    with the filter:     signal intact, CW 84.9 dB down, noise 6.37 dB lower
    without it:          the CW lands at +-0.5 MHz, on top of the signal
```

## The numbers worth remembering

| Quantity | Value | Why |
| :--- | ---: | :--- |
| Decimation | 8 | 20 MHz → 2.5 MHz |
| Filter | 145-tap equiripple | 0.096 dB ripple, 84.8 dB stopband |
| Same spec, window method | 202 taps | Equiripple is the design; the window method is the baseline |
| Polyphase cost | 145 MACs per output | Filter-then-discard does 1160, and the outputs are identical |
| SNR gain | **6.37 dB** | `10 log10 8` = 9.03, minus 3.01 because a real ADC's noise occupies only `fs/2`, plus 0.35 because ENBW is 0.92 `fs_out` |
| Interferer suppression | 84.9 dB | Two stopband lines per real tone, so the bound is `A_s − 3 dB` |
| Output noise vs the ideal receiver | +0.62 dB | ENBW 2.31 MHz, against the 2 MHz the ideal receiver assumes |

## Channel mismatch: the sidelobe picture (C2)

Every channel has its own gain (1 dB rms), phase (20° rms) and group delay (20 ns rms). The beam
is a 35 dB Taylor:

```
   PSL (dB, median over 400 draws)
   -10 |
   -15 |  ###  uncalibrated            -15.4
   -20 |
   -25 |        ###  1-tone pilot,     -23.6   <- the delay is still there
   -30 |             at band edge
   -35 |  - - - - - - - - - - - - - -  ideal -34.7
       |              ###  9-tone pilot, band edge  -34.1
```

A single tone measures each channel at one frequency, and a group delay is a phase that changes
with frequency. A pilot that spans the band measures the slope as well, and the sidelobes come back
to within about 1 dB of ideal. The pilot is one source split to every channel, and channel 0 is
the reference. The calibration routine returns gains and delays. It never edits the data; the
beamformer is handed them.

## IQ imbalance and phase noise (C3)

**IQ imbalance** comes from two analog mixers that do not quite match. At 0.5 dB and 3° the image
of a tone at +f appears at −f, **28.2 dB** down:

```
   IRR = (1 + 2g cos(phi) + g^2) / (1 - 2g cos(phi) + g^2),   g = Q/I gain RATIO
```

Sample at IF and down-convert digitally, and there is no image: the NCO is exact, and the image
falls in the filter's stopband. That is half the reason radars sample at IF.

The current fix is blind: a proper complex signal has `E[z²] = 0`, so choose the compensator that
restores it. It needs no pilot and keeps up with drift. A pilot estimate taken before the hardware
drifted to 0.8 dB and 5° leaves 32 dB IRR. Re-estimating blind on the same data reaches about
49 dB.

**Phase noise** from a common LO is applied twice, once on transmit and once on receive, `τ`
apart. What survives is filtered by `4 sin²(π f τ)`:

| Offset | `4 sin²(πfτ)`, τ = 10 µs (1.5 km) | τ = 160 µs (24.0 km) |
| :--- | ---: | ---: |
| 195 Hz | −38.2 dB | −14.2 dB |
| 977 Hz | −24.2 dB | −0.5 dB |
| 4.98 kHz | −10.1 dB | +1.5 dB |
| 6.25 kHz | −8.2 dB | **null** (`1/τ`) |

Close targets barely see the oscillator's close-in noise. At 1.5 km and 195 Hz offset the skirt is
37.6 dB below what two independent oscillators would give. Far targets see all of it.

What calibration can and cannot remove:

| Error | Removable? |
| :--- | :--- |
| Gain and phase per channel | yes, any pilot |
| Group delay per channel | yes, with a pilot that spans the band |
| IQ imbalance | yes; blind if it drifts |
| Per-channel, per-pulse LO phase | yes, the part that differs between channels (a pilot gate each PRI) |
| Common LO phase noise at the target's delay | **no**: buy a better oscillator |

Common LO noise is the same in every channel, so it never touches the beam pattern.

## Where to look next

- The specification: [`spec/scenario-009-receiver-chain.md`](../../spec/scenario-009-receiver-chain.md)
- The waveform it reuses (S2): [scenario 001](scenario-001-xband.md)
- Mutual coupling, element-position errors and calibration without an injected reference are
  scenario 010's.

# Scenario 004 — High-speed target: keystone, polynomial phase and the datachip

A visual companion for someone meeting this scenario for the first time. **Not normative**: where
this disagrees with [`spec/scenario-004-keystone.md`](../../spec/scenario-004-keystone.md), the
specification wins.

> **Not runnable yet.** The scenario is specified, not implemented. `core/migration.py`,
> `core/datachip.py` and the three TOML files do not exist. The command below is the one that
> *will* work once they do.

## What it does

A wideband X-band tracking radar sited downrange of a sounding-rocket launch pad watches one fast
target: a sounding rocket flying straight towards it along its line of sight. The bandwidth is
500 MHz, so
a range bin is only 25 cm wide, and in one 20 ms burst of 1024 pulses the target crosses
**28 of them**. Ordinary range-Doppler processing assumes a target stays in its bin. This one
doesn't, so the energy is spread across the bins and the peak drops by **26.5 dB**, and a target
that should stand 29 dB above the noise disappears into it.

The scenario puts that energy back, one cause at a time, and then measures the target more finely
than one bin: range to millimetres, speed to centimetres per second, acceleration to a few m/s².

## Run it (once implemented)

```bash
# Not yet runnable: these files arrive with the implementation PR.
uv run python scripts/run_scenario.py scenarios/scenario_004_keystone_k1.toml --out out/k1
uv run python scripts/run_scenario.py scenarios/scenario_004_keystone_k2.toml --out out/k2
uv run python scripts/run_scenario.py scenarios/scenario_004_keystone_k3.toml --out out/k3
```

## Why scenario 001 could never show this

| | Scenario 001 S2 | Scenario 004 |
| :--- | ---: | ---: |
| Bandwidth | 2 MHz | 500 MHz |
| Range bin | 74.95 m | 0.25 m |
| Pulses per burst | 256 | 1024 |
| Walk over one burst | **0.011 bin** at 80 m/s | **27.9 bins** at 340 m/s |

How far a target can walk while its Doppler stays unambiguous is set by the *fractional*
bandwidth: at most `N·B/(2·f0)` resolution cells, which is 26 here and about 0.03 for scenario 001.
Making the target faster does not help, because its Doppler folds first.

## The three variants

```
  K1  constant velocity      ##                     walk only
          340 m/s              ##
                                 ##                 fix: keystone
  K2  + 15 g acceleration    ## ## ##               walk + Doppler smear (4 bins)
          340 m/s, 147 m/s^2     ## ## ##           fix: keystone, then Doppler-rate filter
  K3  Doppler-ambiguous      ##########             walk of 90 bins, Doppler folded once
          1100 m/s                 ##########       fix: keystone with fold factor F = 1
```

| | **K1** | **K2** | **K3** |
| :--- | ---: | ---: | ---: |
| Sounding rocket | coasting | under thrust | after burnout |
| Speed, acceleration | 340 m/s, 0 | 340 m/s, 15 g | 1100 m/s, 0 |
| Walk (bins) | 27.9 | 27.9 | 90.2 |
| Doppler spread (bins) | 0 | 4.0 | 0 |
| Fold factor | 0 | 0 | **1** |
| Loss, no compensation | −26.5 dB | −25.5 dB | −31.8 dB |
| Loss, compensated | −0.05 dB | −0.06 dB | −0.06 dB |

(The remaining 0.05 dB is the measuring grid, not the method.)

## The pipeline, one fix at a time

```
  matched filter  ->  data_rfst  (range x slow time)
                          |
          +---------------+-----------------------------+
          |                                             |
     Doppler FFT                                 core/migration.py
     (the BASELINE: MTD)                          1. range spectrum: FFT along range
     target smeared over                          2. KEYSTONE: stretch slow time by
     28 bins, -26.5 dB                               f0/(f0+f) at each range frequency f
                                                  3. fold factor F (K3)
                                                  4. Doppler-rate matched filter (K2)
                                                  5. back to data_rfst
                                                        |
                                                   Doppler FFT -> CFAR -> detection
                                                        |
                                                 core/datachip.py
                                                   crop a small chip around it
                                                   fine range, speed, acceleration
                                                   + their Cramer-Rao bounds
```

**Why the keystone works.** In the range-frequency domain, a target's phase is
`−4π(f0 + f)·R(t)/c`. The walk is the part that depends on both `f` and `t`: higher frequencies
see the motion as more cycles. Stretching slow time by `f0/(f0 + f)` at each frequency makes every
frequency see the motion as `f0` would. The `f·t` coupling is then gone, for every target at
once, whatever its speed. Plotted as frequency against slow time, the resampling grid looks like
a keystone, which is where the name comes from.

**Why K2 needs a second step.** At 15 g the range history bends by only 7.7 mm, 3% of a bin, so
range is not the problem. The phase is: 3.2 rad of quadratic phase across the burst smears the
Doppler peak over four bins. The keystone fixes the walk and leaves the smear (−4.1 dB). Dechirping
fixes the smear and leaves the walk (−26.5 dB). You need both, in that order.

**Why K3 needs the fold factor.** At 1100 m/s the Doppler is 1.44 times the PRF, so the sampled
signal shows the *alias*. Keystoning the alias stretches the wrong frequency, and the result
(−32.9 dB) is worse than not compensating at all. Try F = −3 … 3 and keep the one that integrates
best. F = 1 wins by 33 dB.

## The datachip

```
   parent range-Doppler map                         DataChip
   +-----------------------------------+            +-----------+
   |                                   |            |  data     |  (n_pulses or n_doppler,
   |              detection            |   crop     |           |   n_range[, n_rx])
   |                  X  ------------------------>  +-----------+
   |                                   |            axis0_offset, range_offset
   +-----------------------------------+            parent bin-centre axes
                                                    f0, B, fs, PRF, FFT sizes, windows,
                                                    keystone / F / a applied
```

A chip is small enough that the expensive, optimal estimator (a local maximum-likelihood search,
the generalised Radon–Fourier transform) is cheap on it. Its results always go back out in the
parent's coordinates: metres, and m/s with the fold factor applied. Scenario 006 uses the same
chip, with a receive-channel axis on the end, for angle estimation.

## Finer than one bin, and the estimator to avoid

| Estimator | Bias, no noise | RMSE ÷ Cramér–Rao bound at 28.9 dB |
| :--- | ---: | ---: |
| Parabolic on \|X\| | **0.234 bin** | **12.2** |
| Quinn | 1.2·10⁻⁶ bin | 1.29 |
| Jacobsen | 1.2·10⁻⁶ bin | 1.55 |
| Candan | 3.9·10⁻⁷ bin | 1.55 |
| Aboutanios–Mulgrew | 5·10⁻¹³ bin | **1.01** |

Parabolic interpolation is what most code does, and its error here is almost all bias. The
default is Aboutanios–Mulgrew.

## One honest limit: jerk

The estimator fits jerk, but a 20 ms burst cannot measure it: the Cramér–Rao bound on jerk at
this SNR is about 2300 m/s³, while an abrupt thrust change (burnout or staging) is about 300 m/s³. The scenario reports jerk
with its bound and checks that it is consistent with zero. It tests jerk *recovery* separately, on
a signal long enough to show it.

## Where to look next

- The specification: [`spec/scenario-004-keystone.md`](../../spec/scenario-004-keystone.md)
- The pulsed chain it extends: [scenario 001](scenario-001-xband.md), variant S2
- Data layout names (`data_rfst`, `data_rv`): [`docs/conventions/style.md`](../conventions/style.md)

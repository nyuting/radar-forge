# 1. The radar and its link budget

Before a single sample is simulated, a radar is a short list of numbers: a carrier frequency, a
bandwidth, a power, two antenna gains, a chirp length, a repetition rate, a sample rate and a
noise figure. This page turns that list into the quantities that decide what the radar can see:
how much power comes back from a target, how much noise it competes with, how finely it
separates two targets in range, and how far and how fast a target can be before its reading
folds. It sits beside the whole [chain](README.md#the-chain), because every later stage reads
these numbers. It covers `constants`, `targets`, `radar` and `radar_equation` in
`radar_forge.core`.

## What this stage does

`radar.py` describes the radar. A `Transmitter` and a `Receiver` are frozen value objects:
they check their inputs and then only answer questions. `Radar` puts the pair at one site
(monostatic). `BistaticRadar` puts the transmitter and the receiver at two different sites.
Nothing here simulates anything; that is the job of `signal` and `dsp`. So you can build a
radar, print its limits and check them before you generate any data.

`radar_equation.py` is the link budget: how much of the transmitted power comes back from a
target. `targets.py` holds the target, a point with a fixed radar cross-section (RCS). The RCS is
how large the target looks to the radar, in square metres. `constants.py` holds the physical
constants; every other module imports them from there.

Two numbers matter most when a target shows up in the wrong place: the unambiguous range
$R_{ua}$ and the unambiguous velocity $v_{ua}$. A target beyond either one *folds*: it appears
at a wrong range or velocity, wrapped back into the window the radar can see. A radar cannot
make both large at once, and the three variants of scenario 001 each make a different choice.

```
 P_t, G_t ──► spreads over 4πR² ──► target, σ ──► spreads over 4πR² ──► antenna G_r λ²/4π ──► P_r
                                                                                             vs
                                                       receiver noise N_0 = k T_0 F, per hertz
```

## Quantities and equations

| Quantity | Symbol | Equation | In the code | Scenario 001 S1 |
| :-- | :-- | :-- | :-- | :-- |
| Wavelength | $\lambda$ | $\lambda = c / f_c$ | `Transmitter.wavelength_m` | 30.5911 mm |
| RCS from dBsm | $\sigma$ | $\sigma = 10^{\sigma_{\mathrm{dBsm}}/10}$ | `dbsm_to_m2`, `PointTarget.from_dbsm` | 10.0 dBsm = 10.0 m² |
| Received power, monostatic | $P_r$ | $P_r = \dfrac{P_t G_t G_r \lambda^2 \sigma}{(4\pi)^3 R^4 L}$ | `received_power_w` | 4.9590e-15 W at $R$ = 17 560.7 m |
| Received power, bistatic | $P_r$ | $P_r = \dfrac{P_t G_t G_r \lambda^2 \sigma_b}{(4\pi)^3 R_t^2 R_r^2 L}$ | `bistatic_received_power_w` | — |
| Noise density | $N_0$ | $N_0 = k T_0 F$ | `Receiver.noise_density_w_per_hz` | 7.9888e-21 W/Hz |
| Noise per IQ sample | — | $N_0 f_s$ | `Receiver.noise_power_per_sample_w` | 7.9888e-15 W |
| Noise of a matched receiver | — | $N_0 B = k T_0 B F$ | `Radar.noise_power_matched_w` | 1.5978e-14 W |
| Signal-to-noise ratio | SNR | $\mathrm{SNR} = P_r / (k T_0 B F)$ | `received_power_w / noise_power_matched_w` | −5.08 dB |
| Range resolution | $\Delta R$ | $\Delta R = c / 2B$ | `Radar.range_resolution_m` | 74.9481 m |
| Pulse repetition interval | $T_{\mathrm{PRI}}$ | $T_{\mathrm{PRI}} = 1 / \mathrm{PRF}$ | `Transmitter.pulse_repetition_interval_s` | 1.000 ms |
| Duty cycle | $d$ | $d = T_c \cdot \mathrm{PRF}$ | `Transmitter.duty_cycle_linear` | 1.00 |
| Sweep rate | $\alpha$ | $\alpha = B / T_c$ | `Transmitter.sweep_rate_hzps` | 2.0000e9 Hz/s |
| Unambiguous delay | $\tau_{ua}$ | $f_s / 2\alpha$ (FMCW), $1/\mathrm{PRF}$ (pulsed) | `Radar.unambiguous_delay_s` | 250 µs |
| Unambiguous range | $R_{ua}$ | $c f_s / 4\alpha$ (FMCW), $c / 2\,\mathrm{PRF}$ (pulsed) | `Radar.unambiguous_range_m` | 37.474 km |
| Unambiguous velocity | $v_{ua}$ | $v_{ua} = \lambda\,\mathrm{PRF} / 4$ | `Radar.unambiguous_velocity_mps` | ±7.6478 m/s |
| Baseline | $L_b$ | distance between the two sites | `BistaticRadar.baseline_m` | — |
| Bistatic angle | $\beta$ | $\cos\beta = (R_t^2 + R_r^2 - L_b^2) / (2 R_t R_r)$ | `BistaticRadar.bistatic_angle_rad` | — |
| Bistatic range resolution | $\Delta R_b$ | $\Delta R_b = c / (2B\cos(\beta/2))$ | `BistaticRadar.range_resolution_at_bistatic_angle_m` | — |
| Unambiguous range sum | $(R_t + R_r)_{ua}$ | $c\,\tau_{ua}$ | `BistaticRadar.unambiguous_range_m` | — |

The S1 received power and SNR are for the target's true range at frame 0, 17 560.7 m, from the
run's `truth.csv`. The bistatic rows have no S1 value because S1 is monostatic. Scenario 002
(B1) flies the S1 waveform over a 19.601 km baseline; its range-sum limit is 74.948 km, and its
range resolution grows from 74.95 m at $\beta = 0$ to 86.54 m at 60° and 105.99 m at 90°.

Symbols used only on this page:

- $L$: total loss factor, linear, $L \ge 1$ (`loss_linear`);
- $\sigma_b$: bistatic RCS, m² (`bistatic_rcs_m2`);
- $N_0$: noise power spectral density, W/Hz (`noise_density_w_per_hz`);
- $L_b$: baseline between the two sites of a bistatic pair, m (`baseline_m`);
- $\Delta R$, $\Delta R_b$: monostatic and bistatic range resolution, m;
- $T_{\mathrm{PRI}}$: pulse repetition interval, s; $d$: duty cycle, linear;
- $\tau_{ua}$, $R_{ua}$, $v_{ua}$: unambiguous delay, range and velocity.

The radar equation. Power falls as $1/R^4$, so doubling the range costs 12 dB:

$$P_r = \frac{P_t\, G_t\, G_r\, \lambda^2\, \sigma}{(4\pi)^3\, R^4\, L}$$

The noise. Every noise power is $N_0$ times a bandwidth; which bandwidth depends on the question.
A matched receiver admits the transmitted band $B$, and that is the radar-equation SNR. A complex
sample at rate $f_s$ holds the sampled band, $N_0 f_s$, and that is what `signal` adds to the cube:

$$N_0 = k\, T_0\, F, \qquad \mathrm{SNR} = \frac{P_r}{N_0 B}$$

The two unambiguous limits. Range folds when an echo arrives too late to be told apart; velocity
folds when the phase turns more than half a cycle between chirps:

$$R_{ua} = \frac{c}{2\,\mathrm{PRF}}\ \text{(pulsed)}, \qquad R_{ua} = \frac{c\, f_s}{4\alpha}\ \text{(FMCW)}, \qquad v_{ua} = \frac{\lambda\, \mathrm{PRF}}{4}$$

For a pulsed radar the product is fixed: $R_{ua}\, v_{ua} = c\lambda/8$, whatever the PRF. Raise
the PRF to see faster targets and you see less far. For FMCW the PRF does not appear in
$R_{ua}$, so the product becomes $(c\lambda/8) \cdot d \cdot f_s / 2B$. S1 has $d = 1$ and
$f_s / 2B = 0.25$, so it reaches only a quarter of $c\lambda/8$; S3 samples at $f_s = 2B$ and
reaches all of it. The trade has moved from the PRF to the sample rate.

## How it works, step by step

1. **Constants.** Import `SPEED_OF_LIGHT_MPS`, `BOLTZMANN_JPK` and
   `STANDARD_NOISE_TEMPERATURE_K` from `radar_forge.core.constants`. Never type 3e8.
2. **The target.** `PointTarget.from_dbsm(10.0)` gives a target of 10 m². `dbsm_to_m2` and
   `m2_to_dbsm` convert arrays either way; `m2_to_dbsm` refuses an area of zero or less.
3. **The transmitter.** `Transmitter(carrier_frequency_hz, bandwidth_hz, transmit_power_w,
   gain_tx_dbi, chirp_duration_s, prf_hz, waveform)` checks that every quantity is positive and
   that the duty cycle is at most 1. It derives `wavelength_m`, `pulse_repetition_interval_s`,
   `duty_cycle_linear` and `sweep_rate_hzps`.
4. **The receiver.** `Receiver(sample_rate_hz, gain_rx_dbi, noise_figure_db)` derives
   `noise_figure_linear`, `noise_density_w_per_hz` and `noise_power_per_sample_w`.
5. **The siting.** `Radar(transmitter, receiver, latitude_deg, longitude_deg, altitude_m)` or
   `BistaticRadar(transmitter, receiver, <transmitter site>, <receiver site>)`. Both reject a
   pulsed receiver that samples below the bandwidth. Both give `range_resolution_m`,
   `unambiguous_delay_s`, `unambiguous_velocity_mps`, `noise_power_matched_w`,
   `n_samples_per_chirp` and `n_samples_per_pri`. Each turns the delay into its own
   `unambiguous_range_m`: half of $c\,\tau_{ua}$ for one site, all of it (a range sum) for two.
   `n_samples_per_pri` is the row length of the `(n_pulses, n_samples)` IQ cube.
6. **The geometry, if bistatic.** `BistaticRadar.target_ranges_m(latitude_deg, longitude_deg,
   altitude_m)` gives $R_t$ and $R_r$, each of shape `(n_targets,)`. `bistatic_angle_rad`
   turns them into $\beta$, and `range_resolution_at_bistatic_angle_m` turns $\beta$ into
   $\Delta R_b$.
7. **The power.** `received_power_w` (or `bistatic_received_power_w`) gives $P_r$. Gains enter
   linear: `10.0 ** (gain_tx_dbi / 10.0)`. Every argument broadcasts, so a range array of shape
   `(n_targets,)` gives a power array of the same shape.

## Try it

This loads S1, computes its link budget at the target's frame-0 range and prints its limits:

```python
import numpy as np

from radar_forge.core.radar_equation import received_power_w
from radar_forge.pipelines.scenarios import load_scenario

scenario = load_scenario("scenarios/scenario_001_fmcw_low_prf.toml")
radar = scenario.bursts[0]  # a Radar: one Transmitter and one Receiver at one site
tx, rx = radar.transmitter, radar.receiver
power_w = received_power_w(
    transmit_power_w=tx.transmit_power_w,
    gain_tx_linear=10.0 ** (tx.gain_tx_dbi / 10.0),
    gain_rx_linear=10.0 ** (rx.gain_rx_dbi / 10.0),
    wavelength_m=radar.wavelength_m,
    rcs_m2=scenario.target.rcs_m2,
    range_m=17_560.7,  # the true range at frame 0, from truth.csv
)
print(f"wavelength      {radar.wavelength_m * 1e3:.4f} mm")
print(f"rcs             {scenario.target.rcs_dbsm:.1f} dBsm = {scenario.target.rcs_m2:.1f} m^2")
print(f"received power  {power_w:.4e} W")
print(f"N0              {rx.noise_density_w_per_hz:.4e} W/Hz")
print(f"noise / sample  {rx.noise_power_per_sample_w:.4e} W")
print(f"noise, matched  {radar.noise_power_matched_w:.4e} W")
print(f"SNR, matched    {10 * np.log10(power_w / radar.noise_power_matched_w):.2f} dB")
print(f"SNR / sample    {10 * np.log10(power_w / rx.noise_power_per_sample_w):.2f} dB")
print(f"range res.      {radar.range_resolution_m:.4f} m")
print(f"PRI, duty       {tx.pulse_repetition_interval_s * 1e3:.3f} ms, {tx.duty_cycle_linear:.2f}")
print(f"sweep rate      {tx.sweep_rate_hzps:.4e} Hz/s")
print(f"R_ua            {radar.unambiguous_range_m / 1e3:.3f} km")
print(f"v_ua            +-{radar.unambiguous_velocity_mps:.4f} m/s")
```

```text
wavelength      30.5911 mm
rcs             10.0 dBsm = 10.0 m^2
received power  4.9590e-15 W
N0              7.9888e-21 W/Hz
noise / sample  7.9888e-15 W
noise, matched  1.5978e-14 W
SNR, matched    -5.08 dB
SNR / sample    -2.07 dB
range res.      74.9481 m
PRI, duty       1.000 ms, 1.00
sweep rate      2.0000e+09 Hz/s
R_ua            37.474 km
v_ua            +-7.6478 m/s
```

A matched SNR of −5 dB looks hopeless, but it is the SNR before any integration. One chirp's
range FFT gains $B T_c = 2000$ (33.01 dB), so one chirp holds $E/N_0 = P_r T_c / N_0$ of about
27.9 dB, and the Doppler FFT over 256 chirps adds 24.1 dB more. The per-sample SNR is 3.01 dB
higher than the matched one because S1 samples at $f_s = B/2$: each sample holds half the noise.

Now the trade. This prints $R_{ua}$, $v_{ua}$ and their product, as a fraction of $c\lambda/8$,
for each burst of the three scenario-001 variants:

```python
from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.pipelines.scenarios import load_scenario

for name in ("fmcw_low_prf", "pulsed_medium_prf", "fmcw_dual_prf"):
    for radar in load_scenario(f"scenarios/scenario_001_{name}.toml").bursts:
        range_m = radar.unambiguous_range_m
        velocity_mps = radar.unambiguous_velocity_mps
        limit = SPEED_OF_LIGHT_MPS * radar.wavelength_m / 8.0  # c * lambda / 8
        print(
            f"{name:18s} {radar.transmitter.prf_hz:7.0f} Hz  R_ua {range_m / 1e3:7.3f} km  "
            f"v_ua {velocity_mps:8.4f} m/s  R*v / (c lambda/8) {range_m * velocity_mps / limit:.3f}"
        )
```

```text
fmcw_low_prf          1000 Hz  R_ua  37.474 km  v_ua   7.6478 m/s  R*v / (c lambda/8) 0.250
pulsed_medium_prf    25000 Hz  R_ua   5.996 km  v_ua 191.1942 m/s  R*v / (c lambda/8) 1.000
fmcw_dual_prf         5000 Hz  R_ua  29.979 km  v_ua  38.2388 m/s  R*v / (c lambda/8) 1.000
fmcw_dual_prf         6000 Hz  R_ua  24.983 km  v_ua  45.8866 m/s  R*v / (c lambda/8) 1.000
```

S1 sees far and folds in velocity; S2 sees fast and folds in range. S3's two bursts each sit on
the $c\lambda/8$ limit, and page 4 shows how the pair escapes it.

## Try it in a scenario

The companion is [scenario 001](../scenarios/scenario-001-xband.md). Run S1:

```bash
uv run python scripts/run_scenario.py scenarios/scenario_001_fmcw_low_prf.toml --out out/s1 --frames 10
```

Now copy the TOML, set `prf_hz = 800.0` (from 1.0e3) and run the copy. The chirp stays 1 ms
long, so the duty cycle falls to 0.8 and the sweep rate does not change. In `metadata.json`,
`unambiguous_velocity_mps` falls from 7.6478 to 6.1182 m/s, while `unambiguous_range_m` stays at
37 474 m: for FMCW, lowering the PRF costs velocity and buys no range. In `rd_00000.png` the
red truth cross, the true −28.08 m/s folded onto the map, moves from +2.51 m/s to −3.61 m/s, and
the velocity axis shrinks to ±6.12 m/s. `truth.csv` is identical in both runs; only where the
radar puts the target changes.

To go further, see [Task 1: Where does the tracker break?](../../interns/tracker-breakpoint.md),
which starts from this page's link budget and lowers `transmit_power_w` until the radar misses.

## Things that look like bugs and aren't

**S1 samples at 1 MHz, half its 2 MHz bandwidth.** An FMCW receiver deramps first, so it samples
the beat tone, not the sweep. The limit is on the sweep rate, not on $B$. A pulsed receiver
samples the pulse itself, and `Radar` raises a `ValueError` if its $f_s < B$.

**The noise in the IQ cube is 3.01 dB below $k T_0 B F$.** The cube holds $N_0 f_s$ per sample,
and S1 has $f_s = B/2$. `noise_power_matched_w` is the radar-equation noise, not the cube's. An
earlier version added $k T_0 B F$ to every sample; commit 4508b85 fixed it.

**The range-Doppler map runs to 74.9 km, but `unambiguous_range_m` says 37.474 km.** The
property uses the real-sampling limit $f_s/2$. Complex samples tell apart all of $[0, f_s)$, so
the map spans twice that. A real sawtooth also needs the delay shorter than the chirp, which caps
range at $c T_c / 2$ and is not modelled. `pipelines.scenarios.range_axis_m`'s Notes and
`spec/scenario-003-tracking.md` §14.11 explain it.

**A bistatic pair's unambiguous range is twice the monostatic one.** It bounds the range *sum*
$R_t + R_r$, which is itself about twice a one-way range. It is not a bistatic bonus.

## Limits

- Only the non-fluctuating Swerling 0 target is implemented. Swerling 1–4 and extended,
  multi-scatterer targets are not.
- RCS does not change with aspect angle. A real aircraft's changes by tens of decibels as it
  turns.
- The scenario pipeline calls the radar equation without a loss, so $L = 1$.
- Noise is referred to $T_0$ = 290 K through the noise figure; antenna and sky temperature are
  not modelled.
- The radar, and both bistatic sites, are stationary.
- The bistatic RCS is one number per target, the same at every angle, and forward scatter near
  $\beta = \pi$ is not modelled.
- A bistatic pair is perfectly synchronised in time and phase, and the direct signal from
  transmitter to receiver is not modelled.
- The FMCW $R_{ua}$ uses $f_s/2$ and ignores the $c T_c/2$ cap of a real sawtooth (see above).

## References

- M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014,
  §2.2.1 (the point-target radar range equation), §2.2.3 (radar cross section), §2.2.7
  (Swerling models), §3.1.3 (Doppler and range ambiguities), §5.3 (pulse Doppler processing).
- M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill, 2001, §2.3 (receiver
  noise and the signal-to-noise ratio), §2.7 (radar cross section of targets), §2.8 (radar
  cross-section fluctuations), §8.4 (atmospheric refraction).
- A. Harrison, *Introduction to Radar Using Python and MATLAB*, Artech House, 2020, ch. 4 (the
  radar range equation).
- G. L. Charvat, *Small and Short-Range Radar Systems*, CRC Press, 2014, ch. 3, "Mathematics of
  FMCW Radar".
- N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005, §1.2 (the bistatic angle),
  ch. 3 (geometry), §4.1 (the bistatic range equation), §4.3 (ovals of Cassini), §7.1 (range
  resolution).
- CODATA 2018 recommended values, https://physics.nist.gov/cuu/Constants/archive2018.html.

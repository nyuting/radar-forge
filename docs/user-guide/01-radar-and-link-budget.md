# 1. The radar and its link budget

Before a single sample is simulated, a radar is a short list of numbers: a carrier frequency, a
bandwidth, a power, two antenna gains, a chirp length, a repetition rate, a sample rate, a noise
figure and where its antennas stand. This page turns that list into the quantities that decide
what the radar can see: how much power comes back from a target, how much noise it competes
with, how finely it separates two targets in range, and how far and how fast a target can be
before its reading folds. It sits beside the whole [chain](README.md#the-chain), because every
later stage reads these numbers. It covers `radar_forge.constants`, `radar_forge.radar`, and the
radar equation and targets in `radar_forge.propagation`.

## What this stage does

`radar.py` describes the radar. A `Transmitter` and a `Receiver` are frozen value objects: they
check their inputs and then only answer questions. A `Site` is where an antenna stands.
`Radar(transmitter, receiver, tx_site, rx_site=None)` puts them together. With no `rx_site` the
receiver stands at the transmitter (monostatic); with one, the pair is bistatic. One class covers
both, because a monostatic radar is a bistatic one with a zero baseline. Nothing here simulates
anything; that is the job of `signal` and `dsp`. So you can build a radar, print its limits and
check them before you generate any data.

`received_power_w` in `propagation.py` is the link budget: how much of the transmitted power
comes back from a target. A target is a point with a radar cross-section (RCS), how large it
looks to the radar, in square metres. `constants.py` holds the physical constants; every other
module imports them from there.

Every range and rate the library carries is **bistatic**: the length of the whole route,
$R_b = R_T + R_R$ from transmitter to target to receiver, and its time derivative $\dot R_b$,
positive receding. For a monostatic radar that is twice the slant range and twice the radial
velocity. Only a display halves them ([page 2](02-geometry-and-propagation.md)).

Two numbers matter most when a target shows up in the wrong place: the unambiguous bistatic
range $R_{b,ua}$ and the span of rates $\dot R_{b,ua}$ the radar can tell apart. A target beyond
either one *folds*: it appears at a wrong range or rate, wrapped back into the window the radar
can see. A radar cannot make both large at once, and the three variants of scenario 001 each
make a different choice.

```
 P_t, G_t ──► spreads over 4πR_T² ──► target, σ ──► spreads over 4πR_R² ──► antenna G_r λ²/4π ──► P_r
                                                                                                 vs
                                                           receiver noise N_0 = k T_0 F, per hertz
```

## Quantities and equations

| Quantity | Symbol | Equation | In the code | Scenario 001 S1 |
| :-- | :-- | :-- | :-- | :-- |
| Wavelength | $\lambda$ | $\lambda = c / f_c$ | `Transmitter.wavelength_m` | 30.5911 mm |
| RCS from dBsm | $\sigma$ | $\sigma = 10^{\sigma_{\mathrm{dBsm}}/10}$ | `dbsm_to_m2`, `PointTarget.from_dbsm` | 10.0 dBsm = 10.0 m² |
| Received power | $P_r$ | $P_r = \dfrac{P_t G_t G_r \lambda^2 \sigma}{(4\pi)^3 R_T^2 R_R^2 L}$ | `received_power_w` | 4.9590e-15 W at $R_T = R_R$ = 17 560.7 m |
| Noise density | $N_0$ | $N_0 = k T_0 F$ | `Receiver.noise_density_w_per_hz` | 7.9888e-21 W/Hz |
| Noise per IQ sample | — | $N_0 f_s$ | `Receiver.noise_power_per_sample_w` | 7.9888e-15 W |
| Noise of a matched receiver | — | $N_0 B = k T_0 B F$ | `Radar.noise_power_matched_w` | 1.5978e-14 W |
| Signal-to-noise ratio | SNR | $\mathrm{SNR} = P_r / (k T_0 B F)$ | `received_power_w / noise_power_matched_w` | −5.08 dB |
| Range resolution | $\Delta R$ | $\Delta R = c / 2B$ | `Radar.range_resolution_m` | 74.9481 m |
| Bistatic range resolution | $\Delta R(\beta)$ | $\Delta R(\beta) = c / (2B\cos(\beta/2))$ | `range_resolution_m / np.cos(beta / 2)` | 74.9481 m ($\beta$ = 0) |
| Pulse repetition interval | $T_{\mathrm{PRI}}$ | $T_{\mathrm{PRI}} = 1 / \mathrm{PRF}$ | `Transmitter.pulse_repetition_interval_s` | 1.000 ms |
| Duty cycle | $d$ | $d = T_c \cdot \mathrm{PRF}$ | `Transmitter.duty_cycle_linear` | 1.00 |
| Sweep rate | $\alpha$ | $\alpha = B / T_c$ | `Transmitter.sweep_rate_hzps` | 2.0000e9 Hz/s |
| Unambiguous delay | $\tau_{ua}$ | $f_s / \alpha$ (FMCW), $1/\mathrm{PRF}$ (pulsed) | `Radar.unambiguous_delay_s` | 500 µs |
| Unambiguous bistatic range | $R_{b,ua}$ | $c\,\tau_{ua}$: $c f_s / \alpha$ (FMCW), $c / \mathrm{PRF}$ (pulsed) | `Radar.unambiguous_bistatic_range_m` | 149.896 km (74.948 km slant) |
| Unambiguous rate span | $\dot R_{b,ua}$ | $\dot R_{b,ua} = \lambda\,\mathrm{PRF}$, shown as $\pm\dot R_{b,ua}/2$ | `Radar.unambiguous_bistatic_range_rate_mps` | 30.5911 m/s: ±15.2955 m/s (±7.6478 m/s radial) |
| Baseline | $L_b$ | distance between the two sites | `Radar.baseline_m` | 0.0 m |
| Bistatic angle | $\beta$ | $\cos\beta = (R_T^2 + R_R^2 - L_b^2) / (2 R_T R_R)$ | `TargetGeometry.bistatic_angle_rad` | 0 |

The S1 received power and SNR are for the target's true range at frame 0, 17 560.7 m, from
`truth_geometry`. S1 is monostatic, so $R_T = R_R$, $L_b = 0$ and $\beta = 0$. Scenario 002
(B1) flies the S1 waveform over a 19 601.4 m baseline. Its $R_{b,ua}$ is the same 149.896 km,
because $c\,\tau_{ua}$ does not depend on where the antennas stand. Its range resolution grows
from 74.95 m at $\beta = 0$ to 86.54 m at 60° and 105.99 m at 90°: measured in $R_b$ itself a
resolution cell is always $c/B$ = 149.9 m, and $\Delta R(\beta)$ is that cell on the ground,
along the bisector of $\beta$.

Symbols used only on this page:

- $R_T$, $R_R$: transmitter-to-target and target-to-receiver range, m (`range_tx_m`,
  `range_rx_m`); for one site both are the slant range $R$;
- $R_b = R_T + R_R$: bistatic range, m (`bistatic_range_m`); $\dot R_b$: its rate, m/s,
  positive receding (`bistatic_range_rate_mps`);
- $L$: total loss factor, linear, $L \ge 1$ (`loss_linear`);
- $\sigma$: with two sites, the bistatic RCS, taken as given (`rcs_m2`);
- $N_0$: noise power spectral density, W/Hz (`noise_density_w_per_hz`);
- $L_b$: baseline between the two sites, m (`baseline_m`);
- $\Delta R$, $\Delta R(\beta)$: monostatic and bistatic range resolution, m;
- $T_{\mathrm{PRI}}$: pulse repetition interval, s; $d$: duty cycle, linear;
- $\tau_{ua}$, $R_{b,ua}$, $\dot R_{b,ua}$: unambiguous delay, bistatic range and rate span.

The radar equation, in Willis's bistatic form. One site has $R_T = R_R = R$, so power falls as
$1/R^4$ and doubling the range costs 12 dB:

$$P_r = \frac{P_t\, G_t\, G_r\, \lambda^2\, \sigma}{(4\pi)^3\, R_T^2\, R_R^2\, L}$$

The noise. Every noise power is $N_0$ times a bandwidth; which bandwidth depends on the question.
A matched receiver admits the transmitted band $B$, and that is the radar-equation SNR. A complex
sample at rate $f_s$ holds the sampled band, $N_0 f_s$, and that is what `signal` adds to the cube:

$$N_0 = k\, T_0\, F, \qquad \mathrm{SNR} = \frac{P_r}{N_0 B}$$

The two unambiguous limits. Range folds when an echo arrives too late to be told apart; rate
folds when the carrier phase, $-2\pi R_b/\lambda$, turns more than half a cycle between chirps:

$$R_{b,ua} = \frac{c}{\mathrm{PRF}}\ \text{(pulsed)}, \qquad R_{b,ua} = \frac{c\, f_s}{\alpha}\ \text{(FMCW)}, \qquad |\dot R_b| < \frac{\dot R_{b,ua}}{2} = \frac{\lambda\, \mathrm{PRF}}{2}$$

For a pulsed radar the product is fixed: $R_{b,ua}\, \dot R_{b,ua} = c\lambda$, whatever the
PRF. Halve both for one site and it is the textbook $R_{ua}\, v_{ua} = c\lambda/8$, with
$R_{ua} = c/2\,\mathrm{PRF}$ and $v_{ua} = \lambda\,\mathrm{PRF}/4$. Raise the PRF to see faster
targets and you see less far. For FMCW the PRF does not appear in $R_{b,ua}$, so the product
becomes $c\lambda \cdot d \cdot f_s / B$. S1 has $d = 1$ and $f_s / B = 0.5$, so it reaches
half of $c\lambda$; S3 samples at $f_s = 2B$ and reaches twice it. The trade has moved from the
PRF to the sample rate.

## How it works, step by step

1. **Constants.** Import `SPEED_OF_LIGHT_MPS`, `BOLTZMANN_JPK` and
   `STANDARD_NOISE_TEMPERATURE_K` from `radar_forge.constants`. Never type 3e8.
2. **The target.** `PointTarget.from_dbsm(10.0)` gives a target of 10 m². `dbsm_to_m2` and
   `m2_to_dbsm` convert arrays either way; `m2_to_dbsm` refuses an area of zero or less. A
   scenario's `[[target]]` gives `rcs_dbsm`, and `Target.rcs_m2` converts it.
   `swerling_rcs_m2(mean_rcs_m2, case, shape, rng=rng)` draws fluctuating cross-sections about a
   mean, Swerling cases 1 to 4.
3. **The transmitter.** `Transmitter(carrier_frequency_hz, bandwidth_hz, transmit_power_w,
   gain_tx_dbi, chirp_duration_s, prf_hz, waveform)` checks that every quantity is positive and
   that the duty cycle is at most 1. It derives `wavelength_m`, `pulse_repetition_interval_s`,
   `duty_cycle_linear` and `sweep_rate_hzps`.
4. **The receiver.** `Receiver(sample_rate_hz, gain_rx_dbi, noise_figure_db)` derives
   `noise_figure_linear`, `noise_density_w_per_hz` and `noise_power_per_sample_w`.
5. **The sites.** `Site(latitude_deg, longitude_deg, altitude_m)`, with the altitude above the
   WGS-84 ellipsoid. `Radar(transmitter, receiver, tx_site, rx_site=None)` rejects a pulsed
   receiver that samples below the bandwidth. It gives `is_monostatic`, `baseline_m`,
   `range_resolution_m`, `unambiguous_delay_s`, `unambiguous_bistatic_range_m`,
   `unambiguous_bistatic_range_rate_mps`, `noise_power_matched_w`, `n_samples_per_chirp` and
   `n_samples_per_pri`, the same properties for either siting. `n_samples_per_pri` is the row
   length of the `(n_pulses, n_samples)` IQ cube.
6. **The geometry.** `Radar.geometry(position_enu_m, velocity_enu_mps)` gives a
   `TargetGeometry` whose `range_tx_m`, `range_rx_m`, `bistatic_range_m`,
   `bistatic_range_rate_mps` and `bistatic_angle_rad` are each of shape `(n_targets,)`.
   [Page 2](02-geometry-and-propagation.md) covers it. The bistatic range resolution is
   `radar.range_resolution_m / np.cos(geometry.bistatic_angle_rad / 2)`.
7. **The power.** `received_power_w` takes keywords only, among them `range_tx_m` and
   `range_rx_m`; for one site, pass the slant range to both. Gains enter linear:
   `10.0 ** (gain_tx_dbi / 10.0)`. Every argument broadcasts, so range arrays of shape
   `(n_targets,)` give a power array of the same shape.

## Try it

This loads S1, computes its link budget at the target's frame-0 range and prints its limits:

```python
import numpy as np

from radar_forge.propagation import received_power_w
from radar_forge.scenario import load_scenario, truth_geometry

scenario = load_scenario("config/scenario_001_fmcw_low_prf.toml")
radar = scenario.radar  # one Transmitter and one Receiver; no [site.rx], so monostatic
tx, rx = radar.transmitter, radar.receiver
target = scenario.targets[0]
truth = truth_geometry(scenario)[0]  # frame 0
power_w = received_power_w(
    transmit_power_w=tx.transmit_power_w,
    gain_tx_linear=10.0 ** (tx.gain_tx_dbi / 10.0),
    gain_rx_linear=10.0 ** (rx.gain_rx_dbi / 10.0),
    wavelength_m=radar.wavelength_m,
    rcs_m2=target.rcs_m2,
    range_tx_m=truth.range_tx_m[0],
    range_rx_m=truth.range_rx_m[0],
)
rate_span_mps = radar.unambiguous_bistatic_range_rate_mps
print(f"monostatic      {radar.is_monostatic}, baseline {radar.baseline_m:.1f} m")
print(f"R_T, R_R        {truth.range_tx_m[0]:.1f} m, {truth.range_rx_m[0]:.1f} m")
print(f"wavelength      {radar.wavelength_m * 1e3:.4f} mm")
print(f"rcs             {target.rcs_dbsm:.1f} dBsm = {target.rcs_m2:.1f} m^2")
print(f"received power  {power_w:.4e} W")
print(f"N0              {rx.noise_density_w_per_hz:.4e} W/Hz")
print(f"noise / sample  {rx.noise_power_per_sample_w:.4e} W")
print(f"noise, matched  {radar.noise_power_matched_w:.4e} W")
print(f"SNR, matched    {10 * np.log10(power_w / radar.noise_power_matched_w):.2f} dB")
print(f"SNR / sample    {10 * np.log10(power_w / rx.noise_power_per_sample_w):.2f} dB")
print(f"range res.      {radar.range_resolution_m:.4f} m")
print(f"PRI, duty       {tx.pulse_repetition_interval_s * 1e3:.3f} ms, {tx.duty_cycle_linear:.2f}")
print(f"sweep rate      {tx.sweep_rate_hzps:.4e} Hz/s")
print(f"R_b,ua          {radar.unambiguous_bistatic_range_m / 1e3:.3f} km")
print(f"rate span       {rate_span_mps:.4f} m/s, +-{rate_span_mps / 4:.4f} m/s radial")
```

```text
monostatic      True, baseline 0.0 m
R_T, R_R        17560.7 m, 17560.7 m
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
R_b,ua          149.896 km
rate span       30.5911 m/s, +-7.6478 m/s radial
```

A matched SNR of −5 dB looks hopeless, but it is the SNR before any integration. One chirp's
range FFT gains $B T_c = 2000$ (33.01 dB), so one chirp holds $E/N_0 = P_r T_c / N_0$ of about
27.9 dB, and the Doppler FFT over 256 chirps adds 24.1 dB more: 52.01 dB on the map. The
per-sample SNR is 3.01 dB higher than the matched one because S1 samples at $f_s = B/2$: each
sample holds half the noise.

Now the trade. This prints $R_{b,ua}$, $\dot R_{b,ua}$ and their product, as a fraction of
$c\lambda$, for each burst of the three scenario-001 variants:

```python
from radar_forge.constants import SPEED_OF_LIGHT_MPS
from radar_forge.scenario import load_scenario

for name in ("fmcw_low_prf", "pulsed_medium_prf", "fmcw_dual_prf"):
    for radar in load_scenario(f"config/scenario_001_{name}.toml").radars:
        range_m = radar.unambiguous_bistatic_range_m
        rate_mps = radar.unambiguous_bistatic_range_rate_mps
        limit = SPEED_OF_LIGHT_MPS * radar.wavelength_m  # c * lambda
        print(
            f"{name:18s} {radar.transmitter.prf_hz:6.0f} Hz  R_b,ua {range_m / 1e3:6.3f} km  "
            f"span {rate_mps:7.3f} m/s  product / (c lambda) {range_m * rate_mps / limit:.3f}"
        )
```

```text
fmcw_low_prf         1000 Hz  R_b,ua 149.896 km  span  30.591 m/s  product / (c lambda) 0.500
pulsed_medium_prf   25000 Hz  R_b,ua 11.992 km  span 764.777 m/s  product / (c lambda) 1.000
fmcw_dual_prf        5000 Hz  R_b,ua 119.917 km  span 152.955 m/s  product / (c lambda) 2.000
fmcw_dual_prf        6000 Hz  R_b,ua 99.931 km  span 183.546 m/s  product / (c lambda) 2.000
```

S1 sees far and folds in rate; S2 sees fast and folds in range. S3's two bursts each reach
$2c\lambda$, and page 4 shows how the pair unfolds a rate beyond either burst's span.

## Try it in a scenario

The scenario file is [scenario 001](../../config/scenario_001_fmcw_low_prf.toml). Run three
frames of S1; the run goes to `runs/scenario_001_fmcw_low_prf/`:

```bash
uv run python scripts/run_scenario.py config/scenario_001_fmcw_low_prf.toml --frames 3
```

Each stage prints what it measured, checks it against the TOML's `[verify]` table, and gives its
wall time. The `geometry` stage is this page's link budget:

```text
[geometry]
  received_power_dbw = -143.046
  noise_power_per_sample_dbw = -140.975
  predicted_snr_db = 52.0115
  baseline_m = 0
  bistatic_angle_min_deg = 0
  bistatic_angle_max_deg = 0
  unambiguous_bistatic_range_m = 149896
  unambiguous_bistatic_range_rate_mps = 30.5911
  truth_in_map_fraction = 0
  check truth_in_map_fraction = 0 in [0.0, 0.25]: pass
  check predicted_snr_db = 52.0115 in [52.0, 52.019999999999996]: pass
```

Now copy the TOML to `config/` under a new name, so that its `trajectory_path` still resolves,
set `prf_hz = 800.0` (from 1.0e3) and run the copy. The chirp stays 1 ms long, so the duty cycle
falls to 0.8 and the sweep rate does not change. `unambiguous_bistatic_range_rate_mps` falls from
30.5911 to 24.4729 m/s (±7.6478 to ±6.1182 m/s radial), while `unambiguous_bistatic_range_m`
stays at 149896 m: for FMCW, lowering the PRF costs rate and buys no range. In
`rd_00000.png` the red truth cross, receding at 28.08 m/s, is off the map in both runs. The
dashed crosshair, where the folded truth must appear, moves from −2.51 m/s to +3.61 m/s, and the
velocity axis shrinks from ±7.65 to ±6.12 m/s. `truth.csv` is identical in both runs; only where the radar puts the
target changes. Every check still passes.

To go further, see [Task 1: Where does the tracker break?](../../interns/tracker-breakpoint.md),
which starts from this page's link budget and lowers `transmit_power_w` until the radar misses.

## Things that look like bugs and aren't

**S1 samples at 1 MHz, half its 2 MHz bandwidth.** An FMCW receiver deramps first, so it samples
the beat tone, not the sweep. The limit is on the sweep rate, not on $B$. A pulsed receiver
samples the pulse itself, and `Radar` raises a `ValueError` if its $f_s < B$.

**The noise in the IQ cube is 3.01 dB below $k T_0 B F$.** The cube holds $N_0 f_s$ per sample,
and S1 has $f_s = B/2$. `noise_power_matched_w` is the radar-equation noise, not the cube's. An
earlier version added $k T_0 B F$ to every sample; commit 4508b85 fixed it.

**`unambiguous_bistatic_range_rate_mps` is 30.59 m/s, but the map's rate axis runs only from
−15.30 to +15.30 m/s.** The property is the span of rates the map holds, $\lambda\,\mathrm{PRF}$,
and the map centres it on zero.

**S1's limits, 149.9 km and 30.59 m/s, are twice what a monostatic textbook gives.** They
bound $R_T + R_R$ and its rate, which for one site are $2R$ and twice the radial velocity. B1,
with the same waveform on two sites, has the same two limits, because they bound the same
route length. A monostatic figure or CSV halves them (`plots.display_axes`).

## Limits

- Every scenario uses the non-fluctuating Swerling 0 target. `swerling_rcs_m2` draws Swerling
  1–4 cross-sections, but nothing in the scenario chain calls it. Extended, multi-scatterer
  targets are not modelled.
- RCS does not change with aspect angle. A real aircraft's changes by tens of decibels as it
  turns.
- `line_of_sight_paths` calls the radar equation without a loss, so $L = 1$.
- Noise is referred to $T_0$ = 290 K through the noise figure; antenna and sky temperature are
  not modelled.
- Both sites are stationary.
- The bistatic RCS is one number per target, the same at every angle, and forward scatter near
  $\beta = \pi$ is not modelled.
- A bistatic pair is perfectly synchronised in time and phase, and the direct signal from
  transmitter to receiver is not modelled.
- The FMCW $R_{b,ua}$ ignores the $c T_c$ cap of a real sawtooth, whose delay must be shorter
  than the chirp.

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

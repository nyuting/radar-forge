# 2. Geometry and propagation

This stage turns *where things are* into *what the receiver hears*. It takes a target's latitude,
longitude and altitude, works out its route from the transmitter to the receiver and how fast
that route grows, and turns those into a delay, a Doppler shift and an amplitude for each echo.
It is the second and third boxes of [the chain](README.md#the-chain). It covers the geodesy
functions, `Radar.geometry` and `TargetGeometry` in `radar_forge.radar`, and
`radar_forge.propagation`.

## What this stage does

A scenario says where the sites and the target are in the coordinates the world uses: WGS-84
latitude, longitude and height above the ellipsoid (the **geodetic** frame). Radar geometry
wants a flat Cartesian frame with a site at the origin. The geodesy functions are the bridge.
They go through Earth-centred, Earth-fixed (**ECEF**) metres to a local **east-north-up (ENU)**
frame on the tangent plane at the site. They are the only code in the library that knows the
Earth is round.

From ENU it is a short step to range, azimuth and elevation. Azimuth is a compass bearing: zero
at true north, increasing clockwise towards east. `Radar.geometry` then measures the target's
**route**: $R_T$ from the transmitter, $R_R$ to the receiver, the **bistatic range**
$R_b = R_T + R_R$, its rate $\dot R_b$, positive receding, and the bistatic angle $\beta$. It
returns them as a `TargetGeometry`. A monostatic radar is the case where the two sites coincide:
$R_T = R_R = R$, so $R_b = 2R$, $\dot R_b$ is twice the radial velocity, and $\beta = 0$.

`propagation` then turns that geometry into **paths**. A path is one way energy gets from the
transmitter to the receiver, described by four things: a delay, a Doppler shift, a complex
amplitude and a pair of angles. Today there is one path per target, the direct line of sight in
free space. The next module, `signal` (page 3), turns paths into samples.

```
 latitude, longitude, height      geodetic_to_ecef_m      ecef_to_enu_m
 (WGS-84, per site and target) ------------------> ECEF ----------------> ENU position, velocity
                                                                            |
                                                             Radar.geometry |
                                                                            v
 PropagationPaths <------ line_of_sight_paths ------ TargetGeometry (R_T, R_R, R_b, dR_b/dt, beta)
 (delay, Doppler, amplitude, angles)  ---> signal (page 3)
```

That split between paths and samples is decision **D1** in `spec/data-001-formats.md` §8: a
propagation model (this module now, a ray tracer later) computes paths, and only `signal`
computes samples. A new propagation model therefore needs no new signal code. Scenario 002 puts
this to the test by moving the transmitter 19.6 km away from the receiver. The same
`TargetGeometry`, the same `line_of_sight_paths` and the same signal code serve it unchanged
(decision D6).

## Quantities and equations

The monostatic rows are for scenario 001 S1 at frame 0 (t = 0 s, the target's first fix). The
bistatic rows, marked B1, are for scenario 002 B1 at its frame 0 (t = 13329 s). Every value comes
from the two examples under [Try it](#try-it).

| Quantity | Symbol | Equation | In the code | Scenario 001 S1, frame 0 |
| :-- | :-- | :-- | :-- | :-- |
| Target's geodetic position | $\varphi, \ell, h$ | from the track CSV; $h$ is the target's fixed altitude | `Target.trajectory_path`, `Target.altitude_m` | 35.90608°, −79.0945°, 1500 m |
| Prime-vertical radius | $N$ | $N = a / \sqrt{1 - e^2 \sin^2\varphi}$ | inside `geodetic_to_ecef_m` | — |
| ECEF position | $(x, y, z)$ | $x = (N + h)\cos\varphi\cos\ell$, $y = (N + h)\cos\varphi\sin\ell$, $z = (N(1 - e^2) + h)\sin\varphi$ | `geodetic_to_ecef_m` | (978741.2, −5079904.8, 3720635.4) m |
| ENU position | $(e, n, u)$ | translate to the site, then rotate by the site's $\varphi_0, \ell_0$ | `ecef_to_enu_m`, `geodetic_to_enu_m` | (−13859.8, −10690.2, 1416.0) m |
| Slant range | $R$ | $R = \sqrt{e^2 + n^2 + u^2}$ | `enu_to_range_azimuth_elevation_rad` | 17560.7 m |
| Azimuth | $\mathrm{az}$ | $\mathrm{az} = \operatorname{atan2}(e, n)$, wrapped to $[0, 2\pi)$ | `enu_to_range_azimuth_elevation_rad` | 232.36° |
| Elevation | $\mathrm{el}$ | $\mathrm{el} = \arcsin(u / R)$ | `enu_to_range_azimuth_elevation_rad` | 4.625° |
| Bistatic range | $R_b$ | $R_b = R_T + R_R$; one site, $2R$ | `TargetGeometry.bistatic_range_m` | 35121.4 m |
| Bistatic range rate | $\dot R_b$ | $\dot R_b = \vec v \cdot \hat u_T + \vec v \cdot \hat u_R$, positive receding | `TargetGeometry.bistatic_range_rate_mps` | +56.163 m/s (receding at 28.081 m/s) |
| Wavelength | $\lambda$ | $\lambda = c / f_c$ | `Radar.wavelength_m` | 30.591 mm |
| Delay | $\tau$ | $\tau = R_b / c$ | `propagation_delay_s`, `PropagationPaths.delay_s` | 117.1523 µs |
| Doppler shift | $f_D$ | $f_D = -\dot R_b / \lambda$ | `doppler_shift_hz`, `PropagationPaths.doppler_hz` | −1835.9 Hz |
| Received power | $P_r$ | $P_r = \lvert a \rvert^2 = P_t G_t G_r \lambda^2 \sigma / ((4\pi)^3 R_T^2 R_R^2)$ | `abs(PropagationPaths.amplitude_linear) ** 2` | 4.959 × 10⁻¹⁵ W |
| Carrier phase | $\arg a$ | $-2\pi R_b / \lambda$ | `propagation_phase_rad` | — |
| Transmit and receive range (B1) | $R_T$, $R_R$ | ENU distance from each site | `TargetGeometry.range_tx_m`, `.range_rx_m` | 14800.6 m, 7523.6 m |
| Baseline (B1) | $L$ | distance between the two sites | `Radar.baseline_m` | 19601.4 m |
| Bistatic angle (B1) | $\beta$ | $\cos\beta = (R_T^2 + R_R^2 - L^2) / (2 R_T R_R)$ | `TargetGeometry.bistatic_angle_rad` | 119.17° |
| Delay (B1) | $\tau$ | $\tau = (R_T + R_R) / c$ | `PropagationPaths.delay_s` | 74.4653 µs |
| Bistatic range (B1) | $R_b$ | $R_b = R_T + R_R = c\tau$ | `PropagationPaths.bistatic_range_m` | 22324.1 m |
| Bistatic range rate (B1) | $\dot R_b$ | as above | `TargetGeometry.bistatic_range_rate_mps` | −19.228 m/s (closing) |
| Doppler shift (B1) | $f_D$ | $f_D = -\dot R_b / \lambda$ | `PropagationPaths.doppler_hz` | +628.6 Hz |

Symbols used only on this page:

- $\varphi$, $\ell$, $h$: geodetic latitude, longitude and height above the WGS-84 ellipsoid (the
  geodesy docstrings write longitude as $\lambda$; here $\lambda$ is kept for the wavelength).
  $\varphi_0$, $\ell_0$ are the site's own.
- $a$, $e^2$: the ellipsoid's semi-major axis (6378137 m) and first eccentricity squared,
  $e^2 = f(2 - f)$ from the flattening $f$ (`WGS84_SEMI_MAJOR_AXIS_M`, `WGS84_FLATTENING`).
- $N$: radius of curvature in the prime vertical, m.
- $(x, y, z)$: ECEF coordinates, m. $(e, n, u)$: east, north, up, m.
- $\mathrm{az}$, $\mathrm{el}$: azimuth and elevation, rad (`aoa_rad` and `aod_rad` hold them
  as pairs).
- $a$ (in $\lvert a \rvert$, $\arg a$): a path's complex amplitude (`amplitude_linear`).
- $R_T$, $R_R$: transmitter-to-target and target-to-receiver range, m (`range_tx_m`,
  `range_rx_m`).
- $R_b$, $\dot R_b$: bistatic range $R_T + R_R$, m (`bistatic_range_m`), and its time
  derivative, m/s, positive receding (`bistatic_range_rate_mps`).
- $f_D$: Doppler shift, Hz, positive closing (`doppler_hz`).
- $L$: baseline between transmitter and receiver, m (`baseline_m`).
- $\vec{v}$: target velocity in ENU; $\hat{u}_T$, $\hat{u}_R$: unit vectors from the transmitter
  and from the receiver towards the target.

The equations that matter most:

$$\mathrm{az} = \operatorname{atan2}(e, n), \qquad \mathrm{el} = \arcsin(u / R)$$

The arguments are (east, north), not the usual (y, x). That swap is what puts zero at north and
makes azimuth grow clockwise.

$$R_b = R_T + R_R, \qquad \tau = \frac{R_b}{c}, \qquad \dot R_b = \vec{v}\cdot\hat{u}_T + \vec{v}\cdot\hat{u}_R, \qquad f_D = -\frac{\dot R_b}{\lambda}$$

The delay is the whole route over $c$, and the Doppler shift is the rate the route grows, in
wavelengths, with its sign flipped: a receding target lengthens its route and shifts the echo
down. `doppler_shift_hz` is the one place that minus sign is written, and everything else calls
it. A monostatic radar has $R_T = R_R = R$, so $R_b = 2R$ and the familiar pair follows:

$$\tau = \frac{2R}{c}, \qquad f_D = -\frac{2\dot R}{\lambda}$$

`propagation` writes each relation once, as a function of $R_b$ (`propagation_delay_s`,
`propagation_phase_rad`, `doppler_shift_hz`), and both sitings call the same functions. The
range-Doppler map keeps $R_b$ and $\dot R_b$ on its axes too. Only a display halves them:
`plots.display_axes(radar)` gives a monostatic radar a scale of 0.5 and the names `range_m` and
`range_rate_mps`, which the figures and the stepper's CSVs use. A bistatic display keeps $R_b$.

## How it works, step by step

1. **Geodetic to ECEF.** `geodetic_to_ecef_m(latitude_deg, longitude_deg, altitude_m)` returns
   shape `(..., 3)`: one `(x, y, z)` per input, in a closed form with no iteration. Altitude is
   height above the ellipsoid, not above sea level; over North Carolina the two differ by about
   33 m.
2. **ECEF to ENU.** `ecef_to_enu_m(ecef_m, reference_latitude_deg, …)` subtracts the site's ECEF
   position and rotates the offset into the site's east, north and up. It is a pure rotation, so
   the length of the ENU vector is the straight-line distance. `geodetic_to_enu_m` does steps 1
   and 2 in one call; `Site.enu_m(origin)` places one site in another's frame.
3. **ENU to range and angles.** `enu_to_range_azimuth_elevation_rad(enu_m)` returns three arrays
   of shape `(...,)`: range in metres, and azimuth and elevation in radians.
   `range_azimuth_elevation_to_enu_m` goes back the other way.
4. **Motion.** `truth_geometry(scenario)` interpolates each target's track onto the frame times,
   converts it to ENU about the receiving site, and takes its velocity with `numpy.gradient`. A
   `[[target]]` given `position_enu_m` and `velocity_enu_mps` instead moves at constant velocity.
   The geodesy functions themselves have no notion of time.
5. **The route.** `Radar.geometry(position_enu_m, velocity_enu_mps)` takes `(n_targets, 3)`
   positions and velocities and returns a `TargetGeometry` of `(n_targets,)` arrays:
   `range_tx_m`, `range_rx_m`, `bistatic_range_m`, `bistatic_range_rate_mps` and
   `bistatic_angle_rad`, plus `aod_rad` and `aoa_rad` of shape `(n_targets, 2)`, each in its own
   site's frame. It raises `ValueError` for a target exactly on a site.
   `TargetGeometry.monostatic(range_m, radial_velocity_mps)` builds one by hand from a slant range
   and a radial velocity, positive receding, and doubles both.
6. **Paths.** `line_of_sight_paths(radar, geometry, rcs_m2)` returns a `PropagationPaths` with
   one path per target, for either siting. Its fields are `delay_s`, `doppler_hz` and
   `amplitude_linear`, each of shape `(n_paths,)`; `aoa_rad` and `aod_rad` of shape
   `(n_paths, 2)` as (azimuth, elevation); `n_bounces`, which is 1 for a direct return; and
   `doppler_rate_hzps`, zero here. `bistatic_range_m` is `delay_s` times $c$. The amplitude's
   magnitude comes from `received_power_w` (page 1), and a negative `rcs_m2` raises `ValueError`.
7. **Hand-off.** `signal.fmcw_deramp_baseband(paths, radar, n_pulses, rng=rng)` or
   `signal.pulsed_baseband` turns the paths into a cube of shape `(n_pulses, n_samples_per_pri)`.
   They use only `abs(amplitude_linear)` and rebuild the phase from `bistatic_range_at_m`, so
   that the phase moves with the delay from chirp to chirp.

## Try it

This takes the S1 target's first fix through every frame, then builds its route and its path:

```python
import csv

import numpy as np

from radar_forge.plots import display_axes
from radar_forge.propagation import line_of_sight_paths
from radar_forge.radar import (
    enu_to_range_azimuth_elevation_rad,
    geodetic_to_ecef_m,
    geodetic_to_enu_m,
)
from radar_forge.scenario import load_scenario, truth_geometry

scenario = load_scenario("config/scenario_001_fmcw_low_prf.toml")
radar, target = scenario.radar, scenario.targets[0]
site = (radar.tx_site.latitude_deg, radar.tx_site.longitude_deg, radar.tx_site.altitude_m)
with target.trajectory_path.open(newline="", encoding="utf-8") as handle:
    fix = next(csv.DictReader(handle))  # t = 0 s
point = (float(fix["latitude_deg"]), float(fix["longitude_deg"]), target.altitude_m)

x_m, y_m, z_m = geodetic_to_ecef_m(*point)
print(f"ECEF: x = {x_m:.1f} m, y = {y_m:.1f} m, z = {z_m:.1f} m")
east_m, north_m, up_m = enu_m = geodetic_to_enu_m(*point, *site)
print(f"ENU:  e = {east_m:.1f} m, n = {north_m:.1f} m, u = {up_m:.1f} m")
range_m, azimuth_rad, elevation_rad = enu_to_range_azimuth_elevation_rad(enu_m)
azimuth_deg, elevation_deg = np.degrees(azimuth_rad), np.degrees(elevation_rad)
print(f"R = {range_m:.1f} m, az = {azimuth_deg:.2f} deg, el = {elevation_deg:.3f} deg")

truth = truth_geometry(scenario)[0]  # frame 0: the same geometry, plus the rate
rate_mps = truth.bistatic_range_rate_mps[0]
print(f"R_T = R_R = {truth.range_tx_m[0]:.1f} m, R_b = {truth.bistatic_range_m[0]:.1f} m")
print(f"rate = {rate_mps:+.3f} m/s, lambda = {radar.wavelength_m * 1e3:.3f} mm")
paths = line_of_sight_paths(radar, truth, target.rcs_m2)
print(f"tau = {paths.delay_s[0] * 1e6:.4f} us, f_D = {paths.doppler_hz[0]:.1f} Hz")
print(f"P_r = {abs(paths.amplitude_linear[0]) ** 2:.3e} W")
show = display_axes(radar)  # one site: halve, and call it range and radial velocity
shown_m, shown_mps = truth.bistatic_range_m[0] * show.scale, rate_mps * show.scale
print(f"{show.range_name} = {shown_m:.1f}, {show.rate_name} = {shown_mps:+.3f}")
```

```text
ECEF: x = 978741.2 m, y = -5079904.8 m, z = 3720635.4 m
ENU:  e = -13859.8 m, n = -10690.2 m, u = 1416.0 m
R = 17560.7 m, az = 232.36 deg, el = 4.625 deg
R_T = R_R = 17560.7 m, R_b = 35121.4 m
rate = +56.163 m/s, lambda = 30.591 mm
tau = 117.1523 us, f_D = -1835.9 Hz
P_r = 4.959e-15 W
range_m = 17560.7, range_rate_mps = +28.081
```

The target is 13.9 km west and 10.7 km south of the radar, so it lies south-west, at 232°. It is
moving away, so its rate is positive and its Doppler shift negative. The last line is what the
stepper writes to `truth.csv` for this frame.

The bistatic pair in scenario 002 B1 uses the same waveform, with the transmitter at
Raleigh-Durham Airport. The code is the same; only the TOML has a `[site.rx]`:

```python
import numpy as np

from radar_forge.constants import SPEED_OF_LIGHT_MPS
from radar_forge.propagation import line_of_sight_paths
from radar_forge.scenario import load_scenario, truth_geometry

scenario = load_scenario("config/scenario_002_bistatic_xband.toml")
radar = scenario.radar
truth = truth_geometry(scenario)[0]  # frame 0, t = 13329 s
paths = line_of_sight_paths(radar, truth, scenario.targets[0].rcs_m2)
r_t, r_r = truth.range_tx_m[0], truth.range_rx_m[0]

print(f"monostatic: {radar.is_monostatic}")
print(f"R_T = {r_t:.1f} m, R_R = {r_r:.1f} m, L = {radar.baseline_m:.1f} m")
print(f"beta = {np.degrees(truth.bistatic_angle_rad[0]):.2f} deg")
sum_delay_us = (r_t + r_r) / SPEED_OF_LIGHT_MPS * 1e6
print(f"tau = {paths.delay_s[0] * 1e6:.4f} us, (R_T + R_R)/c = {sum_delay_us:.4f} us")
print(f"R_b = {paths.bistatic_range_m[0]:.1f} m")
print(f"rate = {truth.bistatic_range_rate_mps[0]:+.3f} m/s, f_D = {paths.doppler_hz[0]:+.1f} Hz")
```

```text
monostatic: False
R_T = 14800.6 m, R_R = 7523.6 m, L = 19601.4 m
beta = 119.17 deg
tau = 74.4653 us, (R_T + R_R)/c = 74.4653 us
R_b = 22324.1 m
rate = -19.228 m/s, f_D = +628.6 Hz
```

Here the route is shortening, so the rate is negative and the Doppler shift positive.

## Try it in a scenario

Read the scenario file [scenario 002, bistatic](../../config/scenario_002_bistatic_xband.toml),
then run ten frames of B1 into `runs/scenario_002_bistatic_xband/`:

```bash
uv run python scripts/run_scenario.py config/scenario_002_bistatic_xband.toml --frames 10
```

**The knob:** in a copy of the TOML, change `carrier_frequency_hz = 9.8e9` to `2.8e9`. That is
exactly what `scenario_002_bistatic_sband.toml` (B2) does, so run that file the same way.

**What to look at.** The `geometry` stage prints `unambiguous_bistatic_range_rate_mps` =
30.5911 at X-band and 107.069 at S-band: the map holds rates within ±15.30 m/s, or ±53.53 m/s.
Then open `truth.csv` in both folders. `bistatic_range_m`, `range_tx_m`, `range_rx_m`,
`bistatic_angle_deg` and `bistatic_range_rate_mps` are identical, row for row: geometry does
not depend on the carrier, and nor does the delay (74.4653 µs at frame 0). What changes is
$\lambda$, from 30.591 mm to 107.069 mm, and with it the Doppler shift. At frame 9
(t = 13338 s) the rate of −40.649 m/s gives $f_D$ = +1328.8 Hz at X-band, but +379.7 Hz at
S-band. The chirp rate is 1 kHz, so only shifts within ±500 Hz are seen as they are. Now open
`rd_00009.png` in each folder; a bistatic map is labelled in $R_T + R_R$ and its rate, not
halved. At X-band the dashed crosshair, where the folded truth must appear, sits at
−10.06 m/s, and the red truth cross at −40.65 m/s is off the bottom of the map. At S-band nothing
folds, and the truth cross sits on the detection at −40.65 m/s.

No intern task builds on this stage yet; the [intern README](../../interns/README.md) asks for
bug reports on any module, and this one is a good place to look.

## Things that look like bugs and aren't

**The azimuth is 232.36°, but `np.degrees(np.arctan2(north, east))` gives 217.64°.** That is the
maths angle, counter-clockwise from east. The library uses the compass angle, clockwise from
north, which is `atan2(east, north)`. `enu_to_range_azimuth_elevation_rad` says so in its
docstring.

**S1's rate is positive but its Doppler shift is negative.** The rate is the time derivative of
$R_b$, positive receding; the Doppler shift is positive closing. `doppler_shift_hz` relates them
once, $f_D = -\dot R_b / \lambda$.

**S1's frame 0 has a Doppler shift of −1835.9 Hz, far outside the ±500 Hz the radar can see.**
Paths carry the true, unfolded shift. Folding happens later, when the slow-time FFT samples it
once per chirp: −1835.9 Hz folds to +164.1 Hz, a bistatic rate of −5.02 m/s, so the figure shows
−2.51 m/s of radial velocity for a target receding at 28.08 m/s. Page 4 covers this.

**The map's arrays put the S1 target at 35121.4 m, twice its range.** The map, `TargetGeometry`
and `PropagationPaths` all carry $R_b = R_T + R_R$, which for one site is $2R$. The figure and
the CSVs halve it through `plots.display_axes`, so they read 17.56 km. A bistatic map is not
halved: surfaces of constant $R_b$ are ellipsoids with the two sites at the foci, and half of
$R_b$ is not a distance from either site.

**The bistatic delay (74.4653 µs) is longer than a real bistatic receiver would measure.** A real
receiver times echoes from the direct pulse it hears from the transmitter, so it would measure
$(R_T + R_R - L)/c$, here 9.0821 µs. The library keeps the absolute delay on purpose, because it
carries the right carrier phase, and it models no direct signal (`radar`'s module docstring).

## Limits

- **Free space only** (`propagation`'s module docstring): no atmospheric absorption, no rain, no
  multipath, no clutter, no obstruction. One direct path per target.
- **No refraction.** Elevations are geometric, not the slightly higher apparent angle a real radar
  sees. The difference grows at low elevation and long range;
  `constants.FOUR_THIRDS_EARTH_RADIUS_M` is the standard correction, and nothing applies it yet.
- **ENU is a flat tangent plane.** Over 20 km the Earth curves away by about 30 m. Range is still
  exact, because ENU is a rotation of ECEF.
- **Ellipsoidal heights only.** There is no geoid model, so a height above sea level must be
  converted before it goes in.
- **Only the amplitude's magnitude reaches `signal`.** A backend cannot impose an extra phase,
  such as a complex reflection coefficient, through `amplitude_linear` today
  (`PropagationPaths`'s docstring).
- **Angles are carried but not used.** `aoa_rad` and `aod_rad` wait for the array modules; the
  generators assume one isotropic channel.
- **Bistatic:** the sites are perfectly synchronised in time and phase, the direct signal from
  transmitter to receiver is not modelled, and the bistatic RCS is one fixed number at every
  $\beta$ (`radar`'s module docstring, and scenario 002).
- **The velocity is a smoothed difference.** `truth_geometry` differences a track interpolated
  between fixes, so the rate is the mean over the seconds between fixes, not the instantaneous
  one.

## References

- National Imagery and Mapping Agency, *Department of Defense World Geodetic System 1984*, NIMA
  TR8350.2, 3rd ed., Amendment 1, 2000, §3.2 (defining parameters) and §4.3, eqs. (4-14)–(4-15)
  (geodetic to ECEF).
- P. Misra and P. Enge, *Global Positioning System: Signals, Measurements, and Performance*, 2nd
  ed., Ganga-Jamuna Press, 2006, app. 4.A (coordinate conversion; azimuth and elevation from
  ENU).
- M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014, §2.2 (the
  radar range equation), §2.6.1 (the two-way propagation phase and Doppler shift).
- N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005, §4.1 (the bistatic range
  equation), §6.1 (target Doppler) and ch. 3 (the bistatic range ellipse).
- `spec/data-001-formats.md` §8, decisions D1 (paths, not samples) and D6 (`PropagationPaths` covers
  bistatic geometry unchanged).

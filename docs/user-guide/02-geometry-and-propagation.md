# 2. Geometry and propagation

This stage turns *where things are* into *what the receiver hears*. It takes a target's latitude,
longitude and altitude, works out its range, angles and radial velocity from the radar, and turns
those into a delay, a Doppler shift and an amplitude for each echo. It is the second and third
boxes of [the chain](README.md#the-chain), and it covers two modules:
`radar_forge.core.geodesy` and `radar_forge.core.propagation`.

## What this stage does

A scenario says where the radar and the target are in the coordinates the world uses: WGS-84
latitude, longitude and height above the ellipsoid (the **geodetic** frame). Radar geometry wants
a flat Cartesian frame with the radar at the origin. `geodesy` is the bridge. It goes through
Earth-centred, Earth-fixed (**ECEF**) metres to a local **east-north-up (ENU)** frame on the
tangent plane at the radar. It is the only module in the library that knows the Earth is round.

From ENU it is a short step to the three numbers a radar measures: range, azimuth and elevation.
Azimuth is a compass bearing: zero at true north, increasing clockwise towards east.

`propagation` then turns that geometry into **paths**. A path is one way energy gets from the
transmitter to the receiver, described by four things: a delay, a Doppler shift, a complex
amplitude and a pair of angles. Today there is one path per target, the direct line of sight in
free space. The next module, `signal` (page 3), turns paths into samples.

```
 latitude, longitude, height      geodetic_to_ecef_m      ecef_to_enu_m
 (WGS-84, per site and target) ------------------> ECEF ----------------> ENU (e, n, u)
                                                                            |
                                         enu_to_range_azimuth_elevation_deg |
                                                                            v
 PropagationPaths <-- line_of_sight_paths / bistatic_line_of_sight_paths -- R, az, el, v_r
 (delay, Doppler, amplitude, angles)  ---> signal (page 3)
```

That split between paths and samples is decision **D1** in `spec/structure.md`: a propagation
model (this module now, a ray tracer later) computes paths, and only `signal` computes samples.
A new propagation model therefore needs no new signal code. Scenario 002 put this to the test by
moving the transmitter 19.6 km away from the receiver. The same `PropagationPaths` class and the
same signal code serve it unchanged (decision D6).

## Quantities and equations

The monostatic rows are for scenario 001 S1 at frame 0 (t = 0 s, the target's first fix). The
bistatic rows, marked B1, are for scenario 002 B1 at its frame 0 (t = 13329 s). Every value comes
from the two examples under [Try it](#try-it).

| Quantity | Symbol | Equation | In the code | Scenario 001 S1, frame 0 |
| :-- | :-- | :-- | :-- | :-- |
| Target's geodetic position | $\varphi, \ell, h$ | from the track CSV; $h$ is the scenario's fixed altitude | `load_flight_csv` | 35.90608°, −79.0945°, 1500 m |
| Prime-vertical radius | $N$ | $N = a / \sqrt{1 - e^2 \sin^2\varphi}$ | inside `geodetic_to_ecef_m` | — |
| ECEF position | $(x, y, z)$ | $x = (N + h)\cos\varphi\cos\ell$, $y = (N + h)\cos\varphi\sin\ell$, $z = (N(1 - e^2) + h)\sin\varphi$ | `geodetic_to_ecef_m` | (978741.2, −5079904.8, 3720635.4) m |
| ENU position | $(e, n, u)$ | translate to the radar, then rotate by the radar's $\varphi_0, \ell_0$ | `ecef_to_enu_m`, `geodetic_to_enu_m` | (−13859.8, −10690.2, 1416.0) m |
| Slant range | $R$ | $R = \sqrt{e^2 + n^2 + u^2}$ | `enu_to_range_azimuth_elevation_deg` | 17560.7 m |
| Azimuth | $\mathrm{az}$ | $\mathrm{az} = \operatorname{atan2}(e, n)$, wrapped to $[0°, 360°)$ | `enu_to_range_azimuth_elevation_deg` | 232.36° |
| Elevation | $\mathrm{el}$ | $\mathrm{el} = \arcsin(u / R)$ | `enu_to_range_azimuth_elevation_deg` | 4.625° |
| Radial velocity | $v_r$ | $v_r = -\,dR/dt$, closing positive | `Frame.radial_velocity_mps` | −28.082 m/s (opening) |
| Wavelength | $\lambda$ | $\lambda = c / f_c$ | `Radar.wavelength_m` | 30.591 mm |
| Round-trip delay | $\tau$ | $\tau = 2R / c$ | `PropagationPaths.delay_s` | 117.1523 µs |
| Doppler shift | $f_d$ | $f_d = 2 v_r / \lambda$ | `PropagationPaths.doppler_hz` | −1835.9 Hz |
| Received power | $P_r$ | $P_r = \lvert a \rvert^2 = P_t G_t G_r \lambda^2 \sigma / ((4\pi)^3 R^4)$ | `abs(PropagationPaths.amplitude_linear) ** 2` | 4.959 × 10⁻¹⁵ W |
| Two-way phase | $\arg a$ | $-4\pi R / \lambda$ | `PropagationPaths.amplitude_linear` | — |
| Transmit and receive range (B1) | $R_t$, $R_r$ | ENU distance from each site | `Frame.range_tx_m`, `Frame.range_rx_m` | 14800.6 m, 7523.6 m |
| Baseline (B1) | $L$ | distance between the two sites | `BistaticRadar.baseline_m` | 19601.4 m |
| Bistatic angle (B1) | $\beta$ | $\cos\beta = (R_t^2 + R_r^2 - L^2) / (2 R_t R_r)$ | `BistaticRadar.bistatic_angle_rad` | 119.17° |
| Bistatic delay (B1) | $\tau$ | $\tau = (R_t + R_r) / c$ | `PropagationPaths.delay_s` | 74.4653 µs |
| Path range (B1) | $R$ | $R = \tau c / 2 = (R_t + R_r) / 2$ | `PropagationPaths.range_m` | 11162.1 m |
| Bisector velocity (B1) | $v_b$ | $v_b = -\tfrac{1}{2}\, d(R_t + R_r)/dt$ | `Frame.radial_velocity_mps` | 9.614 m/s |
| Bistatic Doppler (B1) | $f_d$ | $f_d = 2 v_b / \lambda = (\vec{v}\cdot\hat{u}_t + \vec{v}\cdot\hat{u}_r) / \lambda$ | `bistatic_doppler_hz`, `PropagationPaths.doppler_hz` | 628.6 Hz |

Symbols used only on this page:

- $\varphi$, $\ell$, $h$: geodetic latitude, longitude and height above the WGS-84 ellipsoid (the
  `geodesy` docstrings write longitude as $\lambda$; here $\lambda$ is kept for the wavelength).
  $\varphi_0$, $\ell_0$ are the radar's own.
- $a$, $e^2$: the ellipsoid's semi-major axis (6378137 m) and first eccentricity squared,
  $e^2 = f(2 - f)$ from the flattening $f$ (`WGS84_SEMI_MAJOR_AXIS_M`, `WGS84_FLATTENING`).
- $N$: radius of curvature in the prime vertical, m.
- $(x, y, z)$: ECEF coordinates, m. $(e, n, u)$: east, north, up, m.
- $\mathrm{az}$, $\mathrm{el}$: azimuth and elevation (`azimuth_deg`, `elevation_deg`).
- $a$ (in $\lvert a \rvert$, $\arg a$): a path's complex amplitude (`amplitude_linear`).
- $L$: baseline between transmitter and receiver, m (`baseline_m`).
- $v_b$: bisector velocity, m/s, closing positive (`bisector_velocity_mps`).
- $\vec{v}$: target velocity in ENU; $\hat{u}_t$, $\hat{u}_r$: unit vectors from the target towards
  the transmitter and the receiver.

The four equations that matter most:

$$\mathrm{az} = \operatorname{atan2}(e, n), \qquad \mathrm{el} = \arcsin(u / R)$$

The arguments are (east, north), not the usual (y, x). That swap is what puts zero at north and
makes azimuth grow clockwise.

$$\tau = \frac{2R}{c}, \qquad f_d = \frac{2 v_r}{\lambda}$$

The echo travels out and back, so it is delayed by twice the range and shifted by twice the
closing speed in wavelengths.

$$\tau = \frac{R_t + R_r}{c}, \qquad f_d = \frac{1}{\lambda}\left(\vec{v}\cdot\hat{u}_t + \vec{v}\cdot\hat{u}_r\right)$$

With the sites apart, the delay is the whole route over $c$, and the Doppler shift is the rate the
whole route shortens, in wavelengths. Put the sites together and both become the monostatic forms.

$$R = \frac{R_t + R_r}{2}, \qquad v_b = -\frac{1}{2}\frac{d(R_t + R_r)}{dt}$$

A bistatic path reports half the route and half its rate. The signal code only ever uses range
as $2R/c$ and velocity as $2v/\lambda$, so the factor of a half cancels the factor of two, and it
gets exactly the bistatic delay and Doppler. That is why the same signal code serves both sitings.
The module docstring of `propagation` explains this in full.

## How it works, step by step

1. **Geodetic to ECEF.** `geodetic_to_ecef_m(latitude_deg, longitude_deg, altitude_m)` returns
   shape `(..., 3)`: one `(x, y, z)` per input, in a closed form with no iteration. Altitude is
   height above the ellipsoid, not above sea level; over North Carolina the two differ by about
   33 m.
2. **ECEF to ENU.** `ecef_to_enu_m(ecef_m, reference_latitude_deg, …)` subtracts the radar's ECEF
   position and rotates the offset into the radar's east, north and up. It is a pure rotation, so
   the length of the ENU vector is the straight-line distance. `geodetic_to_enu_m` does steps 1
   and 2 in one call, and is the one a scenario uses.
3. **ENU to radar coordinates.** `enu_to_range_azimuth_elevation_deg(enu_m)` returns three arrays
   of shape `(...,)`: range in metres and azimuth and elevation in degrees.
   `enu_to_range_azimuth_elevation_rad` is the same in radians, and the tracker's measurement model
   calls it too, so there is one copy of this geometry. `range_azimuth_elevation_to_enu_m` goes
   back the other way.
4. **Radial velocity.** The scenario pipeline gets $v_r$ by differencing range over time
   (`pipelines.trajectories.to_radar_frame`, with `numpy.gradient`), and puts it on each `Frame`.
   `geodesy` itself has no notion of time.
5. **Paths.** `line_of_sight_paths(radar, range_m, radial_velocity_mps, rcs_m2)` takes one value
   per target and returns a `PropagationPaths` with `n_paths` = number of targets. Its fields are
   `delay_s`, `doppler_hz` and `amplitude_linear`, each shape `(n_paths,)`; `aoa_rad` and `aod_rad`
   of shape `(n_paths, 2)` as (azimuth, elevation); and `bounce_count`, which is 1 for a direct
   return. The amplitude's magnitude comes from `radar_equation.received_power_w`. This module
   adds no link-budget maths of its own.
6. **Bistatic paths.** For a `BistaticRadar`, `bistatic_line_of_sight_paths(radar, range_tx_m,
   range_rx_m, bisector_velocity_mps, bistatic_rcs_m2)` does the same with two ranges and the
   bisector velocity, and uses `radar_equation.bistatic_received_power_w`. If you have the target's
   velocity vector rather than $v_b$, `bistatic_doppler_hz` gives $f_d$ from the vector form.
7. **Hand-off.** `signal.fmcw_deramp_baseband(paths, radar, n_pulses)` or `signal.pulsed_baseband`
   turns the paths into a cube of shape `(n_pulses, n_samples)`. They use only
   `abs(amplitude_linear)` and rebuild the phase from `delay_s`, so that the phase moves with the
   delay from chirp to chirp.

## Try it

This takes the S1 target's first fix through every frame, then builds its path:

```python
from radar_forge.core.geodesy import (
    enu_to_range_azimuth_elevation_deg,
    geodetic_to_ecef_m,
    geodetic_to_enu_m,
)
from radar_forge.core.propagation import line_of_sight_paths
from radar_forge.pipelines.scenarios import iterate_frames, load_scenario
from radar_forge.pipelines.trajectories import load_flight_csv

scenario = load_scenario("scenarios/scenario_001_fmcw_low_prf.toml")
radar = scenario.bursts[0]
site = (radar.latitude_deg, radar.longitude_deg, radar.altitude_m)
track = load_flight_csv(scenario.trajectory_path, altitude_m=scenario.target_altitude_m)
target = (track.latitude_deg[0], track.longitude_deg[0], track.altitude_m[0])  # t = 0 s

x_m, y_m, z_m = geodetic_to_ecef_m(*target)
print(f"ECEF: x = {x_m:.1f} m, y = {y_m:.1f} m, z = {z_m:.1f} m")
east_m, north_m, up_m = enu_m = geodetic_to_enu_m(*target, *site)
print(f"ENU:  e = {east_m:.1f} m, n = {north_m:.1f} m, u = {up_m:.1f} m")
range_m, azimuth_deg, elevation_deg = enu_to_range_azimuth_elevation_deg(enu_m)
print(f"R = {range_m:.1f} m, az = {azimuth_deg:.2f} deg, el = {elevation_deg:.3f} deg")

frame = next(iterate_frames(scenario))  # frame 0: the same geometry, plus v_r
v_r = frame.radial_velocity_mps
paths = line_of_sight_paths(radar, frame.range_m, v_r, scenario.target.rcs_m2)
print(f"v_r = {v_r:.3f} m/s, lambda = {radar.wavelength_m * 1e3:.3f} mm")
print(f"tau = {paths.delay_s[0] * 1e6:.4f} us, f_d = {paths.doppler_hz[0]:.1f} Hz")
print(f"P_r = {abs(paths.amplitude_linear[0]) ** 2:.3e} W")
```

```text
ECEF: x = 978741.2 m, y = -5079904.8 m, z = 3720635.4 m
ENU:  e = -13859.8 m, n = -10690.2 m, u = 1416.0 m
R = 17560.7 m, az = 232.36 deg, el = 4.625 deg
v_r = -28.082 m/s, lambda = 30.591 mm
tau = 117.1523 us, f_d = -1835.9 Hz
P_r = 4.959e-15 W
```

The target is 13.9 km west and 10.7 km south of the radar, so it lies south-west, at 232°. It is
moving away, so $v_r$ and $f_d$ are negative.

The bistatic pair in scenario 002 B1 uses the same waveform, with the transmitter at
Raleigh-Durham Airport:

```python
import numpy as np

from radar_forge.core.constants import SPEED_OF_LIGHT_MPS
from radar_forge.core.propagation import bistatic_line_of_sight_paths
from radar_forge.pipelines.scenarios import iterate_frames, load_scenario

scenario = load_scenario("scenarios/scenario_002_bistatic_xband.toml")
pair = scenario.bursts[0]
frame = next(iterate_frames(scenario))  # frame 0, t = 13329 s
r_t, r_r, v_b = frame.range_tx_m, frame.range_rx_m, frame.radial_velocity_mps
paths = bistatic_line_of_sight_paths(pair, r_t, r_r, v_b, scenario.target.rcs_m2)

print(f"R_t = {r_t:.1f} m, R_r = {r_r:.1f} m, L = {pair.baseline_m:.1f} m")
print(f"beta = {np.degrees(pair.bistatic_angle_rad(r_t, r_r)):.2f} deg")
sum_delay_us = (r_t + r_r) / SPEED_OF_LIGHT_MPS * 1e6
print(f"tau = {paths.delay_s[0] * 1e6:.4f} us, (R_t + R_r)/c = {sum_delay_us:.4f} us")
print(f"range_m = {paths.range_m[0]:.1f} m (the half-sum)")
print(f"v_b = {v_b:.3f} m/s, f_d = {paths.doppler_hz[0]:.1f} Hz")
```

```text
R_t = 14800.6 m, R_r = 7523.6 m, L = 19601.4 m
beta = 119.17 deg
tau = 74.4653 us, (R_t + R_r)/c = 74.4653 us
range_m = 11162.1 m (the half-sum)
v_b = 9.614 m/s, f_d = 628.6 Hz
```

## Try it in a scenario

Read the companion page [scenario 002, bistatic](../scenarios/scenario-002-bistatic.md), then run
B1:

```bash
uv run python scripts/run_scenario.py scenarios/scenario_002_bistatic_xband.toml --out out/b1 --frames 10
```

**The knob:** in a copy of the TOML, change `carrier_frequency_hz = 9.8e9` to `2.8e9`. That is
exactly what `scenario_002_bistatic_sband.toml` (B2) does. Run it into `out/b2`.

**What to look at.** First `truth.csv` in both folders. Range, $R_t$, $R_r$, $\beta$ and $v_b$ are
identical, row for row: geometry does not depend on the carrier, and nor does the delay
(74.4653 µs at frame 0). What changes is $\lambda$, from 30.591 mm to 107.069 mm, and with it
the Doppler. At frame 9 (t = 13338 s), $v_b$ = 20.324 m/s gives $f_d$ = 1328.8 Hz at X-band, but
379.6 Hz at S-band. The chirp rate is 1 kHz, so only shifts within ±500 Hz are seen as they
are. Now open `rd_00009.png` in each folder. At X-band the peak sits at +5.02 m/s, folded, with
the true +20.3 m/s off the top of the map. At S-band it sits at +20.28 m/s, where it belongs,
because the unambiguous velocity grew from ±7.6478 m/s to ±26.7672 m/s.

No intern task builds on this stage yet; the [intern README](../../interns/README.md) asks for
bug reports on any module, and this one is a good place to look.

## Things that look like bugs and aren't

**The azimuth is 232.36°, but `np.degrees(np.arctan2(north, east))` gives 217.64°.** That is the
maths angle, counter-clockwise from east. The library uses the compass angle, clockwise from
north, which is `atan2(east, north)`. `enu_to_range_azimuth_elevation_rad` says so in its
docstring.

**S1's frame 0 has a Doppler shift of −1835.9 Hz, far outside the ±500 Hz the radar can see.**
Paths carry the true, unfolded shift. Folding happens later, when the slow-time FFT samples it
once per chirp: −1835.9 Hz folds to +164.1 Hz, so the map shows +2.51 m/s for a target opening at
28.08 m/s. Page 4 covers this.

**A bistatic path's `range_m` (11162.1 m) is neither $R_t$ (14800.6 m) nor $R_r$ (7523.6 m).**
It is the half-sum, the range a monostatic radar would need to give the same delay. A bistatic
map's range axis is in this unit too, labelled "bistatic mean range". Constant half-sum surfaces
are ellipsoids with the two sites at the foci, not spheres.

**The bistatic delay (74.4653 µs) is longer than a real bistatic receiver would measure.** A real
receiver times echoes from the direct pulse it hears from the transmitter, so it would measure
$(R_t + R_r - L)/c$, here 9.0821 µs. The library keeps the absolute delay on purpose, because it
carries the right carrier phase. `bistatic_line_of_sight_paths`'s Notes say so.

## Limits

- **Free space only** (the "Approximations" section of `propagation`'s module docstring): no
  atmospheric absorption, no rain, no multipath, no clutter, no obstruction. One direct path per
  target.
- **No refraction.** Elevations are geometric, not the slightly higher apparent angle a real radar
  sees. The difference grows at low elevation and long range;
  `constants.FOUR_THIRDS_EARTH_RADIUS_M` is the standard correction, and nothing applies it yet.
- **ENU is a flat tangent plane.** Over 20 km the Earth curves away by about 30 m
  (`ecef_to_enu_m`'s Notes). Range is still exact, because ENU is a rotation of ECEF.
- **Ellipsoidal heights only.** There is no geoid model, so a height above sea level must be
  converted before it goes in.
- **Only the amplitude's magnitude reaches `signal`.** A backend cannot impose an extra phase,
  such as a complex reflection coefficient, through `amplitude_linear` today
  (`PropagationPaths`'s docstring calls this a gap in D1).
- **Angles are carried but not used.** `aoa_rad` and `aod_rad` wait for the array modules; the
  generators assume one isotropic channel.
- **Bistatic:** the sites are perfectly synchronised in time and phase, the direct signal from
  transmitter to receiver is not modelled, and the bistatic RCS is one fixed number at every
  $\beta$ (`BistaticRadar`'s Notes, and the scenario 002 companion).
- **Radial velocity is a smoothed difference.** The pipeline differences an interpolated range,
  so $v_r$ is the mean closing rate between fixes, not the instantaneous one
  (`to_radar_frame`'s Notes).

## References

- National Imagery and Mapping Agency, *Department of Defense World Geodetic System 1984*, NIMA
  TR8350.2, 3rd ed., Amendment 1, 2000, §3.2 (defining parameters) and §4.3, eqs. (4-14)–(4-15)
  (geodetic to ECEF).
- P. Misra and P. Enge, *Global Positioning System: Signals, Measurements, and Performance*, 2nd
  ed., Ganga-Jamuna Press, 2006, app. 4.A (coordinate conversion; azimuth and elevation from
  ENU).
- M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014, §2.2 (the
  radar range equation), §2.6.1 (the two-way propagation phase and Doppler shift).
- N. J. Willis, *Bistatic Radar*, 2nd ed., SciTech Publishing, 2005, §6.1 (target Doppler) and
  ch. 3 (the range-sum ellipse).
- `spec/structure.md`, decisions D1 (paths, not samples) and D6 (`PropagationPaths` covers
  bistatic geometry unchanged).

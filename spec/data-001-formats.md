# Data 001 — Files a run writes

> Status: schema version **2.0.0**. This is the contract for what the stepper
> (`radar_forge.evaluate`, run by `scripts/run_scenario.py`) writes into a run directory. Each
> stage writes its own files; the columns are the keys of the rows each stage builds.

| § | Section |
| :--- | :--- |
| [1](#1-scope) | Scope |
| [2](#2-principles) | Principles |
| [3](#3-which-format-for-which-data) | Which format for which data |
| [4](#4-the-run-directory) | The run directory |
| [5](#5-conventions-every-file-shares) | Conventions every file shares |
| [6](#6-file-specifications) | File specifications |
| [7](#7-versioning) | Versioning |
| [8](#8-decisions) | Decisions |
| [—](#references) | References |

## 1. Scope

In scope: the files a scenario run writes, from truth to tracks and metrics, with their columns,
dtypes and units. Out of scope: the physics that produces the numbers (the scenario specs and the
code), real-time streaming, and the ASTERIX binary encoding (§6 borrows its vocabulary, not its
bytes).

## 2. Principles

1. **One format per data shape.** Nobody should have to ask which format a file is in.
2. **Units in every name.** `docs/conventions/style.md` §2 applies to column names as it does to
   parameters: `range_m`, `peak_power_w`, `azimuth_deg`.
3. **Linear on disk.** Stored quantities are linear SI. A `_db` column may sit beside its linear
   source (`snr_db` beside `peak_power_w`) but never replaces it.
4. **Floats round-trip.** Every float is written in its shortest exact representation (Python's
   `repr`), never a fixed `%.3f`, so a file read back equals what was written and two runs can be
   compared by hash.

## 3. Which format for which data

| Data | Format | Why |
| :--- | :--- | :--- |
| Scenario, radar and waveform inputs | TOML | The project's configuration format, read with stdlib `tomllib` by `radar_forge/scenario.py` |
| Tables: trajectory, truth, detections, tracks, metrics | CSV (RFC 4180, UTF-8), one row per entity per frame | Opens in every tool including a spreadsheet; no dependency |
| Arrays: raw IQ | NumPy `.npz`, one file per frame | The core stack reads and writes it; no dependency |
| Run metadata and provenance | JSON | Machine output, not configuration |
| Figures | PNG, MP4 | For people, never parsed back |

## 4. The run directory

```text
runs/<name>/               # --out overrides it
├── truth.csv              # §6.4, geometry stage
├── iq_00000.npz …         # §6.7, iq stage, only with --save-iq
├── rd{suffix}_00000.png … # §6.8, rd stage, unless --no-plots
├── detections.csv         # §6.5, detect stage
├── tracks.csv             # §6.6, track stage
├── metrics.csv            # §6.9, track stage
└── track.png              # §6.8, track stage, unless --no-plots
```

A stage that the scenario's `[verify]` table does not list writes nothing. The run's checks and
measured values are printed, and returned by `evaluate.run` as a `Report`; they are not written.

## 5. Conventions every file shares

| Topic | Convention |
| :--- | :--- |
| **Time** | `time_s` is seconds on the trajectory files' own time base, float64, monotonic within a file: every target's file and the frame times share it, and a file is never rebased to its first fix. There is no absolute epoch and no `time_utc` |
| **Frame** | `frame` is a zero-based integer, one revisit (a *scan* to the tracker). With `time_s` it is the join key across files. A *burst* is one coherent processing interval inside a frame |
| **Frames of reference** | Geodetic WGS-84 → ECEF → local ENU tangent plane, per `radar_forge/radar.py`. The ENU origin is the receiver site, which is the radar's when monostatic |
| **Angles** | `azimuth_deg` is a compass bearing from true north, increasing clockwise; `elevation_deg` is up from the local horizontal. Radians inside the library, degrees in files |
| **Range and range rate** | **Range rate is the time derivative of range, positive receding; Doppler = -rate/wavelength.** `range_m` is the slant range for a monostatic radar and the bistatic range `R_T + R_R` (`bistatic_range_m`) for a bistatic one, and its rate is `bistatic_range_rate_mps = d(bistatic_range_m)/dt`, so `doppler_hz = -bistatic_range_rate_mps / wavelength_m`. The range-Doppler axes and the tracker's state carry bistatic range and rate; the files of a monostatic run halve both and name them `range_m` and `range_rate_mps` (`plots.display_axes`), and a bistatic run writes `bistatic_range_m` and `bistatic_range_rate_mps`. Every rate column follows it |
| **Units** | The `style.md` §2 suffix table. Compound units: `_m2`, `_m2ps2`, `_mps2` |
| **Identifiers** | `track_id` is an integer, never reused. `detection_id` is an integer unique within a frame, so `(frame, detection_id)` is the detection key |
| **Vocabulary** | A *detection* is the CFAR-derived point: a connected group of threshold crossings in one burst's range-Doppler map (`radar_forge.detection.Detection`); `detections.csv` holds them in physical units. A *track* is a filtered state built from detections over scans |
| **Missing values** | An empty cell means not applicable or not yet known (an unfolded rate before a fold is selected). `NaN` means computed and invalid. They are never interchanged |
| **Booleans** | `0` / `1` |
| **Enumerations** | Lowercase ASCII: `tentative`, `confirmed`, `coasting`, `deleted`; `accepted`, `missing_pair`, `ambiguous_pair`, `unresolved_rate` |
| **Array axis order** | The library's in-memory order from `style.md` §3.1: slow time before fast time, receive channels last |

## 6. File specifications

Columns are written in the order listed; readers select them by name. The "Req." column says when
a column is present: **R** always, **B** bistatic runs only. `<range>` and `<rate>` stand for
`range_m` and `range_rate_mps` in a monostatic run (halved) and `bistatic_range_m` and
`bistatic_range_rate_mps` in a bistatic one (§5).

### 6.1 Run metadata

Not written since 2.0.0. The scenario file and the stepper's printed report say what was run; a
`metadata.json` manifest returns when a consumer needs one.

### 6.2 Scenario input

The scenario that was run is `config/<name>.toml`. Its schema is whatever `radar_forge/scenario.py`
parses; this document does not restate it, because that loader is the single definition of what a
scenario file may contain.

### 6.3 Trajectory input

A CSV, one row per fix, read by `radar_forge/scenario.py`: `time_s` (s), `latitude_deg` and
`longitude_deg` (WGS-84), and optionally `altitude_m` (above the ellipsoid; otherwise the
`[[target]]`'s `altitude_m` holds for every fix). Times must increase strictly, and are read
as written: two targets' files on one clock play in step, and `[frames].start_time_s` is on it too.

### 6.4 `truth.csv`

One row per target per frame: the true, unfolded geometry.

| Column | dtype | Unit | Req. | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `frame`, `time_s` | int, float64 | —, s | R | |
| `target` | string | — | R | The `[[target]]` name |
| `<range>` | float64 | m | R | Route length R_T + R_R, halved monostatic |
| `<rate>` | float64 | m/s | R | Its time derivative, positive receding, halved monostatic |
| `azimuth_deg`, `elevation_deg` | float64 | deg | R | True-north clockwise, from the receiver |
| `range_tx_m`, `range_rx_m` | float64 | m | B | Transmitter to target, target to receiver |
| `bistatic_angle_deg` | float64 | deg | B | Angle at the target between transmitter and receiver |

### 6.5 `detections.csv`

One row per detection per frame, every burst's. The vocabulary follows the ASTERIX CAT048 target
report; the units are SI.

| Column | dtype | Unit | Req. | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `frame`, `time_s` | int, float64 | —, s | R | |
| `detection_id` | int | — | R | Unique within the frame; `pipeline.Plots` row order |
| `burst_index` | int | — | R | The burst whose map the detection came from |
| `status` | enum | — | R | `accepted` (reaches the tracker from burst 0) or, for a dual-PRF pair, `missing_pair`, `ambiguous_pair`, `unresolved_rate` |
| `<range>` | float64 | m | R | Power-weighted centroid range, folded if the map folds |
| `<rate>_folded` | float64 | m/s | R | Rate as read off the map, inside its interval |
| `<rate>` | float64 | m/s | R | Unfolded rate; empty until known |
| `rate_index`, `range_index` | float64 | bins | R | Fractional centroid indices into the burst's map |
| `sigma_range_m`, `sigma_rate_mps` | float64 | m, m/s | R | A bin over √12, halved monostatic |
| `peak_power_w` | float64 | W | R | Peak cell of the cluster |
| `n_cells` | int | — | R | Cells in the cluster |
| `pair_id` | int | — | R | The same integer on both detections of a dual-PRF pair; empty otherwise |
| `unfolded_<range>` | float64 | m | R | The range an accepted pair agrees on (`dsp.unfold_range_dual_prf`); empty otherwise |

### 6.6 `tracks.csv`

One row per track per frame, tentative and coasting tracks included, and a deleted track once. The
vocabulary follows the ASTERIX CAT062 system track. The state is range and rate.

| Column | dtype | Unit | Req. | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `frame`, `time_s` | int, float64 | —, s | R | |
| `track_id` | int | — | R | |
| `status` | enum | — | R | `tentative`, `confirmed`, `coasting` or `deleted` |
| `<range>`, `<rate>` | float64 | m, m/s | R | The state; range wrapped into the map's span |
| `var_range_m2`, `var_rate_m2ps2` | float64 | m², m²/s² | R | Covariance diagonal |
| `cov_0_1` | float64 | m²/s | R | Covariance off-diagonal |
| `nis` | float64 | — | R | Normalised innovation squared of this scan's update; empty otherwise |
| `n_hits`, `n_misses_in_a_row` | int | — | R | |
| `detection_id` | int | — | R | The detection the track took or was born from this scan; empty on a miss |

### 6.7 `iq_NNNNN.npz`

One compressed NumPy archive per frame (`np.savez_compressed`), with one array per burst named
`burst0`, `burst1`, … Each array is `(n_pulses, n_samples)` complex128 in `sqrt(W)`, so `|a|²` is
watts; the bursts of a dual-PRF run differ in shape. The archive carries no axes or units: the
scenario file defines them. Raw IQ is by far the largest output and is regenerable from the
scenario file and seed.

### 6.8 Figures

`rd{suffix}_NNNNN.png`, one per frame and burst, where `suffix` is empty for a single burst and
`_burst{i}` for each burst of a multi-burst run, and one `track.png`. They are decibel renderings
for people, and nothing reads them back.

### 6.9 `metrics.csv`

Tidy and long, one row per metric per target's primary track (the track nearest it in the most
confirmed frames). The unit is in the metric's name.

| Column | dtype | Req. | Meaning |
| :--- | :--- | :--- | :--- |
| `metric` | string | R | `<range>`'s `_rmse_m` (`range_rmse_m` monostatic, halved), `<rate>`'s `_rmse_mps`, `nees_fraction` (frames inside χ²₂'s 95 % interval), `confirmation_latency_scans` |
| `target`, `track_id` | string, int | R | |
| `frame_start`, `frame_end` | int | R | The inclusive frame window the metric covers |
| `value` | float64 | R | |

## 7. Versioning

- The schema version is semver, stated at the top of this document.
- **Minor**: add a column or key. **Major**: rename, remove, re-type, re-unit or re-sign anything.
- Readers select columns by name, ignore unknown ones, and refuse a major version they do not know.

| Version | Change |
| :--- | :--- |
| 1.0.0 | First proposal |
| 1.1.0 | `detections.csv` adds `burst_index`, `status`, `pair_id`; `metadata.json` `tracking` adds `range_period_m`, `unused` |
| 1.2.0 | `detections.csv` adds `range_unfolded_m` |
| 2.0.0 | Range and rate are bistatic, positive receding, halved and renamed in monostatic files (§5); new `truth.csv`, `detections.csv`, `tracks.csv` and `metrics.csv` columns; `metadata.json` and movies dropped; runs under `runs/` |

## 8. Decisions

Decisions the code follows, moved here from the retired structure spec.

### D5 — Velocity sign

Range rate is the time derivative of range, positive receding, in the code and in every file
(§5). Doppler is `-rate/wavelength`, written once in `propagation.doppler_shift_hz`.

### D1 — The path set is the backend contract

A propagation backend returns `PropagationPaths` (delay, Doppler, complex amplitude, angles of
arrival and departure, bounce count), never baseband. `radar_forge/signal.py` turns any path set into
samples, so waveform changes live in one place and backends compare on identical geometry.

### D6 — The path set covers bistatic geometry unchanged

`delay_s·c` is `bistatic_range_m` (`2R` monostatic), and Doppler is `-bistatic_range_rate_mps/λ`.
Each relation is written once for both sitings; the two legs are geometry and belong to the radar,
not the path set.

### D7 — One siting-agnostic pipeline

One `Radar(transmitter, receiver, tx_site, rx_site=None)`, where `rx_site` of `None` means
monostatic: zero baseline, nothing raises. Consumers take one type with no `bistatic_*` twins, and
a scenario file selects siting by the presence of a receiver site table.

## References

- EUROCONTROL, *ASTERIX Category 048: Monoradar Target Reports*, and *Category 062: SDPS Track
  Messages*. https://www.eurocontrol.int/asterix
- Y. Shafranovich, *Common Format and MIME Type for CSV Files*, RFC 4180, 2005.
- ISO 8601-1:2019, *Date and time: representations for information interchange*.
- M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014 (the radar
  data cube and its slow-time and fast-time axes).
- Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and
  Navigation*, Wiley, 2001, §5.4 (NIS and NEES).
- `docs/conventions/style.md` §2, §3.1; `docs/conventions/testing.md`.

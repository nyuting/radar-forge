# Data 001 — Files a run writes

> Status: schema version **1.2.0**. This is the contract for what `scripts/run_scenario.py` writes
> into a run directory, plus the sign change of §5, which lands with the code PRs that rename the
> range-rate quantities (the writer is closing-positive until then). The column lists are the
> tuples `DETECTION_COLUMNS`, `TRACK_COLUMNS` and `METRIC_COLUMNS` in that script.

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
| Scenario, radar and waveform inputs | TOML | The project's configuration format, read with stdlib `tomllib` by `pipelines/scenarios.py` |
| Tables: trajectory, truth, detections, tracks, metrics | CSV (RFC 4180, UTF-8), one row per entity per frame | Opens in every tool including a spreadsheet; no dependency |
| Arrays: raw IQ | NumPy `.npz`, one file per frame | The core stack reads and writes it; no dependency |
| Run metadata and provenance | JSON | Machine output, not configuration |
| Figures | PNG, MP4 | For people, never parsed back |

## 4. The run directory

```text
out/<run>/
├── metadata.json          # §6.1 — read this first
├── truth.csv              # §6.4
├── detections.csv         # §6.5 (tracking runs)
├── tracks.csv             # §6.6 (tracking runs)
├── metrics.csv            # §6.9 (tracking runs)
├── iq_00000.npz …         # §6.7, one per frame; --no-iq skips them
├── rd_00000.png …         # §6.8
└── track_00000.png …
```

## 5. Conventions every file shares

| Topic | Convention |
| :--- | :--- |
| **Time** | `time_s` is seconds since the trajectory's first fix, float64, monotonic within a file. `epoch_utc` in `metadata.json` is the absolute time of that fix, or `null` for a synthetic trajectory. When it is not null, each table also carries `time_utc = epoch_utc + time_s` (ISO 8601, microseconds, `Z`) |
| **Frame** | `frame` is a zero-based integer, one revisit (a *scan* to the tracker). With `time_s` it is the join key across files. A *burst* is one coherent processing interval inside a frame |
| **Frames of reference** | Geodetic WGS-84 → ECEF → local ENU tangent plane, per `core/geodesy.py`. The ENU origin is the radar, or the receiver of a bistatic pair; `metadata.json` `reference_site` names it |
| **Angles** | `azimuth_deg` is a compass bearing from true north, increasing clockwise; `elevation_deg` is up from the local horizontal. Radians inside `core/`, degrees in files |
| **Range and range rate** | **Range rate is the time derivative of range, positive receding; Doppler = -rate/wavelength.** `range_m` is the slant range for a monostatic radar and the bistatic range `R_T + R_R` (`bistatic_range_m`) for a bistatic one, and its rate is `bistatic_range_rate_mps = d(bistatic_range_m)/dt`, so `doppler_hz = -bistatic_range_rate_mps / wavelength_m`. The range-Doppler axes carry bistatic range and rate; a monostatic display halves both. The columns named `radial_velocity_mps`, `velocity_*_mps` and `range_rate_mps` all follow it |
| **Units** | The `style.md` §2 suffix table. Compound units: `_m2`, `_m2ps2`, `_mps2` |
| **Identifiers** | `track_id` is a string and never reused. `detection_id` is an integer unique within a frame, so `(frame, detection_id)` is the detection key |
| **Vocabulary** | A *detection* is the CFAR-derived point: a connected group of threshold crossings in one burst's range-Doppler map (`core.detection.Detection`); `detections.csv` holds them in physical units. A *track* is a filtered state built from detections over scans |
| **Missing values** | An empty cell means not applicable or not yet known (`velocity_unfolded_mps` before a fold is selected). `NaN` means computed and invalid. They are never interchanged |
| **Booleans** | `0` / `1` |
| **Enumerations** | Lowercase ASCII: `tentative`, `confirmed`, `coasting`, `deleted`; `accepted`, `missing_pair`, `ambiguous_pair`, `unresolved_velocity` |
| **Array axis order** | The library's in-memory order from `style.md` §3.1: slow time before fast time, receive channels last |

## 6. File specifications

Columns are written in the order listed; readers select them by name. The "Req." column says when
a column is present: **R** always, **B** bistatic runs only, **T** tracking runs only, **E** when
`epoch_utc` is not null, **O** optional.

### 6.1 `metadata.json`

The run manifest, written by `write_metadata` in `scripts/run_scenario.py`.

| Key | Type | Content |
| :--- | :--- | :--- |
| `schema_version` | string | Semver of this document |
| `run_id` | string | `<scenario-name>-<created_utc as YYYYMMDDTHHMMSSZ>` |
| `created_utc`, `epoch_utc` | string, string or null | ISO 8601 UTC instants (§5 Time) |
| `reference_site` | object | `latitude_deg`, `longitude_deg`, `altitude_m`, `role` (`"monostatic"` or `"receiver"`) |
| `conventions` | object | Fixed strings restating §5 so a reader can check them: `frame`, `azimuth_reference`, `velocity_sign`, `cube_layout`, `rd_layout` |
| `scenario` | object | Name, description, seed, `start_time_s`, `duration_s`, `frame_rate_hz`, `n_frames` (written, which `--frames` can make smaller than the scenario's), `trajectory_path`, `target_altitude_m` |
| `sites`, `target` | object | `siting` (`"monostatic"` or `"bistatic"`) with the site or sites and `baseline_m`; the target's RCS model |
| `bursts` | array | One object per burst: `waveform`, `carrier_frequency_hz`, `bandwidth_hz`, `chirp_duration_s`, `prf_hz`, `sample_rate_hz`, `n_pulses`, `n_samples`, resolutions, unambiguous range and velocity, `noise_power_w` (thermal noise per complex sample) |
| `detection` | object or null | The resolved `[detection]` settings |
| `tracking` | object or null | The resolved `[tracking]` settings, defaults included, plus `simulated_angles`, `state_fields` (the ordered state names, indexing the `cov_<i>_<j>` columns), `range_period_m` (the period when range is measured modulo one, else `null`) and `unused` (settings the chosen estimator did not read) |
| `provenance` | object | `radar_forge_version`, `git_commit`, `python_version`, `numpy_version`, `command` (the argv) |

### 6.2 Scenario input

The scenario that was run is `scenarios/<name>.toml`. Its schema is whatever `pipelines/scenarios.py`
parses; this document does not restate it, because that loader is the single definition of what a
scenario file may contain.

### 6.3 Trajectory input

A CSV, one row per fix, read by `load_flight_csv` in `pipelines/trajectories.py`. Columns:
`time_s` (s, from the first fix), `latitude_deg`, `longitude_deg` (WGS-84), and optionally
`altitude_m` (above the ellipsoid) and `target_id`. It rejects the old `timestamp,lat,lon` header
with a message naming the new columns.

### 6.4 `truth.csv`

One row per frame: the true, unfolded geometry in every scenario variant.

| Column | dtype | Unit | Req. | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `frame` | int | — | R | |
| `time_s` | float64 | s | R | |
| `range_m` | float64 | m | R | Slant range; the bistatic range when bistatic |
| `radial_velocity_mps` | float64 | m/s | R | Time derivative of `range_m`, positive receding |
| `azimuth_deg` | float64 | deg | R | True-north clockwise, from the reference site |
| `elevation_deg` | float64 | deg | R | |
| `range_tx_m` | float64 | m | B | Transmitter to target |
| `range_rx_m` | float64 | m | B | Target to receiver |
| `bistatic_angle_deg` | float64 | deg | B | Angle at the target between transmitter and receiver |

### 6.5 `detections.csv`

One row per detection per frame. The vocabulary follows the ASTERIX CAT048 target report; the
units are SI.

| Column | dtype | Unit | Req. | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `frame`, `time_s` | int, float64 | —, s | R | Every detection in a frame shares `time_s` |
| `time_utc` | string | ISO 8601 | E | |
| `detection_id` | int | — | R | Unique within the frame |
| `sensor_id` | string | — | R | `"rx0"` |
| `range_m` | float64 | m | R | Power-weighted centroid range, folded if the burst folds |
| `velocity_folded_mps` | float64 | m/s | R | Range rate as read off the map, inside the unambiguous interval, positive receding |
| `velocity_unfolded_mps` | float64 | m/s | R | Empty until a fold is selected |
| `fold_index` | int | — | R | Empty until a fold is selected |
| `range_index`, `velocity_index` | float64 | bins | R | Fractional centroid cell indices into the burst's range-Doppler map |
| `peak_power_w`, `total_power_w` | float64 | W | R | Peak cell and cluster sum |
| `snr_db` | float64 | dB | R | `10 log10(peak_power_w / noise_power_w)` against the burst's thermal noise per sample, not the CFAR floor |
| `n_cells` | int | — | R | Cells in the cluster |
| `associated_track_id` | string | — | T | Empty for a false alarm or a track seed |
| `burst_index` | int | — | R | The burst whose map the detection came from |
| `status` | enum | — | R | `accepted` (passed to the tracker) or, for a dual-PRF pair, `missing_pair`, `ambiguous_pair`, `unresolved_velocity` |
| `pair_id` | int | — | R | The same integer on both detections of a dual-PRF pair; empty otherwise |
| `range_unfolded_m` | float64 | m | R | The true range an `accepted` pair agrees on (`core.ambiguity_resolution.unfold_range_dual_prf`); empty otherwise |

### 6.6 `tracks.csv`

One row per track per frame, tentative and coasting tracks included. The vocabulary follows the
ASTERIX CAT062 system track. The state is range and range rate.

| Column | dtype | Unit | Req. | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `frame`, `time_s` | int, float64 | —, s | R | |
| `time_utc` | string | ISO 8601 | E | |
| `track_id` | string | — | R | |
| `status` | enum | — | R | `tentative`, `confirmed`, `coasting` or `deleted` |
| `state_model` | string | — | R | `range_1d` |
| `measurement_dim` | int | — | R | 1 until the track can unfold its Doppler, then 2 |
| `n_hits`, `n_misses_in_a_row` | int | — | R | |
| `range_m`, `range_rate_mps` | float64 | m, m/s | R | The state; the rate is positive receding |
| `var_range_m2`, `var_range_rate_m2ps2` | float64 | m², m²/s² | R | Covariance diagonal |
| `cov_0_1` | float64 | m²/s | R | Covariance off-diagonal, indexed by `tracking.state_fields` |
| `nis` | float64 | — | R | Normalised innovation squared of the last update; empty on a miss |
| `associated` | 0/1 | — | R | Updated this frame |
| `associated_detection_id` | int | — | R | Key back into `detections.csv`; empty on a miss. On a track's first row it names the detection it was born from, with `associated` 0 |

### 6.7 `iq_NNNNN.npz`

One compressed NumPy archive per frame (`np.savez_compressed`), with one array per burst named
`burst0`, `burst1`, … Each array is `(n_pulses, n_samples)` complex128 in `sqrt(W)`, so `|a|²` is
watts; the bursts of a dual-PRF run differ in shape. The archive carries no axes or units: those
come from `metadata.json` `bursts` and `conventions.cube_layout`. Raw IQ is by far the largest
output and is regenerable from the scenario file and seed.

### 6.8 Figures

`rd_NNNNN.png`, `track_NNNNN.png` and an `.mp4` assembled from them, in the run root. They are
decibel renderings for people, and nothing reads them back.

### 6.9 `metrics.csv`

Tidy and long, one row per metric per scope. The unit is in the metric's name, so there is no
`unit` column to disagree with it.

| Column | dtype | Req. | Meaning |
| :--- | :--- | :--- | :--- |
| `metric` | string | R | `range_rmse_m`, `range_rate_rmse_mps`, `mean_nis_dim2`, `track_breaks`. NIS is averaged per measurement dimension `k` because its expected value is `k` |
| `track_id`, `target_id` | string | O | Empty for a run-level metric |
| `frame_start`, `frame_end` | int | R | The inclusive frame window the metric covers |
| `value` | float64 | R | |

## 7. Versioning

- `schema_version` is semver and appears in `metadata.json`.
- **Minor**: add a column or key. **Major**: rename, remove, re-type, re-unit or re-sign anything.
- Readers select columns by name, ignore unknown ones, and refuse a major version they do not know.

| Version | Change |
| :--- | :--- |
| 1.0.0 | First proposal |
| 1.1.0 | `detections.csv` adds `burst_index`, `status`, `pair_id`; `metadata.json` `tracking` adds `range_period_m`, `unused` |
| 1.2.0 | `detections.csv` adds `range_unfolded_m` |
| 2.0.0 | Pending: range rate is positive receding in every file (§5); `conventions.velocity_sign` becomes `"receding_positive"` |

## 8. Decisions

Moved here from the retired structure spec; each binds the code.

### D1 — The path set is the backend contract

A propagation backend returns `PropagationPaths` (delay, Doppler, complex amplitude, angles of
arrival and departure, bounce count), never baseband. `core/signal.py` turns any path set into
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

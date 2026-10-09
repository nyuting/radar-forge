# Data 001 — File formats for radar inputs, outputs and products

> **Reading order.** §1–§5 are the contract every file shares: what goes in which format, where it
> lives, and the conventions (time, frames, signs, units) that make one file readable next to
> another. §6 specifies each file. §7 tells a collaborator in another ecosystem how to read them.
> §8–§12 are how the contract changes, how it is checked, and how the repository gets from today's
> outputs to it.
>
> | § | Section |
> | :--- | :--- |
> | [1](#1-status-purpose-and-scope) | Status, purpose and scope |
> | [2](#2-principles) | Principles |
> | [3](#3-which-format-for-which-data) | Which format for which data |
> | [4](#4-product-levels-and-the-run-directory) | Product levels and the run directory |
> | [5](#5-conventions-every-file-shares) | Conventions every file shares |
> | [6](#6-file-specifications) | File specifications |
> | [7](#7-reading-and-converting) | Reading and converting |
> | [8](#8-versioning-and-compatibility) | Versioning and compatibility |
> | [9](#9-enforcement) | Enforcement |
> | [10](#10-decisions) | Decisions |
> | [11](#11-acceptance-criteria) | Acceptance criteria |
> | [12](#12-migration-and-follow-ups) | Migration and follow-ups |
> | [—](#references) | References |

---

## 1. Status, purpose and scope

Status: **proposed**. Schema version **1.1.0**. `scripts/run_scenario.py` writes `metadata.json`
(§6.1, all but `files`), `detections.csv` (§6.5), `tracks.csv` (§6.6, for the `range_1d` state
model) and `metrics.csv` (§6.9). Nothing else below is implemented yet except where §12 says
today's writer already matches.

Today each scenario specification defines its own output files:
`spec/scenario-001-xband.md` §3.6, `spec/scenario-002-bistatic.md` §3.6 and
`spec/scenario-003-tracking.md` §9. `scripts/run_scenario.py` writes them. That was the right
call while there was one consumer. Now there are four, and they need one contract:

| Collaborator | Reads with | Needs |
| :--- | :--- | :--- |
| Python users | NumPy, pandas, h5py, xarray | Named axes and units, without reverse-engineering the writer |
| MATLAB users | `h5read`, `readtable`, `jsondecode` | Built-in readers only, no toolbox |
| C++ / SDR / hardware users | HDF5 C API or HighFive, SigMF tooling | A binary layout that is written down, and recorded IQ in a format their tools already emit |
| Tracking researchers | Stone Soup | CSV columns that Stone Soup's generic CSV readers load with no conversion script |

This document fixes, for every product in the processing chain from scenario inputs to tracks:
- the file format
- the file name and where it sits in a run directory
- the column or dataset names, dtypes, shapes and units
- the conventions that make two files agree with each other.

Once implemented, it supersedes the output sections of the three scenario specs. They keep their
column lists as history, and §12 records every change.

**In scope:** scenario inputs, trajectories, raw IQ, processed arrays (range × slow time,
range-Doppler, CFAR thresholds), detections, tracks, truth, metrics, run metadata and the COCO
export's place in the layout.

**Out of scope:**
- **Real-time streaming** and network transport.
- **The ASTERIX binary encoding.** §6 borrows ASTERIX's vocabulary, not its bytes.
- **Physics.** How the numbers are produced is the scenario specs' business, not this one's.
- **ML dataset tensors.** `pipelines/datasets.py` is still design only. When it lands it reads
  these files; it does not define new ones.

## 2. Principles

1. **Every file describes itself.** A file separated from its run directory still says what it is:
   CSV headers carry units in the column names, HDF5 files carry attributes, and every file names
   its schema version (§8).
2. **One format per data shape.** Tables are CSV, arrays are HDF5, configuration is TOML and run
   metadata is JSON. Nobody has to ask "which format is this one in?"
3. **Units in every name.** `docs/conventions/style.md` §2 applies to column and dataset names
   exactly as it applies to parameters: `range_m`, `peak_power_w`, `azimuth_deg`.
4. **Linear on disk.** Stored quantities are linear SI. A `_db` column may *accompany* its linear
   source as a convenience (`snr_db` beside `peak_power_w`), but it never replaces it. Decibel-only
   data exists only in exports meant for eyes or for other toolchains: PNG, MP4 and COCO.
5. **Readable without radar-forge.** Every file opens with a stock reader in each of the four
   ecosystems in §1. §7 gives the snippet and §11 tests it.
6. **The contract is data, not prose.** Column lists and dataset layouts are written once, in a
   registry the writers and the validator share (§9). This document is the explanation; the
   registry is the source of truth.

## 3. Which format for which data

| Data | Canonical format | Why | Conversions |
| :--- | :--- | :--- | :--- |
| Scenario, radar and waveform **inputs** | **TOML** | Already the project's configuration format (`CLAUDE.md`), read with stdlib `tomllib` by `pipelines/scenarios.py` | — |
| **Tables**: input trajectories, truth, detections, tracks, metrics | **CSV**: RFC 4180, UTF-8, tidy long format with one row per entity per frame | Opens in every tool, including a spreadsheet. Stone Soup's generic CSV readers load it directly. Needs no dependency | → Parquet mirror for large runs (§7.6); → Stone Soup readers (§7.4) |
| **Arrays**: raw IQ, range × slow time, range-Doppler, CFAR thresholds, later range-Doppler-azimuth | **HDF5**, one file per product level per run, frames on the leading axis | Self-describing (attributes, dimension scales), with chunked partial reads. Native in MATLAB, C++ and Python. MATLAB's own `.mat` v7.3 is HDF5 underneath | ↔ SigMF (§7.5); → xarray (§7.1); legacy `.npz` → HDF5 (§12) |
| **Recorded hardware IQ**, as interchange | **SigMF** v1.2: a `.sigmf-meta` JSON file plus raw `.sigmf-data` | The open standard SDR tools already write. Reading it needs only JSON and `numpy.fromfile` | SigMF ↔ `iq.h5` (§7.5) |
| **Run metadata and provenance** | **JSON** (`metadata.json`) | It is machine output, not configuration. Readable by `json`, MATLAB `jsondecode` and nlohmann/json. The same format SigMF uses for its metadata | — |
| **ML annotation export** | **COCO JSON** | `spec/structure.md` D5, unchanged | — |
| **Figures** | PNG, MP4 (GIF fallback) | For people, never parsed back | — |

**Rejected alternatives.** Each is a reasonable choice somewhere; none of them is right here.

| Alternative | Why not the canonical format |
| :--- | :--- |
| `.npz` (today's IQ format) | No attributes, no axis names and no units. No MATLAB or C++ reader without a custom parser. Every reader has to be told out of band that axis 0 is slow time |
| `.mat` | Proprietary for v5–v7. v7.3 *is* HDF5, so writing HDF5 directly gives MATLAB users the same file without the MATLAB-specific header |
| Parquet as canonical | Needs `pyarrow`, a heavy dependency. Opens awkwardly in Excel and older MATLAB. Kept as an optional mirror with identical columns |
| Zarr | Excellent for cloud-chunked arrays, but MATLAB and C++ support is still immature. Revisit if runs outgrow a single machine |
| Strict NetCDF-4 / CF compliance | CF has no complex data type, and every product before detection is complex. We stay *CF-aligned* (`units` and `long_name` attributes, dimension scales), which keeps the files legible to NetCDF-aware tools without pretending to comply |
| CF-Radial 2 / ODIM_H5 | Built for weather radar: they are organised around sweeps and moments (reflectivity, velocity, spectrum width), not around raw IQ cubes, detections and tracks. Borrowed for their product-level idea (§4), not their layout |
| JSON Lines for detections and tracks | Nested records are a better fit for variable-length fields, but MATLAB and spreadsheet users lose. Nothing in §6 is nested enough to need it |

**Dependency note** (`CLAUDE.md` requires one). `h5py` goes in a new **`io` extra**, never in core.
It buys attributes, labelled axes, chunked partial reads and a file MATLAB and C++ read natively.
The NumPy/SciPy stack offers none of these: `.npz` has no metadata, and `scipy.io.savemat` writes
only the v5 format. CSV and JSON stay stdlib-only, so a core-only install still writes every
product except arrays. It raises the descriptive `ImportError` of `style.md` §6 if asked for one.

## 4. Product levels and the run directory

The levels follow the data-level idea that weather-radar formats (ODIM_H5, CF-Radial) and satellite
missions use: each level is produced from the one before it, and a level can be regenerated from
its predecessor plus the metadata.

| Level | Product | File | Produced by (today) |
| :--- | :--- | :--- | :--- |
| L0 | Scenario and radar configuration | `inputs/scenario.toml` | the user |
| L0 | Target trajectories | `inputs/trajectory.csv` | the user, or `scripts/translate_flight_coordinates.py` |
| L1 | Raw baseband IQ, `data_ftst` | `iq.h5` | `pipelines/scenarios.py` (`Frame.iq`) |
| L2 | Processed arrays: `data_rfst`, `data_rv`, CFAR threshold, noise | `products.h5` | `pipelines/scenarios.py` (`RangeDopplerProduct`), `core/detection.py` |
| L3 | Detections (plots) | `detections.csv` | `pipelines/tracking.py` (`Measurement`) |
| L4 | Tracks | `tracks.csv` | `core/tracking/` via `pipelines/tracking.py` (`FrameTracks`) |
| — | Ground truth | `truth.csv` | `pipelines/trajectories.py` (`TargetTrack`, `BistaticTargetTrack`) |
| — | Performance metrics | `metrics.csv` | acceptance tests, later a metrics module |
| — | Exports | `exports/coco.json` | `pipelines/exporters/` (design only) |
| — | Figures | `figures/*.png`, `figures/*.mp4` | `viz/scopes/` |

```text
out/<run_id>/
├── metadata.json          # §6.1 — read this first; it names every other file
├── inputs/
│   ├── scenario.toml      # §6.2 — byte copy of the scenario file that was run
│   └── trajectory.csv     # §6.3 — the trajectory it read
├── iq.h5                  # §6.7 — L1, the large one; safe to delete, regenerable from L0 + seed
├── products.h5            # §6.8 — L2
├── detections.csv         # §6.5 — L3
├── tracks.csv             # §6.6 — L4
├── truth.csv              # §6.4
├── metrics.csv            # §6.9
├── exports/coco.json      # §6.10
└── figures/
    ├── rd_00000.png
    ├── track_00000.png
    └── rd.mp4
```

L1 and L2 are separate files on purpose. Raw IQ is typically two orders of magnitude larger than
everything else together, and the common operation on a finished run is to delete it and keep the
products. `inputs/` holds copies, not links, so a run directory is still complete after the
original scenario file has been edited.

## 5. Conventions every file shares

| Topic | Convention |
| :--- | :--- |
| **Time** | `time_s` is seconds since `epoch_utc`: float64, monotonic within a file. `epoch_utc` is an ISO 8601 UTC instant in `metadata.json` (e.g. `2026-09-03T00:17:56Z`). It is the absolute time of the trajectory's first fix (`Trajectory.epoch`), or `null` for a synthetic trajectory. When the epoch is not null, every table also carries `time_utc = epoch_utc + time_s` as ISO 8601 with microseconds and a `Z` suffix |
| **Frame index** | `frame` is a zero-based integer. Together with `time_s` it is the join key across every file of a run |
| **Coordinate frames** | Geodetic WGS-84 (`latitude_deg`, `longitude_deg`, `altitude_m` above the **ellipsoid**) → ECEF → local **ENU** tangent plane at the reference site, per `core/geodesy.py`. The reference site is the receiver for bistatic runs, and `metadata.json` names it (§6.1). ENU columns are `east_m`, `north_m`, `up_m` |
| **Angles** | `azimuth_deg` is measured from **true north, increasing clockwise** (a compass bearing). `elevation_deg` is measured up from the local horizontal. Both are seen from the reference site. Radians inside `core/`, degrees in every file (DF6). The name `azimuth_boresight_deg` is reserved for antenna-frame angles when `array/` lands (DF2) |
| **Velocity sign** | **Closing is positive, in every file**, per `spec/structure.md` D5 and `core/tracking/`. That covers `radial_velocity_mps` (truth), `velocity_*_mps` (detections, the RD axis) and `range_rate_mps` (tracks). Stone Soup and ASTERIX use opening-positive range rate, so readers negate at that boundary (§7.4, DF7). ENU rates (`east_rate_mps`, …) are ordinary time derivatives, with no sign convention to choose |
| **Bistatic reading** | `range_m` is the bistatic mean range `(R_t + R_r)/2`, and `radial_velocity_mps` is the bisector rate, per `spec/structure.md` D6. The individual legs appear only in the columns that name them |
| **Units** | The `style.md` §2 suffix table, applied to every column and dataset name. Compound units follow the pattern already in use: `_m2` (m²), `_m2ps2` (m²/s²), `_mps2` (m/s²). An off-diagonal covariance column takes its unit from `metadata.json` (§6.6) |
| **Identifiers** | `target_id`, `track_id`, `sensor_id`: strings or integers, stable for the whole run, never reused. `detection_id` is an integer that is unique **within a frame**, so `(frame, detection_id)` is the detection key |
| **Missing values** | An **empty cell** means *not applicable or not yet known*: `velocity_unfolded_mps` before a track can unfold, or `associated_track_id` for a false alarm. `NaN` means *computed and invalid*. The two are never interchanged. In HDF5, an absent dataset means not applicable |
| **Booleans** | `0` / `1` in CSV (it is what the current writer emits, and every reader parses it). `uint8` in HDF5 attributes |
| **Enumerations** | Lowercase ASCII strings: `tentative`, `confirmed`, `coasting`, `deleted`; `fmcw`, `pulsed`; `range_1d`, `enu_2d`, `enu_3d`; `accepted`, `missing_pair`, `ambiguous_pair`, `unresolved_velocity` |
| **Numeric precision** | Floats are written with the shortest representation that round-trips exactly: Python's `repr` / `str`, or C's `%.17g`. Never a fixed `%.3f` (DF9). Arrays are `float64` / `complex128`, except recorded hardware IQ (§6.7) |
| **Array axis order** | The library's in-memory order from `style.md` §3.1: slow time before fast time, receive channels last, with the frame axis added in front. Axes are named in the file, so no reader has to guess (DF1) |

## 6. File specifications

The "Req." column says when a column must be present: **R** always, **B** bistatic runs only,
**T** tracking runs only, **E** when `epoch_utc` is not null, **O** optional. Writers emit columns in
the order listed. Readers select columns by name (§8).

### 6.1 `metadata.json`

The run manifest. It keeps today's top-level keys (`scenario`, `sites`, `target`, `bursts`,
`tracking`, `detection`, `provenance`, from `scripts/run_scenario.py` `write_metadata`) and adds
the ones marked *new*.

| Key | Type | Content |
| :--- | :--- | :--- |
| `schema_version` *new* | string | Semver of this spec, `"1.1.0"` |
| `run_id` *new* | string | `<scenario-name>-<created_utc as YYYYMMDDTHHMMSSZ>` unless the caller overrides it. The run directory is named after it |
| `created_utc` *new* | string | ISO 8601 UTC instant the run started |
| `epoch_utc` *new* | string or null | §5 Time |
| `reference_site` *new* | object | `latitude_deg`, `longitude_deg`, `altitude_m`, `role` (`"monostatic"` or `"receiver"`). The ENU origin |
| `conventions` *new* | object | Fixed strings restating §5, so a reader can check rather than assume: `frame: "enu"`, `azimuth_reference: "true_north_clockwise"`, `velocity_sign: "closing_positive"`, `cube_layout: "frame,pulse,sample[,rx]"`, `rd_layout: "frame,doppler,range"` |
| `scenario` | object | As today: name, description, seed, `start_time_s`, `duration_s`, `frame_rate_hz`, `n_frames`, `trajectory_path`, `target_altitude_m` |
| `sites`, `target` | object | As today |
| `bursts` | array | As today, one object per burst: waveform, `f0_hz`, `bandwidth_hz`, `prf_hz`, `n_pulses`, `n_samples`, resolutions, unambiguous range and velocity, `noise_power_w` |
| `detection` | object or null | As today: the resolved `[detection]` table |
| `tracking` | object or null | As today, including `simulated_angles` (`spec/scenario-003-tracking.md` §6.3). **Adds** `state_model` and `state_fields`: the ordered list of state-vector column names, e.g. `["east_m", "north_m", "east_rate_mps", "north_rate_mps"]`. §6.6's covariance columns are indexed by it. **Adds** `range_period_m`: when the tracker measures range modulo a period, that period, and every `range_m` in `detections.csv` and `tracks.csv` lies in `[0, range_period_m)`. `null` when range is absolute. **Adds** `unused`: the names of the settings, among those recorded, that the chosen `estimator` did not read, so that no recorded value is mistaken for one the run used |
| `provenance` | object | As today, `radar_forge_version` and `git_commit`. **Adds** `python_version`, `numpy_version`, and `command` (the argv that produced the run) |
| `files` *new* | array | One object per file written: `path` (relative), `level` (`L0`–`L4`, `truth`, `metrics`, `export`, `figure`), `format`, `sha256`, `size_bytes`. The inventory is what lets the validator tell a truncated run from a complete one |

A JSON Schema for this object is a follow-up (§12). Until it exists, this table is the contract.

### 6.2 `inputs/scenario.toml`

A byte-for-byte copy of the scenario file that was run. Its schema is the one `pipelines/scenarios.py`
parses. This spec does not restate it, because that loader is the single definition of what a
scenario file may contain. Collaborators supplying hardware parameters write a scenario file.

### 6.3 `inputs/trajectory.csv`

One row per fix, per target.

| Column | dtype | Unit | Req. | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `time_utc` | string | ISO 8601 UTC | R* | Absolute time of the fix |
| `time_s` | float64 | s | R* | Relative time, for synthetic trajectories with no absolute epoch |
| `target_id` | string | — | O | Defaults to `"0"` when absent; required once a scenario has two targets |
| `latitude_deg` | float64 | deg | R | WGS-84 geodetic latitude |
| `longitude_deg` | float64 | deg | R | WGS-84 geodetic longitude |
| `altitude_m` | float64 | m | O | Height above the ellipsoid. When absent, the scenario's `target_altitude_m` applies, as today |

\* Exactly one of `time_utc` and `time_s`.

`data/flight_coordinates.csv` and its excerpt `tests/data/golden/flight_coordinates_head.csv` use
these names, with `time_s` measured from the first fix, so the shipped track has no epoch. The
recorded track's first fix was at `2026-09-03T00:17:56Z`. `load_flight_csv`
(`pipelines/trajectories.py`) reads exactly this table. It rejects the old `timestamp,lat,lon`
header with a message naming the new columns, and it rejects a file whose `target_id` names more
than one target until multi-target scenarios exist.

### 6.4 `truth.csv`

One row per target per frame. It carries the **true, unfolded** quantities in every variant, as
scenario 001 §3.6 established. It is a superset of today's columns, so the D5 COCO exporter still
consumes it unchanged.

| Column | dtype | Unit | Req. | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `frame` | int | — | R | |
| `time_s` | float64 | s | R | |
| `time_utc` | string | ISO 8601 | E | |
| `target_id` | string | — | R | *new* |
| `range_m` | float64 | m | R | Slant range; the bistatic mean range when bistatic |
| `radial_velocity_mps` | float64 | m/s | R | Closing-positive; the bisector rate when bistatic |
| `azimuth_deg` | float64 | deg | R | True-north clockwise, from the reference site |
| `elevation_deg` | float64 | deg | R | |
| `latitude_deg`, `longitude_deg`, `altitude_m` | float64 | deg, deg, m | R | *new*. Target position, WGS-84 |
| `east_m`, `north_m`, `up_m` | float64 | m | R | *new*. Target position in the ENU frame |
| `range_tx_m` | float64 | m | B | Transmitter to target |
| `range_rx_m` | float64 | m | B | Target to receiver |
| `bistatic_angle_deg` | float64 | deg | B | β at the target |

Example, monostatic, two frames (shown with rounded values for legibility; real files carry full
precision per §5):

```csv
frame,time_s,time_utc,target_id,range_m,radial_velocity_mps,azimuth_deg,elevation_deg,latitude_deg,longitude_deg,altitude_m,east_m,north_m,up_m
0,0.0,2026-09-03T00:17:56.000000Z,0,5326.1,62.4,301.2,16.3,35.90608,-79.0945,1500.0,-4378.0,2653.9,1497.7
1,1.0,2026-09-03T00:17:57.000000Z,0,5263.8,62.3,301.0,16.5,35.90592,-79.09474,1500.0,-4399.6,2636.1,1497.7
```

### 6.5 `detections.csv`

One row per detection (CFAR cluster) per frame. The names are today's, from scenario 003 §9, plus
the additions marked *new*. The vocabulary follows the ASTERIX CAT048 target report: a measured
polar position, a Doppler speed and an amplitude. The units are SI, not ASTERIX's nautical miles.

| Column | dtype | Unit | Req. | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `frame` | int | — | R | |
| `time_s` | float64 | s | R | Frame time. Every detection in a frame shares it |
| `time_utc` | string | ISO 8601 | E | *new* |
| `detection_id` | int | — | R | *new*. Unique within the frame. `(frame, detection_id)` is the key |
| `sensor_id` | string | — | R | *new*. `"rx0"` for today's single receiver. Present now so multi-sensor fusion needs no schema change |
| `range_m` | float64 | m | R | Power-weighted centroid range |
| `velocity_folded_mps` | float64 | m/s | R | As read off the RD map, inside ±v_unamb, closing-positive |
| `velocity_unfolded_mps` | float64 | m/s | R | Empty until a fold is selected |
| `fold_index` | int | — | R | Empty until a fold is selected |
| `azimuth_deg`, `elevation_deg` | float64 | deg | O | *new*. Empty unless angles were measured or simulated. `metadata.json` `tracking.simulated_angles` says which |
| `range_index`, `velocity_index` | float64 | bins | R | *new*. Fractional centroid cell indices into `products.h5` `/rd` (range, Doppler) |
| `peak_power_w` | float64 | W | R | |
| `total_power_w` | float64 | W | R | *new*. Already on `Measurement`; now written |
| `snr_db` | float64 | dB | O | *new*. `10 log10(peak_power_w / noise_power_w)` using the burst's `noise_power_w`. A convenience beside its linear source (§2.4) |
| `n_cells` | int | — | R | Cells in the cluster |
| `associated_track_id` | string | — | T | Empty for an unassociated detection (a false alarm or a new track seed) |
| `burst_index` | int | — | R | *new*. The burst whose map the detection was found in, `0` for a single-burst run. `range_index` and `velocity_index` index that burst's map |
| `status` | enum | — | R | *new*. What became of the detection on its way to the tracker: `accepted` (passed to it); or, for a dual-PRF pair, `missing_pair` (no detection in the other burst within the range tolerance), `ambiguous_pair` (one within it, but not mutually nearest) or `unresolved_velocity` (paired, but the two folded velocities agree on no unfolded one). Only the first burst's `accepted` detections reach the tracker |
| `pair_id` | int | — | R | *new*. The same integer on the two detections of a dual-PRF pair, unique within the frame. Empty for a detection that was not paired |

### 6.6 `tracks.csv`

One row per track per frame, including tentative and coasting tracks. The vocabulary follows the
ASTERIX CAT062 system track: position, velocity, status and estimated accuracies. Its content
closes the gap scenario 003 §9 left open. That spec promised the ENU state columns, but
`scripts/run_scenario.py` still writes only `state[0]`, `state[1]` and two variances (DF3).

| Column | dtype | Unit | Req. | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `frame`, `time_s` | int, float64 | —, s | R | |
| `time_utc` | string | ISO 8601 | E | *new* |
| `track_id` | string | — | R | |
| `status` | enum | — | R | `tentative`, `confirmed`, `coasting` or `deleted` |
| `state_model` | enum | — | R | *new*. `range_1d`, `enu_2d` or `enu_3d` |
| `measurement_dim` | int | — | R | Scenario 003 §5.3 bootstrap |
| `n_hits`, `n_misses_in_a_row` | int | — | R | *new*. Already on `Track`; now written |
| `range_m` | float64 | m | R | For ENU models, computed from the state, so every run has it (scenario 003 §9) |
| `range_rate_mps` | float64 | m/s | R | **Closing-positive**, as for `range_m` |
| `var_range_m2`, `var_range_rate_m2ps2` | float64 | m², m²/s² | R | For ENU models, propagated through the polar Jacobian |
| `east_m`, `north_m`, `up_m` | float64 | m | O | *new*. ENU state position. `up_m` for `enu_3d` only |
| `east_rate_mps`, `north_rate_mps`, `up_rate_mps` | float64 | m/s | O | *new*. ENU state velocity |
| `var_<field>` | float64 | unit² | O | *new*. One per state field, e.g. `var_east_m2`, `var_north_rate_m2ps2` |
| `cov_<i>_<j>` | float64 | unit_i × unit_j | O | *new*. Upper-triangle off-diagonals, `i < j`, indexed by position in `metadata.json` `tracking.state_fields`. With the `var_` columns, the **full covariance** is on disk |
| `latitude_deg`, `longitude_deg`, `altitude_m` | float64 | deg, deg, m | O | *new*. Derived for ENU models, for geodetic consumers |
| `nis` | float64 | — | R | Normalised innovation squared of the last update. Empty on a miss |
| `associated` | 0/1 | — | R | Updated this frame |
| `associated_detection_id` | int | — | R | *new*. Empty on a miss. The key back into `detections.csv`. On a track's first row it names the detection the track was born from, with `associated` 0, since a seed is not an update |

For `range_1d`, `state_fields` is `["range_m", "range_rate_mps"]`, and the state *is* the
`range_m` / `range_rate_mps` columns. There is one off-diagonal, `cov_0_1`, in m²/s. The covariance
is written in the same closing-positive sign as the state it describes.

### 6.7 `iq.h5` — L1 raw baseband

```text
/                                   attrs: schema_version, conventions="radar-forge-run", run_id,
│                                          epoch_utc, created_utc, radar_forge_version, git_commit,
│                                          source ("simulated" | "recorded")
└── iq/
    ├── burst0/                     attrs: waveform, f0_hz, bandwidth_hz, prf_hz, sample_rate_hz,
    │   │                                  chirp_duration_s, n_pulses, n_samples
    │   ├── data                    (n_frames, n_pulses, n_samples[, n_rx])  complex128
    │   ├── frame                   (n_frames,)   int64    — dimension scale, axis 0
    │   ├── frame_time_s            (n_frames,)   float64  — dimension scale, axis 0
    │   ├── slow_time_s             (n_pulses,)   float64  — dimension scale, axis 1
    │   ├── fast_time_s             (n_samples,)  float64  — dimension scale, axis 2
    │   └── rx                      (n_rx,)       int64    — dimension scale, axis 3, if present
    └── burst1/ …                   dual-PRF runs; bursts may differ in shape
```

- **One group per burst**, because a dual-PRF run's bursts have different `(n_pulses, n_samples)`.
  This is today's `burst0`/`burst1` key naming in the `.npz` files, kept.
- **`data` attributes:** `units = "sqrt(W)"` (|a|² is power in watts, per `PropagationPaths`),
  `long_name`, `domain = "ftst"` (the `style.md` §3.1 suffix), and
  `dims = ["frame", "pulse", "sample"(, "rx")]`. The coordinate datasets are also attached as HDF5
  **dimension scales**, which is what makes xarray, `h5dump` and NetCDF-aware tools show named axes.
- **Complex storage:** the HDF5 compound type `{r: float64, i: float64}`. That is h5py's native
  complex mapping, and the memory layout SigMF calls `cf64_le`. MATLAB reads it as a struct with
  fields `r` and `i`; HighFive maps it to `std::complex<double>`.
- **Chunking:** one frame per chunk, `(1, n_pulses, n_samples[, n_rx])`, so reading one frame
  touches one chunk. **Compression: `gzip` level 4 with the `shuffle` filter, and nothing else.**
  Every HDF5 build, including MATLAB's, ships gzip. `lzf`, Blosc and Zstandard are faster but need
  plugins that MATLAB and most C++ installs lack.
- **Recorded hardware captures** (`source = "recorded"`) may store `complex64`, or raw ADC counts
  as the compound `{r: int16, i: int16}` with a `scale_linear` attribute on `data` (sqrt(W) per
  LSB). Simulated runs are always `complex128` (`style.md` §7).
- A run too large for one file may be split into `iq_00000.h5`, `iq_00001.h5`, … with the same
  layout. The `frame` scale says which frames each part holds.

### 6.8 `products.h5` — L2 processed arrays

The same root attributes as §6.7. Every group is optional except `/rd`, which is written whenever
the range-Doppler stage runs (DF5).

```text
/
├── rd/burst{k}/                    the range-Doppler map, data_rv
│   ├── data                        (n_frames, n_doppler_bins, n_range_bins)  complex128
│   ├── range_m                     (n_range_bins,)    float64, not shifted: bin 0 is zero range
│   ├── velocity_mps                (n_doppler_bins,)  float64, fftshifted: zero at n // 2,
│   │                                                  closing-positive; attr fftshifted = 1
│   ├── frame, frame_time_s         (n_frames,)
│   └── noise_power_w               (n_frames,)        float64, the burst's thermal noise power
├── rt/burst{k}/data                (n_frames, n_pulses, n_range_bins)  complex128, data_rfst
├── cfar/burst{k}/threshold_w       (n_frames, n_doppler_bins, n_range_bins)  float64
└── cfar/burst{k}/mask              (n_frames, n_doppler_bins, n_range_bins)  uint8
```

- The `/rd` layout is exactly `RangeDopplerProduct` (`pipelines/scenarios.py`) with a frame axis
  in front: `rd_map`, `range_axis_m` and `velocity_axis_mps`, renamed to dataset names that carry
  the axis's own unit.
- The `data` attributes are as in §6.7, with `domain = "rv"` or `"rfst"`, plus `produced_by`: the
  qualified name of the function that produced the array, and the window it applied. Scaling is
  whatever that function returns. The spec does not renormalise.
- `/rd/burst{k}/noise_power_w` is the burst's thermal noise power, the `bursts[].noise_power_w`
  of §6.1 and the reference `snr_db` is defined against (§6.5). It is not a CFAR estimate: CFAR
  estimates the floor per cell, not per frame, and that estimate is `cfar/burst{k}/threshold_w`
  divided by the threshold factor α. An earlier draft described this dataset as "the noise
  estimate used by CFAR", which contradicted both the name, which means thermal noise everywhere
  else in this spec and in `core/`, and the shape. No file had been written to this layout, so
  the correction moves no data (`docs/audits/core-audit.md` F10).
- A future range-Doppler-azimuth cube goes in `/rda/burst{k}/data`,
  `(n_frames, n_doppler_bins, n_range_bins, n_azimuth_bins)`, following the channel-last rule.
  Its azimuth scale is `azimuth_boresight_deg` (DF2).

### 6.9 `metrics.csv`

One row per metric per scope. Because tables here are tidy and long, the unit goes in the metric's
name rather than in a column name. That keeps the unit-suffix rule without a `unit` column that
could disagree with it.

| Column | dtype | Req. | Meaning |
| :--- | :--- | :--- | :--- |
| `metric` | string | R | Name with unit suffix: `range_rmse_m`, `range_rate_rmse_mps`, `mean_nis_dim2`, `ospa_m`, `track_breaks`. NIS is averaged per measurement dimension, `mean_nis_dim<k>`, because its expected value is `k` |
| `track_id` | string | O | Empty for a run-level metric |
| `target_id` | string | O | |
| `frame_start`, `frame_end` | int | R | The inclusive frame window the metric covers |
| `value` | float64 | R | |

### 6.10 `exports/coco.json`

Specified by `spec/structure.md` D5 and unchanged. The two transformations D5 requires happen here
and nowhere else. The exporter transposes `/rd` into D5's range × Doppler × azimuth order (DF1)
and converts to boresight azimuth and dB (DF2, §2.4).

## 7. Reading and converting

Each snippet is the minimum a collaborator needs to read a run with no radar-forge installed.
§11's acceptance criteria require every one to run against a golden run in CI. The ones outside
Python are run manually until a runner exists.

### 7.1 Python

```python
import h5py
import pandas as pd

tracks = pd.read_csv("out/run/tracks.csv", dtype={"track_id": str})
with h5py.File("out/run/products.h5", "r") as f:
    rd = f["rd/burst0/data"][12]  # one frame: (n_doppler_bins, n_range_bins), complex128
    range_m = f["rd/burst0/range_m"][:]
    velocity_mps = f["rd/burst0/velocity_mps"][:]
```

For named axes, build an `xarray.DataArray` from the same three reads with
`dims=("velocity_mps", "range_m")`. `xarray.open_dataset(engine="h5netcdf")` also sees the
dimension scales, but its complex-number support depends on the h5netcdf version, so it is not the
documented path.

### 7.2 MATLAB

```matlab
tracks = readtable("out/run/tracks.csv", "TextType", "string");
s  = h5read("out/run/products.h5", "/rd/burst0/data");      % struct with fields r, i
rd = complex(s.r, s.i);
rd = permute(rd, ndims(rd):-1:1);   % MATLAB reverses HDF5 axis order; restore (frame, doppler, range)
meta = jsondecode(fileread("out/run/metadata.json"));
```

The `permute` is not optional. MATLAB is column-major and presents every HDF5 dataset with its
axes reversed. The cube arrives as `(n_samples, n_pulses, n_frames)` until you permute it.

### 7.3 C++ (HighFive, header-only over the HDF5 C library)

```cpp
HighFive::File file("out/run/iq.h5", HighFive::File::ReadOnly);
auto data = file.getDataSet("/iq/burst0/data");
auto dims = data.getDimensions();                 // {n_frames, n_pulses, n_samples}
std::vector<std::complex<double>> cube(dims[0] * dims[1] * dims[2]);
data.read_raw(cube.data());                       // row-major, matches the file order
```

For CSV, any RFC 4180 parser works. Note the §5 missing-value rule: an empty cell is not zero.

### 7.4 Stone Soup

```python
from stonesoup.reader.generic import CSVDetectionReader, CSVGroundTruthReader

truth = CSVGroundTruthReader(
    "out/run/truth.csv",
    state_vector_fields=("east_m", "north_m", "up_m"),
    time_field="time_utc",
    path_id_field="target_id",
)
detections = CSVDetectionReader(
    "out/run/detections.csv",
    state_vector_fields=("range_m", "velocity_unfolded_mps"),
    time_field="time_utc",
    metadata_fields=("frame", "detection_id", "peak_power_w", "associated_track_id"),
)
```

- **Sign.** Stone Soup's range-rate measurement models are opening-positive, and radar-forge files
  are closing-positive (§5, DF7). Negate the velocity column before building a range-rate
  measurement. Otherwise every closing target is filtered as an opening one, and the result still
  looks plausible.
- **No epoch.** For a synthetic run with no epoch, point `time_field` at `time_s` and pass
  `timestamp=True`. The reader then treats seconds as Unix time, which is harmless for tracking.

### 7.5 SigMF ↔ `iq.h5`

For hardware and SDR collaborators. A SigMF recording is a flat stream. A radar cube is that
stream reshaped, so the conversion is a mapping, not a transformation.

| SigMF | `iq.h5` |
| :--- | :--- |
| `global.core:datatype` `cf64_le` / `cf32_le` / `ci16_le` | `data` dtype `complex128` / `complex64` / `{int16, int16}` (§6.7) |
| `global.core:sample_rate` | `burst{k}` attr `sample_rate_hz` |
| `captures[n].core:frequency` | `burst{k}` attr `f0_hz` |
| `captures[n].core:datetime` | `epoch_utc + frame_time_s[n]` |
| One capture per frame, `core:sample_start = n · n_pulses · n_samples` | Frame `n` of `data` |
| Extension namespace `radar_forge:` in `global`, declared in `core:extensions`: `n_pulses`, `n_samples`, `prf_hz`, `bandwidth_hz`, `chirp_duration_s`, `waveform` | `burst{k}` attrs of the same names |
| `annotations` (`core:sample_start`, `core:sample_count`, `core:label`) | Not carried into `iq.h5`; truth lives in `truth.csv` |

The sample order is pulse-major (C order): sample index = `pulse · n_samples + sample`. For
multi-channel data, SigMF's convention is one recording per channel, so an `n_rx` cube becomes
`n_rx` recordings sharing a `core:collection`.

### 7.6 CSV → Parquet

```python
pd.read_csv("tracks.csv", dtype={"track_id": str}).to_parquet("tracks.parquet")
```

A Parquet mirror (`<name>.parquet` beside `<name>.csv`) is allowed for large runs. It carries
identical columns and dtypes, and the CSV remains canonical. `pyarrow` stays out of every extra
until something inside radar-forge needs to *write* Parquet.

## 8. Versioning and compatibility

- `schema_version` follows semver and appears in `metadata.json` and in the root attributes of
  every HDF5 file.
- **Minor** (1.0 → 1.1): add a column, dataset, attribute or optional group.
- **Major** (1.x → 2.0): rename, remove, re-type, re-unit or re-sign anything, or change an axis
  order.
- **Readers** select columns and datasets by name, ignore unknown ones, and refuse a major version
  they do not know, with a message naming both versions.
- **Writers** emit columns in §6's order. Order is not part of the reading contract, but a stable
  order keeps diffs of two runs readable.
- This document carries the changelog. Each version bump appends a row to §12's change table.

## 9. Enforcement

Following `style.md`'s tag table, every rule in this document is meant to be enforced by a tool
except the few only a human can judge.

| Tag | Enforcer (all follow-ups, §12) | Checks |
| :--- | :--- | :--- |
| **[registry]** | `radar_forge.pipelines.io.schema`: the column and dataset registry the writers use | Names, order, dtypes and units come from one place, so a writer cannot drift from the validator |
| **[validator]** | `scripts/check_run_output.py <run_dir>`, callable from tests | Required columns and datasets present; unit suffixes on every name; `schema_version` known; `files[]` inventory and hashes match; `conventions` block matches §5; empty cells only in columns that allow them; bistatic columns present iff siting is bistatic |
| **[test]** | `tests/pipelines/test_io_roundtrip.py` | `export → load → identical` for every file type, bit-exact for floats (DF9), per `docs/conventions/testing.md`; one golden mini-run in `tests/data/golden/` read by the §7.1 snippet |
| **[review]** | a human | That a new column belongs to the level it was added to; that a metric name says what it measures |

## 10. Decisions

Each decision settles a place where the code and the existing specs disagreed when this document
was written.

### DF1 — Files keep the library's axis order; the COCO exporter transposes

`RangeDopplerProduct.rd_map` is `(n_doppler_bins, n_range_bins)` (`pipelines/scenarios.py`), while
`spec/structure.md` D5 requires range × Doppler × azimuth. Files follow the library, because they
are read by the library and by collaborators debugging against it. Named axes and the
`conventions.rd_layout` string make the order explicit. D5's order is an interoperability contract
with the FMCW Target Simulator, so it is honoured exactly where that simulator's consumers read
it, in `exports/coco.json`.

### DF2 — `azimuth_deg` is a true-north bearing; boresight azimuth has its own name

`core/geodesy.py` measures azimuth from true north, clockwise. D5 measures it from boresight. These
are different quantities, not two conventions for one quantity, so they get different names.
`azimuth_deg` is the bearing, in every table. `azimuth_boresight_deg` is reserved for antenna-frame
angles and appears in `/rda` and in COCO exports.

### DF3 — `tracks.csv` carries the full state and covariance

Scenario 003 §9 promised ENU state columns for ENU models. `scripts/run_scenario.py`
`write_tracking_csvs` writes only `state[0]`, `state[1]`, `covariance[0, 0]` and
`covariance[1, 1]`, so an ENU run loses its state and every run loses its off-diagonal covariance.
NEES and consistency checks need the full matrix. `state_fields` plus the `var_` and `cov_` columns
keep it tabular without inventing a matrix-in-a-cell encoding.

### DF4 — `epoch_utc` is written whenever the trajectory has one

`Trajectory.epoch` (`pipelines/trajectories.py`) holds the absolute time of the first fix, but
nothing writes it, so no saved run can be aligned with an external log. It goes in
`metadata.json`, and `time_utc` goes in every table.

### DF5 — Range-Doppler maps are persisted

Today only PNGs of the RD map are written, and they are in dB and cropped to a dynamic range, so
nothing downstream can re-run detection on a saved run. `/rd` in `products.h5` is the
`RangeDopplerProduct` on disk.

### DF6 — Degrees on disk, radians in `core/`

`BistaticTargetTrack.bistatic_angle_rad` is the one radian field among the angles that reach a
file. It is converted at write time, as `truth.csv`'s `bistatic_angle_deg` already is. The rule is
`style.md` §2's boundary rule, applied to the file boundary.

### DF7 — Closing-positive velocity everywhere on disk

`core/tracking/` states that range rate is positive closing, per D5, and `range_rate_mps` in
`tracks.csv` inherits it. Stone Soup and ASTERIX use opening-positive range rate. We keep one sign
across every file and flip at the external boundary (§7.4), rather than giving the tracker's output
a different sign from its input. The `conventions.velocity_sign` string lets a reader check it.

### DF8 — HDF5 replaces `.npz`, behind an `io` extra

Argued in §3. The extra keeps core at NumPy and SciPy, per `spec/structure.md` B.2 rule 1.

### DF9 — Floats round-trip exactly

Today's writer formats `time_s` and `range_m` as `%.3f`. A millimetre is physically fine, but
`export → load → identical` fails and so does any hash-based comparison of two runs. Shortest
round-trip representation costs a few bytes per cell.

## 11. Acceptance criteria

| # | Criterion | Checked by |
| :--- | :--- | :--- |
| AC1 | Scenarios 001 (all variants), 002 and 003 each produce a run directory that passes the validator | [validator] in `tests/pipelines/` |
| AC2 | Every CSV and HDF5 file round-trips bit-exactly through the radar-forge reader | [test] |
| AC3 | The §7.1 Python snippet reads the golden run with only `pandas` and `h5py` installed | [test] |
| AC4 | The §7.2 MATLAB snippet reads the golden run and the permuted cube equals the Python read | manual, recorded in the PR |
| AC5 | The §7.4 Stone Soup readers load `truth.csv` and `detections.csv` of the golden run and yield one path and the expected detection count | [test], behind an optional marker |
| AC6 | SigMF → `iq.h5` → SigMF reproduces the `.sigmf-data` bytes | [test] |
| AC7 | A core-only install writes every CSV and JSON file, and raises the `io`-extra `ImportError` for HDF5 | [test] |
| AC8 | `make check` passes, including rule R7 on this document | [hook] |

## 12. Migration and follow-ups

**What changes relative to today's outputs** (`scripts/run_scenario.py`):

| Today | Under this spec | Kind |
| :--- | :--- | :--- |
| `iq_{frame:05d}.npz`, keys `burst0`, `burst1` | `iq.h5`, groups `/iq/burst0`, `/iq/burst1` | format change |
| — | `products.h5` `/rd` | new |
| `truth.csv`, 6 (+3 bistatic) columns | same columns, plus `target_id`, `time_utc`, geodetic and ENU position | additive |
| `detections.csv`, 9 columns | same columns, plus `detection_id`, `sensor_id`, `time_utc`, angles, cell indices, `total_power_w`, `snr_db` | additive |
| `tracks.csv`, 11 columns | same columns, plus state model, counters, ENU state, full covariance, geodetic position, `associated_detection_id` | additive |
| `metadata.json` | same keys, plus `schema_version`, `run_id`, `created_utc`, `epoch_utc`, `reference_site`, `conventions`, `files`, `tracking.state_fields` | additive |
| `%.3f` / `%.6f` / `%.6e` float formatting | shortest round-trip | precision change (DF9) |
| `rd_*.png`, `track_*.png`, `*.mp4` in the run root | `figures/` | move; `clear_previous_frames` globs must follow |
| — | `inputs/scenario.toml`, `inputs/trajectory.csv` | new |
| Trajectory header `timestamp,lat,lon` | `time_s,latitude_deg,longitude_deg` (§6.3) | rename, **done** |

Every existing output column keeps its name, unit and sign, so a reader of today's output files
keeps working on the new ones. The one rename is to an input, the trajectory header, and it has
already landed: the data files, the loader and `scripts/translate_flight_coordinates.py` all use
the §6.3 names.

**Follow-ups, in build order.** Each is its own PR; none is part of adopting this document. None of
the §9 enforcers exists yet: the registry is item 4, and the validator and round-trip test are
item 5. Only rule R7, which AC8 cites, is in place.

1. Bring the scenario specs and their companions onto the §6.3 column names wherever they describe
   the trajectory CSV: `spec/scenario-001-xband.md` §2.2 ("The CSV carries `timestamp,lat,lon`
   only") and the §2 pipeline diagram, and the matching diagram in
   `docs/scenarios/scenario-001-xband.md`. No dependencies.
2. `scripts/regen_golden_flight_coordinates_head.py`, the generator `docs/conventions/testing.md`
   §6 requires and the 51-row excerpt still lacks. No dependencies.
3. Add the `io` extra (`h5py`) to `pyproject.toml`, with the dependency note from §3 in the PR.
4. `src/radar_forge/pipelines/io/`: the schema registry, CSV/JSON writers and readers
   (stdlib-only), and HDF5 writers and readers (lazy `h5py` import).
5. `scripts/check_run_output.py`, the §9 validator, and `tests/pipelines/test_io_roundtrip.py`,
   plus a golden mini-run in `tests/data/golden/` with its `scripts/regen_golden_*.py` generator,
   per `docs/conventions/testing.md`.
6. Move `scripts/run_scenario.py` onto the io module, including writing `inputs/scenario.toml` and
   `inputs/trajectory.csv`, and add `scripts/convert_npz_to_h5.py` for old runs.
7. A one-line note in scenario specs 001 §3.6, 002 §3.6 and 003 §9 pointing here as superseding
   their output schemas. It comes after item 6 because until then those sections still describe
   what is written.
8. `scripts/convert_sigmf.py` (§7.5).
9. A JSON Schema for `metadata.json` at `spec/schemas/metadata.schema.json`.

**Change log.**

| Version | Date | Change |
| :--- | :--- | :--- |
| 1.0.0 | 2026-09-29 | First proposal |
| 1.1.0 | 2026-10-07 | `detections.csv` adds `burst_index`, `status` and `pair_id`; `metadata.json` `tracking` adds `range_period_m` and `unused`, and records the resolved settings, defaults included, beside the `[tracking]` table's own keys |

## References

- SigMF Working Group, *Signal Metadata Format (SigMF) Specification*, v1.2, 2024.
  https://github.com/sigmf/SigMF
- The HDF Group, *HDF5 User's Guide*, "Dimension Scales"; and the *HDF5 Dimension Scale
  Specification and Design Notes*. https://docs.hdfgroup.org/
- B. Eaton et al., *NetCDF Climate and Forecast (CF) Metadata Conventions*, v1.11, 2022.
  https://cfconventions.org/
- NCAR/EOL and WMO, *CF-Radial Data File Format*, v2.0, adopted as WMO FM-301.
- EUMETNET OPERA, *ODIM_H5: OPERA Data Information Model for HDF5*, v2.4, 2021.
- EUROCONTROL, *ASTERIX Category 048: Monoradar Target Reports*, and *ASTERIX Category 062: SDPS
  Track Messages*. https://www.eurocontrol.int/asterix
- P. A. Thomas, J. Barr, B. Balaji and K. White, *An open source framework for tracking and state
  estimation ("Stone Soup")*, Proc. SPIE 10200, 2017; `stonesoup.reader.generic` for the CSV readers.
- T.-Y. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014; data format at
  https://cocodataset.org/#format-data
- Y. Shafranovich, *Common Format and MIME Type for Comma-Separated Values (CSV) Files*, RFC 4180, 2005.
- ISO 8601-1:2019, *Date and time — Representations for information interchange*.
- M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014 (the radar
  data cube and its slow-time/fast-time axes).
- Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and
  Navigation*, Wiley, 2001, §5.4 (NIS and NEES, and why the full covariance must be kept).
- `spec/structure.md` D1, D5, D6; `docs/conventions/style.md` §2, §3.1, §7; `docs/conventions/testing.md`.

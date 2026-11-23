# CLAUDE.md

`radar-forge` is a radar workbench for research and education: current methods with correct,
cited maths, in plain code that shows its intermediate quantities, read by practitioners, interns
and students. Neither half is traded for the other.

## Non-negotiables (and what enforces each)

- `make check` green before you say you are done; `--no-verify` is not a fix — CI runs the same.
  The R-rules are listed once, in the `scripts/check_conventions.py` docstring; each failure
  prints its fix, so rules a hook teaches on first contact are not repeated here.
- Python ≥ 3.11, src layout, all configuration in `pyproject.toml`, everything through `uv`
  (`uv run pytest`), never `pip install` — R1, R2, the hooks.
- SI units with the unit in the name (`range_m`, `gain_tx_dbi`, never `gain_tx`) — R5.
- Physical constants only from `radar_forge.constants`; never `3e8` — R4, mypy `Final`.
- Lean NumPy docstrings: one summary line; a citation with equation number; Parameters/Returns only where a unit, shape or convention is not in the name; no Raises or Examples unless the example is the test — ruff `D`, `tests/test_conventions.py`.
- Complete annotations, `ArrayLike` in, `NDArray` out; every module defines `__all__`; `msg = "…"` then `raise`, no `assert` — `mypy --strict`, ruff `ANN EM S101`, `tests/test_conventions.py`.
- Broadcast, don't tile — R6. Only `scenario`, `pipeline`, `figures` and `verify/` do I/O, logging or timing — R11. Modules warn
  past 500 lines, fail past 800 — R10.
- No generated arrays committed; golden data only in `tests/data/golden/` — pre-commit hook.
- Conventional Commits, `feat(dsp): add Taylor taper`; PRs squash-merge — commit-msg hook, CI.

## Judgement calls

- Never weaken a test tolerance or reseed to pass; find out why the number moved.
- Prefer analytic ground truth in tests. A Python loop that must stay says why in a comment.
- A new dependency says in the PR what it buys over numpy/scipy; optional backends go in an extra.
- Check a spec before implementing it: re-derive its numbers, fix it first in a `docs(spec)` commit.

## Local machine

Windows: run `make` and the hooks from Git Bash (`bash -c "make check"`). Use `uv run python`,
never a bare `python3`. Sync with `uv sync --extra dev --extra visualisation` (add
`--extra notebooks` for Jupyter): `uv sync` uninstalls every extra not named, and never use
`--all-extras` (it pulls in torch). `make test-fast` is the inner loop and the pre-push gate.

## Spec lifecycle

A scenario's spec is deleted in the same PR that implements it, which encodes its acceptance
numbers in the scenario TOML's `[verify]` table. `spec/` holds only unimplemented scenario specs,
plus `spec/data-001-formats.md` (file formats) and `spec/tracker-001.md` (tracking architecture).

## Conventions you cannot grep for

- Range rate is the time derivative of range, positive receding; Doppler = -rate/wavelength, written once in `propagation.doppler_shift_hz`. Map axes and an IQ scenario's tracker state are bistatic, R_T + R_R and its rate (2R monostatic); only `figures.display_axes` halves them, for figures and CSVs. A plot-level scenario (011) tracks ENU position and velocity on the ground plane.
- One word per concept, defined once in the glossary of `docs/user-guide/README.md`: scan (not frame), burst (one CPI; not dwell), detection, plot and measurement (three layers), figure (a picture), range rate, period (span in prose; `*_period_*` in names), clutter density.
- The FMCW generator relies on a down-sweep.
- British spelling (`normalise`, `visualisation`) except where mirroring an outside API.
  SI units internally; dB only at API boundaries, named `_db*`, converted by `radar_forge.units`.

## Where things are: maths

| Relation | Function (file under `src/radar_forge/`) |
| :--- | :--- |
| wavelength; PRI, duty cycle, sweep rate; range resolution c/2B | `Transmitter.wavelength_m`; `.pulse_repetition_interval_s`, `.duty_cycle_linear`, `.sweep_rate_hzps` and `sweep_rate_hzps`; `Radar.range_resolution_m` and `range_resolution_m` (radar.py) |
| unambiguous delay, range, rate | `Radar.unambiguous_delay_s`, `.unambiguous_bistatic_range_m` (c τ, the map's range period), `.unambiguous_bistatic_range_rate_mps` (λ PRF, the whole rate period) (radar.py) |
| route geometry, baseline, bistatic angle | `Radar.geometry` -> `TargetGeometry`, `Radar.baseline_m`, `TargetGeometry.bistatic_angle_rad` (radar.py) |
| geodesy: geodetic, ECEF, ENU, range-azimuth-elevation; on the ground plane; unit direction | `geodetic_to_ecef_m`, `ecef_to_enu_m`, `geodetic_to_enu_m`, `enu_to_range_azimuth_elevation_rad`, `range_azimuth_elevation_to_enu_m`; `east_north_to_range_azimuth_rad`, `range_azimuth_to_east_north_m`; `directions_unit` (coordinates.py) |
| beat frequency ↔ bistatic range | `beat_frequency_hz`, `bistatic_range_from_beat_frequency_m` (signal.py) |
| delay and its inverse, phase, wavenumber 2πf/c, Doppler and its inverse; stop-and-hop | `propagation_delay_s`, `bistatic_range_from_delay_m`, `propagation_phase_rad`, `wavenumber_radpm`, `doppler_shift_hz`, `bistatic_range_rate_from_doppler_mps`; `PropagationPaths.bistatic_range_at_m` (propagation.py) |
| radar equation; RCS, Swerling 0-4 | `received_power_w` (R_T² R_R²); `dbsm_to_m2`, `swerling_rcs_m2` (propagation.py) |
| dB: a power 10 log10, an amplitude 20 log10; antenna gains linear | `power_to_db`, `db_to_power`, `amplitude_to_db`, `db_to_amplitude` (units.py); `Transmitter.gain_tx_linear`, `Receiver.gain_rx_linear` (radar.py) |
| noise N0 = kT0F, per sample, in band | `Receiver.noise_density_w_per_hz`, `.noise_power_per_sample_w`, `Radar.noise_power_matched_w` (radar.py) |
| SNR, single pulse and integrated | `received_power_w` (propagation.py) over `Radar.noise_power_matched_w` (radar.py); times B T N on the map, in the geometry stage (verify/chain.py) |
| thermal noise; synthesis; LFM chirp, noise code, amplifier; ambiguity | `thermal_noise`; `fmcw_deramp_baseband`, `pulsed_baseband`, `stream_baseband`; `lfm_chirp`, `noise_waveform`, `cubic_amplifier`; `ambiguity_function` (signal.py) |
| leakage, near clutter; LS / Wiener-SMI cancellation; CA bound, notch | `leakage_paths`, `static_clutter_paths` (propagation.py); `least_squares_cancel`, `wiener_smi`, `delay_basis_gram`; `expected_cancellation_depth_db`, `doppler_notch_response_linear` (clutter.py) |
| windows, coherent gain, processing loss | `window`, `apply_window`, `coherent_gain_linear`, `processing_loss_db` (dsp.py) |
| matched filter, batched correlation, range/Doppler FFT, map | `matched_filter`, `batched_correlation`, `range_fft`, `doppler_fft`, `range_doppler_map` (dsp.py); `range_doppler_product` (pipeline.py) |
| bin axes, Doppler resolution; MTI | `range_bin_centres_m`, `doppler_bin_centres_mps` (its spacing λ/(N T)); `mti_filter` (dsp.py) |
| fold, dual-PRF unfold | `fold_range_rate_mps`, `unfold_doppler_dual_prf`, `unfold_range_dual_prf` (dsp.py; all take the period) |
| wrap any periodic value into `[lower, lower + period)` | `fold_into_period` (dsp.py); `tracking/`, `clustering`, `coordinates` and `doa` call `fold` (_periodic.py) |
| CFAR Pfa, threshold, detect; clustering of cells (connected, or mean-shift modes) | `cfar_probability_of_false_alarm`, `cfar_threshold_factor`, `cfar_detect`, `cfar_detect_2d`; `label_connected_cells`, `cluster_detections(bandwidth_bins=)` (detection.py) |
| clustering of points: DBSCAN, mean shift; per-column scale and period | `cluster_points`, `mean_shift` (clustering.py) |
| state and measurement coordinates; h(x): linear, spherical, bistatic range and rate, a marginal; initiators | `CoordinateLayout` (tracking/motion.py); `LinearMeasurement`, `RangeAzimuthElevationRangeRate`, `BistaticRangeAndRangeRate`, `marginal` (tracking/measurement.py); `DirectStateInitiator`, `DebiasedSphericalInitiator` (tracking/initiation.py) |
| predict, innovate, update, weighted update; NEES and its interval, GOSPA, trajectory GOSPA | `predict`, `innovate`, `update`, `update_weighted`, `TaylorLinearisation`, `UnscentedTransform` (tracking/filters.py); `nees`, `consistency_interval`, `gospa`, `trajectory_gospa` (tracking/metrics.py) |
| continuous or discrete white-noise acceleration, constant (coordinated) turn; IMM, Π = expm(ΛT) | `ConstantVelocity` (`.bistatic_range`), `ConstantAcceleration`, `ConstantTurn` (tracking/motion.py); `InteractingMultipleModel`, `imm_mix`, `imm_predict`, `imm_update` (tracking/imm.py) |
| χ² gate, GNN, PDA/JPDA/JPDA*/set weights, Murty k-best; M-of-N, JIPDA existence; MHT track score, N-scan; PMBM | `chi_square_gate_threshold`, `non_assignment_cost` ln(P_D/(1 − P_D P_G)), `GlobalNearestNeighbour`, `probabilistic_data_association`, `joint_probabilistic_data_association`, `murty_k_best` (tracking/association.py); `LifecyclePolicy`, `ExistencePolicy` (tracking/lifecycle.py); `TrackOrientedMHT`, `TrackTree` (tracking/hypotheses.py); `PmbmTracker` (tracking/pmbm.py) |
| route acceleration R̈_b; Doppler rate | `Radar.geometry(acceleration_enu_mps2=)` (radar.py); `doppler_rate_hzps` (propagation.py) |
| range gate, its delay axis; compression through it; keystone, fold factor, Doppler-rate filter | `pulsed_baseband(range_gate_start_m=, n_gate_samples=)`, `range_gate_first_sample` (signal.py); `delay_bin_centres_m` (dsp.py); `pulse_compress`; `keystone`, `second_order_keystone`, `fold_factor_search`, `doppler_rate_compensation`, `compensate_migration` (migration.py) |
| data chip; sub-bin and polynomial-phase estimates; CRB | `crop_slow_time`, `crop_range_doppler`, `estimate_tone`, `estimate_fine`, `cramer_rao_bounds` (datachip.py) |
| receive array: ENU to array frame, steering e^{+j2π p·u/λ}, per-channel synthesis; weights w^H x; covariance, smoothing, FB | `fmcw_deramp_baseband(rx_response_linear=)` (signal.py); `enu_to_array_frame`, `steering_vectors`; `conventional_weights`, `mvdr_weights`, `apply_weights`; `sample_covariance`, `spatial_smoothing`, `forward_backward_average` (array.py) |
| DOA: Bartlett, Capon, MUSIC, root-MUSIC, ESPRIT, IAA, SPICE, ML; deterministic CRB, Smith's limit, monopulse | `estimate_doa`, `bartlett_spectrum` ... `deterministic_ml`; `crb_deterministic`, `resolution_snr_db`, `monopulse_ratio` (doa.py) |
| trajectories (arcs); plot merge P_u = exp(-d²/2), Pd, clutter per m·rad, a rotating radar's scans; clutter map | `Arc`, `Leg` (trajectory.py); `resolve`, `detect`, `measure`, `ClutterComponent`, `plot_scans`, `sector_index` (plot_simulator.py); `ClutterMap` (tracking/clutter_map.py) |
| one scan to the tracker: one batch staring, a batch a sector rotating, tracks outside a sector left waiting | `Scan`, `Tracker.process`, `Coverage` (tracking/tracker.py); `ScanTracker`, `ScanTracker.process` (scan_tracker.py) |

No function yet: Pd (Swerling detection curves).

## Where things are: files

| Path | What |
| :--- | :--- |
| `src/radar_forge/propagation.py` | The style exemplar; copy its shape |
| `src/radar_forge/coordinates.py` | Geodetic, ECEF, ENU and range-azimuth conversions |
| `src/radar_forge/constants.py` | Every physical constant, defined once |
| `docs/user-guide/` | One page a stage, each example run by the tests; its README holds the notation and the glossary |
| `docs/conventions/` | `style.md`, `testing.md`, `review.md` (also `/pr-review`), `commits.md` |
| `.githooks/`,`Makefile`, `scripts/check_conventions.py` | Hooks; every gate defined once; rules R1–R12 |
| `config/*.toml`, `scripts/run_scenario.py` | Scenarios with their `[verify]` numbers; the stepper (`radar_forge.verify`, `verify/stepper.py`) runs them |
| `spec/` | Unimplemented scenarios, data-001 formats, tracker-001 architecture |

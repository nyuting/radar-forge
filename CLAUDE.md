# CLAUDE.md

`radar-forge` is a radar workbench for research and education: current methods with correct,
cited maths, in plain code that shows its intermediate quantities, read by practitioners, interns
and students. Neither half is traded for the other ([`spec/starter.md` §1.1](spec/starter.md#11-the-bar)).

## Non-negotiables (and what enforces each)

- Python ≥ 3.11, src layout; only Python under `src/radar_forge/` — ruff `UP`, R2.
- All configuration in `pyproject.toml`; no `setup.py`, `setup.cfg`, `tox.ini`, `requirements*.txt` — R1.
- Everything through `uv` (`uv run pytest`), never `pip install` — the hooks call `uv run`.
- `make check` green before you say you are done; `--no-verify` is not a fix — CI runs the same.
- SI units with the unit in the name (`range_m`, `gain_tx_dbi`, never `gain_tx`) — R5.
- Physical constants only from `radar_forge.core.constants`; never `3e8` — R4, mypy `Final`.
- NumPy docstrings, doctests that run — ruff `D`, R3, `tests/test_docstrings.py`. `References` and shapes `(n_pulses, n_samples)` are review items.
- Complete annotations, `ArrayLike` in, `NDArray` out — `mypy --strict`, ruff `ANN`, `tests/test_public_api.py`.
- Every module defines `__all__` — `tests/test_public_api.py`. `msg = "…"` then `raise`, no `assert` — ruff `EM`, `S101`.
- Broadcast, don't tile, unless `# broadcast-exempt: <reason>` — R6. No banners, `# Inputs:` or control chars — R8, R9.
- Modules warn past 500 lines, fail past 800 — R10. `core/` does no I/O, logging or timing — R11.
- No `np.errstate`/`np.seterr` in `src/` without `# errstate-exempt: <reason>` — R12.
- No generated arrays committed; golden data only in `tests/data/golden/` — pre-commit hook.
- Conventional Commits, `feat(core): add Taylor taper`; PRs squash-merge — commit-msg hook, CI.

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

- Range rate: today's code is closing-positive (data-001 §5); the Phase 2 physics PR switches the whole repo to the time derivative of range, positive receding, with Doppler = -rate/wavelength written once. Until then, negate at any Stone Soup boundary.
- The FMCW generator relies on a down-sweep.
- British spelling (`normalise`, `visualisation`) except where mirroring an outside API.
  SI units internally; dB only at API boundaries, named `_db*`.

## Where things are: maths

| Relation | Function (file under `src/radar_forge/`) |
| :--- | :--- |
| wavelength; PRI, duty cycle, sweep rate | `Transmitter.wavelength_m`; `.pulse_repetition_interval_s`, `.duty_cycle_linear`, `.sweep_rate_hzps` (core/radar.py) |
| range resolution c/2B | `range_resolution_m` (core/waveforms.py); `_RadarChain.range_resolution_m` (core/radar.py) |
| bistatic range resolution | `BistaticRadar.range_resolution_at_bistatic_angle_m` (core/radar.py) |
| unambiguous delay, range, velocity | `_RadarChain.unambiguous_delay_s`, `Radar.unambiguous_range_m`, `BistaticRadar.unambiguous_range_m` (cτ, a range sum), `_RadarChain.unambiguous_velocity_mps` (core/radar.py) |
| beat frequency ↔ range | `beat_frequency_hz`, `range_from_beat_frequency_m` (core/waveforms.py) |
| delay, phase, Doppler, range-sum rate | `propagation_delay_s`, `propagation_phase_rad`, `doppler_shift_hz`, `range_sum_rate_mps` (core/propagation.py) |
| stop-and-hop range evolution | `PropagationPaths.range_sum_at_m` (core/propagation.py) |
| baseline, bistatic angle | `BistaticRadar.baseline_m`, `.bistatic_angle_rad` (core/radar.py) |
| radar equation | `received_power_w`, `bistatic_received_power_w` (core/radar_equation.py) |
| noise N0 = kT0F, per sample, in band | `Receiver.noise_density_w_per_hz`, `.noise_power_per_sample_w`, `_RadarChain.noise_power_matched_w` (core/radar.py) |
| thermal noise draw; baseband synthesis | `thermal_noise`; `fmcw_deramp_baseband`, `pulsed_baseband` (core/signal.py) |
| LFM chirp | `lfm_chirp` (core/waveforms.py) |
| matched filter, range/Doppler FFT, map | `matched_filter`, `range_fft`, `doppler_fft`, `range_doppler_map` (core/dsp.py) |
| bin axes; MTI | `range_bin_centers_m`, `doppler_bin_centers_mps`; `mti_filter` (core/dsp.py) |
| windows, coherent gain, processing loss | `taper`, `coherent_gain_linear`, `processing_loss_db` (core/windows.py) |
| CFAR Pfa, threshold, detect; clustering | `cfar_probability_of_false_alarm`, `cfar_threshold_factor`, `cfar_detect`, `cfar_detect_2d`; `cluster_detections` (core/detection.py) |
| fold, dual-PRF unfold | `fold_into_period`, `unfold_doppler_dual_prf`, `unfold_range_dual_prf` (core/ambiguity_resolution.py) |
| geodesy | `geodetic_to_enu_m`, `enu_to_range_azimuth_elevation_rad` (core/geodesy.py) |
| RCS dBsm → m² | `dbsm_to_m2` (core/targets.py) |
| Kalman predict/update, NIS | `GaussianFilter`, `innovation_stats` (core/tracking/estimation.py) |
| DWNA/CWNA process noise | `CartesianMotion`, `RadialMotion` (core/tracking/motion.py) |
| χ² gate, GNN | `ChiSquareGate`, `GlobalNearestNeighbour` (core/tracking/association.py) |
| M-of-N confirmation | `LifecyclePolicy` (core/tracking/lifecycle.py) |

No function yet: SNR, Doppler resolution, integration gain, Pd/Swerling.

## Where things are: files

| Path | What |
| :--- | :--- |
| `src/radar_forge/core/radar_equation.py` | The style exemplar; copy its shape |
| `src/radar_forge/core/constants.py` | Every physical constant, defined once |
| `docs/conventions/` | `style.md`, `testing.md`, `review.md` (also `/pr-review`), `commits.md` |
| `.githooks/`,`Makefile`, `scripts/check_conventions.py` | Hooks; every gate defined once; rules R1–R12 |
| `scenarios/*.toml` | Scenario configurations (and, once implemented, their `[verify]` numbers) |
| `spec/` | Unimplemented scenarios, data-001 formats, tracker-001 architecture |

# radar-forge — Module Structure

> `radar-forge` is built on a lightweight core with optional extras. Every block meets the bar in
> [`starter.md` §1.1](starter.md#11-the-bar): the current best published method, with cited
> mathematics, written so a new engineer can read it and a practitioner can see its intermediate
> quantities. Around that it prioritises zero-friction imports and modular backend pluggability.
>
> **Core principles**
> - **State of the art and simple** — both are required; neither is traded for the other.
> - **Lightweight core** — `radar_forge.core` and `radar_forge.array` rely only on NumPy and SciPy.
> - **Zero-breakage imports** — `import radar_forge` always succeeds.
> - **Unified backend contract** — one `raytracing.Scene` powers all backends.
> - **Clean licensing and attribution** — copyleft or unlicensed code is never vendored.

**Status:** design document, partially implemented. `core/` and `pipelines/` are built as far as
scenarios 001, 002 and 003 required, and `core/tracking/` is a package built out further, to
[`tracker-001.md`](tracker-001.md); `array/`, `raytracing/`, `pipelines/exporters/` and most of
`viz/` are still design only. Part B marks the tree as intended, not as built — read it
alongside the source.
**Companion to:** [`starter.md`](starter.md) (project charter and ecosystem survey).
**File formats:** [`data-001-formats.md`](data-001-formats.md) fixes the on-disk form of every
input, output and product the tree below reads or writes.
**Tracking architecture:** [`tracker-001.md`](tracker-001.md) owns the internal design of
`core/tracking/` — its pieces, their protocols, and how the filters are kept from drifting. This
document only places the package in the tree.

This document does two things, in the order a reader needs them. **Part B** is the file-level design
of the `radar_forge` package: the tree as intended, the module-to-upstream mapping, the design rules
and the extras. **Part A** is the survey behind it — each reference project named in the starter
spec, and what its actual core modules are — which is reference material rather than a narrative,
and so follows rather than precedes the design it justifies.

The scenario specifications are the vertical slices through this tree:

| Slice | Specification | What it exercises |
| :--- | :--- | :--- |
| Scenario 1 | [`scenario-001-xband.md`](scenario-001-xband.md) | trajectory → IQ → range-Doppler map, in three ambiguity variants |
| Scenario 2 | [`scenario-002-bistatic.md`](scenario-002-bistatic.md) | the same path over a two-site geometry; D6, D7, D8 |
| Scenario 3 | [`scenario-003-tracking.md`](scenario-003-tracking.md) | CFAR detection → association → Kalman tracking, stacked on scenario 1 |

---

## Part B — `radar_forge` structure

```text
src/radar_forge/
├── __init__.py                  # curated public API; no heavy/optional imports at import time
├── config.py                    # units, constants, global dtype/backend settings
├── core/
│   ├── __init__.py
│   ├── radar.py                 # Transmitter, Receiver, Radar, BistaticRadar, RadarLike
│   ├── radar_equation.py        # range equation (monostatic + bistatic), SNR, max range, link budget
│   ├── geodesy.py               # WGS-84 geodetic ↔ ECEF ↔ ENU; ENU → range/az/el
│   ├── waveforms.py             # FMCW/LFM chirp, pulse train, CW, PMCW; ambiguity function
│   ├── targets.py               # PointTarget, ExtendedTarget, RCS + Swerling 0–4
│   ├── propagation.py           # free-space loss, atmospheric absorption, rain attenuation (deferred)
│   ├── signal.py                # baseband synthesis, superposition, noise, phase noise
│   ├── dsp.py                   # range FFT, Doppler FFT, matched filter, MTI (range axis is bistatic mean range when bistatic)
│   ├── windows.py               # Taylor, Chebyshev, Hamming, Hann tapers; coherent gain, loss
│   ├── ambiguity.py             # velocity folding; dual-PRF Doppler unfolding
│   ├── detection.py             # CA/GO/SO/OS-CFAR (1-D), CA/OS 2-D ring, Pfa calibration, wrap-aware clustering
│   ├── clutter.py               # land/sea clutter models; ECA / Wiener-SMI cancellation
│   └── tracking/                # one tracker, cut at Stone Soup's seams (spec/tracker-001.md §3)
│       ├── __init__.py          # re-exports every public name; import path is radar_forge.core.tracking
│       ├── coordinates.py       # Coordinate, StateLayout (named, unit-bearing, wrap-aware), StateEstimate
│       ├── estimation.py        # Estimator protocol; the one correction core; NIS and log-likelihood
│       ├── kalman.py            # linear KF and EKF as Estimators; predict/update kept as the teaching layer
│       ├── ukf.py               # unscented Kalman filter
│       ├── motion.py            # MotionModel: Cartesian CV/CA, range-only; DWNA process noise
│       ├── measurements.py      # Measurement, MeasurementBatch (planned: today in measurement_models.py)
│       ├── measurement_models.py # MeasurementModel: Cartesian, spherical, monostatic and bistatic range-Doppler
│       ├── sensors.py           # SensorRegistration, SensorPose (planned: today in measurement_models.py)
│       ├── association.py       # chi-square gate; GNN and NN assignment; JPDA extension point
│       ├── initiation.py        # TrackInitiator: start a track from an unassigned measurement
│       ├── lifecycle.py         # M-of-N confirmation; miss, coast-time and covariance deletion
│       ├── tracks.py            # Track, TrackStatus, TrackSnapshot
│       ├── tracker.py           # Tracker: the one scan loop every pipeline uses
│       └── _validation.py       # shared shape and covariance checks
├── array/
│   ├── __init__.py
│   ├── geometry.py              # ULA, URA, circular, cylindrical, spherical, conformal, sparse; subarrays
│   ├── elements.py              # element patterns, polarization frames (Ludwig-3)
│   ├── tapering.py              # Taylor, Chebyshev, Hamming, Hann, custom amplitude tapers
│   ├── beamforming.py           # steering vectors, null steering, multi-beam, MVDR/Capon
│   ├── impairments.py           # phase quantization, mutual coupling, element failures
│   ├── patterns.py              # vectorized array factor, 2D cuts, 3D/UV patterns
│   └── doa.py                   # DoAEstimator base + MUSIC, Root-MUSIC, ESPRIT, Capon, Bartlett
├── raytracing/
│   ├── __init__.py
│   ├── base.py                  # RayTracingBackend protocol: trace(scene, radar) -> paths/baseband
│   ├── scene.py                 # backend-neutral Scene: meshes, materials, poses, trajectories
│   ├── materials.py             # EM material properties, permittivity/conductivity tables
│   └── backends/
│       ├── __init__.py          # lazy registry; missing deps degrade to a clear error, never ImportError at import
│       ├── analytic.py          # always-available fallback: point-target / specular approximation,
│       │                        #   built on core.signal.line_of_sight_paths rather than duplicating it
│       ├── radarsimpy.py        # RadarSimPy backend (extra: radarsimpy)
│       ├── mitsuba.py           # Mitsuba/Dr.Jit backend, RF-Genesis-style (extra: mitsuba)
│       └── ovrtx.py             # NVIDIA Omniverse RTX backend (extra: ovrtx)
├── pipelines/
│   ├── __init__.py
│   ├── scenarios.py             # TOML scenario schema + loader (stdlib tomllib); frame loop
│   ├── trajectories.py          # trajectory load/resample; monostatic and bistatic radar frames
│   ├── tracking.py              # RD map → measurements → core.tracking.Tracker, frame by frame (scenario 3)
│   ├── generate.py              # scene -> baseband -> cube orchestration, batching, seeding
│   ├── datasets.py              # torch Dataset / DataLoader wrappers (extra: ml)
│   └── exporters/
│       ├── __init__.py
│       ├── coco.py              # range-Doppler(-azimuth) cube -> COCO annotations
│       ├── range_doppler.py     # cube serialization per spec/data-001-formats.md (HDF5); dB scaling
│       └── labels.py            # shared label schema: range, velocity, azimuth, x, y, w, h, heading, obstruction
└── viz/
    ├── __init__.py
    ├── app.py                   # PySide6 application shell (extra: gui)
    ├── scopes/
    │   ├── __init__.py
    │   ├── ascope.py            # amplitude vs range
    │   ├── bscope.py            # range vs azimuth
    │   ├── ppi.py               # plan position indicator
    │   ├── rd_map.py            # live range-Doppler map, with detection and gate overlays
    │   └── track_plot.py        # range vs time: truth, detections, track history, current gate
    └── plotting.py              # matplotlib helpers shared by notebooks and scopes
```

Notebooks are not part of the installed package. Chapter-style guided notebooks live in a top-level
`notebooks/` directory beside `src/`, and run with the `notebooks` extra.

### B.1 Module-to-upstream mapping

| radar-forge module | Draws from |
| :--- | :--- |
| `core/radar.py` | RadarSimPy `radar.py` (Transmitter/Receiver/Radar composition) |
| `core/radar_equation.py`, `core/propagation.py` | RadarBook `pyradar`; RadarSim physics modules |
| `core/waveforms.py` | RadarBook (ambiguity function); AIRadarLib `waveform_utils` (PLL quantization, phase noise); RadarSim LFM |
| `core/targets.py` | RadarSimPy `tools` (Swerling); FMCW Target Simulator target classes |
| `core/signal.py` | FMCW Target Simulator `modelBasebandSignal.m`; RadarSimPy simulator semantics |
| `core/dsp.py`, `core/detection.py` | RadarBook; RadarSimPy `processing.py`; RadarSim CFAR variants; pyAPRiL `detector`/`hitProcessor` (structure only) |
| `core/clutter.py` | pyAPRiL `clutterCancellation` (reimplemented from papers); RadarSim land/sea clutter |
| `pipelines/tracking.py` | motpy's `step(detections)` loop shape; Stone Soup's detection → gate → associate → update decomposition |
| `viz/scopes/track_plot.py` | Tracktable's plan-view track rendering (prior art only; not a dependency) |
| `core/tracking/` (all) | `unified-extensible-tracker`, contributed under MIT (`tracker-001.md` §1.2); Stone Soup's data model and seams |
| `core/tracking/estimation.py`, `kalman.py`, `ukf.py` | FilterPy filter formulation and `UnscentedKalmanFilter` shape; Wan & van der Merwe (2000) for the UKF; Bar-Shalom, Li & Kirubarajan (2001); RadarBook tracking-filter chapters |
| `core/tracking/motion.py` | FilterPy `Q_discrete_white_noise`; Bar-Shalom, Li & Kirubarajan (2001) DWNA model |
| `core/tracking/association.py` | Stone Soup gater and data-associator seams; `scipy.optimize.linear_sum_assignment` (Crouse 2016) |
| `core/tracking/initiation.py`, `lifecycle.py`, `tracks.py` | Bar-Shalom & Li (1995) logic-based track formation; Stone Soup initiator/deleter split; RadarSim track management |
| `core/tracking/tracker.py` | motpy's `step(detections)` loop shape; Stone Soup's tracker |
| `array/*` | Phased-Array-Antenna-Model (near one-to-one module split) |
| `array/doa.py` | pyroomacoustics `doa` base-class pattern; RadarSimPy and pyroomacoustics estimators as benchmarks |
| `raytracing/base.py`, `scene.py` | RF-Genesis pipeline staging; RadarSimPy scene/mesh API |
| `raytracing/backends/mitsuba.py` | RF-Genesis `genesis/` Mitsuba + Dr.Jit usage |
| `raytracing/backends/ovrtx.py` | ovrtx sensor-simulation API |
| `pipelines/exporters/*` | FMCW Radar Target Simulator `JSONCoco.py` and its label schema |
| `pipelines/scenarios.py` | RadarSim YAML scenario files (radar-forge uses TOML: `CLAUDE.md` requires it) |
| `pipelines/datasets.py` | AIRadarLib PyTorch dataset/training wrappers; torchcvnn `datasets`/`transforms` (complex SAR loader layout); Steinmetz Neural Networks (complex-valued I/Q feature convention) |
| `viz/scopes/*`, `viz/app.py` | RadarSim PySide6 GUI (PPI, RHI, A-Scope) |
| `notebooks/` (top level) | RadarBook `jupyter/`; RadarSimNb |

### B.2 Design rules

1. **Core is dependency-light.** `radar_forge.core` and `radar_forge.array` require only NumPy and SciPy.
   Everything heavier lives behind an extra.
2. **Optional never breaks import.** `import radar_forge` must succeed with zero extras installed.
   Backends register lazily; a missing backend raises a descriptive `BackendUnavailableError` when
   *selected*, not when imported.
3. **One scene, many backends.** `raytracing.Scene` is backend-neutral; swapping `analytic` for
   `mitsuba` changes fidelity and runtime, not user code.
4. **Licence hygiene is a design constraint.** GPL and unlicensed references are reimplemented from
   published equations, with the source cited in the module docstring. See §A.14.
5. **Every block is state of the art and simple** ([`starter.md` §1.1](starter.md#11-the-bar)).
   The current best published method is the end state; a classical method stays only as a
   baseline or a test oracle. The code reads top to bottom for a new engineer, exposes its
   intermediate quantities (NIS, SNR budget terms, losses), pairs with a notebook in
   `notebooks/`, and is tested against analytic ground truth or a published value.
6. **A block that outgrows one module becomes a subpackage**, re-exported from its
   `__init__.py` so the public import path never changes. `core/tracking/` is the first (D2).

### B.3 Proposed extras

| Extra | Pulls in | Enables |
| :--- | :--- | :--- |
| *(none)* | numpy, scipy | `core`, `array`, `raytracing.backends.analytic` |
| `raytracing` | mitsuba, drjit | Mitsuba backend |
| `radarsimpy` | radarsimpy | RadarSimPy backend (note: GPL-3.0 — user-installed) |
| `ovrtx` | ovrtx, ovstage | Omniverse RTX backend (NVIDIA licence, RTX GPU) |
| `ml` | torch | `pipelines.datasets`, training loops |
| `cvnn` | torch, torchcvnn, complexPyTorch | complex-valued layers and SAR dataset loaders for `pipelines.datasets` |
| `viz` | matplotlib | `viz` plots and scopes |
| `notebooks` | jupyterlab | the top-level `notebooks/` |
| `gui` | PySide6 | `viz/app.py` live scopes (added when `app.py` lands) |
| `dev` | pytest, ruff, mypy, build tooling | development |

---

## Part A — Comparative module map

The survey behind Part B. Each entry follows the same shape: a link, the language and licence, a
table of the project's own modules and what each is responsible for, and a closing paragraph on
what `radar-forge` borrows from it. Nothing here is vendored; §A.14 is the licence matrix that says
why.

| § | Project | Licence | What radar-forge takes from it |
| :--- | :--- | :--- | :--- |
| [A.1](#a1-radarsimpy) | RadarSimPy | GPL-3.0 | Transmitter/Receiver/Radar composition; an arm's-length ray-tracing backend |
| [A.2](#a2-rf-genesis) | RF-Genesis | MIT | GPU ray-tracing pipeline staging for `raytracing/` |
| [A.3](#a3-phased-array-antenna-model) | Phased-Array-Antenna-Model | MIT | Near one-to-one module split for `array/` |
| [A.4](#a4-fmcw-radar-target-simulator) | FMCW Radar Target Simulator | MIT (MATLAB) | Baseband synthesis semantics and the COCO label schema |
| [A.5](#a5-radarbook-software) | RadarBook Software | none declared | Canonical DSP blocks, written from the published equations (D4) |
| [A.6](#a6-pyapril) | pyAPRiL | GPL-3.0 | Clutter cancellation and bistatic processing structure, reimplemented |
| [A.7](#a7-raspnet-dataset) | RASPNet | none declared | ML dataset conventions; an evaluation benchmark |
| [A.8](#a8-radarsim-gui) | RadarSim (GUI) | MIT | Blueprint for `viz/scopes/`; the scenario-file idea |
| [A.9](#a9-airadarlib-pypi-optional) | AIRadarLib | unstated | Chirp generation and PyTorch dataset wrappers, as reference |
| [A.10](#a10-ovrtx-pypi-optional) | ovrtx | NVIDIA proprietary | An opt-in, hardware-gated backend (D3) |
| [A.11](#a11-pyroomacoustics) | pyroomacoustics | MIT | The DoA estimator base-class pattern and DoA benchmarks |
| [A.12](#a12-complex-valued-neural-networks) | torchcvnn, complexPyTorch, Steinmetz | MIT / undeclared | The complex-valued I/Q feature convention `pipelines/datasets.py` exports against |
| [A.13](#a13-tracking-and-data-fusion) | Stone Soup, FilterPy, motpy, Tracktable, labeledRFS | MIT / BSD-3 | The comparative map for `core/tracking/` and `tracker-001.md` |
| [A.14](#a14-licence-summary) | *(licence summary)* | — | The matrix that governs every row above |

### A.1 RadarSimPy

- **Link:** https://github.com/radarsimx/radarsimpy
- **Language / licence:** Python + C++ backend · GPL-3.0

| Module | Responsibility |
| :--- | :--- |
| `radar.py` | `Transmitter` (CW / FMCW / PMCW / pulse waveforms, modulation, TX array), `Receiver` (sampling, noise figure, range gating), `Radar` (composed system, motion) |
| `simulator` | Compiled binaries: baseband simulation from point targets and 3D meshes, ray-tracing, interference, phase noise |
| `processing.py` | Range/Doppler processing, CFAR, beamforming, DoA (MUSIC, Root-MUSIC, ESPRIT, IAA, Capon, Bartlett) |
| `tools` | RCS characterisation, Swerling detection probability |

**What radar-forge borrows:** the `Transmitter` / `Receiver` / `Radar` composition is the cleanest
sensor-description API in the ecosystem and is the model for `core/radar.py`. GPL-3.0 means the code is
**not** vendored — RadarSimPy is wrapped behind an optional backend in `raytracing/backends/`, where the
dependency stays at arm's length.

### A.2 RF-Genesis

- **Link:** https://github.com/Asixa/RF-Genesis
- **Language / licence:** Python 3.10 (CUDA required) · MIT

| Directory | Responsibility |
| :--- | :--- |
| `genesis/` | The core pipeline: text-to-environment, text-to-motion (diffusion/MDM), Mitsuba ray-tracing, radar signal synthesis, point-cloud processing |
| `models/` | Radar configurations (TI AWR1843, 3TX/4RX MIMO) and model weights, incl. the RFLoRA adapter |
| `docs/` | Setup and pipeline documentation |

**What radar-forge borrows:** the staged pipeline shape — *scene → motion → propagation → signal →
product* — is the template for `pipelines/`. Its Mitsuba/Dr.Jit usage informs
`raytracing/backends/mitsuba.py`. Hard CUDA dependency keeps this an optional extra.

### A.3 Phased-Array-Antenna-Model

- **Link:** https://github.com/jman4162/Phased-Array-Antenna-Model
- **Language / licence:** Python (vectorized NumPy) · MIT

| Module | Responsibility |
| :--- | :--- |
| `core.py` | Vectorized array factor, FFT-based patterns, steering vectors, element patterns |
| `geometry.py` | Rectangular, triangular, circular, cylindrical, spherical, sparse/thinned arrays; subarrays |
| `beamforming.py` | Amplitude tapering, null steering, multi-beam, adaptive weights |
| `impairments.py` | Mutual coupling, phase quantization, element failure, scan blindness |
| `wideband.py` | True-time-delay steering, beam squint, bandwidth effects |
| `polarization.py`, `vector_patterns.py` | Jones/Stokes, axial ratio, XPD, Ludwig-3; co/cross-pol vector array factors |
| `coordinates.py`, `export.py`, `visualization.py`, `utils.py` | Frame transforms, CSV/JSON/NumPy export, 2D/3D/UV plotting |

**What radar-forge borrows:** the most direct model in the survey. MIT licence and a permissive,
NumPy-only design make its module split the blueprint for `array/`, essentially one-to-one.

### A.4 FMCW Radar Target Simulator

- **Link:** https://github.com/thomaswengerter/FMCW_Radar_Target_Simulator
- **Language / licence:** MATLAB (Phased Array System Toolbox) + Python helpers · MIT
- **Note:** an earlier draft of [`starter.md`](starter.md) credited this to Fraunhofer FHR. It is an
  independent project by Thomas Wengerter; FHR's ATRIUM is a separate hardware-in-the-loop simulator.
  `starter.md` has been corrected.

| File | Responsibility |
| :--- | :--- |
| `SimulateTargetList.m` | Drives sequences of multi-target urban traffic scenarios |
| `TargetSimulation.m` | Single-target measurement |
| `FMCWradar.m` | Radar device class: frequency, chirp shape, antenna properties |
| `TrajectoryPlanner.m` | Target motion paths |
| `modelBasebandSignal.m` | Per-reflector baseband signal, superposed with clutter and noise |
| `JSONCoco.py`, `JSONCoco_3SeqRGB.py` | Range-Doppler-azimuth cube → COCO annotations; 3-measurement RGB overlay |

Target models: pedestrian (`backscatterPedestrian`), bicyclist, car, point reflector. Labels carry
`range, velocity, azimuth, radar velocity, x, y, width, height, heading` (+ `obstruction` in multi-target).

**What radar-forge borrows:** the annotation schema and the cube→COCO conversion are the functional
specification for `pipelines/exporters/`. Being MATLAB, nothing is reused directly — the label format is
what transfers.

### A.5 RadarBook Software

- **Link:** https://github.com/RadarBook/software
- **Language / licence:** Python + MATLAB · **no LICENSE file in the repository** — treat as
  all-rights-reserved; use as a textbook reference only, do not copy code.

| Directory | Responsibility |
| :--- | :--- |
| `pyradar/` | Chapter-organised Python library over SciPy/NumPy/Matplotlib: radar range equation, waveforms and ambiguity functions, detection, tracking filters, SAR imaging, RCS |
| `mlradar/` | Machine-learning radar examples |
| `jupyter/` | Notebook walkthroughs per chapter |
| `docs/` | Book-companion documentation |

**What radar-forge borrows:** the canonical formulations for `core/` (range equation, ambiguity function,
CFAR, SAR primitives) and the chapter→notebook pedagogy for `notebooks/`. Implementations are
written from the published equations, not transcribed.

### A.6 pyAPRiL

- **Link:** https://github.com/pyapril/pyapril
- **Language / licence:** Python · **GPL-3.0** (copyleft)

| Module | Responsibility |
| :--- | :--- |
| `channelPreparation` | Reference-signal regeneration, clutter-cancellation wrappers |
| `clutterCancellation` | Time domain: Wiener-SMI, Wiener-SMI-MRE, ECA variants. Space domain: Max-SIR, MVDR, eigenvalue beamformer, beamsteering |
| `detector` | Cross-correlation detection (time, frequency, batched), Doppler windowing |
| `hitProcessor` | CA-CFAR and target DoA estimation on detections |
| `metricExtract` | Clutter attenuation, noise-floor reduction, SINR, dynamic-range compression |
| `targetParameterCalculator`, `targetLocalization` | Expected range/Doppler from geodetic data; localisation |
| `RDTools`, `sim` | Range-Doppler matrix plotting; FM-based simulator |

**What radar-forge borrows:** algorithm *structure* only. GPL-3.0 is incompatible with radar-forge's MIT
licence, so ECA and Wiener-SMI clutter cancellation are reimplemented from the published papers for
`core/clutter.py`. No pyAPRiL code enters the tree, and it is not a runtime dependency.

### A.7 RASPNet *(dataset)*

- **Link:** https://github.com/shyamven/RASPNet
- **Language / licence:** Python (examples) · **no LICENSE file in the repository** — treat as
  all-rights-reserved. Dataset itself is distributed via the AFRL SDMS portal under its own terms.

| Area | Responsibility |
| :--- | :--- |
| Processed datasets | Feature-label pairs (EXAMPLES and CVNN variants) over several numbered airborne-radar scenarios |
| `examples/` | Radar target localization models; transfer learning from pre-trained baselines |
| Data access | Downloaded separately from the AFRL SDMS collection portal, not bundled with the repo |

**What radar-forge borrows:** the *dataset conventions* only — feature/label pair layout, scenario
indexing, and the target-localization and transfer-learning task definitions, as a benchmark to
check `pipelines/` exporters against. No code enters the tree and no RASPNet data is committed;
it is not a runtime dependency. The complex-valued model trained on this dataset — Steinmetz Neural
Networks, by the same author — is surveyed in §A.12.

### A.8 RadarSim (GUI)

- **Link:** https://github.com/SpaceEngineerSS/RadarSim
- **Language / licence:** Python + PySide6 (Qt6) · MIT

| Area | Responsibility |
| :--- | :--- |
| `src/` physics | Radar equation, Swerling models, atmospheric absorption and rain attenuation, land/sea clutter, noise figure |
| `src/` signal processing | LFM pulse generation, complex-IQ echoes, matched filtering, MTI, Doppler FFT, CA/GO/SO/OS-CFAR with Pfa calibration |
| `src/` tracking | Kalman and extended Kalman filters, gating and assignment, multi-sensor track fusion, covariance intersection |
| `src/` imaging | Stripmap SAR raw data + range-Doppler focusing; ISAR with profile alignment |
| GUI | PPI, RHI, A-Scope, 3D tactical view, SAR/ISAR products, recording playback |
| `scenarios/` | YAML scenario configuration files |

**What radar-forge borrows:** the scope layout and the PySide6 real-time-display architecture for
`viz/`, and the YAML-scenario idea for `pipelines/scenarios.py`. Its CFAR-variant and tracking
coverage is a useful checklist for `core/`.

### A.9 AIRadarLib *(PyPI, optional)*

- **Link:** https://pypi.org/project/AIRadarLib/ · v0.1.0 (2025-06-29) · licence unstated

| Module | Responsibility |
| :--- | :--- |
| `AIradar_datasetv4.py` | Synthetic radar scene/dataset generation with configurable parameters |
| `AIradar_processing.py` | Range-Doppler processing for FMCW, OFDM, sine and hybrid waveforms; FFT + CFAR |
| `waveform_utils.py` | `generate_adf4159_fmcw_chirp` — realistic chirps with PLL quantization and phase noise |
| `AIradar_transformer.py` | Transformer over time-domain radar data, learnable FFT / CNN backbones |
| `*_train.py`, `*_comparison.py` | PyTorch training and model comparison utilities |

**What radar-forge borrows:** the hardware-realistic chirp model (PLL quantization, phase noise) for
`core/waveforms.py`, and the PyTorch dataset-wrapper pattern for `pipelines/datasets.py`. Unstated
licence ⇒ reference only, never a dependency.

### A.10 ovrtx *(PyPI, optional)*

- **Link:** https://github.com/NVIDIA-Omniverse/ovrtx · https://pypi.org/project/ovrtx/
- **Language / licence:** C + Python · **proprietary** (NVIDIA Software Licence Agreement)
- **Requirements:** RTX-capable NVIDIA GPU; Windows x86_64 / Linux x86_64 / Linux aarch64; Python 3.10–3.13

Exposes physically accurate real-time camera, lidar and radar sensor simulation over Omniverse RTX.

**What radar-forge borrows:** nothing structural — it is a highest-fidelity optional backend behind
`raytracing/backends/ovrtx.py`, gated on hardware and licence acceptance, never installed by default and
never imported at package import time.

### A.11 pyroomacoustics

- **Link:** https://github.com/LCAV/pyroomacoustics
- **Language / licence:** Python · MIT (EPFL-LCAV)

| Subpackage | Responsibility |
| :--- | :--- |
| `doa` | Direction-of-arrival estimators (MUSIC, SRP-PHAT, CSSM, WAVES, TOPS, FRIDA) behind a common base class |
| `beamforming` | Microphone array geometry and beamformer weights (delay-and-sum, MVDR) |
| `transform` | Block and streaming STFT |
| `adaptive`, `bss`, `denoise` | NLMS/RLS, AuxIVA/ILRMA/FastMNMF, spectral subtraction and Wiener |

**What radar-forge borrows:** the `doa` subpackage's pluggable-estimator base class is the design
pattern for `array/doa.py`, and its estimators serve as the cross-check benchmark for radar-forge's own
MUSIC/ESPRIT. MIT licence permits direct adaptation with attribution.

### A.12 Complex-valued neural networks

Radar data is natively complex: every stage from baseband I/Q through the range-Doppler cube carries
amplitude *and* phase. The design question these three projects answer is what a network does with
that — keep the signal complex end to end, or flatten it into two real channels and hope the network
relearns the coupling. The first two are general-purpose libraries; the third is a specific
architecture, and the reason this section exists at all.

**A.12a torchcvnn**

- **Link:** https://github.com/torchcvnn/torchcvnn · PyPI `torchcvnn`
- **Language / licence:** Python (PyTorch) · **MIT**
- **Origin:** CentraleSupélec — V. Dhédin, J. Levi, J. Fix, Q. Gabot et al. (IJCNN '25)

| Module | Responsibility |
| :--- | :--- |
| `torchcvnn.nn` | Complex-valued layers, activations and initialisers, including the ones needing genuinely complex implementations rather than a real-valued layer applied twice |
| `torchcvnn.datasets` | Complex-valued SAR and radar loaders: `MSTAR`, `SAMPLE`, `PolSF`, `ALOS2`, `SLC` (UAVSAR), `S1SLC`, `Bretigny`, `ATRNet-STAR`; plus `MICCAI2023` MRI k-space |
| `torchcvnn.transforms` | Amplitude/phase handling, FFT, crop and resize, `LogAmplitude`, `RandomPhase` |
| `examples/` | End-to-end training scripts over the above |

**What radar-forge borrows:** primarily `datasets` and `transforms` — the layout its SAR loaders
present to PyTorch is the reference `pipelines/datasets.py` should match, so that a radar-forge
export drops into an existing complex-valued training loop unchanged. The `nn` layer API is the
model for anything complex-valued radar-forge later wraps. MIT ⇒ this may be a real optional
dependency behind the `cvnn` extra (§B.3) and is adaptable with attribution.

**A.12b complexPyTorch**

- **Link:** https://github.com/wavefrontshaping/complexPyTorch · PyPI `complexPyTorch`
- **Language / licence:** Python (PyTorch) · **MIT**
- **Paper:** follows C. Trabelsi et al., *Deep Complex Networks*, ICLR 2018.

| Area | Responsibility |
| :--- | :--- |
| Layers | `ComplexLinear`, `ComplexConv2d`, `ComplexConvTranspose2d`, complex max/avg pooling, `ComplexDropout2d`, complex GRU and BN-GRU cells |
| Activations | ℂReLU, complex sigmoid and tanh |
| Normalisation | Complex batch norm (1d/2d) in two forms: the covariance method of Trabelsi et al., and a naive per-part method that is faster and often comparable |

**What radar-forge borrows:** the complex batch-norm formulation and the minimal layer set. Older
and smaller than A.12a, and carried mainly because A.12c is built on it — but MIT, so equally
available behind the `cvnn` extra.

**A.12c Steinmetz Neural Networks**

- **Link:** https://github.com/shyamven/SteinmetzNeuralNetworks
- **Language / licence:** Python (PyTorch, on `complexPyTorch`) · **no LICENSE file in the
  repository** — treat as all-rights-reserved, the same footing as A.7.
- **Paper:** S. Venkatasubramanian, A. Pezeshki and V. Tarokh, *Steinmetz Neural Networks for
  Complex-Valued Data*, AISTATS '25.

| Area | Responsibility |
| :--- | :--- |
| `models/` | Steinmetz and analytic-signal network definitions: complex-valued layers carrying I/Q as an analytic pair, with a consistency penalty on the Steinmetz decomposition |
| `utils/` | Training, evaluation and complex-tensor helpers |
| `main.py` | Driver for the classification and regression tasks |
| `RASPNet.ipynb` | Complex-valued regression on the §A.7 dataset — dataset and model as one benchmark |
| `FSDD.ipynb` | Complex-valued regression on spoken digits; the non-radar comparison task |

**What radar-forge borrows:** the architectural idea — an analytic-signal representation held
together by a consistency penalty on the Steinmetz decomposition, rather than two unconstrained real
channels — and, concretely, the complex-valued feature convention that follows from it, which is the
contract `pipelines/datasets.py` exports against. Its `RASPNet.ipynb` is the bridge back to §A.7:
the two together are the dataset-plus-model benchmark for the `pipelines/` exporters. Licence
undeclared ⇒ no code enters the tree and it is never a runtime dependency.

### A.13 Tracking and data fusion

`core/tracking/` has no upstream in §A.1–A.12 beyond two textbook companions. These five
projects are its comparative map; its scope — KF, EKF and UKF behind one interface, GNN and NN
assignment, and extension points for IMM and JPDA — is fixed by
[`tracker-001.md` §1.1](tracker-001.md#11-scope). None is vendored and none is a runtime
dependency: `scipy.optimize.linear_sum_assignment` covers assignment, and `scipy` is already in
core.

**A.13a Stone Soup**

- **Link:** https://github.com/dstl/Stone-Soup
- **Language / licence:** Python (NumPy/SciPy) · MIT (Dstl)
- **Paper:** P. A. Thomas, J. Barr, B. Balaji and K. White, *An open source framework for tracking
  and state estimation ("Stone Soup")*, Proc. SPIE 10200, 2017.

| Module | Responsibility |
| :--- | :--- |
| `types/` | The data model: `State`, `Detection`, `Track`, `Hypothesis`, `GaussianState` |
| `predictor/`, `updater/` | Kalman, extended, unscented, particle and information filters, split predict/update |
| `hypothesiser/`, `gater/` | Distance and probability hypothesisers; distance, elliptical and filtered gating |
| `dataassociator/` | Nearest neighbour, global nearest neighbour, PDA, JPDA, multi-hypothesis |
| `initiator/`, `deleter/` | M-of-N and single-point track initiation; time-based and covariance-based deletion |
| `models/` | Transition (constant velocity/acceleration, singer) and measurement models, incl. IMM |
| `metricgenerator/` | OSPA, GOSPA, SIAP — the standard tracking performance metrics |

**What radar-forge borrows:** the *vocabulary and the seams*, not the code. `core/tracking/`
follows the detection → hypothesiser → gater → associator → updater → initiator/deleter
decomposition one module per seam ([`tracker-001.md` §3](tracker-001.md#3-architecture-strict-decoupling)),
cut coarser in two places: predictor and updater are one `Estimator`, and the hypothesiser is the
gate. Stone Soup is a framework for comparing trackers; radar-forge has one tracker, each piece
small enough to read on its own, so a student who moves on to Stone Soup recognises the cut. Its
metric generators are the reference for acceptance criteria beyond
`spec/scenario-003-tracking.md` §12, and its JPDA and IMM implementations are the references for
the extension points in tracker-001 [§7.4](tracker-001.md#74-extension-point-jpda) and
[§9](tracker-001.md#9-multiple-models-imm). MIT, so code could be borrowed with attribution; the
reason not to is size and readability, not licence.

**A.13b FilterPy**

- **Link:** https://github.com/rlabbe/filterpy
- **Language / licence:** Python (NumPy/SciPy) · MIT
- **Companion text:** R. R. Labbe, *Kalman and Bayesian Filters in Python* — a worked,
  executable derivation of every filter in the library.

| Module | Responsibility |
| :--- | :--- |
| `kalman/` | `KalmanFilter`, `ExtendedKalmanFilter`, `UnscentedKalmanFilter`, IMM, fixed-lag smoothers |
| `common/` | `Q_discrete_white_noise`, `Q_continuous_white_noise`, `Saver`, van Loan discretisation |
| `gh/`, `hinfinity/`, `monte_carlo/` | g-h filters, H-infinity filters, particle-filter resampling |
| `stats/` | NEES/NIS consistency statistics, covariance ellipses, plotting helpers |

**What radar-forge borrows:** the *formulation*. `Q_discrete_white_noise` is the exact
discrete-white-noise-acceleration construction `core/tracking/motion.py` reimplements from
reference [2] of the scenario spec; FilterPy's `log_likelihood`/NIS bookkeeping is the model for
`core/tracking/estimation.py`, whose NIS scenario 003 §12 asserts on; and `core/tracking/ukf.py`
keeps `UnscentedKalmanFilter`'s shape, one object holding the estimate and changing it in place.
It is a reference to reimplement rather than a dependency: written from the cited papers, each
equation stays visible in a docstring a reader can check, and carrying a filter library to get
them would fail `CLAUDE.md`'s test for adding a dependency.

**A.13c motpy**

- **Link:** https://github.com/wmuron/motpy
- **Language / licence:** Python · MIT

| Module | Responsibility |
| :--- | :--- |
| `tracker.py` | `MultiObjectTracker`: the whole predict → match → update → prune loop in one file |
| `core.py` | `Box`, `Detection`, `Track` value types |
| `metrics.py` | IoU and Euclidean cost matrices for the assignment step |
| `model.py` | Constant-velocity and constant-acceleration motion models with a single order knob |

**What radar-forge borrows:** the *loop shape*. motpy is tracking-by-detection with Hungarian
matching and a staleness-based track manager, complete, in roughly one module. Its
`MultiObjectTracker.step(detections) -> tracks` signature is the shape
`core/tracking/tracker.py`'s `Tracker.process(batch)` follows: one scan in, the current tracks
out. It is a computer-vision tracker, so nothing about its cost metrics or box model transfers;
the loop structure is the whole borrowing.

**A.13d Tracktable**

- **Link:** https://github.com/sandialabs/tracktable
- **Language / licence:** C++ core with Python bindings · BSD-3-Clause (Sandia National
  Laboratories, DOE contract DE-NA0003525)

| Module | Responsibility |
| :--- | :--- |
| `domain/` | Terrestrial, Cartesian-2D and Cartesian-3D coordinate domains with unit-aware points |
| `core/` | `Trajectory` and `TrajectoryPoint` containers, timestamped and property-annotated |
| `analysis/` | Assembly of points into trajectories, DBSCAN clustering, R-tree spatial indexing, distance geometry |
| `render/` | Cartopy/matplotlib map rendering of trajectories and heatmaps |
| `applications/` | Trajectory assembly, filtering and rendering as command-line tools |

**What radar-forge borrows:** the *trajectory data model*, as prior art for
`pipelines/trajectories.py` and for `viz/scopes/track_plot.py` — in particular the separation
of a coordinate domain from the trajectory container, which is what lets the same analysis run in
geodetic and Cartesian frames. It is explicitly **not** a dependency: radar-forge needs one
plan-view plot, matplotlib already draws it, and a C++/Boost build to draw it would fail
`CLAUDE.md`'s dependency test. If large-scale trajectory analytics ever become a goal of
`pipelines/`, the BSD licence makes it the first thing to reach for.

**A.13e labeledRFS and VisualRFS**

- **Link:** https://github.com/linh-gist/labeledRFS and https://github.com/linh-gist/VisualRFS
- **Language / licence:** Python (labeledRFS, ported from Ba-Tuong Vo's MATLAB) and C++/Python
  (VisualRFS) · both MIT. **The original MATLAB sources on Prof. Vo's site carry their own terms;
  the papers, not the ports, are the reference of record.**
- **Papers:** B.-T. Vo and B.-N. Vo, *Labeled random finite sets and multi-object conjugate
  priors*, IEEE TSP 61(13), 2013; B.-N. Vo, B.-T. Vo and H. G. Hoang, *An efficient
  implementation of the generalized labeled multi-Bernoulli filter*, IEEE TSP 65(8), 2017.

| Module | Responsibility |
| :--- | :--- |
| `GLMB`, `LMB` | Generalized labeled multi-Bernoulli and labeled multi-Bernoulli filters with joint prediction and update |
| `gms`, `jointpredictupdate` | Gaussian-mixture components; the joint step that removes the separate gating stage |
| `assignment/` | Murty's k-best and Gibbs-sampled ranked assignment |
| `ospa`, `run_filter` | OSPA/OSPA(2) error metrics and the scenario drivers |

**What radar-forge borrows:** nothing yet, by design — it is the **target state** for high-clutter
multi-target tracking, recorded here so that the extension has a reference rather than an
improvisation. The relevant idea is that a random-finite-set filter propagates a distribution over
*sets* of targets and so needs no heuristic gate, no M-of-N initiator and no deleter: exactly the
three components `spec/scenario-003-tracking.md` §7 has to size by hand and defend in
§3. That contrast is worth teaching even before the filter is implemented. Under the bar of
[`starter.md` §1.1](starter.md#11-the-bar), random-finite-set filters (GLMB, PMBM) are the
state of the art for high-clutter multi-target tracking, and the GNN tracker is the baseline
they are measured against. [`tracker-001.md` §7.5](tracker-001.md#75-out-of-scope-pmbm) records
why they are out of scope for now: implementation is gated on a multi-target, high-clutter
scenario existing to justify it, and would be written from the papers.

### A.14 Licence summary

| Project | Licence | Usable as dependency? | Usable as code source? |
| :--- | :--- | :--- | :--- |
| Phased-Array-Antenna-Model | MIT | yes | yes, with attribution |
| Stone Soup | MIT | yes | yes, with attribution — not taken, size not licence |
| FilterPy | MIT | yes | yes, with attribution — reimplemented instead, from the cited papers |
| motpy | MIT | no (computer-vision domain) | loop structure only |
| Tracktable | BSD-3-Clause | yes, if trajectory analytics is ever a goal | yes, with attribution |
| labeledRFS / VisualRFS | MIT (ports); original MATLAB terms differ | no | **no** — write from the papers |
| pyroomacoustics | MIT | yes | yes, with attribution |
| RF-Genesis | MIT | optional extra | yes, with attribution |
| RadarSim (GUI) | MIT | reference | yes, with attribution |
| FMCW Radar Target Simulator | MIT | no (MATLAB) | format spec only |
| RadarSimPy | GPL-3.0 | optional extra only, arm's length | **no** |
| pyAPRiL | GPL-3.0 | **no** | **no** — reimplement from papers |
| RASPNet | none declared | no (data via SDMS portal) | **no** — dataset conventions only |
| torchcvnn | MIT | yes (extra: `cvnn`) | yes, with attribution |
| complexPyTorch | MIT | yes (extra: `cvnn`) | yes, with attribution |
| Steinmetz Neural Networks | none declared | no | **no** — architectural reference only |
| RadarBook Software | none declared | no | **no** — textbook reference only |
| AIRadarLib | unstated | no | **no** — reference only |
| ovrtx | NVIDIA proprietary | optional extra, user-accepted | **no** |

---

## Decisions

These were open questions on first draft; all are now settled. Recorded here with their rationale so the
reasoning is not lost when implementation starts.

### D1 — Backend contract returns propagation paths

`RayTracingBackend.trace(scene, radar)` returns a **path set**, not baseband:

```python
@dataclass
class PropagationPaths:
    delay: np.ndarray  # (n_paths,) seconds
    doppler: np.ndarray  # (n_paths,) Hz
    amplitude: np.ndarray  # (n_paths,) complex
    aoa: np.ndarray  # (n_paths, 2) az/el, radians
    aod: np.ndarray  # (n_paths, 2) az/el, radians
    bounce_count: np.ndarray
```

`core/signal.py` owns the path→baseband synthesis for every backend. This keeps waveform changes in one
place, makes backends comparable on identical geometry, and makes them testable without synthesising
signals. Backends that natively produce baseband (RadarSimPy, ovrtx) may additionally implement an
optional `trace_baseband()` fast path; `base.py` declares it, and `generate.py` prefers it only when the
requested waveform matches what the backend supports natively.

### D2 — Promote a `core/` module to a subpackage when it outgrows one module

**The rule.** A `core/` module becomes a subpackage once it carries more than roughly one
module's worth of responsibility, and not before. The subpackage re-exports every public name
from its `__init__.py`, so the public import path never changes (§B.2 rule 6). Do not pre-split;
do not let a file drift past its trigger.

**Tracking: promoted.** Scenario 3 first built `core/tracking.py` out as one module — three
state models, GNN association and M-of-N track management — and that did not fire the trigger:
per `spec/scenario-003-tracking.md` §6.2 the state models were data returned by a factory, not a
class hierarchy, and one association strategy was not two. The trigger was a second association
strategy or a second filter, and PR #2 (`91a3454`) brought both: the UKF, and nearest-neighbour
assignment beside GNN. `core/tracking.py` became the package `core/tracking/`, and the original
module survives as `core/tracking/kalman.py`. The split follows Stone Soup's seams, as specified
in [`tracker-001.md` §3](tracker-001.md#3-architecture-strict-decoupling), not the
`{filters,association,fusion}` this decision first sketched. Track-to-track fusion is out of
scope (tracker-001 §1.1); its design belongs to tracker-001 from here on.

**Clutter: stays a single module.** Promote `core/clutter.py` to `core/clutter/` when it carries
more than two clutter models plus the cancellation algorithms.

### D3 — ovrtx stays an opt-in, hardware-gated backend

Proprietary NVIDIA licence and an RTX GPU requirement. Never a default dependency, never imported at
package import time, and its absence must degrade to a descriptive `BackendUnavailableError`.

### D4 — Everything from RadarBook is written from the published equations

`RadarBook/software` has no LICENSE file (confirmed: no `LICENSE` at the repo root, null `spdx_id` from
the GitHub API), so it is all-rights-reserved by default. No code is transcribed. Each affected module in
`core/` cites the book chapter and equation number in its docstring, and its test asserts against a value
published in the book — facts and formulae are not copyrightable, the expression is.

### D5 — COCO label-schema parity is the bar; no MATLAB harness

**The decision:** radar-forge matches the FMCW Radar Target Simulator's *label schema and cube
conventions* exactly. It does not attempt bit-level agreement with MATLAB's simulation output, and no
MATLAB validation harness is built.

**Why this is enough.** The value of that project to radar-forge is interoperability — a model trained on
its output should train on radar-forge output unchanged, and vice versa. That is entirely a function of
the annotation contract, not of the physics matching. Bit-level parity would in any case be unreachable:
it depends on MATLAB's `backscatterPedestrian` scattering-centre model, Phased Array System Toolbox
internals, and its RNG stream — none of which are reproducible in NumPy, and two of which are closed
source. Chasing it would anchor radar-forge's physics to a toolbox the project deliberately does not
depend on.

**What parity concretely requires** — this is the acceptance checklist for `pipelines/exporters/`:

| Item | Contract |
| :--- | :--- |
| Label fields | `range, velocity, azimuth, radar velocity, x, y, width, height, heading`, plus `obstruction` for multi-target scenes — same names, same order, same units (m, m/s, degrees) |
| Cube axes | range × Doppler × azimuth, in that order |
| Cube scaling | decibel, matching the upstream dB reference |
| COCO reduction | azimuth collapsed by maximum across channels to form the 2D image plane |
| Sign conventions | closing velocity positive, azimuth zero at boresight and increasing clockwise |
| Sequence packing | the 3-measurement RGB overlay of `JSONCoco_3SeqRGB.py` available as an export option |
| Bounding boxes | COCO `[x, y, width, height]`, top-left origin, pixel coordinates in the reduced range-Doppler plane |

**How it is verified.** A golden-file test: check a small number of upstream-produced `.mat` cubes and
their COCO JSON into `tests/fixtures/`, and assert that radar-forge's exporter, handed the *same* cube
array, emits byte-comparable JSON. This tests the exporter — the part that must interoperate — without
testing the physics. A second test runs a radar-forge-generated cube through a standard COCO loader
(`pycocotools`) to confirm the output is schema-valid.

**What is deliberately not tested:** agreement between radar-forge's baseband and MATLAB's for the same
scenario. If a physics cross-check is ever wanted, the reference of choice is RadarSimPy or the RadarBook
worked examples — both Python, both already in the dependency story — not MATLAB.

### D6 — `PropagationPaths` covers bistatic geometry unchanged

Settled by `spec/scenario-002-bistatic.md`, which was the first slice to put the
transmitter and the receiver at different sites. D1's dataclass survives without an edit:

| Field | Monostatic reading | Bistatic reading |
| :--- | :--- | :--- |
| `delay_s` | `2R/c` | `(R_t + R_r)/c` |
| `range_m` property, `delay_s·c/2` | `R` | the **bistatic mean range** `(R_t + R_r)/2` |
| `doppler_hz` | `2v/λ` | `(Ṙ_t + Ṙ_r)/λ` — the same expression over the bisector range rate |

Both monostatic forms are the special case `R_t = R_r`. `range_tx_m` and `range_rx_m` are
deliberately **not** fields: the two ranges are geometry, and geometry belongs to `BistaticRadar`.
A path set records what a propagation model produced, not how the radar was arranged. This is the
strongest evidence so far that D1 was drawn in the right place, because it was drawn before the
bistatic case was considered.

### D7 — One siting-agnostic pipeline, via `RadarLike`

`RadarLike = Radar | BistaticRadar`. Consumers widen to the union rather than growing `bistatic_*`
twins: the signal generators and the range-Doppler processing depend only on waveform and receiver
attributes that both classes carry, so each needed a type widening and no new mathematics. In
`pipelines/scenarios.py` the siting is selected by the presence of a `[transmitter_site]` table, so
a monostatic scenario file is unchanged by the feature's existence.

### D8 — Bistatic RCS is taken as given, and angle-independent

`PointTarget` keeps a single `rcs_m2`, used as σ_b. The monostatic-equivalence theorem licenses
that only for smooth bodies at small bistatic angles away from resonance, and scenario 002 runs to
β = 129.3°, so **absolute** power in a bistatic scenario is an approximation. Its acceptance
criteria test range, velocity and resolution — geometry — and deliberately assert nothing about
absolute SNR. Forward scatter is not modelled at all.

---

## Open questions

1. **GPU story — deferred.** Three backends (Mitsuba, RadarSimPy, ovrtx) need CUDA or an RTX GPU. Whether
   CI exercises any of them, or backend tests stay mock-only with hardware validation left manual, is a
   question for when the first real backend lands. Until then `raytracing/backends/analytic.py` is the only
   backend under test, and it runs anywhere. Revisit before merging the first GPU backend.

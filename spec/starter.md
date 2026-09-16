# Repository Specification: `radar-forge`

## 1. Project Overview

| Field | Value |
| :--- | :--- |
| Project name | `radar-forge` |
| PyPI package | `radar-forge` (imports via `import radar_forge`) |
| Primary goal | An open-source, modular Python library and educational workbench combining 3D ray-tracing, phased array beamforming, tracking and machine-learning dataset synthesis |
| Target audience | Interns, students, researchers and algorithm developers in radar / wireless sensing |
| Python | >= 3.11, src layout |
| Configuration | TOML only, in `pyproject.toml`; scenarios in `scenarios/*.toml` |

**Companion documents.** [`structure.md`](structure.md) is the file-level design and the decision
list. The vertical slices through it are specified one document each:

| Slice | Specification | What it builds |
| :--- | :--- | :--- |
| Scenario 1 | [`scenario-001-xband.md`](scenario-001-xband.md) | trajectory -> IQ -> range-Doppler map, in three ambiguity variants |
| Scenario 2 | [`scenario-002-bistatic.md`](scenario-002-bistatic.md) | the same path over a two-site bistatic geometry |
| Scenario 3 | [`scenario-003-tracking.md`](scenario-003-tracking.md) | CFAR detection -> association -> Kalman tracking |

---

## 2. Directory Layout & Module Structure

The tree a reader should have in mind before meeting the dependency survey in section 3. The
authoritative, file-level version — every module, with the upstream it draws from — is Part B of
[`structure.md`](structure.md).

```text
radar-forge/
|-- pyproject.toml               # the single configuration file: deps, ruff, mypy, pytest
|-- README.md
|-- LICENSE
|-- Makefile                     # the single definition of every gate; `make check` is CI
|-- data/                        # tracked input data (the ADS-B-style flight track)
|-- scenarios/                   # TOML scenario configurations, one per variant
|-- scripts/                     # run_scenario.py, check_conventions.py, setup-dev.sh
|-- spec/                        # this document, structure.md, the scenario specifications
|-- docs/                        # developer & intern documentation
|   |-- conventions/             # commits, style, testing -- what the hooks enforce
|   `-- scenarios/               # visual companions to the scenario specifications
|-- tests/                       # pytest unit and integration tests, mirroring src/
`-- src/
    `-- radar_forge/             # core import package
        |-- __init__.py
        |-- core/                # radar equation, geodesy, waveforms, signal, dsp, detection, tracking
        |-- array/               # phased array pattern generation, tapering, null steering
        |-- raytracing/          # wrappers for Mitsuba/RadarSimPy ray-tracing backends
        |-- pipelines/           # scenario loop, trajectories, tracking, ML dataset exporters
        `-- teaching/            # interactive GUI scopes (PPI, A-Scope) and Jupyter notebooks
```

---

## 3. Dependencies & Ecosystem Integration

### 3.1 Related Open-Source GitHub Repositories

The codebase draws architectural inspiration from the following open-source frameworks. None of
them are vendored into the tree; where a project is available at runtime it is reached through an
optional, arm's-length backend. See [`structure.md`](structure.md) Part A for the per-project module
map and §A.14 for the licence compatibility matrix that governs what may be borrowed.

#### 3.1.1 Propagation and ray-tracing backends

What `raytracing/` wraps, and what `core/signal.py` produces baseband from.

| Project Name | Description | Reason for Inclusion / Core Utility |
| :--- | :--- | :--- |
| **RadarSimPy** | Python/C++ Ray-Tracing Radar Simulator | High-performance physical propagation and automotive RCS simulation backend. Wrapped at arm's length as an optional ray-tracing backend (GPL-3.0 — no code vendored). |
| **RF-Genesis / WiTwin Radar** | Differentiable mmWave radar simulator (UCSD) built on Mitsuba/Dr.Jit. | Provides zero-shot domain adaptation and synthetic generative dataset workflows. Used as a reference for GPU-accelerated ray-tracing pipelines. |
| **FMCW Radar Target Simulator** (Wengerter) | MATLAB/Phased Array System Toolbox simulator for urban traffic scenarios. | Provides functional specification for baseband signal exporting into standard deep learning annotation formats (e.g., COCO bounding boxes with range-Doppler cubes). |

#### 3.1.2 Phased arrays and beamforming

The comparative map for `array/`, the one subpackage with a near one-to-one upstream.

| Project Name | Description | Reason for Inclusion / Core Utility |
| :--- | :--- | :--- |
| **Phased-Array-Antenna-Model** | Vectorized 2D/3D radiation pattern computation library. | Fills the phased array gap in existing simulators. Supplies conformal array geometries, spatial tapering (Taylor/Chebyshev), phase quantization, and beamforming math. |

#### 3.1.3 DSP, detection and bistatic processing

Canonical blocks for `core/dsp.py`, `core/detection.py` and `core/clutter.py`. Per `structure.md` D4, RadarBook material is written from the published equations and never vendored.

| Project Name | Description | Reason for Inclusion / Core Utility |
| :--- | :--- | :--- |
| **RadarBook Software** | Companion code for *Introduction to Radar Using Python and MATLAB*. | Used as standard baseline reference for canonical DSP blocks (range FFT, Doppler FFT, CFAR, SAR primitives). |
| **pyapril** | Python library for passive radar signal processing (U. Budapest). | Reference implementation for space-time clutter cancellation (Wiener-SMI, ECA) and bistatic processing geometries. |

#### 3.1.4 Tracking and data fusion

The comparative map for `core/tracking.py`, built by scenario 3. None is a runtime dependency: `scipy.optimize.linear_sum_assignment` covers the only algorithm the scenario needs.

| Project Name | Description | Reason for Inclusion / Core Utility |
| :--- | :--- | :--- |
| **Stone Soup** (Dstl) | Target-tracking and state-estimation framework: predictors and updaters, gaters and hypothesisers, data associators (NN, GNN, PDA, JPDA, MHT), M-of-N initiators and deleters, OSPA/GOSPA metrics. | The architectural reference for `core/tracking.py` — the detection → gate → associate → update → initiate decomposition and the `Detection`/`Track`/`Hypothesis` vocabulary, reproduced at a fraction of the surface area. MIT; no code vendored, size rather than licence is the reason. |
| **FilterPy** | Kalman, extended, unscented and particle filters with the *Kalman and Bayesian Filters in Python* companion text. | Reference formulation for the two-state constant-velocity filter and the discrete-white-noise-acceleration process noise (`Q_discrete_white_noise`), plus the NIS consistency statistic the scenario acceptance criteria assert on. MIT; reimplemented in ~40 vectorised lines rather than carried as a dependency. |
| **motpy** | Minimal tracking-by-detection multi-object tracker: predict, Hungarian assignment, update, staleness-based pruning, in about one module. | Evidence for the decision in `structure.md` D2 to keep `core/tracking.py` a single module until a second association strategy arrives, and the shape reference for `TrackManager.step(detections)`. MIT; a computer-vision tracker, so only the loop structure transfers. |
| **Tracktable** (Sandia) | C++/Python moving-object trajectory analysis: unit-aware coordinate domains, trajectory assembly, R-tree spatial indexing, DBSCAN clustering, Cartopy map rendering. | Prior art for the trajectory data model in `pipelines/trajectories.py` and for plan-view track rendering, especially the separation of coordinate domain from trajectory container. BSD-3-Clause — usable, but deliberately **not** a dependency: one matplotlib plot does not justify a C++/Boost build. |
| **labeledRFS / VisualRFS** (Vo, ported by Linh Ma) | GLMB and LMB random-finite-set multi-target filters with joint predict-update, Gibbs-sampled ranked assignment and OSPA/OSPA(2) metrics. | The target state for high-clutter multi-target tracking: an RFS filter propagates a distribution over *sets* of targets and so needs no heuristic gate, no M-of-N initiation and no deletion logic — the three things scenario 003 must size by hand. Ports are MIT, the original MATLAB terms differ; written from the papers if implemented. |

#### 3.1.5 ML datasets and complex-valued networks

The conventions `pipelines/datasets.py` and `pipelines/exporters/` export against, behind the `ml` and `cvnn` extras.

| Project Name | Description | Reason for Inclusion / Core Utility |
| :--- | :--- | :--- |
| **RASPNet** (SDMS) | Benchmark dataset and example code for radar adaptive signal processing, spanning multiple scenarios of airborne radar returns. Data downloaded separately from the AFRL SDMS portal. | Reference for ML dataset conventions: feature-label pair layout, scenario indexing, and transfer-learning baselines for target localization. Evaluation benchmark for the `pipelines` dataset exporters; no code vendored, no data committed. |
| **torchcvnn** | Complex-valued neural network library for PyTorch (CentraleSupélec, IJCNN '25): complex layers, activations and transforms, plus SAR/radar dataset loaders (MSTAR, SAMPLE, PolSF, ALOS2, UAVSAR SLC, S1SLC). | Reference layout for how complex radar data is presented to PyTorch, so `pipelines` exports drop into an existing complex-valued training loop unchanged. MIT — the one CVNN project usable directly, as an optional `cvnn` extra. |
| **complexPyTorch** | Complex-valued layers, activations and batch normalisation for PyTorch, following Trabelsi et al. (ICLR 2018). | Reference for the complex batch-norm formulation and a minimal complex layer set; the library Steinmetz Neural Networks is built on. MIT — usable as an optional `cvnn` extra. |
| **Steinmetz Neural Networks** | Complex-valued architecture (AISTATS '25) by the RASPNet author, keeping I/Q as an analytic pair under a Steinmetz-decomposition consistency penalty; ships a worked RASPNet regression notebook. | The complex-valued I/Q feature convention `pipelines/datasets.py` exports against — features stay complex rather than split into two real channels — and, with RASPNet, a paired dataset-plus-model benchmark. Licence undeclared; no code vendored, never a dependency. |

#### 3.1.6 Teaching and visualisation

The blueprint for `teaching/scopes/` and the intern onboarding path.

| Project Name | Description | Reason for Inclusion / Core Utility |
| :--- | :--- | :--- |
| **RadarSim (GUI)** | Educational pulse-Doppler visualizer with real-time scopes. | Blueprint for interactive educational modules (A-Scope, B-Scope, PPI display scopes) for intern onboarding. |

---

### 3.2 PyPI Packages & Other Sources
The codebase is also inspired by the following PyPI packages:

| PyPI Package | Role & Description | Inclusion Purpose |
| :--- | :--- | :--- |
| **`AIRadarLib`** *(Optional)* | FMCW/OFDM radar simulation library | Reference for time-domain chirp generation and PyTorch transformer dataset wrappers. |
| **`ovrtx`** *(Optional)* | NVIDIA Omniverse RTX Sensor SDK | High-fidelity GPU point cloud / target map simulation reference. |
| **`pyroomacoustics`** | Spatial array processing & beamforming | Direction-of-Arrival (DoA) estimation benchmarks (MUSIC, ESPRIT). |

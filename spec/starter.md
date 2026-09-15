# Repository Specification: `radar-forge`

## 1. Project Overview
- **Project Name:** `radar-forge`
- **PyPI Package Name:** `radar-forge` (Imports via `import radar_forge`)
- **Primary Goal:** An open-source, modular Python library and educational workbench combining 3D ray-tracing, phased array beamforming, tracking and machine learning dataset synthesis for radar interns and academic collaborators.
- **Target Audience:** Interns, students, researchers, and algorithm developers in radar/wireless sensing.

---

## 2. Dependencies & Ecosystem Integration

### 2.1 Related Open-Source GitHub Repositories
The codebase draws architectural inspiration from the following open-source frameworks. None of them are vendored into the tree; where a project is available at runtime it is reached through an optional, arm's-length backend. See [`structure.md`](structure.md) for the per-project module map and the licence compatibility matrix that governs what may be borrowed.

| Project Name | Description | Reason for Inclusion / Core Utility |
| :--- | :--- | :--- |
| **RadarSimPy** | Python/C++ Ray-Tracing Radar Simulator | High-performance physical propagation and automotive RCS simulation backend. Wrapped at arm's length as an optional ray-tracing backend (GPL-3.0 — no code vendored). |
| **RF-Genesis / WiTwin Radar** | Differentiable mmWave radar simulator (UCSD) built on Mitsuba/Dr.Jit. | Provides zero-shot domain adaptation and synthetic generative dataset workflows. Used as a reference for GPU-accelerated ray-tracing pipelines. |
| **Phased-Array-Antenna-Model** | Vectorized 2D/3D radiation pattern computation library. | Fills the phased array gap in existing simulators. Supplies conformal array geometries, spatial tapering (Taylor/Chebyshev), phase quantization, and beamforming math. |
| **FMCW Radar Target Simulator** (Wengerter) | MATLAB/Phased Array System Toolbox simulator for urban traffic scenarios. | Provides functional specification for baseband signal exporting into standard deep learning annotation formats (e.g., COCO bounding boxes with range-Doppler cubes). |
| **RadarBook Software** | Companion code for *Introduction to Radar Using Python and MATLAB*. | Used as standard baseline reference for canonical DSP blocks (range FFT, Doppler FFT, CFAR, SAR primitives). |
| **RASPNet** (SDMS) | Benchmark dataset and example code for radar adaptive signal processing, spanning multiple scenarios of airborne radar returns. Data downloaded separately from the AFRL SDMS portal. | Reference for ML dataset conventions: feature-label pair layout, scenario indexing, and transfer-learning baselines for target localization. Evaluation benchmark for the `pipelines` dataset exporters; no code vendored, no data committed. |
| **torchcvnn** | Complex-valued neural network library for PyTorch (CentraleSupélec, IJCNN '25): complex layers, activations and transforms, plus SAR/radar dataset loaders (MSTAR, SAMPLE, PolSF, ALOS2, UAVSAR SLC, S1SLC). | Reference layout for how complex radar data is presented to PyTorch, so `pipelines` exports drop into an existing complex-valued training loop unchanged. MIT — the one CVNN project usable directly, as an optional `cvnn` extra. |
| **complexPyTorch** | Complex-valued layers, activations and batch normalisation for PyTorch, following Trabelsi et al. (ICLR 2018). | Reference for the complex batch-norm formulation and a minimal complex layer set; the library Steinmetz Neural Networks is built on. MIT — usable as an optional `cvnn` extra. |
| **Steinmetz Neural Networks** | Complex-valued architecture (AISTATS '25) by the RASPNet author, keeping I/Q as an analytic pair under a Steinmetz-decomposition consistency penalty; ships a worked RASPNet regression notebook. | The complex-valued I/Q feature convention `pipelines/datasets.py` exports against — features stay complex rather than split into two real channels — and, with RASPNet, a paired dataset-plus-model benchmark. Licence undeclared; no code vendored, never a dependency. |
| **pyapril** | Python library for passive radar signal processing (U. Budapest). | Reference implementation for space-time clutter cancellation (Wiener-SMI, ECA) and bistatic processing geometries. |
| **RadarSim (GUI)** | Educational pulse-Doppler visualizer with real-time scopes. | Blueprint for interactive educational modules (A-Scope, B-Scope, PPI display scopes) for intern onboarding. |
| **Stone Soup** (Dstl) | Target-tracking and state-estimation framework: predictors and updaters, gaters and hypothesisers, data associators (NN, GNN, PDA, JPDA, MHT), M-of-N initiators and deleters, OSPA/GOSPA metrics. | The architectural reference for `core/tracking.py` — the detection → gate → associate → update → initiate decomposition and the `Detection`/`Track`/`Hypothesis` vocabulary, reproduced at a fraction of the surface area. MIT; no code vendored, size rather than licence is the reason. |
| **FilterPy** | Kalman, extended, unscented and particle filters with the *Kalman and Bayesian Filters in Python* companion text. | Reference formulation for the two-state constant-velocity filter and the discrete-white-noise-acceleration process noise (`Q_discrete_white_noise`), plus the NIS consistency statistic the scenario acceptance criteria assert on. MIT; reimplemented in ~40 vectorised lines rather than carried as a dependency. |
| **motpy** | Minimal tracking-by-detection multi-object tracker: predict, Hungarian assignment, update, staleness-based pruning, in about one module. | Evidence for the decision in `structure.md` D2 to keep `core/tracking.py` a single module until a second association strategy arrives, and the shape reference for `TrackManager.step(detections)`. MIT; a computer-vision tracker, so only the loop structure transfers. |
| **Tracktable** (Sandia) | C++/Python moving-object trajectory analysis: unit-aware coordinate domains, trajectory assembly, R-tree spatial indexing, DBSCAN clustering, Cartopy map rendering. | Prior art for the trajectory data model in `pipelines/trajectories.py` and for plan-view track rendering, especially the separation of coordinate domain from trajectory container. BSD-3-Clause — usable, but deliberately **not** a dependency: one matplotlib plot does not justify a C++/Boost build. |
| **labeledRFS / VisualRFS** (Vo, ported by Linh Ma) | GLMB and LMB random-finite-set multi-target filters with joint predict-update, Gibbs-sampled ranked assignment and OSPA/OSPA(2) metrics. | The target state for high-clutter multi-target tracking: an RFS filter propagates a distribution over *sets* of targets and so needs no heuristic gate, no M-of-N initiation and no deletion logic — the three things scenario 003 must size by hand. Ports are MIT, the original MATLAB terms differ; written from the papers if implemented. |

### 2.2 PyPI Packages & Other Sources
The codebase is also inspired by the following PyPI packages:

| PyPI Package | Role & Description | Inclusion Purpose |
| :--- | :--- | :--- |
| **`AIRadarLib`** *(Optional)* | FMCW/OFDM radar simulation library | Reference for time-domain chirp generation and PyTorch transformer dataset wrappers. |
| **`ovrtx`** *(Optional)* | NVIDIA Omniverse RTX Sensor SDK | High-fidelity GPU point cloud / target map simulation reference. |
| **`pyroomacoustics`** | Spatial array processing & beamforming | Direction-of-Arrival (DoA) estimation benchmarks (MUSIC, ESPRIT). |

---

## 3. Directory Layout & Module Structure

```text
radar-forge/
├── pyproject.toml
├── README.md
├── LICENSE
├── docs/                       # Developer & intern documentation
├── tests/                      # Pytest unit and integration tests
└── src/
    └── radar_forge/            # Core import package
        ├── __init__.py
        ├── core/               # Radar equation, target models, baseband signal math
        ├── array/              # Phased array pattern generation, tapering, null steering
        ├── raytracing/         # Wrappers for Mitsuba/RadarSimPy ray-tracing backends
        ├── pipelines/          # ML dataset generation & COCO / range-Doppler exporter
        └── teaching/           # Interactive GUI scopes (PPI, A-Scope) and Jupyter notebooks

> **Note on attribution:** an earlier draft of this spec credited the FMCW Radar Target Simulator to
> Fraunhofer FHR. It is in fact an independent MATLAB project by Thomas Wengerter
> (<https://github.com/thomaswengerter/FMCW_Radar_Target_Simulator>, MIT). Fraunhofer FHR's ATRIUM is a
> separate, unrelated hardware-in-the-loop radar target simulator.

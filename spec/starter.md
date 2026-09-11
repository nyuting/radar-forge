# Repository Specification: `radar-forge`

## 1. Project Overview
- **Project Name:** `radar-forge`
- **PyPI Package Name:** `radar-forge` (Imports via `import radar_forge`)
- **Primary Goal:** An open-source, modular Python library and educational workbench combining 3D ray-tracing, phased array beamforming, tracking and machine learning dataset synthesis for radar interns and academic collaborators.
- **Target Audience:** Interns, students, researchers, and algorithm developers in radar/wireless sensing.

---

## 2. Dependencies & Ecosystem Integration

### 2.1 Related Open-Source GitHub Repositories
The codebase may draw architectural inspiration from the following open-source frameworks:

| Project Name | Description | Reason for Inclusion / Core Utility |
| :--- | :--- | :--- |
| **RadarSimPy** | Python/C++ Ray-Tracing Radar Simulator | High-performance physical propagation and automotive RCS simulation backend. Integrated as an optional ray-tracing engine module. |
| **RF-Genesis / WiTwin Radar** | Differentiable mmWave radar simulator (UCSD) built on Mitsuba/Dr.Jit. | Provides zero-shot domain adaptation and synthetic generative dataset workflows. Used as a reference for GPU-accelerated ray-tracing pipelines. |
| **Phased-Array-Antenna-Model** | Vectorized 2D/3D radiation pattern computation library. | Fills the phased array gap in existing simulators. Supplies conformal array geometries, spatial tapering (Taylor/Chebyshev), phase quantization, and beamforming math. |
| **FMCW Radar Target Simulator (FHR)** | MATLAB/Phased Array System Toolbox simulator for traffic scenarios. | Provides functional specification for baseband signal exporting into standard deep learning annotation formats (e.g., COCO bounding boxes with range-Doppler cubes). |
| **RadarBook Software** | Companion code for *Introduction to Radar Using Python and MATLAB*. | Used as standard baseline reference for canonical DSP blocks (range FFT, Doppler FFT, CFAR, SAR primitives). |
| **pyapril** | Python library for passive radar signal processing (U. Budapest). | Reference implementation for space-time clutter cancellation (Wiener-SMI, ECA) and bistatic processing geometries. |
| **RadarSim (GUI)** | Educational pulse-Doppler visualizer with real-time scopes. | Blueprint for interactive educational modules (A-Scope, B-Scope, PPI display scopes) for intern onboarding. |

### 2.2 PyPI Packages & Other Sources
The codebase may also draw architectural inspiration from the following PyPI packages:

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
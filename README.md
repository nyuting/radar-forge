# radar-forge

**Open-source radar simulation and signal processing toolkit for education and research.**

`radar-forge` is a modular Python library and educational workbench that brings 3D ray-tracing,
phased array beamforming, tracking, and machine-learning dataset synthesis under one consistent API.
It is built for radar interns, students, researchers, and algorithm developers in radar and wireless
sensing who want to go from the radar range equation to a labelled deep-learning dataset without
stitching together seven incompatible codebases.

- **Package name (PyPI):** `radar-forge`
- **Import name:** `import radar_forge`
- **Audience:** interns, students, researchers, algorithm developers

---

## Status

**Pre-alpha — specification stage.** No code has been published yet. The design documents live in
[`spec/`](spec/):

- [`spec/starter.md`](spec/starter.md) — project charter, ecosystem survey, high-level layout
- [`spec/structure.md`](spec/structure.md) — comparative module map of the reference projects and the
  file-level design of `radar_forge`
- [`spec/scenario-001-singapore-xband.md`](spec/scenario-001-singapore-xband.md) — the first vertical
  slice: a real aircraft trajectory to baseband IQ to a range-Doppler map, in three ambiguity variants
- [`spec/scenario-003-singapore-tracking.md`](spec/scenario-003-singapore-tracking.md) — the second
  slice, stacked on it: CFAR detection, data association and Kalman tracking

APIs shown below are targets, not shipping behaviour.

---

## Scope

| Area | What it covers |
| :--- | :--- |
| **`core`** | Radar range equation, target/RCS and Swerling models, FMCW and pulsed waveforms, baseband signal math, range/Doppler FFT, CFAR, clutter, tracking |
| **`array`** | Phased array geometries (linear, planar, circular, conformal), Taylor/Chebyshev tapering, beam and null steering, phase quantization, DoA estimation |
| **`raytracing`** | A single scene abstraction over pluggable ray-tracing backends (RadarSimPy, Mitsuba/Dr.Jit, NVIDIA Omniverse RTX), all optional extras |
| **`pipelines`** | Scenario generation and ML dataset synthesis, with COCO-annotation and range-Doppler-cube exporters plus PyTorch dataset wrappers |
| **`teaching`** | Interactive scopes (A-Scope, B-Scope, PPI) and Jupyter notebooks for intern onboarding |

---

## Installation

> Not yet published to PyPI. The commands below describe the intended install path.

```bash
pip install radar-forge                  # core library (NumPy/SciPy only)
pip install "radar-forge[raytracing]"    # + ray-tracing backends
pip install "radar-forge[ml]"            # + PyTorch dataset pipelines
pip install "radar-forge[teaching]"      # + GUI scopes and notebook extras
```

From source, for development:

```bash
git clone https://github.com/<org>/radar-forge.git
cd radar-forge
pip install -e ".[dev]"
pytest
```

---

## Quickstart

> Illustrative target API — this does not run yet.

```python
import numpy as np
import radar_forge as rf

# 1. Describe the sensor
radar = rf.core.Radar(
    waveform=rf.core.FMCW(f0=77e9, bandwidth=1e9, chirp_time=40e-6, n_pulses=128),
    tx=rf.array.ULA(n=3, spacing=0.5),
    rx=rf.array.ULA(n=4, spacing=0.5),
)

# 2. Place a scene
scene = rf.Scene(targets=[rf.core.PointTarget(range_m=42.0, velocity=-12.0, rcs_dbsm=5.0)])

# 3. Simulate baseband and process
baseband = radar.simulate(scene)
rd_map = rf.core.range_doppler(baseband)
detections = rf.core.cfar_2d(rd_map, guard=(2, 2), train=(4, 4), pfa=1e-4)

# 4. Export a labelled training sample
rf.pipelines.export_coco(rd_map, detections, out_dir="dataset/")
```

---

## Package layout

```text
radar-forge/
├── pyproject.toml
├── README.md
├── LICENSE
├── docs/                       # Developer & intern documentation
├── spec/                       # Design specifications
├── tests/                      # Pytest unit and integration tests
└── src/
    └── radar_forge/            # Core import package
        ├── __init__.py
        ├── core/               # Radar equation, target models, baseband signal math, DSP
        ├── array/              # Phased array pattern generation, tapering, null steering, DoA
        ├── raytracing/         # Wrappers for Mitsuba / RadarSimPy / ovrtx backends
        ├── pipelines/          # ML dataset generation & COCO / range-Doppler exporters
        └── teaching/           # Interactive GUI scopes (PPI, A-Scope) and Jupyter notebooks
```

See [`spec/structure.md`](spec/structure.md) for the file-level breakdown and the upstream module
each part draws from.

---

## Related work & references

`radar-forge` does not vendor these projects. It is inspired by them architecturally and, where licences
and hardware permit, reaches them through optional arm's-length backends. Licences are listed because they constrain how
much can be borrowed — GPL-licensed code is treated as a **reference to reimplement**, not to copy.

### Reference frameworks

| Project | Licence | Description | Utility to radar-forge |
| :--- | :--- | :--- | :--- |
| [RadarSimPy](https://github.com/radarsimx/radarsimpy) | GPL-3.0 | Python/C++ ray-tracing radar simulator | High-performance propagation and automotive RCS backend; wrapped at arm's length as an optional ray-tracing backend |
| [RF-Genesis](https://github.com/Asixa/RF-Genesis) | MIT | Differentiable mmWave radar simulator (UCSD, SenSys '23) on Mitsuba/Dr.Jit | Reference for GPU-accelerated ray-tracing pipelines and generative synthetic-dataset workflows |
| [Phased-Array-Antenna-Model](https://github.com/jman4162/Phased-Array-Antenna-Model) | MIT | Vectorized 2D/3D radiation pattern computation library | Fills the phased array gap: conformal geometries, Taylor/Chebyshev tapering, phase quantization, beamforming math |
| [FMCW Radar Target Simulator](https://github.com/thomaswengerter/FMCW_Radar_Target_Simulator) | MIT | MATLAB/Phased Array System Toolbox simulator for urban traffic scenarios | Functional spec for exporting baseband data as COCO bounding boxes over range-Doppler cubes |
| [RadarBook Software](https://github.com/RadarBook/software) | *(no licence declared)* | Companion code for *Introduction to Radar Using Python and MATLAB* | Baseline reference for canonical DSP blocks (range FFT, Doppler FFT, CFAR, SAR primitives) |
| [RASPNet](https://github.com/shyamven/RASPNet) | *(no licence declared)* | Benchmark dataset and example code for radar adaptive signal processing (US Air Force SDMS); data hosted on the AFRL SDMS portal | Reference for ML dataset conventions — feature/label layout, scenario indexing, transfer-learning baselines for target localization — and an external benchmark for the `pipelines` exporters |
| [torchcvnn](https://github.com/torchcvnn/torchcvnn) | MIT | Complex-valued PyTorch layers, transforms and SAR dataset loaders — MSTAR, SAMPLE, PolSF, ALOS2, UAVSAR SLC, S1SLC (CentraleSupélec, IJCNN '25) | Reference layout for complex radar data in `pipelines/datasets.py`; usable directly as an optional `cvnn` extra |
| [complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch) | MIT | Complex-valued layers, activations and batch normalisation after Trabelsi et al. (ICLR '18) | Complex batch-norm formulation and minimal layer set; the library Steinmetz builds on |
| [Steinmetz Neural Networks](https://github.com/shyamven/SteinmetzNeuralNetworks) | *(no licence declared)* | Complex-valued networks keeping I/Q as an analytic pair (AISTATS '25) by the RASPNet author; ships a RASPNet regression notebook | Reference for the complex-valued I/Q feature convention in `pipelines/datasets.py`; paired with RASPNet as a dataset-plus-model benchmark |
| [pyAPRiL](https://github.com/pyapril/pyapril) | GPL-3.0 | Passive radar signal processing library (BME, Budapest) | Reference for space-time clutter cancellation (Wiener-SMI, ECA) and bistatic geometries |
| [RadarSim (GUI)](https://github.com/SpaceEngineerSS/RadarSim) | MIT | Educational pulse-Doppler visualizer with real-time PySide6 scopes | Blueprint for the interactive teaching modules (A-Scope, B-Scope, PPI, RHI) |
| [Stone Soup](https://github.com/dstl/Stone-Soup) | MIT | Dstl's target-tracking and state-estimation framework: filters, gaters, data associators, initiators, OSPA/GOSPA metrics | The architectural reference for `core/tracking.py` — its predictor/updater/associator seams and its `Detection`/`Track`/`Hypothesis` vocabulary, at a fraction of the surface area; the reference for JPDA and IMM when those arrive |
| [FilterPy](https://github.com/rlabbe/filterpy) | MIT | Kalman, extended, unscented and particle filters, with the *Kalman and Bayesian Filters in Python* companion text | Reference formulation for the constant-velocity KF and the `Q_discrete_white_noise` process-noise construction, reimplemented in `core/tracking.py` rather than depended on |
| [motpy](https://github.com/wmuron/motpy) | MIT | Minimal tracking-by-detection multi-object tracker: predict, Hungarian match, update, prune | Evidence that the whole tracking loop fits in one module, which is why `spec/structure.md` D2 keeps `core/tracking.py` unsplit; shape reference for `TrackManager` |
| [Tracktable](https://github.com/sandialabs/tracktable) | BSD-3-Clause | Sandia's C++/Python moving-object trajectory analysis: coordinate domains, spatial indexing, clustering, map rendering | Prior art for the trajectory data model in `pipelines/trajectories.py` and the plan-view plot; **not a dependency** — matplotlib already draws the one plot needed |
| [labeledRFS](https://github.com/linh-gist/labeledRFS) / [VisualRFS](https://github.com/linh-gist/VisualRFS) | MIT (ports of Ba-Tuong Vo's MATLAB, whose own terms differ) | GLMB/LMB random-finite-set multi-target trackers with Gibbs-sampled ranked assignment and OSPA metrics | The target state for high-clutter multi-target tracking: no heuristic gate, no M-of-N initiator, no deleter. Written from the papers if implemented; nothing vendored |

> **Attribution note:** the FMCW Radar Target Simulator is sometimes miscredited to Fraunhofer FHR. It is
> an independent MATLAB project by Thomas Wengerter; FHR's ATRIUM is a separate, unrelated
> hardware-in-the-loop target simulator.

### PyPI packages

| Package | Licence | Role | Inclusion purpose |
| :--- | :--- | :--- | :--- |
| [`AIRadarLib`](https://pypi.org/project/AIRadarLib/) *(optional)* | *(unstated)* | FMCW/OFDM radar simulation library | Reference for time-domain chirp generation (PLL quantization, phase noise) and PyTorch transformer dataset wrappers |
| [`ovrtx`](https://pypi.org/project/ovrtx/) *(optional)* | Proprietary (NVIDIA SLA) | NVIDIA Omniverse RTX Sensor SDK — [source](https://github.com/NVIDIA-Omniverse/ovrtx) | High-fidelity GPU point cloud / target map simulation reference; strictly an optional backend |
| [`pyroomacoustics`](https://github.com/LCAV/pyroomacoustics) | MIT | Spatial array processing & beamforming | Direction-of-arrival estimation benchmarks (MUSIC and relatives) |

---

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) — one command
(`./scripts/setup-dev.sh`) installs the environment and activates the git hooks that enforce
everything below.

| Guide | Covers |
| :--- | :--- |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, the development loop, what each git hook does |
| [docs/conventions/commits.md](docs/conventions/commits.md) | Conventional Commits, branch naming, PR titles |
| [docs/conventions/style.md](docs/conventions/style.md) | Naming, the SI unit-suffix convention, docstrings, typing |
| [docs/conventions/testing.md](docs/conventions/testing.md) | Test layout, tolerances, analytic ground truth, markers |
| [CLAUDE.md](CLAUDE.md) | The same conventions, condensed for AI agents |

Design feedback on the documents in [`spec/`](spec/) is equally welcome.

## Licence

MIT — see [LICENSE](LICENSE).

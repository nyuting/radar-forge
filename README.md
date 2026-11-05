> 🎓 **Interns, look here!** Research tasks for intern candidates and interns:
> [interns/README.md](interns/README.md)
>
> 📘 **New to the library?** Start with the [user guide](docs/user-guide/README.md): each stage
> from geometry to tracks, with its equations, an example to run and a scenario to try.

# radar-forge

**A state-of-the-art, simple radar workbench for research and education.**

`radar-forge` is a research and educational workbench for radar that is both state of the art
and simple: the current best method for every block, with correct and cited mathematics, written
so a new engineer can follow it and a practitioner can learn from it. It is a modular, open-source
Python library that brings 3D ray-tracing, phased-array beamforming, detection and tracking, and
machine-learning dataset synthesis under one consistent API. It is built for radar practitioners
and researchers, new staff, interns, students, and algorithm developers in radar and wireless
sensing who want to go from the radar range equation to a labelled deep-learning dataset without
stitching together seven incompatible codebases.

- **Package name (PyPI):** `radar-forge`
- **Import name:** `import radar_forge`
- **Audience:** interns, students, researchers, algorithm developers

---

## Status

**Pre-alpha.** Nothing is on PyPI yet. The radar chain is built and tested as flat modules,
from geometry to tracks, and scenarios 001–003 run end to end through the stepper; the other
areas under [Scope](#scope) are designed but not built. The [user guide](docs/user-guide/README.md)
explains each module. The design documents live in [`spec/`](spec/):

- [`spec/data-001-formats.md`](spec/data-001-formats.md): the files a run writes, with columns,
  units and conventions
- [`spec/tracker-001.md`](spec/tracker-001.md): the tracking architecture, one Gaussian filter, a
  batched track bank and one scan loop
- `spec/scenario-*.md`: one specification per scenario not yet built; a built scenario's
  acceptance numbers live in its `config/*.toml` `[verify]` table, and the table in
  [`docs/README.md`](docs/README.md#scenarios) says which run today

The [Quickstart](#quickstart) below runs on today's `main`.

---

## Scope

| Area | What it covers |
| :--- | :--- |
| **The radar chain** | `radar` (sites, transmitter, receiver, bistatic geometry), `propagation` (the radar equation, Swerling models), `signal` (FMCW and pulsed waveforms, IQ), `dsp` (range-Doppler processing, windows, MTI, dual-PRF unfolding), `detection` (CFAR and clustering), `tracking/` (KF, EKF and UKF over a batched track bank) |
| **Scenarios** | `scenario` (a TOML file and its truth), `pipeline` (frames to tracks), `evaluate` (the stepper, which checks each stage against the file's `[verify]` table), `plots` |
| **`array`** | Phased array geometries (linear, planar, circular, conformal), Taylor/Chebyshev tapering, beam and null steering, phase quantization, DoA estimation |
| **`raytracing`** | A single scene abstraction over pluggable ray-tracing backends (RadarSimPy, Mitsuba/Dr.Jit, NVIDIA Omniverse RTX), all optional extras |
| **Datasets** | ML dataset synthesis, with COCO-annotation and range-Doppler-cube exporters plus PyTorch dataset wrappers (planned) |

---

## Installation

> Not yet published to PyPI. The commands below describe the intended install path.

```bash
pip install radar-forge                  # core library (NumPy/SciPy only)
pip install "radar-forge[raytracing]"    # + ray-tracing backends
pip install "radar-forge[ml]"            # + PyTorch dataset pipelines
pip install "radar-forge[visualisation]" # + matplotlib figures
pip install "radar-forge[notebooks]"     # + JupyterLab, for the tutorial notebooks
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

Run a scenario through the stepper. Each stage prints what it measured and its wall time,
checks the scenario's `[verify]` table and writes its files under `runs/<name>/`:

```bash
uv sync --extra dev --extra visualisation
uv run python scripts/run_scenario.py config/scenario_001_fmcw_low_prf.toml --frames 3
```

`--stop-after rd` stops after a stage, `--save-iq` keeps the cubes, and a failed check exits 1
and names it. The same chain, one dwell by hand:

```python
import numpy as np
import radar_forge as rf

# 1. The sensor: an X-band FMCW radar at one site (no receiver site: monostatic).
transmitter = rf.Transmitter(
    carrier_frequency_hz=9.8e9,
    bandwidth_hz=2.0e6,
    transmit_power_w=100.0,
    gain_tx_dbi=30.0,
    chirp_duration_s=1.0e-3,
    prf_hz=1.0e3,
)
receiver = rf.Receiver(sample_rate_hz=1.0e6, gain_rx_dbi=30.0, noise_figure_db=3.0)
radar = rf.Radar(transmitter, receiver, rf.Site(35.0, -78.0, 60.0))

# 2. A 10 m^2 target 12 km out, opening at 6 m/s, and its propagation path.
geometry = rf.TargetGeometry.monostatic(12_000.0, 6.0)
paths = rf.line_of_sight_paths(radar, geometry, [10.0])

# 3. Complex baseband IQ, shape (n_pulses, n_samples).
iq = rf.fmcw_deramp_baseband(paths, radar, 256, rng=np.random.default_rng(0))

# 4. The range-Doppler map, and CFAR along range.
rd_map = rf.range_doppler_map(iq)
detected, threshold_w = rf.cfar_detect(np.abs(rd_map) ** 2, 1e-4, n_train=8, n_guard=2)

# 5. The brightest cell in physical units: bistatic range R_T + R_R and its rate.
rate_bin, range_bin = np.unravel_index(np.abs(rd_map).argmax(), rd_map.shape)
range_axis_m = rf.range_bin_centers_m(rd_map.shape[1], 2.0e6, 1.0e-3, 1.0e6)
rate_axis_mps = rf.doppler_bin_centers_mps(rd_map.shape[0], 1.0e-3, radar.wavelength_m)
print(round(range_axis_m[range_bin], 1), round(rate_axis_mps[rate_bin], 2))  # 23983.4 11.95
```

The map's axes carry the bistatic range R_T + R_R and its rate, positive receding: twice the
slant range and the radial velocity for a monostatic radar, which only a display halves
(`radar_forge.plots.display_axes`). A second site is `rf.Radar(transmitter, receiver,
tx_site, rx_site)`, and nothing else changes.

---

## Package layout

```text
radar-forge/
├── pyproject.toml
├── README.md
├── LICENSE
├── docs/                       # Developer & intern documentation
├── spec/                       # Design specifications
├── config/                     # Scenario TOMLs, each with its [verify] table
├── scripts/                    # run_scenario.py (the stepper's CLI), regen_golden.py, hooks' checks
├── tests/                      # Pytest unit and integration tests
└── src/
    └── radar_forge/            # The import package: flat modules, signal-path order
        ├── constants.py        # Every physical constant, once
        ├── radar.py            # Transmitter, receiver, sites, bistatic geometry, WGS-84
        ├── propagation.py      # Propagation paths, the radar equation, Swerling draws
        ├── signal.py           # LFM chirp, thermal noise, FMCW and pulsed IQ synthesis
        ├── dsp.py              # Windows, matched filter, range and Doppler FFTs, unfolding
        ├── detection.py        # CFAR (CA, GO, SO, OS; 1-D and 2-D) and clustering
        ├── tracking/           # Models, filters, association, tracker, metrics
        ├── scenario.py         # A scenario TOML and its truth
        ├── pipeline.py         # Frames, range-Doppler products, plots, scans
        ├── evaluate.py         # The stepper: geometry, iq, rd, detect, track
        └── plots.py            # Figures (the visualisation extra)
```

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
| [pyAPRiL](https://github.com/pyapril/pyapril) | GPL-3.0 | Bistatic radar signal-processing library (BME, Budapest) | Structure reference for least-squares clutter and leakage cancellation (Wiener-SMI, ECA); GPL-3.0, so reimplemented from the papers, never vendored |
| [RASPNet](https://github.com/shyamven/RASPNet) | *(no licence declared)* | Benchmark dataset and example code for radar adaptive signal processing (US Air Force SDMS); data hosted on the AFRL SDMS portal | Reference for ML dataset conventions — feature/label layout, scenario indexing, transfer-learning baselines for target localization — and an external benchmark for the planned exporters |
| [torchcvnn](https://github.com/torchcvnn/torchcvnn) | MIT | Complex-valued PyTorch layers, transforms and SAR dataset loaders — MSTAR, SAMPLE, PolSF, ALOS2, UAVSAR SLC, S1SLC (CentraleSupélec, IJCNN '25) | Reference layout for complex radar data in a planned `datasets` module; usable directly as an optional `cvnn` extra |
| [complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch) | MIT | Complex-valued layers, activations and batch normalisation after Trabelsi et al. (ICLR '18) | Complex batch-norm formulation and minimal layer set; the library Steinmetz builds on |
| [Steinmetz Neural Networks](https://github.com/shyamven/SteinmetzNeuralNetworks) | *(no licence declared)* | Complex-valued networks keeping I/Q as an analytic pair (AISTATS '25) by the RASPNet author; ships a RASPNet regression notebook | Reference for the complex-valued I/Q feature convention in a planned `datasets` module; paired with RASPNet as a dataset-plus-model benchmark |
| [RadarSim (GUI)](https://github.com/SpaceEngineerSS/RadarSim) | MIT | Educational pulse-Doppler visualizer with real-time PySide6 scopes | Blueprint for interactive scopes beside `plots` (A-Scope, B-Scope, PPI, RHI) |
| [Stone Soup](https://github.com/dstl/Stone-Soup) | MIT | Dstl's target-tracking and state-estimation framework: filters, gaters, data associators, initiators, OSPA/GOSPA metrics | The architectural reference for `tracking/` — its predictor/updater/associator seams and its `Detection`/`Track`/`Hypothesis` vocabulary, at a fraction of the surface area; the reference for JPDA and IMM when those arrive |
| [FilterPy](https://github.com/rlabbe/filterpy) | MIT | Kalman, extended, unscented and particle filters, with the *Kalman and Bayesian Filters in Python* companion text | Reference formulation for the constant-velocity KF and the `Q_discrete_white_noise` process-noise construction, reimplemented in `tracking/` from the cited papers rather than depended on |
| [motpy](https://github.com/wmuron/motpy) | MIT | Minimal tracking-by-detection multi-object tracker: predict, Hungarian match, update, prune | Shape reference for the scan loop: one scan in, the current tracks out, as `tracking.Tracker.process(scan)` |
| [Tracktable](https://github.com/sandialabs/tracktable) | BSD-3-Clause | Sandia's C++/Python moving-object trajectory analysis: coordinate domains, spatial indexing, clustering, map rendering | Prior art for the trajectory handling in `scenario.py` and the range-time plot; **not a dependency** — matplotlib already draws the one plot needed |
| [labeledRFS](https://github.com/linh-gist/labeledRFS) / [VisualRFS](https://github.com/linh-gist/VisualRFS) | MIT (ports of Ba-Tuong Vo's MATLAB, whose own terms differ) | GLMB/LMB random-finite-set multi-target trackers with Gibbs-sampled ranked assignment and OSPA metrics | The target state for high-clutter multi-target tracking: no heuristic gate, no M-of-N initiator, no deleter. Written from the papers if implemented; nothing vendored |

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

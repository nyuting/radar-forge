# radar-forge — Module Structure

**Status:** design document, partially implemented. `core/` and `pipelines/` are built as far as
scenarios 001 and 002 required; `array/`, `raytracing/`, `pipelines/exporters/` and most of
`teaching/` are still design only. Part B marks the tree as intended, not as built — read it
alongside the source.
**Companion to:** [`starter.md`](starter.md) (project charter and ecosystem survey).

This document does two things. **Part A** surveys each reference project named in the starter spec and
records what its actual core modules are. **Part B** turns that survey into a file-level design for the
`radar_forge` package, naming for each module the upstream work it draws from.

---

## Part A — Comparative module map

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
CFAR, SAR primitives) and the chapter→notebook pedagogy for `teaching/notebooks/`. Implementations are
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
`teaching/`, and the YAML-scenario idea for `pipelines/scenarios.py`. Its CFAR-variant and tracking
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

`core/tracking.py` is the one module in Part B with no upstream in §A.1–A.12 beyond two textbook
companions. These five projects are its comparative map. None is vendored and none is a runtime
dependency; `scipy.optimize.linear_sum_assignment` covers the only algorithm
`spec/scenario-003-singapore-tracking.md` actually needs, and `scipy` is already in core.

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

**What radar-forge borrows:** the *vocabulary and the seams*, not the code. The
detection → hypothesiser → gater → associator → updater → initiator/deleter decomposition is the
one this repository's `core/tracking.py` follows, at a fraction of the surface area: Stone Soup is
a framework for comparing trackers, and radar-forge needs one tracker an intern can read end to
end. Its metric generators are the reference for any future acceptance criterion beyond
`spec/scenario-003-singapore-tracking.md` §12, and its JPDA and IMM implementations are the
reference for the extensions that would fire D2's promotion trigger. MIT, so code could be
borrowed with attribution; the reason not to is size, not licence.

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
discrete-white-noise-acceleration construction `core/tracking.py` reimplements from reference [2]
of the scenario spec, and FilterPy's `log_likelihood`/NIS bookkeeping is the model for the
consistency statistic that scenario 003 §12 asserts on. It is a reference to reimplement rather
than a dependency: the two-state constant-velocity case is about forty vectorised lines, and
carrying a filter library to get them would fail `CLAUDE.md`'s test for adding a dependency.

**A.13c motpy**

- **Link:** https://github.com/wmuron/motpy
- **Language / licence:** Python · MIT

| Module | Responsibility |
| :--- | :--- |
| `tracker.py` | `MultiObjectTracker`: the whole predict → match → update → prune loop in one file |
| `core.py` | `Box`, `Detection`, `Track` value types |
| `metrics.py` | IoU and Euclidean cost matrices for the assignment step |
| `model.py` | Constant-velocity and constant-acceleration motion models with a single order knob |

**What radar-forge borrows:** the *proof of scale*. motpy is tracking-by-detection with Hungarian
matching and a staleness-based track manager, complete, in roughly one module — which is the
evidence behind `spec/structure.md` D2's decision to keep `core/tracking.py` unsplit until a
second association strategy arrives. Its `MultiObjectTracker.step(detections) -> tracks` signature
is the shape `TrackManager` follows. It is a computer-vision tracker, so nothing about its cost
metrics or box model transfers; the loop structure is the whole borrowing.

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
`pipelines/trajectories.py` and for `teaching/scopes/track_plot.py` — in particular the separation
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
three components `spec/scenario-003-singapore-tracking.md` §7 has to size by hand and defend in
§3. That contrast is worth teaching even before the filter is implemented. Implementation is
gated on a multi-target, high-clutter scenario existing to justify it, and would be written from
the papers.

### A.14 Licence summary

| Project | Licence | Usable as dependency? | Usable as code source? |
| :--- | :--- | :--- | :--- |
| Phased-Array-Antenna-Model | MIT | yes | yes, with attribution |
| Stone Soup | MIT | yes | yes, with attribution — not taken, size not licence |
| FilterPy | MIT | yes | yes, with attribution — reimplemented instead, ~40 lines |
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
│   ├── detection.py             # CA/GO/SO/OS-CFAR, Pfa calibration, detection clustering
│   ├── clutter.py               # land/sea clutter models; ECA / Wiener-SMI cancellation
│   └── tracking.py              # KF, EKF, gating, assignment, simple track manager
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
│   ├── trajectories.py          # target motion planning (traffic, pedestrian, bicyclist)
│   ├── generate.py              # scene -> baseband -> cube orchestration, batching, seeding
│   ├── datasets.py              # torch Dataset / DataLoader wrappers (extra: ml)
│   └── exporters/
│       ├── __init__.py
│       ├── coco.py              # range-Doppler(-azimuth) cube -> COCO annotations
│       ├── range_doppler.py     # cube serialization: .npz / .mat / dB scaling
│       └── labels.py            # shared label schema: range, velocity, azimuth, x, y, w, h, heading, obstruction
└── teaching/
    ├── __init__.py
    ├── app.py                   # PySide6 application shell (extra: teaching)
    ├── scopes/
    │   ├── __init__.py
    │   ├── ascope.py            # amplitude vs range
    │   ├── bscope.py            # range vs azimuth
    │   ├── ppi.py               # plan position indicator
    │   └── rd_map.py            # live range-Doppler map
    ├── plotting.py              # matplotlib helpers shared by notebooks and scopes
    └── notebooks/               # chapter-style guided notebooks for intern onboarding
```

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
| `core/tracking.py` | Stone Soup data model and predictor/updater/associator seams; FilterPy filter and `Q_discrete_white_noise` formulation; motpy's single-module tracking loop; RadarSim tracking and fusion; RadarBook tracking-filter chapters |
| `array/*` | Phased-Array-Antenna-Model (near one-to-one module split) |
| `array/doa.py` | pyroomacoustics `doa` base-class pattern; RadarSimPy and pyroomacoustics estimators as benchmarks |
| `raytracing/base.py`, `scene.py` | RF-Genesis pipeline staging; RadarSimPy scene/mesh API |
| `raytracing/backends/mitsuba.py` | RF-Genesis `genesis/` Mitsuba + Dr.Jit usage |
| `raytracing/backends/ovrtx.py` | ovrtx sensor-simulation API |
| `pipelines/exporters/*` | FMCW Radar Target Simulator `JSONCoco.py` and its label schema |
| `pipelines/scenarios.py` | RadarSim YAML scenario files (radar-forge uses TOML: `CLAUDE.md` requires it) |
| `pipelines/datasets.py` | AIRadarLib PyTorch dataset/training wrappers; torchcvnn `datasets`/`transforms` (complex SAR loader layout); Steinmetz Neural Networks (complex-valued I/Q feature convention) |
| `teaching/scopes/*`, `teaching/app.py` | RadarSim PySide6 GUI (PPI, RHI, A-Scope) |
| `teaching/notebooks/` | RadarBook `jupyter/`; RadarSimNb |

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
5. **Every core algorithm is teachable.** Each `core/` and `array/` module pairs with a notebook in
   `teaching/notebooks/` and a numerical test against a textbook-published value.

### B.3 Proposed extras

| Extra | Pulls in | Enables |
| :--- | :--- | :--- |
| *(none)* | numpy, scipy | `core`, `array`, `raytracing.backends.analytic` |
| `raytracing` | mitsuba, drjit | Mitsuba backend |
| `radarsimpy` | radarsimpy | RadarSimPy backend (note: GPL-3.0 — user-installed) |
| `ovrtx` | ovrtx, ovstage | Omniverse RTX backend (NVIDIA licence, RTX GPU) |
| `ml` | torch | `pipelines.datasets`, training loops |
| `cvnn` | torch, torchcvnn, complexPyTorch | complex-valued layers and SAR dataset loaders for `pipelines.datasets` |
| `teaching` | PySide6, matplotlib, jupyter | GUI scopes and notebooks |
| `dev` | pytest, ruff, mypy, build tooling | development |

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

### D2 — Clutter and tracking stay in `core/`, with a promotion trigger

`core/clutter.py` and `core/tracking.py` remain single modules for now.

> **Note.** Promote either to a subpackage as soon as it exceeds roughly one module's worth of
> responsibility — concretely, when `tracking.py` gains a second association strategy beyond
> nearest-neighbour gating (JPDA, MHT) or a track-fusion layer, or when `clutter.py` carries more than two
> clutter models plus the cancellation algorithms. The promotion is `core/tracking.py` →
> `core/tracking/{filters,association,fusion}.py`, re-exported from `core/tracking/__init__.py` so the
> public import path never changes. Do not pre-split; do not let either file drift past the trigger.

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

Settled by `spec/scenario-002-singapore-bistatic.md`, which was the first slice to put the
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
β = 100.5°, so **absolute** power in a bistatic scenario is an approximation. Its acceptance
criteria test range, velocity and resolution — geometry — and deliberately assert nothing about
absolute SNR. Forward scatter is not modelled at all.

---

## Open questions

1. **GPU story — deferred.** Three backends (Mitsuba, RadarSimPy, ovrtx) need CUDA or an RTX GPU. Whether
   CI exercises any of them, or backend tests stay mock-only with hardware validation left manual, is a
   question for when the first real backend lands. Until then `raytracing/backends/analytic.py` is the only
   backend under test, and it runs anywhere. Revisit before merging the first GPU backend.

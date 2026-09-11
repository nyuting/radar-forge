# radar-forge — Module Structure

**Status:** design document, pre-implementation.
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

### A.7 RadarSim (GUI)

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

### A.8 AIRadarLib *(PyPI, optional)*

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

### A.9 ovrtx *(PyPI, optional)*

- **Link:** https://github.com/NVIDIA-Omniverse/ovrtx · https://pypi.org/project/ovrtx/
- **Language / licence:** C + Python · **proprietary** (NVIDIA Software Licence Agreement)
- **Requirements:** RTX-capable NVIDIA GPU; Windows x86_64 / Linux x86_64 / Linux aarch64; Python 3.10–3.13

Exposes physically accurate real-time camera, lidar and radar sensor simulation over Omniverse RTX.

**What radar-forge borrows:** nothing structural — it is a highest-fidelity optional backend behind
`raytracing/backends/ovrtx.py`, gated on hardware and licence acceptance, never installed by default and
never imported at package import time.

### A.10 pyroomacoustics

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

### A.11 Licence summary

| Project | Licence | Usable as dependency? | Usable as code source? |
| :--- | :--- | :--- | :--- |
| Phased-Array-Antenna-Model | MIT | yes | yes, with attribution |
| pyroomacoustics | MIT | yes | yes, with attribution |
| RF-Genesis | MIT | optional extra | yes, with attribution |
| RadarSim (GUI) | MIT | reference | yes, with attribution |
| FMCW Radar Target Simulator | MIT | no (MATLAB) | format spec only |
| RadarSimPy | GPL-3.0 | optional extra only, arm's length | **no** |
| pyAPRiL | GPL-3.0 | **no** | **no** — reimplement from papers |
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
│   ├── radar.py                 # Transmitter, Receiver, Radar
│   ├── radar_equation.py        # range equation, SNR, max range, link budget
│   ├── waveforms.py             # FMCW/LFM chirp, pulse train, CW, PMCW; ambiguity function
│   ├── targets.py               # PointTarget, ExtendedTarget, RCS + Swerling 0–4
│   ├── propagation.py           # free-space loss, atmospheric absorption, rain attenuation
│   ├── signal.py                # baseband synthesis, superposition, noise, phase noise
│   ├── dsp.py                   # range FFT, Doppler FFT, windowing, matched filter, MTI
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
│       ├── analytic.py          # always-available fallback: point-target / specular approximation
│       ├── radarsimpy.py        # RadarSimPy backend (extra: radarsimpy)
│       ├── mitsuba.py           # Mitsuba/Dr.Jit backend, RF-Genesis-style (extra: mitsuba)
│       └── ovrtx.py             # NVIDIA Omniverse RTX backend (extra: ovrtx)
├── pipelines/
│   ├── __init__.py
│   ├── scenarios.py             # YAML/JSON scenario schema + loader
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
| `core/tracking.py` | RadarSim tracking and fusion; RadarBook tracking-filter chapters |
| `array/*` | Phased-Array-Antenna-Model (near one-to-one module split) |
| `array/doa.py` | pyroomacoustics `doa` base-class pattern; RadarSimPy and pyroomacoustics estimators as benchmarks |
| `raytracing/base.py`, `scene.py` | RF-Genesis pipeline staging; RadarSimPy scene/mesh API |
| `raytracing/backends/mitsuba.py` | RF-Genesis `genesis/` Mitsuba + Dr.Jit usage |
| `raytracing/backends/ovrtx.py` | ovrtx sensor-simulation API |
| `pipelines/exporters/*` | FMCW Radar Target Simulator `JSONCoco.py` and its label schema |
| `pipelines/scenarios.py` | RadarSim YAML scenario files |
| `pipelines/datasets.py` | AIRadarLib PyTorch dataset/training wrappers |
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
   published equations, with the source cited in the module docstring. See §A.11.
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

---

## Open questions

1. **GPU story — deferred.** Three backends (Mitsuba, RadarSimPy, ovrtx) need CUDA or an RTX GPU. Whether
   CI exercises any of them, or backend tests stay mock-only with hardware validation left manual, is a
   question for when the first real backend lands. Until then `raytracing/backends/analytic.py` is the only
   backend under test, and it runs anywhere. Revisit before merging the first GPU backend.

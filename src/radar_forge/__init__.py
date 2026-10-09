"""radar-forge: state-of-the-art, simple radar simulation and DSP for research and education.

The public API is exactly what this module re-exports; anything reached through a
submodule path is subject to change without notice. See ``docs/conventions/style.md``.

Subpackages
-----------
core
    Radar range equation, target and RCS models, waveforms, baseband signal
    math, range/Doppler processing, CFAR, clutter and tracking.
array
    Phased array geometries, tapering, beam and null steering, DoA estimation.
raytracing
    A single scene abstraction over pluggable ray-tracing backends.
pipelines
    Scenario generation and machine-learning dataset synthesis.
viz
    Plots and scopes (range-Doppler maps, track histories) for reading results.
"""

from __future__ import annotations

from radar_forge import core, pipelines

__version__ = "0.0.0"

__all__ = ["__version__", "core", "pipelines"]

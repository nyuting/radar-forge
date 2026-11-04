# Testing and numerics

Numerical code does not crash when it is wrong; it returns a plausible number. Configuration is
in `pyproject.toml` `[tool.pytest.ini_options]`: `--strict-markers`, `--import-mode=importlib`,
and every warning is an error.

## 1. Layout

`tests/` mirrors `src/radar_forge/` (`tests/core/`, `tests/core/tracking/`, `tests/pipelines/`,
`tests/visualisation/`), plus `tests/scripts/` and `tests/tools/` for `scripts/` and `tools/`,
package-wide checks at the top level (`test_import.py`, `test_public_api.py`,
`test_docstrings.py`), and `tests/data/golden/`. One test module per source module. A test name
states the property: `test_inverse_fourth_power_scaling`, not `test_received_power_2`. Fixtures
live in the `conftest.py` of the directory that uses them.

## 2. Float comparison

Never `==` on floats. Use `np.testing.assert_allclose` with an explicit `rtol` (and `atol` near
zero) and a comment saying why that number: about `1e-12` for a few float64 operations,
`1e-10`…`1e-8` for an FFT round trip or linear algebra, `1e-6` for an iterative estimator. A
Monte Carlo statistic gets a confidence interval, not a tolerance. Near a null, `rtol` means
nothing: assert the null is far below the peak.

**Never loosen a tolerance to make a test pass.** If you do not know why the number moved, you
have found a bug. A tolerance change is justified in the commit body.

## 3. Ground truth, best first

1. **Closed form**: a target on an exact bin, a null at a computable angle, 12 dB per 4× range.
2. **Independent re-derivation** of the textbook expression in the test, citing it. Two of our
   own implementations that share a core are not independent: keep one test against an
   independent reference.
3. **Invariants**: Parseval, linearity, symmetry, monotonicity (good with `hypothesis`).
4. **Golden data**, last: it proves behaviour did not change, never that it is right.

Code implementing a cited reference reproduces one of its worked examples, citing it.

## 4. Fixtures and randomness

Name shared parameter sets for the scenario ("nominal 77 GHz front end"), not `params1`.
Fixtures build; they do not assert. Every random draw is seeded
(`np.random.default_rng(20260911)`), once, in a fixture. Tests write only under `tmp_path`.

## 5. Property-based tests

Use `hypothesis` when a property should hold over a domain; bound strategies to physically
sensible values.

## 6. Golden data

Only where no analytic truth exists, only in `tests/data/golden/` (the pre-commit hook blocks
binary arrays elsewhere), kilobytes not megabytes, with a `scripts/regen_golden_*.py` that
recreates it from a seed. Regenerating it is a behaviour change with its own commit. Prefer
derived statistics (peak position, detection count) over a raw array.

## 7. Markers

| Marker | Meaning |
| :--- | :--- |
| `slow` | Over about a second. Excluded by `make test-fast` and the pre-push hook; CI runs it |
| `gpu` | Needs a CUDA/Metal device |
| `raytracing` | Needs an optional backend; use `pytest.importorskip` |
| `benchmark` | Performance regression, not correctness |

A slow test says why in its docstring or a comment above the marker. If tests share an expensive
module-scoped fixture, mark them all, or the fixture still runs in the fast suite. Find
candidates with `uv run pytest -m "not slow" --durations=20`.

## 8. Coverage

`make cov` reports it; there is no threshold. Use it to find questions. What must hold: every
public function has a test against known-correct values, every documented `Raises` is
triggered, and every edge case in `Notes` is tested.

## 9. Regression tests

The test goes in first, fails for the right reason, and its name states the violated property.

## 10. Radar-domain verification

### 10.1 What earns a place **[review]**

1. **It names the bug it catches**, in one docstring sentence.
2. **It is the cheapest test that catches it**: deterministic before Monte Carlo, unit before
   pipeline. Size a Monte Carlo run from the error it must detect (about 100 expected events).
3. **It tests our code**, not NumPy's FFT or SciPy's `chi2.ppf`.
4. **It fits the budget**: unmarked tests run in well under a second.
5. **Its setup makes physical sense**: plausible targets and radar parameters that could coexist,
   a window suited to what is measured, an unfolding test whose target actually folds.

Not tests: whole-curve reproductions (notebooks), two-target resolution demos, features that do
not exist yet.

### 10.2 Front end and IQ synthesis

LFM slope K = B/T_p (catches a missing ½ in the phase). Thermal noise variance k_B·T·F·B, half per
I and Q, in a χ² interval. Swerling cases by KS test against the right χ² degrees of freedom.

### 10.3 Fast/slow-time DSP and the range-Doppler map

A target on an exact bin lands in that range and Doppler bin, with f_d = 2v_r/λ and its sign.
Window sidelobes and processing loss per Harris (1978). Compression gain B·T_p minus the window's
loss. MTI nulls at k·PRF. The −3 dB width of an unweighted response is about 0.886·c/(2B);
c/(2B) is the peak-to-first-null spacing. Code on a circular axis is tested a fraction of a bin
inside the edge and across it, not only on a bin centre.

### 10.4 Detection

Design Pfa inside the Clopper–Pearson interval of the measured rate
(`assert_rate_consistent_with_design` in `tests/core/test_detection.py`); GO/SO/OS behaviour at
a clutter edge; exact vertex recovery for centroid interpolation.

### 10.5 Measurements and coordinates

Spherical–Cartesian round trip at `rtol=1e-12`; analytic Jacobian against a central difference;
converted covariance symmetric with a Cholesky that succeeds.

### 10.6 Tracking

NEES and NIS summed over N runs inside the 99.9 % χ² interval, for every estimator; a gate that
accepts the fraction it claims; confirmation on exactly the M-th hit and deletion on the K-th
miss; crossing targets keep their IDs. `spec/tracker-001.md` is the authority.

### 10.7 ML

A learned detector holds its Pfa on noise only (the §10.4 test); outputs stay finite across
−20 to +40 dB SNR. "Pd no worse than classical" is a `benchmark`.

### 10.8 End to end

Two per pipeline: the processed SNR of a point target equals the radar equation plus
integration gain minus window losses; and Pd at one SNR point against an exact form (Swerling 1,
one pulse: Pd = Pfa^(1/(1+SNR))).

### 10.9 When a statistical test fails

With a fixed seed it always passes or always fails, so a new failure means the code changed.
Changing the seed until it passes is loosening a tolerance.

### 10.10 Pruning a suite **[review]**

Each test is **sound** (keep), **redundant** (a kept test catches its bug, or it tests NumPy:
delete), **weak** (unconvincing setup or assertion: rewrite) or **wrong** (asserts something
untrue: fix, and say what it hid). Every deletion names its survivor; coverage of `src/` does not
drop; a sample of deletions is checked by breaking what they guarded.

### 10.11 Ways a test passes for the wrong reason **[review]**

- Symmetric inputs hide an asymmetric bug: make the gains, coordinates or delays unequal.
- The assertion runs over an empty set (`np.all([])` is `True`): assert it is non-empty.
- The bound is looser than the bug: assert the exact bin, the closed-form value.
- The identity holds by construction: compare with something that shares no code.
- Two quantities coincide in the fixture (`time_s = arange(n)` is the frame index).
- The geometry makes the effect vanish: choose one where it is large.
- Only "something happened" is asserted: assert the data, with non-default arguments.

Break the behaviour on purpose and check that the test goes red.

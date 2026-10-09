# Testing and numerics

Numerical code fails differently from ordinary software: it does not crash, it returns a
plausible number that is wrong. Everything below follows from that.

`pytest` configuration lives in `pyproject.toml` under `[tool.pytest.ini_options]`.

---

## 1. Layout

`tests/` mirrors `src/radar_forge/`:

```
tests/
├── conftest.py                 # shared fixtures
├── test_import.py              # package smoke test
├── core/
│   └── test_radar_equation.py  # mirrors src/radar_forge/core/radar_equation.py
├── array/
├── pipelines/
└── data/
    └── golden/                 # the only place binary reference data may live
```

One test module per source module, same name with a `test_` prefix. If you cannot decide
where a test goes, the module under test is probably doing two things.

Test names state the property, not the mechanics:
`test_inverse_fourth_power_scaling`, not `test_received_power_2`.

---

## 2. Float comparison

**Never compare floats with `==`.** Always `np.testing.assert_allclose` with an *explicit*
`rtol` (and `atol` where values approach zero), plus a comment justifying the number:

```python
# rtol at 1e-12: this is a handful of float64 multiplies, so anything looser
# would hide a genuine algebraic error.
np.testing.assert_allclose(received_power_w(**NOMINAL, range_m=42.0), expected, rtol=1e-12)
```

Rough guidance, to be argued with per case:

| Situation | `rtol` |
| :--- | :--- |
| A few float64 arithmetic operations | `1e-12` |
| FFT round-trip, linear algebra | `1e-10` … `1e-8` |
| Iterative estimator (MUSIC, ESPRIT), accumulated error | `1e-6` |
| Monte-Carlo statistic (CFAR false-alarm rate) | Use a confidence interval, not a tolerance |

**Near zero, `rtol` is meaningless** — a pattern null is "−infinity dB" and relative error
against it is nonsense. Use `atol` scaled to the signal, or assert the null is below a
threshold relative to the peak:

```python
# A ULA null is a cancellation, so its absolute depth is set by float64 round-off.
# Assert it is far below the peak rather than pinning an exact value.
assert pattern_db[null_index] < pattern_db.max() - 60.0
```

**Never loosen a tolerance to make a failing test pass.** A tolerance change is a claim
about numerics and needs a justification in the commit body. If you do not know why the
number moved, you have found a bug, not a tolerance problem.

---

## 3. Prefer analytic ground truth

Ranked best to worst:

1. **Closed-form truth.** A point target at range R appears in a known range bin; an
   unweighted ULA has its first null at a computable angle; a 4× range increase costs
   exactly 12 dB. These tests catch real errors and document the physics.
2. **Independent re-derivation.** Evaluate the textbook expression separately in the test
   from the inputs, and compare. Still catches algebra errors — but not a shared
   misunderstanding, so cite the reference in the test too.
3. **Invariants and properties.** Energy conservation (Parseval), linearity, superposition,
   symmetry, monotonicity, unit consistency. Good with `hypothesis` (§5).
4. **Golden data.** Last resort (§6). It proves only that behaviour has not *changed*, never
   that it is *correct* — and it happily locks in a bug.

Anything implementing a cited reference should have at least one test reproducing a worked
example from that reference, with the citation in the test.

```python
def test_matches_the_textbook_worked_example() -> None:
    """<Author>, <book>, §<section>, <example number>: its inputs give its printed result."""
```

---

## 4. Fixtures

- Shared physically-meaningful parameter sets go in `conftest.py` as fixtures or module
  constants — a "nominal 77 GHz automotive front-end", a "nominal 8-element ULA". Name them
  after the scenario, not `params1`.
- Fixtures build objects; they do not assert. A failing fixture reports as an error rather
  than a failure and is much harder to read.
- `@pytest.mark.parametrize` for the same property across several inputs. Give
  `ids=` when the values alone are not self-describing.
- **Every random draw is seeded**: `np.random.default_rng(20260911)`. An unseeded test that
  fails once a fortnight is worse than no test. Seed from a fixture so it is stated once.

---

## 5. Property-based tests

`hypothesis` is a dev dependency. It is worth reaching for when a property should hold across
a whole domain rather than at a few points:

```python
@given(
    range_m=st.floats(min_value=1.0, max_value=1e5, allow_nan=False),
    scale=st.floats(min_value=1.1, max_value=10.0),
)
def test_power_is_monotonically_decreasing_in_range(range_m: float, scale: float) -> None:
    assert received_power_w(**NOMINAL, range_m=range_m * scale) < received_power_w(
        **NOMINAL, range_m=range_m
    )
```

Good candidates here: Parseval/energy conservation through FFT stages, CFAR false-alarm rate
against its design `pfa`, pattern symmetry for symmetric tapers, exporter round-trips
(`export → load → identical`). Bound the strategies to physically sensible ranges — a
1e300-metre range tests nothing but float overflow.

---

## 6. Golden data

Allowed only where no analytic truth exists — typically a full pipeline's end-to-end output.
Rules:

- Lives **only** in `tests/data/golden/`. The `pre-commit` hook blocks binary arrays
  anywhere else, and `.gitignore` reflects the same rule.
- Keep it small (kilobytes). Store a decimated slice, not a full cube.
- **Ship the generator.** Every golden file has a `scripts/regen_golden_*.py` that recreates
  it deterministically from a seed, committed alongside it.
- Regenerating a golden file is a behaviour change: it needs its own commit, and the body
  must say what changed and why the new values are the correct ones.
- Prefer a checksum of derived statistics (peak position, SNR, detection count) over a raw
  array dump — it fails with a readable message instead of a wall of numbers.

---

## 7. Markers

Registered in `pyproject.toml`; `--strict-markers` means a typo is an error, not a silently
skipped test.

| Marker | Meaning |
| :--- | :--- |
| `slow` | Over a second. Excluded by `make test-fast` |
| `gpu` | Needs a CUDA/Metal device |
| `raytracing` | Needs an optional backend (mitsuba, radarsimpy, ovrtx) |
| `benchmark` | Performance regression, not correctness |

Tests for optional backends skip cleanly when the backend is absent:

```python
mitsuba = pytest.importorskip("mitsuba", reason="requires the raytracing extra")
```

`pre-push` runs **the whole suite**, markers included — that is deliberate. `make test-fast`
is for your inner loop, not for the gate. If the full suite grows painful, mark the
expensive tests `slow` and move them behind a nightly CI job; do not weaken the push gate.

---

## 8. Coverage

`make cov` reports it. There is no enforced threshold, because coverage measures lines
executed, not correctness — and in numerical code a fully covered function can still be
wrong in every value it returns.

Use it as a *finder*: an uncovered branch is a question ("is this reachable? does it need a
test? should it exist?"), not a number to raise. What actually matters:

- Every public function has at least one test against known-correct values.
- Every documented `Raises` has a test that triggers it.
- Every edge case the docstring's `Notes` mentions has a test.

---

## 9. Writing a regression test for a bug

When you fix a numerical bug, the test goes in **first**, fails for the right reason, and its
name records the bug:

```python
def test_range_equation_uses_two_way_propagation() -> None:
    """Regression for #23: the denominator used R^2 (one-way) instead of R^4."""
```

Name the property that was violated, not the issue number alone — the test must still make
sense to someone who never reads the issue.

---

## 10. Radar-domain verification

§1–§9 make a test trustworthy. This section says which radar tests are worth having. A
pipeline can pass every unit test and still put the target in the wrong Doppler bin, quote
an SNR 3 dB high, or run a tracker that is confident and wrong. Every domain check is one
of three kinds:

| Kind | Assert as |
| :--- | :--- |
| Closed form (bin index, gain, null, sidelobe level) | `assert_allclose`, tolerance justified (§2) |
| Monte Carlo statistic (Pfa, noise power, a distribution) | Confidence interval or a seeded KS test, never a tolerance |
| Estimator consistency (NEES, NIS) | χ² interval over N independent runs |

### 10.1 What earns a place **[review]**

Every test runs on every push, so a domain test must clear all five:

1. **It names the bug it catches.** The docstring states, in one sentence, a plausible error
   that would fail it. If you cannot name one, do not write the test.

   ```python
   def test_a_receding_target_lands_in_a_negative_doppler_bin() -> None:
       """Catches a flipped Doppler sign or a missing fftshift: a target placed on bin -8 lands there."""
   ```

2. **It is the cheapest test that catches that bug.** Deterministic before Monte Carlo, unit
   before pipeline. A pipeline test does not re-check what a unit test already pins down.
3. **It tests our code, not NumPy's.** Parseval for `np.fft`, the orthogonality of `I` and `Q`
   in `exp(jωt)`, `chi2.ppf`: those are other projects' tests. Test *our* normalisation,
   sign conventions and algebra.
4. **It fits the budget.**
   - Unmarked tests run in under about 0.5 s.
   - Size a Monte Carlo run from the error it must detect, not from "big N". About 100
     expected events gives a ±20 % interval. That catches a wrong threshold factor or a wrong
     exponent, and those are the bugs that occur. Pfa at 1e-3 or 1e-4 catches the same bugs as
     1e-6 at a hundredth of the cost.
   - Share an expensive run between its tests with a module-scoped fixture.
   - Anything over a second is marked `slow` with the reason in its docstring
     (`uv run pytest --durations=20` finds them).
5. **Its setup makes physical sense.** A test can pass and still convince nobody. Check that:
   - each target's range, altitude and RCS are plausible for its class, and its accelerations
     and turn rates are ones that kind of aircraft can fly;
   - the radar parameters could exist together in one sensor (PRF, duty cycle, pulse width);
   - the window suits what the test measures: a low-sidelobe taper does not belong in a
     resolution test;
   - an expected bin index is right for the right reason, and a threshold sits near a plausible
     operating point;
   - an unfolding test's target actually folds;
   - the test still passes away from its nominal parameters.

   A test that fails this check is either **weak**, meaning it passes for an unconvincing
   reason, or **wrong**, meaning it asserts something untrue or unphysical. Rewrite a weak
   test's setup or assertion. Fix a wrong one, and say in the commit what it was hiding.
   Labelling the test isn't enough.

Not tests:

- Whole-curve reproductions (ROC sweeps, Pd against SNR, pattern plots). These belong in a
  notebook; the test suite checks one point on the curve.
- Two-target resolution demonstrations. They follow from the exact-bin and window tests.
- Checks for features that do not exist yet.

The tables below list what each stage should have *once its module exists*. Where a test
already does the job, the row names it.

### 10.2 Front end and IQ synthesis

| Property | Truth | Bug it catches |
| :--- | :--- | :--- |
| LFM instantaneous frequency | Sweeps f_c → f_c + B linearly over T_p; slope K = B/T_p | Missing ½ in φ(t) = 2π(f_c t + ½Kt²), so the slope doubles |
| Thermal noise power | Complex variance k_B·T·F·B, **half per I and Q channel**; χ² interval on the sample variance | Link budget off by 3 dB (per-channel vs total), or off by a factor of B |
| I/Q imbalance injection | Recover the ε and Δφ you injected from a test tone | Imbalance applied to the wrong channel, or with the wrong sign |
| Swerling RCS | Cases 1/2 exponential (χ², 2 DoF), cases 3/4 χ² with 4 DoF; one-sample KS against the CDF. Cases 1/3 hold over a scan, 2/4 decorrelate pulse to pulse | Wrong DoF; scan-to-scan and pulse-to-pulse swapped |
| Clutter amplitude (Weibull, K) | One-sample KS test against the closed-form CDF | Shape and scale parameters swapped |

### 10.3 Fast/slow-time DSP and the RD map

| Property | Truth | Bug it catches |
| :--- | :--- | :--- |
| Target placement | A target set exactly on a bin lands in **that** range bin and **that** Doppler bin, f_d = 2v_r/λ, with its sign | Doppler sign, fftshift, off-by-one, one-way vs two-way delay |
| Scaling | The stage's documented normalisation holds (Parseval, once per convention, not per stage) | An unscaled FFT that moves the noise floor by 10 log₁₀ N |
| Window | Peak sidelobe and processing loss per Harris (1978): Hann −31.5 dB / 1.76 dB, Hamming −42.7 dB / 1.34 dB; see `tests/core/test_windows.py` | Wrong window, or a periodic window where a symmetric one is needed |
| Compression gain | Noise-free SNR gain equals the time–bandwidth product B·T_p minus the window's processing loss | Matched filter not conjugated or not time-reversed, or the window applied twice |
| MTI | Null at f_d = k·PRF below the peak by a stated depth (§2 near-zero rule); blind speeds at k·λ·PRF/2 | Canceller taps or PRF unit wrong |

The −3 dB mainlobe width of an unweighted response is about 0.886·c/(2B). c/(2B) is the
peak-to-first-null spacing, and asserting it as the −3 dB width fails a correct
implementation.

### 10.4 Detection and hit extraction

| Property | Truth | Bug it catches |
| :--- | :--- | :--- |
| False-alarm rate | Design Pfa inside the Clopper–Pearson interval of the measured rate (`assert_rate_consistent_with_design` in `tests/core/test_detection.py`) | Threshold factor wrong by a constant or in its exponent |
| Clutter edge, masking | CA spikes at a step, GO suppresses it; OS and SO hold a target next to a strong one (the existing tests in `test_detection.py`) | A variant behaving like CA |
| Centroid interpolation | Exact vertex recovery on three samples of a true parabola | Offset sign, or the wrong neighbour |

A bias or variance test for interpolation on a real point-spread function is needed only
when the docstring claims an accuracy. The bias depends on the window, so derive the bound
for the window in use.

### 10.5 Measurements and coordinates

| Property | Truth | Bug it catches |
| :--- | :--- | :--- |
| Spherical ↔ Cartesian, with range rate | Round trip to `rtol=1e-12` | Azimuth convention (from north vs from east), elevation vs zenith |
| Jacobian | Analytic matches a central difference | One wrong partial derivative |
| Converted covariance J R Jᵀ | Symmetric to `rtol=1e-12`, and Cholesky succeeds | Transposed Jacobian; loss of positive definiteness |

Monopulse slope and MUSIC/ESPRIT resolution tests arrive with those estimators.

### 10.6 Tracking

`spec/tracker-001.md` §12 is the authority; this is the summary.

| Property | Truth | Bug it catches |
| :--- | :--- | :--- |
| NEES, NIS | Sum over N runs inside the 99.9 % χ² interval, parametrised over every estimator; runs shared through a fixture, `slow` (`tests/core/tracking/test_ukf.py`) | A filter that tracks well but is overconfident |
| Gate | Accepts the fraction of true measurements it claims (`test_association.py`) | Gate on the wrong dimension or the wrong quantile |
| Lifecycle | Confirms on exactly the M-th of N hits; deletes on exactly the K-th miss | Off-by-one in the counters |
| Crossing targets | IDs kept, on geometry where the right answer is unambiguous | Cost matrix transposed |

Innovation whiteness and "covariance grows while coasting" are not required. NEES and NIS
already catch the bugs they would catch, and the second does not hold for every motion model.

### 10.7 ML enhancements

These tests run with the `ml` extra and use `pytest.importorskip("torch")`.

| Property | Truth | Bug it catches |
| :--- | :--- | :--- |
| **A learned detector holds its Pfa** | The same Clopper–Pearson test as §10.4, on noise only | A model that buys Pd with false alarms, which a Pd figure alone hides |
| Robustness | Outputs finite over −20 to +40 dB SNR | NaN or overflow outside the training range |
| Flow invertibility | ‖f⁻¹(f(x)) − x‖ ≤ 1e-5 in float32 | A non-invertible layer |
| Equivariance | Shifting the input shifts the output, only where the architecture guarantees it (e.g. circular-padded convolutions) | Padding that breaks the guarantee |

"Pd no worse than the classical detector" belongs under the `benchmark` marker. CPU and GPU
outputs must agree to a stated tolerance, not bitwise, under the `gpu` marker.

### 10.8 End to end

Two tests per pipeline, and no more:

- **Link budget.** A point target goes through the whole chain. The processed SNR equals the
  radar equation (`core/radar_equation.py`), plus integration gain, minus window and
  processing losses, within a tolerance justified in a comment.
- **Detection probability at one SNR point** per fluctuation model, against an exact form:
  - Swerling 1, one pulse, square-law detector: Pd = Pfa^(1/(1+SNR)).
  - Swerling 0: Marcum Q.

  Use Albersheim or Shnidman only where no exact form exists, with their published error as
  the tolerance.

### 10.9 When a statistical test fails

A statistical test fails 0.1 % of the time by design at 99.9 %. With a fixed seed, though, it
either always passes or always fails, so a new failure means the code changed. Changing the
seed until it passes is loosening a tolerance (§2). Raising the confidence level needs the
same justification in the commit body.

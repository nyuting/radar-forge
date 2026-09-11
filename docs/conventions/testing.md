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
def test_matches_richards_worked_example() -> None:
    """Richards, FRSP 2e, §2.2: the worked example returns -103.6 dBm at 10 km."""
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

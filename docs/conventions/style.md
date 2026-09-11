# Coding style

`ruff` (formatting + lint) and `mypy --strict` enforce the mechanical half of this document;
`make fmt` fixes most of what they catch. What follows explains the half a linter cannot
check — and, for the mechanical rules, *why* they are switched on.

The single best reference is the worked example:
[`src/radar_forge/core/radar_equation.py`](../../src/radar_forge/core/radar_equation.py).
It demonstrates every convention below at once. Copy its shape.

---

## 0. The audience rule

This library is read by interns and students learning radar. When clarity and cleverness
conflict, clarity wins — every time, without discussion. A vectorised one-liner that needs a
paragraph to explain is worse than four named intermediate steps.

Concretely: name the intermediate quantities of a derivation after the symbols in the
reference, so a reader holding the textbook can follow along.

```python
# Good — the reader can match this against Richards eq. 2.11.
numerator = transmit_power_w * gain_tx * gain_rx * wavelength_m**2 * rcs_m2
denominator = (4 * np.pi) ** 3 * range_m**4 * loss
return numerator / denominator
```

---

## 1. Language and configuration

- **Python ≥ 3.11.** Use modern syntax: `X | None` over `Optional[X]`, builtin generics,
  `match` where it genuinely reads better. `from __future__ import annotations` at the top of
  every module regardless.
- **All configuration is TOML, in `pyproject.toml`** — build, deps, ruff, mypy, pytest,
  coverage. No `setup.py`, `setup.cfg`, `.flake8`, `tox.ini`, `requirements*.txt`, or a
  standalone `ruff.toml`/`mypy.ini`. `scripts/check_conventions.py` blocks them at commit
  time. One file means one place to look, and no chance of two tools disagreeing.
- **src layout.** Everything importable as `radar_forge` lives under `src/radar_forge/`;
  nothing else does. This makes it impossible to accidentally test against the source tree
  instead of the installed package.
- Line length **100**. Radar code carries long but meaningful names; 79 would force
  abbreviation, which costs more than it saves.

---

## 2. Units — the most important convention here

Unit confusion is the single most common source of wrong answers in radar code, and it is
silent: the array still has the right shape. So we put the unit in the name.

**Every physical quantity carries its unit as a suffix.**

| Suffix | Unit | Example |
| :--- | :--- | :--- |
| `_m` | metres | `range_m`, `wavelength_m`, `spacing_m` |
| `_s` | seconds | `chirp_time_s`, `pri_s`, `delay_s` |
| `_hz` | hertz | `f0_hz`, `bandwidth_hz`, `doppler_hz` |
| `_rad` / `_deg` | angle | `azimuth_rad`, `beamwidth_deg` |
| `_w` / `_dbm` / `_dbw` | power | `transmit_power_w`, `noise_power_dbm` |
| `_m2` / `_dbsm` | RCS | `rcs_m2`, `rcs_dbsm` |
| `_mps` | velocity | `radial_velocity_mps` |
| `_db` / `_dbi` | ratio / gain | `snr_db`, `gain_dbi` |

Rules:

1. **Linear internally, decibels only at the boundary.** Every internal computation is in
   linear SI. Convert at the API edge, and name the converted quantity `_db*`. A function
   never accepts "gain" and guesses whether you meant 30 or 1000.
2. **Angles in radians internally** (`_rad`), because that is what NumPy's trig takes.
   User-facing constructors may accept `_deg` for convenience — but then the parameter is
   named `_deg` and converted immediately, on the first line.
3. **No bare unit names.** `range`, `power`, `gain`, `freq` are all rejected in review; they
   shadow builtins, read ambiguously, or hide a dB/linear mix-up.
4. **State the unit in the docstring too**, including the linear-vs-dB expectation and how
   to convert. See `received_power_w` for the pattern.

---

## 3. Naming

- `snake_case` functions and variables, `PascalCase` classes, `UPPER_SNAKE` module constants.
- Array-dimension counts are `n_`-prefixed: `n_chirps`, `n_samples`, `n_elements`, `n_targets`.
  Use the same names in shape annotations so `(n_chirps, n_samples)` is unambiguous everywhere.
- Single letters are allowed **only** where they are the standard symbol in the cited
  reference, and only inside a function whose docstring maps them: `R`, `sigma`, `lambda_`
  (trailing underscore — `lambda` is a keyword). Never as a parameter of a public function.
- Verb-first function names for actions (`compute_`, `estimate_`, `export_`, `simulate_`);
  noun names for quantities returned (`received_power_w`, `beam_pattern_db`).
- `_leading_underscore` for everything not in the public API (§6).

---

## 4. Docstrings

**NumPy style** (enforced by `ruff` `pydocstyle` with `convention = "numpy"`), on every
public module, class and function. Sections, in order:

```
Summary line, imperative, one line, ends with a period.

Extended description. The maths, in LaTeX, matching the reference's notation.

Parameters
Returns
Raises          (whenever the function validates its inputs)
Notes           (assumptions, limitations, numerical caveats)
References      (required for any physics or DSP — see below)
Examples        (doctest-formatted; they are real, runnable examples)
```

Two requirements specific to this project:

**Cite your source.** Any function implementing physics or a DSP block carries a
`References` section naming the textbook section, equation number, or paper. This is what
makes the library teachable, and it is what lets a reviewer check the maths rather than
trusting it. Use the reference's own notation in the LaTeX so the two can be read side by
side.

```
References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §2.2 (eq. 2.11).
```

**Write out array shapes.** Every array parameter and return states its shape using the
`n_` names, and says what the axes *mean*:

```
Parameters
----------
baseband : numpy.ndarray
    Complex baseband cube of shape ``(n_chirps, n_samples, n_rx)``; fast time
    along axis 1, slow time along axis 0.
```

---

## 5. Typing

`mypy --strict` runs over `src/` and must pass.

- **Annotate everything public, completely.** Defaults included.
- Arrays: accept `numpy.typing.ArrayLike`, return `numpy.typing.NDArray[np.float64]` (or
  `np.complex128`). Being liberal on input and precise on output means callers can pass a
  list, and the type checker still knows what came back.
- Be explicit about dtype in the return type. `NDArray[Any]` defeats the point.
- No `# type: ignore` without a trailing comment saying why and, ideally, an issue link.
- `TYPE_CHECKING` imports for anything needed only for annotations, so optional heavy
  backends (`torch`, `mitsuba`) never become an import-time cost.

Shapes are not in the type system; that is why §4 requires them in the docstring. If shape
errors become a recurring source of bugs, runtime validation belongs in a shared helper, not
in scattered `assert`s.

---

## 6. Public API

**The public API is exactly what a package's `__init__.py` re-exports in `__all__`.**
Anything else — even without a leading underscore — is an implementation detail that may
change in any release.

- Each subpackage `__init__.py` imports its public names and lists them in `__all__`.
- Private helpers get a `_` prefix. Modules that are wholly internal get one too
  (`_backends.py`).
- Keep `__init__.py` import-light. Optional backends are imported lazily, inside the function
  that needs them, and raise a clear `ImportError` naming the extra to install:

  ```python
  msg = "Ray tracing requires the raytracing extra: pip install 'radar-forge[raytracing]'"
  raise ImportError(msg) from exc
  ```

---

## 7. NumPy practice

- **Vectorise.** Python loops over array elements are rejected in review. If a loop is
  genuinely unavoidable (a sequential filter, a tracker update), write a comment saying why.
- Broadcast rather than tile: a function taking scalars should also accept arrays for those
  arguments, with no extra code. `received_power_w` is the model — every argument broadcasts.
- Always pass `dtype=np.float64` / `np.complex128` explicitly at array construction.
  Silent float32 demotion is very hard to spot and changes results at the tolerances we test at.
- Use `np.asarray` (no copy when possible), not `np.array`, to normalise inputs.
- `rng = np.random.default_rng(seed)` — never the legacy `np.random.*` global functions.
  Any function with randomness takes a `rng` or `seed` parameter; reproducibility is not
  optional in a dataset-synthesis library.
- Prefer `np.pi`-based expressions in a named constant when reused (`_FOUR_PI_CUBED`), so
  the reader sees the equation rather than a magic number.

---

## 8. Errors

- Validate inputs at public API boundaries, and say what is wrong *and what is expected*:

  ```python
  msg = "range_m must be strictly positive; the range equation diverges at R = 0."
  raise ValueError(msg)
  ```

  Assign the message to `msg` first (ruff `EM`-style, and it keeps the `raise` line short).
- `ValueError` for bad values, `TypeError` for bad types, `ImportError` for a missing
  optional backend. Never a bare `Exception`, never a silent clamp.
- Do not use `assert` for validation — `python -O` removes it.
- `warnings.warn` for a recoverable situation the user should know about (aliasing, an
  under-sampled pattern), with a `stacklevel` so it points at the caller.

---

## 9. Imports and modules

- Ordered by `ruff` isort: stdlib, third-party, first-party `radar_forge`, local.
- Absolute imports (`from radar_forge.core import ...`), never relative.
- **Every module has a docstring** saying what it models and citing its reference.
  `scripts/check_conventions.py` enforces the existence; review enforces the content.
- One coherent concept per module. When a module passes ~400 lines, that is a prompt to ask
  whether it is doing two things.

---

## 10. Comments

Comment the **why**, never the what. The code already says what it does.

```python
# Bad
range_arr = np.asarray(range_m)  # convert range to an array

# Good
# asarray rather than array: callers frequently pass an existing ndarray in a
# hot loop, and a defensive copy per call showed up in the profiler.
range_arr = np.asarray(range_m, dtype=np.float64)
```

Mark deliberate deviations from a reference, and known approximations, explicitly — a future
reader must be able to tell an approximation from a bug.

```python
# Small-angle approximation: valid to <0.1 dB within the 3 dB beamwidth,
# which is the only region export_coco() samples. See issue #17.
```

# Coding style

Copy the shape of [`src/radar_forge/propagation.py`](../../src/radar_forge/propagation.py) (the
radar equation is its `received_power_w`); it follows every rule here. The tools enforce most of
them, and `make fmt` fixes much of what they catch. This page says where the enforced rules
are listed, then gives the judgement calls a tool cannot make, which are what review checks.

## What the tools enforce

`make check` runs ruff (the `select` list in `pyproject.toml` names each rule family and why),
`mypy --strict`, [`scripts/check_conventions.py`](../../scripts/check_conventions.py) and
pytest. The checker's module docstring is the one list of the R-rules, and each failure prints
the rule, the fix and where it is documented. `tests/test_conventions.py` checks `__all__`,
`ArrayLike` in and `NDArray` out, and that every doctest runs. When a judgement call below turns
out to be checkable, add it there or to a test rather than writing more prose.

## 0. The audience rule

The library is read by interns and students learning radar. When clarity and cleverness
conflict, clarity wins. Name the intermediate quantities of a derivation after the symbols in
the cited reference, so a reader holding the textbook can follow along.

## 1. Language and configuration

Enforced (table above). Use `X | None`, builtin generics and `from __future__ import annotations`.

## 2. Units

R5 checks parameter names. The rest is judgement:

1. **Linear internally, decibels only at the API boundary**, and the converted name says `_db*`.
2. **Radians internally.** A constructor may take `_deg` for convenience and converts on its
   first line.
3. **Locals follow the same rule.** R5 checks only parameters; review checks the rest.
4. **The docstring states the unit**, including linear or dB, as `received_power_w` does.

Suffixes: `_m _m2 _s _hz _rad _deg _w _dbm _dbw _db _dbi _dbsm _mps _k _linear _bins _samples`
and the scaled forms in `UNIT_SUFFIXES`. Counts are `n_`-prefixed. To add a constant, give it
`Final`, a docstring with its source, an `__all__` entry, and add its near-miss approximations
to `CONSTANT_LITERALS`.

## 3. Naming

- Counts are `n_pulses`, `n_samples`, `n_targets`, and shape annotations use the same names.
- `_index` is a position in an ordered sequence; `_id` is a key that survives reordering.
  Never index an array with an id. Never write `_idx`.
- Single letters only where they are the cited reference's symbol, inside a function whose
  docstring maps them, never as a public parameter.
- Verb-first names for actions (`estimate_`, `export_`), noun names for returned quantities
  (`received_power_w`). A verb keeps one meaning across the package.
- Name a thing for what it does or holds, not who it is for (`visualisation`, not `teaching`).
- Never reuse a term the field or the repo already owns: a CFAR estimate is
  `cfar_noise_estimate_w` because `noise_power_w` is thermal noise.
- Never shadow a parameter, builtin or module with a local or loop variable.

### 3.1 Data layout and array names

The cube is `(n_pulses, n_samples)`: slow time on axis 0 (becomes Doppler), fast time on axis 1
(becomes range); extra channels go last. A *pulse* is one slow-time sample on both paths, one
chirp of an FMCW dwell included. An array that crosses a function boundary is named for its
domains, fast axis first: `data_ftst` (raw baseband), `data_rfst` (after range compression),
`data_rv` (the range-Doppler map). `img` and `img_db` are for display only; never feed one back
into a detector. On disk, the layout is [data-001](../../spec/data-001-formats.md)'s.

### 3.2 Spelling

British spelling (`normalise`, `behaviour`, `visualisation`, `licence` as a noun) in names,
docs, comments and commits, except a name that mirrors an outside API (matplotlib's `color=`),
a quotation, or a released name that renaming would break. Say why in a comment if not obvious.

### 3.3 Circular axes

Doppler is always a circle; range is one when returns come from beyond one repetition interval;
so is azimuth. Declare the period once where the axis is made (`Coordinate.period`,
`RangeDopplerProduct.range_period_m`) and apply it where values are consumed (`wrap`,
`StateLayout.residual`). Never pad, roll or `np.interp` across the edge. Convert an index to a
value with the axis's own formula, so a fractional index wraps. Keep a fold half-open (`x % p` can
round to `p`). Average angles about a nearby reference, not as unit vectors. Unwrap a history
before fitting a slope to it.

## 4. Docstrings

Lean NumPy style: one summary line; a citation with equation number; Parameters and Returns only
where a unit, shape or convention is not already in the name; no Raises or Examples unless the
example is the test. Doctests that are present run (`tests/test_conventions.py`).

- **Cite the source.** Physics or DSP carries a `References` entry naming the section or
  equation, e.g. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., §2.2.1.
- **Write out array shapes** with the `n_` names, where a shape is documented.
- **LaTeX goes in a raw string**, `r"""…"""`. Never write source through a string that reads
  escapes (Python in a shell heredoc): `\alpha` lands as BEL (R9 catches it).

### 4.1 Checking a citation

A citation passes only if the record matches the publisher's (authors, title, venue, pages,
year), the cited section or equation number can be seen in the source or its table of contents,
and that section's subject supports the sentence. Evidence is the source, its scanned contents,
or the publisher or Crossref record; memory and search summaries are not. If the number cannot
be seen, cite the level you can see or remove the citation. Every `References` entry is cited
in the text. The commit message records each correction with its evidence.

## 5. Typing

Accept `ArrayLike`, return `NDArray[np.float64]` (or `complex128`), never `NDArray[Any]`. A
`# type: ignore` names its error code and says why. Optional heavy backends are imported under
`TYPE_CHECKING` or inside the function that needs them.

## 6. Public API

The public API is what a package's `__init__.py` re-exports in `__all__`; every module also
defines `__all__` (`tests/test_conventions.py`). Private helpers and wholly internal modules get a
leading `_`. Keep `__init__.py` import-light: an optional backend is imported inside the function
and raises `ImportError` naming the extra to install.

## 7. NumPy practice

- **Vectorise.** A Python loop that must stay (a sequential filter, a tracker update) carries a
  comment saying why.
- **Broadcast** so scalar arguments also accept arrays with no extra code (R6 for `np.tile` and `np.repeat`).
- Pass `dtype=np.float64`/`np.complex128` explicitly at construction; use `np.asarray` on input.
- Randomness takes an `rng: np.random.Generator` or a `seed` from the caller.
- **A floating-point warning is a finding.** If the NaN or inf is the documented answer, mask it
  explicitly or scope the narrowest `np.errstate` category to one expression with
  `# errstate-exempt: <reason>` (R12), and test that it appears. If rearranging the maths avoids
  it, do that. If it means bad input, raise `ValueError` at the boundary. Never `np.seterr` or a
  `warnings` filter in `src/`.

## 8. Errors

- **Guards first.** Validation is one block of `if …: msg = "…"; raise ValueError(msg)` at the
  top; the maths follows uninterrupted, so code that mixes the two is harder to read.
- The message says what is wrong and what is expected.
- `ValueError` for bad values, `TypeError` for bad types, `ImportError` for a missing extra.
  Never a bare `Exception`, never a silent clamp.
- **Never drop data silently.** Dispatch on a kind through a dict; an unknown key raises.
- **Raise your own error at a module boundary**, chained with `from`.
- **An error leaves the state as it was**: a method that changes several objects finishes or
  restores them all before re-raising (`Tracker.process` does).
- `warnings.warn(…, stacklevel=2)` for a recoverable condition the user should know about.

## 9. Imports and modules

Absolute imports only. One coherent concept per module; R10 warns at 500 lines and fails at 800.
The module docstring (ruff D100) says what the module models and cites its reference.

## 10. Comments

Comment the **why**, never the what. Mark a deliberate deviation from the reference, or an
approximation, so the next reader can tell it from a bug. A `# noqa` is a last resort and names
its rule and reason on the same line.

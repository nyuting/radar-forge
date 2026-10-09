<!-- The PR title becomes the squash-merge subject: it must be a valid
     Conventional Commit, e.g. `fix(core): correct 1/R^4 term in range equation`.
     A breaking change takes `!` after the scope and a `BREAKING CHANGE:` paragraph below.
     See docs/conventions/commits.md. -->

## What and why

<!-- What changed, why, and what a reviewer should check. One logical change per PR.
     Breaking change: add a paragraph starting `BREAKING CHANGE:` saying what callers must do. -->

## Related issues

<!-- Delete if none. Otherwise `Closes #123` — one per line. -->

## References

<!-- New physics or DSP: textbook section or paper implemented. For a numerical
     change: what the number was, what it is now, and the source for the right value. -->

## Plots

<!-- Delete if none. Range-Doppler maps, spectra or before/after comparisons for a
     DSP or physics change. These help the reviewer; they do not replace a test. -->

## New dependencies

<!-- Delete if none. Otherwise: what it buys, why numpy/scipy cannot do it,
     and whether it belongs in an extra rather than core. -->

## Checklist

- [ ] `make check` passes (lint, types, conventions, full tests). No `--no-verify`.
- [ ] Physical quantities are SI with the unit in the name (`range_m`, `f0_hz`); dB only at API boundaries, named `_db*`
- [ ] Constants come from `radar_forge.core.constants`; no hard-coded values such as `3e8`
- [ ] Public API has NumPy-style docstrings with `References` and array shapes
- [ ] Tests check against analytic ground truth where one exists; no tolerance was loosened
- [ ] No generated arrays committed; small golden data only in `tests/data/golden/`
- [ ] Any breaking API change has `!` in the title and a `BREAKING CHANGE:` paragraph

# Commit and branch conventions

Enforced by `.githooks/commit-msg` (locally) and the commit-lint CI job (on PRs).

## Why

A readable history is what lets you answer "when did the CFAR false-alarm rate change, and
why" six months from now. Conventional Commits also give us a machine-readable changelog for
free, which matters once the package is on PyPI and users need to know what broke.

---

## Commit subject

```
<type>(<scope>)?<!>?: <summary>
```

- **≤ 72 characters**, imperative mood ("add", not "added" or "adds"), **no trailing period**.
- Lowercase after the colon, unless the first word is a proper noun (`FFT`, `Mitsuba`).
- Describe the *change*, not the file: `fix(core): correct 1/R^4 term in the range equation`,
  not `fix(core): update radar_equation.py`.

### Types

| Type | Use for |
| :--- | :--- |
| `feat` | New user-visible capability |
| `fix` | Bug fix — including a numerical correction |
| `docs` | Documentation only |
| `style` | Formatting only, no behaviour change |
| `refactor` | Restructuring with no behaviour change |
| `perf` | Performance improvement (say the speedup in the body) |
| `test` | Adding or fixing tests only |
| `build` | Packaging, `pyproject.toml`, dependencies |
| `ci` | CI workflows |
| `chore` | Tooling, hooks, housekeeping |
| `revert` | Reverting a previous commit |

### Scopes

`core`, `array`, `raytracing`, `pipelines`, `teaching`, `docs`, `ci`, `deps`, `hooks`.
These mirror the package layout. The scope is optional but nearly always worth adding; the
hook checks its *shape* (lowercase, hyphens) rather than its membership in this list, so a
new subpackage does not require a hook change.

### Examples

```
feat(array): add Taylor tapering for planar arrays
fix(core): correct 1/R^4 term in the range equation
perf(raytracing): batch ray casts, 8x faster on the highway scene
docs: document the SI unit suffix convention
test(core): add analytic CFAR false-alarm-rate check
build(deps): require numpy>=1.26 for NEP 50 casting rules
refactor(pipelines): extract the COCO writer from the exporter
```

Rejected:

```
update stuff                  no type
feat: Added Taylor tapering.  past tense, trailing period
fix(core): fix bug            says nothing
```

---

## Commit body

Optional, but include one whenever the *why* is not obvious from the subject. Wrap at 72
columns, separated from the subject by a blank line. For anything numerical, say what the
number was, what it is now, and what it should be according to which reference:

```
fix(core): correct 1/R^4 term in the range equation

The denominator used R^2, which is the one-way form. Two-way propagation
loses another R^2 on the return path.

Richards, Fundamentals of Radar Signal Processing 2e, eq. 2.11.
Received power at 100 m drops 40 dB, now matching the textbook worked
example in §2.2 to within 0.01 dB.
```

## Breaking changes

Append `!` after the type/scope and explain in the body under a `BREAKING CHANGE:` footer:

```
feat(array)!: return gain in dBi rather than linear

BREAKING CHANGE: ULA.pattern() now returns dBi. Multiply by 10**(x/10)
to recover the previous linear values.
```

Pre-1.0 this is still required — the point is that users can see it, not that semver
obliges us yet.

## Trailers

```
Closes #42
Co-Authored-By: Name <email>
```

---

## Branches

```
<type>/<short-kebab-summary>
```

Same type vocabulary as commits: `feat/taylor-tapering`, `fix/range-equation-exponent`,
`docs/units-convention`, `chore/pin-numpy`.

- Branch from `main`; keep branches short-lived.
- `main` is protected by the `pre-push` hook. Override for solo maintenance with
  `RADAR_FORGE_ALLOW_MAIN_PUSH=1 git push`.
- Rebase on `main` rather than merging it in, so the history stays linear.
- Tidy up before pushing for review: `git rebase -i main` to squash `fixup!` noise. Once
  someone has reviewed, stop rewriting — push follow-up commits so the reviewer can see
  what changed.

## Pull requests

- **The PR title must itself be a valid Conventional Commit** — we squash-merge, so it
  becomes the commit subject on `main`.
- The PR description carries the commit body: what changed, why, what a reviewer should
  check, and the reference for any new physics.

## Changelog and releases

`CHANGELOG.md` is generated from the commit history, grouped by type, once the first
release is cut. Nothing to maintain by hand — which is the whole return on the discipline
above.

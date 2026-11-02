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
| `refactor` | Restructuring with no behaviour change ([refactoring.md](refactoring.md)) |
| `perf` | Performance improvement (say the speedup in the body) |
| `test` | Adding or fixing tests only |
| `build` | Packaging, `pyproject.toml`, dependencies |
| `ci` | CI workflows |
| `chore` | Tooling, hooks, housekeeping |
| `revert` | Reverting a previous commit |

### Scopes

`core`, `array`, `raytracing`, `pipelines`, `visualisation`, `docs`, `ci`, `deps`, `hooks`.
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

Richards, Fundamentals of Radar Signal Processing 2e, §2.2.1.
Received power at 100 m drops 40 dB, the extra factor of R^2 = 10^4.
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
- Before review, rebase on `main` rather than merging it in. Tidy up then too:
  `git rebase -i main` to squash `fixup!` noise. Once someone has reviewed, stop rewriting.
  Push follow-up commits so the reviewer can see what changed, and merge `main` in if the
  branch needs it.
- Work on a branch in its own worktree: `git worktree add ../radar-forge-<name> <branch>`,
  then `uv sync` the extras there (CLAUDE.md lists them). Sessions that share one checkout
  share its index and `HEAD`, and one session's commit sweeps up another's staged files.
- A branch that replaces an abandoned one keeps its name with a `-v2` suffix, and its PR
  says "Replaces #N".

### Stacked PRs

A PR whose base is another PR's branch merges into that branch, not into `main`. If the base
merges first, the child's merge lands on a branch nothing reads, and its commits never reach
`main` (#47 and #49 were lost this way). Before merging the base, retarget the child to
`main`. After merging any PR, check that `git log origin/main` contains its commits.

### After a merge

```sh
git switch main && git pull --ff-only     # in the main checkout
git worktree remove ../radar-forge-<name>
git branch -d <branch> && git push origin --delete <branch>
```

`--ff-only` fails rather than making a merge commit if local `main` has diverged. That
happens only if something was committed to it by mistake, so find out what before going on.

## Pull requests

- **PRs land as merge commits**, so every commit on the branch reaches `main` as it is and
  must be a valid Conventional Commit on its own. The `commit-msg` hook checks each one.
- **The PR title must itself be a valid Conventional Commit.** It becomes the body of the
  merge commit, and is what the PR list shows.
- The PR description carries the commit body: what changed, why, what a reviewer should
  check, and the reference for any new physics.
- If a rebase hit a conflict, the description names each one and says which side's
  behaviour won and why.

## Changelog and releases

`CHANGELOG.md` is generated from the commit history, grouped by type, once the first
release is cut. Nothing to maintain by hand — which is the whole return on the discipline
above.

# Commits, branches and pull requests

The `commit-msg` hook checks every commit locally; CI checks every commit on a PR and the PR title.

## Commit subject

`<type>(<scope>)?!?: <summary>`: at most 72 characters, imperative ("add", not "added"), lowercase
after the colon unless a proper noun, no trailing period. Describe the change, not the file.

Types: `feat` (new capability), `fix` (bug or numerical correction), `docs`, `style`, `refactor`
(no behaviour change), `perf` (state the speed-up), `test`, `build` (packaging, dependencies),
`ci`, `chore` (tooling, hooks), `revert`. The scope is optional and mirrors the package
(`dsp`, `tracking`, `pipeline`, `figures`, `deps`, `hooks`, …); the hook checks its shape only.

```
fix(propagation): correct 1/R^4 term in the range equation
perf(raytracing): batch ray casts, 8x faster on the highway scene
```

## Commit body

Say why, when the subject does not. For a numerical change: what the number was, what it is now,
and which reference says so. A breaking change takes `!` after the type or scope and a
`BREAKING CHANGE:` paragraph saying what callers must do. Trailers (`Closes #42`,
`Co-Authored-By:`) go last.

## Branches

`<type>/<short-kebab-summary>` from `main` (`feat/taylor-tapering`), short-lived, one per
worktree (`git worktree add ../radar-forge-<name> <branch>`, then `uv sync` the extras there).
The `pre-push` hook blocks a direct push to `main`. Rebase on `main` before review; after review
starts, push follow-up commits instead of rewriting. A stacked PR is retargeted to `main` before
its base merges, or its commits never reach `main`.

## Pull requests

- **PRs land by squash merge.** The PR title becomes the commit subject on `main`, so it must be
  a valid Conventional Commit; CI checks it. The description becomes the body: what changed, why,
  what to check, and the reference for any new physics.
- The branch is deleted on merge. Then `git switch main && git pull --ff-only` and remove the
  worktree.
- The changelog is generated from `main`'s history, grouped by type.

## Integration branches

A change too large for one review (a refactor in several steps) goes through an integration
branch, `refactor/<name>` from `main`, and reaches `main` in one PR. Each step's PR targets the
integration branch and is squash-merged into it once its fast CI jobs and commit-message check
pass, an independent review's findings are fixed and, for a refactor, a run before and after
shows that nothing computes differently. It does not wait for the full `check` jobs, which take
hours. After each merge, run them on the integration branch with
`gh workflow run ci.yml --ref refactor/<name>` (a newer run cancels an older one, and the newest
covers every merge before it) and fix what they find before the PR into `main`, which waits for
the full suite like any other.

A PR stacked on a step that squash-merges still holds that step's own commits. Merge the
integration branch into it before review goes on; when the integration branch's tree is the one
the stack was built on (`git diff <that commit> origin/refactor/<name>` prints nothing),
`git merge -s ours origin/refactor/<name>` records the squash without touching a file.

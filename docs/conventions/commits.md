# Commits, branches and pull requests

The `commit-msg` hook checks every commit locally; CI checks every commit on a PR and the PR title.

## Commit subject

`<type>(<scope>)?!?: <summary>`: at most 72 characters, imperative ("add", not "added"), lowercase
after the colon unless a proper noun, no trailing period. Describe the change, not the file.

Types: `feat` (new capability), `fix` (bug or numerical correction), `docs`, `style`, `refactor`
(no behaviour change), `perf` (state the speed-up), `test`, `build` (packaging, dependencies),
`ci`, `chore` (tooling, hooks), `revert`. The scope is optional and mirrors the package
(`core`, `pipelines`, `visualisation`, `deps`, `hooks`, …); the hook checks its shape only.

```
fix(core): correct 1/R^4 term in the range equation
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

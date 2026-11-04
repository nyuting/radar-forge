---
name: pr-review
description: Review or re-review a pull request against radar-forge's conventions and spec, and draft the review for the author. Use when asked to "review PR", "re-review" or "check PR", or to audit existing code against the spec.
---

# Reviewing a pull request

How to review a PR here and write it up, for people and AI agents alike. Claude Code loads this
file as `/pr-review`. The same lists serve an audit of existing code (§5) and a refactor (§6).

Readers are interns and students, so a review teaches as well as gates. It holds a PR to both
halves of the bar in [`CLAUDE.md`](../../CLAUDE.md) (its opening paragraph): a current method with
correct, cited maths, in code a newcomer can follow. **Every claim is run, not argued**: a root
cause, a bug, a passing test, a number. Anything unrun goes under "discuss", marked unverified.

## 1. Procedure

1. **Fix the base.** `git fetch origin main pull/N/head:pr-N`, then diff against the merge base:
   `git diff --stat origin/main...pr-N` (three dots). Write down both commits. Read every
   modified file, `__init__.py` exports included. Check that CI ran.
2. **Run the gates** in a worktree of the PR: `make check`. Record ruff, mypy, conventions and the
   test count. A green lint is the start, not the review.
3. **List the PR's own claims** (description, docstrings, commit bodies). Each is a check.
4. **Read the spec it touches before the code**: `spec/data-001-formats.md`,
   `spec/tracker-001.md`, the scenario's spec or its TOML `[verify]` table.
5. **Domain pass**: is the maths right and the method current? Check each equation against its
   cited source; name any specific gap from current practice. Rerun any results the PR states.
6. **Engineering pass**, reported apart: can a student follow it? Count files per call,
   protocol hops, registries; grep for duplication against `main`.
7. **Tests**: a test for each behaviour the domain pass cares about; flag tests that cannot fail
   or that lock in what the review asks to change.
8. **Measure what you will assert.** Count with `git grep` and quote the regex; check it returns
   0 on `main`. Time before calling anything slow. Reproduce before calling anything a bug.
9. **Write it up** (§3, §4), then re-check every number, line number and quoted name at the
   pinned head.
10. **Hand it to the maintainer.** An agent drafts; a person posts. Nothing reaches GitHub until
    the maintainer has read it.

## 2. What to check

Blocking first:

- **Licence and provenance.** The repo is MIT; unlicensed or copyleft code is never copied in.
- **Extend, don't fork.** A second pipeline, tracker or CFAR beside `main`'s is the costliest
  pattern; tracking lives in `core/tracking/`.
- **No new lint exemptions.** `per-file-ignores`, `force-exclude` or a `# noqa` in `src/`
  is a departure; the per-line escapes are `# broadcast-exempt:` and `# errstate-exempt:`.

Then, with a run counterexample for each correctness finding:

- **Numerics**: dB/linear and `10·log10` vs `20·log10`; FFT bin centres, `fftshift`, the sign of
  the Doppler axis; angle wrapping; covariance symmetric and positive-definite (Joseph form,
  `solve` not `inv`); float `==`; empty, length-1 and NaN inputs; off-by-one indexing.
- **Lifecycle**: can a tentative track that can no longer reach M-of-N live forever?
- **Reproducibility**: randomness from a caller's `Generator`; same seed, same output.
- **Error paths**: bad shapes and units raise and name the argument; no silent `except`.
- **Unsafe loading**: no `pickle`, `eval` or `allow_pickle=True` on user files.
- **Scale**: think at a scenario's real size; O(n²) loops, copies in loops, `inv` per scan.
- **Design and API**: units in names, no swappable adjacent arguments of one type and unit,
  keyword-only after the leading arrays, a return contract that holds for every input, reuse of
  `core/`, one reader and one writer per file format (data-001), writers that only write.
- **Docs**: a docstring that says something the code does not do costs students the most; read
  ten lines either side before calling it wrong. Every unavoidable loop has a "why" comment.
- **Tests**: analytic ground truth, NEES/NIS for a filter, clutter and missed detections for a
  tracker, and none of the ways a test passes for the wrong reason ([testing.md](testing.md)).

## 3. Layout

A first review: a greeting and a bold **Verdict** line; a summary table with one linked row per
section (R7 checks the anchors), labelled **blocking**, **should-fix**, **discuss** or **nit**;
`## N. Title` sections in the order the author will work; a *habit → what it costs later* table;
what is good, specifically; a re-review checklist; then private notes below a bare `---` line,
which is never posted (`sed '/^---$/q' review.md | sed '$d'` is the posted part). A re-review
opens with the verdict on the checklist and uses `### Topic (#N)` headings that point back.

## 4. Findings, checks and voice

A finding answers, in order: **what is wrong** (what the code *does*, bold, with `path:line`),
**how you know** (the quoted line, the count, the run), **why it matters** (something someone
will see), and **what to do** (a pasteable suggestion). It must make sense with the fix hidden
and read alone. State the conclusion before the trace.

Every checklist ask has a *Do* and a *Check*, ideally a command with today's count, tested both
ways: it finds the problem on the PR and nothing on `main`.

Write for an intern who has not seen the PR: one point per sentence, the code's own names,
abbreviations spelt out once, British spelling, polite and asking ("Suggest: …"), hedged once
and only when unsure. Pin the commit; say how each claim was checked and what was not done.
Name the drafter when an agent did the work, and never push to the contributor's branch.

## 5. Auditing existing code

An audit checks code against the specified behaviour (the equation, the bounds, the documented
contract), not against a rival implementation. Each public name gets a conformance verdict and a
quality verdict on eight dimensions:

- **Q1** Accuracy and radar-physics fidelity: the right thing, to the claimed accuracy.
- **Q2** Clarity: would an intern follow it without the docstring?
- **Q3** Speed and vectorisation, with a measurement behind any claim.
- **Q4** Conciseness: nothing that earns nothing.
- **Q5** Documentation: docstring, shapes, units, references.
- **Q6** Naming: what it does, in radar terms, with units.
- **Q7** Inputs: order, keyword-only where a swap is silent, sensible defaults, `ArrayLike`.
- **Q8** Outputs: type, shape, units and a contract a caller can rely on.

A behaviour change comes with a test that fails on the old code; no tolerance is weakened; where
the spec is wrong, the spec is fixed in a `docs(spec)` commit.

## 6. Refactors

A refactor changes how code is put together, not what it computes:

- **F1** Define each relation, file format and lookup table once; every other form calls it.
- **F2** Delete code a library or local helper already does, and code nothing calls.
- **F3** Make a step that matters a named, public, testable function; split check from action.
- **F4** Split modules along the spec's decisions, not only because a file grew.
- **F5** Rename until the name says what the thing does, and nothing else.
- **F6** Break deliberately: no compatibility shims; `!` and a `BREAKING CHANGE:` footer.
- **F7** Prove nothing moved: a pinning test, a count, unchanged scenario outputs; or say what moved.
- **F8** Code comments describe the code as it is now, not the commit that changed it.
- **F9** Arguments that always travel together become one frozen dataclass.

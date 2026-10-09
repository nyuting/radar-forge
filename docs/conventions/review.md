---
name: pr-review
description: Review or re-review a pull request against radar-forge's conventions and spec, and draft the review for the author. Use when asked to "review PR", "re-review", "check PR", or to draft or update a file under docs/reviews/.
---

# Reviewing a pull request

How to do a review for this repo and how to write it up: where it lives, the order the work is
done in, what to check, how it is laid out, how a finding is phrased, and what to check the PR
against. It applies to people and to AI agents writing a review alike. Claude Code loads this file
as `/pr-review` (through the symlink `.claude/skills/pr-review/SKILL.md`), so there is one guide,
not two. The worked examples are [`docs/reviews/pr1-review.md`](../reviews/pr1-review.md) (a first
review) and `pr2-review.md` (a re-review).

## Why

Our readers are interns and students. A review here is teaching material as well as a gate: the
author learns from it, and so does the next student who reads the PR to see why the code looks the
way it does. The repo must be both state of the art and simple
([`spec/starter.md` §1.1](../../spec/starter.md#11-the-bar)), so a review holds a PR to both: a
current method with correct, cited math, and code a newcomer can follow. An outdated method is a
blocking finding, and so is opaque code. So a review says *why* as well as *what*, points at the line, and gives a fix the
author can paste. A finding that only says "this is wrong" teaches nothing and costs a round trip.

**Every claim is run, not argued.** That covers a root cause, a bug, a test that "passes", a test
that "would have caught" something, and every number. If nobody has run it, it goes under
"discuss", labelled unverified.

---

## 1. Where a review lives

- Write it as `docs/reviews/pr<N>-review.md` on a branch, then post it from there. The file keeps
  the reasoning next to the code, and a re-review can diff against it.
- A re-review of the same PR, or of a PR split out of it, gets its own file
  (`pr2-review.md` re-reviews PR A of `pr1-review.md`'s split). Its sections point back to the
  first review's numbers.
- Nothing goes to GitHub until the maintainer has read it and decided to post it. An agent drafts;
  a person posts.
- Everything above the first line that is exactly `---` is posted; everything below is private
  notes. So the posted part must never contain a bare `---` line.
- The maintainer or another session may edit a draft mid-round. Re-read it before each edit, make
  small targeted edits, and never rewrite the whole file.
- `docs/reviews/` is excluded from `ruff`, so a review can quote a deliberately bad snippet. Other
  docs, this one included, wrap one in `<!-- fmt:off -->` / `<!-- fmt:on -->`.

Posting, once the maintainer approves. Check the cut first: this should end with the sign-off, not
with the notes.

```sh
sed '/^---$/q' docs/reviews/prN-review.md | sed '$d' | tail -5
sed '/^---$/q' docs/reviews/prN-review.md | sed '$d' | gh pr review N --request-changes --body-file -
```

## 2. Doing a review

In this order. Each step makes the next one smaller, and the last one catches what the others got
wrong.

1. **Fix the base.** Fetch first, then diff the PR head against `origin/main`'s merge base, not a
   local `main`. Write down both commits. A stale local `main` makes your own earlier commits look
   like the author's.

   ```sh
   git fetch origin main pull/N/head:pr-N
   git rev-list --count main..origin/main   # non-zero: local main is stale
   git diff --stat origin/main...pr-N       # three dots
   ```

   Read every modified file, not only the new ones; "the PR only adds" is often wrong about
   `__init__.py` exports. Check whether CI ran: GitHub holds workflows for a first-time
   contributor until someone approves them, and "passes locally" isn't CI.
2. **Set up a worktree and run the gates.** One worktree per stream:
   `git worktree add ../radar-forge-pr-N pr-N`. If `VIRTUAL_ENV` points at the main checkout,
   plain `uv run` tests main's package, so use `env -u VIRTUAL_ENV uv sync --extra dev` and
   `env -u VIRTUAL_ENV uv run python -m pytest` (`--all-extras` fails on macOS 13 because of
   torch). Run `make check` and record what passed: ruff, mypy `--strict`, conventions, test count.
3. **List the PR's own claims.** The description, the docstrings and the commit messages make
   promises ("one writer per file", "nothing coerced"). Each one is something to check.
4. **Read the spec the PR touches, before the code.** `spec/structure.md` (layers, B.2 rules, the
   D-decisions), `spec/data-001-formats.md`, and the scenario spec. Most structural findings are
   "the spec already decided this". If the spec post-dates the PR
   (`git log --format=%ad -- spec/<file>` against the PR's commit date), ask the author to follow
   it, but say it's new and don't count the mismatch against them.
5. **The domain pass.** Is the maths right, and is the method current? Check each equation
   against its cited source (§9 item 5). Name any specific gap from current practice (no Joseph
   form, sigma points not vectorised), not "not state of the art". If the PR states results, rerun
   the scenarios and reproduce them.
6. **The engineering pass.** Is the code heavier than the job needs, and can a student follow it?
   Count what makes it hard: the files one call goes through, the protocol hops, the registries.
   Grep for duplication, both inside the PR and against what `main` already has. Check placement
   against [`spec/structure.md`](../../spec/structure.md). Read the new modules in the reading
   view of the PR tree:

   ```sh
   git archive pr-N src | tar -x -C SCRATCH
   uv run python tools/generate_reading_view.py --src SCRATCH/src/radar_forge --out SCRATCH/rv
   ```

   Report this pass apart from the domain pass. "The maths is right; the engineering is what makes
   it hard to read" is a finding, and one doesn't excuse the other.
7. **Tests.** Is there a test for each behaviour the domain pass cares about? Which ones are
   missing ([testing.md](testing.md); `main`'s own tests often already have them to port)? Which
   are nuisance tests, ones that can't fail or that lock in the very thing the review asks to
   change? Flag those in the same review, or the fix will break them.
8. **Measure everything you will assert** (§8). Run it, time it, count it. Count with `git grep`,
   not by eye, and name the regex each count came from:
   - `-w word` misses `words` and `word_index`. Write a regex for the whole family, case included,
     and check it returns 0 on `main`.
   - A banned word is banned in prose too, so grep `README.md docs spec` as well as code. Grep a
     term before calling it foreign to the repo.
   - Quote `git show "${R}:path"`: zsh reads `$R:s…` as a history modifier.
   - Add up the parts of a count and check they match its total.
9. **Write the findings (§5) and the checks (§6).** Work through §3, and look at the
   "Blocking gates" group first.
10. **Run an accuracy pass.** Re-check every number, line number and quoted name at the pinned
    head, and fix the draft. Keep a list of what changed in the private notes. The next review
    learns from it.
11. **Hand it to the maintainer** (§1). Answer each question they asked in the private notes
    ("Yes." / "No." first, then where the review covers it).

## 3. What to check

Grouped by priority. Tags say what enforces each item, as in [style.md](style.md). A `[review]`
item is one only a reviewer catches, so it gets the most attention. A `[ruff]`/`[mypy]`/`[hook]`
item should already be failing CI; if it isn't, find the exemption.

### Blocking gates

- **Licence and provenance first** [review]. The repo is MIT. Code "supplied" from elsewhere,
  with no licence, can't be copied (`spec/structure.md` B.2 rule 4, D4). It goes first because
  merged code is hard to un-publish.
- **Extend, don't fork** [review]. A second package, pipeline, runner or CFAR next to main's is
  the most expensive pattern. `spec/structure.md` D2: tracking lives in the one `core/tracking/`
  package, so a new tracker extends it.
- **No lint or format exemptions** [ruff]. Look for `per-file-ignores`,
  `[tool.ruff.format] exclude` and `force-exclude` in
  `git diff origin/main...pr-N -- pyproject.toml`. `main` exempts only whole tool directories
  (`tests/**`, `scripts/**`), with a reason, and never a `src/` file. The one per-line escape is
  `# broadcast-exempt: <reason>`. A file-level ignore, a format exclude or `force-exclude` in
  `src/` is a departure even when the rule allows it. `force-exclude` matters because the
  pre-commit hook passes staged paths explicitly, and this setting makes it skip them. The fix is
  proper comments and docstrings, not `# noqa`. To measure it, copy the PR's `src` to scratch
  with main's `pyproject.toml`, then run `ruff check --statistics` and `ruff format --check`.

### Correctness

Each finding comes with a counterexample that has been run against the PR: an input with a known
answer, and the wrong number the PR returns.

- **Numerical and logical** [review]:
  - dB/linear mix-ups at API boundaries, and `10·log10` vs `20·log10` (power vs amplitude).
  - FFT bin centres, `fftshift`, Nyquist and the sign of the Doppler axis.
  - Angle wrapping at ±π, and sign conventions (range-rate, phase, azimuth direction).
  - Covariance kept symmetric and positive-definite: a Joseph-form update, or a stated reason
    for doing without one, and `solve` rather than `inv`.
  - Float `==`, and empty, length-1 and NaN inputs.
  - Off-by-one errors in pulse and sample indexing.
- **Tracker lifecycle** [review]. Can a tentative track that can no longer reach M-of-N live
  forever? Feed hit, miss, miss, … and see (PR #1 had this bug). Compare with main's
  `core/tracking/lifecycle.py`.
- **Reproducibility** [review]. Randomness comes from an `np.random.Generator` passed in by the
  caller, never the global `np.random.*` functions or an unseeded `default_rng()` in library
  code. Check that two runs with the same seed give identical output.
- **Error paths** [review]. Bad shapes and units raise, with messages that name the argument. No
  bare or silent `except`, and no `assert` for input validation (`python -O` strips it).
- **Unsafe loading** [review]. No `pickle`, `eval` or `np.load(allow_pickle=True)` on files a
  user supplies. Paths are built with `pathlib`, not string concatenation.
- **CLI flags** [review]. `store_true` with `default=True` is a no-op. The repo's switches are
  `--no-*` opt-outs.

### Performance and scale

- **At real size** [review]. Think at the size a scenario really runs at
  (`n_pulses × n_samples × n_channels`, targets × detections per scan), not the size the tests
  use. Look for:
  - O(n²) pairwise or association loops.
  - Copies inside loops (`astype`, fancy indexing, `np.append`).
  - complex128 where complex64 was intended, or the reverse.
  - `np.linalg.inv` inside a per-scan loop.

  Measure before writing "slow" (§8): `--durations` for tests, and a `timeit` at the realistic
  size for library code. Put the numbers in the notes.

### Tests

- **Ground truth and coverage** [review]. Analytic ground truth over recorded output. A filter
  needs a consistency test (NEES/NIS). A tracker needs clutter, missed-detection and
  several-detections-per-target cases. Start from main's own tests for the same component
  (`tests/core/test_tracking.py` and `tests/core/tracking/` cover most of a tracker checklist) and ask the author to match them,
  citing `file:line`.
- **Nuisance tests** [review]. Ones that assert today's structure (`X is not Y`, module paths,
  re-exports), and so fail once the review's asks are done. And ones that can't fail: compare
  what the inputs actually are, not what the test's name says.

### Design and API

Check every function the PR adds or changes against this group, not only the ones with a
numerical bug. A misleading name, an argument order that can be swapped without an error, or a
return whose shape is undocumented is as much a defect as a wrong constant.

- **Naming and units** [hook: R5 for units; review for the rest]. Units in names, family word
  first, `n_` for counts, main's existing names for existing ideas. Retired words ("leg") mustn't
  come back. No `General*`/`new_`/`v2` names. Keep the standard algorithm names (UKF, IMM, GNN)
  that textbooks use (§9 item 6).
- **Signatures** [review]. No two adjacent arguments of the same type and unit that a caller
  could swap without an error, such as `(range_m, altitude_m)`. Arguments after the leading
  arrays and physical inputs are keyword-only (`*`). A default is given only where one value
  suits most callers. Arrays come in as `ArrayLike`.
- **Return contract** [review]. The docstring states the return's type, shape and units, and
  they hold for every input. A function doesn't return a scalar for some inputs and an array for
  others unless the docstring says so.
- **Reuse `core`** [review]. For each new helper, search `core/` for an existing one, and search
  `main` before suggesting any new file or name. Anything with no caller outside tests is
  deferred to the PR whose scenario needs it.
- **Public API** [review]. New names in any `__all__`, especially `radar_forge/__init__.py` and
  `core/__init__.py`. No imports from another package's `_private` module.
- **File-format compatibility** [review]. Same file name, different columns, is the worst case.
  Compare CSV headers, `metadata.json` keys and TOML tables with main's writers and loaders, and
  with `spec/data-001-formats.md` §6. Read every section of the review that touches a file format
  against the spec together, so they don't contradict each other. Three smells:
  - a matrix in a cell (`*_json`, which DF3 rejects);
  - a column with the same value on every row (it belongs in `metadata.json`);
  - metrics that aren't tidy-long `metrics.csv` (§6.9).
- **Writers that only write** [review]. A writer that also scores against truth (picks the
  primary track, accumulates RMSE) produces metrics nobody can recompute from its files. Ask for
  one `write_<file>` function per file and a separate scoring function. Headers and rows come from
  one column tuple, never two position-matched lists.
- **Repetition inside the PR** [review]. Grep for the same literal compared many times (a variant
  label), the same wrap or formula written several ways, and one default declared in two places.
  Also look for code that does nothing useful: a wrapper that returns its input unchanged, or a
  constructor call that restates nine fields to change two.
  `git grep -nE '"S[123]"'`-style counts make it concrete.

### Docs and comments

- **Docstrings** [review]. NumPy style, with Returns, Raises and shapes written out. Look for
  four failure modes:
  - a docstring example that doesn't run (a misspelt class name, or a keyword that doesn't exist);
  - a banner comment that restates the docstring;
  - a stub reference ("X class references", or a spec file cited as a source);
  - **a docstring that says something the code doesn't do**.

  The last costs students the most. Read each claim against the code, and read ten lines either
  side before calling a docstring wrong.
- **Comments** [review]. `style.md` §10: why, never what. Main's comments are full sentences above
  the code, giving a reason, a citation or a rejected alternative. Look for inline comments that
  restate their line, banner comments, and `Inputs:`/`Outputs:` blocks. A proposed rule R8 would
  catch the last two; check `scripts/check_conventions.py` before tagging them `[hook]`.
- **Loops need a "why"** [review]. CLAUDE.md: every unavoidable Python loop carries a comment.
- **Guard shape and reading-view residue** [review]. Guards come first, as `msg = …; raise`
  (style §8). Values computed only for a guard survive in the reading view with nothing reading
  them. Look for them on the generated pages.
- **Docs that are working logs** [review]. "Approved substitutions", SHA inventories and personal
  notes belong in the PR description, not `docs/` or `spec/`.
- **Conventional Commits** [hook: commit-msg]. Web edits skip the hooks, so check
  `git log --format=%s origin/main..pr-N`.

## 4. Structure

### A first review

1. **A greeting and a verdict line.** Thank the author, then say in bold whether it merges:

   > **Verdict: Request changes.** Two things block merging: the licence question (#1), and the
   > lint and format exemptions in `pyproject.toml` (#2). Everything else is a should-fix, a point
   > for discussion or a nit.

2. **A summary table**, one row per numbered section, each row linked to its section (style.md
   §12, rule R7):

   ```markdown
   | # | Ask | Label | PR |
   | :-- | :-- | :-- | :-- |
   | 7 | [Stop tentative tracks stealing detections](#7-tentative-tracks-compete-with-confirmed-ones) | should-fix | A |
   ```

   Labels: **blocking** (must be settled before merge), **should-fix** (fix in the PR named),
   **discuss** (let's talk), **nit** (small). A label must match §3: anything §3 mandates isn't
   "discuss". Group the rows under bold sub-headings (**Blocking**, **PR A: tracker behaviour
   and logic**, …) when the PR is large. R7 checks the anchors only in tracked files; for an
   untracked draft, call `check_markdown_anchors` from `scripts/check_conventions.py`.
3. **`## N. Title` sections**, one per ask, in table order. Each says what the convention is, what
   the PR does instead, and the fix. Number them in the order the author will work (blocking,
   then shape, then by split PR), not in the order they were drafted. Drafting order splits one
   topic across several places.
4. **Why these matter later.** A table of *habit in this PR → what it costs later*. This is the
   section a student reads to learn the reasoning, so make it general:

   | Habit in this PR | What it costs later |
   | :-- | :-- |
   | Docs that contradict the code (#21) | Students trust a docstring over the code beneath it, so a wrong one teaches the wrong thing |

5. **What's good.** Always. Be specific, as in "the UKF against the closed-form KF at 1e-10", not
   "nice tests". Say where the PR does better than `main`, so that main can change to match.
6. **A re-review checklist** (§6), which the author copies into the PR description.
7. **Private notes**, below a `---`, under `# Notes for <maintainer> (not posted)`. These hold:
   - how each claim was verified;
   - open questions, and answers to the maintainer's;
   - corrections to earlier drafts;
   - who found what (§10);
   - the re-review commands;
   - anything not yet ready to say to the author.

### A re-review

Shorter, and it opens with the verdict on the checklist, not a greeting:

> **Please tick A4.** It's done: confirmed tracks pick first (`tracker.py:386-391`), there are no
> births inside a confirmed gate (`:406-410`), q is derived in `motion.py`'s docstring, and each has
> a test.

Then `### Topic (#N)` headings, where `#N` is the first review's section, such as
`### Reuse (#6): reuse, or say why not`, so the author can see which ask each point closes.

## 5. Findings

Each finding is one bullet or one short paragraph. It answers four questions, in this order, and
the reader should be able to find each answer without hunting:

1. **What is wrong?** One plain sentence, stated as what the code *does*, not what you want it to
   do. This is the bold lead, with the `path:line` next to it: **A tentative track that can never
   be confirmed is never deleted** (`management.py:244-257`).
2. **How do you know?** The evidence. Quote the line, count the cases, or give the run. Don't
   just describe them.
3. **Why does it matter?** The consequence, as something someone will see: a wrong answer, a
   crash, a slower run, a student misled, a check that fails. One clause is often enough. If you
   can't name one, it's a nit, or it isn't a finding.
4. **What to do.** The fix, as a pasteable snippet where there is one. Mark it as a suggestion.

From `pr2-review.md`:

> **`tracker.py:209-210`:** the comment gives the wrong reason for the loop. It says "routes are
> few" and "clearest form", but the loop is there so every route goes through `add_sensor`. A
> student who reads it learns the wrong reason, and may replace the loop with something that
> skips those checks. Suggest:
> `# Each route goes through add_sensor so it gets the same checks as one added later.`

Two tests before you keep a finding:

- **Cover the fix.** Hide step 4. Can the author still say in their own words what is wrong? If
  the finding only makes sense once you read the fix ("`space` → `layout`"), step 1 is missing.
- **Read it alone.** Would it make sense to someone who hasn't read the other findings? Write
  "the duplicated guard conditions (#17)", not "#17's residue". A `#N` goes with the words, not
  instead of them.

For a long investigation (a trace, a table of runs), state the conclusion first, then the trace
that supports it. Don't make the reader follow the trace to find out what it shows. From
`pr1-review.md` #7, before:

> The default demo fragments tracks, and you documented that honestly. Claude traced it. In S1 at
> t = 15 s the detector reports one target four times … Tentative and confirmed tracks share one
> assignment, and both get `deletion_misses = 5`.

After:

> **New tentative tracks take detections from the confirmed track, so the track fragments.**
> Tentative and confirmed tracks compete in one assignment. In S1 at t = 15 s the detector
> reports one target four times … Suggest two-stage association: confirmed tracks first, then
> tentative tracks on the leftover detections.

Ground a finding in the code this repo already holds up as the model, usually
[`core/radar_equation.py`](../../src/radar_forge/core/radar_equation.py), so the author can see
the target rather than guess at it. For example (an illustrative case), a parameter with no unit:

<!-- fmt:off -->
```python
def received_power(power, gain_tx, wavelength, range):  # what the PR has
```
<!-- fmt:on -->

> **`link.py:12`:** `power`, `gain_tx` and `range` don't say their units, so a student can't tell
> watts from dBW or a linear gain from dBi, and R5 rejects them. Suggest the exemplar's names:

```python
def received_power_w(
    *,
    transmit_power_w: ArrayLike,
    gain_tx_linear: ArrayLike,
    gain_rx_linear: ArrayLike,
    wavelength_m: ArrayLike,
    range_m: ArrayLike,
) -> NDArray[np.float64]: ...
```

Or a guard that repeats its condition to build its message (style.md §8):

<!-- fmt:off -->
```python
if len(bursts) == 2 and v_max_mps <= min(b.unambiguous_velocity_mps for b in bursts):
    msg = f"... {min(b.unambiguous_velocity_mps for b in bursts)} m/s ..."
```
<!-- fmt:on -->

> Compute `smallest_mps = min(...)` once above the `if`, or move the check into a `_check_*`
> helper that builds the offenders once and tests `if offenders:`. The reading view drops the
> helper whole.

Group the smaller points under headings that say what kind they are, rather than one long list:
**Where it isn't concise**, **Content that was lost**, **Inaccurate statements** (numbered, so
the author can answer "fixed 1-3, 4 is deliberate"), **Smaller readability fixes** and **Nits**.

### Proposing a fix

Be proactive, but leave the change to the author, since making it is how they learn. Never push to
the contributor's branch.

- A small, mechanical fix goes in a GitHub ```` ```suggestion ```` block on the exact line, so the
  author can apply it with one click.
- A larger fix is a patch the reviewer has written and run in the worktree with `make check`
  passing. It goes in a fenced `diff` block labelled "proposal, tested at `<sha>`".
- A proposed test must fail on the PR and pass with the fix. Run it both ways. Assert over the
  whole run, not one frame, so it can't pass after a fix by luck of the loop length. Keep the
  simulated noise and the declared covariance in one variable, so the test can't fail for its own
  reason.

## 6. Checks

Every ask on the checklist has a *Do* and a *Check*. The check is something the author can run
before asking for re-review, ideally a command with today's count:

```markdown
- [ ] **C1. One pipeline, one file format (#20, #22).** *Do:* … *Check:* `git grep -nE
  '_json"|class TrackingWriter' -- src scripts` returns nothing (it finds 11 lines today).
```

In a re-review, the check sits on its own line under the point it closes:

> *Check:* `git grep -nE '^class (Track|TrackManager)\b|^TrackStatus\b|class TrackStatus' -- src`
> gives one hit per name.

- Name the steps by PR (A1, C4, E2), not by section number, so a step's name never clashes
  with "#6".
- Order the steps so each one makes the next smaller (move, then rename, then reuse).
- An "Every PR" block (E1, E2…) covers what applies to all of them: `make check`, an unchanged
  `pyproject.toml`, Conventional Commits.
- Put each check in the PR that can run it. A scenario rerun, for example, needs the pipeline PR.
- Test every check both ways: on the PR commit it finds the problem, and on `main` it finds
  nothing. A pattern that misses `legs` when it means `leg` gives a false pass. A `pytest -k`
  selection usually matches main's own tests too, so check for added tests with
  `git diff origin/main... -- tests | grep -E '^\+\s*def test_'`.

## 7. Voice

Write for an intern who knows Python and NumPy but hasn't seen this PR, the earlier review, or the
conversation behind it.

- **Say the problem plainly, before anything else** (§5). Describe behaviour someone could
  observe ("the track is never deleted"), not a judgement ("the lifecycle is fragile").
- **Name the consequence.** "`matrices()` runs twice per predict" leaves the reader to guess
  whether that matters. "Every predict builds the same matrices twice, so half that work is
  wasted" doesn't.
- **One point per sentence.** If a sentence needs a colon, a semicolon and a bracket, split it.
- **Use the code's own names, and define terms once.** Spell out an abbreviation or radar term
  the first time it appears in the review (NLL: negative log-likelihood). Don't use nicknames the
  code doesn't ("births", "the reading view") unless you define them.
- **Plain, short sentences, in British spelling** (normalise, behaviour, licence as a noun).
- **Polite, and asking.** "Please …", "Suggest: …", "If you keep X for Y, say so in the Notes."
  The author may know something you don't; a question gets the answer, an order gets compliance.
- **Hedge once, honestly.** Say "I think" when you are unsure, and only then.
- **Mark a proposal as a proposal.** Text Claude drafted for a docstring or spec is a suggestion,
  not spec wording, and the review says which is which. Label any citation section number nobody
  checked.

## 8. Habits

- **Say how a claim was checked.** "I checked this one myself", "Claude ran it at `bedfde0`",
  "Claude counted at bedfde0: 198 banner lines". If a claim wasn't verified, say so or leave
  it out.
- **Pin the commit.** Line numbers are at a named head (`5a9495a`). They move.
- Credit what's good, specifically, every time.
- Quote short code and text in backticks, and give `path:line` for each.
- Count exactly: "about half" becomes "198 of 4,400 lines".
- Check the PR description's own claims against the diff. "Nothing coerced", "one writer per
  file" and "the grep returns nothing" are claims too.
- Recompute a quoted figure from its inputs when you can. A wrong number in a docstring is the one
  a student copies into their report.
- Say what you didn't do ("I didn't run the reading-view generator").
- **Reproduce before you call it a bug.** Write the smallest script that shows it. Check it fails
  at the head *for the reason you claim*, and would pass once fixed. If you think you know why
  something fails, patch it in scratch and rerun before writing the cause down.
- **Time before you call it slow.** The first `matplotlib` import costs seconds. Separate it from
  the test before blaming the test.
- **Grep the spec before citing it.** "The spec retired X" or "style.md says Y" needs the text
  behind it, quoted.

## 9. What to review against

In this order:

1. **CLAUDE.md's non-negotiables.** SI units in names, constants from `core.constants`, NumPy
   docstrings with References and shapes, `mypy --strict`, all config in `pyproject.toml`,
   `make check` green.
2. **[style.md](style.md) and [testing.md](testing.md)**, especially the **[review]**-tagged rules
   that no tool checks: the audience rule, guards first, comments that say why, a "why" on each
   loop, analytic ground truth, a justified tolerance, and a domain test that names the bug it
   catches ([testing.md §10.1](testing.md#101-what-earns-a-place-review)).
3. **The `core/` modules as written.** `radar_equation.py` is the exemplar; `detection.py`,
   `dsp.py` and `ambiguity.py` show the same idioms at more length. Where the PR does something
   differently, compare the two side by side.
4. **[data-001](../../spec/data-001-formats.md)** for any file read or written, and the
   scenario specs for behaviour.
5. **Citations.** Each citation the PR adds or changes passes
   [style.md §4.1](style.md#41-checking-a-citation-review). Don't let a removed source (Blackman & Popoli) or a renumbered one
   creep back in, especially through a merge.
6. **Names, on two axes.** Algorithms keep the standard names from the literature and from
   upstream libraries (Stone Soup, FilterPy). Modules, fields, configs and file formats follow
   this repo. Say which axis a naming finding is on. Also say whether it breaks a style.md rule or
   only departs from a habit of `main`.
7. **Upstream.** Where a module says it follows Stone Soup (`core/tracking/`), check that the
   divergences are deliberate and that the module docstring says so, as `docs/tracking/README.md`
   promises. Check claims about Stone Soup, FilterPy and similar against their GitHub `main`, not
   memory, and say in the notes what was checked there.
8. **Earlier reviews.** Every ask in the first review that this PR was meant to close, and every
   ask in a sibling review that touches the same code.
9. **Main since the PR opened.** A PR opened before a merge may need a rebase, and the rebase must
   keep main's changes.

## 10. For AI agents

What an agent adds to the procedure above. Claude Code loads this file as `/pr-review`.

- **Start the built-in reviews in the background**, in the worktree: `/code-review pr-N high` and
  `/security-review`. Never pass `--comment` (it posts before the maintainer approves) or `--fix`
  (the agent doesn't change the author's branch).
- **Audit, then merge.** Run parallel read-only subagents, one per group in §3, and pool their
  findings with the built-ins'. Every finding is a candidate until it is deduplicated, verified by
  running it, relabelled (§4), and rewritten in this guide's voice with the *why*.
- **Spot-check line numbers.** Subagents and built-ins get them slightly wrong.
- **Name the drafter.** "Claude traced it", "Claude checked it". The review never pretends a human
  did work an agent did.
- **In the notes, say who found what**, and which built-in findings were rejected and why.

## 11. Pitfalls

Mistakes earlier drafts made. Each rule now lives in the step or check named in brackets.

- Diffed against a stale local `main`, so the maintainer's own `EARTH_RADIUS_M` commit showed up as the PR's. [§2 step 1]
- Said "the PR only adds", but it had also changed the `__init__.py` exports. [§2 step 1]
- Blamed NIS and GNN for track fragmentation without running it; they weren't the cause. [§8 Reproduce]
- Said the repo never uses "Duke"; it does, for the receiver site. [§2 step 8]
- `\bGeneral[A-Z]` missed the README's "General tracking" and "general runner". [§2 step 8]
- Counted "82 lines of banner"; they were `Inputs:` comments, and the file had no `####`. [§2 step 8]
- Listed ruff counts whose parts summed to 285 against a total of 279. [§2 step 8]
- Suggested a `track_plot.py` that main already had. [§3 Reuse `core`]
- Overstated doc findings: the check existed a few lines later, or the docstring did say it. [§3 Docstrings]
- Said a test "would have caught #7"; it passed. That bug needed several detections per target. [§3 Tests]
- Wrote "'leg' was retired" and "style.md says family word first" with no text behind either. [§8 Grep the spec]
- Told the author to write `noise_power_w` in one section and `peak_power_w` in another. [§3 File-format compatibility]
- A clutter test drew noise with a different σ from its declared R. [§5 Proposing a fix]
- Asserted a reproducer only on the last frame, so it would pass after a fix by luck of the loop length. [§5 Proposing a fix]
- Used `pytest -k "nees or nis …"` as a check that new tests exist; main's tests already match it. [§6]
- Sections numbered in drafting order split one topic across four places. [§4 item 3]

## 12. Maintaining this guide

Update this file after every review:

- **A new check** in §3 when a review found something it didn't prompt for. That includes any
  verified built-in finding (§10) that §3 missed, so over time the checklist absorbs the
  built-ins' blind spots.
- **A pitfall** in §11 when a draft got something wrong, with its rule folded into the step or
  check it belongs to.
- **A command** when a better one turned up.

Delete anything that stopped being true.

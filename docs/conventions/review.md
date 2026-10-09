# Reviewing a pull request

How to do a review for this repo and how to write it up: where it lives, the order the work is
done in, how it is laid out, how a finding is phrased, and what to check the PR against. It applies to people and to AI agents writing a review alike.
The worked examples are [`docs/reviews/pr1-review.md`](../reviews/pr1-review.md) (a first review)
and `pr2-review.md` (a re-review).

## Why

Our readers are interns and students. A review here is teaching material as well as a gate: the
author learns from it, and so does the next student who reads the PR to see why the code looks the
way it does. The repo must be both state of the art and simple
([`spec/starter.md` §1.1](../../spec/starter.md#11-the-bar)), so a review holds a PR to both: a
current method with correct, cited math, and code a newcomer can follow. An outdated method is a
blocking finding, and so is opaque code. So a review says *why* as well as *what*, points at the line, and gives a fix the
author can paste. A finding that only says "this is wrong" teaches nothing and costs a round trip.

---

## 1. Where a review lives

- Write it as `docs/reviews/pr<N>-review.md` on a branch, then post it from there. The file keeps
  the reasoning next to the code, and a re-review can diff against it.
- A re-review of the same PR, or of a PR split out of it, gets its own file
  (`pr2-review.md` re-reviews PR A of `pr1-review.md`'s split). Its sections point back to the
  first review's numbers.
- Nothing goes to GitHub until the maintainer has read it and decided to post it. An agent drafts;
  a person posts.
- `docs/reviews/` is excluded from `ruff`, so a review can quote a deliberately bad snippet. Other
  docs, this one included, wrap one in `<!-- fmt:off -->` / `<!-- fmt:on -->`.

## 2. Doing a review

In this order. Each step makes the next one smaller, and the last one catches what the others got
wrong.

1. **Fix the base.** Fetch first, then diff the PR head against `origin/main`'s merge base, not a
   local `main`. Write down both commits. A stale local `main` makes your own earlier commits look
   like the author's.
2. **List the PR's own claims.** The description, the docstrings and the commit messages make
   promises ("one writer per file", "nothing coerced"). Each one is something to check.
3. **The domain pass.** Is the maths right, and is the method current? Check each equation
   against its cited source (§8 item 5). Name any specific gap from current practice (no Joseph
   form, sigma points not vectorised), not "not state of the art".
4. **The engineering pass.** Is the code heavier than the job needs, and can a student follow it?
   Count what makes it hard: the files one call goes through, the protocol hops, the registries.
   Grep for duplication, both inside the PR and against what `main` already has. Check placement
   against [`spec/structure.md`](../../spec/structure.md). Report this pass apart from the domain
   pass. "The maths is right; the engineering is what makes it hard to read" is a finding, and
   one doesn't excuse the other.
5. **Tests.** Is there a test for each behaviour the domain pass cares about? Which ones are
   missing ([testing.md](testing.md); `main`'s own tests often already have them to port)? Which
   are nuisance tests, ones that can't fail or that lock in the very thing the review asks to
   change? Flag those in the same review, or the fix will break them.
6. **Measure everything you will assert** (§7). Run it, time it, count it.
7. **Write the findings (§4) and the checks (§5).**
8. **Run an accuracy pass.** Re-check every number, line number and quoted name at the pinned
   head, and fix the draft. Keep a list of what changed in the private notes. The next review
   learns from it.
9. **Hand it to the maintainer** (§1). Answer each question they asked in the private notes
   ("Yes." / "No." first, then where the review covers it).

## 3. Structure

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
   **discuss** (let's talk), **nit** (small). Group the rows under bold sub-headings
   (**Blocking**, **PR A: tracker behaviour and logic**, …) when the PR is large.
3. **`## N. Title` sections**, one per ask, in table order. Each says what the convention is, what
   the PR does instead, and the fix.
4. **Why these matter later.** A table of *habit in this PR → what it costs later*. This is the
   section a student reads to learn the reasoning, so make it general:

   | Habit in this PR | What it costs later |
   | :-- | :-- |
   | Docs that contradict the code (#21) | Students trust a docstring over the code beneath it, so a wrong one teaches the wrong thing |

5. **What's good.** Always. Be specific, as in "the UKF against the closed-form KF at 1e-10", not
   "nice tests". Say where the PR does better than `main`, so that main can change to match.
6. **A re-review checklist** (§5), which the author copies into the PR description.
7. **Private notes**, below a `---`, under `# Notes for <maintainer> (not posted)`. These hold how
   each claim was verified, open questions, and anything not yet ready to say to the author.

### A re-review

Shorter, and it opens with the verdict on the checklist, not a greeting:

> **Please tick A4.** It's done: confirmed tracks pick first (`tracker.py:386-391`), there are no
> births inside a confirmed gate (`:406-410`), q is derived in `motion.py`'s docstring, and each has
> a test.

Then `### Topic (#N)` headings, where `#N` is the first review's section, such as
`### Reuse (#6): reuse, or say why not`, so the author can see which ask each point closes.

## 4. Findings

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

## 5. Checks

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
- Test a grep before writing it down. A pattern that misses `legs` when it means `leg` gives a false
  pass.

## 6. Voice

Write for an intern who knows Python and NumPy but hasn't seen this PR, the earlier review, or the
conversation behind it.

- **Say the problem plainly, before anything else** (§4). Describe behaviour someone could
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
  not spec wording, and the review says which is which.

## 7. Habits

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
  at the head *for the reason you claim*, and would pass once fixed. An assertion that only
  happens to catch the bug at one frame, or a test fixture whose noise doesn't match its declared
  covariance, proves the wrong thing.
- **Time before you call it slow.** The first `matplotlib` import costs seconds. Separate it from
  the test before blaming the test.
- **Compare exemptions with how `main` does it**, not only with the rule. `main` exempts whole
  directories with a reason and single lines with a stated reason (`# broadcast-exempt:`). A
  file-level ignore, a format exclude or `force-exclude` in `src/` is a departure, even when the
  rule allows it.

## 8. What to review against

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
   promises.
8. **Earlier reviews.** Every ask in the first review that this PR was meant to close, and every
   ask in a sibling review that touches the same code.
9. **Main since the PR opened.** A PR opened before a merge may need a rebase, and the rebase must
   keep main's changes.

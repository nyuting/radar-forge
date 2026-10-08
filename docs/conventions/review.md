# Reviewing a pull request

How to write a review for this repo: where it lives, how it is laid out, how a finding is phrased,
and what to check the PR against. It applies to people and to AI agents writing a review alike.
The worked examples are [`docs/reviews/pr1-review.md`](../reviews/pr1-review.md) (a first review)
and `pr2-review.md` (a re-review).

## Why

Our readers are interns and students. A review here is teaching material as well as a gate: the
author learns from it, and so does the next student who reads the PR to see why the code looks the
way it does. So a review says *why* as well as *what*, points at the line, and gives a fix the
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

## 2. Structure

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
6. **A re-review checklist** (§4), which the author copies into the PR description.
7. **Private notes**, below a `---`, under `# Notes for <maintainer> (not posted)`. These hold how
   each claim was verified, open questions, and anything not yet ready to say to the author.

### A re-review

Shorter, and it opens with the verdict on the checklist, not a greeting:

> **Please tick A4.** It's done: confirmed tracks pick first (`tracker.py:386-391`), there are no
> births inside a confirmed gate (`:406-410`), q is derived in `motion.py`'s docstring, and each has
> a test.

Then `### Topic (#N)` headings, where `#N` is the first review's section, such as
`### Reuse (#6): reuse, or say why not`, so the author can see which ask each point closes.

## 3. Findings

Each finding is one bullet or one short paragraph:

1. **A bold lead** naming the file and line, or the topic: **`tracker.py:209-210`:**.
2. **What is wrong.** Quote the line or count the cases. Don't just describe them.
3. **Why it matters to a student.** One clause is often enough.
4. **The fix**, as a pasteable snippet where there is one. Mark it as a suggestion.

From `pr2-review.md`:

> **`tracker.py:209-210`:** the loop is right, but "routes are few" and "clearest form" aren't why
> it's a loop. Suggest: `# Each route goes through add_sensor so it gets the same checks as one
> added later.`

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

## 4. Checks

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

## 5. Voice

- **Plain, short sentences, in British spelling** (normalise, behaviour, licence as a noun).
- **Polite, and asking.** "Please …", "Suggest: …", "If you keep X for Y, say so in the Notes."
  The author may know something you don't; a question gets the answer, an order gets compliance.
- **Say how a claim was checked.** "I checked this one myself", "Claude ran it at `bedfde0`",
  "Claude counted at bedfde0: 198 banner lines". If a claim wasn't verified, say so or leave
  it out.
- **Mark a proposal as a proposal.** Text Claude drafted for a docstring or spec is a suggestion,
  not spec wording, and the review says which is which.
- **Pin the commit.** Line numbers are at a named head (`5a9495a`). They move.

## 6. Habits

- Credit what's good, specifically, every time.
- Quote short code and text in backticks, and give `path:line` for each.
- Count exactly: "about half" becomes "198 of 4,400 lines".
- Check the PR description's own claims against the diff. "Nothing coerced", "one writer per
  file" and "the grep returns nothing" are claims too.
- Recompute a quoted figure from its inputs when you can. A wrong number in a docstring is the one
  a student copies into their report.
- Say what you didn't do ("I didn't run the reading-view generator").

## 7. What to review against

In this order:

1. **CLAUDE.md's non-negotiables.** SI units in names, constants from `core.constants`, NumPy
   docstrings with References and shapes, `mypy --strict`, all config in `pyproject.toml`,
   `make check` green.
2. **[style.md](style.md) and [testing.md](testing.md)**, especially the **[review]**-tagged rules
   that no tool checks: the audience rule, guards first, comments that say why, a "why" on each
   loop, analytic ground truth, a justified tolerance.
3. **The `core/` modules as written.** `radar_equation.py` is the exemplar; `detection.py`,
   `dsp.py` and `ambiguity.py` show the same idioms at more length. Where the PR does something
   differently, compare the two side by side.
4. **[data-001](../../spec/data-001-formats.md)** for any file read or written, and the
   scenario specs for behaviour.
5. **Citations.** A section number must exist and its title must support the claim, as checked in
   the citation pass (#11). Don't let a removed source (Blackman & Popoli) or a renumbered one
   creep back in, especially through a merge.
6. **Upstream.** Where a module says it follows Stone Soup (`core/tracking/`), check that the
   divergences are deliberate and that the module docstring says so, as `docs/tracking/README.md`
   promises.
7. **Earlier reviews.** Every ask in the first review that this PR was meant to close, and every
   ask in a sibling review that touches the same code.
8. **Main since the PR opened.** A PR opened before a merge may need a rebase, and the rebase must
   keep main's changes.

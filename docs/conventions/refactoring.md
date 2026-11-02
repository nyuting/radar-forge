# Refactoring

A refactor changes how the code is put together and leaves what it computes alone. This page lists
what a refactor in this repo is for and how to show it changed nothing. The commit type is
`refactor` ([commits.md](commits.md#types)). The rules that every piece of code follows (naming,
units, the public API) are in [style.md](style.md) and are not repeated here. Each rule cites one or
two commits from the history as worked examples. `git show <hash>` shows the change and its
reasoning.

| Rule | Statement |
| :--- | :--- |
| [F1](#f1-define-each-relation-once) | Define each relation once, and have every other form call it |
| [F2](#f2-delete-what-a-library-already-does) | Delete hand-written code that a library or a local helper already does |
| [F3](#f3-show-the-steps) | Make the step that matters a named, public, testable function |
| [F4](#f4-split-along-the-spec) | Split modules along the spec's decisions, and move in the spec's steps |
| [F5](#f5-rename-until-the-name-holds-up) | Rename until the name says what the thing does, and nothing else |
| [F6](#f6-break-deliberately) | Break deliberately, with no compatibility shims |
| [F7](#f7-prove-nothing-moved) | Show that nothing moved, or say exactly what did |
| [F8](#f8-leave-history-alone) | Records keep the names they were written with |

## The rules

### F1. Define each relation once

When a formula is written twice, the two copies drift apart. Call the helper that already defines
the relation, even when the inline form is one line. `Radar` and `BistaticRadar` each wrote
`c / (2B)` until they called `waveforms.range_resolution_m` (63749e9). When a quantity comes in two
units or forms, write one implementation and make the other a thin wrapper around it. The
degree form of the ENU-to-spherical geometry calls the radians form (3e30ba6).

### F2. Delete what a library already does

Hand-written code that SciPy or a helper in this repo already does is code to read, test and
maintain for nothing. Use the existing function. Keep the comment or docstring that explains the
idea. 3652134 replaced a hand-written summed-area table with `scipy.ndimage.uniform_filter`.
66f8a34 replaced hand padding with `cfar_detect`'s `wrap=`.

### F3. Show the steps

Don't hide a step that matters inside a stand-in argument or a private helper. Give it a name
and a docstring, make it public, and test it on its own. In 624083f, the 2-D CFAR ring had been
calibrated by calling the 1-D function with `n_train=M // 2`. That commit made the step
`cfar_ring_reference_count` and `cfar_threshold_factor_2d`. Clustering also became public as
`label_connected_cells`. This is the [§1.1 bar](../../spec/starter.md#11-the-bar) applied to old
code: the intermediate quantities are visible.

### F4. Split along the spec

A module splits where a spec decision or a module the spec already reserves says it should.
Don't split only because a file got long. d5c7d88 moved the path builders to
`core/propagation.py`, the module [`spec/structure.md`](../../spec/structure.md) had reserved. Move
large changes in the spec's numbered steps, one commit or PR per step. Every public import path
keeps working between steps. In eea7d29 (tracker-001 §13 step 1), `kalman.py` imports the
moved names back and keeps them in its `__all__`.

### F5. Rename until the name holds up

A name that is nearly right causes misreadings in every later reader, so a rename is worth a
breaking change. The naming rules are in [style.md §3](style.md#3-naming). In refactors they have
come up as:

- **Name what it does.** `SensorRoute` held model IDs and a coverage test and built batches, so it
  became `SensorRegistration` (f2259d0).
- **Name what it holds.** In a bistatic file, the scenario TOML's `[radar]` table held the
  receiving site, so it became `[receiver_site]` (7f0369d).
- **One name per concept.** `predict_to(time_s)` became `timestamp_s`, the `Estimator` protocol's
  name (f2259d0). The `viz` subpackage, extra and commit scope share one word (4a0f1f4).

### F6. Break deliberately

The project is young, so fix a wrong name or signature now, before outside code depends on it.

- Delete the old spelling rather than keep a fallback, and make a leftover fail with an error
  that names it. 7f0369d dropped the dual-spelling loader, so a stale `[radar]` table fails by
  name.
- Make an argument keyword-only when swapping it silently gives a plausible wrong answer. Swapping
  `wavelength_m` and `rcs_m2` in a positional call returned a believable power (60af1ee). This is
  [audit.md](audit.md#1-auditing-code-against-the-spec) Q7.
- Take a name out of `__all__` when nothing outside its module reads it (9fb6399).
- Remove a dependency from an extra when the code no longer uses it (4a0f1f4 dropped PySide6, about
  500 MB).

Mark the commit `!` and write the `BREAKING CHANGE:` footer with the migration in it
([commits.md](commits.md#breaking-changes)).

### F7. Prove nothing moved

A refactor makes a claim: the output did not change. Back it with evidence a reader can check:

- a test that pins the old behaviour and passes before and after. In 66f8a34,
  `test_wrap_matches_padding_the_map_by_hand` passes on both sides;
- a count, when tests move with the code: "40 before and after" (d5c7d88);
- a statement that the scenario outputs are unchanged, after running them (7f0369d).

If anything did move, say what, by how much and why, even when it is "different in the last bits
only" (3652134). Never weaken a tolerance to absorb the change. Re-derive the number, as in
[audit.md](audit.md#2-changes-an-audit-makes) C1 and C2. A refactor that needs a behaviour change
is two commits: the `refactor` and a `fix` or `feat`.

### F8. Leave history alone

`docs/reviews/` and `docs/audits/` record what the code was called when they were written. Don't
rename inside them (4a0f1f4). Add a dated Status note instead
([audit.md](audit.md#4-writing-the-report)). Code comments go the other way: they describe the
code as it is now, so they don't cite the PR or commit that changed it (3652134).

## The commit

The body's first paragraph says why the code had to change, and names the failure the old code
allowed. Then list what moved, as bullets when there are several things. End with the evidence
from F7 and the spec step or audit finding the commit carries out. The format is in
[commits.md](commits.md#commit-body).

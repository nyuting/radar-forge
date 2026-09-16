# Refactor 002 — Spec-first reference comparison, qualitative test evaluation, anchor invariant

Status: **specified, partially blocked**. Three items: a function-by-function audit of `src/`
against the specifications that predate it, a qualitative (rather than structural) audit of the
test suite, and a navigational invariant for Markdown summary tables.

Where `spec/refactor-001-standardisation-and-reading-pipeline.md` was mechanical — renames,
restructures, a new tool — this one is a judgement exercise. Its output is as much a set of
findings as a set of diffs, and an honest "no change, here is why" is a valid result for any
function audited.

§4 records where the request's premises did not match the repository. Where §4 contradicts the
body, §4 wins. §1 in particular cannot be executed as literally written.

## 1. Spec-first reference comparison (`spec/` vs `src/`)

### 1.1 Temporal premise

`spec/scenario-001-xband.md`, `spec/scenario-002-bistatic.md` and `spec/scenario-003-tracking.md`
were written before `src/` existed. Early prototype implementations embedded in `spec/` may
therefore carry functional logic that is more accurate, cleaner or more direct than the code that
superseded them.

> **The first half is true; the second does not follow — see §4.1.** `spec/` was committed 48
> minutes before `src/`, so the ordering holds. But `spec/` contains no prototype
> implementations: one 8-line dataclass *interface sketch*, and no `def`, `import`, `return` or
> `np.` line in any other fenced block. There is no prototype code to lose a comparison to.

### 1.2 Function-by-function audit

Compare every function in `src/` against its candidate reference across six dimensions:

| # | Dimension | Asks |
| :--- | :--- | :--- |
| Q1 | Mathematical accuracy and radar-physics fidelity | Does it compute the right thing, to the claimed accuracy? |
| Q2 | Code clarity and intuitiveness | Would an intern follow it without the docstring? |
| Q3 | Execution speed and vector efficiency | Is it vectorised, and is any speed claim measured? |
| Q4 | Conciseness and elimination of bloat | Is anything here that earns nothing? |
| Q5 | Readability and documentation thoroughness | Docstring, shapes, units, references |
| Q6 | **Naming** | Does the name say what it does, in radar nomenclature and PEP 8? Units in identifiers? |
| Q7 | **Inputs** | Is the signature the right shape — argument order, keyword-only where it matters, sensible defaults, `ArrayLike` in? |
| Q8 | **Outputs** | Is the return the right thing — type, array shape, units, and a contract a caller can rely on? |

Q6–Q8 are audited for **every** function, not only those that fail Q1. A function can be
numerically perfect and still be hard to call correctly: a misleading name, an argument order that
invites a silent transposition, or a return whose shape is undocumented are defects in the same
sense that a wrong constant is.

**Reframed scope, per §4.1.** With no prototype code to diff against, "candidate reference" means
the *specified behaviour* rather than a rival implementation:

- the governing equation, as written in the scenario spec or cited from its source;
- the numerical bounds, tolerances and acceptance figures the spec fixes;
- the documented contract — array shapes, coordinate frames, sign conventions, units;
- the decisions in `spec/structure.md` (its own D1–D8, which are *decisions*, not these dimensions) that constrain the implementation.

The audit therefore asks, per function: **does the code compute what the spec says, to the
accuracy the spec claims, as clearly as the spec describes it?** A divergence is a finding
whichever side is wrong — the code may have drifted, or the spec may have been wrong and the code
silently corrected it without recording the correction.

Q3 is measured, not asserted: a claim that one form is faster carries a benchmark or it is not a
finding.

**Decision: judge the implementation on its own merits too.** The audit is not limited to
conformance. Having established that a function computes what the spec says, ask whether it is
the *better implementation* available — neater, clearer, shorter, faster, more robust — and
improve it where it is not. Absence of a rival implementation is not a reason to leave a clumsy
one standing. Both style and performance count, under R1.3.1 and R1.3.2: a behavioural change
needs a test that would have caught the old behaviour, and a speed claim needs a benchmark.

### 1.3 Selective merging

Integrate winning elements into `src/`. Constraints:

| Rule | Statement |
| :--- | :--- |
| R1.3.1 | No behavioural change without a test that would have caught the old behaviour |
| R1.3.2 | No tolerance weakened; a moved number is re-derived from first principles and documented |
| R1.3.3 | Licensing follows `structure.md` §4.2 and D4 — copyleft or unlicensed code is never vendored |
| R1.3.4 | Where the spec is the wrong party, the spec is corrected and the correction recorded |

## 2. Qualitative multidisciplinary test evaluation

`refactor-001` §5.2 audited the suite for **structural** coverage and closed the gaps it found —
911 tests, 99% line and branch coverage. This item asks a different question: not *is it covered*
but **does it make sense**.

The suite can be complete and still be unconvincing: a test can pass over a physically impossible
target, assert a threshold no signal would ever sit near, or pin an implementation detail that
would break under a refactor that changed nothing a caller could observe.

### 2.1 Radar physics

| Ask | Failure it catches |
| :--- | :--- |
| Do test conditions describe physically meaningful scenarios? | A target at an implausible range, altitude or RCS for its class |
| Are target dynamics realisable? | Accelerations or turn rates no aircraft of that type flies |
| Are radar parameters internally consistent? | A PRF, duty cycle and pulse width that could not coexist in one sensor |

### 2.2 Radar signal processing

| Ask | Failure it catches |
| :--- | :--- |
| Is windowing matched to what is being measured? | A taper chosen for sidelobes in a test about resolution |
| Is FFT binning intuitive at the stated parameters? | An expected bin index that happens to be right for the wrong reason |
| Are thresholds matched to expected signal behaviour? | A detection threshold far from any plausible operating point |
| Is Doppler unfolding exercised where folding actually occurs? | An unfolding test whose target never folds |
| Are SNR assertions consistent with the link budget? | An asserted SNR the radar equation does not support |

### 2.3 Software engineering

| Ask | Failure it catches |
| :--- | :--- |
| Are assertions clean and readable? | A test whose failure message says nothing about what broke |
| Are they robust against edge cases? | A test that passes only at the nominal parameters |
| Do they test true invariants? | A test pinned to an implementation detail, not observable behaviour |

### 2.4 Output

Findings are recorded per test or per group, each classified:

| Class | Meaning |
| :--- | :--- |
| **Sound** | Intuitive in all three domains; no action |
| **Weak** | Passes, but for an unconvincing reason; rewrite the setup or the assertion |
| **Wrong** | Asserts something untrue or unphysical; fix, and say what it was hiding |

**Decision: fix weak and wrong, not merely classify them.** An unconvincing setup or assertion is
rewritten so that it convinces; a test asserting something untrue or unphysical is corrected and
the report says what it was hiding. No tolerance is weakened in the process — every rewrite keeps
or tightens the bound it had, and any moved number is re-derived from first principles.

No test is deleted merely for being redundant; redundancy is cheap and a second angle on the same
invariant is not a defect.

## 3. Summary-table hyperlink invariant

Every Markdown **summary table** — a table whose rows stand for sections below it — carries an
explicit anchor link per row, so a reader jumps straight to the detail.

```
| [A.1](#a1-radarsimpy) | RadarSimPy | GPL-3.0 | ... |
        ^^^^^^^^^^^^^^
        anchor to the detailed section below
```

**Scope, per §4.3.** The invariant applies to *navigational* tables only — a table that indexes
sections. It does **not** apply to parameter matrices, acceptance matrices, or comparison tables,
whose rows are data and have no section to point at. Applying it there would produce hundreds of
links to nothing.

`spec/structure.md` Part A already satisfies this (14 linked rows, added in refactor-001 §4.2).
The work is to find any other navigational table that does not, and to state the invariant in
`docs/conventions/style.md` so new ones comply.

| Requirement | Target |
| :--- | :--- |
| R3.1 | Every navigational table row links to its section |
| R3.2 | Every anchor resolves to a real heading in the same file |
| R3.3 | The invariant is written down in `docs/conventions/style.md` |

## 4. Premise audit

### 4.1 `spec/` contains no prototype implementations — §1 is reframed, not executed as written

The temporal claim is correct: `spec/` landed at 13:44 and `src/` at 14:32 on the same day. But
across all six files in `spec/` there is exactly one Python fence — an 8-line `PropagationPaths`
dataclass in `structure.md`, an interface sketch with comments for fields and no logic. No other
fenced block in `spec/` contains a `def`, an `import`, a `return` or a NumPy call.

So there is no rival implementation to cherry-pick from, and "the spec version may be cleaner"
has no referent. §1.2 is therefore reframed: audit the code against the *specified behaviour* —
equations, bounds, contracts, decisions — which is a real and useful audit, and the one the
request is reaching for. It is not the literal function-by-function code diff described.

### 4.2 The upstream comparison was already done, and found nothing adoptable

`refactor-001` §5.1 cross-referenced `src/` against every reference project catalogued in
`structure.md` Part A and adopted nothing — RadarBook, pyAPRiL, RadarSimPy, ovrtx, RASPNet and
AIRadarLib are all GPL, proprietary, or declare no licence, and D4 already requires writing from
published equations; the permissively-licensed references (Phased-Array-Antenna-Model,
pyroomacoustics, RF-Genesis) map to `array/` and raytracing modules that do not yet exist. That
result stands and is not re-litigated here. §1 is about `spec/` versus `src/`, which is a
different comparison.

### 4.3 The anchor invariant is already met where it applies

`structure.md` Part A carries 14 anchored rows. The other 12 tabled files carry 0, but their
tables are parameter and acceptance matrices, not indexes — hence the scoping in §3. The
remaining work is small: state the rule, and verify the existing anchors resolve.

### 4.4 Scenario spec filenames

The request names `spec/scenario-001.md`, `-002`, `-003`. Those files are
`spec/scenario-001-xband.md`, `spec/scenario-002-bistatic.md` and `spec/scenario-003-tracking.md`
since refactor-001 §4.

## 5. Acceptance criteria

| # | Criterion | Verified by |
| :--- | :--- | :--- |
| A1 | `make check` passes | CI, `pre-push` |
| A2 | Every public function in `src/` has an audit verdict recorded | Audit document |
| A3 | Every Q3 performance claim carries a benchmark | Review |
| A4 | Every behavioural change has a test that fails without it | `git diff`, review |
| A5 | No tolerance weakened without a documented first-principles derivation | Review |
| A6 | Every test classified sound / weak / wrong, with weak and wrong addressed | Audit document |
| A7 | Every navigational table row carries a resolving anchor | Link check |
| A8 | The anchor invariant is stated in `docs/conventions/style.md` | `grep` |
| A9 | Spec-vs-code divergences resolved on whichever side was wrong, and recorded | Review |

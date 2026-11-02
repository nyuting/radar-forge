# Instrumentation 001 — Runtime configuration, logging, tracing and health telemetry

> **Reading order.** §1–§3 say what this spec is for, the principles it holds to, and the one
> architectural idea everything else follows: a **hot path** that processes radar data and a
> **cold path** that records what happened, joined by one record per frame. §4–§6 are the
> everyday tools: runtime configuration, logs and timing. §7–§12 are the radar-specific part:
> health metrics that need no ground truth, numerical guards, algorithmic invariants, residual
> monitors, anomaly triggers with a flight recorder, and synthetic injection. §13 puts numbers
> on the cost. §14 says how the new files relate to `data-001`. §15 fixes what a later CPU/GPU
> backend must respect. §16–§20 are enforcement, decisions, acceptance and migration.
>
> | Part | Sections |
> | :--- | :--- |
> | Purpose and principles | [§1](#1-status-purpose-and-scope), [§2](#2-principles) |
> | Architecture | [§3](#3-architecture-hot-path-and-cold-path) |
> | Everyday tools | [§4](#4-runtime-configuration), [§5](#5-logging), [§6](#6-tracing-and-the-real-time-factor) |
> | Health and correctness | [§7](#7-health-telemetry) to [§12](#12-synthetic-injection) |
> | Cost | [§13](#13-cost-of-the-cold-path-the-gil-queues-and-threads) |
> | Relation to other specs | [§14](#14-relation-to-data-001), [§15](#15-constraints-on-a-cpu-or-gpu-backend) |
> | Enforcement, decisions, acceptance, migration | [§16](#16-enforcement) to [§20](#20-open-questions) |
> | Sources | [References](#references) |

## 1. Status, purpose and scope

**Status: proposed.** Nothing below is implemented. Today `src/radar_forge/` has no `logging`
call, no timer and no profiler hook. `core/signal.py` raises one `warnings.warn`, and
`scripts/run_scenario.py` reports progress with `print`. What exists, and what this spec builds
on rather than replaces:

| Exists today | Where | This spec |
| :--- | :--- | :--- |
| Scenario configuration: TOML read with `tomllib` into frozen dataclasses, strict key sets | `pipelines/scenarios.py` `load_scenario`; `pipelines/tracking.py` `configs_from_scenario` | Adds one table, `[runtime]` (§4), loaded the same way |
| A run manifest with provenance | `metadata.json`, `data-001` §6.1 | Adds the resolved runtime settings and a summary (§14) |
| Accuracy metrics scored against truth | `metrics.csv`, `data-001` §6.9 | Adds `telemetry.csv` with the same shape, for metrics that need no truth (§7) |
| Input validation that raises | `core/tracking/_validation.py` (finite, symmetric, positive semidefinite) | Kept as it is. Stage-boundary guards (§8) are a different job |
| NIS and NEES consistency tests | `spec/tracker-001.md` §12, offline Monte Carlo | Adds live, truth-free monitors (§10). NEES stays offline |
| Intermediate quantities exposed by each block | `spec/starter.md` §1.1, "Insightful" | The mechanism this spec relies on: diagnostics leave `core/` as return values |

The spec answers four needs:
1. **Know what a run did** (logs, §5) and **how long each stage took** (spans, §6).
2. **Know whether a run was healthy** without ground truth, as a fielded system would have to
   (§7–§10).
3. **Catch the moment it went wrong** and keep the data around it (§11).
4. **Pay almost nothing for all of this** on the signal-processing path, on CPU today and on GPU
   later (§3, §13, §15).

**In scope:** offline batch runs of `scripts/run_scenario.py` and the pipelines it calls, designed
so the same records can later stream from a live pipeline.

**Out of scope:** network transport, dashboards, a GUI and alert delivery. They are possible
*sinks* for the records defined here (§3); none is specified. The GPU backend itself belongs to a
later `backend-001`. This spec only fixes the interfaces instrumentation needs from it (§15).

**Why "instrumentation" and not "observability".** In this repository *observable* means Kalman
observability: `docs/reviews/pr13-review.md` renamed `observable` to `covers` to keep the meaning
unambiguous. This spec uses *instrumentation* for the whole subject and avoids the bare word.

### 1.1 Practice elsewhere

What comparable projects do, and what this spec takes from each.

| Project | What it does | Taken here |
| :--- | :--- | :--- |
| GNU Radio | Per-block performance counters (work time, buffer fullness) kept by the **scheduler**, not by block code, and read through ControlPort | Timing and counters live at the pipeline boundary, never inside a block (§3) |
| FilterPy `Saver` | Records filter attributes after every step, as data, for later plotting | Diagnostics are data the block returns, not log lines it prints (§2) |
| Stone Soup metric generators | OSPA and SIAP metrics computed beside the tracker, never inside it | Metrics are computed in `pipelines/` from what `core/` returns (§7) |
| PyTorch profiler, NVIDIA NVTX | Named ranges around regions; on GPU, timed by device events without synchronising | Span names double as NVTX range names; GPU timing by events (§6) |
| OpenTelemetry | A vendor-neutral model of traces (spans with parents), metrics and logs | The span and metric *model*, without the dependency. An exporter is an optional extra (§6) |
| Prometheus | Counters and gauges, sampled cheaply, kept separate from logs | Telemetry is a small set of scalars per frame, never a log line per event (§7) |
| Statistical process control (EWMA, CUSUM) | Decades-old, well-characterised change detectors for a stream of scalars | The trigger rules (§11) |

## 2. Principles

1. **The hot path and the cold path are split at the module boundary.** `core/` is the hot path.
   It does no logging, no file I/O, no clock reads and no device-to-host copies for the sake of
   telemetry. Whatever a block knows about its own working (a noise estimate, a threshold, a NIS)
   it *returns*, as `spec/starter.md` §1.1 already asks. `pipelines/` is the only place that
   reads clocks, builds records and hands them to the cold path.
2. **Instrumentation never changes results.** With every instrumentation setting off, and with
   every one on, a run writes byte-identical L1–L4 products (`data-001` §4). Synthetic injection
   (§12) and the future `backend`/`dtype` settings (§15) are the stated exceptions.
3. **Off costs almost nothing; on costs little.** Measured on scenario 003, the overhead is at
   most 1 % of frame time with everything off and at most 5 % with logging at `info`, tracing,
   telemetry and the cheap invariants on. `benchmark` tests hold both numbers (§16).
4. **Standard library first.** `logging`, `contextvars`, `queue`, `threading` and
   `time.perf_counter_ns` are sufficient. OpenTelemetry and structlog are not needed for any of
   it; an OpenTelemetry exporter may come later as an optional extra, per `CLAUDE.md`.
5. **One run directory, one set of join keys.** Every new file lives in `out/<run_id>/` beside the
   data products and joins to them on `frame`, `burst_index` and `time_s` (`data-001` §5). Log
   lines also carry `run_id`, because a log line is often read far from its directory.
6. **Scalars flow, arrays stay.** What crosses from hot to cold is a few scalars per frame plus
   references to arrays that already exist. Arrays are written only when something asks for them:
   a data product, or a snapshot when a trigger fires (§11).
7. **Every check is a small function with a stated tolerance.** Invariants, monitors and trigger
   rules are pure functions, testable alone with analytic cases, and each says where its
   tolerance comes from (`docs/conventions/testing.md`).

## 3. Architecture: hot path and cold path

```text
HOT PATH: core/ functions, called by pipelines/
┌────────────────────────────────────────────────────────────────────┐
│ synthesise ─► range-Doppler ─► CFAR ─► cluster ─► unfold ─► track  │
│      return values: noise estimate, threshold, counts, NIS         │
└──────────────────────────────────┬─────────────────────────────────┘
                                   │  pipelines/ wraps each call: span timing (§6),
                                   │  guards (§8), cheap invariants (§9), injection (§12)
                                   ▼
                     FrameRecord, one per frame:
                     scalars + finished spans + array references
                                   │  queue.put, once per frame
                                   ▼
            ┌────────────────────────────────────────────┐
            │ bounded queue                              │
            │ backpressure: block (batch) or drop (live) │
            └──────────────────────┬─────────────────────┘
                                   ▼
COLD PATH: one recorder thread
        monitors (§10) ─► triggers (§11) ─► sinks
                                   │
                                   ▼
   log.jsonl (§5)      spans.csv (§6)       telemetry.csv (§7)
   events.csv (§11)    snapshots.h5 (§11)   metadata.json summary (§14)
```

- **`FrameRecord`** is a frozen dataclass: `frame`, `time_s`, a mapping of metric name to scalar,
  a list of finished spans, and an optional mapping of product name to array reference for the
  flight recorder. On GPU the scalars are 0-d device arrays; the recorder reads them (§15).
- **The recorder** owns every sink. It runs the monitors and triggers on the scalars, writes rows,
  and on a trigger writes a snapshot. It is the only code that does file I/O for instrumentation.
- **Backpressure.** The queue holds `queue_frames` records (§4). When it is full:
  - `backpressure = "block"` (the default for batch runs) makes `put` wait. A batch run is not
    real time, so waiting costs only wall time and keeps the telemetry complete.
  - `backpressure = "drop"` (for a live pipeline) discards the record and increments
    `n_dropped_records`, which is itself reported. The hot path never waits.
- **Shutdown.** At the end of a run, or on an exception, the pipeline puts a sentinel, joins the
  recorder and then writes `metadata.json`, so the summary sees every record. An exception in the
  recorder is logged and re-raised in the main thread at join time; it never passes silently.

## 4. Runtime configuration

### 4.1 Two kinds of setting

| Kind | Examples | Changes results? | Lives in |
| :--- | :--- | :--- | :--- |
| **Physics** | waveform, sites, target, seed, `[detection]`, `[tracking]` | yes, by design | the existing scenario tables |
| **Runtime** | log level, tracing, telemetry, guards, invariants, triggers, snapshots | **no** (§2.2) | the new `[runtime]` table |
| Runtime, result-affecting | `[[runtime.inject]]` (§12); later `backend`, `dtype` (§15) | yes, and the change is recorded | `[runtime]`, flagged in `metadata.json` |

`metadata.json` gains `provenance.physics_sha256`: the SHA-256 of the canonical JSON of the
resolved physics settings (sorted keys, shortest round-trip floats, `data-001` DF9), excluding
`[runtime]`. Two runs with the same `physics_sha256` and no result-affecting runtime setting must
have identical L1–L4 files. That is what makes principle 2 checkable across runs, not only inside
one test.

### 4.2 The `[runtime]` table

Optional. Every key has a default, so today's scenario files keep working unchanged. Loaded by
`load_scenario` with the existing pattern: an allowed-key frozenset, `_require_keys` for unknown
keys, a frozen `RuntimeConfig` dataclass, and strict type checks with no coercion (as
`pipelines/tracking.py` `_check_value_types` does).

| Key | Type | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `log_level` | string | `"info"` | `debug`, `info`, `warning`, `error` (§5) |
| `log_console` | bool | `true` | Human-readable log lines on stderr, as well as `log.jsonl` |
| `trace` | bool | `false` | Record spans and write `spans.csv` (§6) |
| `telemetry` | bool | `true` | Write `telemetry.csv` (§7). Costs one record per frame |
| `numeric_guard` | string | `"count"` | `off`, `count`, `raise` (§8) |
| `invariants` | string | `"cheap"` | `off`, `cheap`, `full` (§9) |
| `monitors` | bool | `true` | Run the residual monitors (§10). Needs `telemetry` |
| `queue_frames` | int | `64` | Capacity of the recorder queue, in frame records |
| `backpressure` | string | `"block"` | `block` or `drop` (§3) |
| `snapshot_frames` | int | `0` | Frames kept in the flight recorder before a trigger. `0` disables it (§11) |
| `snapshot_post_frames` | int | `0` | Frames still written after a trigger fires |
| `snapshot_levels` | list of string | `["L2"]` | Which product levels a snapshot holds: `L1`, `L2` |
| `max_snapshots` | int | `4` | Snapshots per run; later triggers still log, but no longer write arrays |
| `backend` | string | `"numpy"` | Reserved for `backend-001`. Only `"numpy"` is accepted until it lands |
| `dtype` | string | `"float64"` | Reserved for `backend-001`. Only `"float64"` is accepted until it lands |

Array-of-tables entries: `[[runtime.trigger]]` (§11.1) and `[[runtime.inject]]` (§12).

```toml
[runtime]
log_level = "info"
trace = true
snapshot_frames = 8
snapshot_post_frames = 2

[[runtime.trigger]]
name = "noise_floor_step"
metric = "noise_floor_ratio_linear"
rule = "cusum"
allowance_sigma = 0.5
decision_sigma = 5.0
warmup_frames = 20
action = "snapshot"
```

### 4.3 Overrides and precedence

**Defaults < `[runtime]` in the scenario file < command-line flags.** `scripts/run_scenario.py`
gains `--log-level`, `--trace`, `--no-telemetry`, `--numeric-guard`, `--invariants` and
`--snapshot-frames`. There are no environment variables: a setting that changes what a run
records must be visible in the command line that `metadata.json` already stores
(`provenance.command`).

`metadata.json` records the resolved `RuntimeConfig` under `runtime`, with `cli_overrides` listing
the keys the command line changed, as `tracking.unused` already lists the settings an estimator did
not read.

### 4.4 Seeds

Unchanged: one `np.random.default_rng(scenario.seed)` per run, consumed in frame order
(`pipelines/scenarios.py` `iterate_frames`). Injection (§12) draws no random numbers, so it cannot
shift the stream. When frames are processed in parallel, or on a device, each frame gets its own
generator from `np.random.SeedSequence(seed).spawn(n_frames)`. That is a result-affecting change
and belongs to `backend-001`; it is noted here so no one invents a second scheme.

## 5. Logging

### 5.1 Who configures what

- **Library code** (`src/radar_forge/`) gets loggers with `logging.getLogger(__name__)`, so names
  are `radar_forge.pipelines.tracking` and so on. The package installs a `NullHandler` on
  `radar_forge` and nothing else. This follows the standard library's guidance for libraries: a
  library never configures handlers, levels or formats, because that is the application's choice.
- **`core/` does not log at all** (principle 1; rule R9 in §16). A recoverable problem the *user*
  must see stays a `warnings.warn` with a `stacklevel` (`docs/conventions/style.md` §8). Anything
  else `core/` knows, it returns.
- **The entry point** (`scripts/run_scenario.py`, later any CLI) calls one function,
  `pipelines.runtime.configure_logging(config, run_dir)`, which:
  - attaches a JSON-lines handler writing `log.jsonl` and, if `log_console`, a human-readable
    stderr handler;
  - adds a filter that copies `run_id`, `frame`, `burst_index` and `time_s` from `contextvars`
    into every record, so no function has to pass them down just to log;
  - calls `logging.captureWarnings(True)`, so every `warnings.warn` also lands in `log.jsonl`;
  - replaces today's `print` calls in `scripts/run_scenario.py`.

### 5.2 Levels

| Level | Used for | Volume |
| :--- | :--- | :--- |
| `debug` | Per-frame detail: counts per stage, settings resolved per burst | One or a few lines per frame |
| `info` | Run lifecycle: config resolved, files written, progress every N frames, summary | Tens of lines per run |
| `warning` | Captured warnings, triggers that fired, invariants that failed in `count` mode, dropped records | Rare |
| `error` | An exception, logged once with its traceback before it propagates | Rare |

**Never one line per cell, per detection or per track inside a loop.** A per-item fact is a count
in telemetry (§7) or a row in a data product (`detections.csv`, `tracks.csv`). Log calls use
`%`-style lazy arguments, never f-strings, so a disabled level costs one `isEnabledFor` check.

### 5.3 The `log.jsonl` record

One JSON object per line, UTF-8. Format details are in `data-001` §6.11; the fields are:

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `logged_utc` | string | Wall-clock instant the record was made, ISO 8601 with microseconds and `Z` |
| `level` | string | `debug` … `error` |
| `logger` | string | e.g. `radar_forge.pipelines.tracking` |
| `event` | string | A stable snake_case identifier: `run_started`, `frame_done`, `trigger_fired`, `stop_and_hop_strained`. Readers filter on this, never on `message` |
| `message` | string | The human sentence |
| `run_id` | string | From `metadata.json` |
| `frame`, `burst_index` | int or absent | When the record belongs to one |
| `time_s` | float or absent | Scenario time of that frame (`data-001` §5) |
| `data` | object | Extra key-value pairs, every key with its unit suffix: `{"n_plots": 3, "noise_floor_w": 1.2e-13}` |
| `exc_type`, `exc_message`, `traceback` | string or absent | For a logged exception |

### 5.4 Two clocks, two names

A run has two unrelated times, and mixing them up is a classic logging bug.

| Clock | What it measures | Names |
| :--- | :--- | :--- |
| **Scenario time** | Where in the simulated or recorded data we are | `time_s`, `time_utc`, as in every `data-001` file |
| **Machine time** | When and how long the computer spent | `logged_utc` (wall clock); `elapsed_start_s`, `duration_s` (monotonic clock, §6) |

A log line has both: `time_s` says which frame it is about, `logged_utc` says when it was written.
Spans use only the monotonic clock (`time.perf_counter_ns`), never the wall clock, because the
wall clock can step.

## 6. Tracing and the real-time factor

### 6.1 Spans

A span is a named, timed region with a parent. The API is one context manager:

```python
with tracer.span("cfar", frame=k, burst_index=b):
    detections = cfar_detect(...)
```

- **Where.** Only in `pipelines/`, around calls into `core/`. The standard spans are `run` ⊃
  `frame` ⊃ {`synthesise`, `range_doppler`, `cfar`, `cluster`, `unfold`, `track`, `write`}.
  A pipeline may add finer spans; it may not put one inside `core/`.
- **Off.** When `trace = false`, `tracer` is a shared no-op object whose `span` returns a
  pre-built context manager, so the disabled cost is one method call.
- **On, CPU.** `time.perf_counter_ns` at entry and exit. The finished span goes into the frame's
  `FrameRecord`, not straight to a file.
- **On, GPU (defined now, built with `backend-001`).** Entry and exit record device events on the
  current stream (`cupy.cuda.Event`, `torch.cuda.Event`). The hot path never calls
  `synchronize()` for timing. The recorder resolves the events after the frame's last event has
  completed. The span name is also pushed as an NVTX range, so Nsight Systems shows the same
  names.
- **Inside a stage.** Spans say *which* stage is slow. To see *why*, use a sampling profiler with
  no code change: `py-spy record` on CPU, Nsight Systems on GPU, or `cProfile` for a call graph.
  None of these becomes a dependency.

### 6.2 `spans.csv`

One row per finished span. Columns are in `data-001` §6.12: `frame` (empty for `run`),
`burst_index`, `stage`, `parent_stage`, `elapsed_start_s` (seconds since the run's monotonic
start), `duration_s` and `device` (`cpu`, `cuda:0`).

### 6.3 The real-time factor

The single number that says whether processing keeps up with the data:

```text
rtf = processing time / data time
```

| Metric | Processing time | Data time |
| :--- | :--- | :--- |
| `rtf_processing_linear` | Sum of the `frame` spans **minus** `synthesise` and `write` | `n_frames / frame_rate_hz` |
| `rtf_total_linear` | Wall time of the `run` span | `n_frames / frame_rate_hz` |

`rtf < 1` keeps up with real time. A run that spends 20 minutes on 1 s of data has
`rtf_total_linear = 1200`. The two numbers answer different questions:
- `rtf_processing_linear` is what a fielded receiver would face, because a receiver does not
  synthesise its own input.
- The gap between the two is the cost of simulation.

Both are written per frame to `telemetry.csv` and per run to `metadata.json`. The first thing to
do with a slow run is look at them and at the per-stage spans. A GPU is the answer only if the
slow stage is one a GPU accelerates (§15).

## 7. Health telemetry

Telemetry is a small set of **scalars per frame (and per burst)** that say whether the run is
healthy, computed **without ground truth**. That is the difference from `metrics.csv`, which scores
the run against truth after the fact. A fielded radar has only the first kind.

### 7.1 Where the numbers come from

Each metric is either already returned by a `core/` function, or is one cheap reduction over an
array the stage already holds:
- **Already returned:** CFAR noise estimate (`cfar_noise_estimate_w`), threshold and mask
  (`cfar_detect`), cluster count, NIS (`TrackSnapshot.nis`), track status counts.
- **One extra reduction:** the sum used by the non-finite guard (§8), the maximum magnitude, a
  count over the CFAR mask.

On a GPU each reduction stays on the device as a 0-d array until the recorder reads it (§15).

### 7.2 Metrics

Names follow `data-001` §6.9: unit suffix in the name, counts `n_`-prefixed, linear on disk.

| Area | Metric | Scope | Meaning |
| :--- | :--- | :--- | :--- |
| Signal | `noise_floor_w` | burst | Mean CFAR noise estimate over the valid cells |
| Signal | `noise_floor_ratio_linear` | burst | `noise_floor_w` over the prediction of §9.2. Near 1 when healthy |
| Signal | `peak_to_noise_linear` | burst | Largest RD-cell power over `noise_floor_w` |
| Signal | `n_nonfinite_samples` | burst, per stage | Non-finite values found by the guard (§8) |
| Signal | `n_saturated_samples` | burst | Samples at the converter's full scale, when the receiver model has one (scenario 009); empty otherwise |
| Detection | `n_valid_cells`, `n_threshold_crossings` | burst | CFAR cells with a full reference window, and those over threshold |
| Detection | `crossing_fraction_linear`, `pfa_design_linear` | burst | Their ratio, beside the design Pfa (§10.5) |
| Detection | `n_detections`, `n_plots_accepted`, `n_plots_rejected` | frame | Clusters, and plots by `status` (`data-001` §6.5) |
| Tracking | `n_tracks_tentative`, `n_tracks_confirmed`, `n_tracks_coasting` | frame | Tracks by status after the update |
| Tracking | `n_tracks_born`, `n_tracks_deleted` | frame | Lifecycle transitions this frame |
| Tracking | `mean_nis_dim<k>` | frame | Mean NIS over this frame's updates, per measurement dimension (`data-001` §6.9) |
| Tracking | monitor statistics | track, frame | §10 |
| Runtime | `frame_duration_s`, `rtf_processing_linear`, `rtf_total_linear` | frame | §6.3 |
| Runtime | `n_queued_records`, `n_dropped_records` | frame | Queue depth when this record was put, and drops so far (§3) |
| Runtime | `peak_rss_bytes` | run | Peak resident memory, from the standard library's `resource` module on POSIX; empty where it is unavailable (Windows) |

### 7.3 `telemetry.csv`

Tidy long, like `metrics.csv`, so one reader handles both. The columns are `metrics.csv`'s
(`metric`, `track_id`, `target_id`, `frame_start`, `frame_end`, `value`) plus `time_s` and
`burst_index`. A per-frame value has `frame_start = frame_end`. `target_id` is always empty here,
because telemetry knows no truth. The format is in `data-001` §6.13.

Telemetry is cheap but not free. At 30 scalars per frame and 10⁴ frames, `telemetry.csv` holds
3 × 10⁵ rows, about 15 MB. That is acceptable for a research run. A long live run would rotate
the file, which is left to the streaming work (§20).

## 8. Numerical guards

### 8.1 What exists, and what is added

`core/tracking/_validation.py` already rejects non-finite, asymmetric or indefinite inputs at the
tracking API and raises. That is **input validation**: it protects a function from its caller,
and it stays.

A **stage guard** is different. It checks a stage's *output* in `pipelines/`, so a NaN made in
the range-Doppler map is caught there and not three stages later as a mysterious empty track
list.

### 8.2 The check, at the cost of one reduction

At each stage boundary (`synthesise`, `range_doppler`, `cfar`, `track`), for each array the
stage returns:
1. **Detect** with one sum: `s = x.sum()`. NaN propagates through a sum and ±∞ survives it, so
   `isfinite(s)` is false exactly when some element is not finite, or when the sum overflowed,
   which is itself worth reporting. On a GPU this is one device reduction.
2. **Count** only on failure: `n = count_nonzero(~isfinite(x))`. This second pass runs only when
   the first one fails.

| `numeric_guard` | On failure |
| :--- | :--- |
| `off` | No check |
| `count` (default) | Write `n_nonfinite_samples` for that stage and burst, log a `warning` with `event = "nonfinite_output"`, add a row to `events.csv`. Processing continues |
| `raise` (used by tests) | Raise `FloatingPointError` naming the stage, frame, burst and count |

In `raise` mode the stage also runs under
`np.errstate(divide="raise", invalid="raise", over="raise")`, so the operation that made the NaN
raises, not only the boundary after it.

**Never a silent repair.** No stage calls `np.nan_to_num`, clips or masks a non-finite value to
keep going (`docs/conventions/style.md` §8, "never a silent clamp"). Where a non-finite value is
*meaningful* (`data-001` §5: NaN means "computed and invalid"), the stage documents which output
may carry it, and the guard skips that output by name.

## 9. Algorithmic invariants

### 9.1 Rules

- **Not `assert`.** `python -O` removes `assert` statements, and `docs/conventions/style.md` §8
  already bans them for validation. An invariant is a function.
- **Pure, in `core/`.** Each invariant lives in `core/invariants.py`. It takes arrays the stage
  already returned and gives back an `InvariantResult(name, holds, residual, tolerance)`. It does
  not raise and does not log. Unit tests drive it with a case that holds and a case that is
  violated, both known analytically.
- **Called by `pipelines/`**, by cost level. A failure is an `events.csv` row and a `warning`, or
  an `InvariantError` (a `RuntimeError` subclass) when `numeric_guard = "raise"`.
- **Tagged by cost.** A `cheap` invariant costs at most one pass over an array the stage already
  holds, or O(1). A `full` invariant may cost as much as the stage itself and is for debugging.

### 9.2 The invariants

The equation numbers are checked against each text when the block is implemented
(`docs/conventions/style.md` §4.1).

| # | Level | Invariant | Tolerance and its basis |
| :--- | :--- | :--- | :--- |
| I1 | cheap | **Parseval across the range-Doppler FFTs.** For unnormalised DFTs of lengths `n_range_fft`, `n_doppler_fft` over the windowed cube `x_w`: `Σ\|X\|² = n_range_fft · n_doppler_fft · Σ\|x_w\|²`. Zero-padding does not change it | `rtol = 1e-10`. FFT round-off grows as O(ε log₂ N); 1e-10 leaves margin above it for N ≤ 2²⁰ |
| I2 | cheap | **Noise floor matches its prediction.** White noise of power σ² per sample, through windows w_f and w_s and two unnormalised DFTs, has mean power `σ² Σ\|w_f\|² Σ\|w_s\|²` per cell. For the pulsed path the matched filter `h` replaces `w_f`. Checked as `noise_floor_ratio_linear` | Statistical: within `5 / √M_eff` of 1, with M the valid cells and `M_eff = M / (B_f B_s)`. B is each window's equivalent noise bandwidth in bins (Harris 1978), which accounts for the correlation the window puts between neighbouring bins. The approximation is verified by a test on noise-only frames (§18) |
| I3 | cheap | **Track bookkeeping is conserved.** `n_alive(k) = n_alive(k−1) + n_born(k) − n_deleted(k)`, with alive meaning tentative, confirmed or coasting | Exact |
| I4 | cheap | **Time moves forward.** Frame `time_s` is strictly increasing; every span's `duration_s ≥ 0` | Exact |
| I5 | cheap | **Plots lie inside their map, and unfolding is consistent.** Every `range_index`, `velocity_index` lies inside its burst's map. Every unfolded velocity equals the folded one plus `2 · fold_index · v_unamb` | Indices exact; velocity `rtol = 1e-12` |
| I6 | full | **Covariances stay valid.** Every updated covariance is symmetric and positive semidefinite, checked with the existing `as_covariance` | The round-off allowance `_validation.py` already justifies |
| I7 | full | **The two covariance forms agree on a linear model.** For a Taylor-linearised update, the Joseph form and the short form of `spec/tracker-001.md` §4.3 agree | The tolerance `spec/tracker-001.md` §4.5 states for its guards |

The I2 prediction is analytic, so a failing I2 means a wrong noise power, a wrong window
normalisation or a scaling bug. That is a better check than "the noise floor did not move", which
would accept a scaling bug present from frame 0. The *trend* in the noise floor is a monitor
(§10.4), not an invariant.

## 10. Residual and innovation monitors

### 10.1 What a monitor is

A monitor turns a stream of per-frame quantities into a **test statistic and a flag** that need no
ground truth. Its maths is a pure function in `core/monitoring.py`, beside small frozen state
dataclasses (a window, a running sum). The recorder holds the state and calls the function once per
frame, so a monitor costs the hot path nothing. Every statistic and flag is a `telemetry.csv`
metric, and so a possible trigger input (§11).

The offline consistency tests stay where they are. NEES needs truth, so it remains a Monte-Carlo
test (`spec/tracker-001.md` §12). These monitors are the live, truth-free counterpart.

What monitors need from the tracker: each update's NIS, which `TrackSnapshot.nis` already carries,
and its **whitened innovation** `L⁻¹ ν`, where `L` is the Cholesky factor of `S`, which it does not
carry yet. Adding `whitened_innovation` to `TrackSnapshot` is migration step 6 (§19). It is a
return value, so it keeps principle 1.

### 10.2 The monitors

All follow Bar-Shalom, Li and Kirubarajan (2001) §5.4 unless stated. α is the per-test
false-alarm probability, `alpha` in the trigger, default 0.01.

| # | Monitor | Statistic | Healthy when |
| :--- | :--- | :--- | :--- |
| M1 | **Windowed NIS, per track** | Sum of the last N NIS values of one track | Inside the two-sided 1 − α interval of χ²(N·n_z). Under a correct model the NIS values are independent χ²(n_z) |
| M2 | **Windowed NIS, pooled** | The same over all confirmed tracks' updates this window | Inside χ²(Σ n_z) at 1 − α. More updates give more power to detect a wrong noise model |
| M3 | **Innovation whiteness, per track** | Normalised sample autocorrelation of the whitened innovations at lags 1…J, over a window of N updates | Each lag within `±z_{1−α/2} / √N`. Correlated innovations mean a wrong motion model or a mistimed update |
| M4 | **Mean innovation, per sensor** | Mean whitened innovation per measurement component, over all confirmed tracks in the window; also reported in physical units (`range_innovation_mean_m`, `range_rate_innovation_mean_mps`) | Each component within `±z_{1−α/2} / √M` of 0, M the updates in the window. A persistent offset is a sensor bias: range offset, timing error, Doppler sign |
| M5 | **Crossing fraction against design Pfa, per burst** | `n_threshold_crossings` minus the crossings inside clusters that became accepted plots, over `n_valid_cells` | At or below the upper 1 − α Clopper–Pearson bound of Binomial(`n_valid_cells`, `pfa_design_linear`). Excess crossings mean the noise model under the CFAR is wrong (non-Gaussian clutter, an interferer) |

Monitor M1 is also the classical manoeuvre detector (Bar-Shalom et al. 2001, ch. 11). A flag on
one track during a turn is expected and is information, not a fault. That is why monitors only
produce flags, and §11 decides what a flag means.

## 11. Anomaly triggers and the flight recorder

### 11.1 Triggers

A trigger watches **one scalar metric** from `telemetry.csv` and fires an action. Triggers run in
the recorder, on scalars only, so they cost the hot path nothing. Domain knowledge stays in the
monitors (§10); the trigger rules are generic.

| `rule` | Parameters | Fires when | Source |
| :--- | :--- | :--- | :--- |
| `bounds` | `lower`, `upper` (either optional) | The value leaves `[lower, upper]`. Also the rule for monitor flags (`upper = 0`) | — |
| `ewma_zscore` | `lambda_linear` (0.1), `z_sigma` (4.0), `warmup_frames` (20) | The value is more than `z_sigma` standard deviations from its exponentially weighted mean, with mean and variance estimated over the warm-up and then updated | Roberts (1959); Lucas and Saccucci (1990) |
| `cusum` | `allowance_sigma` (0.5), `decision_sigma` (5.0), `warmup_frames` (20) | A two-sided cumulative sum of standardised deviations crosses `decision_sigma`. Detects small persistent steps that `ewma_zscore` misses | Page (1954); Montgomery (2019) ch. 9 |

Common keys: `name`, `metric`, `scope` (`burst`, `frame`, `track`: which rows of the metric it
watches), `action`, and `cooldown_frames` (default 10). A fired trigger stays quiet for its cooldown,
so a persistent fault gives one event, not one per frame.

The defaults are the textbook settings for detecting a shift of about one standard deviation with
a low false-alarm rate. Their average run lengths are tabulated in Montgomery (2019) ch. 9, and
the trigger tests (§18) check against those tables rather than against recorded output.

### 11.2 Actions

| `action` | Effect |
| :--- | :--- |
| `log` | A row in `events.csv` (`data-001` §6.14) and a `warning` with `event = "trigger_fired"` |
| `snapshot` | As `log`, and the flight recorder writes its buffer (§11.3) |
| `raise` | As `log`, then raise `AnomalyError` in the main thread at the next frame boundary. For debugging and tests |

Invariant failures (§9), guard failures (§8), injection misses (§12) and dropped records (§3) also
write `events.csv` rows, with their own `kind`, so one file lists everything unusual about a run.

### 11.3 The flight recorder

Aviation's flight recorder keeps the last minutes, not the whole flight. This one keeps the last
`snapshot_frames` frames' arrays, so that when a trigger fires there is something to look at.
- **Ring buffer.** A `collections.deque(maxlen=snapshot_frames)` of `FrameRecord` array references,
  for the levels in `snapshot_levels` (L1 cube, L2 RD map, CFAR threshold and mask).
- **No copies on the CPU path.** The ring holds references to arrays the hot path has already
  finished with. **A stage that reuses an output buffer must hand the ring a copy instead**, and
  its docstring says so; otherwise the snapshot would show a later frame's data.
- **On GPU,** the reference is a device array; the copy to host happens only when a trigger fires.
- **When a trigger fires,** the recorder writes the buffered frames, then `snapshot_post_frames`
  more, to `snapshots.h5`, one group per event (`/event_<event_id>/`). Inside it are the
  `iq.h5` and `products.h5` trees (`data-001` §6.7, §6.8), with a frame axis that holds only the
  snapshot's frames, so the same readers work with the group path as a prefix. The `events.csv`
  row names the group.
- **Memory is stated, not discovered.** At start-up the pipeline logs the ring's size,
  `snapshot_frames × bytes per frame`, at `info`.
- **Without `h5py`** (the `io` extra, `data-001` DF8), a `snapshot` action degrades to `log` and
  one `warning` per run. The run does not fail.

## 12. Synthetic injection

### 12.1 Purpose

**What exists.** Scenario 009 injects an analogue pilot tone to calibrate receiver channels, and
scenario 008 places test targets in clutter. Both are about one slice. Neither checks, while a run
executes, that the detection chain still detects.

A **built-in test target** does that. It is a known point target added digitally to a copy of
each frame's cube. The pipeline then checks that it was detected in the expected cell at the
expected SNR. The result is a live check of the processing chain: windowing, FFT scaling, CFAR and
clustering, with no ground truth for real targets needed.

It tests the digital chain only. An RF front-end fault is invisible to it, which is what
scenario 009's analogue pilot is for.

### 12.2 Specification

```toml
[[runtime.inject]]
name = "bit_far"
range_m = 18000.0
radial_velocity_mps = -120.0   # closing-positive (data-001 §5)
snr_sample_db = -10.0          # per sample, before integration
bursts = [0]                   # optional; default all bursts
```

- **Off by default.** It is a result-affecting runtime setting (§4.1), so `metadata.json` records
  it and principle 2 does not apply to a run that uses it.
- **After L1.** Injection is added to a copy of the cube *after* `iq.h5` has its frame, so L1 stays
  the clean cube and bit-identical. L2–L4 carry the injected target.
- **Synthesis is reused.** The injected echo goes through `core/signal.py`'s existing
  path-to-baseband synthesis, with one `PropagationPaths` entry and no noise. There is no second
  echo model.
- **Snapped to bin centres.** The injector moves `range_m` and `radial_velocity_mps` to the nearest
  range and Doppler bin centres, so there is no scalloping loss, and records the snapped values.
- **Expected SNR.** `snr_sample_linear × n_pulses × n_samples × L_w`, where `L_w` is the combined
  processing loss of the two windows (`core/windows.py` `processing_loss_db`, converted to
  linear). This is the coherent integration gain of the radar equation (Richards 2014).
- **Check.** A plot exists within one cell of the expected cell, and its `peak_power_w` over the
  noise floor is within the 1 − α interval of the Rician peak-cell statistic at that SNR. Each
  frame writes `injection_detected` (0 or 1) and `injection_snr_error_db`, a dB convenience beside
  its linear source (`data-001` §2.4), to `telemetry.csv`. A miss writes an `events.csv` row.
- **Tagged and withheld.** The matched plot gets `injected = 1` in `detections.csv`
  (`data-001` §6.5) and is **never passed to the tracker**. Readers filter on the column, and
  nothing downstream mistakes it for a target.
- **Limitation.** A real target within a cell of the injected one confounds the check. The user
  places the injection; the pipeline warns if a confirmed track comes within two cells of it.

## 13. Cost of the cold path: the GIL, queues and threads

Principle 3 sets the budget. This section says why the design fits inside it, and what is measured
rather than assumed.

| Operation | Typical cost (CPython 3.11, one core) | How often | Note |
| :--- | :--- | :--- | :--- |
| Disabled `logger.debug(...)` | ~0.1 µs | a few per frame | `isEnabledFor` is cached |
| Disabled span | ~0.2 µs | ~8 per frame | Shared no-op context manager |
| Enabled span | ~1 µs | ~8 per frame | Two `perf_counter_ns` calls and a small append |
| `queue.Queue.put` | ~1–2 µs | **once per frame** | Takes a lock; this is why records are per frame, not per event |
| One extra reduction (guard sum) | one pass over the array | per stage output | Memory bandwidth, not Python. The dominant term, and still small beside an FFT over the same array |
| JSON-encoding a log record | ~10–30 µs | in the recorder | Off the hot path, but holds the GIL (below) |

The microsecond figures are orders of magnitude for planning. §18 replaces them with measured
numbers on the reference machine, recorded beside the baseline as `spec/tracker-001.md` §11 does.

**Why a thread is enough.**
- The hot path spends its time in NumPy and SciPy kernels. Most ufunc loops, `scipy.fft` and the
  LAPACK calls release the GIL while they run on large non-object arrays.
- A recorder thread therefore competes with the hot path only for the short Python glue between
  kernels, and its own work (formatting rows, appending to files) is short.
- The share of GIL time the recorder takes is **measured**: the AC2 benchmark runs with the
  recorder on and off.

**Why not more.**
- **`asyncio`** helps code that waits on I/O. The hot path waits on arithmetic.
- **A recorder process** (`multiprocessing`, with `shared_memory` for snapshot arrays) removes GIL
  contention, at the cost of pickling records and a more complex shutdown. `FrameRecord` is plain
  data, so the switch is possible later without changing the hot path. It is made only if the AC2
  benchmark shows the GIL binding.
- **Free-threaded CPython** (PEP 703, 3.13t and later) would remove the question. It is noted, not
  relied on: `numpy` and `scipy` support for it is still settling, and the project targets 3.11.

**On GPU** (§15) the cost moves. The hot path must not wait for the device to report a scalar.
Records carry 0-d device arrays, and the recorder waits on the frame's completion event in its own
thread. Waiting in that thread blocks that thread only.

## 14. Relation to data-001

**Logs and telemetry are records *about* a run; data products are the run's *results*.** They are
different kinds of file, but they live in the same run directory and follow the same rules: join
keys, unit suffixes, linear on disk, shortest round-trip floats, and one owner for every format.

| Concern | Owner |
| :--- | :--- |
| File formats, columns, dtypes, units, the run-directory tree, versioning, the validator | `data-001`. It gains §6.11–§6.15 for the new files, and `schema_version` becomes 1.3.0 |
| When a file is written, what each value means, what computes it, what it costs | This spec |

The new files:

| File | Format | Content | `data-001` |
| :--- | :--- | :--- | :--- |
| `log.jsonl` | JSON Lines | Log records (§5.3) | §6.11 |
| `spans.csv` | CSV | Timing spans (§6.2) | §6.12 |
| `telemetry.csv` | CSV | Health metrics, tidy long (§7.3) | §6.13 |
| `events.csv` | CSV | Everything unusual: triggers, invariant and guard failures, injection misses, drops (§11) | §6.14 |
| `snapshots.h5` | HDF5 | Flight-recorder frames (§11.3) | §6.15 |

**Why JSON Lines for logs only.** `data-001` §3 rejected JSON Lines for detections and tracks,
because nothing there is nested enough to justify losing spreadsheet and MATLAB readers. Logs are
different:
- records are heterogeneous (each `event` has its own `data` keys);
- they carry tracebacks;
- they are appended while the run is still going.

JSON Lines is the common format of structured logging, and each line is readable with `json`,
`jsondecode` or nlohmann/json. The decision is recorded as DI4 here and noted in `data-001` §3.

`metadata.json` gains:
- `runtime`: the resolved settings and `cli_overrides` (§4.3);
- `provenance.physics_sha256` (§4.1) and `provenance.machine` (platform, processor, CPU count,
  device names), so a timing can be read against the machine that produced it;
- `telemetry_summary`: `rtf_processing_linear`, `rtf_total_linear`, `wall_duration_s`, `n_events`,
  `n_dropped_records`, `n_nonfinite_samples`.

## 15. Constraints on a CPU or GPU backend

The backend itself is `backend-001`'s subject. Instrumentation needs these from it, and they are
recorded now so that the backend does not have to be retrofitted.

1. **Array API, not NumPy by name.** Hot-path functions take their namespace from their input
   (`xp = array_namespace(x)`, from `array-api-compat` in an extra), so the same code runs on
   NumPy, CuPy or PyTorch. `xp.fft` and `xp.linalg` are part of the standard.
2. **No hidden host transfers.** No `float(x)`, `.item()`, `np.asarray(x)` or Python `if` on a
   device value inside the hot path. Each one forces the host to wait for the device. Metrics are
   0-d device arrays until the recorder reads them.
3. **Batched along the leading axis.** The frame generator's per-frame loop is fine on CPU, but a
   GPU needs work in batches of frames to stay busy. `FrameRecord`s are then produced per frame
   from the batch's outputs, so instrumentation does not change.
4. **Timing by device events** (§6.1), never by synchronising.
5. **Precision is a stated choice.** The default stays float64/complex128. Consumer GPUs run
   float64 at 1/32 to 1/64 of their float32 rate, so a `complex64` mode will be wanted. It must be
   explicit (`dtype = "float32"`), recorded, and checked against the float64 path to a stated
   tolerance under the `gpu` marker (`docs/conventions/testing.md`).
6. **Random numbers differ by device.** CuPy's and PyTorch's generators do not reproduce NumPy's
   streams. GPU synthesis therefore matches CPU synthesis statistically, not bit for bit, and the
   `SeedSequence.spawn` scheme of §4.4 keeps each frame reproducible on its own device.
7. **Measure before porting.** §6.3 shows which stage dominates. Synthesis (many paths × pulses ×
   samples), the FFTs and CFAR are good GPU candidates. The tracker's small matrices and Python
   bookkeeping are not, and should stay on CPU.

## 16. Enforcement

Tags as in `docs/conventions/style.md`.

| Rule | Enforced by |
| :--- | :--- |
| `core/` contains no `import logging`, no `print`, no `time`/`perf_counter` call and no file I/O for instrumentation (§2.1) | **[hook]**: a new rule **R9** in `scripts/check_conventions.py`, scoped to `src/radar_forge/core/` |
| Instrumentation off and on give byte-identical L1–L4 files (§2.2) | **[test]**: AC1 |
| Overhead within budget (§2.3) | **[test]**, `benchmark` marker: AC2 |
| `[runtime]` rejects unknown keys and wrong types (§4.2) | **[test]**: AC4 |
| Every invariant, monitor and trigger rule has an analytic test (§2.7) | **[test]**: AC5–AC7 |
| No log call inside a per-item loop (§5.2) | **[review]**. A loop over cells is easy for a tool to find; a loop over tracks that logs once per track is a judgement call |
| A stage that reuses an output buffer copies into the flight recorder (§11.3) | **[review]** |
| New files match `data-001` §6.11–§6.15 | **[validator]**, once `data-001` §9's validator exists |

## 17. Decisions

### DI1 — `core/` emits nothing; diagnostics are return values

§2.1. A block that logged or timed itself would tie the maths to a run's settings and slow every
caller, including the teaching notebooks. Returning intermediate quantities is already the
project's bar (`spec/starter.md` §1.1), so this costs `core/` nothing new.
*Rejected:* a logger in every module, the common default (it puts I/O and settings into the hot
path, and on GPU a log line that formats a value is a hidden synchronisation).

### DI2 — Instrumentation cannot change results

§2.2. Without this rule, a user who turns tracing on to find a bug can no longer trust they are
looking at the same run. The one-run byte-identity test (AC1) and `physics_sha256` across runs
make the rule checkable. Injection and the future backend settings are the declared exceptions,
and each is recorded in `metadata.json`.

### DI3 — Standard library, not structlog or OpenTelemetry

§2.4. `logging` with a `contextvars` filter gives structured records with context, which is
structlog's main benefit, and needs no dependency. The span and metric model follows OpenTelemetry,
so an exporter is a mapping, not a redesign. *Rejected:* structlog in core (a dependency for what
the standard library does); the OpenTelemetry SDK in core (heavy, built for services, and its value
is in a collector this project does not run).

### DI4 — JSON Lines for logs, and only for logs

§14. Logs are heterogeneous, nested and streamed; tables are not. `data-001` §3's rejection of
JSON Lines for detections and tracks stands.

### DI5 — `telemetry.csv` is its own file, shaped like `metrics.csv`

§7.3. The two answer different questions: is the run healthy (no truth, live), and how good is it
(truth, after the fact). Separate files keep a truth-free consumer from reading truth by accident.
The shared shape means one reader handles both. *Rejected:* one combined file with a `source`
column (it mixes truth-scored and truth-free numbers in one table a live system would read).

### DI6 — One record per frame through a bounded queue; block in batch, drop in live

§3. Per-frame records keep queue overhead at microseconds per frame. Blocking is the right default
for a batch run, where completeness matters and wall time is cheap. Dropping, counted, is right for
a live pipeline, where falling behind the antenna is the failure.
*Rejected:* always drop (a batch run's telemetry would depend on machine load); always block
(a live pipeline would stall on a slow disk).

### DI7 — The recorder is a thread

§13. NumPy and SciPy release the GIL in their kernels, so a thread is enough, and it is the
simplest thing that keeps file I/O off the hot path. A process is allowed later if a benchmark
shows the GIL binding.

### DI8 — Triggers are generic and run on scalars; domain tests are monitors

§10, §11. Keeping the statistics (χ², whiteness, binomial) in monitors and the change detection
(bounds, EWMA, CUSUM) in triggers lets each be tested alone with analytic cases. A trigger never
touches an array, so it costs the hot path nothing.

### DI9 — Invariants are functions with levels, never `assert`

§9. `assert` disappears under `-O`, carries no residual and cannot be switched by cost. A function
returning `InvariantResult` can be unit-tested, logged with its residual, and chosen per run.

### DI10 — `[runtime]` lives in the scenario file, overridden on the command line

§4. One file still describes a run completely, and `provenance.command` records every override. The
physics hash excludes `[runtime]`, so runs that differ only in instrumentation are recognisably the
same physics. *Rejected:* environment variables (invisible in the record); a separate runtime file
(a second file to keep beside every scenario, for little gain at this scale; revisit with
streaming).

### DI11 — Scenario time and machine time have different names

§5.4. `time_s`/`time_utc` are scenario time everywhere in `data-001`. Machine time is
`logged_utc`, `elapsed_start_s` and `duration_s`. Spans use only the monotonic clock.

### DI12 — GPU timing by device events, never by synchronising

§6.1, §15. A `synchronize()` for timing serialises the device and measures a different program
from the one that runs untimed.

### DI13 — Synthetic injection is after L1, tagged, and withheld from the tracker

§12. Injecting after L1 keeps the raw record clean. The `injected` column keeps the plot visible,
and withholding it from the tracker keeps it from becoming a track.

## 18. Acceptance criteria

| # | Criterion | Checked by |
| :--- | :--- | :--- |
| AC1 | Scenarios 001 (FMCW dual-PRF) and 003 run with every instrumentation setting off, then with every one on (`trace`, `telemetry`, `invariants = "full"`, `numeric_guard = "count"`, a trigger with `action = "snapshot"`), injection off: every L1–L4 file is byte-identical, and so is `physics_sha256` | [test] |
| AC2 | On scenario 003, frame time with everything off is within 1 % of a build without instrumentation, and within 5 % with the default settings plus `trace`. The baseline, the factor and the machine are recorded in §13 by the PR that lands step 4 | [test], `benchmark` |
| AC3 | `core/` passes rule R9 | [hook] |
| AC4 | `[runtime]` with an unknown key raises naming the key and the allowed set; a wrong type raises `TypeError`; a scenario file without `[runtime]` loads with the defaults | [test] |
| AC5 | Every invariant I1–I7 returns `holds = True` on an analytic case and `False` on a constructed violation. I2 holds on 200 noise-only frames at the stated tolerance with no more failures than its α allows (binomial interval) | [test], I2 `slow` |
| AC6 | Each monitor M1–M5, on a model-matched simulation, flags at a rate inside the binomial interval of its α; with a stated range bias, coloured process noise or a non-Gaussian clutter amplitude injected, it flags within a stated number of frames | [test], `slow` |
| AC7 | `ewma_zscore` and `cusum` on a stationary Gaussian series and on a one-σ step match the average run lengths tabulated in Montgomery (2019) ch. 9 within their Monte-Carlo interval; `bounds` fires exactly at its edges; `cooldown_frames` suppresses repeats | [test] |
| AC8 | A NaN placed in one frame's cube is counted in `count` mode (event row, telemetry value) and raises `FloatingPointError` naming the stage, frame and burst in `raise` mode | [test] |
| AC9 | An injected target at a stated SNR is detected in its expected cell on every frame at high SNR, its plot has `injected = 1`, it reaches no track, and `iq.h5` equals the run without injection | [test] |
| AC10 | Every line of `log.jsonl` parses as JSON with the §5.3 required fields; a `warnings.warn` raised in `core/` appears as a record; no `print` remains in `scripts/run_scenario.py` | [test] |
| AC11 | A trigger at frame k writes frames `k − snapshot_frames + 1 … k + snapshot_post_frames` to `snapshots.h5`, readable with `data-001` §7.1's snippet; without `h5py` the run completes with one warning | [test] |
| AC12 | The new files pass the `data-001` validator | [validator], once it exists |
| AC13 | This spec passes `scripts/check_conventions.py` rule R7 | [hook] |

## 19. Migration and follow-ups

Each step is its own PR, and each leaves scenario 003's end-to-end numbers where they were
(principle 2).

1. **Rule R9** in `scripts/check_conventions.py`, so nothing creeps into `core/` while the rest
   lands. `core/` passes it today.
2. **`pipelines/runtime/`**: `config.py` (`RuntimeConfig`, the `[runtime]` loader, the CLI
   overrides, `physics_sha256`) and `log.py` (`configure_logging`, the JSON-lines formatter, the
   context filter). Replace the `print` calls in `scripts/run_scenario.py`. Write `runtime` and
   `provenance.machine` to `metadata.json`.
3. **Spans** (`pipelines/runtime/spans.py`), `spans.csv` and the real-time factor. Commit the
   timing baseline for AC2.
4. **Recorder and telemetry**: `FrameRecord`, the queue, the recorder thread, `telemetry.csv` and
   `telemetry_summary`. Record the measured costs in §13.
5. **Guards and the cheap invariants**: `core/invariants.py` (I1–I5), the stage guards, `events.csv`.
6. **Monitors**: `whitened_innovation` on `TrackSnapshot` (a `core/tracking/` change, reviewed
   against `spec/tracker-001.md`), `core/monitoring.py` (M1–M5), the full invariants (I6, I7).
7. **Triggers**: the EWMA and CUSUM updates as pure functions in `core/monitoring.py`, the wiring
   and cooldowns in `pipelines/runtime/triggers.py`, with the `log` and `raise` actions.
8. **Flight recorder** and `snapshots.h5`. Needs `data-001` follow-ups 3 and 4 (the `io` extra and
   the HDF5 writers).
9. **Synthetic injection**, and the `injected` column in `detections.csv`.
10. **`spec/backend-001.md`**: the CPU/GPU backend, written against §15 and informed by step 3's
    timings.

## 20. Open questions

1. **Streaming sinks.** When a live pipeline exists, which transport carries `FrameRecord`s off the
   machine (a socket, ZeroMQ, an OpenTelemetry collector)? The record is designed to serialise;
   the transport is not chosen.
2. **Log and telemetry rotation** for runs long enough that one file per run is too large.
3. **An OpenTelemetry exporter** as an extra: worth it only if someone runs a collector.
4. **Free-threaded CPython**: revisit DI7 when NumPy and SciPy declare support and the project's
   minimum Python allows it.

## References

- Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and
  Navigation*, Wiley, 2001: §5.4 (NIS, time-averaged NIS, innovation whiteness and its test) and
  ch. 11 (the NIS as a manoeuvre detector).
- E. S. Page, "Continuous inspection schemes", *Biometrika* 41(1/2), 100–115, 1954 (CUSUM).
- S. W. Roberts, "Control chart tests based on geometric moving averages", *Technometrics* 1(3),
  239–250, 1959 (EWMA).
- J. M. Lucas and M. S. Saccucci, "Exponentially weighted moving average control schemes:
  properties and enhancements", *Technometrics* 32(1), 1–12, 1990.
- D. C. Montgomery, *Introduction to Statistical Quality Control*, 8th ed., Wiley, 2019, ch. 9
  (CUSUM and EWMA design, average run lengths).
- C. J. Clopper and E. S. Pearson, "The use of confidence or fiducial limits illustrated in the
  case of the binomial", *Biometrika* 26(4), 404–413, 1934.
- M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014 (CFAR and
  its Pfa; coherent integration gain; noise through the range-Doppler chain).
- F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform",
  *Proc. IEEE* 66(1), 51–83, 1978 (equivalent noise bandwidth, processing loss).
- A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010
  (Parseval's relation for the DFT).
- Python Software Foundation, *Logging HOWTO*, "Configuring Logging for a Library"; and the
  `logging`, `contextvars` and `queue` module documentation. https://docs.python.org/3/
- S. Gross, *PEP 703 — Making the Global Interpreter Lock Optional in CPython*, 2023.
- Consortium for Python Data API Standards, *Python array API standard*, v2023.12.
  https://data-apis.org/array-api/
- OpenTelemetry Authors, *OpenTelemetry Specification*, Trace and Metrics API.
  https://opentelemetry.io/docs/specs/otel/
- NVIDIA, *NVTX (NVIDIA Tools Extension Library)* and *Nsight Systems User Guide*.
- GNU Radio project, *Performance Counters* and *ControlPort* documentation. https://wiki.gnuradio.org/
- B. Beyer, C. Jones, J. Petoff and N. R. Murphy (eds.), *Site Reliability Engineering*, O'Reilly,
  2016, ch. 6 (monitoring: symptoms over causes, few signals, cheap to collect).
- `spec/data-001-formats.md` §3–§6, §9; `spec/tracker-001.md` §4.3, §4.5, §11, §12;
  `spec/starter.md` §1.1; `docs/conventions/style.md` §2, §4.1, §8; `docs/conventions/testing.md`.

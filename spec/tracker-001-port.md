# Tracker 001 port inventory and validation

## Port boundary

The general tracker is now `radar_forge.tracking`. The deprecated source contains **15**
Python modules in its tracking directory (the planning count of 16 was incorrect); all 15
are included. This port copies 31 files in total: 20 library modules, seven test modules,
two scripts, and two specifications. Two additional test modules cover the new runner
boundary and coexistence/multi-sensor integration.

The source checkout's HEAD was `45428bac37f03acd28bbf3e4a8ab942973a53410`, but its tracker
and annotations were uncommitted. The hashes below identify the actual working files
ported, not merely that commit. The earlier upstream hashes in
[tracker-001-provenance.md](tracker-001-provenance.md) remain unchanged historical records.

The existing tracker, detector, scenario loader, simulation, scenario configurations,
trajectory data, and scenario runner retain their implementations. Interface simplification,
new tracking algorithms, moving sensors, and broader IQ adapters remain follow-up work.

## Source-to-destination inventory

Paths in the first column are relative to `radar-forge-DEPRECATED`; destination paths are
relative to this repository at the initial port. The subsequent observation-class relocation
is recorded below. Hashes are SHA-256 of the original source file bytes.

| Source | Destination | Source SHA-256 |
| :--- | :--- | :--- |
| `src/radar_forge/core/tracking/__init__.py` | `src/radar_forge/tracking/__init__.py` | `72b02da2a4a1d30b0d1ac770a08fdcf6b6f837985b5732b0263ef77ae3ad97d3` |
| `src/radar_forge/core/tracking/_numerics.py` | `src/radar_forge/tracking/_numerics.py` | `71f3cf2df40a2cde2b6b67f1fa9135f7adee87f7dec036e102f705e68cbf20fc` |
| `src/radar_forge/core/tracking/association.py` | `src/radar_forge/tracking/association.py` | `7681aefc6f6a91b27591d8071e40e3eef13524d39d586c1b9aaf5c7d329d51a8` |
| `src/radar_forge/core/tracking/derived.py` | `src/radar_forge/tracking/derived.py` | `281ab1014ec474cb999622db2431f9e2fd045221f7061a36944f63237abfd1e1` |
| `src/radar_forge/core/tracking/engine.py` | `src/radar_forge/tracking/engine.py` | `950c5bbd88a22ea467d0a6b10e5e95e714643e5564dbc99d9320880135abb3cc` |
| `src/radar_forge/core/tracking/estimation.py` | `src/radar_forge/tracking/estimation.py` | `0432cade0a419774c7b407a618d65a46853c2fa38303ef97bafdab9bf73a477c` |
| `src/radar_forge/core/tracking/imm.py` | `src/radar_forge/tracking/imm.py` | `b1dacd491bb0582da8d38c3211e0a38b86c6b099ac6f092c38322258c8b2a6d0` |
| `src/radar_forge/core/tracking/initiation.py` | `src/radar_forge/tracking/initiation.py` | `2e63927674c251b898124de0c09d66850de5b222687978f70d6f4e493b4f7290` |
| `src/radar_forge/core/tracking/management.py` | `src/radar_forge/tracking/management.py` | `b1d277cd1a481de86ce2609b1fe72599b303ece14d58e48a66e7cc22c6286fee` |
| `src/radar_forge/core/tracking/measurements.py` | `src/radar_forge/tracking/measurements.py` | `cd4ac12fe939c7f1df6682f8a47b86dd65666e79b1824f8681ff3f8156964bb5` |
| `src/radar_forge/core/tracking/motion.py` | `src/radar_forge/tracking/motion.py` | `f3257270006a9063a375577d396ac27be3b699f428d58ee463c2b2e97193c38f` |
| `src/radar_forge/core/tracking/sensors.py` | `src/radar_forge/tracking/sensors.py` | `4bb37d2c66ac882b0ad5ec24d0e679570a9062b05418db6d39afcc51b0a43597` |
| `src/radar_forge/core/tracking/spaces.py` | `src/radar_forge/tracking/spaces.py` | `b05e42e8c21f55bbeb016330d216ab06d91e456da9e96f32ed711cb1c0c45bad` |
| `src/radar_forge/core/tracking/tracks.py` | `src/radar_forge/tracking/tracks.py` | `1b4192986c92dc3a9bf20ab9f117453f10f6e5a62ce8d61c2040566328385871` |
| `src/radar_forge/core/tracking/ukf.py` | `src/radar_forge/tracking/ukf.py` | `55ceb0ce40e3661529e6d09483131e5d447b2b3d941e6e5e6310f02010da91b3` |
| `tests/core/tracking/test_engine.py` | `tests/tracking/test_engine.py` | `1352bddcfbd4f86db6fccff91ecfa0f6d4cf49685a845aa275af7bf7702d4e17` |
| `tests/core/tracking/test_measurements.py` | `tests/tracking/test_measurements.py` | `be9293f959a09a32475c950df4c1b88f8d50dad1476ddbd6b148040e58eb3fcf` |
| `tests/core/tracking/test_motion.py` | `tests/tracking/test_motion.py` | `750f7b3e04440d4925fb415f23dd3ccb6b215306fecfe9ece417a38ae1d31c03` |
| `tests/core/tracking/test_ukf.py` | `tests/tracking/test_ukf.py` | `cb0ce8f815fbe19ef937a95f9c6a0140497d136a4592fd517a31d380a20f536f` |
| `src/radar_forge/core/detection.py` | `src/radar_forge/core/detection_2d.py` | `65e129429aa31ed588d1f1714862e95e6f7b21fefa684da1f0b24ece4aa20ae8` |
| `tests/core/test_detection.py` | `tests/core/test_detection_2d.py` | `41179681f6f3e275e521b5aa2c0e3551531bbe9f23159f0529984e262f9878a3` |
| `src/radar_forge/pipelines/tracking.py` | `src/radar_forge/pipelines/general_tracking.py` | `5574f9300632e8aed49d7813a25e4063e3d4f0de7703b15be7dcb6ac5b57a066` |
| `tests/pipelines/test_tracking.py` | `tests/pipelines/test_general_tracking.py` | `0d864974975ff37e456408d8c5c33a2853cd8615214423cd2effdea9d816ef71` |
| `scripts/run_scenario.py` | `scripts/run_general_tracking.py` | `0d8353aef589d94dfe418db794300f4f88b7c32edc83d3bbddba1f221c01941f` |
| `src/radar_forge/pipelines/tracking_config.py` | `src/radar_forge/pipelines/tracking_config.py` | `d0c851a92011686d02b43f2dde66482e0b8852e94648709a2742557031c4835f` |
| `src/radar_forge/pipelines/tracking_output.py` | `src/radar_forge/pipelines/tracking_output.py` | `79197af2071aca0343f06f40949210634709e5707aee42e029a91ce5e1d98c6a` |
| `src/radar_forge/teaching/scopes/track_history.py` | `src/radar_forge/teaching/scopes/track_history.py` | `aefda33db169d5d3505d46eb5d0d2027c1765499e334c06f75574b36347a5284` |
| `tests/teaching/test_track_history.py` | `tests/teaching/test_track_history.py` | `412484c6d127aeadef8cced67ec6bcd1974fe6780179518db3e8d8544891ce28` |
| `scripts/run_tracking_example.py` | `scripts/run_tracking_example.py` | `6cc77d72c8c9a00ea8b34ea2623c473980f27e16c77bdbbcfd583c48f61766ce` |
| `spec/tracker-001-integration.md` | `spec/tracker-001-integration.md` | `3cdeebfa665d60ee0d0f4a78fb7235bd3bde015cd3386edea6ece54a01d7130f` |
| `spec/tracker-001-provenance.md` | `spec/tracker-001-provenance.md` | `9e6d84c0e1cbc1cf3d38e3b365fbd289cd7622afd283e771a7280b120e4f5c35` |

### Merged source portions and current dependencies

| Source portion | Destination / disposition |
| :--- | :--- |
| `src/radar_forge/core/__init__.py` detector exports | Merge `CfarResult`, `DetectionConfig`, `cfar_2d`, and the `detection_2d` module into the current core exports. |
| `src/radar_forge/pipelines/__init__.py` tracker exports | Merge nonconflicting exports; use `GeneralScenarioTracker` and `GeneralTrackingConfig` aliases for existing name conflicts. Export the three added pipeline modules explicitly. |
| `src/radar_forge/teaching/scopes/__init__.py` history exports | Merge `track_history` and `render_track_history`; preserve the current plot exports. |
| `src/radar_forge/pipelines/scenarios.py` detector/tracker configuration fields and parsing | Keep the current scenario schema. Move the imported strict table parsing into the dedicated runner's separate `--tracker-config` boundary. |
| `README.md` native-tracking section | Port into the current README's General tracking section with updated paths and approved origin. |
| `docs/README.md` tracker links | Merge tracker specification and audit links into the current documentation index. |
| `spec/structure.md` tracker subpackage decision | Port the tracker-specific paragraph under General tracking package; preserve current architecture decisions. |
| Source Scenario 001 specification's Tracking extension section | Merge into `spec/scenario-001-xband.md` with the dedicated runner's entry point. |
| Existing physics, ambiguity, signal processing, trajectories and Scenario 001 data | Reuse current implementations and Duke scenarios; do not overwrite them with deprecated files or datasets. |
| Deprecated output artifacts and caches | Do not import generated artifacts or environments. Rerun examples to generate outputs locally. |

No tracker notebooks or separate tracker fixture datasets were found. The imported pipeline
tests use the current three Scenario 001 TOMLs and their current trajectory data. The source
tracker tests outside `core/tracking`—detector, pipeline, and plotting—are all included.

### Compatibility additions

- Imports target `radar_forge.tracking`, `core.detection_2d`, and `pipelines.general_tracking`.
- Imported consumers use current `Scenario.bursts`, `Scenario.n_pulses`, and
  `form_range_doppler_map`. Runner metadata reads `Transmitter.chirp_duration_s` while retaining
  the imported `chirp_time_s` output key; Pillow conversion uses `Image.Palette.ADAPTIVE`.
- The dedicated runner enables tracking by default and accepts the old `--track` flag.
  `--tracker-config` reads only `[detection]` and `[tracking]` from a separate TOML file;
  unknown tables/keys and incorrect scalar types are rejected. Omitted values use imported
  defaults. Existing scenario tracker tables belong to the original tracker.
- The runner rejects nonpositive `--frames` and bistatic IQ geometry before creating outputs.
  Its radial adapter supports one monostatic FMCW/pulsed burst or two FMCW PRFs.
  The general Python engine retains its bistatic measurement and multi-sensor interfaces.
- Added explicit `__all__` declarations to imported library modules to satisfy this repository's
  API contracts, retaining the original general package exports. Added `radar_forge.tracking`
  to root exports without replacing `radar_forge.core.tracking`.
- Four new NumPy-style docstrings document the previously comment-only helpers in `_numerics.py`.
  Existing comments remain in place.

## Annotation and location audit

Token and AST comparisons checked **1,116 original comment tokens** and **157 original docstrings** across all copied Python files. Every source comment was retained verbatim and in order. Existing docstrings are identical except for
these approved substitutions:

| File | Approved change |
| :--- | :--- |
| `src/radar_forge/pipelines/general_tracking.py` | Module docstring's source location name changed to Duke. |
| `scripts/run_tracking_example.py` | Module docstring's source location name changed to Duke. |
| `scripts/run_general_tracking.py` | Module docstring's usage command names the new runner and wraps the longer path. |

Imported geographic fixture/example origins, including zero-valued geographic origins,
now use the existing Duke receiver origin `(36.00250, -78.94100, 60.0)`. Local Cartesian
positions and offsets retain their original meaning. The mismatched-origin regression
uses a state with no geodetic origin against the Duke sensor origin, preserving the
mismatch rejection without introducing another physical location.

The original provenance specification is byte-identical. Other imported documentation
updates excluded location names and moved paths, separates the two tracker interfaces,
and labels original performance measurements as historical rather than Duke results.

Six source files contain inherited long comments, comment whitespace, or custom docstring
layout that automatic formatting would change: `_numerics`, `association`, `derived`,
`engine`, `estimation`, and `imm`. `pyproject.toml` excludes only those files from formatting
and exempts only their observed annotation lint categories. Remaining lint and type rules
still run. Blank code lines, final newlines, imports, exports, and adapted test call wrapping
were adjusted without rewriting the preserved comments.

An AST comparison of **all 20 copied library modules**, ignoring import relocations, export declarations, and documentation additions, found no algorithm changes.
This is a source-preserving port, not a tracker tuning or interface redesign.

## Validation

The pre-port baseline passed 936 tests, lint, formatting, and conventions. An initial run
using the already available development environment lacked Pygments stubs; installing this
repository's locked dev/teaching dependencies resolved those environment-only type errors.
No runtime dependencies were added and `uv.lock` is unchanged.

Final `make check` passed: formatting and lint passed, mypy reported no issues in 50 source
files, conventions passed, and **1,140 tests passed**. The source annotation/algorithm audit,
excluded-location scan, and byte-for-byte historical provenance check also passed.

Validation covers:

- All original tracker, detector, pipeline and history-plot tests, without weakened tolerances.
- The new runner's three scenario variants, partial/default/custom configuration, malformed
  settings, streaming outputs, modulo range interpretation, and unsupported geometry rejection.
- Both tracking APIs importing together without optional plotting imports or source paths into
  the deprecated checkout; two sensors asynchronously updating both crossing targets.
- All six seeded ENU example combinations (`x`, `xy`, `xyz`, each CV/CA), producing 120 JSONL
  records and a confirmed final track with the configured state dimension.
- Dedicated-runner headless and plotting smoke runs, including the history PNG; the existing
  Scenario 003 runner still writes its existing output files.
- The full repository `make check` gate: formatting, lint, mypy, conventions, and pytest.

### Current Duke 120-second measurements

These runs use the current Scenario 001 configurations, seed 20260911, and the imported
default tracker settings. Evaluation fixes the first confirmed ID and counts only its
confirmed frames. RMSE therefore describes the listed subset, not whole-window accuracy
or continuous identity. S2 range errors are modulo range; its absolute range is unresolved.

| Variant | Births | First confirmation (s) | Confirmed primary frames / 120 | Range RMSE (m) | Velocity RMSE (m/s) | Frames without observations |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 72 | 2.0 | 18 | 32.82 | 9.56 | 0 |
| S2 | 138 | 2.0 | 10 | 11.36 | 4.73 | 6 |
| S3 | 23 | 2.0 | 14 | 12.49 | 8.10 | 7 |

The inherited defaults fragment tracks on these trajectories. The port exposes that behavior;
it does not claim to solve it or adjust thresholds to conceal it. Historical source-window
results remain in [the integration specification](tracker-001-integration.md).

Reproduce each run with the corresponding existing Scenario 001 TOML:

```bash
uv run python scripts/run_general_tracking.py scenarios/scenario_001_fmcw_low_prf.toml \
  --out out/general-s1 --no-iq --no-plots --no-movie
uv run python scripts/run_general_tracking.py scenarios/scenario_001_pulsed_medium_prf.toml \
  --out out/general-s2 --no-iq --no-plots --no-movie
uv run python scripts/run_general_tracking.py scenarios/scenario_001_fmcw_dual_prf.toml \
  --out out/general-s3 --no-iq --no-plots --no-movie
uv run python scripts/run_tracking_example.py --axes xyz --acceleration --out out/enu-xyz-ca.jsonl
make check
```

## Follow-up: observation ownership

The file-boundary refactor retains all 15 tracking modules. `Measurement` and
`MeasurementBatch` moved from `tracking/measurements.py` to `tracking/sensors.py`, alongside
`Sensor` and its batch-building method. Their complete class blocks, including decorators,
comments, docstrings, validation, and fields, were copied verbatim. Measurement models,
`SensorPose`, and `_Geometry` remain in `measurements.py`.

Internal consumers now import observations from their owning module. Both prior public
paths (`radar_forge.tracking` and `radar_forge.tracking.measurements`) explicitly re-export
the same classes, with no wrappers or subclasses. The classes' defining `__module__` is
now `radar_forge.tracking.sensors`; constructor signatures and processing behavior are
unchanged. The snapshot annotation in `Sensor` remains behind `TYPE_CHECKING` to avoid
runtime cycles.

The initial port's source hashes and audit results above remain historical records. For
this follow-up, the two moved class blocks were compared byte-for-byte against the
pre-refactor files, and every tracking class/function AST was compared across the package
before and after the move. Existing comment tokens and docstrings were also checked across
the relocation; only new ownership explanations were added. No physical locations were added.

The updated ownership map is in [structure.md](structure.md). Configuration/builders and
all numerical algorithms retain their existing interfaces and implementations. Regression
coverage in `tests/tracking/test_sensors.py` checks class identity across all three import
paths, detached read-only observations, empty scans, mismatched sensor/timestamp rejection,
invalid shapes/covariance, nonfinite time, and unregistered routes. Existing tests continue
to exercise multi-sensor and multi-target tracking, measurement models, lifecycle, pipelines,
optional-dependency-free imports, and both tracking implementations.

Follow-up validation passed: all 996 existing tracking-package comment tokens and 124
docstrings were preserved, and all 48 top-level class/function ASTs were unchanged.
`sensors.py` is now 184 lines and `measurements.py` is 425 lines, including annotations.
`make check` passed with **1,151 tests**, and three-frame smoke runs of both the original
Scenario 003 runner and the general Scenario 001 runner completed successfully.

## Pull-request preparation

The subsequent reading annotations and `_michael_notes.md` are included unchanged.
Annotation-only lint exceptions and formatter exclusions now cover the additional
annotated tracking modules, preserving their comment and docstring text. Numerical,
import, and type checks remain enabled. The current full `make check` gate passed
with 1,151 tests before preparing the pull request.

The reading notes now live at `docs/tracking/_michael_notes.md` to satisfy the package-layout
rule. Their contents are unchanged. The private `_Geometry` Boolean flag is named
`include_velocity` to distinguish it from a physical velocity; existing annotations are
retained, with an added naming explanation. Formatter exclusions also apply when the
pre-commit hook supplies explicit file paths.

#!/usr/bin/env python3
"""Run a scenario through the stepper: each stage prints, checks and writes what it found.

    uv run python scripts/run_scenario.py config/scenario_001_fmcw_low_prf.toml --frames 3

Thin by design: the stages live in ``radar_forge.evaluate`` and the scenario's ``[verify]``
table says which run and what they check. The exit status is 1 when a check fails.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from radar_forge.evaluate import STAGES, VerificationError, run
from radar_forge.scenario import load_scenario


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scenario through the stepper.")
    parser.add_argument("scenario", type=Path, help="scenario TOML, such as config/<name>.toml")
    parser.add_argument("--frames", type=int, default=None, help="frames to run (default: all)")
    parser.add_argument("--stop-after", choices=list(STAGES), help="last stage to run")
    parser.add_argument("--save-iq", action="store_true", help="write iq_NNNNN.npz per frame")
    parser.add_argument("--no-plots", action="store_true", help="skip the PNG figures")
    parser.add_argument("--out", type=Path, help="run directory (default: runs/<name>)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scenario = load_scenario(args.scenario)
    order = list(STAGES)
    stages = list(scenario.verify.get("stages", order))
    if args.stop_after is not None:
        stages = [name for name in stages if order.index(name) <= order.index(args.stop_after)]
    try:
        report = run(
            scenario,
            stages,
            n_frames=args.frames,
            out_dir=args.out or Path("runs") / args.scenario.stem,
            save_iq=args.save_iq,
            render=not args.no_plots,
        )
    except VerificationError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"{scenario.name}: all {len(report.checks)} check(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

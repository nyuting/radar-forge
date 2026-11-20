#!/usr/bin/env python3
"""Run a scenario through the stepper: each stage prints, checks and writes what it found.

    uv run python scripts/run_scenario.py config/scenario_001_fmcw_low_prf.toml --scans 3

Thin by design: the stages live in ``radar_forge.verify`` and register with its
``stepper``; the scenario's ``[verify]`` table says which run and what they check.
The exit status is 1 when a check fails.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from radar_forge.scenario import load_scenario
from radar_forge.verify import STAGES, VerificationError, run


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scenario through the stepper.")
    parser.add_argument("scenario", type=Path, help="scenario TOML, such as config/<name>.toml")
    parser.add_argument("--scans", type=int, default=None, help="scans to run (default: all)")
    parser.add_argument("--stop-after", choices=list(STAGES), help="last stage to run")
    parser.add_argument("--save-iq", action="store_true", help="write iq_NNNNN.npz per scan")
    parser.add_argument("--no-figures", action="store_true", help="skip the PNG figures")
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
            n_scans=args.scans,
            out_dir=args.out or Path("runs") / args.scenario.stem,
            save_iq=args.save_iq,
            render=not args.no_figures,
        )
    except VerificationError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"{scenario.name}: all {len(report.checks)} check(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

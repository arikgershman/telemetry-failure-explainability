#!/usr/bin/env python
"""Run CPU/offline contract acceptance through the project's public CLIs.

    python scripts/integration_smoke.py --backend stub --output-dir runs/integration-cpu

The default uses mock simulation and stub inference. Select --sim-backend mujoco
for fresh real-physics CPU acceptance using restored release configurations.
Both modes use explicit stub abstentions; real model acceptance is separate.
Successful and failed runs
retain stage logs and integration_manifest.json; reuse requires a fresh output
directory. Exit codes: 0=passed, 1=smoke failure, 2=invalid arguments/setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from telemetry_reasoning.integration import SmokeFailure, run_cpu_smoke


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("stub",), default="stub")
    parser.add_argument("--sim-backend", choices=("mock", "mujoco"), default="mock")
    parser.add_argument("--batch", type=Path, help="Restored batch for MuJoCo candidate configurations")
    parser.add_argument("--parser-config", type=Path, help="Parser profile for fresh MuJoCo acceptance")
    parser.add_argument("--stub-tokenizer-dir", type=Path,
                        help="Optional local pinned tokenizer for exact CPU stub context accounting (MuJoCo only)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=723)
    parser.add_argument("--stage-timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    try:
        if args.sim_backend == "mock":
            if args.batch is not None or args.parser_config is not None or args.stub_tokenizer_dir is not None:
                raise ValueError("--batch, --parser-config and --stub-tokenizer-dir require --sim-backend mujoco")
            manifest = run_cpu_smoke(args.output_dir, seed=args.seed, stage_timeout_s=args.stage_timeout_s)
        else:
            if args.seed != 723:
                raise ValueError("MuJoCo acceptance uses recorded representative seeds; --seed applies only to mock")
            from telemetry_reasoning.sim_integration import DEFAULT_BATCH, run_mujoco_cpu_smoke
            manifest = run_mujoco_cpu_smoke(args.output_dir, batch=args.batch or DEFAULT_BATCH,
                                            parser_config=args.parser_config or Path(__file__).resolve().parents[1] / "configs/parser.yaml",
                                            stub_tokenizer_dir=args.stub_tokenizer_dir,
                                            stage_timeout_s=args.stage_timeout_s)
    except SmokeFailure as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.sim_backend == "mock":
        print("CPU contract smoke passed (SYNTHETIC; no real physics or model acceptance).")
    else:
        print("Fresh MuJoCo CPU acceptance passed (real physics, synthetic stub inference; archive certification separate).")
        if manifest.get("status") == "passed_with_context_refusals":
            refusals = manifest["checks"]["context_refusals"]
            print(f"Context refusals: {refusals['refused']}/{refusals['requested']} requested stub calls; model execution remains incomplete.")
    print(f"Verified {manifest['checks']['prediction_count']} predictions and {len(manifest['checks']['plot_files'])} plots.")
    print(f"Receipt: {Path(manifest['output_dir']) / 'integration_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

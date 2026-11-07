#!/usr/bin/env python
"""Plot raw success/failure telemetry with retained samples and computed divergence.

    python scripts/plot_telemetry.py \\
        --data-dir data/sample \\
        --bt-dir data/processed/smoke/bt \\
        --output-dir runs/smoke/plots

One ``telemetry_<pair_id>.png`` per pair listed in ``<data-dir>/metadata.csv``.
Each figure shows both raw runs, the retained (compressed) samples from the
pair's BT, subtask boundaries, units, and a divergence panel (``null`` is drawn
as "not computed", never as zero). Optional event markers, each with its own
colour and line style because they are different events:

    --manifest FILE      injection time (Track 2 sidecar)
    --predictions FILE   predicted onset of --variant (default baseline_1)

If a pair has no BT file the raw traces are still plotted. Matplotlib runs
headless; no display is needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from telemetry_reasoning.eval.contracts import MODEL_VARIANTS, PredictionRecord, read_predictions
from telemetry_reasoning.eval.metrics import load_manifest, predicted_onset
from telemetry_reasoning.eval.plots import plot_pair_telemetry, save_figure, trace_from_rows

METADATA_FILENAME = "metadata.csv"


# ASSUMPTION: the shared readers take a file path and return validated row objects.
def _read_metadata(path: Path) -> Any:
    from telemetry_reasoning.io import read_metadata_csv

    return read_metadata_csv(path)


def _read_telemetry(path: Path) -> Any:
    from telemetry_reasoning.io import read_telemetry_csv

    return read_telemetry_csv(path)


def _load_bt(path: Path) -> dict[str, Any]:
    from telemetry_reasoning.schemas import PairedBehaviorTree

    return PairedBehaviorTree.model_validate_json(path.read_text()).model_dump(mode="json")


def _value(x: Any) -> Any:
    return getattr(x, "value", x)


def render_pair(
    row: Any, data_dir: Path, bt_dir: Path, output_dir: Path, *,
    injection_time: Optional[float] = None, record: Optional[PredictionRecord] = None,
) -> Path:
    pair_id = str(_value(row.pair_id))
    success = trace_from_rows(_read_telemetry(data_dir / str(row.success_file)))
    failure = trace_from_rows(_read_telemetry(data_dir / str(row.failure_file)))
    bt_path = bt_dir / f"pair_{pair_id}_bt.json"
    bt = _load_bt(bt_path) if bt_path.is_file() else None

    onset = predicted_onset(record) if record is not None else None
    note = "SYNTHETIC: stub predictions" if record is not None and record.backend == "stub" else None
    title = (
        f"Pair {pair_id}: {_value(row.task)}; injected {_value(row.failure_mode)} "
        f"in {_value(row.failed_subtask)}"
    )
    fig = plot_pair_telemetry(
        success, failure, bt=bt, injection_time=injection_time, predicted_onset=onset, title=title, note=note
    )
    return save_figure(fig, output_dir / f"telemetry_{pair_id}.png")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--bt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--variant", default="baseline_1", choices=MODEL_VARIANTS)
    parser.add_argument("--pair-id", action="append", help="plot only this pair (repeatable)")
    args = parser.parse_args(argv)

    rows = list(_read_metadata(args.data_dir / METADATA_FILENAME))
    if args.pair_id:
        wanted = set(args.pair_id)
        rows = [r for r in rows if str(_value(r.pair_id)) in wanted]
        unknown = wanted - {str(_value(r.pair_id)) for r in rows}
        if unknown:
            print(f"error: pair id(s) not in metadata: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
    if not rows:
        print("error: no pairs to plot", file=sys.stderr)
        return 2

    injection = load_manifest(args.manifest).injection_times if args.manifest else {}
    records: dict[str, PredictionRecord] = {}
    if args.predictions:
        for rec in read_predictions(args.predictions).records:
            if rec.model_variant == args.variant:
                records[rec.pair_id] = rec

    for row in rows:
        pid = str(_value(row.pair_id))
        path = render_pair(
            row, args.data_dir, args.bt_dir, args.output_dir,
            injection_time=injection.get(pid), record=records.get(pid),
        )
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

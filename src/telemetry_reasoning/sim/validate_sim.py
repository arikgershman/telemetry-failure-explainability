"""Validate nominal batches or paired failure-injection datasets."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv
from telemetry_reasoning.schemas import TASK_SUBTASKS, Task

ROOT = Path(__file__).resolve().parents[3]
TASKS = {
    'pickup': ('telemetry_reasoning.sim.batch', Task.PICK_UP_OBJECT),
    'transport': ('telemetry_reasoning.sim.transport_batch', Task.TRANSPORT_OBJECT),
}


def stages(rows):
    observed = []
    for row in rows:
        if not observed or row.subtask != observed[-1]:
            observed.append(row.subtask)
    return observed


def validate_trace(path, task, *, allow_partial=False):
    rows = read_telemetry_csv(path)  # full canonical schema + 50 Hz check
    expected = list(TASK_SUBTASKS[task])
    observed = stages(rows)
    if observed != (expected[:len(observed)] if allow_partial else expected):
        raise ValueError(f'{path}: wrong subtask sequence: {observed}')
    if abs(rows[0].timestamp) > 1e-8:
        raise ValueError(f'{path}: timestamp must start at zero')
    return rows


def validate_nominal(name, runs, seed, spread):
    module, task = TASKS[name]
    result = subprocess.run(
        [sys.executable, '-m', module, '--runs', str(runs),
         '--seed', str(seed), '--spread', str(spread)],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode:
        raise RuntimeError(f'{name}: runner failed')
    paths = [Path(line.partition('Output:')[2].strip())
             for line in result.stdout.splitlines() if line.startswith('Output:')]
    if not paths:
        raise ValueError(f'{name}: no new output directory found')

    batch_path = max(paths, key=lambda p: p.stat().st_mtime_ns)
    batch = paths[0]
    with (batch / 'summary.csv').open(newline='') as f:
        summary = list(csv.DictReader(f))
    if len(summary) != runs or any(r['result'] != 'success' for r in summary):
        raise ValueError(f'{name}: nominal execution failures')
    files = sorted(batch.glob(f'{name}_*.csv'))
    if len(files) != runs:
        raise ValueError(f'{name}: expected {runs} traces, found {len(files)}')
    for file in files:
        validate_trace(file, task)
    print(f'PASS {name}: {runs} nominal runs; canonical telemetry and cadence')


def validate_pairs(folder):
    folder = Path(folder)
    metadata = read_metadata_csv(folder / 'metadata.csv')
    diagnostics_path = folder / 'diagnostics.jsonl'
    diagnostics = {}
    if diagnostics_path.exists():
        for line in diagnostics_path.read_text().splitlines():
            record = json.loads(line)
            diagnostics[record['pair_id']] = record
    for pair in metadata:
        nominal = validate_trace(folder / pair.success_file, pair.task)
        failure = validate_trace(folder / pair.failure_file, pair.task, allow_partial=True)
        if len(nominal[0].joint_positions) != len(failure[0].joint_positions):
            raise ValueError(f'{pair.pair_id}: mismatched joint dimensions')
        if pair.pair_id not in diagnostics:
            raise ValueError(f'{pair.pair_id}: missing diagnostics')
        d = diagnostics[pair.pair_id]
        if not d['nominal']['controller_outcome'] == 'success':
            raise ValueError(f'{pair.pair_id}: nominal did not succeed')
        if d['injected']['controller_outcome'] != 'failed':
            raise ValueError(f'{pair.pair_id}: injected episode did not fail')
        if not d['injected']['failure']['activated']:
            raise ValueError(f'{pair.pair_id}: injection not activated')
        if d['nominal']['initial_offset_xy'] != d['injected']['initial_offset_xy']:
            raise ValueError(f'{pair.pair_id}: initial conditions mismatch')
    print(f'PASS {folder}: {len(metadata)} validated pairs')
    print('NOTE: partial failure traces pass CSV validation but need parser support.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['pickup', 'transport', 'all'], default='all')
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--seed', type=int, default=723)
    p.add_argument('--spread', type=float, default=0.015)
    p.add_argument('--pairs-dir', type=Path)
    args = p.parse_args()
    if args.pairs_dir:
        validate_pairs(args.pairs_dir)
    else:
        for name in (TASKS if args.task == 'all' else [args.task]):
            validate_nominal(name, args.runs, args.seed, args.spread)
    print('ALL VALIDATIONS PASSED')


if __name__ == '__main__':
    main()

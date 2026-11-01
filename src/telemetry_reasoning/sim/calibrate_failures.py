# need to calibrate failure parameters so the tasks actually fail

"""Empirically calibrate failure-injection configurations on paired MuJoCo episodes.

Uses the same candidate configurations and episode runner as dataset.py.
Outputs all tested configurations, outcomes, and best candidates per mode/subtask.
"""
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from telemetry_reasoning.schemas import Task
from telemetry_reasoning.sim.dataset import configs_for
from telemetry_reasoning.sim.runner import ROOT, run_episode


def config_dict(config):
    return {
        'mode': config.mode.value,
        'subtask': config.subtask.value,
        'magnitude': float(config.magnitude),
        'axis': int(config.axis),
        'delay_s': float(config.delay_s),
    }


def run_calibration(task, conditions, seed, spread, threshold, output, verbose=False):
    task = Task(task)
    output.mkdir(parents=True, exist_ok=True)
    configs = configs_for(task)

    # Cache nominal outcomes: the same initial conditions are reused for every config.
    nominals = {}
    for condition in range(conditions):
        nominal = run_episode(task=task, seed=seed,
                              initial_condition_id=condition, spread=spread)
        nominals[condition] = nominal
        if verbose:
            print(f'  nominal condition {condition}: '
                  f'{"success" if nominal.task_succeeded else "FAILED"}', flush=True)

    valid_conditions = [i for i, r in nominals.items()
                        if r.task_succeeded and r.diagnostics['complete_subtasks']]
    print(f'[{task.value}] valid nominal conditions: '
          f'{len(valid_conditions)}/{conditions}', flush=True)
    if not valid_conditions:
        raise RuntimeError(f'{task.value}: no successful nominal conditions')

    details = []
    results = []
    for index, config in enumerate(configs, 1):
        cfg = config_dict(config)
        activated = 0
        failures = 0
        controller_failures = 0
        reasons = defaultdict(int)
        for condition in valid_conditions:
            result = run_episode(task=task, seed=seed,
                                 initial_condition_id=condition,
                                 failure=config, spread=spread)
            injection = result.diagnostics['failure'] or {}
            active = bool(injection.get('activated'))
            failed = not result.task_succeeded
            activated += int(active)
            controller_failures += int(failed)
            failures += int(active and failed)
            reason = result.diagnostics['controller_reason']
            reasons[reason or '(none)'] += 1
            details.append({
                'task': task.value, **cfg,
                'condition_id': condition,
                'nominal_success': True,
                'activated': active,
                'task_failed': failed,
                'valid_injected_failure': active and failed,
                'controller_reason': reason,
                'activation_steps': injection.get('activation_steps', 0),
                'first_activation_s': injection.get('first_activation_s'),
                'duration_s': result.diagnostics['duration_s'],
            })
        n = len(valid_conditions)
        row = {
            'task': task.value, **cfg,
            'tested': n,
            'activated': activated,
            'injected_failures': failures,
            'activation_rate': activated / n,
            'failure_rate': failures / n,
            'failure_rate_given_activation': failures / activated if activated else 0.0,
            'meets_threshold': failures / n >= threshold,
            'failure_reasons': json.dumps(dict(reasons), sort_keys=True),
        }
        results.append(row)
        print(f'[{task.value} {index}/{len(configs)}] '
              f'{cfg["subtask"]} / {cfg["mode"]} '
              f'mag={cfg["magnitude"]:g} axis={cfg["axis"]}: '
              f'activated={activated}/{n}, failed={failures}/{n} '
              f'({failures / n:.0%})', flush=True)

    def write_csv(path, rows):
        if not rows:
            return
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output / 'calibration_results.csv', results)
    write_csv(output / 'calibration_episodes.csv', details)

    # Select the smallest effective magnitude for each (subtask, mode, axis).
    # If none meet the threshold, report the best observed candidate but mark it unreliable.
    groups = defaultdict(list)
    for row in results:
        groups[(row['subtask'], row['mode'], row['axis'])].append(row)

    recommendations = []
    for (subtask, mode, axis), rows in sorted(groups.items()):
        eligible = [r for r in rows if r['meets_threshold']]
        if eligible:
            selected = min(eligible, key=lambda r: r['magnitude'])
        else:
            selected = sorted(rows, key=lambda r: (-r['failure_rate'],
                                                   r['magnitude']))[0]
        recommendations.append({
            'task': task.value, 'subtask': subtask, 'mode': mode,
            'axis': axis, 'magnitude': selected['magnitude'],
            'delay_s': selected['delay_s'],
            'failure_rate': selected['failure_rate'],
            'activation_rate': selected['activation_rate'],
            'reliable': selected['meets_threshold'],
            'tested': selected['tested'],
        })

    write_csv(output / 'recommended_configs.csv', recommendations)
    summary = {
        'task': task.value,
        'seed': seed,
        'conditions_requested': conditions,
        'valid_nominal_conditions': len(valid_conditions),
        'threshold': threshold,
        'configurations_tested': len(results),
        'reliable_configurations': sum(r['meets_threshold'] for r in results),
        'recommended_configurations': len(recommendations),
        'reliable_recommendations': sum(r['reliable'] for r in recommendations),
        'output': str(output),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['pickup', 'transport', 'all'], default='all')
    parser.add_argument('--conditions', type=int, default=10)
    parser.add_argument('--seed', type=int, default=723)
    parser.add_argument('--spread', type=float, default=0.015)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.conditions < 1 or args.seed < 0 or args.spread < 0 or not (0 < args.threshold <= 1):
        parser.error('conditions must be positive, seed/spread nonnegative, threshold in (0,1]')
    root = args.output or (ROOT / 'outputs' / 'calibration' /
           f'batch_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}')
    tasks = ([Task.PICK_UP_OBJECT, Task.TRANSPORT_OBJECT] if args.task == 'all'
             else [Task.PICK_UP_OBJECT if args.task == 'pickup'
                   else Task.TRANSPORT_OBJECT])
    for task in tasks:
        run_calibration(task, args.conditions, args.seed, args.spread,
                        args.threshold, root / task.value, args.verbose)
    print(f'Calibration output: {root}', flush=True)


if __name__ == '__main__':
    main()

"""Exercise real generator retry/I/O behavior with synthetic episode results.

Only the runner dependency is replaced when loading the generator. These tests
do not import MuJoCo, load robot assets, or invoke the physics simulator.
"""

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv
from telemetry_reasoning.schemas import TASK_SUBTASKS, Task
from telemetry_reasoning.sim.failures import FailureConfig


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def generator(monkeypatch):
    """Load dataset.py without importing its optional real simulation backend."""
    runner = ModuleType('telemetry_reasoning.sim.runner')
    runner.ROOT = ROOT

    def unexpected_episode(**kwargs):
        raise AssertionError('Test must install fake episodes before generation')

    runner.run_episode = unexpected_episode
    spec = importlib.util.spec_from_file_location(
        'telemetry_reasoning.sim._dataset_retry_tests',
        ROOT / 'src/telemetry_reasoning/sim/dataset.py',
    )
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, 'telemetry_reasoning.sim.runner', runner)
        spec.loader.exec_module(module)
    return module


class FakeEpisodes:
    """Reject the first N injections, then emit complete synthetic pairs."""

    def __init__(self, rejection=None, reject_count=0, nominal_problem=None):
        self.rejection = rejection
        self.reject_count = reject_count
        self.nominal_problem = nominal_problem
        self.calls = []
        self.injection_count = 0
        self.template = read_telemetry_csv(ROOT / 'data/sample/success_000.csv')[0]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        injected = 'failure' in kwargs
        if injected:
            self.injection_count += 1
        rejection = self.rejection if injected and self.injection_count <= self.reject_count else None
        stages = TASK_SUBTASKS[kwargs['task']]
        rows = [self.template.model_copy(update={
            'timestamp': index * 0.02,
            'subtask': stage,
            # Make any accidentally persisted rejected attempt observable.
            'gripper_force': float(self.injection_count if injected else 0),
        }) for index, stage in enumerate(stages)]
        if rejection == 'no_canonical_telemetry':
            rows = []
        succeeded = not injected or rejection == 'injection_did_not_fail'
        if not injected and self.nominal_problem == 'failed':
            succeeded = False
        complete = bool(rows) and not (not injected and self.nominal_problem == 'incomplete')
        diagnostics = {
            'task': kwargs['task'].value,
            'seed': kwargs['seed'],
            'initial_condition_id': kwargs['initial_condition_id'],
            'initial_offset_xy': [0.0, 0.0],
            'complete_subtasks': complete,
            'observed_subtasks': [s.value for s in stages] if rows else [],
            'controller_outcome': 'success' if succeeded else 'failed',
            'failure': {'activated': rejection != 'not_activated'} if injected else None,
            'synthetic_injection_attempt': self.injection_count if injected else None,
        }
        return SimpleNamespace(rows=rows, task_succeeded=succeeded, diagnostics=diagnostics)


def run_generator(generator, monkeypatch, output, episodes, task, max_attempts):
    monkeypatch.setattr(generator, 'run_episode', episodes)
    calibrated = {task: {
        mode: [FailureConfig(mode=mode, subtask=TASK_SUBTASKS[task][0], magnitude=0.05)]
        for mode in generator.DATASET_MODES
    }}
    generator.generate(task=task, pairs=4, seed=723, max_attempts=max_attempts,
                       output=output, spread=0.015, calibrated=calibrated)


@pytest.mark.parametrize('task', list(Task))
@pytest.mark.parametrize('reason', [
    'not_activated', 'injection_did_not_fail', 'no_canonical_telemetry',
])
def test_retries_preserve_slot_and_write_only_accepted_pairs(
    generator, monkeypatch, tmp_path, task, reason,
):
    episodes = FakeEpisodes(reason, reject_count=2)
    run_generator(generator, monkeypatch, tmp_path, episodes, task, max_attempts=6)

    nominal = [call for call in episodes.calls if 'failure' not in call]
    injected = [call for call in episodes.calls if 'failure' in call]
    assert len(nominal) == len(injected) == 6
    assert injected[0] == injected[1] == injected[2]
    for success, failure in zip(nominal, injected):
        assert success == {key: value for key, value in failure.items() if key != 'failure'}
        assert success['seed'] == 723 and success['spread'] == 0.015
    assert Counter(call['initial_condition_id'] for call in injected[2:]) == Counter(range(4))
    assert Counter(call['failure'].mode for call in injected[2:]) == Counter(generator.DATASET_MODES)

    metadata = read_metadata_csv(tmp_path / 'metadata.csv')
    diagnostics = [json.loads(line) for line in (tmp_path / 'diagnostics.jsonl').read_text().splitlines()]
    assert [m.pair_id for m in metadata] == ['000000', '000001', '000002', '000003']
    assert [d['pair_id'] for d in diagnostics] == [m.pair_id for m in metadata]
    assert [d['injected']['synthetic_injection_attempt'] for d in diagnostics] == [3, 4, 5, 6]
    for meta, record, call in zip(metadata, diagnostics, injected[2:]):
        assert meta.task == task and meta.seed == 723
        assert meta.failure_mode == call['failure'].mode
        assert record['initial_condition_id'] == call['initial_condition_id']
        assert record['failure_config']['mode'] == meta.failure_mode.value
        assert record['failure_trace_is_partial'] is False
        failure_rows = read_telemetry_csv(tmp_path / meta.failure_file)
        assert {r.gripper_force for r in failure_rows} == {float(record['injected']['synthetic_injection_attempt'])}
        assert read_telemetry_csv(tmp_path / meta.success_file)
    assert len(list((tmp_path / 'success').glob('*.csv'))) == 4
    assert len(list((tmp_path / 'failure').glob('*.csv'))) == 4
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['accepted_pairs'] == 4 and summary['attempts'] == 6
    assert summary['rejected'] == {reason: 2}
    assert summary['condition_coverage_valid'] and summary['mode_balance_valid']


@pytest.mark.parametrize('reason', [
    'not_activated', 'injection_did_not_fail', 'no_canonical_telemetry',
])
def test_exhausted_retries_fail_without_publishing_rejected_pairs(
    generator, monkeypatch, tmp_path, reason,
):
    episodes = FakeEpisodes(reason, reject_count=3)
    with pytest.raises(RuntimeError, match='generated only 0/4 pairs'):
        run_generator(generator, monkeypatch, tmp_path, episodes, Task.PICK_UP_OBJECT, max_attempts=3)
    assert episodes.injection_count == 3
    assert not (tmp_path / 'metadata.csv').exists()
    assert not (tmp_path / 'diagnostics.jsonl').exists()
    assert not list(tmp_path.glob('*/*.csv'))
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['accepted_pairs'] == 0 and summary['attempts'] == 3
    assert summary['rejected'] == {reason: 3}
    assert not summary['condition_coverage_valid'] and not summary['mode_balance_valid']


def test_attempt_limit_is_shared_across_slots(generator, monkeypatch, tmp_path):
    episodes = FakeEpisodes()
    with pytest.raises(RuntimeError, match='generated only 3/4 pairs'):
        run_generator(generator, monkeypatch, tmp_path, episodes, Task.PICK_UP_OBJECT, max_attempts=3)
    assert episodes.injection_count == 3
    assert len(read_metadata_csv(tmp_path / 'metadata.csv')) == 3
    assert len((tmp_path / 'diagnostics.jsonl').read_text().splitlines()) == 3
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['attempts'] == 3 and summary['accepted_pairs'] == 3
    assert summary['rejected'] == {}
    assert not summary['condition_coverage_valid'] and not summary['mode_balance_valid']


@pytest.mark.parametrize('problem', ['failed', 'incomplete'])
def test_bad_nominal_aborts_without_retrying(generator, monkeypatch, tmp_path, problem):
    episodes = FakeEpisodes(nominal_problem=problem)
    with pytest.raises(RuntimeError, match='nominal episode failed'):
        run_generator(generator, monkeypatch, tmp_path, episodes, Task.PICK_UP_OBJECT, max_attempts=10)
    assert len(episodes.calls) == 1 and episodes.injection_count == 0
    assert not (tmp_path / 'metadata.csv').exists()
    assert not (tmp_path / 'diagnostics.jsonl').exists()
    assert not list(tmp_path.glob('*/*.csv'))

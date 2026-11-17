#!/usr/bin/env python3
"""Read-only audit of paired MuJoCo telemetry. Python standard library only.
Usage: python audit_robot_dataset.py /path/to/outputs/dataset/batch_...
"""
import argparse, csv, json, math, collections
from pathlib import Path

MODES = {'position_error','orientation_error','inadequate_grip_strength','early_gripper_release'}
TASKS = {'pick_up_object','transport_object'}

def readcsv(p):
    with p.open(newline='', encoding='utf-8-sig') as f:
        r=csv.DictReader(f); return (r.fieldnames or []), list(r)

def check_trace(path, expected_subtasks, problems, examples):
    if not path.is_file():
        problems['missing_trace']+=1; examples.setdefault('missing_trace',str(path)); return None
    try: fields, rows=readcsv(path)
    except Exception as e:
        problems['unreadable_trace']+=1; examples.setdefault('unreadable_trace',f'{path}: {e}'); return None
    if not rows:
        problems['empty_trace']+=1; examples.setdefault('empty_trace',str(path)); return None
    if 'timestamp' not in fields or 'subtask' not in fields:
        problems['missing_core_columns']+=1; examples.setdefault('missing_core_columns',f'{path}: {fields}'); return None
    timestamps=[]; invalid=0
    for r in rows:
        try:
            t=float(r['timestamp']); assert math.isfinite(t); timestamps.append(t)
        except Exception: invalid+=1
    if invalid:
        problems['bad_timestamp']+=1; examples.setdefault('bad_timestamp',str(path))
    elif len(timestamps)>1:
        deltas=[b-a for a,b in zip(timestamps,timestamps[1:])]
        if any(d<=0 for d in deltas):
            problems['nonincreasing_time']+=1; examples.setdefault('nonincreasing_time',str(path))
        if any(abs(d-0.02)>0.002 for d in deltas):
            problems['non_50hz_sampling']+=1; examples.setdefault('non_50hz_sampling',f'{path}: sample deltas {deltas[:8]}')
    missing=sum(1 for r in rows for v in r.values() if v is None or not str(v).strip())
    if missing:
        problems['empty_cells_in_trace']+=1; examples.setdefault('empty_cells_in_trace',f'{path}: {missing} empty cells')
    subtasks=[r['subtask'] for r in rows]
    segments=[s for i,s in enumerate(subtasks) if i==0 or s!=subtasks[i-1]]
    if any(s not in expected_subtasks for s in segments):
        problems['unexpected_subtask']+=1; examples.setdefault('unexpected_subtask',f'{path}: {segments}')
    if len(set(segments))!=len(segments):
        problems['repeated_subtask_segment']+=1; examples.setdefault('repeated_subtask_segment',f'{path}: {segments}')
    return {'rows':len(rows),'duration_s':round(timestamps[-1]-timestamps[0],3) if timestamps else None,
            'columns':fields,'segments':segments,'first':rows[0]}

def main():
    p=argparse.ArgumentParser(); p.add_argument('batch',type=Path); p.add_argument('--report',type=Path,default=Path('dataset_audit_report.json')); args=p.parse_args()
    root=args.batch.expanduser().resolve(); report={'batch':str(root),'tasks':{},'issues':{},'examples':{},'limitations':[
        'Controller-reported task failure is not proof that the injected failure mode caused the physical failure.',
        'Initial-condition IDs cannot establish physical distinctness without inspecting runner.py or initial states.',
        'Telemetry field semantics and physical plausibility require schema and simulator-specific checks.'
    ]}
    problems=collections.Counter(); examples={}
    for task in sorted(TASKS):
        folder=root/task; md=folder/'metadata.csv'; diag=folder/'diagnostics.jsonl'; summary=folder/'summary.json'
        tr={'metadata_rows':0,'success_files':0,'failure_files':0,'mode_counts':{},'condition_counts':{},'trace_rows':{},'durations_seconds':{},'columns':{},'summary':None}
        report['tasks'][task]=tr
        if not md.is_file(): problems['missing_metadata']+=1; continue
        try: cols, pairs=readcsv(md)
        except Exception as e: problems['bad_metadata']+=1; examples['bad_metadata']=str(e); continue
        tr['metadata_rows']=len(pairs)
        if summary.exists():
            try: tr['summary']=json.loads(summary.read_text())
            except Exception as e: problems['bad_summary']+=1; examples['bad_summary']=str(e)
        diagnostics={}
        if diag.exists():
            for line in diag.read_text().splitlines():
                if not line.strip(): continue
                try:
                    d=json.loads(line); key=str(d['pair_id'])
                    if key in diagnostics: problems['duplicate_diagnostic_id']+=1
                    diagnostics[key]=d
                except Exception as e: problems['bad_diagnostic_line']+=1; examples.setdefault('bad_diagnostic_line',str(e))
        else: problems['missing_diagnostics']+=1
        if len(diagnostics)!=len(pairs): problems['diagnostic_count_mismatch']+=1
        ids=set(); modes=collections.Counter(); conditions=collections.Counter(); trace_counts={'success':0,'failure':0}; lengths=collections.defaultdict(list); durations=collections.defaultdict(list); colsets=collections.defaultdict(set)
        expected=['approach','align','grasp','lift'] if task=='pick_up_object' else ['maintain_grasp','move_along_path','reach_target']
        for pair in pairs:
            pid=str(pair.get('pair_id',''))
            if pid in ids: problems['duplicate_pair_id']+=1
            ids.add(pid)
            if pair.get('task')!=task: problems['metadata_task_mismatch']+=1
            mode=pair.get('failure_mode',''); modes[mode]+=1
            if mode not in MODES: problems['unexpected_mode']+=1
            d=diagnostics.get(pid)
            if d is None: problems['missing_pair_diagnostics']+=1
            else:
                condition=d.get('initial_condition_id')
                conditions[str(condition)]+=1
                cfg=d.get('failure_config',{})
                if cfg.get('mode')!=mode: problems['diagnostic_mode_mismatch']+=1
                if cfg.get('subtask')!=pair.get('failed_subtask'): problems['diagnostic_subtask_mismatch']+=1
                if d.get('nominal',{}).get('task_succeeded') is False: problems['nominal_not_successful']+=1
                if d.get('injected',{}).get('task_succeeded') is True: problems['injected_not_failed']+=1
                if d.get('injected',{}).get('failure',{}).get('activated') is False: problems['injection_not_activated']+=1
            firsts={}
            for kind in ('success','failure'):
                rel=pair.get(f'{kind}_file','')
                path=(folder/rel).resolve()
                if not path.is_relative_to(folder.resolve()):
                    problems['unsafe_metadata_path']+=1; continue
                t=check_trace(path,expected,problems,examples)
                if t:
                    trace_counts[kind]+=1; lengths[kind].append(t['rows']); durations[kind].append(t['duration_s']); colsets[kind].add(tuple(t['columns'])); firsts[kind]=t['first']
            if len(firsts)==2:
                common=set(firsts['success'])&set(firsts['failure'])
                if any(firsts['success'][c]!=firsts['failure'][c] for c in common if c not in ('timestamp','subtask')):
                    problems['different_first_logged_state']+=1; examples.setdefault('different_first_logged_state',f'{task}/{pid} (may be expected if failure starts immediately)')
        for kind in ('success','failure'):
            files=list((folder/kind).glob('*.csv')) if (folder/kind).is_dir() else []
            if len(files)!=trace_counts[kind]: problems[f'{kind}_file_count_mismatch']+=1
            tr[f'{kind}_files']=len(files)
            vals=lengths[kind]; tr['trace_rows'][kind]={'min':min(vals),'max':max(vals),'mean':round(sum(vals)/len(vals),1)} if vals else None
            vals=durations[kind]; tr['durations_seconds'][kind]={'min':min(vals),'max':max(vals),'mean':round(sum(vals)/len(vals),2)} if vals else None
            tr['columns'][kind]=[list(s) for s in colsets[kind]]
            if len(colsets[kind])>1: problems['inconsistent_columns']+=1
        tr['mode_counts']=dict(modes); tr['condition_counts']=dict(conditions)
        if len(pairs)!=500: problems['not_500_pairs']+=1
        if any(modes[m]!=125 for m in MODES): problems['unbalanced_modes']+=1
        if len(conditions)!=50 or any(conditions[str(i)]!=10 for i in range(50)):
            problems['unbalanced_conditions']+=1
        if tr['summary']:
            if tr['summary'].get('accepted_pairs')!=len(pairs): problems['summary_metadata_disagreement']+=1
    from telemetry_reasoning.sim.validate_dataset import validate_dataset
    report['acceptance'] = validate_dataset(root)
    report['issues']=dict(problems); report['examples']=examples
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Report:',args.report.resolve())
    print('Issues:',dict(problems))
    for t,d in report['tasks'].items(): print(t, 'pairs:',d['metadata_rows'],'success files:',d['success_files'],'failure files:',d['failure_files'],'modes:',d['mode_counts'],'conditions:',len(d['condition_counts']))
    acceptance = report['acceptance']
    print('Parser-ready pairs:', acceptance['accepted_pairs'], 'quarantined:', acceptance['quarantined_pairs'])
    if problems or acceptance['fatal_errors'] or acceptance['quarantined_pairs']:
        raise SystemExit(1)

if __name__=='__main__': main()

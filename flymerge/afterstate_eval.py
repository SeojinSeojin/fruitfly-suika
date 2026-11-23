"""Tune value weights on development seeds and evaluate on fresh paired seeds."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import mean, median
import time

from .policy import np
from .afterstate_data import DEV, ROOT, SHOWCASE, TEST, candidate_boards
from .afterstate_model import ACTIONS, restore_scorer
from .planner_experiment import environment, load, overflow
from .replay import json_file
from .verify import unpack

GRID = ((.5,70.),(1.,100.),(2.,160.),(4.,240.))
METHODS = {'A':'hybrid','B':'planning','R':'robust','C':'compact','D':'flywire'}
CAP = 450


def load_models():
    graph = json.loads(Path('artifacts/graph.json').read_text())
    return load(), {kind:restore_scorer(ROOT/f'{kind}.pt',graph) for kind in ('compact','flywire')}


def run_episode(case, policy, models):
    method, profile, seed, weights = case
    env = environment(profile,cap=CAP)
    obs,_ = env.reset(seed=seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed,7951]))
    strategy = METHODS[method]
    model = models.get(strategy)
    if model:
        model.decide(obs,*weights)
    else:
        policy.decide(obs,np.random.default_rng(0),strategy,trace=False)
    timings = []
    for _ in range(CAP):
        start = time.perf_counter_ns()
        if model:
            action,_ = model.decide(obs,*weights)
        else:
            action,_ = policy.decide(obs,rng,strategy,trace=False)
        timings.append((time.perf_counter_ns()-start)/1e6)
        obs,_,terminated,truncated,info = env.step(action)
        ended = terminated or truncated
        if env.drops == 300 or (ended and env.drops < 300):
            common = dict(**info,overflow=overflow(env))
        if ended:
            break
    return dict(method=method,profile=profile,seed=seed,weights=weights,common=common,
                extended=dict(**info,overflow=overflow(env)),timings=timings)


def evaluate_group(cases):
    policy,models = load_models()
    return [run_episode(case,policy,models) for case in cases]


def run_sets(groups, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [row for group in pool.map(evaluate_group,groups) for row in group]


def evaluate(phase, workers):
    started = time.perf_counter()
    if phase == 'develop':
        groups = [[(m,'joint',seed,weights) for m in ('C','D') for weights in GRID] for seed in DEV]
        rows = run_sets(groups,workers)
        results = {method:[dict(weights=weights,mean=mean(x['common']['score'] for x in rows
                              if x['method']==method and x['weights']==weights)) for weights in GRID]
                   for method in ('C','D')}
        selected = {method:max(values,key=lambda row:row['mean'])['weights'] for method,values in results.items()}
        for r in rows:
            r.pop('timings')
        json_file(ROOT/'development.json',dict(seeds=DEV,selected=selected,results=results,rows=rows,
                                              seconds=time.perf_counter()-started))
        print(results, selected,flush=True)
        return
    selected = json.loads((ROOT/'development.json').read_text())['selected']
    groups = [[(m,profile,seed,selected.get(m)) for m in METHODS] for seed in TEST
              for profile in ('clean','joint')]
    rows = run_sets(groups,workers)
    summary = []
    for p in ('clean','joint'):
        for horizon in ('common','extended'):
            baseline = {r['seed']:r[horizon]['score'] for r in rows if r['method']=='A' and r['profile']==p}
            for m in METHODS:
                group = [r for r in rows if r['method']==m and r['profile']==p]
                scores = [r[horizon]['score'] for r in group]
                delta = [r[horizon]['score']-baseline[r['seed']] for r in group]
                latency = [item for r in group for item in r['timings'][:r[horizon]['length']]]
                summary.append(dict(method=m,profile=p,horizon=horizon,mean=mean(scores),median=median(scores),
                                    above_3000=sum(x>3000 for x in scores),paired_delta=mean(delta),
                                    wins=sum(x>0 for x in delta),ties=delta.count(0),losses=sum(x<0 for x in delta),
                                    mean_highest=mean(r[horizon]['highest'] for r in group),
                                    overflows=sum(r[horizon]['overflow'] for r in group),
                                    alive=[r['seed'] for r in group if not r[horizon]['overflow']],
                                    latency_ms=np.percentile(latency,[50,95]).tolist()))
    joint = [r for r in summary if r['profile']=='joint' and r['horizon']=='common']
    best = max((r for r in joint if r['method'] in ('C','D')),key=lambda r:r['mean'])
    leader = max(joint,key=lambda r:r['mean'])
    # The deployment candidate must beat every comparator on joint mean.
    candidate = best['method'] if best is leader else None
    for r in rows:
        r.pop('timings')
    json_file(ROOT/'evaluation.json',dict(seeds=TEST,showcase=SHOWCASE,weights=selected,comparators=METHODS,
                                          best_learned=best['method'],candidate=candidate,
                                          summary=summary,rows=rows,seconds=time.perf_counter()-started))
    print('best learned',best,'leader',leader,flush=True)


def diagnose():
    records = []
    for capture in (Path('artifacts/transfer/demo-game-over/website.js'),
                    Path('artifacts/planner/website/website.js')):
        for d in unpack(capture)['decisions']:
            if d['score'] is None or d['score_before'] is None:
                continue
            obs = np.array(d['observation'],dtype=np.float32)
            action = d['action']
            _,scores,_ = candidate_boards(obs,np.array([action],dtype=np.int64))
            active = obs[:-2].reshape(-1,3)
            active = active[active[:,0]>=0]
            records.append(dict(capture=capture.parent.name,drop=d['drop'],action=action,
                                edge=action in (0,14),tall=bool(len(active) and active[:,1].min()<.35),
                                predicted=int(scores[0]),observed=d['score']-d['score_before']))
    groups = {}
    for field in ('edge','tall'):
        for value in (False,True):
            matched = [r for r in records if r[field] == value]
            groups[f'{field}={value}'] = dict(n=len(matched),counts=dict(Counter(
                f"{int(r['predicted']>0)}{int(r['observed']>0)}" for r in matched)),
                mean_predicted=mean(r['predicted'] for r in matched),mean_observed=mean(r['observed'] for r in matched))
    json_file(ROOT/'website-diagnostics.json',dict(rows=records,groups=groups,
              confusion_key='predicted positive bit followed by observed positive bit'))
    print(groups,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['develop','evaluate','diagnose'])
    parser.add_argument('--workers',type=int,default=12)
    args = parser.parse_args()
    if args.phase == 'diagnose':
        diagnose()
    else:
        evaluate(args.phase,args.workers)

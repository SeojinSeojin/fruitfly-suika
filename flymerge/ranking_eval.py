"""Paired fresh-seed evaluation with the robust planner as the reference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import mean, median
import time

from .afterstate_eval import load_models
from .afterstate_data import candidate_boards
from .planner_experiment import environment, overflow
from .policy import np
from .ranker import load_ranker
from .ranking_data import ROOT, TEST, SHOWCASE
from .replay import json_file
from .verify import unpack

METHODS = ('robust','old-compact','compact','flywire')


def components():
    policy, old = load_models()
    graph = json.loads(Path('artifacts/graph.json').read_text())
    rankers = {m:load_ranker(m,graph) for m in METHODS[2:]}
    return policy,old,rankers


def select(method, observation, policy,old,rankers,nuisance,trace=False):
    rng = np.random.default_rng(nuisance)
    if method=='robust':
        return policy.decide(observation,rng,'robust',trace=trace)
    if method=='old-compact':
        return old['compact'].decide(observation,.5,70.)
    return rankers[method].choose(observation,policy,old,nuisance,trace=trace)


def evaluate_one(job):
    seed,profile = job
    policy,old,rankers = components()
    results = []
    for method in METHODS:
        env = environment(profile,cap=450)
        obs,_ = env.reset(seed=seed)
        select(method,obs,policy,old,rankers,seed)
        actions,times = [],[]
        for drop in range(1,451):
            time0 = time.perf_counter_ns()
            action,_ = select(method,obs,policy,old,rankers,seed+drop)
            times.append((time.perf_counter_ns()-time0)/1e6)
            actions.append(action)
            obs,_,terminal,truncated,info = env.step(action)
            ended = terminal or truncated
            if drop == 300 or (drop < 300 and ended):
                common = dict(**info,overflow=overflow(env),rightmost=actions.count(14))
            if ended:
                break
        results.append(dict(method=method,seed=seed,profile=profile,common=common,
                            extended=dict(**info,overflow=overflow(env),rightmost=actions.count(14)),times=times))
    return results


def main(workers):
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for batch in pool.map(evaluate_one,[(s,p) for s in TEST for p in ('clean','joint')]) for r in batch]
    summary = []
    for profile in ('clean','joint'):
        for horizon in ('common','extended'):
            baseline = {r['seed']:r[horizon]['score'] for r in rows
                        if r['profile']==profile and r['method']=='robust'}
            for method in METHODS:
                group = [r for r in rows if r['profile']==profile and r['method']==method]
                score = [r[horizon]['score'] for r in group]
                delta = [r[horizon]['score']-baseline[r['seed']] for r in group]
                ms = [t for r in group for t in r['times'][:r[horizon]['length']]]
                summary.append(dict(method=method,profile=profile,horizon=horizon,
                                    mean=mean(score),median=median(score),above_3000=sum(x>3000 for x in score),
                                    mean_delta=mean(delta),wins=sum(x>0 for x in delta),ties=delta.count(0),
                                    losses=sum(x<0 for x in delta),mean_tier=mean(r[horizon]['highest'] for r in group),
                                    overflow=sum(r[horizon]['overflow'] for r in group),
                                    cap_seeds=[r['seed'] for r in group if not r[horizon]['overflow']],
                                    rightmost=sum(r[horizon]['rightmost'] for r in group)/sum(r[horizon]['length'] for r in group),
                                    latency_ms=np.percentile(ms,[50,95]).tolist()))
    joint = [r for r in summary if r['horizon']=='common' and r['profile']=='joint' and r['method'] in METHODS[2:]]
    best = max(joint,key=lambda r:r['mean'])
    candidate = best['method'] if best['mean_delta']>0 and best['wins']>len(TEST)/2 else None
    for row in rows:
        row.pop('times')
    json_file(ROOT/'evaluation.json',dict(held_out=TEST,showcase=SHOWCASE,best_learned=best['method'],
              candidate=candidate,summary=summary,episodes=rows,workers=workers,seconds=time.perf_counter()-start))
    print(best,candidate,flush=True)


def website_diagnostic():
    rows = []
    for path in ('artifacts/planner/website/website.js','artifacts/transfer/demo-game-over/website.js'):
        for r in unpack(Path(path))['decisions']:
            if r['score'] is None or r['score_before'] is None:
                continue
            obs = np.array(r['observation'],dtype=np.float32)
            _, points,_ = candidate_boards(obs,np.array([r['action']]))
            y = obs[1:-2:3]
            rows.append(dict(capture=path,drop=r['drop'],edge=r['action'] in (0,14),
                             crowded=int((y>=0).sum())>=15,
                             tall=bool(((y>=0)&(y<.35)).any()),predicted=float(points[0]),
                             actual=r['score']-r['score_before']))
    groups = {}
    for field in ('edge','crowded','tall'):
        for v in (False,True):
            relevant = [r for r in rows if r[field]==v and r['predicted']>0]
            groups[f'{field}={v}'] = dict(predicted_positive=len(relevant),
                                         observed_positive=sum(r['actual']>0 for r in relevant))
    json_file(ROOT/'website-diagnostics.json',dict(groups=groups,drops=rows))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--diagnose',action='store_true')
    args=parser.parse_args()
    website_diagnostic() if args.diagnose else main(args.workers)

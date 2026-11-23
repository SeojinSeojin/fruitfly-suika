"""Select a rollout variant on development seeds, then run paired held-out games."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import mean
import time

from .afterstate_eval import load_models
from .planner_experiment import environment, overflow
from .policy import np
from .replay import json_file
from .rollout_teacher import paired_futures, score_actions, shortlist, winner
from .rollout_transfer import choose

ROOT = Path('artifacts/rollout-transfer')
DEV = tuple(range(20271301,20271307))
HELD_OUT = tuple(range(20271401,20271425))
EXACT = HELD_OUT[:6]
SHOWCASE = HELD_OUT[0]
BASELINES = ('robust','fixed','adaptive')


def rng_for(seed,drop,stream=0):
    return np.random.default_rng(np.random.SeedSequence([seed,drop,16607,stream]))


def decision(env,obs,policy,models,method,seed,drop,trace=False):
    nuisance = seed+drop
    if method == 'robust':
        return policy.decide(obs,np.random.default_rng(nuisance),'robust',trace=trace)
    rng = rng_for(seed,drop)
    if method == 'exact':
        baseline,actions = shortlist(policy,models,obs,nuisance)
        candidates = score_actions(env,actions,paired_futures(rng))
        return winner(candidates),dict(baseline=baseline,actions=actions,utility=[r['value'] for r in candidates])
    action, details = choose(obs,policy,models,nuisance,rng,method)
    if trace:
        # Search uses simulated circles. This is the frozen checkpoint's observation trace.
        _, diagnostic = policy.decide(obs,None,'greedy',trace=True)
        details.update(diagnostic)
    return action,details


def play(job):
    seed,profile,methods,phase = job
    policy,models = load_models()
    outputs,audit = [],[]
    for method in methods:
        env = environment(profile,cap=450)
        obs,_ = env.reset(seed=seed)
        decision(env,obs,policy,models,method,seed,0)
        times, decisions, budget = [],[],[]
        for drop in range(1,451):
            start = time.perf_counter_ns()
            action,details = decision(env,obs,policy,models,method,seed,drop)
            times.append((time.perf_counter_ns()-start)/1e6)
            if details and 'tails' in details:
                budget.append(len(details['actions'])*details['tails'])
            if phase=='develop' and method=='robust' and drop in (6,50,105,170,240,300):
                for variant in BASELINES[1:]:
                    a,_ = choose(obs,policy,models,seed+drop,rng_for(seed,drop,1),variant)
                    b,_ = choose(obs,policy,models,seed+drop,rng_for(seed,drop,2),variant)
                    audit.append(dict(seed=seed,profile=profile,drop=drop,variant=variant,a=a,b=b))
            obs,gain,terminated,truncated,info = env.step(action)
            decisions.append(dict(action=action,gain=gain,score=info['score']))
            ended = terminated or truncated
            if drop==300 or (drop<300 and ended):
                common = dict(**info,overflow=overflow(env))
            if ended:
                break
        outputs.append(dict(seed=seed,method=method,profile=profile,common=common,
                            extended=dict(**info,overflow=overflow(env)),times=times,budget=budget,
                            decisions=decisions if seed==SHOWCASE and profile=='joint' else []))
    print(profile,seed,{k['method']:k['extended']['score'] for k in outputs},flush=True)
    return outputs,audit


def summarize(rows,methods):
    summary = []
    for profile in ('clean','joint'):
        for method in methods:
            group = [r for r in rows if (r['profile'],r['method'])==(profile,method)]
            if not group:
                continue
            match = {r['seed']:r for r in rows if r['profile']==profile and r['method']=='robust'}
            for horizon in ('common','extended'):
                base = [match[r['seed']][horizon]['score'] for r in group]
                scores = [r[horizon]['score'] for r in group]
                delta = [x-y for x,y in zip(scores,base)]
                milliseconds = [t for r in group for t in r['times'][:r[horizon]['length']]]
                summary.append(dict(profile=profile,method=method,horizon=horizon,episodes=len(group),
                                    mean=mean(scores),robust_mean=mean(base),delta=mean(delta),
                                    wins=sum(x>0 for x in delta),ties=delta.count(0),losses=sum(x<0 for x in delta),
                                    above_3000=sum(s>3000 for s in scores),
                                    cap_seeds=[r['seed'] for r in group if not r[horizon]['overflow']],
                                    latency_ms=np.percentile(milliseconds,[50,95]).tolist()))
    return summary


def main(phase,workers):
    start = time.perf_counter()
    if phase=='develop':
        seeds, chosen = DEV, None
    else:
        seeds = HELD_OUT
        chosen = json.loads((ROOT/'development.json').read_text())['selected']
    jobs = [(seed,p,BASELINES if phase=='develop' else ('robust',chosen,'exact') if seed in EXACT
             else ('robust',chosen),phase) for p in ('clean','joint') for seed in seeds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(play,jobs))
    rows = [r for group,_ in results for r in group]
    metrics = summarize(rows,BASELINES if phase=='develop' else ('robust',chosen,'exact'))
    if phase=='develop':
        winner = max((r for r in metrics if r['profile']=='joint' and r['horizon']=='extended'
                      and r['method'] in BASELINES[1:]),key=lambda r:r['mean'])
        chosen = winner['method']
    primary = next(r for r in metrics if (r['method'],r['profile'],r['horizon'])==(chosen,'joint','extended'))
    eligible = phase=='evaluate' and primary['delta']>0
    audit = [a for _,group in results for a in group]
    budgets = {}
    for method in ('fixed','adaptive'):
        entries = [v for r in rows if r['method']==method for v in r['budget']]
        if entries:
            budgets[method] = dict(mean=mean(entries),maximum=max(entries))
    for r in rows:
        r.pop('times')
        r.pop('budget')
    json_file(ROOT/('development.json' if phase=='develop' else 'results.json'),
              dict(phase=phase,seeds=seeds,workers=workers,selected=chosen,live_eligible=eligible,
                   summary=metrics,episodes=rows,within_board_repeat=audit,budget=budgets,
                   seconds=time.perf_counter()-start,exact_seeds=EXACT if phase=='evaluate' else []))
    print(primary, 'live eligible:',eligible,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['develop','evaluate'])
    parser.add_argument('--workers',type=int,default=12)
    args = parser.parse_args()
    main(args.phase,args.workers)

"""Fresh paired episodes, resumable per seed, with an exact-state subset."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
from statistics import mean
import time

from .afterstate_eval import load_models
from .planner_experiment import environment,overflow
from .policy import np
from .replay import json_file
from .rollout_eval import decision
from .state_calibration import ROOT
from .state_estimation import local_history
from .state_search import METHODS,decide

SEEDS=tuple(range(20271601,20271625))
CLEAN=SEEDS[:12]
EXACT=SEEDS[:6]
SHOWCASE=SEEDS[0]


def play(job):
    profile,seed = job
    policy,models=load_models()
    config=json.loads((ROOT/'frozen.json').read_text())
    rows=[]
    for method in (*METHODS,'exact') if seed in EXACT else METHODS:
        env=environment(profile,cap=450,capture=True)
        obs,_=env.reset(seed=seed)
        times,show,accepted=[],[],0
        for drop in range(451):
            start=time.perf_counter_ns()
            if method=='exact':
                action,detail=decision(env,obs,policy,models,method,seed,drop)
            else:
                history=local_history(env,obs,seed,max(1,drop)) if method in ('tracking','both') else []
                action,detail=decide(obs,history,policy,models,method,seed,drop,config)
            if drop==0:
                continue
            times.append((time.perf_counter_ns()-start)/1e6)
            accepted += detail.get('velocities_used',0) if detail else 0
            obs,gain,terminated,truncated,info=env.step(action)
            if seed==SHOWCASE and profile=='joint':
                show.append(dict(action=action,gain=gain,score=info['score']))
            end=terminated or truncated
            if drop==300 or (end and drop<300):
                common=dict(**info,overflow=overflow(env))
            if end:
                break
        rows.append(dict(profile=profile,seed=seed,method=method,common=common,
                         extended=dict(**info,overflow=overflow(env)),times=times,show=show,
                         nonzero_velocities=accepted))
    print(profile,seed,{r['method']:r['extended']['score'] for r in rows},flush=True)
    return rows


def main(workers):
    start=time.perf_counter()
    jobs=[(p,s) for p,seeds in (('clean',CLEAN),('joint',SEEDS)) for s in seeds]
    cache = Path('runs/state-episodes')
    cache.mkdir(parents=True,exist_ok=True)
    def path(job):
        return cache/f'{job[0]}-{job[1]}.json'
    pending=[job for job in jobs if not path(job).is_file()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(play,job):job for job in pending}
        for completed in as_completed(futures):
            dest=path(futures[completed])
            temp=dest.with_suffix('.part')
            json_file(temp,completed.result())
            temp.replace(dest)
    rows=[r for job in jobs for r in json.loads(path(job).read_text())]
    summary=[]
    for profile in ('clean','joint'):
        reference={r['seed']:r for r in rows if r['profile']==profile and r['method']=='robust'}
        for method in (*METHODS,'exact'):
            group=[r for r in rows if r['profile']==profile and r['method']==method]
            for horizon in ('common','extended'):
                values=[r[horizon]['score'] for r in group]
                delta=[v-reference[r['seed']][horizon]['score'] for r,v in zip(group,values)]
                timings=[t for r in group for t in r['times'][:r[horizon]['length']]]
                summary.append(dict(profile=profile,method=method,horizon=horizon,episodes=len(group),
                                mean=mean(values),delta=mean(delta),wins=sum(v>0 for v in delta),
                                ties=delta.count(0),losses=sum(v<0 for v in delta),
                                above_3000=sum(v>3000 for v in values),
                                alive_at_cap=[r['seed'] for r in group if not r[horizon]['overflow']],
                                latency_ms=np.percentile(timings,[50,95]).tolist()))
    best=max((r for r in summary if r['profile']=='joint' and r['horizon']=='extended'
              and r['method'] in METHODS[1:]),key=lambda r:r['mean'])
    for row in rows:
        row.pop('times')
    json_file(ROOT/'results.json',dict(seeds=SEEDS,clean_seeds=CLEAN,exact_seeds=EXACT,showcase=SHOWCASE,
              summary=summary,episodes=rows,best=best['method'],live_eligible=best['delta']>0,
              workers=workers,resumed_jobs=len(jobs)-len(pending),seconds=time.perf_counter()-start))
    print(best,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=12)
    main(parser.parse_args().workers)

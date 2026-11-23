"""Matched observation errors and exact-state utility regret."""
import json

from .afterstate_eval import load_models
from .planner_experiment import environment
from .policy import np
from .replay import json_file
from .rollout_eval import rng_for
from .rollout_teacher import PENALTY,clean_branch,shortlist
from .rollout_transfer import futures,reconstruct
from .state_calibration import DEV,ROOT
from .state_estimation import local_history,match
from .state_experiment import EXACT
from .state_search import METHODS,decide


def settled_motion(policy):
    data={}
    for profile in ('clean','joint'):
        samples=[]
        for seed in DEV:
            env=environment(profile,cap=450,capture=True)
            obs,_=env.reset(seed=seed)
            for drop in range(1,301):
                if drop>1 and drop%5==0:
                    before=env.frames[19][0]
                    visible=np.zeros((len(before),5))
                    visible[:,[0,1,4]]=before
                    true=env.body[:env.n]
                    pairs=match(true,visible,30.,0.)
                    hits=np.flatnonzero(pairs>=0)
                    samples.extend(np.linalg.norm(true[hits,:2]-visible[pairs[hits],:2],axis=1))
                action,_=policy.decide(obs,np.random.default_rng(seed+drop),'robust',trace=False)
                obs,_,term,trunc,_=env.step(action)
                if term or trunc:
                    break
        data[profile]=dict(tracks=len(samples),median_p95_px=np.percentile(samples,[50,95]).tolist(),
                           fraction_at_most_2px=float((np.asarray(samples)<=2).mean()))
    return data


def forecasts(policy,models,config):
    rows=[]
    for profile in ('clean','joint'):
        for seed in EXACT:
            env=environment(profile,cap=450,capture=True)
            obs,_=env.reset(seed=seed)
            for drop in range(1,301):
                if drop in (15,80,160,240,295):
                    history=local_history(env,obs,seed,drop)
                    baseline,actions=shortlist(policy,models,obs,seed+drop)
                    _,_,_,preview=reconstruct(obs)
                    tails=futures(preview,rng_for(seed,drop),3)
                    exact=[]
                    for action in actions:
                        values=[]
                        for next_tier,tail in tails:
                            a,b,ended,_=clean_branch(env.body,env.n,env.current,next_tier,env.physics,action,tail)
                            values.append(a+b-PENALTY*ended)
                        exact.append(np.mean(values))
                    for method in METHODS:
                        action=baseline if method=='robust' else decide(obs,history,policy,models,method,seed,drop,config)[0]
                        regret=max(exact)-exact[actions.index(action)]
                        rows.append(dict(profile=profile,seed=seed,drop=drop,method=method,
                                         action=action,exact_action=actions[int(np.lexsort((actions,-np.array(exact)))[0])],
                                         regret=float(regret)))
                action,_=policy.decide(obs,np.random.default_rng(seed+drop),'robust',trace=False)
                obs,_,term,trunc,_=env.step(action)
                if term or trunc:
                    break
    summary=[]
    for profile in ('clean','joint'):
        for method in METHODS:
            rows_at=[r for r in rows if r['profile']==profile and r['method']==method]
            summary.append(dict(profile=profile,method=method,boards=len(rows_at),
                regret=float(np.mean([r['regret'] for r in rows_at])),
                exact_action_matches=sum(r['action']==r['exact_action'] for r in rows_at)))
    return rows,summary


if __name__=='__main__':
    policy,models=load_models()
    config=json.loads((ROOT/'frozen.json').read_text())
    motion=settled_motion(policy)
    rows,summary=forecasts(policy,models,config)
    json_file(ROOT/'diagnostics.json',dict(motion=motion,summary=summary,rows=rows))

"""Fixed-size ablation using tracked starts and/or sampled simulator settings."""
import numpy as np

from .rollout_eval import rng_for
from .rollout_teacher import clean_branch,shortlist
from .rollout_transfer import futures,NOMINAL,reconstruct,utilities
from .state_estimation import track

METHODS=('robust','current','tracking','physics','both')


def decide(obs,history,policy,models,method,seed,drop,config,trace=False):
    nuisance=seed+drop
    if method=='robust':
        return policy.decide(obs,np.random.default_rng(nuisance),'robust',trace=trace)
    baseline,actions=shortlist(policy,models,obs,nuisance)
    bank = config['physics'] if method in ('physics','both') else [NOMINAL]*3
    order = rng_for(seed,drop,909).permutation(3)
    state,accepted = track(obs,history,config['tracking_velocity']) if method in ('tracking','both') else (reconstruct(obs),0)
    board,count,current,preview=state
    if current<0:
        return baseline,dict(fallback='unknown current fruit')
    tails=futures(preview,rng_for(seed,drop),3)
    samples=[]
    for action in actions:
        group=[]
        for k,(next_tier,tail) in enumerate(tails):
            physics=tuple(bank[order[k]])
            a,b,end,steps=clean_branch(board,count,current,next_tier,physics,action,tail)
            group.append([float(a),float(b),bool(end),int(steps)])
        samples.append(group)
    value=utilities(samples).mean(1)
    index=int(np.lexsort((actions,-value))[0])
    details=dict(baseline=baseline,actions=actions,utility=value.tolist(),velocities_used=accepted)
    if trace:
        _,diagnostic=policy.decide(obs,None,'greedy')
        result=board[:count].tolist()
        details.update(estimate=result,frames=[dict(seconds=t,detections=np.asarray(f).tolist()) for t,f in history],
                       **diagnostic)
    return actions[index],details

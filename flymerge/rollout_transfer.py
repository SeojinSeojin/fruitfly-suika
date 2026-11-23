"""Twenty-drop rollout search from the detector's observation."""
import numpy as np

from .game import HEIGHT, SLOTS, TRANSFER_WIDTH
from .planning import drop_values
from .rollout_teacher import HORIZON, PENALTY, clean_branch, shortlist

NOMINAL = (.46, .97, .19, .55)
INITIAL_TAILS = 3
MAX_TAILS = 6
BASE_CANDIDATES = 6
MAX_CANDIDATES = 8
NOMINAL_MARGIN = .15
UTILITY_MARGIN = 25.


def reconstruct(obs):
    points = obs[:-2].reshape(-1,3)
    valid = points[(points[:,0]>=0)&(points[:,1]>=0)]
    body = np.zeros((SLOTS+1,5))
    body[:len(valid),:2] = valid[:,:2]*np.array([TRANSFER_WIDTH,HEIGHT])
    body[:len(valid),4] = np.clip(np.round(valid[:,2]*10),0,10)
    return body,len(valid),round(float(obs[-2])*10),round(float(obs[-1])*10)


def futures(preview,rng,count):
    # Draw the hypothetical preview only when the detector's preview is unknown.
    result = []
    for _ in range(count):
        known = preview if preview >= 0 else int(rng.integers(0,5))
        result.append((known,rng.integers(0,5,HORIZON)))
    return result


def outcomes(state,actions,tails):
    body, count, current,_ = state
    samples = []
    for action in actions:
        row = []
        for preview,tail in tails:
            immediate,later,ended,steps = clean_branch(body,count,current,preview,NOMINAL,action,tail)
            row.append((float(immediate),float(later),bool(ended),int(steps)))
        samples.append(row)
    return samples


def utilities(samples):
    return np.asarray([[a+b-PENALTY*end for a,b,end,_ in row] for row in samples])


def choose(obs,policy,models,nuisance,rng,variant):
    baseline, actions = shortlist(policy,models,obs,nuisance)
    state = reconstruct(obs)
    if state[2] < 0:
        return baseline,dict(actions=[baseline],rollouts=[],fallback='unknown held fruit')
    adaptive = variant == 'adaptive'
    if adaptive:
        nominal = drop_values(obs)
        order = np.argsort(-nominal,kind='stable')
        if nominal[order[0]]-nominal[order[1]] <= NOMINAL_MARGIN:
            actions = list(dict.fromkeys([*actions,*map(int,order)]))[:MAX_CANDIDATES]
    tails = futures(state[3],rng,INITIAL_TAILS)
    samples = outcomes(state,actions,tails)
    net = utilities(samples)
    if adaptive:
        order = np.lexsort((actions,-net.mean(1)))
        paired = net[order[0]]-net[order[1]]
        standard_error = paired.std(ddof=1)/np.sqrt(len(paired))
        if paired.mean() <= max(UTILITY_MARGIN,standard_error):
            more = outcomes(state,actions,futures(state[3],rng,MAX_TAILS-INITIAL_TAILS))
            samples = [row+extra for row,extra in zip(samples,more)]
            net = utilities(samples)
    idx = int(np.lexsort((actions,-net.mean(1)))[0])
    return actions[idx],dict(baseline=baseline,actions=actions,rollouts=samples,
                             utility=net.mean(1).tolist(),tails=len(samples[0]))

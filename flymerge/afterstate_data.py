"""Generate paired clean and reconstructed afterstates with censored 20-drop targets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

from numba import njit

from .policy import np
from .game import AIMS, HEIGHT, SLOTS, TRANSFER_RADII, TRANSFER_WIDTH
from .planning import drop_values, settle
from .planner_experiment import environment, load, logp, overflow
from .replay import json_file

HORIZON = 20
ROOT = Path('artifacts/afterstate')
DATA = Path('runs/afterstate.npz')
TRAIN = tuple(range(20270301, 20270401)) + tuple(range(20270401, 20270449))
VALID = tuple(range(20270501, 20270509))
DEV = tuple(range(20270601, 20270609))
TEST = tuple(range(20270701, 20270725))
SHOWCASE = TEST[0]


@njit(cache=True)
def after_observation(body, count, visible_next):
    out = np.full(SLOTS*3+2, -1., dtype=np.float32)
    keys = np.empty(count)
    for i in range(count):
        # A stable iteration order resolves position ties before final packing.
        keys[i] = body[i, 1]
    order = np.argsort(keys, kind='mergesort')
    for i, p in enumerate(order[:SLOTS]):
        out[3*i] = min(1., max(0., body[p,0]/TRANSFER_WIDTH))
        out[3*i+1] = min(1., max(0., body[p,1]/HEIGHT))
        out[3*i+2] = body[p,4]/10
    out[-2] = visible_next
    return out


@njit(cache=True)
def candidate_boards(obs, actions):
    board = np.zeros((SLOTS+1, 5))
    count = 0
    for i in range(SLOTS):
        x, y, tier = obs[3*i:3*i+3]
        if x >= 0 and y >= 0:
            board[count,0], board[count,1] = x*TRANSFER_WIDTH, y*HEIGHT
            board[count,4] = min(10,max(0,round(tier*10)))
            count += 1
    views = np.empty((len(actions), SLOTS*3+2), dtype=np.float32)
    scores = np.zeros(len(actions))
    ends = np.zeros(len(actions),dtype=np.bool_)
    tier = min(10,max(0,round(obs[-2]*10)))
    for i, action in enumerate(actions):
        trial, n, scores[i] = settle(board,count,tier,AIMS[action])
        views[i] = after_observation(trial,n,obs[-1])
        ends[i] = n >= SLOTS
        for j in range(n):
            ends[i] |= trial[j,1]-TRANSFER_RADII[int(trial[j,4])] < 11
    return views,scores,ends


def targets(gains, ended, horizon=HORIZON):
    """Return score, risk, score-known and risk-known for each created board."""
    gains = np.asarray(gains)
    n = len(gains)
    ends = np.minimum(np.arange(n)+1+horizon,n)
    remaining = n-np.arange(n)-1
    prefix = np.concatenate(([0],np.cumsum(gains)))
    outcome = prefix[ends]-prefix[np.arange(n)+1]
    risk = ended & (remaining > 0) & (remaining <= horizon)
    known = (remaining >= horizon) | ended
    # An overflow on the creating move already terminates the candidate.
    if ended:
        known[-1] = False
    return outcome,risk,known,known.copy()


def collect(job):
    seed, profile, method, split = job
    policy = load() if method == 'hybrid' else None
    rng = np.random.default_rng(np.random.SeedSequence([seed, 5203]))
    env = environment(profile,cap=500)
    obs,_ = env.reset(seed=seed)
    clean, sim, moves, gains, sim_gains = [],[],[],[],[]
    while True:
        score = drop_values(obs)
        if policy is not None:
            score = score + .20*logp(policy,obs)
        ordered = np.argsort(-score,kind='stable')
        action = int(rng.choice(ordered[:4])) if rng.random() < .20 else int(ordered[0])
        visible_next = obs[-1]
        board, points, _ = candidate_boards(obs,np.array([action],dtype=np.int64))
        obs, gain, terminated, truncated,_ = env.step(action)
        clean.append(after_observation(env.body,env.n,visible_next))
        sim.append(board[0])
        moves.append(action)
        gains.append(gain)
        sim_gains.append(points[0])
        if terminated or truncated:
            break
    ended = overflow(env)
    score,risk,known_score,known_risk = targets(gains,ended)
    return dict(clean=np.stack(clean),sim=np.stack(sim),action=np.array(moves,np.int8),
                immediate=np.array(gains,np.int16),sim_immediate=np.array(sim_gains,np.int16),
                target=score.astype(np.float32),risk=risk,score_known=known_score,risk_known=known_risk,
                seed=np.full(len(gains),seed,np.int32),split=np.full(len(gains),split,np.int8),
                profile=np.full(len(gains),profile=='joint',np.int8),
                policy=np.full(len(gains),method=='hybrid',np.int8),
                drop=np.arange(1,len(gains)+1,dtype=np.int16))


def main(workers):
    start = time.perf_counter()
    jobs = [(s,p,m,i) for i,seeds in enumerate((TRAIN,VALID)) for s in seeds
            for p in ('clean','joint') for m in ('hybrid','planning')]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        groups = list(pool.map(collect,jobs,chunksize=1))
    arrays = {k:np.concatenate([r[k] for r in groups]) for k in groups[0]}
    DATA.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(DATA,**arrays)
    json_file(ROOT/'dataset.json',dict(horizon=HORIZON,train=TRAIN,validation=VALID,development=DEV,
              test=TEST,showcase=SHOWCASE,trajectories=len(jobs),rows=len(arrays['seed']),
              complete_score=int(arrays['score_known'].sum()),future_overflow=int(arrays['risk'].sum()),
              exploratory_fraction=.20,exploratory_actions=4,cap=500,workers=workers,
              seconds=time.perf_counter()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=12)
    main(parser.parse_args().workers)

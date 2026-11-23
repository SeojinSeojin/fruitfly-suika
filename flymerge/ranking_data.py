"""Generate grouped action labels. Every group shares one true parent board."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

from .afterstate_eval import load_models
from .afterstate_data import after_observation,candidate_boards
from .planner_experiment import environment
from .policy import np
from .replay import json_file
from .rollout_teacher import ROOT, PENALTY, paired_futures, score_actions, settle_exact, shortlist

DATA = Path('runs/ranking.npz')
TRAIN = tuple(range(20271001,20271033))
DEV = tuple(range(20271101,20271109))
TEST = tuple(range(20271201,20271225))
SHOWCASE = TEST[0]


def group(env,obs,policy,models,seed,rng):
    robust, actions = shortlist(policy,models,obs,seed)
    actions = np.array(actions,dtype=np.int64)
    simulated, immediate, terminal = candidate_boards(obs,actions)
    exact, exact_immediate, exact_terminal = [],[],[]
    for action in actions:
        board = env.body.copy()
        count, gain, ended = settle_exact(board,env.n,env.current,int(action),env.physics)
        exact.append(after_observation(board,count,obs[-1]))
        exact_immediate.append(gain)
        exact_terminal.append(ended)
    rollouts = score_actions(env,actions,paired_futures(rng))
    targets = [[v['immediate']+v['later']-PENALTY*v['overflow'] for v in row['rollouts']] for row in rollouts]
    return dict(simulated=simulated,exact=np.stack(exact),sim_features=np.column_stack((immediate,terminal)),
                exact_features=np.column_stack((exact_immediate,exact_terminal)),
                actions=actions,robust=robust,scores=np.array(targets),
                risks=np.array([[x['overflow'] for x in row['rollouts']] for row in rollouts]),
                truncated=sum(x['truncated'] for row in rollouts for x in row['rollouts']))


def build(job):
    seed,profile,behavior,split = job
    policy,models = load_models()
    env = environment(profile,cap=450)
    obs,_ = env.reset(seed=seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed,10813]))
    schedule = (8,35,65,100,135,175,215,255,295,335,375) if split == 0 else (12,75,150,240,320)
    rows = []
    while True:
        drop = env.drops+1
        if drop in schedule:
            r = group(env,obs,policy,models,seed+drop,rng)
            rows.append(dict(**r,seed=seed,profile=profile=='joint',behavior=behavior=='robust',split=split,drop=drop))
        action,_ = policy.decide(obs,np.random.default_rng(seed+drop),behavior,trace=False)
        obs,_,terminal,truncated,_ = env.step(action)
        if terminal or truncated:
            break
    return rows


def main(workers):
    start = time.perf_counter()
    jobs = [(seed,p,b,i) for i,seeds in enumerate((TRAIN,DEV,TEST)) for seed in seeds
            for p in ('clean','joint') for b in (('robust','planning') if i<2 else ('robust',))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [x for batch in pool.map(build,jobs,chunksize=1) for x in batch]
    arrays = {k:np.stack([r[k] for r in rows]) for k in rows[0]}
    assert not arrays['truncated'].any()
    np.savez_compressed(DATA,**arrays)
    json_file(ROOT/'dataset.json',dict(train=TRAIN,development=DEV,independent_test=TEST,
              board_groups=len(rows),split_counts=np.bincount(arrays['split']).tolist(),
              candidates=len(rows)*6,rollouts=len(rows)*18,trajectories=len(jobs),
              workers=workers,seconds=time.perf_counter()-start))
    print(len(rows),'board groups',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=12)
    main(parser.parse_args().workers)

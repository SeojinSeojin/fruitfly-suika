"""Counterfactual first moves with paired, hidden future fruit."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import json
from pathlib import Path
import time
from statistics import mean

from numba import njit

from .afterstate_eval import load_models
from .game import AIMS, HEIGHT, SLOTS, TRANSFER_RADII, TRANSFER_WIDTH, tick
from .planner_experiment import environment, overflow
from .policy import np
from .planning import drop_values
from .replay import json_file

ROOT = Path('artifacts/drop-ranking')
HORIZON = 20
FUTURES = 3
PENALTY = 150.
AUDIT_SEEDS = tuple(range(20270801,20270809))
DEV = tuple(range(20270901,20270913))
SHOWCASE = DEV[0]
AUDIT_DROPS = (5,45,100,170,230,285)


@njit(cache=True)
def continuation(board, count, current, upcoming):
    """Fixed first-contact heuristic using visible circles and preview only."""
    radius = TRANSFER_RADII[current]
    best, selected = -1e9, 0
    for action,aim in enumerate(AIMS):
        x = radius+(TRANSFER_WIDTH-2*radius)*aim
        floor = HEIGHT-radius
        hit = -1
        for i in range(count):
            size = radius+TRANSFER_RADII[int(board[i,4])]
            dx = x-board[i,0]
            chord = size*size-dx*dx
            if chord > 0:
                contact = board[i,1]-np.sqrt(chord)
                if contact < floor:
                    floor,hit = contact,i
        opportunity = 0.
        if hit >= 0 and board[hit,4] == current and current < 10:
            opportunity = 1.4 + .07*current
        preview = .04 if hit >= 0 and board[hit,4] == upcoming and upcoming != current else 0.
        value = opportunity + floor/HEIGHT + preview
        if floor-radius < 11 and not opportunity:
            value -= 5
        if value > best:
            best,selected = value,action
    return selected


@njit(cache=True)
def settle_exact(board, count, fruit, action, physics):
    radius = TRANSFER_RADII[fruit]
    x = radius+(TRANSFER_WIDTH-2*radius)*AIMS[action]
    board[count,0] = x
    board[count,1] = -25.
    board[count,2] = board[count,3] = 0.
    board[count,4] = fruit
    n, gain = count+1,0
    for _ in range(80):
        n, reward = tick(board,n,TRANSFER_WIDTH,TRANSFER_RADII,*physics)
        gain += reward
    ended = n >= SLOTS
    for j in range(n):
        ended |= board[j,1]-TRANSFER_RADII[int(board[j,4])] < 11
    return n,gain,ended


@njit(cache=True)
def clean_branch(body,count,current,upcoming,physics,action,tail):
    board = body.copy()
    n,immediate,ended = settle_exact(board,count,current,action,physics)
    later,steps = 0,0
    for i in range(HORIZON):
        if ended:
            break
        fruit = upcoming if i == 0 else tail[i-1]
        preview = tail[i]
        move = continuation(board,n,fruit,preview)
        n,gain,ended = settle_exact(board,n,fruit,move,physics)
        later += gain
        steps += 1
    return immediate,later,ended,steps


def visible_action(obs):
    fruit = obs[:-2].reshape(-1,3).copy()
    fruit = fruit[fruit[:,0]>=0]
    body = np.zeros((len(fruit),5))
    body[:,:2] = fruit[:,:2]*np.array([TRANSFER_WIDTH,HEIGHT])
    body[:,4] = np.maximum(0,np.round(fruit[:,2]*10))
    return continuation(body,len(body),round(float(obs[-2])*10),round(float(obs[-1])*10))


def noisy_branch(env, action, future_seed, vision_seeds):
    clone = copy.deepcopy(env)
    clone.capture = False
    clone.drop_limit = env.drops+1+HORIZON
    clone.np_random = np.random.default_rng(future_seed)
    clone.vision_rng = np.random.default_rng(vision_seeds[0])
    obs,immediate,_,_,_ = clone.step(action)
    ended = overflow(clone)
    later,steps = 0,0
    for i in range(HORIZON):
        if ended:
            break
        clone.vision_rng = np.random.default_rng(vision_seeds[i+1])
        obs,gain,_,_,_ = clone.step(visible_action(obs))
        ended = overflow(clone)
        later += gain
        steps += 1
    return immediate,later,ended,steps


def paired_futures(rng):
    seeds = rng.integers(0,2**32,FUTURES)
    vision = rng.integers(0,2**32,(FUTURES,HORIZON+1))
    return [(int(seed),np.random.default_rng(seed).integers(0,5,HORIZON),view)
            for seed,view in zip(seeds,vision)]


def score_actions(env, actions, futures):
    rows = []
    for action in actions:
        rollouts = []
        for seed,tail,vision in futures:
            if env.vision_noise:
                immediate,later,ended,steps = noisy_branch(env,action,seed,vision)
            else:
                immediate,later,ended,steps = clean_branch(env.body,env.n,env.current,env.upcoming,
                                                           env.physics,action,tail)
            rollouts.append(dict(immediate=float(immediate),later=float(later),overflow=bool(ended),
                                 subsequent_drops=int(steps),truncated=not ended and steps < HORIZON))
        values = [r['immediate']+r['later']-PENALTY*r['overflow'] for r in rollouts]
        rows.append(dict(action=int(action),value=mean(values),sd=float(np.std(values)),
                         overflow=mean(r['overflow'] for r in rollouts),rollouts=rollouts))
    return rows


def shortlist(policy, models, obs, seed):
    robust,_ = policy.decide(obs,np.random.default_rng(seed),'robust',trace=False)
    compact,_ = models['compact'].decide(obs,.5,70.)
    fly,_ = models['flywire'].decide(obs,.5,70.)
    physics_values = drop_values(obs)
    order = np.argsort(-physics_values,kind='stable')
    return robust,list(dict.fromkeys([robust,compact,fly,*map(int,order)]))[:6]


def teacher(env, policy, models, obs, rng, nuisance):
    baseline, actions = shortlist(policy,models,obs,nuisance)
    tested = score_actions(env,actions,paired_futures(rng))
    return max(tested,key=lambda row:(row['value'],-row['action']))['action'],baseline,tested


def audit_job(job):
    seed,source,profile = job
    policy,models = load_models()
    env = environment(profile,cap=450)
    obs,_ = env.reset(seed=seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed,3901]))
    rows = []
    while True:
        drop = env.drops+1
        nuisance = seed+drop
        if drop in AUDIT_DROPS:
            a,candidates = shortlist(policy,models,obs,nuisance)
            first,second = paired_futures(rng),paired_futures(rng)
            candidate_rows = score_actions(env,candidates,first)
            replicate = score_actions(env,candidates,second)
            all_rows = score_actions(env,range(15),first) if drop in (5,100,230) else None
            rows.append(dict(seed=seed,source=source,profile=profile,drop=drop,fruits=env.n,
                             highest_center=float(env.body[:env.n,1].min()) if env.n else HEIGHT,
                             baseline=a,candidates=candidate_rows,independent=replicate,all_actions=all_rows))
        action,_ = policy.decide(obs,np.random.default_rng(nuisance),source,trace=False)
        obs,_,terminated,truncated,_ = env.step(action)
        if terminated or truncated:
            break
    return rows


def episode_job(seed):
    policy,models = load_models()
    episodes = []
    for method in ('robust','teacher'):
        env = environment('clean',cap=450)
        obs,_ = env.reset(seed=seed)
        rng = np.random.default_rng(np.random.SeedSequence([seed,4891]))
        decisions = []
        for _ in range(450):
            start = time.perf_counter()
            nuisance = seed+env.drops+1
            if method == 'teacher':
                action,baseline,tested = teacher(env,policy,models,obs,rng,nuisance)
            else:
                action,_ = policy.decide(obs,np.random.default_rng(nuisance),'robust',trace=False)
                baseline,tested = action,None
            ms = (time.perf_counter()-start)*1000
            obs,gain,terminated,truncated,info = env.step(action)
            decisions.append(dict(drop=env.drops,action=action,baseline=baseline,reward=gain,
                                  score=env.score,ms=ms,candidates=tested))
            ended = terminated or truncated
            if env.drops == 300 or (ended and env.drops < 300):
                common = dict(**info,overflow=overflow(env))
            if ended:
                break
        episodes.append(dict(method=method,seed=seed,at_300=common,
                             extended=dict(**info,overflow=overflow(env)),decisions=decisions))
    return episodes


def winner(rows):
    return max(rows,key=lambda row:(row['value'],-row['action']))['action']


def diagnostic(audits):
    result = {}
    for profile in ('all','clean','joint'):
        group = [r for r in audits if profile=='all' or r['profile']==profile]
        advantages = []
        for row in group:
            initial = {r['action']:r['value'] for r in row['candidates']}
            replica = {r['action']:r['value'] for r in row['independent']}
            selected, reference = winner(row['candidates']),row['baseline']
            advantages.append((initial[selected]-initial[reference],replica[selected]-replica[reference]))
        result[profile] = dict(boards=len(group),
             selected_over_robust_same_rollout=mean(a for a,_ in advantages),
             selected_over_robust_independent_rollout=mean(b for _,b in advantages),
             predicted_gain_lost=sum(a>0 and b<0 for a,b in advantages),
             crowded_20_plus=sum(r['fruits']>=20 for r in group),
             high_center_below_80_px_y=sum(r['highest_center']<80 for r in group))
    return result


def main(workers):
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        audits = [row for group in pool.map(audit_job,[(s,m,p) for s in AUDIT_SEEDS
                  for m in ('robust','planning') for p in ('clean','joint')]) for row in group]
        audit_seconds = time.perf_counter()-start
        games = list(pool.map(episode_job,DEV))
    rows = []
    for horizon in ('at_300','extended'):
        deltas = [b[horizon]['score']-a[horizon]['score'] for a,b in games]
        rows.append(dict(horizon=horizon,baseline=mean(a[horizon]['score'] for a,b in games),
                         teacher=mean(b[horizon]['score'] for a,b in games),delta=mean(deltas),
                         deltas=deltas,wins=sum(x>0 for x in deltas),ties=deltas.count(0),
                         losses=sum(x<0 for x in deltas)))
    audit_sample = [x for x in audits if x['all_actions']]
    repeat = sum(winner(x['candidates'])==winner(x['independent']) for x in audits)
    audit = dict(boards=len(audits),repeat_winner=repeat,all_action_boards=len(audit_sample),
                 retained_winner=sum(winner(x['all_actions']) in [r['action'] for r in x['candidates']]
                                     for x in audit_sample),seconds=audit_seconds)
    threshold = rows[0]['delta'] >= 100 and rows[0]['wins'] > len(DEV)/2
    latency = {name:np.percentile([d['ms'] for pair in games for r in pair if r['method']==name
                                 for d in r['decisions'][1:]],[50,95]).tolist()
               for name in ('robust','teacher')}
    json_file(ROOT/'teacher-results.json',dict(audit=audit,summary=rows,latency_ms=latency,
              trained_ranker_eligible=threshold,workers=workers,seconds=time.perf_counter()-start,
              episodes=[{k:v for k,v in row.items() if k!='decisions'} for pair in games for row in pair]))
    json_file(ROOT/'teacher-diagnostics.json',diagnostic(audits))
    Path('runs').mkdir(exist_ok=True)
    with gzip.open(Path('runs')/'teacher-ledger.json.gz','wt') as f:
        json.dump(dict(audit=audits,games=games),f,separators=(',',':'))
    with gzip.open(ROOT/'showcase-ledger.json.gz','wt') as f:
        json.dump(next(rows for rows in games if rows[0]['seed']==SHOWCASE),f,separators=(',',':'))
    print(audit,rows,latency,threshold,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=12)
    main(parser.parse_args().workers)

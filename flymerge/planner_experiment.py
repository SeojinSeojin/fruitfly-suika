"""Paired one-drop planner experiment and website replay diagnostics."""
import argparse
import copy
import gzip
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from statistics import mean
import time

from .policy import np, restore, torch
from .game import Orchard, observe
from .planning import drop_values
from .replay import json_file
from .verify import unpack

ROOT = Path('artifacts/planner')
CHECKPOINT = Path('artifacts/transfer/checkpoints/flywire-seed11-final.pt')
WEIGHTS = (0., .05, .20)
NEAR = .001
DEVELOPMENT = (20270101, 20270102, 20270103)
HELD_OUT = tuple(range(20270201, 20270225))
SHOWCASE = HELD_OUT[0]
PROFILES = ('clean', 'joint', 'stress')


def load():
    return restore(CHECKPOINT, json.loads(Path('artifacts/graph.json').read_text()))[0]


@torch.no_grad()
def logp(policy, observation):
    x = torch.from_numpy(observation[None])
    return np.log(policy.action_logits(policy.circuit(x), x).softmax(-1)[0]
                  .numpy().astype(float).clip(1e-8))


def scores(policy, obs):
    physics, prior = drop_values(obs), logp(policy, obs)
    variants = []
    for weight in WEIGHTS:
        total = physics + weight*prior
        winner = int(total.argmax())
        margin = float(total[winner] - np.partition(total, -2)[-2])
        variants.append(dict(weight=weight, scores=total.tolist(), action=winner, margin=margin,
                             exact=np.flatnonzero(total == total[winner]).tolist(),
                             near=np.flatnonzero(total >= total[winner]-NEAR).tolist()))
    return dict(physics=physics.tolist(), log_probability=prior.tolist(), variants=variants)


def diagnose():
    policy = load()
    source = unpack(Path('artifacts/transfer/demo-game-over/website.js'))
    rows = []
    for d in source['decisions']:
        obs = np.array(d['observation'], dtype=np.float32)
        fruits = [(1-x,y,round(float(t)*10)) for x,y,t in obs[:-2].reshape(-1,3) if x >= 0]
        mirror = observe(fruits, *(round(float(t)*10) for t in obs[-2:]))
        rows.append(dict(drop=d['drop'], logged=d['action'], original=scores(policy, obs),
                         mirrored=scores(policy, mirror)))
    summary = []
    for i, weight in enumerate(WEIGHTS):
        original = [r['original']['variants'][i] for r in rows]
        mirrored = [r['mirrored']['variants'][i] for r in rows]
        summary.append(dict(weight=weight, rightmost=sum(x['action']==14 for x in original),
                            mirrored_rightmost=sum(x['action']==14 for x in mirrored),
                            mirrored_leftmost=sum(x['action']==0 for x in mirrored),
                            reproduced=sum(x['action']==r['logged'] for x,r in zip(original,rows)),
                            exact_ties=sum(len(x['exact'])>1 for x in original),
                            near_ties=sum(len(x['near'])>1 for x in original),
                            rightmost_exact=sum(x['action']==14 and len(x['exact'])>1 for x in original),
                            rightmost_near=sum(x['action']==14 and len(x['near'])>1 for x in original),
                            mirror_equivariant=sum(a['action']==14-b['action'] for a,b in zip(original,mirrored)),
                            median_margin=float(np.median([x['margin'] for x in original]))))
    json_file(ROOT/'edge-diagnostics.json', dict(checkpoint=str(CHECKPOINT),
              sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), near_tolerance=NEAR,
              tie_rule='first index wins exact ties', recovered=len(rows), summary=summary, drops=rows))
    print(summary)


def environment(profile, cap=450, capture=False):
    physics, noise = {'clean': (False, 0.), 'joint': (True, 1.), 'stress': (True, 1.5)}[profile]
    return Orchard(rules='transfer', randomize_physics=physics, vision_noise=noise,
                   drop_limit=cap, capture=capture)


def overflow(env):
    block = env.body[:env.n]
    return bool(env.n >= len(env.body)-1 or np.any(
        block[:,1]-env.radii[block[:,4].astype(int)] < env.overflow_y))


def run_pair(job):
    from .robust_planner import choose

    profile, seed = job
    policy = load()
    # Warm compiled paths outside timed decisions.
    empty = Orchard().reset(seed=0)[0]
    choose(empty, logp(policy, empty), np.random.default_rng(0))
    rows, changes, audit = [], [], []
    for label in ('A', 'B', 'C'):
        env = environment(profile)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 8051]))
        times, at_300 = [], None
        for drop in range(1, 451):
            start = time.perf_counter_ns()
            if label == 'C':
                prior = logp(policy, obs)
                random_state = copy.deepcopy(rng.bit_generator.state) if drop in (25,100,200) else None
                action, details = choose(obs, prior, rng)
            else:
                action, _ = policy.decide(obs, rng, 'hybrid' if label == 'A' else 'planning', trace=False)
            times.append((time.perf_counter_ns()-start)/1e6)
            if label == 'C':
                baseline = int(np.argmax(details['nominal']))
                if random_state:
                    all_rng = np.random.default_rng()
                    all_rng.bit_generator.state = random_state
                    full, _ = choose(obs, prior, all_rng, shortlist=15)
                    audit.append(dict(drop=drop, shortlist=action, full=full, included=full in details['candidates']))
                if action != baseline:
                    alternate = copy.deepcopy(env)
                    _, alt_gain, _, _, _ = alternate.step(baseline)
                    change = dict(drop=drop, baseline=baseline, action=action, alternate_gain=alt_gain,
                                  alternate_overflow=overflow(alternate), **details)
            obs, gain, terminated, truncated, info = env.step(action)
            ended = terminated or truncated
            if label == 'C' and action != baseline:
                changes.append(dict(**change, gain=gain, overflow=overflow(env)))
            if drop == 300 or (ended and drop < 300):
                at_300 = dict(**info, overflow=ended and overflow(env))
            if ended:
                break
        rows.append(dict(method=label, seed=seed, profile=profile, at_300=at_300,
                         extended=dict(**info, overflow=overflow(env)),
                         timing_ms=times))
    return rows, dict(seed=seed, profile=profile, changes=changes, audit=audit)


def evaluate(phase, workers):
    from .robust_planner import SHORTLIST, RISK_WEIGHT, WEIGHT

    seeds = DEVELOPMENT if phase == 'develop' else HELD_OUT
    if phase == 'evaluate':
        assert (ROOT/'development.json').is_file(), 'Complete development and freeze settings first'
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_pair, [(p,s) for p in PROFILES for s in seeds]))
    rows = [r for group,_ in results for r in group]
    details = [trace for _,trace in results]
    summary = []
    for profile in PROFILES:
        for horizon in ('at_300','extended'):
            a = {r['seed']:r[horizon]['score'] for r in rows if r['profile']==profile and r['method']=='A'}
            for method in ('A','B','C'):
                group = [r for r in rows if r['profile']==profile and r['method']==method]
                delta = [r[horizon]['score']-a[r['seed']] for r in group]
                ms = [m for r in group for m in r['timing_ms'][:r[horizon]['length']]]
                summary.append(dict(profile=profile, horizon=horizon, method=method,
                                    score=mean(r[horizon]['score'] for r in group), paired_delta=mean(delta),
                                    wins=sum(x>0 for x in delta), ties=sum(x==0 for x in delta), losses=sum(x<0 for x in delta),
                                    tier=mean(r[horizon]['highest'] for r in group),
                                    overflow=sum(r[horizon]['overflow'] for r in group),
                                    alive_at_cap=sum(not r[horizon]['overflow'] for r in group),
                                    latency_ms=np.percentile(ms,[50,95]).tolist()))
    for r in rows:
        r.pop('timing_ms')
    audited = [x for group in details for x in group['audit']]
    changed = [x for group in details for x in group['changes']]
    aggregate = dict(sampled_boards=len(audited), contained=sum(a['included'] for a in audited),
                     same_winner=sum(a['shortlist']==a['full'] for a in audited), changed=len(changed),
                     gains_better=sum(d['gain']>d['alternate_gain'] for d in changed),
                     gains_worse=sum(d['gain']<d['alternate_gain'] for d in changed),
                     median_simulated_disagreement=float(np.median([d['selected_sd'] for d in changed])) if changed else None)
    filename = 'development' if phase == 'develop' else 'results'
    json_file(ROOT/f'{filename}.json', dict(seeds=seeds, showcase=SHOWCASE, profiles=PROFILES,
              seconds=time.perf_counter()-start, shortlist=SHORTLIST, risk_weight=RISK_WEIGHT,
              ppo_weight=WEIGHT, groups=summary, changes=aggregate, rows=rows))
    with gzip.open(ROOT/f'{filename}-decisions.json.gz','wt') as out:
        json.dump(details, out, separators=(',',':'))
    print(aggregate, flush=True)


def benchmark():
    policy = load()
    source = unpack(Path('artifacts/transfer/demo-game-over/website.js'))
    rows = []
    names = ('hybrid', 'planning', 'robust')
    rngs = [np.random.default_rng(4241) for _ in names]
    for index, saved in enumerate(source['decisions']):
        obs = np.array(saved['observation'], dtype=np.float32)
        if index == 0:
            for name in names:
                policy.decide(obs, np.random.default_rng(0), name, trace=False)
        row = dict(drop=saved['drop'])
        for i in ((index+j)%3 for j in range(3)):
            start = time.perf_counter_ns()
            policy.decide(obs, rngs[i], names[i], trace=False)
            row[names[i]] = (time.perf_counter_ns()-start)/1e6
        rows.append(row)
    json_file(ROOT/'latency.json', dict(boards=len(rows), units='milliseconds', measurements=rows,
              percentiles={name:np.percentile([r[name] for r in rows],[50,95]).tolist() for name in names}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['diagnose','develop','evaluate','benchmark'])
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.command == 'diagnose':
        diagnose()
    elif args.command == 'benchmark':
        benchmark()
    else:
        evaluate(args.command, args.workers)

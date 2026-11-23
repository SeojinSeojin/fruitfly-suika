"""Instrumented collection: how training-style episodes end (overflow vs drop cap).

Workers match flymerge.train: Orchard(missing=.12, rules='transfer', drop_limit=300) with fixed physics
(deployed checkpoint's training profile) or randomized physics plus unit vision noise (joint pilot).
Actions are sampled from the frozen deployed checkpoint; no PPO updates are performed. Overflow is
recomputed independently after every step so that an overflow on the cap drop is visible.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .game import Orchard
from .planner_experiment import load, overflow
from .replay import json_file

OUT = Path('artifacts/live-baseline/gameover-audit/training-endings.json')


def collect(job):
    profile, worker, drops = job
    policy = load()
    joint = profile == 'joint'
    env = Orchard(missing=.12, rules='transfer', randomize_physics=joint, vision_noise=1. if joint else 0.)
    rng = np.random.default_rng(9000+worker)
    obs, _ = env.reset(seed=7700*(1+joint)+worker)
    episodes, total = [], 0
    while total < drops:
        action, _ = policy.decide(obs, rng, 'sample', trace=False)
        obs, _, terminal, truncated, info = env.step(action)
        total += 1
        if terminal or truncated:
            spilled = overflow(env)
            episodes.append(dict(profile=profile, worker=worker, length=info['length'], score=info['score'],
                                 terminal=bool(terminal), truncated=bool(truncated), overflow=spilled,
                                 overflow_on_cap_drop=bool(truncated and spilled)))
            obs, _ = env.reset()
    return episodes


def main(args):
    jobs = [(p, w, args.drops) for p in ('fixed', 'joint') for w in range(args.workers)]
    with Pool(len(jobs)) as pool:
        episodes = [e for chunk in pool.map(collect, jobs) for e in chunk]
    summary = {}
    for profile in ('fixed', 'joint'):
        group = [e for e in episodes if e['profile'] == profile]
        summary[profile] = dict(
            completed_episodes=len(group),
            ended_by_overflow_before_cap=sum(e['terminal'] for e in group),
            ended_at_cap=sum(e['truncated'] for e in group),
            cap_alive=sum(e['truncated'] and not e['overflow'] for e in group),
            overflow_on_cap_drop_labeled_truncated=sum(e['overflow_on_cap_drop'] for e in group),
            mean_length=float(np.mean([e['length'] for e in group])),
            mean_score=float(np.mean([e['score'] for e in group])))
    json_file(OUT, dict(description=__doc__.strip(), drops_per_worker=args.drops, workers_per_profile=args.workers,
                        label='new measurement (no archived training episode logs record the ending)',
                        summary=summary, episodes=episodes))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--drops', type=int, default=6144)
    main(parser.parse_args())

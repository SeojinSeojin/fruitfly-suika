"""Paired evaluation of PPO continuations with screenshot-like observations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import mean

from .game import Orchard
from .policy import np, restore
from .replay import json_file

SEEDS = (51, 52, 53)
FRUIT_SEEDS = tuple(range(20261201, 20261213))
CONDITIONS = ('fixed', 'physics', 'both')
PROFILES = {'clean': (False, 0.), 'physics': (True, 0.),
            'vision': (False, 1.), 'both': (True, 1.), 'stress': (True, 1.5)}


def evaluate(checkpoint, condition, seed, graph):
    policy, metadata = restore(checkpoint, graph)
    rows = []
    for profile, (physics, noise) in PROFILES.items():
        for fruit_seed in FRUIT_SEEDS:
            strategies = ('sample', 'hybrid') if profile == 'both' else ('sample',)
            for strategy in strategies:
                env = Orchard(rules='transfer', randomize_physics=physics, vision_noise=noise)
                observation, _ = env.reset(seed=fruit_seed)
                rng = np.random.default_rng(fruit_seed+1771)
                while True:
                    action, _ = policy.decide(observation, rng, strategy, trace=False)
                    observation, _, terminal, truncated, info = env.step(action)
                    if terminal or truncated:
                        break
                rows.append(dict(condition=condition, training_seed=seed, profile=profile,
                                 fruit_seed=fruit_seed, strategy=strategy, **info))
    print(condition, seed, len(rows), flush=True)
    return dict(path=str(checkpoint), drops=metadata['drops'], updates=metadata['updates'],
                training_seed=seed, condition=condition, rows=rows)


def main(args):
    graph = json.loads(Path('artifacts/graph.json').read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(evaluate, args.runs/f'{condition}-{seed}'/'final.pt', condition, seed, graph)
                for seed in SEEDS for condition in CONDITIONS]
        runs = [job.result() for job in jobs]
    rows = [row for run in runs for row in run.pop('rows')]
    scores = {(r['condition'], r['training_seed'], r['profile'], r['fruit_seed'], r['strategy']): r['score']
              for r in rows}
    paired = []
    for profile in PROFILES:
        for strategy in (('sample', 'hybrid') if profile == 'both' else ('sample',)):
            for seed in SEEDS:
                for condition in CONDITIONS:
                    group = [scores[condition, seed, profile, f, strategy] for f in FRUIT_SEEDS]
                    delta = {control: mean(s-scores[control, seed, profile, f, strategy]
                                           for s, f in zip(group, FRUIT_SEEDS))
                             for control in CONDITIONS if control != condition}
                    paired.append(dict(profile=profile, strategy=strategy, seed=seed, condition=condition,
                                       mean_score=mean(group), paired_delta=delta))
    json_file(args.out, dict(runs=runs, fruit_seeds=FRUIT_SEEDS, rows=rows, by_training_seed=paired))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs/vision-suite'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/transfer/results/vision-robustness.json'))
    parser.add_argument('--workers', type=int, default=6)
    main(parser.parse_args())

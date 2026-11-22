"""Compare frozen policy action selection on the original held-out sequences."""
import argparse
import json
from pathlib import Path
import time

from .game import Orchard
from .policy import np, restore
from .replay import json_file


def main(distilled, aggregated, spatial, continuations, append):
    started = time.perf_counter()
    root = Path('artifacts')
    graph = json.loads((root/'graph.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    policy, _ = restore(root/'transfer/checkpoints/flywire-seed11-final.pt', graph)
    cases = [(policy, strategy, strategy) for strategy in ('sample', 'planning', 'hybrid', 'lookahead')]
    if distilled:
        student, _ = restore(distilled, graph)
        cases.extend(((policy, 'greedy', 'ppo_greedy'), (student, 'greedy', 'distilled_greedy'),
                      (student, 'hybrid', 'distilled_hybrid')))
    if aggregated:
        student, _ = restore(aggregated, graph)
        cases.extend(((student, 'greedy', 'aggregated_greedy'),
                      (student, 'hybrid', 'aggregated_hybrid')))
    if spatial:
        student, _ = restore(spatial, graph)
        cases.extend(((student, 'greedy', 'spatial_greedy'),
                      (student, 'hybrid', 'spatial_hybrid')))
    for label, path in zip(('continued_ppo', 'continued_spatial'), continuations):
        model, _ = restore(path, graph)
        cases.extend((model, strategy, f'{label}_{strategy}') for strategy in ('sample', 'greedy'))
    result_path = root/'transfer/results/planning.json'
    existing = json.loads(result_path.read_text()) if append else dict(rows=[], seconds=0.)
    rows = existing['rows']
    if append:
        evaluated = {row['strategy'] for row in rows}
        cases = [case for case in cases if case[2] not in evaluated]

    for seed in protocol['held_out_fruit_seeds']:
        for model, strategy, label in cases:
            env = Orchard(rules='transfer')
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(seed+protocol['evaluation_action_seed_offset'])
            for _ in range(300):
                action, _ = model.decide(obs, rng, strategy, trace=False)
                obs, _, terminal, truncated, info = env.step(action)
                if terminal or truncated:
                    break
            rows.append(dict(seed=seed, strategy=label, **info))
            print(seed, label, info, flush=True)
    json_file(result_path, dict(seconds=existing['seconds'] + time.perf_counter()-started, rows=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distilled', type=Path)
    parser.add_argument('--aggregated', type=Path)
    parser.add_argument('--spatial', type=Path)
    parser.add_argument('--continuations', nargs=2, type=Path, default=())
    parser.add_argument('--append', action='store_true')
    args = parser.parse_args()
    main(args.distilled, args.aggregated, args.spatial, args.continuations, args.append)

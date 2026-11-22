"""Check screenshot-to-simulator merge agreement and randomized training robustness."""
import argparse
import json
from pathlib import Path

from .game import Orchard
from .planning import predict_drops
from .policy import np, restore
from .replay import json_file
from .verify import unpack


def main(checkpoint):
    root = Path('artifacts')
    transfer = root/'transfer'
    graph = json.loads((root/'graph.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    windows = []
    for path in sorted(transfer.glob('demo*/website.js')):
        replay = unpack(path)
        for decision in replay['decisions']:
            before, after = decision['score_before'], decision['score']
            if before is None or after is None:
                continue
            _, predictions = predict_drops(np.array(decision['observation'], dtype=np.float32))
            windows.append(dict(capture=path.parent.name, drop=decision['drop'],
                                predicted_score=int(predictions[decision['action']]), actual_score=after-before))
    trials = []
    baseline = transfer/'checkpoints/flywire-seed44-continued-ppo.pt'
    for label, path in [('fixed', baseline), ('randomized', checkpoint)]:
        policy, _ = restore(path, graph)
        for evaluation in ('fixed', 'randomized'):
            for seed in protocol['held_out_fruit_seeds']:
                rng = np.random.default_rng(seed+protocol['evaluation_action_seed_offset'])
                env = Orchard(rules='transfer', randomize_physics=evaluation == 'randomized')
                obs, _ = env.reset(seed=seed)
                while True:
                    action, _ = policy.decide(obs, rng, trace=False)
                    obs, _, terminal, truncated, info = env.step(action)
                    if terminal or truncated:
                        break
                trials.append(dict(training=label, evaluation=evaluation, seed=seed, **info))
    json_file(transfer/'results/physics.json', dict(local=trials, transfer=windows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    main(parser.parse_args().checkpoint)

"""Train a FlyWire readout to predict the local simulator's preferred drops."""
import argparse
import json
from pathlib import Path

from .game import Orchard
from .planning import drop_values
from .policy import np, restore, torch


def train(args):
    graph = json.loads(Path('artifacts/graph.json').read_text())
    model, checkpoint = restore(args.source, graph)
    rng = np.random.default_rng(53001)
    examples, estimates = [], []
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=.002)
    for round_index in range(args.rounds):
        for seed in range(53001+round_index*args.episodes, 53001+(round_index+1)*args.episodes):
            env = Orchard(rules='transfer')
            obs, _ = env.reset(seed=seed)
            while True:
                values = drop_values(obs)
                examples.append(obs)
                estimates.append(values)
                draw = rng.random()
                if draw < .15:
                    action = int(rng.integers(15))
                elif round_index and draw < .65:
                    action, _ = model.decide(obs, rng, trace=False)
                else:
                    action = int(values.argmax())
                obs, _, terminal, truncated, _ = env.step(action)
                if terminal or truncated:
                    break
            print(seed, len(examples), flush=True)
        x = torch.as_tensor(np.stack(examples))
        with torch.no_grad():
            features = torch.cat([model.features(b) for b in x.split(256)])
            teacher = torch.as_tensor(np.stack(estimates), dtype=torch.float32)
            target = (teacher / .20).softmax(-1)
        model.train()
        for _ in range(35):
            for batch in np.array_split(rng.permutation(len(x)), (len(x)+127)//128):
                predicted = model.actor(features[batch]).log_softmax(-1)
                loss = -(target[batch] * predicted).sum(-1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(**{k: v for k, v in checkpoint.items() if k != 'state'}, state=model.state_dict(),
                    imitation_examples=len(x), imitation_episodes=args.episodes*args.rounds,
                    imitation_rounds=args.rounds), args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/transfer/checkpoints/flywire-seed11-final.pt'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--rounds', type=int, default=1)
    args = parser.parse_args()
    if args.rounds < 1 or args.episodes < 1:
        parser.error('Positive rounds and episodes are required')
    train(args)

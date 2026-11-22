"""CPU PPO with synchronous batches and milestone checkpoints."""
import argparse
import json
import multiprocessing as mp
import resource
import time
from pathlib import Path

from .policy import Policy, np, torch
from .game import Orchard
from torch.distributions import Categorical


def worker(pipe, seed):
    env = Orchard(missing=.12)
    pipe.send(env.reset(seed=seed)[0])
    while (action := pipe.recv()) is not None:
        start = time.perf_counter()
        observation, reward, terminal, truncated, info = env.step(action)
        done = terminal or truncated
        if done:
            observation = env.reset()[0]
        pipe.send((observation, reward, done, info if done else None,
                   time.perf_counter() - start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def train(args):
    started = time.perf_counter()
    graph = json.loads(args.graph.read_text()) if args.kind != 'mlp' else None
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(exist_ok=True, parents=True)
    timings = dict(inference=0., environment_wall=0., worker_physics_sum=0.,
                   communication_and_wait=0., ppo=0., checkpoint=0.)
    Orchard().reset(seed=0)
    warm = Orchard()
    warm.reset(seed=0)
    warm.step(0)
    pipes, workers = [], []
    for i in range(args.workers):
        host, guest = mp.Pipe()
        process = mp.Process(target=worker, args=(guest, args.seed * 1000 + i), daemon=True)
        process.start()
        guest.close()
        pipes.append(host)
        workers.append(process)
    observation = np.stack([pipe.recv() for pipe in pipes])
    policy = Policy(args.kind, graph, args.sparse)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    total = updates = 0
    peaks = [0] * args.workers
    marks = {'init': 0, 'drop5': 5, 'drop1000': 1000, 'early': args.budget * 5 // 100,
             'middle': args.budget // 2, 'final': args.budget}
    snapshots = {}

    def save(label):
        start = time.perf_counter()
        path = args.out / f'{label}.pt'
        torch.save(dict(kind=args.kind, seed=args.seed, drops=total, updates=updates,
                        requested=marks[label], state=policy.state_dict()), path)
        snapshots[label] = dict(file=path.name, requested=marks[label], drops=total, updates=updates)
        timings['checkpoint'] += time.perf_counter() - start

    save('init')
    learning_started = time.perf_counter()
    with (args.out / 'episodes.jsonl').open('w') as episodes, (args.out / 'learning.jsonl').open('w') as learning:
        while total < args.budget:
            feats, actions, old_logp, values, rewards, done_rows = [], [], [], [], [], []
            length = min(args.rollout // args.workers, (args.budget - total) // args.workers)
            for _ in range(length):
                start = time.perf_counter()
                with torch.no_grad():
                    features = policy.features(torch.from_numpy(observation))
                    logits, value = policy.heads(features)
                    distribution = Categorical(logits=logits)
                    action = distribution.sample()
                    logp = distribution.log_prob(action)
                timings['inference'] += time.perf_counter() - start
                start = time.perf_counter()
                for pipe, move in zip(pipes, action.tolist()):
                    pipe.send(move)
                result = [pipe.recv() for pipe in pipes]
                elapsed = time.perf_counter() - start
                timings['environment_wall'] += elapsed
                obs, reward, done, infos, physics, memory = zip(*result)
                timings['worker_physics_sum'] += sum(physics)
                timings['communication_and_wait'] += max(0., elapsed - max(physics))
                peaks = [max(a, b) for a, b in zip(peaks, memory)]
                observation = np.stack(obs)
                total += args.workers
                for i, info in enumerate(infos):
                    if info:
                        episodes.write(json.dumps(dict(drops=total, worker=i, **info)) + '\n')
                feats.append(features)
                actions.append(action)
                old_logp.append(logp)
                values.append(value)
                rewards.append(reward)
                done_rows.append(done)
                if total % args.rollout and total < args.budget:
                    for label, goal in marks.items():
                        if label not in snapshots and total >= goal:
                            save(label)
            start = time.perf_counter()
            with torch.no_grad():
                _, next_value = policy.heads(policy.features(torch.from_numpy(observation)))
                advantages = torch.empty((length, args.workers))
                gae = 0.
                for t in reversed(range(length)):
                    alive = 1. - torch.tensor(done_rows[t], dtype=torch.float32)
                    delta = torch.tensor(rewards[t], dtype=torch.float32) + .99 * next_value * alive - values[t]
                    gae = delta + .99 * .95 * alive * gae
                    advantages[t] = gae
                    next_value = values[t]
                targets = advantages.flatten() + torch.cat(values)
                advantages = advantages.flatten()
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            feature_batch, action_batch, logp_batch = map(torch.cat, (feats, actions, old_logp))
            for _ in range(4):
                indexes = rng.permutation(len(targets))
                for begin in range(0, len(targets), 256):
                    i = indexes[begin:begin + 256]
                    logits, value = policy.heads(feature_batch[i])
                    dist = Categorical(logits=logits)
                    ratio = (dist.log_prob(action_batch[i]) - logp_batch[i]).exp()
                    loss = (-torch.minimum(ratio * advantages[i], ratio.clamp(.8, 1.2) * advantages[i]).mean()
                            + .5 * (value - targets[i]).square().mean() - .01 * dist.entropy().mean())
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
                    optimizer.step()
            updates += 1
            timings['ppo'] += time.perf_counter() - start
            learning.write(json.dumps(dict(drops=total, updates=updates, loss=float(loss.detach()),
                                           seconds=time.perf_counter() - learning_started)) + '\n')
            learning.flush()
            for label, goal in marks.items():
                if label not in snapshots and total >= goal:
                    save(label)
    seconds = time.perf_counter() - learning_started
    for pipe in pipes:
        pipe.send(None)
    for process in workers:
        process.join()
    summary = dict(kind=args.kind, seed=args.seed, budget=total, workers=args.workers, sparse=args.sparse,
                   updates=updates, seconds=seconds, runtime=time.perf_counter() - started,
                   drops_per_second=total / seconds, learner_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   worker_peak_kib=peaks, parameters=sum(p.numel() for p in policy.parameters()),
                   timings=timings, snapshots=snapshots)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('kind', 'seed', 'budget', 'workers', 'seconds', 'drops_per_second')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['flywire', 'rewired', 'mlp'], required=True)
    parser.add_argument('--graph', type=Path, default=Path('artifacts/graph.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--budget', type=int, default=100352)
    parser.add_argument('--rollout', type=int, default=1024)
    parser.add_argument('--sparse', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.budget <= 0 or args.rollout <= 0 or args.budget % args.workers or args.rollout % args.workers:
        parser.error('Positive budget and rollout must be divisible by the worker count')
    train(args)

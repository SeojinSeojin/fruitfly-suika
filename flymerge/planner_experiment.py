"""Paired one-drop planner experiment and website replay diagnostics."""
import argparse
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['diagnose'])
    diagnose()

"""Plot paired held-out evaluation and actual training episode curves."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('artifacts/results')
STYLES = {'flywire': ('FlyWire', '#16746c'), 'rewired': ('Rewired', '#b56131'), 'mlp': ('MLP', '#5c61a7')}


def main(root):
    results = json.loads((root/'evaluation.json').read_text())
    seeds = sorted({entry['seed'] for entry in results})
    budget = max(entry['drops'] for entry in results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for kind, (label, color) in STYLES.items():
        grouped = defaultdict(list)
        for entry in results:
            if entry['kind'] == kind:
                grouped[entry['drops'], entry['seed']].append(entry['score'])
        drops = sorted({index[0] for index in grouped})
        means = np.array([[np.mean(grouped[step, seed]) for seed in seeds] for step in drops])
        axes[0].plot(drops, means.mean(1), 'o-', color=color, label=label)
        axes[0].fill_between(drops, means.min(1), means.max(1), color=color, alpha=.12)
        pooled = defaultdict(list)
        for path in (root/'runs').glob(f'{kind}-*/episodes.jsonl'):
            for line in path.read_text().splitlines():
                episode = json.loads(line)
                pooled[(episode['drops']-1)//5000].append(episode['score'])
        bins = sorted(pooled)
        axes[1].plot([min((b+.5)*5000, budget) for b in bins],
                     [np.mean(pooled[b]) for b in bins], color=color, label=label)
    axes[0].set(title='Held-out evaluation', ylabel='Mean episode score', xlabel='Training drops')
    axes[1].set(title='Training episodes · 5,000-drop bins', xlabel='Training drops')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.ticklabel_format(axis='x', style='plain')
        axis.legend(frameon=False)
    fig.savefig(root/'learning-curves.png', dpi=150)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    main(parser.parse_args().output)

"""Counterfactual of mask memory and the strict gate on recorded website boards (no physics: labels only)."""
import json
from pathlib import Path

import numpy as np

from .replay import json_file
from .rollout_transfer import reconstruct
from .spawn_contact import SPAWN_OFFSET_PX, carry_hidden, gate, label_actions
from .verify import unpack

RUNS = Path('artifacts/spawn-contact/live')
OUT = Path('artifacts/spawn-contact/strict-replay.json')


def replay(folder):
    capture = unpack(folder/'website.js')
    box = tuple(capture['meta']['board_bounds'])
    rows, remembered, previous_hand = [], [], None
    for d in capture['decisions']:
        body, count, current, _ = reconstruct(np.asarray(d['observation'], np.float32))
        seen = [(float(body[i, 0]), float(body[i, 1]), int(body[i, 4])) for i in range(count)]
        seen += [tuple(f) for f in d['band']]
        hand = None if previous_hand is None else (previous_hand, box[1]+SPAWN_OFFSET_PX, current)
        carried = carry_hidden(remembered, seen, hand, box)
        remembered = seen+carried
        base = [l['label'] for l in label_actions(seen, current, box)]
        labels = label_actions(remembered, current, box)
        names = [l['label'] for l in labels]
        executed = d['action']
        block, allowed = gate(labels, executed, strict=True)
        rows.append(dict(drop=d['drop'], executed_aim=int(executed)+1, recorded_label=base[executed],
                         memory_label=names[executed], carried=[[round(x, 1), round(y, 1), int(t)] for x, y, t in carried],
                         strict_blocks=block, allowed_aims=[int(i)+1 for i in np.flatnonzero(allowed)] if block else None))
        previous_hand = d.get('hand_x')
    return rows


def main():
    report = {}
    for k in (1, 2, 3):
        rows = replay(RUNS/f'attempt-{k}')
        report[f'attempt-{k}'] = dict(
            decisions=len(rows), boards_with_carried_fruit=sum(bool(r['carried']) for r in rows),
            labels_changed_by_memory=[r['drop'] for r in rows if r['memory_label'] != r['recorded_label']],
            strict_would_block=[r['drop'] for r in rows if r['strict_blocks']],
            final=rows[-1], last5=rows[-5:])
        print(k, json.dumps({a: b for a, b in report[f'attempt-{k}'].items() if a != 'last5'}))
    json_file(OUT, dict(note='labels only: the outcome of any changed aim is unknown', attempts=report))


if __name__ == '__main__':
    main()

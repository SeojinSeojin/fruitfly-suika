"""Replay archived boards: original detection, improved detection, improved detection plus contact check.

Live attempts 1-3 have raw timestamped frames; each decision is re-detected on the readiness frame the
browser used (the last settle frame before that drop's aim). The two older game-over captures only
have annotated screenshots, whose overlays cover the board; for them the saved in-box detections are
kept and only the unannotated band above the box is searched. Archived outcomes describe executed
actions only; other actions are evaluated by the contact rule, not by outcome.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .game import observe
from .planner_experiment import load, logp
from .replay import json_file
from .robust_planner import choose
from .site import Vision, board_extended, compose_background, extended_box, hand_pixels, hand_position
from .spawn_contact import SITE_RADII, gate, label_actions
from .verify import unpack

ROOT = Path('artifacts/spawn-contact')
LIVE = Path('runs/live-baseline')
ARCHIVED = {'archived-robust-1673': (Path('artifacts/planner/website'), 'robust'),
            'archived-hybrid-1725': (Path('artifacts/transfer/demo-game-over'), 'hybrid')}
CACHE = Path('data/site')


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def find_hand(vision, pixels, box):
    image, _ = hand_pixels(pixels, box)
    return hand_position(vision.match(image, range(5), (.64, .68, .72), (0,), .047), box)


def band_background(vision, empty, paths, box):
    """Empty-board frame with band rows from a per-column median over frames with the hand elsewhere."""
    shots = []
    for path in paths:
        pixels = load_rgb(path)
        hand = find_hand(vision, pixels, box)
        if hand:
            shots.append((hand[0], pixels))
    background, uncovered = compose_background(empty, shots, box)
    return background, dict(frames=len(shots), hand_x=sorted(round(float(x)) for x, _ in shots),
                            uncovered_columns=uncovered)


def detect_both(legacy, improved, pixels, box, background, ext_background, hand):
    x0, y0, x1, y1 = box
    old, cov_old = legacy.board(pixels[y0:y1, x0:x1], background, fast=True)
    if cov_old < .86:
        old, cov_old = legacy.board(pixels[y0:y1, x0:x1], background)
    new, cov_new = board_extended(improved, pixels, box, ext_background, hand, fast=True)
    if cov_new < .86:
        new, cov_new = board_extended(improved, pixels, box, ext_background, hand)
    return (old, cov_old), (new, cov_new)


def to_local(found):
    return [(x*410, y*440, tier) for x, y, tier, _ in found]


def compare_sets(a, b, box):
    """Count fruit in b without a same-tier partner in a within half a radius, and vice versa."""
    scale = (box[2]-box[0])/410
    def unmatched(p, q):
        return [f for f in p if not any(f[2] == g[2] and np.hypot(f[0]-g[0], f[1]-g[1]) < .5*SITE_RADII[f[2]] for g in q)]
    return unmatched(b, a), unmatched(a, b)


def decide(policy, obs, state, allowed=None):
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return choose(obs, logp(policy, obs), rng, allowed=allowed)[0]


def replay_live(name):
    path = LIVE/name
    capture = unpack(path/'website.js')
    meta, decisions = capture['meta'], capture['decisions']
    record = path/'record'
    frames = json.loads((record/'frames.json').read_text())['frames']
    box = tuple(meta['board_bounds'])
    x0, y0, x1, y1 = box
    legacy = Vision(CACHE, 4)
    improved = Vision(CACHE, 4, watermelon_inner=True)
    policy = load()
    settle = {}
    for f in frames:
        if f['tag'] == 'settle':
            settle.setdefault(f['drop'], []).append(f)
    empty = load_rgb(record/settle[0][-1]['image'])
    background = empty[y0:y1, x0:x1].copy()
    early = [record/f['image'] for f in frames if f['drop'] <= 8 and f['tag'] in ('settle', 'aim')]
    ext_background, hand_xs = band_background(improved, empty, early, box)
    rng = np.random.default_rng(meta['action_seed'])
    rows = []
    for d in decisions:
        state = rng.bit_generator.state
        obs_saved = np.asarray(d['observation'], np.float32)
        executed, _ = choose(obs_saved, logp(policy, obs_saved), rng)
        assert executed == d['action']
        frame = settle[d['drop']-1][-1]
        pixels = load_rgb(record/frame['image'])
        hand = find_hand(improved, pixels, box)
        (old, cov_old), (new, cov_new) = detect_both(legacy, improved, pixels, box, background, ext_background, hand)
        current, upcoming = round(float(obs_saved[-2])*10), round(float(obs_saved[-1])*10)
        inbox = [f for f in new if f[1] >= 0]
        band = [f for f in new if f[1] < 0]
        obs_new = observe([f[:3] for f in inbox], current, upcoming)
        improved_action = decide(policy, obs_new, state)
        labels_new = label_actions(to_local(new), current, box)
        labels_old = label_actions(to_local(old), current, box)
        block, allowed = gate(labels_new, improved_action)
        gated = decide(policy, obs_new, state, allowed) if block else improved_action
        added, removed = compare_sets(to_local([f for f in old]), to_local(inbox), box)
        rows.append(dict(
            drop=d['drop'], current=current+1, executed=executed+1, frame=frame['image'],
            hand=None if hand is None else [round(float(hand[0]), 1), round(float(hand[1]), 1), hand[2]+1],
            original=dict(count=len(old), coverage=round(cov_old, 3)),
            improved=dict(count=len(new), coverage=round(cov_new, 3), inbox=len(inbox),
                          band=[[round(x*410, 1), round(y*440, 1), t+1, round(q, 3)] for x, y, t, q in band],
                          added_inbox=[[round(a, 1), round(b, 1), t+1] for a, b, t in added],
                          removed_inbox=[[round(a, 1), round(b, 1), t+1] for a, b, t in removed]),
            saved_count=int((obs_saved[:-2].reshape(-1, 3)[:, 0] >= 0).sum()),
            executed_label_original=labels_old[executed]['label'],
            executed_label_improved=labels_new[executed]['label'], executed_contact=labels_new[executed],
            improved_action=improved_action+1, improved_action_label=labels_new[improved_action]['label'],
            gated_action=gated+1, blocked=block,
            labels_improved=[l['label'] for l in labels_new], labels_original=[l['label'] for l in labels_old]))
    terminal = meta['outcome'] == 'game over'
    extra = {}
    if name == 'attempt-3':
        # Frames after the last release, where the watermelon formed and the readiness check gave up.
        tail = settle.get(len(decisions), [])
        checks = []
        for f in tail:
            pixels = load_rgb(record/f['image'])
            hand = find_hand(improved, pixels, box)
            (old, cov_old), (new, cov_new) = detect_both(legacy, improved, pixels, box, background, ext_background, hand)
            checks.append(dict(frame=f['image'], t=f['t'], original_coverage=round(cov_old, 3),
                               original_watermelon=any(t == 10 for _, _, t, _ in old),
                               improved_coverage=round(cov_new, 3),
                               improved_watermelon=[[round(x*410, 1), round(y*440, 1)] for x, y, t, _ in new if t == 10],
                               improved_count=len(new)))
        extra['after_last_release'] = checks
    return name, dict(outcome=meta['outcome'], terminal_drop=decisions[-1]['drop'] if terminal else None,
                      band_background_hand_x=hand_xs, rows=rows, **extra)


def replay_archived(name):
    path, strategy = ARCHIVED[name]
    capture = unpack(path/'website.js')
    meta, decisions = capture['meta'], capture['decisions']
    box = tuple(meta['board_bounds'])
    x0, y0, x1, y1 = box
    ex0, ey0, ex1, ey1 = extended_box(box)
    improved = Vision(CACHE, 4, watermelon_inner=True)
    policy = load()
    screens = [path/d['image'] for d in decisions]
    ext_background, hand_xs = band_background(improved, load_rgb(screens[0]), screens[:12], box)
    rng = np.random.default_rng(meta['action_seed'])
    rows = []
    for d, screen in zip(decisions, screens):
        state = rng.bit_generator.state
        obs = np.asarray(d['observation'], np.float32)
        if strategy == 'robust':
            executed, _ = choose(obs, logp(policy, obs), rng)
            assert executed == d['action']
        executed = d['action']
        pixels = load_rgb(screen)
        hand = find_hand(improved, pixels, box)
        new, coverage = board_extended(improved, pixels, box, ext_background, hand, fast=True)
        band = [f for f in new if f[1] < -12/565]   # centers clearly above the annotated box outline
        saved = [(x, y, round(t*10), 0.) for x, y, t in obs[:-2].reshape(-1, 3) if x >= 0 and y >= 0]
        current = round(float(obs[-2])*10)
        labels_old = label_actions(to_local(saved), current, box)
        labels_new = label_actions(to_local(saved+band), current, box)
        block, allowed = gate(labels_new, executed)
        gated = executed
        if block and strategy == 'robust':
            gated = decide(policy, obs, state, allowed)
        rows.append(dict(drop=d['drop'], current=current+1, executed=executed+1,
                         band=[[round(x*410, 1), round(y*440, 1), t+1, round(q, 3)] for x, y, t, q in band],
                         executed_label_original=labels_old[executed]['label'],
                         executed_label_improved=labels_new[executed]['label'], executed_contact=labels_new[executed],
                         blocked=block, gated_action=gated+1,
                         labels_improved=[l['label'] for l in labels_new], labels_original=[l['label'] for l in labels_old]))
    return name, dict(outcome=meta['outcome'], strategy=strategy, terminal_drop=decisions[-1]['drop'],
                      band_background_hand_x=hand_xs, note='annotated screenshots: saved in-box detections plus band search',
                      rows=rows)


def run(job):
    kind, name = job
    return replay_live(name) if kind == 'live' else replay_archived(name)


def summarise(name, run):
    rows = run['rows']
    terminal = run.get('terminal_drop')
    body = [r for r in rows if r['drop'] != terminal]
    def count(key, value, group):
        return sum(r[key] == value for r in group)
    out = dict(releases=len(rows), nonterminal=len(body),
               nonterminal_executed_would_block=sum(r['executed_label_improved'] == 'fatal' and
                                                    any(l != 'fatal' for l in r['labels_improved']) for r in body),
               nonterminal_executed_fatal_improved=count('executed_label_improved', 'fatal', body),
               nonterminal_executed_uncertain_improved=count('executed_label_improved', 'uncertain', body),
               nonterminal_executed_fatal_original=count('executed_label_original', 'fatal', body),
               boards_with_band_detections=sum(bool(r.get('band') or r.get('improved', {}).get('band')) for r in rows))
    if 'improved' in rows[0]:
        out.update(nonterminal_blocked_any=sum(r['blocked'] for r in body),
                   improved_action_differs=sum(r['improved_action'] != r['executed'] for r in body),
                   inbox_added=sum(len(r['improved']['added_inbox']) for r in rows),
                   inbox_removed=sum(len(r['improved']['removed_inbox']) for r in rows),
                   original_rerun_matches_saved_count=sum(r['original']['count'] == r['saved_count'] for r in rows),
                   improved_coverage_below_gate=sum(r['improved']['coverage'] < .86 for r in rows),
                   original_coverage_below_gate=sum(r['original']['coverage'] < .86 for r in rows))
    if terminal:
        t = next(r for r in rows if r['drop'] == terminal)
        labels = t['labels_improved']
        out['terminal'] = dict(drop=terminal, executed=t['executed'], current=t['current'],
                               executed_label_original=t['executed_label_original'],
                               executed_label_improved=t['executed_label_improved'],
                               contact=t['executed_contact'], blocked=t['blocked'], gated_action=t['gated_action'],
                               improved_action=t.get('improved_action'), improved_action_label=t.get('improved_action_label'),
                               fatal_actions=[i+1 for i, l in enumerate(labels) if l == 'fatal'],
                               uncertain_actions=[i+1 for i, l in enumerate(labels) if l == 'uncertain'],
                               clear_actions=[i+1 for i, l in enumerate(labels) if l == 'clear'],
                               band=t.get('band') or t.get('improved', {}).get('band'),
                               added_inbox=t.get('improved', {}).get('added_inbox'))
    return out


def main(args):
    jobs = [('live', f'attempt-{k}') for k in (1, 2, 3)]+[('archived', n) for n in ARCHIVED]
    with Pool(len(jobs)) as pool:
        runs = dict(pool.map(run, jobs))
    summary = {name: summarise(name, r) for name, r in runs.items()}
    ROOT.mkdir(parents=True, exist_ok=True)
    json_file(ROOT/'replay-ledger.json', dict(description=__doc__.strip(), summary=summary, runs=runs))
    print(json.dumps(summary, indent=1))
    if 'after_last_release' in runs.get('attempt-3', {}):
        print(json.dumps(runs['attempt-3']['after_last_release'], indent=1))


if __name__ == '__main__':
    main(argparse.ArgumentParser(description=__doc__).parse_args())

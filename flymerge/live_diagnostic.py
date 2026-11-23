"""Replay executed website releases in the local simulator and compare observable outcomes.

For every release with a following settled observation, the local transfer physics is run from the
pre-drop detections with the executed aim. Predicted merge events, points, and the settled layout are
compared with the website's visible score change and the next settled detections. A tier-count
transition that cannot explain the visible score change marks a perception inconsistency.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from .game import AIMS, HEIGHT, SLOTS, TRANSFER_RADII, TRANSFER_WIDTH, tick
from .replay import json_file
from .rollout_transfer import NOMINAL, reconstruct
from .verify import unpack

ROOT = Path('artifacts/live-baseline')
RUNS = Path('runs/live-baseline')
ARCHIVED = {'archived-hybrid-1725': Path('artifacts/transfer/demo-game-over'),
            'archived-robust-1673': Path('artifacts/planner/website')}
MATCH_RADIUS = 40.
# Collision radii read from the site's public sprite sizes and its setSize/setScale(0.75) rules,
# converted to local units (0.963 screen px per game px, 526/410 screen px per local unit).
SITE_RADII = np.array([13.8, 17.7, 24.8, 27.0, 34.3, 43.9, 51.2, 62.8, 70.7, 87.0, 101.3])
VARIANTS = {'nominal': (NOMINAL, TRANSFER_RADII),
            'site radii': (NOMINAL, SITE_RADII),
            'site radii, no bounce': ((NOMINAL[0], NOMINAL[1], 0., NOMINAL[3]), SITE_RADII)}
HOLDOUT = 'attempt-3'


def points(tier):
    return (tier+1)*(tier+2)//2


def histogram(tiers):
    return np.bincount(np.asarray(tiers, int), minlength=11)[:11]


def solve_merges(before, after):
    """Unique merge counts per tier explaining a tier histogram change, or None."""
    merges, carry = [], 0
    for tier in range(11):
        diff = int(before[tier])+carry-int(after[tier])
        if diff < 0 or diff % 2 or (tier == 10 and diff):
            return None
        merges.append(diff//2)
        carry = diff//2
    return merges


def merge_points(merges):
    return int(sum(m*points(t) for t, m in enumerate(merges)))


def simulate(obs, action, physics=NOMINAL, ticks=80, radii=TRANSFER_RADII, drop=True):
    body, count, current, _ = reconstruct(obs)
    radius = TRANSFER_RADII[current]
    body[count] = radius+(TRANSFER_WIDTH-2*radius)*AIMS[action], -25., 0., 0., current
    if not drop:
        body[count] = 0
    n, gain, events, first = count+int(drop), 0, [], None
    for step in range(ticks):
        before = histogram(body[:n, 4])
        n, reward = tick(body, n, TRANSFER_WIDTH, radii, *physics)
        if reward:
            merges = solve_merges(before, histogram(body[:n, 4]))
            events.append(dict(tick=step, points=int(reward), merges=merges))
            first = step if first is None else first
        gain += reward
    top = min((body[j, 1]-radii[int(body[j, 4])] for j in range(n)), default=HEIGHT)
    speed = float(np.abs(body[:n, 2:4]).max()) if n else 0.
    return dict(body=body[:n].copy(), points=int(gain), events=events, top=float(top),
                overflow=bool(n >= SLOTS or top < 11), max_speed=speed,
                merges=solve_merges(histogram([*body[:count, 4], *([current] if drop else [])]), histogram(body[:n, 4])))


def layout(predicted, observed):
    """Same-tier optimal assignment; returns matched distances and unmatched counts."""
    distances, unmatched_pred, unmatched_obs = [], 0, 0
    for tier in range(11):
        a = predicted[predicted[:, 4] == tier, :2]
        b = observed[observed[:, 4] == tier, :2]
        if len(a) and len(b):
            cost = np.linalg.norm(a[:, None]-b[None], axis=2)
            rows, cols = linear_sum_assignment(cost)
            good = cost[rows, cols] <= MATCH_RADIUS
            distances += cost[rows, cols][good].tolist()
            unmatched_pred += len(a)-good.sum()
            unmatched_obs += len(b)-good.sum()
        else:
            unmatched_pred += len(a)
            unmatched_obs += len(b)
    return distances, int(unmatched_pred), int(unmatched_obs)


def category(predicted, observed):
    if predicted == observed:
        return 'agree'
    if predicted > 0 and observed == 0:
        return 'simulator merge, site none'
    if predicted == 0 and observed > 0:
        return 'site merge, simulator none'
    return 'both merged, different points'


def load_run(path):
    capture = unpack(path/'website.js')
    return capture['meta'], capture['decisions']


def analyse(name, path, frames=None):
    meta, decisions = load_run(path)
    rows = []
    for i, d in enumerate(decisions):
        obs = np.asarray(d['observation'], np.float32)
        body, count, current, upcoming = reconstruct(obs)
        nominal = simulate(obs, d['action'])
        long = simulate(obs, d['action'], ticks=300)
        row = dict(run=name, drop=d['drop'], action=d['action'], current=current, next=upcoming,
                   detections=count, score_before=d.get('score_before'), score=d.get('score'),
                   predicted_points=nominal['points'], predicted_merges=nominal['merges'],
                   predicted_events=nominal['events'], predicted_top=nominal['top'],
                   predicted_overflow=nominal['overflow'], residual_speed_80=nominal['max_speed'],
                   points_300_ticks=long['points'], overflow_300_ticks=long['overflow'],
                   hand_x=d.get('hand_x'), target_x=d.get('target_x'), aim_error_px=d.get('aim_error_px'),
                   release_time=d.get('release_time'), ready_time=d.get('ready_time'),
                   post_release_wait=d.get('post_release_wait'), timestamp=d.get('timestamp'))
        if d.get('score') is not None and d.get('score_before') is not None:
            row['observed_points'] = d['score']-d['score_before']
            row['category'] = category(nominal['points'], row['observed_points'])
        if i+1 < len(decisions):
            after = np.asarray(decisions[i+1]['observation'], np.float32)
            nbody, ncount, _, _ = reconstruct(after)
            observed_merges = solve_merges(histogram([*body[:count, 4], current]), histogram(nbody[:ncount, 4]))
            consistent = (observed_merges is not None and 'observed_points' in row
                          and merge_points(observed_merges) == row['observed_points'])
            distances, missing, extra = layout(nominal['body'], nbody[:ncount])
            row.update(next_detections=ncount, observed_merges=observed_merges,
                       perception_consistent=bool(consistent),
                       merge_events_agree=bool(consistent and observed_merges == nominal['merges']),
                       layout_median_px=float(np.median(distances)) if distances else None,
                       layout_matched=len(distances), layout_unmatched_predicted=missing,
                       layout_unmatched_observed=extra,
                       carry_over=(decisions[i+1].get('score_before') or 0)-(d.get('score') or 0))
            # Compare against a no-physics reference: the pre-drop board alone.
            still, _, _ = layout(np.vstack([body[:count]]), nbody[:ncount])
            row['static_matched'] = len(still)
        row['variants'] = {}
        for label, (physics, radii) in VARIANTS.items():
            sim = simulate(obs, d['action'], physics, radii=radii)
            alone = simulate(obs, d['action'], physics, radii=radii, drop=False)
            row['variants'][label] = dict(points=sim['points'], board_alone_points=alone['points'],
                                          overflow=sim['overflow'], top=sim['top'])
        rows.append(row)
    return meta, rows


def variant_summary(rows):
    usable = [r for r in rows if 'observed_points' in r and 'perception_consistent' in r]
    out = {}
    for label in VARIANTS:
        pts = [(r['variants'][label]['points'], r['observed_points']) for r in usable]
        out[label] = dict(compared=len(pts), exact_points=sum(a == b for a, b in pts),
                          increase_agreement=sum((a > 0) == (b > 0) for a, b in pts),
                          simulator_merge_site_none=sum(a > 0 and b == 0 for a, b in pts),
                          site_merge_simulator_none=sum(a == 0 and b > 0 for a, b in pts),
                          board_alone_merges=sum(r['variants'][label]['board_alone_points'] > 0 for r in usable),
                          mean_abs_points=float(np.mean([abs(a-b) for a, b in pts])) if pts else None)
    return out


def summarise(rows):
    usable = [r for r in rows if 'perception_consistent' in r]
    counts = Counter(r['category'] for r in usable)
    consistent = [r for r in usable if r['perception_consistent']]
    inconsistent = [r for r in usable if not r['perception_consistent']]
    medians = [r['layout_median_px'] for r in usable if r['layout_median_px'] is not None]
    return dict(
        releases=len(rows), compared=len(usable), categories=dict(counts),
        score_event_agreement=sum(r['category'] == 'agree' for r in usable),
        scoring_drops_observed=sum(r['observed_points'] > 0 for r in usable),
        scoring_drops_predicted=sum(r['predicted_points'] > 0 for r in usable),
        perception_consistent=len(consistent), perception_inconsistent=len(inconsistent),
        categories_when_consistent=dict(Counter(r['category'] for r in consistent)),
        categories_when_inconsistent=dict(Counter(r['category'] for r in inconsistent)),
        merge_events_agree=sum(r['merge_events_agree'] for r in consistent),
        horizon_300_changes_points=sum(r['points_300_ticks'] != r['predicted_points'] for r in rows),
        median_layout_px=float(np.median(medians)) if medians else None,
        mean_unmatched_predicted=float(np.mean([r['layout_unmatched_predicted'] for r in usable])) if usable else None,
        mean_unmatched_observed=float(np.mean([r['layout_unmatched_observed'] for r in usable])) if usable else None,
        nonzero_carry_over=sum(r['carry_over'] != 0 for r in usable))


def board_pixels(meta, body):
    x0, y0, x1, y1 = meta['board_bounds']
    for x, y, tier in body:
        yield x0+x/TRANSFER_WIDTH*(x1-x0), y0+y/HEIGHT*(y1-y0), TRANSFER_RADII[int(tier)]*(x1-x0)/TRANSFER_WIDTH, int(tier)


def overlay(image, meta, body, color, label):
    draw = ImageDraw.Draw(image)
    for x, y, r, tier in board_pixels(meta, body):
        draw.ellipse((x-r, y-r, x+r, y+r), outline=color, width=3)
        draw.text((x-4, y-6), str(tier+1), fill=color, stroke_width=2, stroke_fill='black')
    x0, y0, x1, y1 = meta['board_bounds']
    draw.rectangle((x0, y1+6, x1, y1+30), fill='#172e2d')
    draw.text((x0+8, y1+12), label, fill='white')
    return image


def panel(meta, decisions, frames_root, frames, row, out):
    """Pre-drop frame with detections | simulated settle | observed settled frame with detections."""
    x0, y0, x1, y1 = meta['board_bounds']
    crop = (x0-20, y0-110, x1+20, y1+36)
    drop = row['drop']
    i = drop-1
    def frame(tag, which):
        chosen = [f for f in frames if f['drop'] == which[0] and f['tag'] == tag]
        return Image.open(frames_root/chosen[which[1]]['image']).convert('RGB') if chosen else None
    pre = frame('aim', (drop, -1))
    post = frame('settle', (drop, -1))
    if pre is None or post is None:
        return None
    obs = np.asarray(decisions[i]['observation'], np.float32)
    body, count, current, _ = reconstruct(obs)
    sim = simulate(obs, row['action'])
    left = overlay(pre.copy(), meta, body[:count, [0, 1, 4]], '#ffffff',
                   f'Drop {drop} pre-release detections · aim {row["action"]+1}/15 · current {current+1}')
    middle = overlay(pre.copy(), meta, sim['body'][:, [0, 1, 4]], '#ff4fa3',
                     f'Local 80-tick replay · predicted +{row["predicted_points"]}')
    right_body = None
    if i+1 < len(decisions):
        nb, nc, _, _ = reconstruct(np.asarray(decisions[i+1]['observation'], np.float32))
        right_body = nb[:nc, [0, 1, 4]]
    right = overlay(post.copy(), meta, right_body if right_body is not None else np.zeros((0, 3)), '#7cf0ff',
                    f'Website settled · observed +{row.get("observed_points")}')
    tiles = [im.crop(crop) for im in (left, middle, right)]
    w, h = tiles[0].size
    canvas = Image.new('RGB', (3*w+20, h), '#162628')
    for k, tile in enumerate(tiles):
        canvas.paste(tile, (k*(w+10), 0))
    canvas.save(out, quality=82, optimize=True)
    return str(out)


def main(args):
    reports, runs = {}, {}
    names = [f'attempt-{k}' for k in range(1, 4)]
    for name in names:
        path = RUNS/name
        if not (path/'website.js').exists():
            continue
        meta, rows = analyse(name, path)
        runs[name] = dict(meta=meta, rows=rows, path=path)
        reports[name] = dict(outcome=meta['outcome'], final_score=meta['final_score'],
                             drops=meta['completed_drops'], **summarise(rows))
    for name, path in ARCHIVED.items():
        meta, rows = analyse(name, path)
        runs[name] = dict(meta=meta, rows=rows, path=path)
        reports[name] = dict(outcome=meta['outcome'], final_score=meta['final_score'],
                             drops=meta['completed_drops'], **summarise(rows))
    live_rows = [r for n in names if n in runs for r in runs[n]['rows']]
    reports['live-total'] = summarise(live_rows)
    fit_rows = [r for n in names if n in runs and n != HOLDOUT for r in runs[n]['rows']]
    fit = variant_summary(fit_rows)
    chosen = max(VARIANTS, key=lambda k: (fit[k]['increase_agreement'], -fit[k]['mean_abs_points']))
    calibration = dict(rule='choose the variant with the most score-increase agreements on attempts 1-2; report attempt 3 untouched',
                       fit=fit, chosen=chosen, holdout=HOLDOUT,
                       holdout_result=variant_summary(runs[HOLDOUT]['rows']) if HOLDOUT in runs else None,
                       archived={n: variant_summary(runs[n]['rows']) for n in ARCHIVED})
    examples = {}
    frames_out = ROOT/'frames'
    frames_out.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in runs:
            continue
        record = runs[name]['path']/'record'
        if not (record/'frames.json').exists():
            continue
        frames = json.loads((record/'frames.json').read_text())['frames']
        _, decisions = load_run(runs[name]['path'])
        usable = [r for r in runs[name]['rows'] if 'perception_consistent' in r]
        for key in ('simulator merge, site none', 'site merge, simulator none', 'both merged, different points'):
            for consistent in (True, False):
                picked = [r for r in usable if r['category'] == key and r['perception_consistent'] == consistent][:args.examples]
                for r in picked:
                    slug = key.split(',')[0].replace(' ', '-')
                    out = frames_out/f'{name}-drop{r["drop"]:03}-{slug}-{"consistent" if consistent else "inconsistent"}.jpg'
                    made = panel(runs[name]['meta'], decisions, record, frames, r, out)
                    if made:
                        examples.setdefault(key, []).append(dict(run=name, drop=r['drop'], consistent=consistent,
                                                                 image=str(out.relative_to(ROOT))))
    json_file(ROOT/'physics-diagnostic.json', dict(
        description=__doc__.strip(), physics=NOMINAL, ticks=80, match_radius_local_px=MATCH_RADIUS,
        summary=reports, calibration=calibration, examples=examples,
        rows={n: [{k: v for k, v in r.items()} for r in runs[n]['rows']] for n in runs}))
    print(json.dumps(reports, indent=1))
    print(json.dumps(calibration, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=int, default=2)
    main(parser.parse_args())

"""Audit the final drops of website runs that ended at the site's own game-over screen.

Each logged decision is recomputed from its saved observation with the frozen checkpoint and the
run's action seed (robust) or deterministically (hybrid), and the executed action must reproduce.
Geometry uses local transfer units: the board box top is y = 0, the local overflow check is a fruit
top above y = 11 after 80 ticks. The site's game-over test (public script) fires on a collision
between different fruits when body.y - centerOffset.y/2 < 70 game px, i.e. center - r/2 < 70 for a
circle; the release center is at game y = 80.
"""
import json
from pathlib import Path

import numpy as np

from .game import AIMS, HEIGHT, TRANSFER_RADII, TRANSFER_WIDTH
from .planner_experiment import load, logp
from .planning import drop_values
from .replay import json_file
from .robust_planner import RISK_WEIGHT, WEIGHT, choose
from .rollout_transfer import NOMINAL, reconstruct
from .rollout_teacher import settle_exact
from .verify import unpack

ROOT = Path('artifacts/live-baseline/gameover-audit')
GAME_TO_LOCAL = .963/(526/410)       # local units per game px
BOX_TOP_GAME = .164*724              # board box top in game px (box = canvas y + 0.164 h)
LOCAL_LINE = 11.
RELEASE_LOCAL = (80-BOX_TOP_GAME)*GAME_TO_LOCAL
RUNS = {
    'attempt-1': (Path('runs/live-baseline/attempt-1'), 'robust'),
    'attempt-2': (Path('runs/live-baseline/attempt-2'), 'robust'),
    'archived-robust-1673': (Path('artifacts/planner/website'), 'robust'),
    'archived-hybrid-1725': (Path('artifacts/transfer/demo-game-over'), 'hybrid'),
}
SKIPPED = {'attempt-3': 'detector stop (fruit centers or ready cue remained uncertain)'}


def site_threshold(radius_local):
    """Center height (local y) below which the site's collision test fires for that radius."""
    return (70-BOX_TOP_GAME)*GAME_TO_LOCAL + radius_local/2


def tops(body, count):
    rows = sorted(((float(body[i, 1]-TRANSFER_RADII[int(body[i, 4])]), int(body[i, 4])+1,
                    float(body[i, 0]), float(body[i, 1])) for i in range(count)))
    return [dict(top=round(t, 1), tier=k, x=round(x, 1), center_y=round(y, 1)) for t, k, x, y in rows[:4]]


def nominal_outcome(obs, action):
    body, count, current, _ = reconstruct(obs)
    board = body.copy()
    n, gain, ended = settle_exact(board, count, current, action, NOMINAL)
    top = min(board[j, 1]-TRANSFER_RADII[int(board[j, 4])] for j in range(n))
    return dict(points=int(gain), top=float(top), clearance=float(top-LOCAL_LINE), overflow=bool(ended))


def audit_run(name, path, strategy, policy, last=10):
    capture = unpack(path/'website.js')
    meta, decisions = capture['meta'], capture['decisions']
    rng = np.random.default_rng(meta['action_seed'])
    ledger = []
    for d in decisions:
        obs = np.asarray(d['observation'], np.float32)
        prior = logp(policy, obs)
        if strategy == 'robust':
            action, trace = choose(obs, prior, rng)
        else:
            physics = drop_values(obs)
            action = int(np.argmax(physics+WEIGHT*prior))
            trace = dict(physics=physics.tolist())
        assert action == d['action'], (name, d['drop'], action, d['action'])
        if d['drop'] <= len(decisions)-last:
            continue
        body, count, current, upcoming = reconstruct(obs)
        every = {a: nominal_outcome(obs, a) for a in range(15)}
        row = dict(drop=d['drop'], action=action, aim=action+1, current_tier=current+1, next_tier=upcoming+1,
                   detections=count, coverage=d.get('coverage'), score_before=d.get('score_before'),
                   score_after=d.get('score'), detected_tops=tops(body, count),
                   predicted=every[action],
                   surviving_actions_nominal=[a for a in range(15) if not every[a]['overflow']],
                   ppo_term=float(WEIGHT*prior[action]))
        if strategy == 'robust':
            candidates = trace['candidates']
            i = candidates.index(action)
            values = np.asarray(trace['outcomes'])
            risks = np.asarray(trace['overflow_risk'])
            row['candidates'] = [dict(action=c, physics_term=float(values[k].mean()),
                                      risk_term=float(-RISK_WEIGHT*risks[k].mean()),
                                      ppo_term=float(WEIGHT*prior[c]), combined=float(trace['combined'][k]),
                                      overflow_risk=float(risks[k].mean()),
                                      scenarios_overflowing=int((risks[k] >= 1).sum()), scenarios=int(len(risks[k])),
                                      nominal_overflow=every[c]['overflow'])
                                 for k, c in enumerate(candidates)]
            chosen = row['candidates'][i]
            row.update(physics_term=chosen['physics_term'], risk_term=chosen['risk_term'],
                       overflow_risk=chosen['overflow_risk'],
                       all_scenarios_overflow=chosen['scenarios_overflowing'] == chosen['scenarios'],
                       safer_candidate_available=any(c['scenarios_overflowing'] < c['scenarios'] for c in row['candidates'])
                       and chosen['scenarios_overflowing'] == chosen['scenarios'])
        else:
            row.update(physics_term=float(trace['physics'][action]), risk_term=None,
                       overflow_risk=1. if every[action]['overflow'] else 0.,
                       all_scenarios_overflow=every[action]['overflow'],
                       safer_candidate_available=every[action]['overflow'] and bool(row['surviving_actions_nominal']),
                       note='hybrid: overflow enters the physics term as the -8 rank penalty; no scenarios')
        ledger.append(row)
    return meta, ledger


def main():
    policy = load()
    out = dict(description=__doc__.strip(), local_line=LOCAL_LINE, release_center_local=RELEASE_LOCAL,
               site_threshold_examples={f'tier {t+1}': round(site_threshold(TRANSFER_RADII[t]), 1) for t in range(5)},
               skipped=SKIPPED, runs={})
    for name, (path, strategy) in RUNS.items():
        meta, ledger = audit_run(name, path, strategy, policy)
        out['runs'][name] = dict(strategy=strategy, outcome=meta['outcome'], final_score=meta['final_score'],
                                 drops=meta['completed_drops'], ledger=ledger)
        last = ledger[-1]
        print(name, meta['final_score'], 'terminal drop', last['drop'], 'aim', last['aim'],
              'pred', last['predicted'], 'top detected', last['detected_tops'][:1],
              'all-scen-overflow', last['all_scenarios_overflow'], 'safer', last['safer_candidate_available'],
              'nominal survivors', len(last['surviving_actions_nominal']))
        for r in ledger:
            print('  ', r['drop'], 'aim', r['aim'], 'cur', r['current_tier'], 'mintop', r['detected_tops'][0]['top'] if r['detected_tops'] else None,
                  'pred top %.1f' % r['predicted']['top'], 'ovf', r['predicted']['overflow'], 'risk', round(r['overflow_risk'], 2),
                  'terms', [round(r['physics_term'], 2), r['risk_term'] and round(r['risk_term'], 2), round(r['ppo_term'], 2)],
                  'survivors', len(r['surviving_actions_nominal']), 'safer', r['safer_candidate_available'])
    ROOT.mkdir(parents=True, exist_ok=True)
    json_file(ROOT/'decision-ledger.json', out)



def spawn_contact(obs, target_x, board_left, radii):
    """First fruit a straight-down release would touch, and whether the site's test would fire.

    Uses detected centers, the site-derived radii and the screen x actually targeted. The release
    center starts at game y = 80; the site test fires for a different-tier contact while
    center - r/2 < 70 game px.
    """
    body, count, current, _ = reconstruct(obs)
    if current < 0:
        return None
    r = radii[current]
    x = (target_x-board_left)/(526/410)
    best = None
    for i in range(count):
        ri = radii[int(body[i, 4])]
        dx = abs(body[i, 0]-x)
        if dx >= r+ri:
            continue
        y = body[i, 1]-np.sqrt((r+ri)**2-dx**2)
        if best is None or y < best[0]:
            best = (float(y), int(body[i, 4]))
    limit = (70-BOX_TOP_GAME)*GAME_TO_LOCAL+r/2
    if best is None:
        return dict(contact_center=None, fatal=False, limit=limit)
    y, tier = best
    fatal = y < max(limit, RELEASE_LOCAL) and tier != current
    return dict(contact_center=round(y, 1), contact_tier=tier+1, limit=round(limit, 1), fatal=bool(fatal))


def compare_rules():
    from .live_diagnostic import SITE_RADII
    runs = dict(RUNS, **{'attempt-3': (Path('runs/live-baseline/attempt-3'), 'robust')})
    report = {}
    for name, (path, _) in runs.items():
        capture = unpack(path/'website.js')
        meta, decisions = capture['meta'], capture['decisions']
        left = meta['board_bounds'][0]
        rows = []
        for d in decisions:
            obs = np.asarray(d['observation'], np.float32)
            spawn = spawn_contact(obs, d['target_x'], left, SITE_RADII)
            local = nominal_outcome(obs, d['action'])
            body, count, _, _ = reconstruct(obs)
            detected_top = min((body[i, 1]-TRANSFER_RADII[int(body[i, 4])] for i in range(count)), default=HEIGHT)
            rows.append(dict(drop=d['drop'], spawn=spawn, local_overflow_80=local['overflow'],
                             detected_board_over_local_line=bool(detected_top < LOCAL_LINE)))
        terminal = rows[-1] if meta['outcome'] == 'game over' else None
        body = rows[:-1] if terminal else rows
        report[name] = dict(
            outcome=meta['outcome'], releases=len(rows),
            nonterminal_spawn_rule_fatal=sum(bool(r['spawn'] and r['spawn']['fatal']) for r in body),
            nonterminal_local_80_overflow=sum(r['local_overflow_80'] for r in body),
            nonterminal_detected_board_over_local_line=sum(r['detected_board_over_local_line'] for r in body),
            first_drop_detected_over_local_line=next((r['drop'] for r in rows if r['detected_board_over_local_line']), None),
            terminal=terminal)
    return report


def knowing_check():
    """Over every robust website decision: chosen move overflowing in all scenarios while another survives."""
    policy = load()
    runs = dict(RUNS, **{'attempt-3': (Path('runs/live-baseline/attempt-3'), 'robust')})
    out = {}
    for name, (path, strategy) in runs.items():
        if strategy != 'robust':
            continue
        capture = unpack(path/'website.js')
        rng = np.random.default_rng(capture['meta']['action_seed'])
        n = every = knowing = nonzero = 0
        peak = 0.
        for d in capture['decisions']:
            obs = np.asarray(d['observation'], np.float32)
            action, trace = choose(obs, logp(policy, obs), rng)
            assert action == d['action']
            risks = np.asarray(trace['overflow_risk'])
            full = (risks >= 1).all(1)
            i = trace['candidates'].index(action)
            n += 1
            every += bool(full[i])
            knowing += bool(full[i] and (~full).any())
            nonzero += bool(risks[i].mean() > 0)
            peak = max(peak, float(risks[i].mean()))
        out[name] = dict(decisions=n, chosen_all_scenarios_overflow=every, knowingly_fatal_with_safer=knowing,
                         chosen_nonzero_risk=nonzero, max_chosen_mean_risk=round(peak, 3))
    return out


def write_all():
    main()
    json_file(ROOT/'rule-comparison.json', compare_rules())
    json_file(ROOT/'knowingly-fatal-check.json', knowing_check())


if __name__ == '__main__':
    write_all()

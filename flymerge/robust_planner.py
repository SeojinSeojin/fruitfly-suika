"""One-drop scoring under common, antithetic sensor and physics perturbations."""
import numpy as np
from numba import njit

from .game import AIMS, HEIGHT, SLOTS, TRANSFER_RADII, TRANSFER_WIDTH
from .planning import drop_values, rank_trial, settle

WEIGHT = .20
SHORTLIST = 4
RISK_WEIGHT = 2.


@njit(cache=True)
def simulate(observation, candidates, offsets, shifts):
    count = 0
    base = np.zeros((SLOTS+1, 5))
    for row in observation[:-2].reshape(-1, 3):
        if row[0] >= 0 and row[1] >= 0:
            base[count, 0] = row[0]*TRANSFER_WIDTH
            base[count, 1] = row[1]*HEIGHT
            base[count, 4] = min(10, max(0, round(row[2]*10)))
            count += 1
    values = np.zeros((len(candidates), len(shifts)))
    risks = values.copy()
    if observation[-2] < 0:
        return values, risks
    current = min(10, round(observation[-2]*10))
    upcoming = round(observation[-1]*10)
    for j, (aim, gravity, impact) in enumerate(shifts):
        board = base.copy()
        for k in range(count):
            radius = TRANSFER_RADII[int(base[k, 4])]
            board[k, 0] = min(TRANSFER_WIDTH-radius, max(radius, base[k, 0]+offsets[j,k,0]))
            board[k, 1] = base[k, 1]+offsets[j,k,1]
        if j == 0:
            board = base
        for i, action in enumerate(candidates):
            trial, n, score = settle(board, count, current, AIMS[action], .46+gravity, .55+impact, aim)
            values[i,j] = rank_trial(trial, count, n, score, upcoming)
            top = HEIGHT
            for k in range(n):
                top = min(top, trial[k,1]-TRANSFER_RADII[int(trial[k,4])])
            risks[i,j] = 1. if n >= SLOTS else min(1., max(0., (31-top)/20))
    return values, risks


def choose(observation, log_probability, random, shortlist=SHORTLIST, allowed=None):
    physics = drop_values(observation)
    nominal = physics + WEIGHT*log_probability
    order = np.argsort(-nominal, kind='stable')
    if allowed is not None:
        # Spawn-contact gate: shortlist only permitted actions; random draws are unchanged.
        order = order[np.asarray(allowed, bool)[order]]
    candidates = order[:shortlist]
    offsets = random.uniform(-1, 1, (3, SLOTS, 2))*np.array([4., 5.])
    shifts = random.uniform(-1, 1, (3, 3))*np.array([6., .025, .045])
    offsets = np.concatenate((np.zeros((1,SLOTS,2)), offsets, -offsets))
    shifts = np.concatenate((np.zeros((1,3)), shifts, -shifts))
    values, risks = simulate(observation, candidates, offsets, shifts)
    combined = values.mean(1)-RISK_WEIGHT*risks.mean(1)+WEIGHT*log_probability[candidates]
    best = np.lexsort((candidates,-combined))[0]
    action = int(candidates[best])
    return action, dict(nominal=nominal.tolist(), candidates=candidates.tolist(),
                        outcomes=values.tolist(), overflow_risk=risks.tolist(),
                        combined=combined.tolist(), selected_sd=float(values[best].std()))

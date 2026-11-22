"""Rank legal drops using counterfactuals in the local physics model."""
from numba import njit
import numpy as np

from .game import AIMS, HEIGHT, SLOTS, TRANSFER_RADII, TRANSFER_WIDTH, tick


@njit(cache=True)
def settle(board, count, current, aim):
    trial = board.copy()
    radius = TRANSFER_RADII[current]
    trial[count, 0] = radius + (TRANSFER_WIDTH-2*radius)*aim
    trial[count, 1] = -25
    trial[count, 2] = trial[count, 3] = 0.
    trial[count, 4] = current
    n, score = count+1, 0
    for _ in range(80):
        n, gain = tick(trial, n, TRANSFER_WIDTH, TRANSFER_RADII)
        score += gain
    return trial, n, score


@njit(cache=True)
def rank_trial(trial, before, n, score, upcoming):
    high = HEIGHT
    load = 0.
    preview = 0.
    for i in range(n):
        tier = int(trial[i, 4])
        top = trial[i, 1]-TRANSFER_RADII[tier]
        high = min(high, top)
        load += (1-top/HEIGHT)**2
        if tier == upcoming:
            preview = max(preview, 1-top/HEIGHT)
    value = np.log1p(score) + .9*(before+1-n) + high/HEIGHT - .12*load + .25*preview
    return value - 8. if high < 11 else value


@njit(cache=True)
def predict_drops(observation, future=False):
    board = np.zeros((SLOTS+1, 5))
    count = 0
    for i in range(SLOTS):
        x, y, tier = observation[3*i:3*i+3]
        if x >= 0 and y >= 0:
            board[count, 0] = x * TRANSFER_WIDTH
            board[count, 1] = y * HEIGHT
            board[count, 4] = min(10, max(0, round(tier*10)))
            count += 1
    values = np.zeros(15)
    scores = np.zeros(15, dtype=np.int32)
    if observation[-2] < 0:
        return values, scores
    current = min(10, round(observation[-2]*10))
    upcoming = round(observation[-1]*10)
    preview = future and 0 <= upcoming <= 10 and count < SLOTS
    boards = np.empty((15 if preview else 0, SLOTS+1, 5))
    counts = np.zeros(15, dtype=np.int32)
    for action, aim in enumerate(AIMS):
        trial, n, score = settle(board, count, current, aim)
        scores[action] = score
        values[action] = rank_trial(trial, count, n, score, upcoming)
        if preview:
            boards[action] = trial
            counts[action] = n
    if preview:
        candidates = np.argsort(values)
        extended = np.full(15, -np.inf)
        for action in candidates[-4:]:
            n = counts[action]
            best = -8.
            if values[action] > -5 and n < SLOTS:
                for aim in AIMS:
                    trial, after, score = settle(boards[action], n, upcoming, aim)
                    best = max(best, rank_trial(trial, n, after, score, -1))
            extended[action] = values[action] + .55 * best
        values = extended
    return values, scores


@njit(cache=True)
def drop_values(observation, future=False):
    return predict_drops(observation, future)[0]

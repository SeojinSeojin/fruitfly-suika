"""Rank legal drops using one-step counterfactuals in the local physics model."""
from numba import njit
import numpy as np

from .game import AIMS, HEIGHT, SLOTS, TRANSFER_RADII, TRANSFER_WIDTH, tick


@njit(cache=True)
def drop_values(observation):
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
    if observation[-2] < 0:
        return values
    current = min(10, round(observation[-2]*10))
    upcoming = round(observation[-1]*10)
    radius = TRANSFER_RADII[current]
    for action, aim in enumerate(AIMS):
        trial = board.copy()
        trial[count, 0] = radius + (TRANSFER_WIDTH-2*radius)*aim
        trial[count, 1] = -25
        trial[count, 4] = current
        n, score = count+1, 0
        for _ in range(80):
            n, gain = tick(trial, n, TRANSFER_WIDTH, TRANSFER_RADII)
            score += gain
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
        values[action] = np.log1p(score) + .9*(count+1-n) + high/HEIGHT - .12*load + .25*preview
        if high < 11:
            values[action] -= 8.
    return values
